#!/usr/bin/env python3
"""
client_debug.py – Version de débogage pour diagnostiquer les problèmes audio
"""

import os
import time
import tempfile
import audioop
import struct
import numpy as np
from collections import deque
from enum import Enum

import webrtcvad
import pyaudio
import requests
from robot_hat import Pin, Music
from dotenv import load_dotenv, find_dotenv

# ─────────────── CONFIG ────────────────────────────────────────────
load_dotenv(find_dotenv())

API_ENDPOINT = "http://192.168.110.35:8000/ask"
API_TOKEN    = os.getenv("API_TOKEN")
SR           = 16000
FRAME_MS     = 30
CHUNK        = int(SR * FRAME_MS / 1000)

# LED sur le Robot HAT
led = Pin('LED')


# ─────────────── ÉTAT DE DÉTECTION ─────────────────────────────────
class SpeechState(Enum):
    SILENCE = "SILENCE"
    MAYBE_SPEECH = "MAYBE_SPEECH"
    SPEECH = "SPEECH"
    ENDING = "ENDING"


# ─────────────── ENREGISTREUR AUDIO AVEC DEBUG ─────────────────────
class AudioRecorder:
    def __init__(self):
        self.frames = []
        self.pre_buffer = deque(maxlen=10)  # 300ms de pré-buffer
        self.min_audio_level = 50  # Seuil très bas pour debug
        
    def add_to_prebuffer(self, frame):
        """Ajoute une frame au pré-buffer"""
        self.pre_buffer.append(frame)
        
    def start_recording(self):
        """Démarre l'enregistrement avec le pré-buffer"""
        self.frames = list(self.pre_buffer)
        
    def add_frame(self, frame):
        """Ajoute une frame à l'enregistrement"""
        self.frames.append(frame)
        
    def is_valid_recording(self):
        """Vérifie si l'enregistrement contient vraiment de la parole"""
        if not self.frames:
            print("  DEBUG: Aucune frame enregistrée")
            return False
            
        # Calculer le niveau moyen et max
        levels = []
        for frame in self.frames:
            level = audioop.rms(frame, 2)
            levels.append(level)
            
        avg_level = np.mean(levels)
        max_level = np.max(levels)
        min_level = np.min(levels)
        
        print(f"\n  DEBUG Enregistrement:")
        print(f"    - Frames: {len(self.frames)}")
        print(f"    - Niveau moyen: {avg_level:.1f}")
        print(f"    - Niveau max: {max_level:.1f}")
        print(f"    - Niveau min: {min_level:.1f}")
        print(f"    - Seuil requis: {self.min_audio_level}")
        print(f"    - Valide: {avg_level > self.min_audio_level}")
        
        return avg_level > self.min_audio_level
        
    def get_wav(self):
        """Retourne le WAV complet avec header"""
        if not self.frames:
            return None
            
        # Concaténer l'audio SANS normalisation pour debug
        data = b"".join(self.frames)
        
        # Header WAV
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            len(data) + 36,
            b'WAVE',
            b'fmt ',
            16, 1, 1, SR, SR*2, 2, 16,
            b'data',
            len(data)
        )
        
        return header + data
        
    def clear(self):
        """Réinitialise l'enregistreur"""
        self.frames = []
        self.pre_buffer.clear()


# ─────────────── DÉTECTEUR SIMPLIFIÉ POUR DEBUG ────────────────────
class SimpleSpeechDetector:
    def __init__(self):
        self.vad = webrtcvad.Vad(1)  # Mode 1 = plus sensible
        self.noise_floor = 100
        self.threshold = 200  # Seuil fixe bas
        self.state = SpeechState.SILENCE
        self.speech_frames = 0
        self.silence_frames = 0
        
    def calibrate(self, stream, duration=2.0):
        """Calibration simple"""
        print("🎤 Calibration rapide...")
        levels = []
        
        for _ in range(int(duration * 1000 / FRAME_MS)):
            pcm = stream.read(CHUNK, exception_on_overflow=False)
            level = audioop.rms(pcm, 2)
            levels.append(level)
            
        self.noise_floor = np.mean(levels)
        self.threshold = self.noise_floor * 2  # Seuil = 2x le bruit
        
        print(f"✓ Calibration:")
        print(f"  - Bruit moyen: {self.noise_floor:.0f}")
        print(f"  - Seuil: {self.threshold:.0f}")
        
    def process_frame(self, pcm):
        """Traite une frame audio"""
        level = audioop.rms(pcm, 2)
        is_speech = level > self.threshold
        
        action = None
        
        if self.state == SpeechState.SILENCE:
            if is_speech:
                self.speech_frames += 1
                if self.speech_frames >= 3:  # 90ms de parole
                    self.state = SpeechState.SPEECH
                    action = "START_RECORDING"
                    print(f"\n  DEBUG: Début détecté - niveau {level}")
            else:
                self.speech_frames = 0
                
        elif self.state == SpeechState.SPEECH:
            if not is_speech:
                self.silence_frames += 1
                if self.silence_frames >= 30:  # 900ms de silence
                    self.state = SpeechState.SILENCE
                    self.speech_frames = 0
                    self.silence_frames = 0
                    action = "STOP_RECORDING"
                    print(f"\n  DEBUG: Fin détectée après {self.silence_frames} frames de silence")
            else:
                self.silence_frames = 0
                
        return action, level


# ─────────────── CLIENT DEBUG ─────────────────────────────────────
class DebugClient:
    def __init__(self):
        # Music : gère la sortie haut-parleur PiCar-X
        self.music = Music()
        self.music.music_set_volume(70)
        
        # Audio
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self._setup_audio_input()
        
        # Détection et enregistrement
        self.detector = SimpleSpeechDetector()
        self.recorder = AudioRecorder()
        
        # Calibration
        self.detector.calibrate(self.stream)
        
    def _setup_audio_input(self):
        """Configure le flux audio"""
        try:
            print("🔍 Périphériques audio disponibles:")
            for i in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  [{i}] {info['name']} ({info['maxInputChannels']} ch)")
            
            # Utiliser le périphérique par défaut
            self.stream = self.pa.open(
                rate=SR,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=CHUNK,
            )
            print("✓ Flux audio configuré (périphérique par défaut)")
        except Exception as e:
            print(f"❌ Erreur audio: {e}")
            raise
            
    def listen_forever(self):
        """Boucle principale d'écoute"""
        print("\n🎤 Mode DEBUG - Seuils très bas")
        print("   Parlez pour tester la détection\n")
        
        recording = False
        frames_recorded = 0
        
        while True:
            try:
                # Lecture audio
                pcm = self.stream.read(CHUNK, exception_on_overflow=False)
                
                # Ajout au pré-buffer
                if not recording:
                    self.recorder.add_to_prebuffer(pcm)
                
                # Analyse
                action, level = self.detector.process_frame(pcm)
                
                # Affichage
                bar = '█' * int(level / 50)
                state_char = '●' if recording else '·'
                print(f"[{state_char}] {level:5.0f} {bar:<40}", end='\r')
                
                # Actions
                if action == "START_RECORDING":
                    print(f"\n🔴 ENREGISTREMENT DÉMARRÉ")
                    led.on()
                    recording = True
                    frames_recorded = 0
                    self.recorder.start_recording()
                    
                elif recording:
                    self.recorder.add_frame(pcm)
                    frames_recorded += 1
                    
                    if action == "STOP_RECORDING":
                        led.off()
                        recording = False
                        duration_ms = frames_recorded * FRAME_MS
                        
                        print(f"\n⏹️  ENREGISTREMENT TERMINÉ ({duration_ms}ms)")
                        
                        if self.recorder.is_valid_recording():
                            print("✅ Enregistrement valide - envoi au serveur")
                            wav_data = self.recorder.get_wav()
                            if wav_data:
                                # Sauvegarder pour debug
                                with open("debug_recording.wav", "wb") as f:
                                    f.write(wav_data)
                                print("   Sauvegardé dans debug_recording.wav")
                                
                                self._send_and_play(wav_data)
                        else:
                            print("❌ Enregistrement rejeté")
                            
                        self.recorder.clear()
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Erreur: {e}")
                recording = False
                self.recorder.clear()
                led.off()
                
    def _send_and_play(self, wav_bytes: bytes):
        """Envoie l'audio au serveur et joue la réponse"""
        try:
            print("📡 Envoi au serveur...")
            
            resp = requests.post(
                API_ENDPOINT,
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                files={"file": ("voice.wav", wav_bytes, "audio/wav")},
                timeout=20,
            )
            resp.raise_for_status()
            
            response_data = resp.json()
            
            if "answer" in response_data:
                print(f"🤖: {response_data['answer']}")
                
            if "audio" in response_data:
                mp3_data = bytes.fromhex(response_data["audio"])
                self.play_mp3(mp3_data)
                
        except Exception as e:
            print(f"❌ Erreur: {e}")
            
    def play_mp3(self, mp3_bytes: bytes):
        """Lecture via l'API Music"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(mp3_bytes)
            path = f.name
        try:
            self.music.sound_play(path)
        finally:
            os.remove(path)
            
    def cleanup(self):
        """Nettoyage des ressources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
        led.off()


# ─────────────── MAIN ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🔧 CLIENT DEBUG - Diagnostic audio")
    print("=" * 60)
    print("Ce client utilise des seuils très bas pour diagnostiquer")
    print("les problèmes de détection audio.")
    print("=" * 60)
    
    client = DebugClient()
    
    try:
        client.listen_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt")
    finally:
        client.cleanup()
        print("✓ Terminé")
