#!/usr/bin/env python3
"""
client.py – PiCar-X : écoute continue → enregistrement → envoi serveur → lecture réponse
Version optimisée pour Raspberry Pi 5 avec détection de parole améliorée
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


# ─────────────── DÉTECTEUR OPTIMISÉ ────────────────────────────────
class OptimizedSpeechDetector:
    def __init__(self):
        # WebRTC VAD
        self.vad = webrtcvad.Vad(2)  # Mode 2 = équilibré
        
        # Calibration
        self.noise_floor = 0
        self.noise_std = 0
        self.threshold_low = 0
        self.threshold_high = 0
        
        # État
        self.state = SpeechState.SILENCE
        self.state_duration = 0
        
        # Buffers
        self.level_history = deque(maxlen=10)  # 300ms d'historique
        self.speech_confidence = 0
        
        # Paramètres ajustables
        self.min_speech_frames = 10  # 300ms minimum de parole
        self.max_silence_frames = 50  # 1.5s max de silence
        self.pre_speech_frames = 5   # 150ms de pré-buffer
        
    def calibrate(self, stream, duration=2.0):
        """Calibration du bruit ambiant"""
        print("🎤 Calibration du bruit ambiant... Ne parlez pas.")
        levels = []
        
        for _ in range(int(duration * 1000 / FRAME_MS)):
            pcm = stream.read(CHUNK, exception_on_overflow=False)
            level = audioop.rms(pcm, 2)
            levels.append(level)
            
        # Statistiques du bruit
        self.noise_floor = np.mean(levels)
        self.noise_std = np.std(levels)
        noise_max = np.percentile(levels, 95)
        
        # Seuils adaptatifs
        self.threshold_low = max(noise_max * 1.5, self.noise_floor + 3 * self.noise_std)
        self.threshold_high = self.threshold_low * 1.5
        
        print(f"✓ Calibration terminée:")
        print(f"  - Bruit moyen: {self.noise_floor:.0f}")
        print(f"  - Seuil bas: {self.threshold_low:.0f}")
        print(f"  - Seuil haut: {self.threshold_high:.0f}")
        
    def process_frame(self, pcm):
        """Traite une frame audio et retourne l'action à effectuer"""
        level = audioop.rms(pcm, 2)
        self.level_history.append(level)
        
        # Calcul des métriques
        avg_level = np.mean(self.level_history)
        level_variation = np.std(self.level_history)
        is_vad_speech = self.vad.is_speech(pcm, SR)
        
        # Détection multi-critères
        is_loud = level > self.threshold_low
        is_very_loud = level > self.threshold_high
        has_variation = level_variation > (self.noise_std * 2)
        
        # Score de confiance
        confidence = 0
        if is_loud: confidence += 1
        if is_very_loud: confidence += 1
        if is_vad_speech: confidence += 2
        if has_variation: confidence += 1
        
        # Machine d'état
        action = None
        
        if self.state == SpeechState.SILENCE:
            if confidence >= 3:
                self.speech_confidence += 1
                if self.speech_confidence >= 2:  # 2 frames consécutives
                    self.state = SpeechState.MAYBE_SPEECH
                    self.state_duration = 0
            else:
                self.speech_confidence = 0
                
        elif self.state == SpeechState.MAYBE_SPEECH:
            self.state_duration += 1
            if confidence >= 2:
                if self.state_duration >= 3:  # 90ms de confirmation
                    self.state = SpeechState.SPEECH
                    action = "START_RECORDING"
            else:
                self.state = SpeechState.SILENCE
                self.speech_confidence = 0
                
        elif self.state == SpeechState.SPEECH:
            self.state_duration += 1
            if confidence < 2:
                self.state = SpeechState.ENDING
                self.state_duration = 0
                
        elif self.state == SpeechState.ENDING:
            self.state_duration += 1
            if confidence >= 2:
                # Retour à la parole
                self.state = SpeechState.SPEECH
            elif self.state_duration >= self.max_silence_frames:
                # Fin de la parole
                self.state = SpeechState.SILENCE
                self.speech_confidence = 0
                action = "STOP_RECORDING"
                
        return action, level, confidence
        
    def should_recalibrate(self):
        """Détermine si une recalibration est nécessaire"""
        # Recalibrer toutes les 5 minutes ou si trop de faux positifs
        return False  # Pour l'instant, calibration manuelle seulement


# ─────────────── ENREGISTREUR AUDIO ────────────────────────────────
class AudioRecorder:
    def __init__(self):
        self.frames = []
        self.pre_buffer = deque(maxlen=10)  # 300ms de pré-buffer
        
    def add_to_prebuffer(self, frame):
        """Ajoute une frame au pré-buffer"""
        self.pre_buffer.append(frame)
        
    def start_recording(self):
        """Démarre l'enregistrement avec le pré-buffer"""
        self.frames = list(self.pre_buffer)
        
    def add_frame(self, frame):
        """Ajoute une frame à l'enregistrement"""
        self.frames.append(frame)
        
    def get_wav(self):
        """Retourne le WAV complet avec header"""
        if not self.frames:
            return None
            
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


# ─────────────── CLIENT PRINCIPAL ─────────────────────────────────
class Client:
    def __init__(self):
        # Music : gère la sortie haut-parleur PiCar-X
        self.music = Music()
        self.music.music_set_volume(70)
        
        # Audio
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self._setup_audio_input()
        
        # Détection et enregistrement
        self.detector = OptimizedSpeechDetector()
        self.recorder = AudioRecorder()
        
        # Calibration initiale
        self.detector.calibrate(self.stream)
        
    def _setup_audio_input(self):
        """Configure le flux audio"""
        try:
            self.stream = self.pa.open(
                rate=SR,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=CHUNK,
            )
        except Exception as e:
            print(f"Erreur audio: {e}")
            raise
            
    def listen_forever(self):
        """Boucle principale d'écoute"""
        print("\n🎤 En attente de parole... (Ctrl+C pour arrêter)")
        
        recording = False
        frames_since_start = 0
        
        while True:
            try:
                # Lecture audio
                pcm = self.stream.read(CHUNK, exception_on_overflow=False)
                
                # Ajout au pré-buffer
                if not recording:
                    self.recorder.add_to_prebuffer(pcm)
                
                # Analyse
                action, level, confidence = self.detector.process_frame(pcm)
                
                # Affichage du niveau (optionnel)
                bar = '█' * int(level / 100)
                state_char = {'SILENCE': '·', 'MAYBE_SPEECH': '?', 
                             'SPEECH': '●', 'ENDING': '○'}[self.detector.state.value]
                print(f"[{state_char}] {level:5.0f} {bar}", end='\r')
                
                # Actions
                if action == "START_RECORDING":
                    print("\n🔊 Parole détectée ! Enregistrement...")
                    led.on()
                    recording = True
                    frames_since_start = 0
                    self.recorder.start_recording()
                    
                elif recording:
                    self.recorder.add_frame(pcm)
                    frames_since_start += 1
                    
                    if action == "STOP_RECORDING":
                        led.off()
                        recording = False
                        
                        # Vérifier la durée minimale
                        if frames_since_start >= self.detector.min_speech_frames:
                            print(f"\n✓ Enregistrement terminé ({frames_since_start * FRAME_MS}ms)")
                            wav_data = self.recorder.get_wav()
                            if wav_data:
                                self._send_and_play(wav_data)
                        else:
                            print("\n✗ Trop court, ignoré")
                            
                        self.recorder.clear()
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nErreur: {e}")
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
                
        except requests.exceptions.Timeout:
            print("❌ Timeout serveur")
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur réseau: {e}")
        except Exception as e:
            print(f"❌ Erreur: {e}")
            
    def play_mp3(self, mp3_bytes: bytes):
        """Lecture via l'API Music → haut-parleur PiCar-X"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(mp3_bytes)
            path = f.name
        try:
            self.music.sound_play(path)
        finally:
            os.remove(path)
            
    def recalibrate(self):
        """Recalibration manuelle du bruit"""
        print("\n📊 Recalibration...")
        self.detector.calibrate(self.stream)
        
    def cleanup(self):
        """Nettoyage des ressources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
        led.off()


# ─────────────── MAIN ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 PiCar-X Voice Assistant - Version optimisée")
    print("=" * 50)
    
    client = Client()
    
    try:
        client.listen_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt demandé")
    finally:
        client.cleanup()
        print("✓ Nettoyage terminé")
