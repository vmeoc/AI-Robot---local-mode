#!/usr/bin/env python3
"""
client.py – PiCar-X : écoute continue → enregistrement → envoi serveur → lecture réponse
Version optimisée pour Raspberry Pi 5 avec détection de parole améliorée et réduction de bruit
"""

import os
import time
import tempfile
import audioop
import struct
import numpy as np
from collections import deque
from enum import Enum
from scipy import signal

import webrtcvad
import pyaudio
import requests
from robot_hat import Pin, Music
from dotenv import load_dotenv, find_dotenv
from piper import PiperVoice  # Pour TTS de secours

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


# ─────────────── UTILITAIRES AUDIO ─────────────────────────────────
class AudioProcessor:
    """Traitement audio : filtrage, normalisation, etc."""
    
    @staticmethod
    def high_pass_filter(data, cutoff=80, fs=SR):
        """Filtre passe-haut pour éliminer les basses fréquences"""
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        
        # Convertir bytes en numpy array
        audio_array = np.frombuffer(data, dtype=np.int16)
        
        # Créer et appliquer le filtre
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        filtered = signal.filtfilt(b, a, audio_array)
        
        # Reconvertir en bytes
        return filtered.astype(np.int16).tobytes()
    
    @staticmethod
    def normalize_audio(data, target_level=3000):
        """Normalise le niveau audio"""
        # Calculer le niveau RMS actuel
        current_level = audioop.rms(data, 2)
        
        if current_level > 0:
            # Calculer le facteur de normalisation
            factor = target_level / current_level
            # Limiter le facteur pour éviter la saturation
            factor = min(factor, 10.0)
            factor = max(factor, 0.1)
            
            # Appliquer la normalisation
            return audioop.mul(data, 2, factor)
        
        return data
    
    @staticmethod
    def compute_snr(signal_data, noise_floor):
        """Calcule le rapport signal/bruit"""
        signal_level = audioop.rms(signal_data, 2)
        if noise_floor > 0:
            return 20 * np.log10(signal_level / noise_floor)
        return 0


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
        
        # Statistiques
        self.total_detections = 0
        self.false_positives = 0
        
    def calibrate(self, stream, duration=3.0):
        """Calibration améliorée du bruit ambiant"""
        print("🎤 Calibration du bruit ambiant... Ne parlez pas.")
        levels = []
        
        for _ in range(int(duration * 1000 / FRAME_MS)):
            pcm = stream.read(CHUNK, exception_on_overflow=False)
            # Appliquer le filtre passe-haut
            filtered = AudioProcessor.high_pass_filter(pcm)
            level = audioop.rms(filtered, 2)
            levels.append(level)
            
            # Affichage de progression
            progress = len(levels) / (duration * 1000 / FRAME_MS)
            bar = '█' * int(progress * 20)
            print(f"\r  [{bar:<20}] {progress*100:.0f}%", end='')
            
        print()  # Nouvelle ligne
        
        # Statistiques du bruit avec percentiles
        levels_array = np.array(levels)
        self.noise_floor = np.median(levels_array)
        self.noise_std = np.std(levels_array)
        noise_p75 = np.percentile(levels_array, 75)
        noise_p95 = np.percentile(levels_array, 95)
        
        # Seuils adaptatifs basés sur les percentiles
        self.threshold_low = max(noise_p95 * 1.5, self.noise_floor + 4 * self.noise_std)
        self.threshold_high = self.threshold_low * 1.8
        
        print(f"✓ Calibration terminée:")
        print(f"  - Bruit médian: {self.noise_floor:.0f}")
        print(f"  - Bruit P95: {noise_p95:.0f}")
        print(f"  - Seuil bas: {self.threshold_low:.0f}")
        print(f"  - Seuil haut: {self.threshold_high:.0f}")
        print(f"  - Écart-type: {self.noise_std:.0f}")
        
    def process_frame(self, pcm):
        """Traite une frame audio et retourne l'action à effectuer"""
        # Appliquer le filtre passe-haut
        filtered = AudioProcessor.high_pass_filter(pcm)
        
        level = audioop.rms(filtered, 2)
        self.level_history.append(level)
        
        # Calcul des métriques
        avg_level = np.mean(self.level_history)
        level_variation = np.std(self.level_history)
        is_vad_speech = self.vad.is_speech(filtered, SR)
        
        # Calcul du SNR
        snr = AudioProcessor.compute_snr(filtered, self.noise_floor)
        
        # Détection multi-critères
        is_loud = level > self.threshold_low
        is_very_loud = level > self.threshold_high
        has_variation = level_variation > (self.noise_std * 2)
        good_snr = snr > 10  # SNR > 10 dB
        
        # Score de confiance amélioré
        confidence = 0
        if is_loud: confidence += 1
        if is_very_loud: confidence += 1
        if is_vad_speech: confidence += 2
        if has_variation: confidence += 1
        if good_snr: confidence += 1
        
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
            if confidence >= 3:
                if self.state_duration >= 3:  # 90ms de confirmation
                    self.state = SpeechState.SPEECH
                    action = "START_RECORDING"
                    self.total_detections += 1
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
                
        return action, level, confidence, snr
        
    def should_recalibrate(self):
        """Détermine si une recalibration est nécessaire"""
        # Recalibrer si trop de faux positifs
        if self.total_detections > 10 and self.false_positives / self.total_detections > 0.3:
            return True
        return False


# ─────────────── ENREGISTREUR AUDIO ────────────────────────────────
class AudioRecorder:
    def __init__(self):
        self.frames = []
        self.pre_buffer = deque(maxlen=10)  # 300ms de pré-buffer
        self.min_audio_level = 100  # Niveau minimum pour valider l'enregistrement (abaissé)
        
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
            return False
            
        # Calculer le niveau moyen
        total_level = 0
        for frame in self.frames:
            total_level += audioop.rms(frame, 2)
        avg_level = total_level / len(self.frames)
        
        return avg_level > self.min_audio_level
        
    def get_wav(self):
        """Retourne le WAV complet avec header, normalisé"""
        if not self.frames:
            return None
            
        # Concaténer et normaliser l'audio
        data = b"".join(self.frames)
        data = AudioProcessor.normalize_audio(data)
        
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


# ─────────────── SÉLECTEUR DE PÉRIPHÉRIQUE ────────────────────────
class DeviceSelector:
    """Sélectionne automatiquement le meilleur périphérique audio"""
    
    @staticmethod
    def find_best_microphone(pa):
        """Trouve le meilleur microphone disponible"""
        print("🔍 Recherche du meilleur microphone...")
        
        best_device = None
        best_score = -1
        
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            
            # Vérifier si c'est un périphérique d'entrée
            if info['maxInputChannels'] > 0:
                score = 0
                name = info['name'].lower()
                
                # Scoring basé sur le nom
                if 'usb' in name:
                    score += 3
                if 'microphone' in name or 'mic' in name:
                    score += 2
                if 'webcam' in name or 'camera' in name:
                    score -= 1  # Éviter les micros de webcam
                if 'default' in name:
                    score += 1
                    
                # Préférer les périphériques avec plus de canaux
                score += info['maxInputChannels']
                
                print(f"  [{i}] {info['name']} (score: {score})")
                
                if score > best_score:
                    best_score = score
                    best_device = i
                    
        if best_device is not None:
            selected = pa.get_device_info_by_index(best_device)
            print(f"✓ Microphone sélectionné: [{best_device}] {selected['name']}")
            return best_device
        
        return None


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

        # Initialisation TTS de secours
        self.tts_voice = None
        try:
            self.tts_voice = PiperVoice.load("TTS/fr_FR-siwis-medium.onnx")
            print("✓ TTS de secours initialisé")
        except Exception as e:
            print(f"⚠️ Impossible d'initialiser le TTS de secours: {e}")
        
        # Statistiques
        self.stats = {
            'recordings_sent': 0,
            'recordings_valid': 0,
            'total_duration': 0,
            'last_snr': 0
        }
        
        # Calibration initiale
        self.detector.calibrate(self.stream)
        
    def _setup_audio_input(self):
        """Configure le flux audio avec sélection automatique du micro"""
        try:
            # Trouver le meilleur microphone
            device_index = DeviceSelector.find_best_microphone(self.pa)
            
            self.stream = self.pa.open(
                rate=SR,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=device_index,
            )
            print("✓ Flux audio configuré")
        except Exception as e:
            print(f"❌ Erreur audio: {e}")
            raise
            
    def listen_forever(self):
        """Boucle principale d'écoute"""
        print("\n🎤 En attente de parole... (Ctrl+C pour arrêter)")
        print("   [·] Silence  [?] Peut-être  [●] Parole  [○] Fin")
        print("   [R] pour recalibrer manuellement\n")
        
        recording = False
        frames_since_start = 0
        last_calibration = time.time()
        
        while True:
            try:
                # Lecture audio
                pcm = self.stream.read(CHUNK, exception_on_overflow=False)
                
                # Ajout au pré-buffer
                if not recording:
                    self.recorder.add_to_prebuffer(pcm)
                
                # Analyse
                action, level, confidence, snr = self.detector.process_frame(pcm)
                self.stats['last_snr'] = snr
                
                # Affichage du niveau
                bar = '█' * int(level / 100)
                state_char = {
                    'SILENCE': '·', 
                    'MAYBE_SPEECH': '?', 
                    'SPEECH': '●', 
                    'ENDING': '○'
                }[self.detector.state.value]
                
                snr_indicator = f"SNR:{snr:4.1f}dB" if snr > 0 else "SNR: ---"
                print(f"[{state_char}] {level:5.0f} {bar:<30} {snr_indicator} C:{confidence}", end='\r')
                
                # Actions
                if action == "START_RECORDING":
                    print(f"\n🔊 Parole détectée ! Enregistrement... (SNR: {snr:.1f}dB)")
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
                        duration_ms = frames_since_start * FRAME_MS
                        
                        # Vérifier la durée et la validité
                        if frames_since_start >= self.detector.min_speech_frames:
                            if self.recorder.is_valid_recording():
                                print(f"\n✓ Enregistrement terminé ({duration_ms}ms)")
                                wav_data = self.recorder.get_wav()
                                if wav_data:
                                    self._send_and_play(wav_data)
                                    self.stats['recordings_valid'] += 1
                            else:
                                print("\n✗ Niveau audio trop faible, ignoré")
                                self.detector.false_positives += 1
                        else:
                            print("\n✗ Trop court, ignoré")
                            self.detector.false_positives += 1
                            
                        self.recorder.clear()
                        
                # Recalibration automatique toutes les 5 minutes
                if time.time() - last_calibration > 300:
                    if not recording and self.detector.should_recalibrate():
                        print("\n\n📊 Recalibration automatique...")
                        self.detector.calibrate(self.stream)
                        last_calibration = time.time()
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Erreur: {e}")
                recording = False
                self.recorder.clear()
                led.off()
                
        # Afficher les statistiques
        self._print_stats()
                
    def _send_and_play(self, wav_bytes: bytes):
        """Envoie l'audio au serveur et joue la réponse avec fallback TTS"""
        try:
            print("📡 Envoi au serveur...")
            self.stats['recordings_sent'] += 1
            
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
                    if not self.play_mp3(mp3_data) and self.tts_voice:
                        # Fallback TTS si lecture MP3 échoue
                        self._play_with_tts_fallback(response_data["answer"])
                elif self.tts_voice:
                    # Réponse texte seulement -> utiliser TTS
                    self._play_with_tts_fallback(response_data["answer"])
                
        except requests.exceptions.Timeout:
            print("❌ Timeout serveur")
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur réseau: {e}")
        except Exception as e:
            print(f"❌ Erreur: {e}")
            
    def play_mp3(self, mp3_bytes: bytes):
        """Lecture via l'API Music → haut-parleur PiCar-X avec fallback TTS"""
        print("🔊 Tentative de lecture MP3...")
        
        # Vérifier si des données audio sont présentes
        if not mp3_bytes or len(mp3_bytes) < 100:  # 100 bytes min
            print("⚠️ Données MP3 absentes ou trop courtes")
            return False
            
        # Créer un fichier temporaire
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                f.write(mp3_bytes)
                path = f.name
                print(f"📂 Fichier temporaire créé ({len(mp3_bytes)} bytes)")
                
            # Tentative de lecture
            try:
                print("▶️ Lancement de la lecture...")
                self.music.sound_play(path)
                print("✓ Lecture MP3 réussie")
                return True
            except Exception as e:
                print(f"❌ Échec lecture MP3: {e}")
                return False
        finally:
            try:
                os.remove(path)
            except:
                pass
            
    def recalibrate(self):
        """Recalibration manuelle du bruit"""
        print("\n� Recalibration manuelle...")
        self.detector.calibrate(self.stream)
        
    def _print_stats(self):
        """Affiche les statistiques de la session"""
        print("\n\n📊 Statistiques de la session:")
        print(f"  - Enregistrements envoyés: {self.stats['recordings_sent']}")
        print(f"  - Enregistrements valides: {self.stats['recordings_valid']}")
        if self.detector.total_detections > 0:
            accuracy = (1 - self.detector.false_positives / self.detector.total_detections) * 100
            print(f"  - Précision de détection: {accuracy:.1f}%")
        print(f"  - Dernier SNR: {self.stats['last_snr']:.1f} dB")
        
    def _play_with_tts_fallback(self, text: str):
        """Joue le texte via TTS de secours"""
        if not self.tts_voice:
            print("⚠️ TTS de secours non disponible")
            return
            
        print("🔊 Utilisation du TTS de secours...")
        
        try:
            # Créer un fichier WAV temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                path = f.name
                print(f"📂 Génération TTS pour: '{text}'")
                
                # Synthétiser la voix
                self.tts_voice.synthesize(text, f)
                
                # Lire le fichier généré
                print("▶️ Lecture TTS...")
                self.music.sound_play(path)
                print("✓ Lecture TTS réussie")
        except Exception as e:
            print(f"❌ Erreur TTS: {e}")
        finally:
            try:
                os.remove(path)
            except:
                pass

    def cleanup(self):
        """Nettoyage des ressources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
        led.off()


# ─────────────── MAIN ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 PiCar-X Voice Assistant - Version optimisée avec réduction de bruit")
    print("=" * 60)
    
    # Vérifier les dépendances
    try:
        import scipy
    except ImportError:
        print("⚠️  scipy n'est pas installé. Installation recommandée:")
        print("   pip install scipy")
        print("   Continuons sans le filtre passe-haut...")
        
    client = Client()
    
    try:
        client.listen_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt demandé")
    finally:
        client.cleanup()
        print("✓ Nettoyage terminé")
