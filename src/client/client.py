#!/usr/bin/env python3
"""
client.py – PiCar-X : écoute continue → enregistrement → envoi serveur → lecture réponse
Compatible Raspberry Pi 5 + Robot HAT v2.0
"""

import os
import time
import tempfile
import audioop
import math

import webrtcvad
import pyaudio
import requests
from robot_hat import Pin, Music
from dotenv import load_dotenv, find_dotenv

# ─────────────── CONFIG ────────────────────────────────────────────
load_dotenv(find_dotenv())            # charge .env s'il existe

API_ENDPOINT = "http://192.168.110.35:8000/ask"   # IP du serveur
API_TOKEN    = os.getenv("API_TOKEN")             # Bearer token
VAD_AGGR     = 1                                  # 1 = least aggressive (most strict)
SILENCE_TMO  = 1.5                                # arrêt après 1.5 s de silence
MIN_SPEECH_DURATION = 0.5                         # min seconds of speech to trigger
SR           = 16000                              # sample-rate
FRAME_MS     = 30                                 # longueur trame (ms)
CHUNK        = int(SR * FRAME_MS / 1000)          # = 480 échantillons

# LED sur le Robot HAT
led = Pin('LED')


# ─────────────── AUDIO RECORD ──────────────────────────────────────
class AudioRecorder:
    """VAD + buffer circulaire – renvoie un WAV brut (bytes)"""
    def __init__(self):
        self.vad   = webrtcvad.Vad(VAD_AGGR)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []

    def _open_stream(self):
        return self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SR,
            input=True,
            frames_per_buffer=CHUNK
        )

    def record_until_silence(self):
        self.frames = []
        self.stream = self._open_stream()

        # Add WAV header (44 bytes)
        self.frames.append(self._create_wav_header(0))  # Placeholder for final size

        silent_cnt   = 0
        silent_max   = int(SILENCE_TMO * 1000 / FRAME_MS)

        while True:
            frame = self.stream.read(CHUNK, exception_on_overflow=False)
            speech = self.vad.is_speech(frame, SR)

            if speech:
                silent_cnt = 0
                self.frames.append(frame)
            else:
                silent_cnt += 1
                if silent_cnt > silent_max:
                    break
                self.frames.append(frame)

        self.stream.stop_stream()
        self.stream.close()

        # Update WAV header with actual data size
        data = b"".join(self.frames[1:])  # Skip header
        header = self._create_wav_header(len(data))
        return header + data

    def _create_wav_header(self, data_size):
        """Generate proper WAV header for 16-bit mono PCM"""
        import struct
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            data_size + 36,  # Total file size - 8
            b'WAVE',
            b'fmt ',
            16,  # fmt chunk size
            1,   # PCM format
            1,   # Mono
            SR,  # Sample rate
            SR * 2,  # Byte rate (sample rate * bytes per sample)
            2,   # Block align (bytes per sample * channels)
            16,  # Bits per sample
            b'data',
            data_size
        )
        return header


# ─────────────── CLIENT PRINCIPAL ─────────────────────────────────
class Client:
    def __init__(self):
        # Music : gère la sortie haut-parleur PiCar-X
        self.music = Music()
        self.music.music_set_volume(70)            # 0-100

        self.recorder = AudioRecorder()
        self._setup_audio_input()


    # ---------------- Micro ----------------
    def _find_input_device(self):
        return 0                               # fonction annulée 
    def _setup_audio_input(self):
        self.pa = pyaudio.PyAudio()
        try:
            self.wk_stream = self.pa.open(
                rate=SR,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=self._find_input_device(),
            )
        except OSError:
            self.wk_stream = self.pa.open(
                rate=SR,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=CHUNK,
            )

    def listen_forever(self):
        print("🎤 En attente de parole…")
        threshold = 2000  # ajuste en fonction de ton micro
        min_speech_frames = int(MIN_SPEECH_DURATION * 1000 / FRAME_MS)
        speech_frames = 0
        
        while True:
            pcm = self.wk_stream.read(CHUNK, exception_on_overflow=False)

            # ← Nouveau bloc pour afficher que le micro capte bien quelque chose
            level = audioop.rms(pcm, 2)  # RMS sur des échantillons 16-bit
            bar = '#' * (level // threshold)
            print(f"[LEVEL {level:5d}] {bar}", end="\r")

            if self.recorder.vad.is_speech(pcm, SR):
                speech_frames += 1
                if speech_frames >= min_speech_frames:
                    print("\n🔊 Parole détectée !")
                    led.on()
                    wav = self.recorder.record_until_silence()
                    led.off()
                    self._send_and_play(wav)
                    speech_frames = 0
            else:
                speech_frames = 0

      # ---------------- API + playback ----------------
    def _send_and_play(self, wav_bytes: bytes):
        try:
            resp = requests.post(
                API_ENDPOINT,
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                files={"file": ("voice.wav", wav_bytes, "audio/wav")},
                timeout=20,
            )
            resp.raise_for_status()
            response_data = resp.json()
            if "audio" in response_data:
                mp3_data = bytes.fromhex(response_data["audio"])
                self.play_mp3(mp3_data)
            if "answer" in response_data:
                print(f"🤖: {response_data['answer']}")
        except Exception as exc:
            print("❌ Erreur serveur :", exc)

    def play_mp3(self, mp3_bytes: bytes):
        """Lecture via l’API Music → haut-parleur PiCar-X"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(mp3_bytes)
            path = f.name
        self.music.sound_play(path)
        os.remove(path)

    # ---------------- Clean exit ----------------
    def cleanup(self):
        self.wk_stream.close()
        self.pa.terminate()


# ─────────────── MAIN ─────────────────────────────────────────────
if __name__ == "__main__":
    client = Client()
    try:
        client.listen_forever()
    except KeyboardInterrupt:
        print("\nArrêt demandé par l’utilisateur.")
    finally:
        client.cleanup()
