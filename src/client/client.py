#!/usr/bin/env python3
"""
client.py – PiCar-X : écoute mot-clé → enregistrement → envoi serveur → lecture réponse
Compatible Raspberry Pi 5 + Robot HAT v2.0
"""

import os
import time
import tempfile

import webrtcvad
import pyaudio
import requests
import pvporcupine            # plus besoin de LIBRARY_PATH / MODEL_PATH
from pvporcupine import Porcupine
from robot_hat import Pin, Music
from dotenv import load_dotenv, find_dotenv

# ─────────────── CONFIG ────────────────────────────────────────────
load_dotenv(find_dotenv())            # charge .env s'il existe

API_ENDPOINT = "http://192.168.110.35:8000/ask"   # IP du serveur
API_TOKEN    = os.getenv("API_TOKEN")             # Bearer token
WAKE_WORD    = "Mars réveille toi"                # label descriptif
VAD_AGGR     = 3                                  # 1 (doux) → 3 (très agressif)
SILENCE_TMO  = 0.5                                # arrêt après 0,5 s de silence
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
        return b"".join(self.frames)


# ─────────────── CLIENT PRINCIPAL ─────────────────────────────────
class Client:
    def __init__(self):
        # Music : gère la sortie haut-parleur PiCar-X
        self.music = Music()
        self.music.music_set_volume(70)            # 0-100

        self.recorder = AudioRecorder()
        self._setup_wake_word()
        self._setup_audio_input()

    # ---------------- Wake-word ----------------
    def _setup_wake_word(self):
        access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        if not access_key:
            raise ValueError("PORCUPINE_ACCESS_KEY manquant dans .env")

        keyword_path = os.path.join(
            os.path.dirname(__file__),
            "Mars-réveille-toi_fr_raspberry-pi_v3_0_0.ppn"
        )
        model_path = os.path.join(
            os.path.dirname(__file__),
            "porcupine_params_fr.pv"
        )

        if not os.path.exists(keyword_path):
            raise FileNotFoundError(f"Wake-word absent : {keyword_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle absent : {model_path}")

        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[keyword_path],
            model_path=model_path,
            sensitivities=[0.5],
        )


    # ---------------- Micro ----------------
    def _find_input_device(self):
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            name = pa.get_device_info_by_index(i)["name"].lower()
            if "mic" in name or "i2s" in name:
                return i
        return None                               # laisser PyAudio choisir
def _setup_audio_input(self):
        self.pa = pyaudio.PyAudio()
        frames_per_buf = self.porcupine.frame_length // 2
        try:
            self.wk_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=frames_per_buf,
                input_device_index=self._find_input_device(),
            )
        except OSError:
            self.wk_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=frames_per_buf,
            )

    def listen_forever(self):
        print("💤 En attente du mot-clé…")
        audio_frames = self.porcupine.frame_length // 2
        while True:
            pcm = self.wk_stream.read(audio_frames,
                                      exception_on_overflow=False)
            # pcm est désormais exactement 512 octets
            if self.porcupine.process(pcm) >= 0:
                print("🔊 Wake-word détecté !")
                led.on()
                wav = self.recorder.record_until_silence()
                led.off()
                self._send_and_play(wav)

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
            mp3_data = resp.json()["audio"]        # ← le serveur renvoie les octets
            self.play_mp3(mp3_data)
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
        self.porcupine.delete()


# ─────────────── MAIN ─────────────────────────────────────────────
if __name__ == "__main__":
    client = Client()
    try:
        client.listen_forever()
    except KeyboardInterrupt:
        print("\nArrêt demandé par l’utilisateur.")
    finally:
        client.cleanup()
