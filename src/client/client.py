import os
import time
import threading
import webrtcvad
import pyaudio
import requests
from pvporcupine import Porcupine
from robot_hat import Pin
import subprocess
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())  # charge un éventuel fichier .env

# Configuration
API_ENDPOINT = "http://192.168.110.35:8000/ask"  # À modifier avec l'IP du serveur
API_TOKEN = os.getenv("API_TOKEN")  # Token d'authentification
WAKE_WORD = "Mars réveille toi"  # Mot de réveil
VAD_AGGRESSIVENESS = 3  # Niveau d'agressivité du VAD (1-3)
SILENCE_TIMEOUT = 0.5  # Temps de silence pour arrêter l'enregistrement (secondes)
SAMPLE_RATE = 16000  # Taux d'échantillonnage (Hz)
FRAME_DURATION = 30  # Durée d'une trame audio (ms)
CHUNK_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # Taille d'un chunk audio

# Initialisation des LEDs
led = Pin('LED')

class AudioRecorder:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.frames = []
        
    def start_recording(self):
        self.recording = True
        self.frames = []
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
    def stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        return b''.join(self.frames)
        
    def record_until_silence(self):
        self.start_recording()
        silent_frames = 0
        silence_threshold = int(SILENCE_TIMEOUT * 1000 / FRAME_DURATION)
        
        while self.recording:
            frame = self.stream.read(CHUNK_SIZE)
            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
            
            if is_speech:
                silent_frames = 0
                self.frames.append(frame)
            else:
                silent_frames += 1
                if silent_frames > silence_threshold:
                    break
                self.frames.append(frame)
                
        return self.stop_recording()

class Client:
    def __init__(self):
        self.recorder = AudioRecorder()
        self.porcupine = Porcupine(
            access_key=os.getenv("PORCUPINE_ACCESS_KEY"),
            keyword_paths=[os.path.join(os.path.dirname(__file__), "Mars-réveille-toi_fr_raspberry-pi_v3_0_0.ppn")]
        )
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )
        
    def listen_for_wake_word(self):
        while True:
            pcm = self.stream.read(self.porcupine.frame_length)
            result = self.porcupine.process(pcm)
            
            if result >= 0:  # Wake word détecté
                led.on()  # Allumer LED
                audio_data = self.recorder.record_until_silence()
                led.off()  # Éteindre LED
                self.process_audio(audio_data)
                
    def process_audio(self, audio_data):
        # Enregistrer temporairement le WAV
        temp_file = "/tmp/voice_command.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_data)
            
        # Envoyer au serveur
        try:
            response = requests.post(
                API_ENDPOINT,
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                files={"file": open(temp_file, "rb")},
                timeout=10
            )
            
            if response.status_code == 200:
                self.play_mp3(response.json()["audio"])
                
        except Exception as e:
            print(f"Erreur lors de la communication avec le serveur: {e}")
            
    def play_mp3(self, mp3_data):
        # Jouer le MP3 avec mpg123
        process = subprocess.Popen(
            ["mpg123", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        process.communicate(input=mp3_data)
        
    def cleanup(self):
        self.stream.close()
        self.audio.terminate()
        self.porcupine.delete()

def main():
    client = Client()
    try:
        client.listen_for_wake_word()
    except KeyboardInterrupt:
        pass
    finally:
        client.cleanup()

if __name__ == "__main__":
    main()
