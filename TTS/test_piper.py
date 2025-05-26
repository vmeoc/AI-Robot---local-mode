from piper import PiperVoice

# Initialiser la voix
voice = PiperVoice.load("TTS/fr_FR-siwis-medium.onnx")

# Synthétiser un échantillon audio
import wave
with wave.open("test.wav", "wb") as wav_file:
    voice.synthesize("Installation de Piper TTS réussie.", wav_file)

print("Synthèse vocale terminée - vérifiez test.wav")
