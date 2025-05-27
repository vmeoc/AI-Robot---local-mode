# ask_server.py  – pipeline STT → LLM → TTS avec une seule voix Piper
import os
import io
import time
import random
import tempfile
import subprocess
import requests
import wave

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from faster_whisper import WhisperModel
import pydub

# ───────────────  CONFIG  ──────────────────────────────────────────
load_dotenv(find_dotenv())  # charge un éventuel fichier .env
API_TOKEN  = os.getenv("API_TOKEN", "")
print("[INFO] API_TOKEN chargé :", API_TOKEN[:8] if API_TOKEN else "<vide>")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"       # Ollama « robot-mistral »
VOICE_PATH = r"C:\Users\vince\Documents\VS Code\Dev\AI Robot - local mode\TTS\fr_FR-siwis-medium.onnx"
PIPER_TTS_EXE = r"C:\Users\vince\Documents\VS Code\Dev\AI Robot - local mode\.venv\Scripts\piper-tts.exe"
# ───────────────────────────────────────────────────────────────────

# ❶  Sécurité – fail-delay
def check_auth(hdr: str | None):
    if hdr is None or not hdr.startswith("Bearer "):
        _delay_fail()
    token = hdr.split(None, 1)[1]
    if token != API_TOKEN:
        _delay_fail()

def _delay_fail():
    time.sleep(random.uniform(2, 4))
    raise HTTPException(status_code=401, detail="unauthorized")

# ❷  STT multilingue (Whisper base, GPU si dispo)
# Utilisation du modèle "base" pour une meilleure précision
# compute_type="float16" pour plus de précision si GPU disponible, sinon "int8"
try:
    # Essayer d'utiliser float16 si GPU disponible
    stt = WhisperModel("small", compute_type="float16")
    print("[INFO] Whisper small chargé avec compute_type=float16 (GPU)")
except Exception:
    # Fallback sur int8 si pas de GPU
    stt = WhisperModel("small", compute_type="int8")
    print("[INFO] Whisper small chargé avec compute_type=int8 (CPU)")

# ❸  Appel Ollama
def llama(prompt: str) -> str:
    try:
        rsp = requests.post(
            OLLAMA_URL,
            json={
                "model": "mars-ia-llama3-8B-instruct-q4",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        rsp.raise_for_status()  # Lève une exception pour les erreurs HTTP (4xx ou 5xx)
        
        response_json = rsp.json()
        
        # Pour débogage, si nécessaire:
        # print(f"[LLAMA DEBUG] Réponse complète: {response_json}") 
        
        if "message" in response_json and isinstance(response_json["message"], dict) and "content" in response_json["message"]:
            return response_json["message"]["content"]
        elif "error" in response_json:
            error_message = response_json["error"]
            print(f"[LLAMA ERROR] Ollama a retourné une erreur: {error_message}")
            raise HTTPException(status_code=502, detail=f"Ollama API error: {error_message}")
        else:
            # Cas où la structure attendue n'est pas là, mais ce n'est pas non plus une clé "error" standard d'Ollama
            # Cela pourrait être une réponse valide mais différente, ou une erreur non standard.
            # Par exemple, si "choices" est utilisé à la place de "message" par certains modèles/versions d'Ollama.
            # Pour l'instant, on considère que c'est inattendu pour "robot-mistral".
            # On pourrait inspecter `response_json.keys()` pour voir ce qui est disponible.
            # Exemple: if "choices" in response_json and response_json["choices"] and "message" in response_json["choices"][0] ...
            print(f"[LLAMA ERROR] Structure de réponse inattendue d'Ollama: {response_json}")
            raise HTTPException(status_code=502, detail=f"Unexpected response structure from Ollama: {response_json}")
            
    except requests.exceptions.RequestException as e:
        print(f"[LLAMA ERROR] La requête vers Ollama a échoué: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: Could not connect to Ollama: {e}")
    except KeyError as e:
        # Cette capture spécifique de KeyError devrait être moins probable avec les vérifications ci-dessus,
        # mais elle reste une sécurité pour d'autres KeyError potentielles lors du parsing de rsp.json().
        response_text = rsp.text if 'rsp' in locals() and hasattr(rsp, 'text') else 'Response object or text not available'
        print(f"[LLAMA ERROR] Erreur lors du parsing de la réponse d'Ollama (KeyError: {e}). Réponse: {response_text}")
        raise HTTPException(status_code=502, detail=f"Error parsing Ollama response: KeyError {e}")
    except Exception as e: # Capture générique pour toute autre exception non prévue
        response_text = rsp.text if 'rsp' in locals() and hasattr(rsp, 'text') else 'Response object or text not available'
        print(f"[LLAMA ERROR] Erreur inattendue dans la fonction llama: {e}. Réponse: {response_text}")
        raise HTTPException(status_code=500, detail=f"Unexpected error in Llama call: {e}")

# ❹  TTS – unique voix FR (fonctionne même pour EN/ES)
def synthesize(text: str) -> bytes:
    from piper import PiperVoice
    import wave
    
    # Créer un buffer en mémoire pour le WAV
    wav_io = io.BytesIO()
    
    # Utiliser wave pour créer un fichier WAV valide
    with wave.open(wav_io, 'wb') as wav_file:
        voice = PiperVoice.load(VOICE_PATH)
        voice.synthesize(text, wav_file)
    
    # Convertir le WAV en MP3
    wav_io.seek(0)
    mp3 = (
        pydub.AudioSegment.from_wav(wav_io)
        .export(format="mp3")
        .read()
    )
    return mp3

# ───────────  API FastAPI  ─────────────────────────────────────────
app = FastAPI()

@app.post("/ask")
async def ask(
    file: UploadFile = File(...),
    authorization: str | None = Header(None),
):
    check_auth(authorization)

    # 1) Sauvegarde des fichiers entrants pour analyse
    os.makedirs("received_audio", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = f"received_audio/audio_{timestamp}.wav"
    
    audio_bytes = await file.read()
    with open(save_path, "wb") as f:
        f.write(audio_bytes)
    
    # Crée aussi un fichier temporaire pour Whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # 2) Transcription avec paramètres optimisés
        stt_start = time.time()
        segments, info = stt.transcribe(
            tmp_path,
            beam_size=5,  # Augmenté pour plus de précision
            language="fr",  # Forcer le français
            vad_filter=True,  # Activer le VAD interne
            vad_parameters=dict(
                threshold=0.5,  # Seuil de détection de parole
                min_speech_duration_ms=250,  # Durée minimale de parole
                max_speech_duration_s=float('inf'),  # Pas de limite max
                min_silence_duration_ms=1000,  # Silence minimum entre segments
                speech_pad_ms=400  # Padding autour de la parole détectée
            ),
            word_timestamps=False,  # Pas besoin des timestamps par mot
            condition_on_previous_text=True,  # Meilleure cohérence
            compression_ratio_threshold=2.4,  # Seuil de compression
            log_prob_threshold=-1.0,  # Seuil de probabilité log
            no_speech_threshold=0.6,  # Seuil de non-parole
            temperature=0.0,  # Pas de sampling aléatoire
            initial_prompt="Ceci est une conversation en français.",  # Aide le modèle
        )
        stt_duration = time.time() - stt_start
        
        # Afficher plus d'informations sur la transcription
        print(f"[STT TIME] {stt_duration:.2f}s")
        print(f"[STT INFO] Langue détectée: {info.language}, Probabilité: {info.language_probability:.2f}")
    finally:
        os.unlink(tmp_path)  # supprime le WAV temporaire

    # Assembler le texte avec gestion des segments vides
    text_segments = []
    for segment in segments:
        if segment.text.strip():  # Ignorer les segments vides
            text_segments.append(segment.text.strip())
    
    text = " ".join(text_segments)
    
    # Vérifier si on a bien capturé du texte
    if not text:
        print("[STT] Aucune parole détectée dans l'audio")
        return {"answer": "Je n'ai pas compris. Pouvez-vous répéter ?", "audio": ""}
    
    print(f"[STT {info.language}] {text}")

    # 3) LLM
    llm_start = time.time()
    answer = llama(text)
    llm_duration = time.time() - llm_start
    print(f"[LLM TIME] {llm_duration:.2f}s")
    print(f"[LLM] {answer}")

    # 4) TTS
    tts_start = time.time()
    mp3_bytes = synthesize(answer)
    tts_duration = time.time() - tts_start
    print(f"[TTS TIME] {tts_duration:.2f}s")
    return {"answer": answer, "audio": mp3_bytes.hex()}

# ───────────  Endpoint de test  ────────────────────────────────────
@app.get("/health")
async def health_check():
    """Endpoint pour vérifier que le serveur fonctionne"""
    return {
        "status": "ok",
        "whisper_model": "small",
        "llm_model": "mars-ia-llama3-8B-instruct-q4",
        "tts_voice": "fr_FR-siwis-medium"
    }

# ───────────  Lancement direct  ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Démarrage du serveur AI Robot")
    print("=" * 60)
    print(f"📍 Endpoint: http://0.0.0.0:8000/ask")
    print(f"🔐 Token requis: {'Oui' if API_TOKEN else 'Non'}")
    print(f"🎙️  Modèle STT: Whisper base")
    print(f"🤖 Modèle LLM: mars-ia-llama3-8B-instruct-q4")
    print(f"🔊 Voix TTS: fr_FR-siwis-medium")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
