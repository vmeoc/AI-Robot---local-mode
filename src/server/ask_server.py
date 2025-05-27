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

# ❷  STT multilingue (Whisper tiny-int8, GPU si dispo)
stt = WhisperModel("tiny", compute_type="int8")

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
        # 2) Transcription
        stt_start = time.time()
        segments, info = stt.transcribe(
            tmp_path,
            beam_size=2,
            language="fr",
        )
        stt_duration = time.time() - stt_start
        print(f"[STT TIME] {stt_duration:.2f}s")
    finally:
        os.unlink(tmp_path)  # supprime le WAV temporaire

    text = "".join(s.text for s in segments).strip()
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

# ───────────  Lancement direct  ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
