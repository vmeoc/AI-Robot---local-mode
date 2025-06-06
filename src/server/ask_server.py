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
import json

# ───────────────  CONFIG  ──────────────────────────────────────────
load_dotenv(find_dotenv())  # charge un éventuel fichier .env
API_TOKEN  = os.getenv("API_TOKEN", "")
print("[INFO] API_TOKEN chargé :", API_TOKEN[:8] if API_TOKEN else "<vide>")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"       # Ollama « robot-mistral »
VOICE_PATH = r"C:\Users\vince\Documents\VS Code\Dev\AI Robot - local mode\TTS\fr_FR-siwis-medium.onnx"
PIPER_TTS_EXE = r"C:\Users\vince\Documents\VS Code\Dev\AI Robot - local mode\.venv\Scripts\piper-tts.exe"
LLM= "mars-ia-llama3-8B-instruct-q4" #mars-ia-llama3-8B-instruct-q4 ou gemma3:1b
SYSTEM_PROMPT = """Tu es un robot serviable et amusant avec des roues, conçu pour interagir avec des enfants.
Réponds TOUJOURS en fournissant un objet JSON valide. Cet objet JSON DOIT contenir deux clés :
1. \"answer_text\": une chaîne de caractères contenant la réponse textuelle que tu donneras à voix haute. Cette réponse doit être courte, engageante, et adaptée aux enfants. N'utilise pas d'onomatopées ou de symboles comme '*' dans ce texte.
2. "actions_list": une liste de chaînes de caractères. Chaque chaîne est le nom d'une action que le robot doit effectuer. Les actions possibles sont EXACTEMENT : "shake head", "nod", "wave hands", "resist", "act cute", "rub hands", "think", "twist body", "celebrate", "depressed", "honking", "start engine". Choisis parmi cette liste. Si aucune action n'est appropriée, retourne une liste vide [].

Exemple de format de réponse attendu :
{
  \"answer_text\": \"Bonjour les amis ! Prêts à jouer ?\",
  "actions_list": ["wave hands"]
}
Ou, si aucune action n'est nécessaire :
{
  \"answer_text\": \"Je suis un robot très intelligent.\",
  \"actions_list\": []
}
Assure-toi que ta réponse est UNIQUEMENT cet objet JSON, sans texte avant ou après."""

# Conversation history
conversation_history = []
MAX_HISTORY_TURNS = 5  # Number of user/assistant pairs to keep
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
def llama(prompt: str) -> tuple[str, list]:
    global conversation_history # Declare intent to modify global variable

    # Prepare messages for Ollama, including history
    messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages_payload.extend(conversation_history)
    messages_payload.append({"role": "user", "content": prompt})

    try:
        rsp = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM,
                "messages": messages_payload,
                "stream": False,
                "format": "json" # Request JSON format from Ollama if supported
            },
            timeout=60,
        )
        rsp.raise_for_status()  # Lève une exception pour les erreurs HTTP (4xx ou 5xx)
        
        response_json_ollama = rsp.json()
        
        if "message" in response_json_ollama and isinstance(response_json_ollama["message"], dict) and "content" in response_json_ollama["message"]:
            assistant_response_str = response_json_ollama["message"]["content"]
            
            final_answer_text = "Je ne sais pas quoi répondre."
            final_actions_list = []
            try:
                parsed_llm_output = json.loads(assistant_response_str)
                final_answer_text = parsed_llm_output.get("answer_text", "Pardon, je n'ai pas réussi à formuler une réponse claire.")
                final_actions_list = parsed_llm_output.get("actions_list", [])
                if not isinstance(final_actions_list, list):
                    print(f"[LLAMA WARNING] 'actions_list' from LLM was not a list: {final_actions_list}. Defaulting to empty list.")
                    final_actions_list = []
            except json.JSONDecodeError:
                print(f"[LLAMA ERROR] Failed to parse LLM JSON response: {assistant_response_str}")
                final_answer_text = assistant_response_str # Fallback to using the raw response as text
                final_actions_list = []
            except AttributeError: # Handles if parsed_llm_output is not a dict
                print(f"[LLAMA ERROR] LLM response was valid JSON but not a dictionary: {assistant_response_str}")
                final_answer_text = str(parsed_llm_output) if 'parsed_llm_output' in locals() else assistant_response_str
                final_actions_list = []

            # Add current exchange to history (using the textual part for clarity)
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": final_answer_text})
            
            # Prune history if it's too long
            if len(conversation_history) > MAX_HISTORY_TURNS * 2:
                conversation_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]
                
            return final_answer_text, final_actions_list
        elif "error" in response_json_ollama:
            error_message = response_json_ollama["error"]
            print(f"[LLAMA ERROR] Ollama a retourné une erreur: {error_message}")
            raise HTTPException(status_code=502, detail=f"Ollama API error: {error_message}")
        else:
            print(f"[LLAMA ERROR] Structure de réponse inattendue d'Ollama: {response_json_ollama}")
            raise HTTPException(status_code=502, detail=f"Unexpected response structure from Ollama: {response_json_ollama}")
            
    except requests.exceptions.RequestException as e:
        print(f"[LLAMA ERROR] La requête vers Ollama a échoué: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: Could not connect to Ollama: {e}")
    except KeyError as e:
        response_text = rsp.text if 'rsp' in locals() and hasattr(rsp, 'text') else 'Response object or text not available'
        print(f"[LLAMA ERROR] Erreur lors du parsing de la réponse d'Ollama (KeyError: {e}). Réponse: {response_text}")
        raise HTTPException(status_code=502, detail=f"Error parsing Ollama response: KeyError {e}")
    except Exception as e:
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
        return {"answer": "Je n'ai pas compris. Pouvez-vous répéter ?", "actions": [], "audio": ""}
    
    print(f"[STT {info.language}] {text}")

    # 3) LLM
    llm_start = time.time()
    answer_text, actions_list = llama(text)
    llm_duration = time.time() - llm_start
    print(f"[LLM TIME] {llm_duration:.2f}s")
    print(f"[LLM] Answer: {answer_text}, Actions: {actions_list}")

    # 4) TTS
    tts_start = time.time()
    mp3_bytes = synthesize(answer_text)
    tts_duration = time.time() - tts_start
    print(f"[TTS TIME] {tts_duration:.2f}s")
    return {"answer": answer_text, "actions": actions_list, "audio": mp3_bytes.hex()}

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
    print(f"🤖 Modèle LLM: {LLM}")
    print(f"🔊 Voix TTS: fr_FR-siwis-medium")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
