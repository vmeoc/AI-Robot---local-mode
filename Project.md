# 🤖 AI Robot – Voice Interaction Project

## 1 · Overview

This repo contains the **minimal, fully‑local voice pipeline** for my PiCar‑X robot:

```
Pi 5 (wake‑word + VAD)  ──WAV──▶  FastAPI /ask  ──MP3──▶  Pi 5 loud‑speaker
                       (LAN < 10 ms)
        STT  Whisper‑tiny   ⇣      LLM  “robot‑mistral” (Ollama)
                                  ⇡
                       TTS  Piper (fr_FR‑siwis)
```

* **All computation stays on my network** – no OpenAI cloud calls.
* **Multilingual**: Whisper auto‑detects FR / EN / ES; the answer is voiced in Siwis‑FR for now.
* **Security**: every request needs a `Bearer` API\_TOKEN (+ fail‑delay).

---

## 2 · File map

| Path                           | Role                                                          |
| ------------------------------ | ------------------------------------------------------------- |
| `sc/server/ask_server.py`                | FastAPI endpoint `/ask` → STT → LLM → TTS → MP3               |
| `src/client/client.py` *(to be finished)* | Runs on the Pi 5 – wake‑word, VAD, POST wav, play mp3         |
| `.env`                         | Stores `API_TOKEN=` so the server never hard‑codes the secret |
| `TTS/fr_FR‑siwis‑medium.onnx`  | Single Piper voice (≈ 60 MB)                                  |
| `Models/**`      | Ollama personality & system prompt                            |

---

## 3 · ask\_server.py responsibilities

1. **Auth** – `check_auth()` validates `Authorization: Bearer <TOKEN>` and adds a 2‑4 s delay on bad tokens.
2. **STT** – writes the uploaded WAV to a temp file, runs `faster‑whisper` (tiny INT8, GPU if available).
3. **LLM** – calls Ollama (`model: robot‑mistral`) with the user text, gets a reply string.
4. **TTS** – pipes the reply into Piper (`fr_FR‑siwis‑medium.onnx`), converts stdout WAV → MP3 using *pydub + ffmpeg*.
5. **Response** – returns JSON `{ answer, audio }` where `audio` is the MP3 hex string.

> ⚠ Dependencies: `python-dotenv fastapi uvicorn faster-whisper[CUDA] piper-tts pydub ffmpeg` + Torch 2.5.1 cu121 and CTranslate2 GPU.

### Quick start (PC Windows)

```powershell
$Env:API_TOKEN = "<your‑token>"
pip install -r requirement.txt  # see versions pinned
ollama serve                    # robot‑mistral must be pulled
python ask_server.py            # launches on http://localhost:8000
```

### Test with curl.exe

```powershell
curl.exe -X POST http://localhost:8000/ask ^
 -H "Authorization: Bearer %API_TOKEN%" ^
 -F "file=@test.wav" > reply.json
python utils\play_json.py reply.json  # writes response.mp3 & plays it
```

---

## 4 · Client responsibilities (Raspberry Pi)

* **Wake‑word** with Porcupine: triggers "Hey Mars".
* **VAD capture** (webrtcvad) until 500 ms silence.
* **POST** to `/ask` with token header.
* **Play** the returned MP3 (`mpg123 -`).
* **Loop** back to listening (disable wake‑word while audio is playing).

Implementation stub is in `client.py`; next tasks:

1. Glue Porcupine + VAD + HTTP.
2. Add LED "listening" feedback.

---

## 5 · Current state

### ✅ Working

* ask_server.py boots, loads token from `.env`.
* Ollama & Whisper GPU run; ffmpeg installed.
* LLM communication works correctly with mars-ia-mistral-nemo model.
* TTS with Piper fully operational using Python API.


### ⬜ To finish

| Item                     | Notes                                                 |
| ------------------------ | ----------------------------------------------------- |
| **Client loop on Pi**    | record, send, play – needs final glue.                |
| **Continuous wake-word** | disable during playback to avoid self-trigger.        |
| **Voice variety**        | add EN/ES Piper voices + language→voice map.          |
| **Robo-commands**        | later: Rhino or LLM function-calling to drive motors. |


### Known Issues

1. **Piper-TTS Dependency Conflict** (Résolu):
   ```
   Solution: Utiliser piper-phonemize-fix à la place de piper-phonemize
   Commandes:
   pip uninstall piper-tts
   pip install piper-tts==1.2.0 --no-deps
   pip install piper-phonemize-fix==1.2.1
   ```

2. **Utilisation de Piper TTS**:
   ```python
   from piper import PiperVoice
   import wave
   
   # Charger la voix
   voice = PiperVoice.load("chemin/vers/voix.onnx")
   
   # Synthétiser un échantillon audio
   with wave.open("output.wav", "wb") as wav_file:
       voice.synthesize("Texte à synthétiser", wav_file)
   ```


### 🐞 Known issues

* `pydub` warns if ffmpeg isn’t in PATH – solved by installing *Gyan FFmpeg*.
* `API_TOKEN` must be in env **before** launching the server, or .env + `python‑dotenv`.
* Upload must be a **wav** file; other formats will need an ffmpeg decode step.

---

## 6 · Next steps

1. Finish `client.py` and test the full LAN round‑trip.
2. Benchmark latency & tokens/s, tweak Whisper size (tiny → base) if accuracy too low.
3. Introduce Piper EN/ES or switch to Orca if we need faster TTS.
4. Add command intents (Rhino) or Ollama function‑calling for robot motion.

Happy hacking! 🚀
