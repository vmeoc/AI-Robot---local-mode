# 🤖 AI Robot – Voice Interaction Project

## 1 · Overview

This repo contains the **minimal, fully‑local voice pipeline** for my PiCar‑X robot:

```
Pi 5 (wake‑word + VAD)  ──WAV──▶  FastAPI /ask  ──MP3──▶  Pi 5 loud‑speaker
                       (LAN < 10 ms)
        STT  Whisper‑base   ⇣      LLM  "robot‑mistral" (Ollama)
                                  ⇡
                       TTS  Piper (fr_FR‑siwis)
```

* **All computation stays on my network** – no OpenAI cloud calls.
* **Multilingual**: Whisper auto‑detects FR / EN / ES; the answer is voiced in Siwis‑FR for now.
* **Security**: every request needs a `Bearer` API\_TOKEN (+ fail‑delay).

---

## 2 · File map

| Path                           | Role                                                          |
| ------------------------------ | ------------------------------------------------------------- |
| `src/server/ask_server.py`     | FastAPI endpoint `/ask` → STT → LLM → TTS → MP3               |
| `src/client/client.py`         | Runs on the Pi 5 – wake‑word, VAD, POST wav, play mp3         |
| `.env`                         | Stores `API_TOKEN=` so the server never hard‑codes the secret |
| `TTS/fr_FR‑siwis‑medium.onnx`  | Single Piper voice (≈ 60 MB)                                  |
| `Models/**`                    | Ollama personality & system prompt                            |
| `src/client/AUDIO_IMPROVEMENTS.md` | Documentation des améliorations audio                     |

---

## 3 · ask\_server.py responsibilities

1. **Auth** – `check_auth()` validates `Authorization: Bearer <TOKEN>` and adds a 2‑4 s delay on bad tokens.
2. **STT** – writes the uploaded WAV to a temp file, runs `faster‑whisper` (base model, float16/int8, GPU if available).
3. **LLM** – calls Ollama (`model: mars-ia-llama3-8B-instruct-q4`) with the user text, gets a reply string.
4. **TTS** – pipes the reply into Piper (`fr_FR‑siwis‑medium.onnx`), converts stdout WAV → MP3 using *pydub + ffmpeg*.
5. **Response** – returns JSON `{ answer, audio }` where `audio` is the MP3 hex string.

> ⚠ Dependencies: `python-dotenv fastapi uvicorn faster-whisper[CUDA] piper-tts pydub ffmpeg` + Torch 2.5.1 cu121 and CTranslate2 GPU.

### Quick start (PC Windows)

```powershell
$Env:API_TOKEN = "<your‑token>"
pip install -r requirement.txt  # see versions pinned
ollama serve                    # mars-ia-llama3-8B-instruct-q4 must be pulled
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

## 4 · Client responsibilities (Raspberry Pi)

* **Audio Processing** – High-pass filter, normalization, SNR calculation
* **Smart VAD** – Multi-criteria detection with confidence scoring
* **Auto-calibration** – Adapts to ambient noise with percentile-based thresholds
* **Device Selection** – Automatically finds the best microphone
* **POST** to `/ask` with token header and normalized audio
* **Play** the returned MP3 via robot_hat Music API
* **Statistics** – Real-time SNR monitoring and session stats

---

## 5 · Current state

### ✅ Working

* ask_server.py boots, loads token from `.env`.
* Ollama & Whisper GPU run; ffmpeg installed.
* LLM communication works correctly with mars-ia-llama3-8B-instruct-q4 model.
* TTS with Piper fully operational using Python API.
* **NEW**: Client audio capture optimized with filtering and normalization
* **NEW**: Whisper base model for better STT accuracy
* **NEW**: Multi-criteria speech detection with SNR monitoring
* **NEW**: Automatic noise floor calibration
* **NEW**: Real-time audio quality indicators

### ⬜ To finish

| Item                     | Notes                                                 |
| ------------------------ | ----------------------------------------------------- |
| **Wake-word integration** | Re-enable Porcupine for hands-free activation       |
| **Voice variety**        | add EN/ES Piper voices + language→voice map.          |
| **Robo-commands**        | later: Rhino or LLM function-calling to drive motors. |

### 🎤 Audio Quality Improvements

1. **Signal Processing**:
   - High-pass filter (80Hz cutoff) removes low-frequency noise
   - Audio normalization ensures consistent levels
   - Real-time SNR calculation for quality monitoring

2. **Speech Detection**:
   - Multi-criteria scoring: audio level + VAD + variation + SNR
   - Adaptive thresholds based on noise percentiles
   - Pre-buffer captures speech onset
   - Automatic recalibration on high false-positive rate

3. **Whisper Optimization**:
   - Upgraded from `tiny` to `base` model
   - VAD filter enabled with tuned parameters
   - Beam size increased to 5
   - Initial prompt helps with French recognition

### Known Issues

1. **Piper-TTS Dependency Conflict** (Résolu):
   ```
   Solution: Utiliser piper-phonemize-fix à la place de piper-phonemize
   Commandes:
   pip uninstall piper-tts
   pip install piper-tts==1.2.0 --no-deps
   pip install piper-phonemize-fix==1.2.1
   ```

2. **Scipy Installation on Raspberry Pi**:
   ```bash
   # Install system dependencies first
   sudo apt-get install libatlas-base-dev gfortran
   # Then install scipy
   pip install scipy==1.10.1
   ```

### 🐞 Known issues

* `pydub` warns if ffmpeg isn't in PATH – solved by installing *Gyan FFmpeg*.
* `API_TOKEN` must be in env **before** launching the server, or .env + `python‑dotenv`.
* Upload must be a **wav** file; other formats will need an ffmpeg decode step.

---

## 6 · Next steps

1. **Hardware**: Test with USB microphone for better audio quality (recommended: Blue Yeti Nano, Samson Go Mic)
2. **Local STT**: Consider Vosk for on-device transcription to reduce latency
3. **Wake Word**: Re-integrate Porcupine with improved audio pipeline
4. **Voice Commands**: Add command intents (Rhino) or Ollama function‑calling for robot motion

### Usage Tips

* Monitor SNR in real-time - aim for > 10 dB
* Speak 20-30cm from microphone
* Run `update_audio_system.sh` on Pi for optimal setup
* Check `AUDIO_IMPROVEMENTS.md` for detailed documentation

Happy hacking! 🚀
