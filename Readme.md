# 🧠🤖 Voice-Controlled PiCar-X with Local LLMs

Transform your [Sunfounder PiCar-X](https://www.sunfounder.com/products/picar-x?ref=luckyday&gad_source=1&gad_campaignid=22592763779&gbraid=0AAAAA_u_cfN7qILs3TPP89J_CodjDeyXX&gclid=Cj0KCQjwotDBBhCQARIsAG5pinOjfLEk2BrIwIBAsutfu-dz9eeVdjQR9jZwXNEfIrKJVoDinXrwccsaArTKEALw_wcB) into a smart, voice-activated robot powered by local Large Language Models (LLMs).
No cloud. No lag. Just fast, private AI at your fingertips.

---

## 🚀 Features

* **Wake word detection** — say the magic word to wake up your robot.
* **Voice request capture** — record your question directly from the robot.
* **On-premise processing** — audio is sent to your local computer for analysis.
* **LLM-powered intelligence** — local LLM (via [Ollama](https://ollama.com/)) generates a reply.
* **Text-to-speech (TTS)** — reply is converted to audio and played back on the robot.

All processing is done locally — ensuring fast responses and data privacy.

---

## 📂 Project Architecture

```plaintext
User → PiCar-X Mic → (client.py) → Local PC (ask_server.py + Ollama) → LLM → TTS → Audio → PiCar-X Speaker
```

---

## 📂 Components

### 👤 `ask_server.py` (runs on your local computer)

* Receives audio from PiCar-X
* Transcribes speech
* Sends prompt to LLM via Ollama
* Converts response into speech (TTS)
* Returns audio file to PiCar-X

> ✅ Requires [Ollama](https://ollama.com) with a compatible LLM installed (e.g. `llama3`, `mistral`, etc.)

---

### 🤖 `client.py` (runs on the PiCar-X / Raspberry Pi)

* Detects wake word using Porcupine
* Records voice input
* Sends audio to `ask_server.py`
* Receives audio reply and plays it through speaker

---

## 📦 Installation

### On Local Computer

1. Install [Ollama](https://ollama.com) and pull your preferred LLM.
2. Clone this repo and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the server:

   ```bash
   python ask_server.py
   ```

### On Raspberry Pi (PiCar-X)

1. Enable microphone and speaker support.
2. Install Python dependencies:

   ```bash
   sudo apt install portaudio19-dev
   python3 -m venv .venv
   source .venv/bin/activate
   pip install pvporcupine webrtcvad pyaudio requests
   ```
3. Run the client:

   ```bash
   python client.py
   ```

---

##

---

## 📃 License

MIT License

---

## 👌 Credits

* Based on the [PiCar-X by Sunfounder](https://www.sunfounder.com/products/picar-x)
* Wake-word detection via [Picovoice Porcupine](https://github.com/Picovoice/porcupine)
* Local LLM support powered by [Ollama](https://ollama.com)

---

> Questions or ideas? Contributions welcome!
