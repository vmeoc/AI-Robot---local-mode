# 🧠🤖 Voice-Controlled PiCar-X robot with Local LLMs

Transform your [Sunfounder PiCar-X](https://www.sunfounder.com/products/picar-x?ref=luckyday&gad_source=1&gad_campaignid=22592763779&gbraid=0AAAAA_u_cfN7qILs3TPP89J_CodjDeyXX&gclid=Cj0KCQjwotDBBhCQARIsAG5pinOjfLEk2BrIwIBAsutfu-dz9eeVdjQR9jZwXNEfIrKJVoDinXrwccsaArTKEALw_wcB) into a smart, voice-activated robot powered by local Large Language Models (LLMs).
No cloud. No lag. Just fast, private AI at your fingertips.
Can also be used to turn your Raspberry Pi in a google home like powered by your LLM of choice

---

## 🚀 Features

* **Voice request capture** — record your question directly from the robot.
* **On-premise processing** — audio is sent to your local computer for analysis.
* **LLM-powered intelligence** — local LLM (via [Ollama](https://ollama.com/)) generates a reply (Mistral, Gemma, Llama, etc..).
* **Text-to-speech (TTS)** — reply is converted to audio and played back on the robot.
* **Robot Actions** — LLM can command the robot to perform pre-defined actions (e.g., wave hands, nod, express emotions) synchronized with its speech.

All processing is done locally — ensuring fast responses and data privacy.

---

## 📂 Project Architecture

```plaintext
User → PiCar-X Mic → (client.py) → Local PC (ask_server.py + Ollama) → LLM (generates Text + Actions) → TTS (for Text) → Server sends (Audio + Action List) → PiCar-X (Speaker plays Audio, client.py executes Actions)
```

---

## 📂 Components

### 👤 `ask_server.py` (runs on your local computer)

* Receives audio from PiCar-X
* Transcribes speech
* Sends prompt to LLM via Ollama
* Extracts text and action commands from LLM's JSON response
* Converts textual response into speech (TTS)
* Returns both the audio file and the list of actions to PiCar-X

> ✅ Requires [Ollama](https://ollama.com) with a compatible LLM installed (e.g. `llama3`, `mistral`, etc.)

---

### 🤖 `client.py` (runs on the PiCar-X / Raspberry Pi)

* Detects sound environment and caliber the mic input settings
* Records voice input
* Sends audio to `ask_server.py`
* Receives audio reply and a list of action commands
* Plays the audio reply through the speaker
* Executes the received robot actions using `preset_actions.py`

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
1.1.
Create `/root/.asoundrc` or `/etc/asound.conf` with (see `Examples/.asoundrc`):
############################################################
# 1) Mixeur logiciel pour le playback sur le Robot HAT
pcm.dmixer {
    type     dmix
    ipc_key  1024
    slave {
        pcm         "hw:0,0"    # HAT = card 0, device 0
        rate        48000       # fréquence native du DAC
        period_size 1024
        buffer_size 4096
    }
}

############################################################
# 2) Plug pour la capture USB Mic
#    convertit 16 kHz → 48 kHz pour le hardware
pcm.capplug {
    type plug
    slave {
        pcm    "hw:1,0"    # Micro USB = card 1, device 0
        rate   48000       # fréquence native du mic
    }
}

############################################################
# 3) Périphérique asymétrique (playback vs capture)
pcm.asym {
    type          asym
    playback.pcm  "dmixer"
    capture.pcm   "capplug"
}

############################################################
# 4) Définit 'default' sur cet asynchrone avec resampling
pcm.!default {
    type     plug
    slave.pcm "asym"
}

ctl.!default {
    type hw
    card 0            # contrôle global sur la carte 0 (HAT)
}

2. Install Python dependencies:

   ```bash
   sudo apt install portaudio19-dev
   python3 -m venv .venv
   source .venv/bin/activate
   pip install pvporcupine webrtcvad pyaudio requests
   ```
3. Run the client in the root session:

   ```bash
   python client.py
   # Optional: specify an alternate microphone index
   python client.py --input-device 1
   ```

   The client uses the capture device defined in your `.asoundrc` by default.
   See `Examples/.asoundrc` for a ready-to-use configuration. Use
   `--input-device` to override the PyAudio index when required.

---
usage: client.py [-h] [--with-movements]
---

##
Read Project.md for more information about the project technical details and progresss status.
##

---

## 📃 License

MIT License

---

## 👌 Credits

* Based on the [PiCar-X by Sunfounder](https://www.sunfounder.com/products/picar-x)

* Local LLM support powered by [Ollama](https://ollama.com)

---

> Questions or ideas? Contributions welcome!
