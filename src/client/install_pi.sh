#!/bin/bash
# Script d'installation pour Raspberry Pi 5
# Usage: bash install_pi.sh

echo "🤖 Installation du client PiCar-X Voice Assistant"
echo "================================================"

# Mise à jour du système
echo "📦 Mise à jour du système..."
sudo apt update

# Installation des dépendances système
echo "📦 Installation des dépendances système..."
sudo apt install -y \
    portaudio19-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    libasound2-dev \
    libatlas-base-dev \
    libopenblas-dev

# Création de l'environnement virtuel
echo "🐍 Création de l'environnement virtuel..."
python3 -m venv .venv
source .venv/bin/activate

# Installation des paquets Python
echo "📦 Installation des paquets Python..."
pip install --upgrade pip
pip install -r requirements_pi.txt

# Configuration audio (optionnel)
echo "🔊 Configuration audio..."
# Augmenter la priorité du processus audio
echo "@audio - rtprio 95" | sudo tee -a /etc/security/limits.conf
echo "@audio - memlock unlimited" | sudo tee -a /etc/security/limits.conf

# Ajouter l'utilisateur au groupe audio
sudo usermod -a -G audio $USER

echo "✅ Installation terminée !"
echo ""
echo "Pour lancer le client :"
echo "  source .venv/bin/activate"
echo "  python client.py"
echo ""
echo "⚠️  Redémarrez le Raspberry Pi pour appliquer les changements audio"
