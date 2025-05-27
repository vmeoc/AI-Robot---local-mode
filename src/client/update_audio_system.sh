#!/bin/bash
# Script de mise à jour du système audio pour Raspberry Pi 5
# Amélioration de la qualité STT

echo "🔧 Mise à jour du système audio pour AI Robot"
echo "============================================"

# Vérifier si on est sur un Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "⚠️  Ce script est conçu pour Raspberry Pi"
    exit 1
fi

# Mise à jour du système
echo "📦 Mise à jour des paquets système..."
sudo apt-get update

# Installation des dépendances pour scipy
echo "📦 Installation des dépendances pour scipy..."
sudo apt-get install -y libatlas-base-dev gfortran

# Installation des outils audio
echo "📦 Installation des outils audio..."
sudo apt-get install -y alsa-utils pulseaudio pulseaudio-utils

# Configuration ALSA pour optimiser l'audio
echo "🎤 Configuration ALSA..."
cat > ~/.asoundrc << 'EOF'
# Configuration ALSA optimisée pour l'enregistrement
pcm.!default {
    type asym
    playback.pcm "plughw:0,0"
    capture.pcm "mic"
}

pcm.mic {
    type plug
    slave {
        pcm "hw:1,0"  # Ajuster selon votre configuration
        rate 16000
        channels 1
        format S16_LE
    }
}

# Réduction du bruit matériel
pcm.mic_boost {
    type softvol
    slave.pcm "mic"
    control {
        name "Mic Boost"
        card 1
    }
    min_dB -5.0
    max_dB 20.0
    resolution 100
}
EOF

# Test des périphériques audio
echo "🔍 Périphériques audio détectés:"
arecord -l

# Mise à jour des dépendances Python
echo "📦 Mise à jour des dépendances Python..."
pip install --upgrade pip
pip install --upgrade -r requirements_pi.txt

# Test du microphone
echo "🎤 Test du microphone..."

# Détection du bon périphérique
DEFAULT_DEVICE=$(arecord -l | grep -E "card [0-9]+" | head -n1 | sed -E 's/card ([0-9]+):.*/\1/')
if [ -z "$DEFAULT_DEVICE" ]; then
    echo "⚠️  Aucun périphérique audio trouvé"
    echo "   Vérifiez que votre microphone est bien connecté"
else
    echo "   Utilisation du périphérique: card $DEFAULT_DEVICE"
    echo "   Parlez maintenant pour tester le niveau audio (5 secondes)..."
    arecord -D plughw:${DEFAULT_DEVICE},0 -f S16_LE -r 16000 -c 1 -d 5 test_audio.wav 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ Enregistrement terminé"
        
        # Analyse du niveau audio
        if [ -f test_audio.wav ]; then
            if command -v sox &> /dev/null; then
                echo "📊 Analyse du niveau audio:"
                sox test_audio.wav -n stats 2>&1 | grep -E "Maximum amplitude|RMS"
            else
                echo "ℹ️  Installez sox pour l'analyse audio: sudo apt-get install sox"
            fi
            # Nettoyage
            rm -f test_audio.wav
        fi
    else
        echo "⚠️  Erreur lors de l'enregistrement. Essayez avec un microphone USB externe."
    fi
fi

echo ""
echo "✅ Mise à jour terminée!"
echo ""
echo "📋 Prochaines étapes:"
echo "1. Redémarrez le client: python3 client.py"
echo "2. Le système va automatiquement calibrer le bruit ambiant"
echo "3. Surveillez les indicateurs SNR (Signal-to-Noise Ratio)"
echo ""
echo "💡 Conseils pour améliorer la qualité:"
echo "- Utilisez un microphone USB externe si possible"
echo "- Éloignez le microphone des sources de bruit"
echo "- Parlez clairement à 20-30cm du microphone"
echo "- Un SNR > 10dB est recommandé pour une bonne transcription"
