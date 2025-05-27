# 🎤 Améliorations Audio pour AI Robot

## Vue d'ensemble

Ce document décrit les améliorations apportées au système de capture audio et de transcription STT (Speech-to-Text) pour améliorer la qualité de reconnaissance vocale sur Raspberry Pi 5.

## 🚀 Améliorations Principales

### 1. **Client (Raspberry Pi) - Capture Audio Améliorée**

#### Traitement du Signal
- **Filtre passe-haut** : Élimine les basses fréquences parasites (< 80Hz)
- **Normalisation audio** : Ajuste automatiquement le niveau audio
- **Calcul du SNR** : Mesure en temps réel du rapport signal/bruit

#### Détection de Parole Optimisée
- **Calibration améliorée** : Utilise des percentiles pour mieux s'adapter au bruit ambiant
- **Score de confiance multi-critères** :
  - Niveau audio (seuils adaptatifs)
  - WebRTC VAD
  - Variation du signal
  - Rapport signal/bruit (SNR)
- **Pré-buffer** : Capture 300ms avant le début détecté de la parole

#### Sélection Automatique du Microphone
- Détecte et sélectionne automatiquement le meilleur microphone disponible
- Préférence pour les microphones USB externes
- Évite les microphones de webcam (qualité inférieure)

#### Statistiques et Monitoring
- Affichage en temps réel du SNR
- Statistiques de session (précision de détection)
- Recalibration automatique si trop de faux positifs

### 2. **Serveur - Whisper Base**

#### Modèle Amélioré
- Passage de `tiny` à `base` pour une meilleure précision
- Support GPU avec `float16` si disponible
- Fallback automatique sur CPU avec `int8`

#### Paramètres Optimisés
- **Beam size** : Augmenté à 5 pour plus de précision
- **VAD interne** : Activé avec paramètres optimisés
- **Initial prompt** : Guide le modèle pour le français
- **Gestion des segments vides** : Filtre automatique

## 📊 Indicateurs de Qualité

### SNR (Signal-to-Noise Ratio)
- **< 5 dB** : Qualité très faible, transcription difficile
- **5-10 dB** : Qualité acceptable
- **10-15 dB** : Bonne qualité
- **> 15 dB** : Excellente qualité

### Niveaux Audio
- **Bruit médian** : Niveau de référence du bruit ambiant
- **Seuil bas** : Niveau minimum pour détecter la parole
- **Seuil haut** : Niveau pour une détection certaine

## 🛠️ Installation

### Sur le Raspberry Pi

1. **Exécuter le script de mise à jour** :
   ```bash
   cd src/client
   chmod +x update_audio_system.sh
   ./update_audio_system.sh
   ```

2. **Ou manuellement** :
   ```bash
   # Installer les dépendances système
   sudo apt-get update
   sudo apt-get install -y libatlas-base-dev gfortran alsa-utils
   
   # Installer les dépendances Python
   pip install -r requirements_pi.txt
   ```

### Sur le Serveur

1. **Télécharger le modèle Whisper base** :
   ```bash
   # Le modèle sera téléchargé automatiquement au premier lancement
   python ask_server.py
   ```

## 🎯 Conseils d'Utilisation

### Configuration Matérielle
1. **Microphone USB externe** : Fortement recommandé
   - Blue Yeti Nano
   - Samson Go Mic
   - Tout microphone USB de qualité

2. **Positionnement** :
   - 20-30 cm de la bouche
   - Éviter les sources de bruit (ventilateurs, etc.)
   - Utiliser un support anti-vibration si possible

### Utilisation du Client

1. **Calibration** :
   - Au démarrage, restez silencieux pendant 3 secondes
   - La calibration s'adapte au bruit ambiant

2. **Indicateurs visuels** :
   - `[·]` : Silence
   - `[?]` : Détection possible
   - `[●]` : Enregistrement en cours
   - `[○]` : Fin de parole détectée

3. **Monitoring** :
   - Surveillez le SNR en temps réel
   - Si SNR < 10 dB, améliorez l'environnement audio

## 🐛 Dépannage

### Problème : "Niveau audio trop faible"
- Rapprochez-vous du microphone
- Vérifiez le volume d'entrée : `alsamixer`
- Utilisez un microphone externe

### Problème : "Trop de faux positifs"
- Le système recalibrera automatiquement
- Réduisez les sources de bruit ambiant
- Ajustez manuellement les seuils dans le code si nécessaire

### Problème : "Transcription incorrecte"
- Vérifiez le SNR (doit être > 10 dB)
- Parlez plus clairement et lentement
- Assurez-vous que le serveur utilise bien Whisper base

## 📈 Performances Attendues

### Avant les améliorations
- Détection de parole peu fiable
- Transcriptions souvent incorrectes
- Sensible au bruit ambiant

### Après les améliorations
- Détection de parole précise à 90%+
- Transcriptions correctes avec SNR > 10 dB
- Adaptation automatique au bruit ambiant
- Temps de réponse optimisé

## 🔄 Prochaines Étapes Possibles

1. **STT Local** : Implémenter Vosk sur le Pi pour réduire la latence
2. **Microphone Directionnel** : Utiliser un array de microphones
3. **Annulation d'Écho** : Pour éviter les boucles audio
4. **Modèle Whisper Large** : Si les ressources serveur le permettent
