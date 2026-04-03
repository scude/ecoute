pip install torch torchaudio silero-vad faster-whisper streamlit pandas

# Optionnel: réduction de bruit RNNoise
sudo apt update
sudo apt install -y git autoconf automake libtool build-essential pkg-config
cd ~
git clone https://github.com/xiph/rnnoise.git
cd rnnoise
./autogen.sh
./configure
make

# 1) Configurer le flux RTSP
cp .env.sample .env
# puis éditer .env avec vos accès caméra

# 2) Capturer en continu (24/7) en segments WAV + réessais auto + rétention 7 jours
bash capture_rtsp_audio.sh

# 3) Segmenter avec VAD (sur les WAV à traiter)
python vad_segment.py
# - Les fichiers traités sont déplacés dans audios_processed/
# - Les segments exportés incluent leurs offsets (start/end) dans le nom de fichier

# 4) Transcrire avec Whisper vers JSON
python transcribe_segments.py
# - Chaque ligne de transcription est préfixée avec l'horodatage absolu de capture
# - Résultat: transcriptions/transcriptions.json

# 5) Lancer la GUI locale
streamlit run app.py

Ordre d'exécution recommandé: **VAD -> transcription JSON -> GUI**.

Fonctionnalités de la GUI (`app.py`):
- chargement de `transcriptions/transcriptions.json`,
- tableau trié par `timestamp_abs`,
- filtre texte + seuil de confiance,
- bouton **Écouter** par ligne (`st.audio`) pour lire le WAV `audio_path`,
- message utilisateur clair si le fichier WAV est manquant,
- code couleur du niveau de confiance (faible/moyen/élevé).


Note: le script détecte automatiquement si votre version de ffmpeg supporte `-rw_timeout` et désactive cette option sinon.
