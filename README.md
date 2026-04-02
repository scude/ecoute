pip install torch torchaudio silero-vad faster-whisper

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

# 4) Transcrire avec Whisper
python transcribe_segments.py
# - Chaque ligne de transcription est préfixée avec l'horodatage absolu de capture


Note: le script détecte automatiquement si votre version de ffmpeg supporte `-rw_timeout` et désactive cette option sinon.
