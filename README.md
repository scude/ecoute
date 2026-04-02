pip install torch torchaudio silero-vad faster-whisper

pip install faster-whisper

python vad_segment.py

python transcribe_segments.py

conda activate ecoute

sudo apt update
sudo apt install -y git autoconf automake libtool build-essential pkg-config
cd ~
git clone https://github.com/xiph/rnnoise.git
cd rnnoise
./autogen.sh
./configure
make

pip install librosa