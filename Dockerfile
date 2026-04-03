FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        bash \
        build-essential \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install \
        numpy \
        pandas \
        pyyaml \
        streamlit \
        torch \
        torchaudio \
        silero-vad \
        faster-whisper

COPY . /app

RUN chmod +x /app/capture_rtsp_audio.sh /app/rnnoise_denoise.sh

CMD ["python", "pipeline_runner.py", "--loop"]
