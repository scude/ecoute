# Écoute: RTSP Audio Pipeline -> VAD -> Transcription (Whisper) -> UI

This project captures an audio stream (RTSP), automatically segments speech with VAD, transcribes the segments with Whisper, then exposes results in a Streamlit interface.

---

## 1) Overview

The pipeline runs in this order:

1. **RTSP capture** (`capture_rtsp_audio.sh`)  
   Generates WAV files in `audios/`.
2. **VAD** (`vad_segment.py`)  
   Detects spoken portions, exports segments to `speech_segments/`, then deletes processed source WAV files (and removes a possible duplicate copy in `audios_processed/`).
3. **Incremental transcription** (`transcribe_segments.py`)  
   Reads unprocessed segments, writes results to SQLite, optional JSON export.
4. **Streamlit UI** (`app.py`)  
   Search and playback for transcriptions.

The **`pipeline_runner.py`** script orchestrates steps 2 and 3 in one-shot or continuous loop mode.

---

## 2) Recommended run mode: Docker Compose

The project is now intended to run via Docker Compose (capture + pipeline + UI),
which replaces manual dependency installation in most cases.

### 2.1 Prerequisites

- Docker Engine
- Docker Compose (the `docker compose` plugin)

### 2.2 Configuration

```bash
cp .env.sample .env
# edit at least RTSP_URL in .env
```

Useful variables in `.env` (non-exhaustive):

- `RTSP_URL`
- `PIPELINE_INTERVAL_SECONDS`
- `PIPELINE_MAX_BACKOFF_SECONDS`
- `WHISPER_MODEL_SIZE_OR_PATH`
- `WHISPER_DEVICE`
- `WHISPER_COMPUTE_TYPE`

### 2.3 Start

```bash
docker compose up -d --build
```

Started services:

- `capture`: runs `capture_rtsp_audio.sh` in a loop.
- `pipeline`: runs `python pipeline_runner.py --loop`.
- `ui`: runs `streamlit run app.py --server.address 0.0.0.0 --server.port 8501`.

UI available at: <http://localhost:8501>

### 2.4 Stop and logs

```bash
docker compose logs -f
docker compose down
```

### 2.5 Persistent volumes

The following directories are mounted as persistent host volumes:

- `./audios`
- `./audios_processed`
- `./speech_segments`
- `./transcriptions`
- `./runtime`

### 2.6 `capture` service health

The `capture` service writes a JSON heartbeat to `runtime/capture_heartbeat.json` (inside the container: `/app/runtime/capture_heartbeat.json`) at a regular interval during capture and during state transitions.

The Docker Compose `healthcheck` runs `./scripts/health_capture.sh` and applies the following criteria:

- **Healthy** if the heartbeat file exists, contains `updated_at_epoch`, its age is less than or equal to `CAPTURE_HEARTBEAT_MAX_AGE_SECONDS` (180s by default), and `state` is in `{starting, connecting, capturing, reconnecting, restarting}`.
- **Unhealthy** if the heartbeat is missing/invalid, too old (older than the threshold), or if the state is not allowed.

The heartbeat includes:

- `updated_at` / `updated_at_epoch`
- `last_segment_at` / `last_segment_epoch` (latest WAV segment produced)
- `reconnect_errors` (reconnection error counter)
- `state` (current capture state)

---

## 3) Manual mode (legacy / debugging)

This mode is still useful for local debugging outside Docker, but it is no longer the primary mode.

### Python dependencies

```bash
pip install torch torchaudio silero-vad faster-whisper streamlit pandas pyyaml
```

### Manual startup

```bash
bash capture_rtsp_audio.sh
python pipeline_runner.py --once
streamlit run app.py
```

---

## 4) `pipeline_runner.py` orchestrator

Two modes are available:

- `--once`: runs exactly one `VAD -> transcription` sequence, then exits.
- `--loop`: runs the sequence continuously with a configurable interval.

Examples:

```bash
python pipeline_runner.py --once
python pipeline_runner.py --loop
```

### Built-in robustness

- **Lockfile**: `/tmp/ecoute_pipeline.lock` (prevents concurrent runs).
- **Structured JSON logs**: level, message, per-step duration, file counters.
- **Exponential backoff** on errors (up to `pipeline.max_backoff_seconds`).

---

## 5) Configuration

Configuration is centralized in **`config.yaml`**.  
Environment variables can override file values.

### 5.1 `config.yaml` file

Main sections:

- `paths`: input/output paths
- `pipeline`: interval and backoff
- `whisper`: model and inference parameters (including `banned_phrases` for filtering hallucinations)
- `vad`: speech detection parameters

### 5.2 Supported environment variables

> Priority: **environment variables > config.yaml > internal defaults**

#### Pipeline

- `PIPELINE_INTERVAL_SECONDS`  
  Interval between runs in `--loop` mode.
- `PIPELINE_MAX_BACKOFF_SECONDS`  
  Maximum backoff duration after an error.

#### Paths

- `INPUT_AUDIO_DIR`  
  Capture WAV directory (e.g., `audios`).
- `SPEECH_SEGMENTS_DIR`  
  VAD segments directory (e.g., `speech_segments`).
- `PROCESSED_AUDIO_DIR`  
  Optional cleanup directory for old processed source WAV files (e.g., `audios_processed`).
- `TRANSCRIPTIONS_DB_PATH`  
  SQLite path for transcriptions.
- `TRANSCRIPTIONS_JSON_OUTPUT`  
  JSON export file path.

#### Whisper

- `WHISPER_MODEL_SIZE_OR_PATH`  
  Model name (e.g., `small`, `medium`, `large-v3`) or local model path.
- `WHISPER_DEVICE`  
  Inference device (`cpu`, `cuda`, etc.).
- `WHISPER_COMPUTE_TYPE`  
  Compute/quantization type used by faster-whisper (e.g., `int8`, `float16`, `float32`).
- `WHISPER_LANGUAGE`  
  Forced transcription language (e.g., `fr`, `en`).
- `WHISPER_BEAM_SIZE`  
  Beam search size (larger = potentially more accurate but slower).
- `WHISPER_BEST_OF`  
  Number of candidate decodes evaluated to select the best result.
- `WHISPER_PATIENCE`  
  Beam search decode patience parameter.
- `WHISPER_CONDITION_ON_PREVIOUS_TEXT` (`true/false`, `1/0`, `yes/no`)  
  If enabled, previous text influences the current segment (higher coherence, higher drift risk).
- `WHISPER_TEMPERATURE`  
  Decoding temperature (higher = more diversity, less deterministic).
- `WHISPER_COMPRESSION_RATIO_THRESHOLD`  
  Filter threshold for outputs considered abnormal/over-compressed.
- `WHISPER_LOG_PROB_THRESHOLD`  
  Minimum log-probability threshold to accept an output.
- `WHISPER_NO_SPEECH_THRESHOLD`  
  Whisper-level no-speech detection threshold.
- `WHISPER_BANNED_PHRASES`  
  Comma-separated list of phrases to hide from the UI (e.g., "Merci.,Sous-titrage ST' 501"). Only excludes segments matching the exact phrase.

#### VAD

- `VAD_SAMPLE_RATE`  
  Expected input sampling rate (Hz), typically `16000`.
- `VAD_THRESHOLD`  
  Speech detection threshold (higher = stricter).
- `VAD_MIN_SPEECH_DURATION_MS`  
  Minimum duration (ms) to validate a speech segment.
- `VAD_MIN_SILENCE_DURATION_MS`  
  Minimum silence duration (ms) to split two segments.
- `VAD_SPEECH_PAD_MS`  
  Padding margin (ms) added before/after each detected segment.

### 5.3 Override example

```bash
export PIPELINE_INTERVAL_SECONDS=30
export WHISPER_MODEL_SIZE_OR_PATH=small
export WHISPER_DEVICE=cpu
python pipeline_runner.py --loop
```

---

## 6) Using standalone scripts

### VAD only

```bash
python vad_segment.py
```

### Transcription only

```bash
python transcribe_segments.py --export-json
```

Useful options:

```bash
python transcribe_segments.py \
  --input-dir speech_segments \
  --db-path transcriptions/transcriptions.sqlite \
  --json-output transcriptions/transcriptions.json \
  --export-json
```

---

## 7) Service mode (outside Docker, optional)

### Option A - systemd

Create `/etc/systemd/system/ecoute-pipeline.service`:

```ini
[Unit]
Description=Ecoute pipeline runner (VAD + transcription)
After=network.target

[Service]
Type=simple
WorkingDirectory=/workspace/ecoute
Environment=PIPELINE_INTERVAL_SECONDS=60
ExecStart=/usr/bin/python3 /workspace/ecoute/pipeline_runner.py --loop
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ecoute-pipeline.service
sudo systemctl status ecoute-pipeline.service
```

### Option B - cron

Periodic one-shot execution:

```cron
*/2 * * * * cd /workspace/ecoute && /usr/bin/python3 pipeline_runner.py --once >> /var/log/ecoute_pipeline.log 2>&1
```

The lockfile prevents overlap if a previous run is still active.

---

## 8) Streamlit interface

`app.py` provides:

- sorted transcription display,
- text filtering,
- confidence threshold,
- audio snippet playback (`st.audio`),
- explicit message when the WAV file no longer exists,
- **filtering of banned phrases** (hallucinations) as defined in the configuration.

---

## 9) Operational notes

- The capture script automatically detects whether `ffmpeg` supports `-rw_timeout`.
- For a standard deployment, prefer Docker Compose.
- `systemd`/`cron` remain valid for outside-Docker mode.
- Regularly monitor disk usage (`audios/`, `speech_segments/`, SQLite database).
