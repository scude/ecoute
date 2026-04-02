#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${RTSP_URL:?RTSP_URL is not set. Define it in .env or environment.}"

OUTPUT_DIR="${OUTPUT_DIR:-audios}"
OUTPUT_PATTERN="${OUTPUT_PATTERN:-rtsp_audio_%Y-%m-%d_%H-%M-%S.wav}"
AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-16000}"
AUDIO_CHANNELS="${AUDIO_CHANNELS:-1}"
RTSP_TRANSPORT="${RTSP_TRANSPORT:-tcp}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-60}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
RECONNECT_DELAY_SECONDS="${RECONNECT_DELAY_SECONDS:-5}"
FFMPEG_RW_TIMEOUT_US="${FFMPEG_RW_TIMEOUT_US:-15000000}"

mkdir -p "$OUTPUT_DIR"

cleanup_old_audio() {
  find "$OUTPUT_DIR" -type f -name '*.wav' -mtime "+${RETENTION_DAYS}" -print -delete || true
}

ffmpeg_supports_rw_timeout() {
  ffmpeg -hide_banner -h full 2>/dev/null | grep -q -- '-rw_timeout'
}

run_capture_once() {
  local -a input_options
  input_options=(
    -rtsp_transport "$RTSP_TRANSPORT"
  )

  if ffmpeg_supports_rw_timeout; then
    input_options+=( -rw_timeout "$FFMPEG_RW_TIMEOUT_US" )
  else
    echo "Warning: ffmpeg does not support -rw_timeout; running without it." >&2
  fi

  ffmpeg \
    -nostdin \
    -hide_banner \
    -loglevel warning \
    "${input_options[@]}" \
    -i "$RTSP_URL" \
    -vn \
    -acodec pcm_s16le \
    -ar "$AUDIO_SAMPLE_RATE" \
    -ac "$AUDIO_CHANNELS" \
    -f segment \
    -segment_time "$SEGMENT_SECONDS" \
    -reset_timestamps 1 \
    -strftime 1 \
    "$OUTPUT_DIR/$OUTPUT_PATTERN"
}

trap 'echo "Stopping capture..."; exit 0' INT TERM

echo "Starting RTSP capture loop..."
echo "Output directory: $OUTPUT_DIR"
echo "Segment length: ${SEGMENT_SECONDS}s"
echo "Retention: ${RETENTION_DAYS} days"

while true; do
  cleanup_old_audio

  if run_capture_once; then
    echo "ffmpeg exited cleanly; restarting in ${RECONNECT_DELAY_SECONDS}s..."
  else
    echo "ffmpeg crashed or stream disconnected; reconnecting in ${RECONNECT_DELAY_SECONDS}s..." >&2
  fi

  sleep "$RECONNECT_DELAY_SECONDS"
done
