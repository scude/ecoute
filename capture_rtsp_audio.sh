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
HEARTBEAT_FILE="${HEARTBEAT_FILE:-runtime/capture_heartbeat.json}"
HEARTBEAT_INTERVAL_SECONDS="${HEARTBEAT_INTERVAL_SECONDS:-5}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$HEARTBEAT_FILE")"

RECONNECT_ERRORS=0
CURRENT_STATE="starting"
LAST_SEGMENT_EPOCH=""

cleanup_old_audio() {
  find "$OUTPUT_DIR" -type f -name '*.wav' -mtime "+${RETENTION_DAYS}" -print -delete || true
}

ffmpeg_supports_rw_timeout() {
  ffmpeg -hide_banner -h full 2>/dev/null | grep -q -- '-rw_timeout'
}

iso8601_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

format_epoch_iso8601() {
  local epoch="$1"
  if [[ -z "$epoch" ]]; then
    printf 'null'
    return
  fi
  date -u -d "@${epoch}" +"%Y-%m-%dT%H:%M:%SZ"
}

update_last_segment_timestamp() {
  local newest
  newest="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.wav' -printf '%T@\n' 2>/dev/null | sort -nr | head -n1 || true)"

  if [[ -n "$newest" ]]; then
    LAST_SEGMENT_EPOCH="${newest%.*}"
  fi
}

write_heartbeat() {
  local updated_at
  local last_segment_at
  local last_segment_at_json

  updated_at="$(iso8601_now)"
  last_segment_at="$(format_epoch_iso8601 "$LAST_SEGMENT_EPOCH")"

  if [[ "$last_segment_at" == "null" ]]; then
    last_segment_at_json="null"
  else
    last_segment_at_json="\"${last_segment_at}\""
  fi

  cat > "$HEARTBEAT_FILE" <<JSON
{
  "updated_at": "${updated_at}",
  "updated_at_epoch": $(date -u +%s),
  "state": "${CURRENT_STATE}",
  "last_segment_epoch": ${LAST_SEGMENT_EPOCH:-null},
  "last_segment_at": ${last_segment_at_json},
  "reconnect_errors": ${RECONNECT_ERRORS},
  "output_dir": "${OUTPUT_DIR}",
  "segment_seconds": ${SEGMENT_SECONDS}
}
JSON
}

monitor_capture_heartbeat() {
  local ffmpeg_pid="$1"

  while kill -0 "$ffmpeg_pid" 2>/dev/null; do
    update_last_segment_timestamp
    write_heartbeat
    sleep "$HEARTBEAT_INTERVAL_SECONDS"
  done

  update_last_segment_timestamp
  write_heartbeat
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
    "$OUTPUT_DIR/$OUTPUT_PATTERN" &

  local ffmpeg_pid=$!
  monitor_capture_heartbeat "$ffmpeg_pid" &
  local monitor_pid=$!

  local ffmpeg_exit=0
  wait "$ffmpeg_pid" || ffmpeg_exit=$?

  wait "$monitor_pid" 2>/dev/null || true
  return "$ffmpeg_exit"
}

trap 'CURRENT_STATE="stopped"; write_heartbeat; echo "Stopping capture..."; exit 0' INT TERM

echo "Starting RTSP capture loop..."
echo "Output directory: $OUTPUT_DIR"
echo "Segment length: ${SEGMENT_SECONDS}s"
echo "Retention: ${RETENTION_DAYS} days"

write_heartbeat

while true; do
  CURRENT_STATE="connecting"
  write_heartbeat
  cleanup_old_audio

  CURRENT_STATE="capturing"
  write_heartbeat

  if run_capture_once; then
    CURRENT_STATE="restarting"
    write_heartbeat
    echo "ffmpeg exited cleanly; restarting in ${RECONNECT_DELAY_SECONDS}s..."
  else
    RECONNECT_ERRORS=$((RECONNECT_ERRORS + 1))
    CURRENT_STATE="reconnecting"
    write_heartbeat
    echo "ffmpeg crashed or stream disconnected; reconnecting in ${RECONNECT_DELAY_SECONDS}s..." >&2
  fi

  sleep "$RECONNECT_DELAY_SECONDS"
done
