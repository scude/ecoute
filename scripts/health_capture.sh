#!/usr/bin/env bash
set -euo pipefail

HEARTBEAT_FILE="${HEARTBEAT_FILE:-/app/runtime/capture_heartbeat.json}"
MAX_AGE_SECONDS="${CAPTURE_HEARTBEAT_MAX_AGE_SECONDS:-180}"

if [[ ! -s "$HEARTBEAT_FILE" ]]; then
  echo "heartbeat missing: $HEARTBEAT_FILE"
  exit 1
fi

updated_epoch="$(sed -n 's/.*"updated_at_epoch"[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' "$HEARTBEAT_FILE" | head -n1)"
state="$(sed -n 's/.*"state"[[:space:]]*:[[:space:]]*"\([^"]\+\)".*/\1/p' "$HEARTBEAT_FILE" | head -n1)"

if [[ -z "$updated_epoch" ]]; then
  echo "invalid heartbeat: missing updated_at_epoch"
  exit 1
fi

now_epoch="$(date -u +%s)"
age="$((now_epoch - updated_epoch))"

if (( age > MAX_AGE_SECONDS )); then
  echo "heartbeat stale: ${age}s > ${MAX_AGE_SECONDS}s"
  exit 1
fi

case "$state" in
  capturing|connecting|reconnecting|restarting|starting)
    ;;
  *)
    echo "invalid capture state: ${state:-unknown}"
    exit 1
    ;;
esac

echo "capture healthy (state=$state age=${age}s)"
