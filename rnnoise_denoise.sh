#!/usr/bin/env bash
set -e

INPUT="speech_segments/speech_segment_001.wav"
OUTPUT="speech_segments/speech_segment_denoised_001.wav"

TMP_IN="/tmp/rnnoise_in.raw"
TMP_OUT="/tmp/rnnoise_out.raw"

echo "→ Convert to RAW 48k mono"
ffmpeg -y -i "$INPUT" -ac 1 -ar 48000 -f s16le "$TMP_IN"

echo "→ RNNoise processing"
"$HOME/rnnoise/examples/rnnoise_demo" "$TMP_IN" "$TMP_OUT"

echo "→ Back to WAV"
ffmpeg -y -f s16le -ar 48000 -ac 1 -i "$TMP_OUT" "$OUTPUT"

echo "✅ Done → $OUTPUT"
