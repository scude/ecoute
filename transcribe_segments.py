from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timedelta
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

from faster_whisper import WhisperModel

from storage import DEFAULT_DB_PATH, SQLiteStorage


CAPTURE_STEM_RE = re.compile(r".*?(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
SEGMENT_OFFSET_RE = re.compile(r"_start(\d+)ms_end(\d+)ms\\.wav$")


def iter_audio_files(directory: Path) -> Iterable[Path]:
    """
    Yield WAV files recursively from a directory in sorted order.
    """
    yield from sorted(directory.rglob("*.wav"))


def parse_capture_start(audio_file: Path) -> Optional[datetime]:
    """
    Parse capture start datetime from parent directory name (source file stem).
    Example stem: rtsp_audio_2026-04-02_12-30-00
    """
    stem = audio_file.parent.name
    match = CAPTURE_STEM_RE.match(stem)
    if not match:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def parse_segment_start_ms(audio_file: Path) -> int:
    """
    Parse VAD segment start offset from file name.
    """
    match = SEGMENT_OFFSET_RE.search(audio_file.name)
    if not match:
        return 0
    return int(match.group(1))


def parse_segment_end_ms(audio_file: Path) -> int:
    """
    Parse VAD segment end offset from file name.
    """
    match = SEGMENT_OFFSET_RE.search(audio_file.name)
    if not match:
        return 0
    return int(match.group(2))


def compute_segment_id(audio_file: Path, vad_start_ms: int, vad_end_ms: int) -> str:
    stat = audio_file.stat()
    payload = f"{audio_file.resolve()}|{vad_start_ms}|{vad_end_ms}|{stat.st_mtime}|{stat.st_size}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def normalize_confidence(avg_logprob: float) -> float:
    """
    Convert average log-probability to a readable confidence score in [0.0, 1.0].
    """
    return max(0.0, min(1.0, math.exp(avg_logprob)))


def transcribe_file(
    model: WhisperModel,
    audio_file: Path,
    segment_id: str,
    language: str = "fr",
) -> List[Dict[str, Any]]:
    """
    Transcribe a single audio file using faster-whisper and return structured entries.
    """
    segments, _ = model.transcribe(
        str(audio_file),
        language=language,
        vad_filter=False,
        beam_size=1,
        best_of=1,
        patience=1.0,
        condition_on_previous_text=False,
        temperature=0.0,
        compression_ratio_threshold=1.8,
        log_prob_threshold=-0.9,
        no_speech_threshold=0.3,
    )

    capture_start = parse_capture_start(audio_file)
    segment_start_ms = parse_segment_start_ms(audio_file)
    segment_end_ms = parse_segment_end_ms(audio_file)
    audio_path = str(audio_file.resolve())

    entries: List[Dict[str, Any]] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        segment_start_sec = (segment_start_ms / 1000.0) + float(segment.start)
        segment_end_sec = (segment_start_ms / 1000.0) + float(segment.end)

        if capture_start is not None:
            absolute_dt = capture_start + timedelta(
                milliseconds=segment_start_ms + int(segment.start * 1000)
            )
            timestamp_abs = absolute_dt.isoformat(timespec="milliseconds")
        else:
            timestamp_abs = None

        entries.append(
            {
                "segment_id": segment_id,
                "timestamp_abs": timestamp_abs,
                "text": text,
                "confidence": round(normalize_confidence(segment.avg_logprob), 4),
                "audio_path": audio_path,
                "vad_start_ms": segment_start_ms,
                "vad_end_ms": segment_end_ms,
                "segment_start_sec": round(segment_start_sec, 3),
                "segment_end_sec": round(segment_end_sec, 3),
            }
        )

    return entries


def main() -> None:
    """
    Transcribe every WAV file found in speech_segments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("speech_segments"))
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--export-json", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("transcriptions/transcriptions.json"),
        help="JSON compatibility export path",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    storage = SQLiteStorage(args.db_path)

    print("Loading faster-whisper model...")
    model = WhisperModel(
        model_size_or_path="medium",
        device="cpu",
        compute_type="int8",
    )

    audio_files = list(iter_audio_files(input_dir))
    if not audio_files:
        print("No WAV files found.")
        return

    new_rows_count = 0
    for audio_file in audio_files:
        segment_start_ms = parse_segment_start_ms(audio_file)
        segment_end_ms = parse_segment_end_ms(audio_file)
        segment_id = compute_segment_id(audio_file, segment_start_ms, segment_end_ms)

        if storage.is_segment_done(segment_id):
            print(f"[skip] already processed: {audio_file.name}")
            continue

        print(f"\n=== {audio_file.name} ===")
        entries = transcribe_file(model, audio_file, segment_id=segment_id, language="fr")
        storage.append_transcription_rows(entries)

        stat = audio_file.stat()
        storage.mark_segment_done(
            segment_id,
            {
                "audio_path": str(audio_file.resolve()),
                "vad_start_ms": segment_start_ms,
                "vad_end_ms": segment_end_ms,
                "audio_mtime": stat.st_mtime,
                "audio_size": stat.st_size,
            },
        )

        if entries:
            for entry in entries:
                display_ts = entry["timestamp_abs"] or f"+{entry['segment_start_sec']:.2f}s"
                print(f"[{display_ts}] ({entry['confidence']:.2f}) {entry['text']}")
            new_rows_count += len(entries)
        else:
            print("[empty transcription]")

    if args.export_json:
        storage.export_json(args.json_output)
        print(f"\nJSON export generated: {args.json_output}")

    print(f"\nDone. New rows inserted: {new_rows_count}")


if __name__ == "__main__":
    main()
