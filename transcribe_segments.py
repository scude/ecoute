from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Iterable, List, Optional

from faster_whisper import WhisperModel


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


def transcribe_file(model: WhisperModel, audio_file: Path, language: str = "fr") -> List[str]:
    """
    Transcribe a single audio file using faster-whisper and return timestamped lines.
    """
    segments, _ = model.transcribe(
        str(audio_file),
        language=language,
        vad_filter=False,
        beam_size=5,
        condition_on_previous_text=False,
        temperature=0.0,
    )

    capture_start = parse_capture_start(audio_file)
    segment_start_ms = parse_segment_start_ms(audio_file)

    lines: List[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        if capture_start is not None:
            absolute_dt = capture_start + timedelta(
                milliseconds=segment_start_ms + int(segment.start * 1000)
            )
            timestamp = absolute_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # fallback: timestamp relative to current chunk
            timestamp = f"+{segment.start:.2f}s"

        lines.append(f"[{timestamp}] {text}")

    return lines


def main() -> None:
    """
    Transcribe every WAV file found in speech_segments/.
    """
    input_dir = Path("speech_segments")

    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    print("Loading faster-whisper model...")
    model = WhisperModel(
        model_size_or_path="large-v3",
        device="cpu",
        compute_type="int8",
    )

    audio_files = list(iter_audio_files(input_dir))
    if not audio_files:
        print("No WAV files found.")
        return

    full_transcription: List[str] = []

    for audio_file in audio_files:
        print(f"\n=== {audio_file.name} ===")
        lines = transcribe_file(model, audio_file, language="fr")

        if lines:
            for line in lines:
                print(line)
            full_transcription.extend(lines)
        else:
            print("[empty transcription]")

    print("\n=== MERGED TRANSCRIPTION ===")
    if full_transcription:
        for line in full_transcription:
            print(line)
    else:
        print("[empty transcription]")


if __name__ == "__main__":
    main()
