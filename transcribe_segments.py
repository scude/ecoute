from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from faster_whisper import WhisperModel


def iter_audio_files(directory: Path) -> Iterable[Path]:
    """
    Yield WAV files recursively from a directory in sorted order.
    """
    yield from sorted(directory.rglob("*.wav"))


def transcribe_file(model: WhisperModel, audio_file: Path, language: str = "fr") -> str:
    """
    Transcribe a single audio file using faster-whisper.

    Args:
        model: Initialized WhisperModel instance.
        audio_file: Path to the WAV file.
        language: Language hint for transcription.

    Returns:
        Final concatenated transcription text.
    """
    segments, info = model.transcribe(
        str(audio_file),
        language=language,
        vad_filter=False,
        beam_size=5,
        condition_on_previous_text=False,
        temperature=0.0,
    )

    texts: List[str] = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            texts.append(text)

    return " ".join(texts).strip()


def main() -> None:
    """
    Transcribe every WAV file found in speech_segments/.
    """
    input_dir = Path("speech_segments")

    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

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

    full_transcription: List[str] = []

    for audio_file in audio_files:
        print(f"\n=== {audio_file.name} ===")
        text = transcribe_file(model, audio_file, language="fr")

        if text:
            print(text)
            full_transcription.append(text)
        else:
            print("[empty transcription]")

    merged_text = " ".join(full_transcription).strip()

    print("\n=== MERGED TRANSCRIPTION ===")
    if merged_text:
        print(merged_text)
    else:
        print("[empty transcription]")


if __name__ == "__main__":
    main()