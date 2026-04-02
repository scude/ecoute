from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import wave

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad


SampleRate = int
TimeRange = Tuple[float, float]


def load_wav_mono_16k(file_path: Path, target_sample_rate: SampleRate = 16000) -> torch.Tensor:
    """
    Load a PCM WAV file, convert it to mono float32, and ensure it is already 16 kHz.
    """
    with wave.open(str(file_path), "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(num_frames)

    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width * 8} bits")

    if sample_rate != target_sample_rate:
        raise ValueError(
            f"Unexpected sample rate: {sample_rate} Hz. Expected {target_sample_rate} Hz."
        )

    audio = np.frombuffer(raw_data, dtype=np.int16).copy()

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    audio_tensor = torch.from_numpy(audio).float() / 32768.0
    return audio_tensor


def speech_timestamps_to_seconds(
    speech_timestamps: List[dict],
    sample_rate: SampleRate,
) -> List[TimeRange]:
    """
    Convert Silero speech timestamps from sample indices to seconds.
    """
    ranges: List[TimeRange] = []

    for segment in speech_timestamps:
        start_sec = segment["start"] / sample_rate
        end_sec = segment["end"] / sample_rate
        ranges.append((start_sec, end_sec))

    return ranges


def save_chunk_wav(output_path: Path, chunk: torch.Tensor, sample_rate: int) -> None:
    """
    Save a mono float32 torch tensor to a PCM 16-bit WAV file.
    """
    chunk_np = chunk.detach().cpu().clamp(-1.0, 1.0).numpy()
    chunk_int16 = (chunk_np * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(chunk_int16.tobytes())


def save_speech_segments(
    audio: torch.Tensor,
    speech_timestamps: List[dict],
    output_dir: Path,
    sample_rate: SampleRate,
) -> List[Path]:
    """
    Save detected speech segments as individual WAV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    for index, segment in enumerate(speech_timestamps, start=1):
        start = segment["start"]
        end = segment["end"]

        chunk = audio[start:end]
        output_path = output_dir / f"speech_segment_{index:03d}.wav"

        save_chunk_wav(output_path, chunk, sample_rate)
        saved_files.append(output_path)

    return saved_files


def main() -> None:
    input_path = Path("audios/output.wav")
    output_dir = Path("speech_segments")
    sample_rate: SampleRate = 16000

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading model...")
    model = load_silero_vad()

    print(f"Loading audio: {input_path}")
    audio = load_wav_mono_16k(input_path, target_sample_rate=sample_rate)

    print("Running voice activity detection...")
    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sample_rate,
        threshold=0.5,
        min_speech_duration_ms=300,
        min_silence_duration_ms=700,
        speech_pad_ms=300,
        return_seconds=False,
    )

    if not speech_timestamps:
        print("No speech detected.")
        return

    ranges = speech_timestamps_to_seconds(speech_timestamps, sample_rate)

    print("\nDetected speech segments:")
    for index, (start_sec, end_sec) in enumerate(ranges, start=1):
        duration = end_sec - start_sec
        print(f"  #{index:03d}  start={start_sec:.2f}s  end={end_sec:.2f}s  duration={duration:.2f}s")

    saved_files = save_speech_segments(audio, speech_timestamps, output_dir, sample_rate)

    print(f"\nSaved {len(saved_files)} speech segment(s) into: {output_dir.resolve()}")
    for file_path in saved_files:
        print(f"  - {file_path.name}")


if __name__ == "__main__":
    main()