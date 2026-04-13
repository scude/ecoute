from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import wave

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from ecoute.pipeline_config import DEFAULT_CONFIG_PATH, load_config


SampleRate = int
TimeRange = Tuple[float, float]


def load_wav_mono_16k(
    file_path: Path,
    target_sample_rate: SampleRate = 16000,
) -> torch.Tensor:
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

    File names embed VAD offsets in milliseconds:
    speech_segment_001_start000123ms_end001456ms.wav
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    for index, segment in enumerate(speech_timestamps, start=1):
        start = segment["start"]
        end = segment["end"]

        chunk = audio[start:end]
        start_ms = int((start / sample_rate) * 1000)
        end_ms = int((end / sample_rate) * 1000)
        output_path = (
            output_dir
            / f"speech_segment_{index:03d}_start{start_ms:010d}ms_end{end_ms:010d}ms.wav"
        )

        save_chunk_wav(output_path, chunk, sample_rate)
        saved_files.append(output_path)

    return saved_files


def cleanup_processed_file(input_path: Path, processed_dir: Path) -> None:
    """
    Delete a source WAV file once VAD segments are written.

    Also removes a stale copy in audios_processed/ when present.
    """
    if input_path.exists():
        input_path.unlink()

    stale_archived_path = processed_dir / input_path.name
    if stale_archived_path.exists():
        stale_archived_path.unlink()


def process_file(
    input_path: Path,
    base_output_dir: Path,
    model: torch.nn.Module,
    sample_rate: SampleRate,
    vad_config: Dict[str, Any],
) -> int:
    """
    Process a single WAV file and save its speech segments into a dedicated subdirectory.
    """
    print(f"\n=== Processing: {input_path.name} ===")

    audio = load_wav_mono_16k(input_path, target_sample_rate=sample_rate)

    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sample_rate,
        threshold=float(vad_config["threshold"]),
        min_speech_duration_ms=int(vad_config["min_speech_duration_ms"]),
        min_silence_duration_ms=int(vad_config["min_silence_duration_ms"]),
        speech_pad_ms=int(vad_config["speech_pad_ms"]),
        return_seconds=False,
    )

    if not speech_timestamps:
        print("No speech detected.")
        return 0

    ranges = speech_timestamps_to_seconds(speech_timestamps, sample_rate)

    print("Detected speech segments:")
    for index, (start_sec, end_sec) in enumerate(ranges, start=1):
        duration = end_sec - start_sec
        print(
            f"  #{index:03d}  start={start_sec:.2f}s  end={end_sec:.2f}s  duration={duration:.2f}s"
        )

    output_dir = base_output_dir / input_path.stem
    saved_files = save_speech_segments(audio, speech_timestamps, output_dir, sample_rate)

    print(f"Saved {len(saved_files)} speech segment(s) into: {output_dir.resolve()}")
    for file_path in saved_files:
        print(f"  - {file_path.name}")

    return len(saved_files)


def process_pending_wavs(config: Dict[str, Any]) -> Dict[str, int]:
    paths_cfg = config["paths"]
    vad_cfg = config["vad"]
    input_dir = Path(paths_cfg["input_audio_dir"])
    output_dir = Path(paths_cfg["speech_segments_dir"])
    processed_input_dir = Path(paths_cfg["processed_audio_dir"])
    sample_rate: SampleRate = int(vad_cfg["sample_rate"])

    wav_files = sorted(input_dir.glob("*.wav"))

    if not wav_files:
        return {
            "found_files": 0,
            "candidate_files": 0,
            "processed_files": 0,
            "generated_segments": 0,
        }

    # The RTSP capture script always has one segment file currently being written.
    # Skip the most recently modified WAV file to avoid reading a partial file.
    file_being_written = max(wav_files, key=lambda path: path.stat().st_mtime)
    wav_files = [path for path in wav_files if path != file_being_written]

    if not wav_files:
        print(
            "Only one WAV file found and it appears to be the file currently being written. "
            "Nothing to process yet."
        )
        return {
            "found_files": 1,
            "candidate_files": 0,
            "processed_files": 0,
            "generated_segments": 0,
        }

    print("Loading model...")
    model = load_silero_vad()

    print(f"Found {len(wav_files)} WAV file(s) to process.")

    processed_files = 0
    generated_segments = 0

    for wav_file in wav_files:
        try:
            generated = process_file(
                input_path=wav_file,
                base_output_dir=output_dir,
                model=model,
                sample_rate=sample_rate,
                vad_config=vad_cfg,
            )
            cleanup_processed_file(wav_file, processed_input_dir)
            print(f"Deleted processed source file: {wav_file}")
            processed_files += 1
            generated_segments += generated
        except Exception as exc:
            print(f"Error while processing {wav_file.name}: {exc}")

    print("\nAll files processed.")
    return {
        "found_files": len(wav_files) + 1,
        "candidate_files": len(wav_files),
        "processed_files": processed_files,
        "generated_segments": generated_segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    stats = process_pending_wavs(config)
    print(f"VAD summary: {stats}")


if __name__ == "__main__":
    main()
