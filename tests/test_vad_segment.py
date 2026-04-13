import wave
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vad_segment import (
    load_wav_mono_16k,
    speech_timestamps_to_seconds,
    save_chunk_wav,
    save_speech_segments,
    cleanup_processed_file,
    process_file,
)


def create_dummy_wav(path: Path, sample_rate=16000, num_channels=1, duration_sec=1.0):
    """Helper to create a simple valid PCM 16-bit WAV file."""
    num_frames = int(sample_rate * duration_sec)
    # Sine wave
    t = np.linspace(0, duration_sec, num_frames, endpoint=False)
    data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    if num_channels > 1:
        data = np.repeat(data[:, np.newaxis], num_channels, axis=1).flatten()

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data.tobytes())


def test_load_wav_mono_16k(tmp_path):
    wav_path = tmp_path / "test.wav"
    create_dummy_wav(wav_path, sample_rate=16000, num_channels=1)
    
    tensor = load_wav_mono_16k(wav_path)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 1
    assert len(tensor) == 16000
    assert torch.all(tensor <= 1.0) and torch.all(tensor >= -1.0)

    # Test multi-channel
    create_dummy_wav(wav_path, sample_rate=16000, num_channels=2)
    tensor_stereo = load_wav_mono_16k(wav_path)
    assert tensor_stereo.dim() == 1 # Should be flattened to mono

    # Test invalid sample rate
    create_dummy_wav(wav_path, sample_rate=44100)
    with pytest.raises(ValueError, match="Unexpected sample rate"):
        load_wav_mono_16k(wav_path)


def test_speech_timestamps_to_seconds():
    timestamps = [
        {"start": 0, "end": 16000},
        {"start": 32000, "end": 48000},
    ]
    ranges = speech_timestamps_to_seconds(timestamps, 16000)
    assert ranges == [(0.0, 1.0), (2.0, 3.0)]


def test_save_chunk_wav(tmp_path):
    output_path = tmp_path / "chunk.wav"
    chunk = torch.sin(torch.linspace(0, 2 * 3.14159 * 440, 16000))
    save_chunk_wav(output_path, chunk, 16000)
    
    assert output_path.exists()
    with wave.open(str(output_path), "rb") as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        assert f.getframerate() == 16000
        assert f.getnframes() == 16000


def test_save_speech_segments(tmp_path):
    audio = torch.zeros(48000)
    speech_timestamps = [
        {"start": 0, "end": 16000},
        {"start": 32000, "end": 48000},
    ]
    output_dir = tmp_path / "segments"
    
    saved_files = save_speech_segments(audio, speech_timestamps, output_dir, 16000)
    
    assert len(saved_files) == 2
    assert (output_dir / "speech_segment_001_start0000000000ms_end0000001000ms.wav").exists()
    assert (output_dir / "speech_segment_002_start0000002000ms_end0000003000ms.wav").exists()


def test_cleanup_processed_file(tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.write_text("dummy")
    
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    stale_path = processed_dir / "input.wav"
    stale_path.write_text("stale")
    
    cleanup_processed_file(input_path, processed_dir)
    
    assert not input_path.exists()
    assert not stale_path.exists()


@patch("vad_segment.load_wav_mono_16k")
@patch("vad_segment.get_speech_timestamps")
@patch("vad_segment.save_speech_segments")
def test_process_file(mock_save, mock_get_ts, mock_load, tmp_path):
    input_path = Path("test.wav")
    mock_load.return_value = torch.zeros(16000)
    mock_get_ts.return_value = [{"start": 0, "end": 16000}]
    mock_save.return_value = [Path("seg1.wav")]
    
    vad_config = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
        "speech_pad_ms": 30,
    }
    
    count = process_file(input_path, tmp_path, MagicMock(), 16000, vad_config)
    
    assert count == 1
    mock_load.assert_called_once()
    mock_get_ts.assert_called_once()
    mock_save.assert_called_once()
