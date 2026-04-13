import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_segments import (
    parse_capture_start,
    parse_segment_start_ms,
    parse_segment_end_ms,
    compute_segment_id,
    normalize_confidence,
    transcribe_file,
)


def test_parse_capture_start():
    # Valid pattern
    path = Path("rtsp_audio_2023-03-15_12-30-00/segment.wav")
    expected = datetime(2023, 3, 15, 12, 30, 0)
    assert parse_capture_start(path) == expected

    # Invalid pattern
    path_invalid = Path("invalid_folder/segment.wav")
    assert parse_capture_start(path_invalid) is None


def test_parse_segment_offsets():
    path = Path("speech_segment_001_start0000001234ms_end0000005678ms.wav")
    assert parse_segment_start_ms(path) == 1234
    assert parse_segment_end_ms(path) == 5678

    path_no_match = Path("just_audio.wav")
    assert parse_segment_start_ms(path_no_match) == 0
    assert parse_segment_end_ms(path_no_match) == 0


def test_compute_segment_id(tmp_path):
    audio_file = tmp_path / "test.wav"
    audio_file.write_text("dummy")
    
    vad_start = 100
    vad_end = 500
    
    seg_id = compute_segment_id(audio_file, vad_start, vad_end)
    assert len(seg_id) == 40 # SHA1 length
    
    # Same file, same offsets -> same ID
    seg_id_2 = compute_segment_id(audio_file, vad_start, vad_end)
    assert seg_id == seg_id_2
    
    # Different offset -> different ID
    seg_id_3 = compute_segment_id(audio_file, vad_start, 600)
    assert seg_id != seg_id_3


def test_normalize_confidence():
    import math
    assert normalize_confidence(0.0) == 1.0 # exp(0) = 1
    assert normalize_confidence(-1.0) == pytest.approx(math.exp(-1.0))
    assert normalize_confidence(-100.0) == 0.0 # very small, capped at 0 (though exp is never < 0)
    assert normalize_confidence(1.0) == 1.0 # capped at 1.0


@patch("transcribe_segments.parse_capture_start")
def test_transcribe_file(mock_parse_capture, tmp_path):
    # Setup mock model and its return value
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = " Hello world "
    mock_segment.start = 1.0
    mock_segment.end = 2.5
    mock_segment.avg_logprob = -0.1
    mock_model.transcribe.return_value = ([mock_segment], None)

    # Setup file path
    audio_dir = tmp_path / "rtsp_audio_2023-03-15_12-00-00"
    audio_dir.mkdir()
    audio_file = audio_dir / "speech_segment_001_start0000001000ms_end0000005000ms.wav"
    audio_file.write_text("...")

    mock_parse_capture.return_value = datetime(2023, 3, 15, 12, 0, 0)
    
    whisper_config = {
        "language": "fr",
        "beam_size": 5,
        "best_of": 5,
        "patience": 1.0,
        "condition_on_previous_text": True,
        "temperature": 0.0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }

    entries = transcribe_file(mock_model, audio_file, "some-id", whisper_config)
    
    assert len(entries) == 1
    entry = entries[0]
    assert entry["text"] == "Hello world"
    assert entry["segment_id"] == "some-id"
    # start_ms is 1000 (1s), segment.start is 1.0s -> abs start is 12:00:00 + 2s = 12:00:02
    assert "2023-03-15T12:00:02" in entry["timestamp_abs"]
    assert entry["segment_start_sec"] == 2.0 # 1.0 (offset) + 1.0 (internal)
    assert entry["segment_end_sec"] == 3.5 # 1.0 (offset) + 2.5 (internal)
