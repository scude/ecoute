import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline_runner import (
    PipelineLock,
    compute_backlog_stats,
    write_pipeline_status,
    run_pipeline_once,
)


@pytest.fixture
def temp_lock_path(tmp_path):
    return tmp_path / "test.lock"


def test_pipeline_lock_acquire_release(temp_lock_path):
    lock = PipelineLock(temp_lock_path)
    
    # Mock fcntl.flock to avoid real OS locking issues in tests
    with patch("fcntl.flock") as mock_flock:
        assert lock.acquire() is True
        assert temp_lock_path.exists()
        assert temp_lock_path.read_text() == str(os.getpid())
        
        lock.release()
        mock_flock.assert_any_call(pytest.any, 8) # LOCK_UN is usually 8


def test_compute_backlog_stats(tmp_path):
    audio_dir = tmp_path / "audios"
    audio_dir.mkdir()
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    
    (audio_dir / "1.wav").write_text("...")
    (segments_dir / "s1.wav").write_text("...")
    (segments_dir / "s2.wav").write_text("...")
    
    config = {
        "paths": {
            "input_audio_dir": str(audio_dir),
            "speech_segments_dir": str(segments_dir),
        }
    }
    
    stats = compute_backlog_stats(config)
    assert stats["pending_wavs"] == 1
    assert stats["pending_segments"] == 2


def test_write_pipeline_status(tmp_path):
    status_path = tmp_path / "status.json"
    with patch("pipeline_runner.STATUS_PATH", status_path):
        write_pipeline_status(
            last_run_started_at="start",
            last_run_finished_at="end",
            last_run_success=True,
            last_error=None,
            vad_stats={"files": 1},
            transcription_stats={"rows": 5},
            next_scheduled_in_seconds=60,
            backlog_stats={"pending_wavs": 0},
        )
        
        assert status_path.exists()
        data = json.loads(status_path.read_text())
        assert data["last_run_success"] is True
        assert data["vad_stats"]["files"] == 1


@patch("pipeline_runner.process_pending_wavs")
@patch("pipeline_runner.transcribe_pending_segments")
def test_run_pipeline_once(mock_transcribe, mock_vad):
    mock_vad.return_value = {"processed_files": 2, "generated_segments": 10}
    mock_transcribe.return_value = {"new_segments": 10, "new_rows": 8}
    
    config = {"some": "config"}
    summary = run_pipeline_once(config)
    
    assert summary["vad"]["processed_files"] == 2
    assert summary["transcription"]["new_rows"] == 8
    assert "duration_seconds" in summary
    mock_vad.assert_called_once_with(config)
    mock_transcribe.assert_called_once_with(config, export_json=True)
