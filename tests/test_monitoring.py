import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import monitoring
from monitoring import (
    _parse_datetime,
    _human_duration,
    _format_bytes,
    _folder_size_bytes,
    _threshold_level,
    read_runtime_json,
    get_monitoring_snapshot,
)


def test_parse_datetime():
    # Test None
    assert _parse_datetime(None) is None
    
    # Test epoch (int/float)
    epoch = 1678886400.0
    expected = datetime.fromtimestamp(epoch, tz=timezone.utc)
    assert _parse_datetime(epoch) == expected
    assert _parse_datetime(int(epoch)) == expected

    # Test ISO string
    iso_str = "2023-03-15T13:20:00Z"
    expected_iso = datetime(2023, 3, 15, 13, 20, 0, tzinfo=timezone.utc)
    assert _parse_datetime(iso_str) == expected_iso

    # Test invalid string
    assert _parse_datetime("invalid-date") is None
    
    # Test other types
    assert _parse_datetime([]) is None


def test_human_duration():
    assert _human_duration(None) == "N/A"
    assert _human_duration(30) == "30s"
    assert _human_duration(90) == "1m 30s"
    assert _human_duration(3661) == "1h 1m"
    assert _human_duration(0) == "0s"


def test_format_bytes():
    assert _format_bytes(500) == "500.0 B"
    assert _format_bytes(1024) == "1.0 KB"
    assert _format_bytes(1024 * 1024) == "1.0 MB"
    assert _format_bytes(1024 * 1024 * 1024) == "1.0 GB"
    assert _format_bytes(1024**4) == "1.0 TB"


def test_folder_size_bytes(tmp_path):
    test_dir = tmp_path / "test_size"
    test_dir.mkdir()
    assert _folder_size_bytes(test_dir) == 0
    
    file1 = test_dir / "file1.txt"
    file1.write_text("hello") # 5 bytes
    
    subdir = test_dir / "sub"
    subdir.mkdir()
    file2 = subdir / "file2.txt"
    file2.write_text("world!") # 6 bytes
    
    assert _folder_size_bytes(test_dir) == 11
    
    # Test non-existent path
    assert _folder_size_bytes(tmp_path / "non_existent") == 0


def test_threshold_level():
    # Using default thresholds: warn=80, critical=90
    assert _threshold_level(50).level == "ok"
    assert _threshold_level(80).level == "warning"
    assert _threshold_level(85).level == "warning"
    assert _threshold_level(90).level == "critical"
    assert _threshold_level(95).level == "critical"


def test_read_runtime_json(tmp_path):
    path = tmp_path / "test.json"
    
    # File doesn't exist
    assert read_runtime_json(path) is None
    
    # Valid JSON
    data = {"key": "value"}
    path.write_text(json.dumps(data))
    assert read_runtime_json(path) == data
    
    # Invalid JSON
    path.write_text("{invalid")
    assert read_runtime_json(path) is None


@patch("monitoring.shutil.disk_usage")
@patch("monitoring._now_utc")
def test_get_monitoring_snapshot(mock_now, mock_disk_usage, tmp_path):
    # Setup paths and environment
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    audios_dir = tmp_path / "audios"
    audios_dir.mkdir()
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    
    # Mocking environment variables
    with patch.dict(os.environ, {
        "RUNTIME_DIR": str(runtime_dir),
        "INPUT_AUDIO_DIR": str(audios_dir),
        "SPEECH_SEGMENTS_DIR": str(segments_dir),
    }):
        # Update monitoring globals that depend on environ
        monitoring.RUNTIME_DIR = runtime_dir
        monitoring.CAPTURE_HEARTBEAT_PATH = runtime_dir / "capture_heartbeat.json"
        monitoring.PIPELINE_STATUS_PATH = runtime_dir / "pipeline_status.json"
        monitoring.AUDIOS_DIR = audios_dir
        monitoring.SPEECH_SEGMENTS_DIR = segments_dir
        monitoring.AUDIOS_PROCESSED_DIR = tmp_path / "processed"
        monitoring.TRANSCRIPTIONS_DIR = tmp_path / "transcriptions"

        now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_now.return_value = now
        
        mock_disk_usage.return_value = MagicMock(total=1000, used=500, free=500)
        
        # Create heartbeat file (recent)
        heartbeat = {"updated_at_epoch": now.timestamp() - 10, "state": "running"}
        monitoring.CAPTURE_HEARTBEAT_PATH.write_text(json.dumps(heartbeat))
        
        # Create pipeline status file
        pipeline = {"last_run_success": True, "last_run_finished_at": (now - timedelta(minutes=5)).isoformat()}
        monitoring.PIPELINE_STATUS_PATH.write_text(json.dumps(pipeline))
        
        # Add some dummy files for pending count
        (audios_dir / "1.wav").write_text("...")
        (segments_dir / "seg1.wav").write_text("...")
        (segments_dir / "seg2.wav").write_text("...")

        snapshot = get_monitoring_snapshot()
        
        assert snapshot["capture"]["recent"] is True
        assert snapshot["capture"]["age_seconds"] == 10
        assert snapshot["pipeline"]["last_run_success"] is True
        assert snapshot["pending"]["audios"] == 1
        assert snapshot["pending"]["speech_segments"] == 2
        assert snapshot["pending"]["total"] == 3
        assert snapshot["storage"]["used_percent"] == 50.0
        assert snapshot["storage"]["usage_level"] == "ok"
