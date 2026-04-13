import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ui.app import (
    resolve_audio_path,
    confidence_badge,
    format_timestamp_fr,
    _format_capture_status,
    _format_pipeline_status,
)


def test_resolve_audio_path(tmp_path):
    # Test file exists
    existing_file = tmp_path / "exists.wav"
    existing_file.write_text("...")
    assert resolve_audio_path(str(existing_file)) == existing_file

    # Test remapping
    # Requested path: /some/host/path/speech_segments/abc/def.wav
    # Local path: {tmp_path}/speech_segments/abc/def.wav
    host_path = "/home/user/ecoute/speech_segments/session1/seg1.wav"
    local_root = tmp_path / "speech_segments"
    local_file = local_root / "session1" / "seg1.wav"
    local_file.parent.mkdir(parents=True)
    local_file.write_text("...")

    with patch.dict(os.environ, {"SPEECH_SEGMENTS_DIR": str(local_root)}):
        resolved = resolve_audio_path(host_path)
        assert resolved == local_file

    # Test remapping not found
    assert resolve_audio_path("/other/path/nothing.wav") == Path("/other/path/nothing.wav")


def test_confidence_badge():
    assert "N/A" in confidence_badge(None)
    assert "N/A" in confidence_badge(float("nan"))
    assert "Faible" in confidence_badge(0.2)
    assert "Moyen" in confidence_badge(0.5)
    assert "Élevé" in confidence_badge(0.8)


def test_format_timestamp_fr():
    ts = "2023-03-15T12:00:00"
    assert format_timestamp_fr(ts) == "15/03/2023 12:00:00"
    assert format_timestamp_fr(None) == "N/A"
    assert format_timestamp_fr("invalid") == "N/A"


def test_format_status_helpers():
    assert "active" in _format_capture_status({"recent": True})
    assert "stale" in _format_capture_status({"recent": False})

    assert "OK" in _format_pipeline_status({"last_run_success": True})
    assert "échec" in _format_pipeline_status({"last_run_success": False})
    assert "inconnu" in _format_pipeline_status({"last_run_success": None})
