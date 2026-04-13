import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ecoute.storage import SQLiteStorage


@pytest.fixture
def temp_db_path(tmp_path):
    """Provides a temporary database path for testing."""
    return tmp_path / "test.sqlite"


@pytest.fixture
def storage(temp_db_path):
    """Provides an SQLiteStorage instance with a temporary database."""
    s = SQLiteStorage(temp_db_path)
    yield s


def test_init_creates_db_and_schema(temp_db_path):
    """Test that __init__ creates the database file and initializes the schema."""
    assert not temp_db_path.exists()
    storage = SQLiteStorage(temp_db_path)
    assert temp_db_path.exists()

    # Verify tables exist
    with sqlite3.connect(temp_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "segments" in tables
        assert "transcriptions" in tables


def test_exists(storage, temp_db_path):
    """Test the exists method."""
    assert storage.exists()
    # Test when the file doesn't exist
    temp_db_path.unlink()
    assert not storage.exists()


def test_is_segment_done_and_mark_segment_done(storage):
    """Test marking a segment as done and checking its status."""
    segment_id = "test_segment_1"
    payload = {
        "audio_path": "/path/to/audio.wav",
        "vad_start_ms": 0,
        "vad_end_ms": 1000,
        "audio_mtime": 1678886400.0,
        "audio_size": 12345,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    assert not storage.is_segment_done(segment_id)
    storage.mark_segment_done(segment_id, payload)
    assert storage.is_segment_done(segment_id)

    # Test updating an existing segment
    updated_payload = payload.copy()
    updated_payload["audio_size"] = 54321
    storage.mark_segment_done(segment_id, updated_payload)
    assert storage.is_segment_done(segment_id) # Still done

    # Verify the data was updated
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT audio_size FROM segments WHERE segment_id = ?", (segment_id,)
        ).fetchone()
        assert row["audio_size"] == 54321


def test_append_transcription_rows_and_query_transcriptions(storage):
    """Test appending transcription rows and querying them."""
    segment_id_1 = "seg_1"
    segment_id_2 = "seg_2"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = [
        {
            "segment_id": segment_id_1,
            "timestamp_abs": now,
            "text": "Hello world",
            "confidence": 0.9,
            "audio_path": "/audio/1.wav",
            "vad_start_ms": 0,
            "vad_end_ms": 1000,
            "segment_start_sec": 0.0,
            "segment_end_sec": 1.0,
        },
        {
            "segment_id": segment_id_1,
            "timestamp_abs": now,
            "text": "Another sentence",
            "confidence": 0.75,
            "audio_path": "/audio/1.wav",
            "vad_start_ms": 1000,
            "vad_end_ms": 2000,
            "segment_start_sec": 1.0,
            "segment_end_sec": 2.0,
        },
        {
            "segment_id": segment_id_2,
            "timestamp_abs": now,
            "text": "Goodbye world",
            "confidence": 0.5,
            "audio_path": "/audio/2.wav",
            "vad_start_ms": 0,
            "vad_end_ms": 1500,
            "segment_start_sec": 0.0,
            "segment_end_sec": 1.5,
        },
    ]

    storage.append_transcription_rows(rows)

    # Test basic query
    results = storage.query_transcriptions()
    assert len(results) == 3
    assert results[0]["text"] == "Hello world"
    assert results[2]["text"] == "Goodbye world"

    # Test text query
    results_hello = storage.query_transcriptions(text_query="hello")
    assert len(results_hello) == 1
    assert results_hello[0]["text"] == "Hello world"

    results_world = storage.query_transcriptions(text_query="world")
    assert len(results_world) == 2

    # Test confidence filter
    results_high_conf = storage.query_transcriptions(min_confidence=0.8)
    assert len(results_high_conf) == 1
    assert results_high_conf[0]["text"] == "Hello world"

    # Test text and confidence filter
    results_filtered = storage.query_transcriptions(text_query="sentence", min_confidence=0.7)
    assert len(results_filtered) == 1
    assert results_filtered[0]["text"] == "Another sentence"

    # Test limit and offset
    results_limit_1 = storage.query_transcriptions(limit=1)
    assert len(results_limit_1) == 1
    assert results_limit_1[0]["text"] == "Hello world"

    results_offset_1 = storage.query_transcriptions(offset=1)
    assert len(results_offset_1) == 2
    assert results_offset_1[0]["text"] == "Another sentence"

    results_limit_1_offset_1 = storage.query_transcriptions(limit=1, offset=1)
    assert len(results_limit_1_offset_1) == 1
    assert results_limit_1_offset_1[0]["text"] == "Another sentence"


def test_count_transcriptions(storage):
    """Test counting transcriptions."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        {"segment_id": "s1", "timestamp_abs": now, "text": "apple", "confidence": 0.9},
        {"segment_id": "s2", "timestamp_abs": now, "text": "banana", "confidence": 0.6},
        {"segment_id": "s3", "timestamp_abs": now, "text": "orange", "confidence": 0.4},
    ]
    storage.append_transcription_rows(rows)

    assert storage.count_transcriptions() == 3
    assert storage.count_transcriptions(text_query="apple") == 1
    assert storage.count_transcriptions(min_confidence=0.5) == 2
    assert storage.count_transcriptions(text_query="a", min_confidence=0.5) == 1 # banana


def test_export_json(storage, tmp_path):
    """Test exporting transcriptions to a JSON file."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        {
            "segment_id": "s1",
            "timestamp_abs": now,
            "text": "Export test 1",
            "confidence": 0.9,
            "audio_path": "/audio/export1.wav",
            "vad_start_ms": 0,
            "vad_end_ms": 1000,
            "segment_start_sec": 0.0,
            "segment_end_sec": 1.0,
        },
        {
            "segment_id": "s2",
            "timestamp_abs": now,
            "text": "Export test 2",
            "confidence": 0.8,
            "audio_path": "/audio/export2.wav",
            "vad_start_ms": 1000,
            "vad_end_ms": 2000,
            "segment_start_sec": 1.0,
            "segment_end_sec": 2.0,
        },
    ]
    storage.append_transcription_rows(rows)

    output_file = tmp_path / "exported_transcriptions.json"
    storage.export_json(output_file)

    assert output_file.exists()
    with output_file.open("r", encoding="utf-8") as f:
        exported_data = json.load(f)

    assert len(exported_data) == 2
    assert exported_data[0]["text"] == "Export test 1"
    assert exported_data[1]["confidence"] == 0.8
    assert exported_data[0]["timestamp_abs"] == now
    assert exported_data[1]["audio_path"] == "/audio/export2.wav"

    # Ensure the parent directory is created if it doesn't exist
    nested_output_file = tmp_path / "nested" / "exported.json"
    storage.export_json(nested_output_file)
    assert nested_output_file.exists()
