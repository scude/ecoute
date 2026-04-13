from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNTIME_DIR = Path(os.environ.get("RUNTIME_DIR", "runtime"))
CAPTURE_HEARTBEAT_PATH = RUNTIME_DIR / "capture_heartbeat.json"
PIPELINE_STATUS_PATH = RUNTIME_DIR / "pipeline_status.json"

AUDIOS_DIR = Path(os.environ.get("INPUT_AUDIO_DIR", "audios"))
SPEECH_SEGMENTS_DIR = Path(os.environ.get("SPEECH_SEGMENTS_DIR", "speech_segments"))
AUDIOS_PROCESSED_DIR = Path(os.environ.get("PROCESSED_AUDIO_DIR", "audios_processed"))
TRANSCRIPTIONS_DIR = Path(os.environ.get("TRANSCRIPTIONS_DIR", "transcriptions"))

CAPTURE_HEARTBEAT_MAX_AGE_SECONDS = int(
    os.environ.get("MONITORING_CAPTURE_HEARTBEAT_MAX_AGE_SECONDS", "180")
)
STORAGE_WARN_THRESHOLD_PERCENT = float(os.environ.get("MONITORING_STORAGE_WARN_PERCENT", "80"))
STORAGE_CRITICAL_THRESHOLD_PERCENT = float(
    os.environ.get("MONITORING_STORAGE_CRITICAL_PERCENT", "90")
)


@dataclass
class ThresholdLevel:
    level: str
    label: str


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def read_runtime_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _file_age_seconds(last_update: datetime | None) -> float | None:
    if last_update is None:
        return None
    return max(0.0, (_now_utc() - last_update.astimezone(timezone.utc)).total_seconds())


def _human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def _folder_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def _threshold_level(percent_value: float) -> ThresholdLevel:
    if percent_value >= STORAGE_CRITICAL_THRESHOLD_PERCENT:
        return ThresholdLevel(level="critical", label="🔴 Critique")
    if percent_value >= STORAGE_WARN_THRESHOLD_PERCENT:
        return ThresholdLevel(level="warning", label="🟠 Alerte")
    return ThresholdLevel(level="ok", label="🟢 OK")


def _format_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def get_monitoring_snapshot() -> dict[str, Any]:
    capture_payload = read_runtime_json(CAPTURE_HEARTBEAT_PATH) or {}
    pipeline_payload = read_runtime_json(PIPELINE_STATUS_PATH) or {}

    capture_updated = _parse_datetime(
        capture_payload.get("updated_at_epoch") or capture_payload.get("updated_at")
    )
    capture_age_seconds = _file_age_seconds(capture_updated)
    capture_recent = (
        capture_age_seconds is not None and capture_age_seconds <= CAPTURE_HEARTBEAT_MAX_AGE_SECONDS
    )

    pipeline_finished_at = _parse_datetime(pipeline_payload.get("last_run_finished_at"))
    pipeline_age_seconds = _file_age_seconds(pipeline_finished_at)

    pending_audios = len(list(AUDIOS_DIR.glob("*.wav"))) if AUDIOS_DIR.exists() else 0
    pending_segments = len(list(SPEECH_SEGMENTS_DIR.glob("*.wav"))) if SPEECH_SEGMENTS_DIR.exists() else 0

    disk = shutil.disk_usage(Path("."))
    used_percent = (disk.used / disk.total * 100) if disk.total > 0 else 0.0
    usage_level = _threshold_level(used_percent)

    critical_dirs = {
        "audios": AUDIOS_DIR,
        "audios_processed": AUDIOS_PROCESSED_DIR,
        "speech_segments": SPEECH_SEGMENTS_DIR,
        "transcriptions": TRANSCRIPTIONS_DIR,
    }
    folder_sizes = {}
    for name, path in critical_dirs.items():
        size_bytes = _folder_size_bytes(path)
        folder_sizes[name] = {
            "path": str(path),
            "bytes": size_bytes,
            "human": _format_bytes(size_bytes),
        }

    return {
        "capture": {
            "state": capture_payload.get("state", "unknown"),
            "recent": capture_recent,
            "age_seconds": capture_age_seconds,
            "age_human": _human_duration(capture_age_seconds),
            "max_age_seconds": CAPTURE_HEARTBEAT_MAX_AGE_SECONDS,
        },
        "pipeline": {
            "last_run_success": pipeline_payload.get("last_run_success"),
            "last_error": pipeline_payload.get("last_error"),
            "age_seconds": pipeline_age_seconds,
            "age_human": _human_duration(pipeline_age_seconds),
            "last_run_started_at": pipeline_payload.get("last_run_started_at"),
            "last_run_finished_at": pipeline_payload.get("last_run_finished_at"),
        },
        "pending": {
            "audios": pending_audios,
            "speech_segments": pending_segments,
            "total": pending_audios + pending_segments,
        },
        "storage": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "total_human": _format_bytes(disk.total),
            "used_human": _format_bytes(disk.used),
            "free_human": _format_bytes(disk.free),
            "used_percent": used_percent,
            "usage_level": usage_level.level,
            "usage_label": usage_level.label,
            "warn_threshold_percent": STORAGE_WARN_THRESHOLD_PERCENT,
            "critical_threshold_percent": STORAGE_CRITICAL_THRESHOLD_PERCENT,
            "folder_sizes": folder_sizes,
        },
    }
