from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, Optional

from pipeline_config import DEFAULT_CONFIG_PATH, load_config
from transcribe_segments import transcribe_pending_segments
from vad_segment import process_pending_wavs


LOCK_PATH = Path("/tmp/ecoute_pipeline.lock")
STATUS_PATH = Path("runtime/pipeline_status.json")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


class PipelineLock:
    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self._fd: Any = None

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self.lock_path, "w", encoding="utf-8")
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fd.close()
            self._fd = None
            return False

        self._fd.seek(0)
        self._fd.truncate(0)
        self._fd.write(str(os.getpid()))
        self._fd.flush()
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
        self._fd.close()
        self._fd = None


logger = logging.getLogger("ecoute.pipeline")


def setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)


def log(level: int, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"extra_fields": fields})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_backlog_stats(config: Dict[str, Any]) -> Dict[str, int]:
    input_audio_dir = Path(config["paths"]["input_audio_dir"])
    speech_segments_dir = Path(config["paths"]["speech_segments_dir"])

    pending_wavs = len(list(input_audio_dir.glob("*.wav"))) if input_audio_dir.exists() else 0
    pending_segments = (
        len(list(speech_segments_dir.glob("*.wav"))) if speech_segments_dir.exists() else 0
    )

    return {
        "pending_wavs": pending_wavs,
        "pending_segments": pending_segments,
    }


def write_pipeline_status(
    *,
    last_run_started_at: Optional[str],
    last_run_finished_at: Optional[str],
    last_run_success: bool,
    last_error: Optional[str],
    vad_stats: Optional[Dict[str, Any]],
    transcription_stats: Optional[Dict[str, Any]],
    next_scheduled_in_seconds: Optional[int],
    backlog_stats: Optional[Dict[str, int]],
) -> None:
    payload: Dict[str, Any] = {
        "last_run_started_at": last_run_started_at,
        "last_run_finished_at": last_run_finished_at,
        "last_run_success": last_run_success,
        "last_error": last_error,
        "vad_stats": vad_stats,
        "transcription_stats": transcription_stats,
        "next_scheduled_in_seconds": next_scheduled_in_seconds,
    }

    if backlog_stats is not None:
        payload["backlog_stats"] = backlog_stats

    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_pipeline_once(config: Dict[str, Any]) -> Dict[str, Any]:
    run_started = time.monotonic()

    vad_started = time.monotonic()
    vad_stats = process_pending_wavs(config)
    vad_duration = round(time.monotonic() - vad_started, 3)
    log(
        logging.INFO,
        "vad_step_completed",
        step="vad",
        duration_seconds=vad_duration,
        files_processed=vad_stats["processed_files"],
        segments_generated=vad_stats["generated_segments"],
    )

    tr_started = time.monotonic()
    tr_stats = transcribe_pending_segments(config, export_json=True)
    tr_duration = round(time.monotonic() - tr_started, 3)
    log(
        logging.INFO,
        "transcription_step_completed",
        step="transcription",
        duration_seconds=tr_duration,
        files_processed=tr_stats["new_segments"],
        rows_inserted=tr_stats["new_rows"],
    )

    total_duration = round(time.monotonic() - run_started, 3)
    summary = {
        "duration_seconds": total_duration,
        "vad": vad_stats,
        "transcription": tr_stats,
    }
    log(
        logging.INFO,
        "pipeline_run_completed",
        step="pipeline",
        duration_seconds=total_duration,
        files_processed=vad_stats["processed_files"] + tr_stats["new_segments"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one cycle and exit")
    mode.add_argument("--loop", action="store_true", help="Run in infinite loop")
    args = parser.parse_args()

    setup_logging()

    config = load_config(args.config)
    interval_seconds = int(config["pipeline"]["interval_seconds"])
    max_backoff_seconds = int(config["pipeline"]["max_backoff_seconds"])

    lock = PipelineLock(LOCK_PATH)
    if not lock.acquire():
        log(logging.WARNING, "pipeline_lock_already_held", lock_path=str(LOCK_PATH))
        return

    backoff_seconds = max(1, min(5, interval_seconds))
    loop_mode = args.loop or not args.once

    log(
        logging.INFO,
        "pipeline_runner_started",
        mode="loop" if loop_mode else "once",
        interval_seconds=interval_seconds,
        lock_path=str(LOCK_PATH),
    )

    try:
        while True:
            started_at = utc_now_iso()
            finished_at: Optional[str] = None
            try:
                run_summary = run_pipeline_once(config)
                finished_at = utc_now_iso()
                next_scheduled = interval_seconds if loop_mode else None
                write_pipeline_status(
                    last_run_started_at=started_at,
                    last_run_finished_at=finished_at,
                    last_run_success=True,
                    last_error=None,
                    vad_stats=run_summary["vad"],
                    transcription_stats=run_summary["transcription"],
                    next_scheduled_in_seconds=next_scheduled,
                    backlog_stats=compute_backlog_stats(config),
                )
                backoff_seconds = max(1, min(5, interval_seconds))
            except Exception as exc:
                finished_at = utc_now_iso()
                write_pipeline_status(
                    last_run_started_at=started_at,
                    last_run_finished_at=finished_at,
                    last_run_success=False,
                    last_error=str(exc),
                    vad_stats=None,
                    transcription_stats=None,
                    next_scheduled_in_seconds=backoff_seconds if loop_mode else None,
                    backlog_stats=compute_backlog_stats(config),
                )
                log(
                    logging.ERROR,
                    "pipeline_run_failed",
                    error=str(exc),
                    retry_in_seconds=backoff_seconds,
                )
                if not loop_mode:
                    raise
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, max_backoff_seconds)
                continue

            if not loop_mode:
                break

            log(logging.INFO, "pipeline_sleep", sleep_seconds=interval_seconds)
            time.sleep(interval_seconds)
    finally:
        lock.release()
        log(logging.INFO, "pipeline_runner_stopped")


if __name__ == "__main__":
    main()
