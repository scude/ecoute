from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict

from pipeline_config import DEFAULT_CONFIG_PATH, load_config
from transcribe_segments import transcribe_pending_segments
from vad_segment import process_pending_wavs


LOCK_PATH = Path("/tmp/ecoute_pipeline.lock")


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
            try:
                run_pipeline_once(config)
                backoff_seconds = max(1, min(5, interval_seconds))
            except Exception as exc:
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
