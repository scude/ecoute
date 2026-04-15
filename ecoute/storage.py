from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DB_PATH = Path("transcriptions/transcriptions.sqlite")


class SQLiteStorage:
    def __init__(
        self, 
        db_path: Path = DEFAULT_DB_PATH,
        banned_phrases: list[str] | None = None
    ) -> None:
        self.db_path = db_path
        self.banned_phrases = banned_phrases or []
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id TEXT PRIMARY KEY,
                    audio_path TEXT NOT NULL,
                    vad_start_ms INTEGER NOT NULL,
                    vad_end_ms INTEGER NOT NULL,
                    audio_mtime REAL NOT NULL,
                    audio_size INTEGER NOT NULL,
                    processed_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id TEXT NOT NULL,
                    timestamp_abs TEXT,
                    text TEXT,
                    confidence REAL,
                    audio_path TEXT,
                    vad_start_ms INTEGER,
                    vad_end_ms INTEGER,
                    segment_start_sec REAL,
                    segment_end_sec REAL,
                    FOREIGN KEY(segment_id) REFERENCES segments(segment_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_transcriptions_timestamp_abs
                ON transcriptions(timestamp_abs)
                """
            )

    def exists(self) -> bool:
        return self.db_path.exists()

    def is_segment_done(self, segment_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM segments WHERE segment_id = ? LIMIT 1",
                (segment_id,),
            ).fetchone()
            return row is not None

    def mark_segment_done(self, segment_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO segments(
                    segment_id, audio_path, vad_start_ms, vad_end_ms,
                    audio_mtime, audio_size, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment_id,
                    payload["audio_path"],
                    int(payload["vad_start_ms"]),
                    int(payload["vad_end_ms"]),
                    float(payload["audio_mtime"]),
                    int(payload["audio_size"]),
                    payload.get("processed_at")
                    or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def append_transcription_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        values = [
            (
                row["segment_id"],
                row.get("timestamp_abs"),
                row.get("text"),
                row.get("confidence"),
                row.get("audio_path"),
                row.get("vad_start_ms"),
                row.get("vad_end_ms"),
                row.get("segment_start_sec"),
                row.get("segment_end_sec"),
            )
            for row in rows
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO transcriptions(
                    segment_id, timestamp_abs, text, confidence,
                    audio_path, vad_start_ms, vad_end_ms,
                    segment_start_sec, segment_end_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )

    def query_transcriptions(
            self,
            *,
            text_query: str = "",
            min_confidence: float | None = None,
            limit: int = 200,
            offset: int = 0,
            sort_desc: bool = False,
    ) -> dict[str, list[dict[str, Any]]]: # Modified return type
        sql = (
            "SELECT segment_id, timestamp_abs, text, confidence, audio_path, " # Added segment_id
            "vad_start_ms, vad_end_ms, segment_start_sec, segment_end_sec "
            "FROM transcriptions WHERE 1=1"
        )
        params: list[Any] = []

        if min_confidence is not None and min_confidence > 0:
            sql += " AND COALESCE(confidence, 0.0) >= ?"
            params.append(min_confidence)

        if text_query and text_query.strip():
            sql += " AND LOWER(COALESCE(text, '')) LIKE ?"
            params.append(f"%{text_query.lower()}%")
            
        # Filtre des phrases bannies (exact match)
        if self.banned_phrases:
            sql += " AND TRIM(COALESCE(text, '')) NOT IN ({})".format(
                ",".join(["?"] * len(self.banned_phrases))
            )
            params.extend(self.banned_phrases)

        order = "DESC" if sort_desc else "ASC"

        # On ajoute un tri secondaire pour garantir un ordre stable même si les timestamps sont identiques
        sql += (
            f" ORDER BY timestamp_abs {order}, confidence {order}, "
            f"audio_path {order}, segment_start_sec {order} "
            f"LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        # Group transcriptions by segment_id
        grouped_transcriptions: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            row_dict = dict(row)
            segment_id = row_dict.pop("segment_id") # Remove segment_id from individual transcription dict
            if segment_id not in grouped_transcriptions:
                grouped_transcriptions[segment_id] = []
            grouped_transcriptions[segment_id].append(row_dict)

        return grouped_transcriptions

    def count_transcriptions(self, text_query: str = "", min_confidence: float | None = None) -> int:
        sql = "SELECT COUNT(*) AS c FROM transcriptions WHERE 1=1"
        params: list[Any] = []

        if min_confidence is not None and min_confidence > 0:
            sql += " AND COALESCE(confidence, 0.0) >= ?"
            params.append(min_confidence)

        if text_query and text_query.strip():
            sql += " AND LOWER(COALESCE(text, '')) LIKE ?"
            params.append(f"%{text_query.lower()}%")
            
        # Filtre des phrases bannies (exact match)
        if self.banned_phrases:
            sql += " AND TRIM(COALESCE(text, '')) NOT IN ({})".format(
                ",".join(["?"] * len(self.banned_phrases))
            )
            params.extend(self.banned_phrases)

        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return int(row["c"]) if row else 0

    def export_json(self, output_file: Path) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp_abs, text, confidence, audio_path,
                       vad_start_ms, vad_end_ms, segment_start_sec, segment_end_sec
                FROM transcriptions
                ORDER BY timestamp_abs ASC, audio_path ASC, segment_start_sec ASC
                """
            ).fetchall()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = [dict(r) for r in rows]
        output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["SQLiteStorage", "DEFAULT_DB_PATH"]
