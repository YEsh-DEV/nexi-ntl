"""Durable, PII-free operational telemetry store using standard library SQLite."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from agent.config import Settings

DB_FILENAME = "nexi.sqlite3"


class TelemetryStore:
    def __init__(self, settings: Settings) -> None:
        self.db_dir = settings.chroma_dir.parent / "telemetry"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / DB_FILENAME
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS call_logs (
                    call_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    turns INTEGER NOT NULL,
                    tool_calls INTEGER NOT NULL,
                    no_results INTEGER NOT NULL,
                    errors INTEGER NOT NULL,
                    timeouts INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    knowledge_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    usage_summary TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS stage_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    language TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_gaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT NOT NULL,
                    language TEXT NOT NULL,
                    knowledge_version TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                """
            )

    def record_call_sync(
        self,
        call_id: str,
        started_at: float,
        duration: float,
        channel: str,
        language: str,
        status: str,
        turns: int,
        tool_calls: int,
        no_results: int,
        errors: int,
        timeouts: int,
        knowledge_version: str,
        prompt_version: str,
        usage: dict,
    ) -> None:
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        # Defensive check against raw monotonic timestamps
        if started_at < 1_000_000_000:
            start_dt = datetime.fromtimestamp(
                now_dt.timestamp() - max(0.0, float(duration)), tz=timezone.utc
            )
        else:
            start_dt = datetime.fromtimestamp(started_at, tz=timezone.utc)

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO call_logs (
                    call_id, started_at, ended_at, channel, language, status,
                    turns, tool_calls, no_results, errors, timeouts,
                    duration_seconds, knowledge_version, prompt_version, usage_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    call_id,
                    start_dt.isoformat(),
                    now_iso,
                    channel,
                    language,
                    status,
                    turns,
                    tool_calls,
                    no_results,
                    errors,
                    timeouts,
                    duration,
                    knowledge_version,
                    prompt_version,
                    json.dumps(usage, ensure_ascii=False),
                ),
            )

    async def record_call(self, **kwargs: object) -> None:
        await asyncio.to_thread(self.record_call_sync, **kwargs)

    def record_stage_metrics_sync(
        self,
        call_id: str,
        channel: str,
        language: str,
        stage: str,
        measurements: dict,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO stage_metrics (
                    call_id, channel, language, stage, metrics_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    call_id,
                    channel,
                    language,
                    stage,
                    json.dumps(measurements, ensure_ascii=False),
                    now_iso,
                ),
            )

    async def record_stage_metrics(self, **kwargs: object) -> None:
        await asyncio.to_thread(self.record_stage_metrics_sync, **kwargs)

    def record_knowledge_gap_sync(
        self,
        call_id: str,
        language: str,
        knowledge_version: str,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_gaps (
                    call_id, language, knowledge_version, recorded_at
                ) VALUES (?, ?, ?, ?);
                """,
                (call_id, language, knowledge_version, now_iso),
            )

    async def record_knowledge_gap(self, **kwargs: object) -> None:
        await asyncio.to_thread(self.record_knowledge_gap_sync, **kwargs)

    def query_recent_calls(self, limit: int = 20) -> list[dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM call_logs ORDER BY ended_at DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def query_knowledge_gaps(self, limit: int = 20) -> list[dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM knowledge_gaps ORDER BY recorded_at DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
