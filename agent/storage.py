"""Durable operational telemetry store using SQLite with migrations, integrity recovery, and privacy."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import secrets
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from agent.config import Settings

logger = logging.getLogger(__name__)

DB_FILENAME = "nexi.sqlite3"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def compute_caller_hash(phone_number: str, pii_key: str) -> str:
    """HMAC-SHA256 of telephone number, returning first 16 hex characters."""
    clean = re.sub(r"[^\d+]", "", phone_number)
    key_bytes = pii_key.encode("utf-8") if pii_key else b"dev-insecure-key"
    digest = hmac.new(key_bytes, clean.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:16]


def mask_caller_number(phone_number: str) -> str:
    """Mask phone number keeping country prefix and last 2 digits, e.g. +91********12."""
    clean = re.sub(r"[^\d+]", "", phone_number)
    if not clean:
        return "***"
    if clean.startswith("+"):
        prefix = clean[:3]
        suffix = clean[-2:] if len(clean) >= 5 else ""
        stars = "*" * max(4, len(clean) - len(prefix) - len(suffix))
        return f"{prefix}{stars}{suffix}"
    else:
        prefix = clean[:2]
        suffix = clean[-2:] if len(clean) >= 4 else ""
        stars = "*" * max(4, len(clean) - len(prefix) - len(suffix))
        return f"{prefix}{stars}{suffix}"


def redact_query_text(query: str, extra_patterns: list[str] | None = None) -> tuple[str, str]:
    """Redact PII from question text: emails, phone numbers, digit runs >=6, extra regexes.

    Returns (redacted_query, query_hash).
    """
    text = query.strip()
    # 1. Emails
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[redacted]", text)
    # 2. Phone numbers (7+ digits allowing spaces, dashes, brackets)
    text = re.sub(r"(?:\+?\d[\d\s\-()]{5,}\d)", "[redacted]", text)
    # 3. Digit runs of 6 or more (roll numbers, student IDs, pins)
    text = re.sub(r"\b\d{6,}\b", "[redacted]", text)
    # 4. Extra patterns
    if extra_patterns:
        for pat in extra_patterns:
            try:
                text = re.sub(pat, "[redacted]", text)
            except re.error as pat_err:
                logger.debug("Skipping invalid regex extra pattern %r: %s", pat, pat_err)

    # Normalize whitespace and truncate to 200 characters
    norm = " ".join(text.split())[:200]
    q_hash = hashlib.sha256(norm.lower().encode("utf-8")).hexdigest()
    return norm, q_hash


class TelemetryStore:
    def __init__(self, settings: Settings, db_path: Path | None = None) -> None:
        self.settings = settings
        if db_path is not None:
            self.db_path = db_path
            self.db_dir = db_path.parent
        else:
            self.db_dir = settings.chroma_dir.parent / "telemetry"
            self.db_path = self.db_dir / DB_FILENAME

        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.degraded = False
        self.recovered = False

        # Verify or generate PII hash key
        self.pii_key = settings.pii_hash_key
        if not self.pii_key:
            if settings.environment == "production":
                raise ValueError("PII_HASH_KEY is strictly required in production")
            logger.warning("WARNING: PII_HASH_KEY not set. Using ephemeral random key for development.")
            self.pii_key = secrets.token_hex(16)

        self._check_integrity_and_migrate()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _check_integrity_and_migrate(self) -> None:
        if self.db_path.is_file():
            try:
                with self._connection() as conn:
                    cursor = conn.execute("PRAGMA integrity_check;")
                    row = cursor.fetchone()
                    if not row or row[0] != "ok":
                        self._handle_corrupt_database("integrity_check failed")
            except Exception as exc:
                logger.error("Failed opening database for integrity check: %s", exc)
                self._handle_corrupt_database(str(exc))

        self._apply_migrations()

    def _handle_corrupt_database(self, reason: str) -> None:
        now_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        corrupt_target = self.db_dir / f"{self.db_path.name}.corrupt-{now_str}"
        logger.warning(
            "WARNING: Corrupt database detected (%s). Renaming %s to %s and creating fresh store.",
            reason,
            self.db_path,
            corrupt_target,
        )
        try:
            if self.db_path.is_file():
                shutil.move(str(self.db_path), str(corrupt_target))
            wal = self.db_path.with_name(f"{self.db_path.name}-wal")
            shm = self.db_path.with_name(f"{self.db_path.name}-shm")
            if wal.is_file():
                shutil.move(str(wal), str(self.db_dir / f"{wal.name}.corrupt-{now_str}"))
            if shm.is_file():
                shutil.move(str(shm), str(self.db_dir / f"{shm.name}.corrupt-{now_str}"))
            self.recovered = True
        except Exception as exc:
            logger.error("Failed renaming corrupt database: %s", exc)
            self.degraded = True

    def _apply_migrations(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at_utc TEXT NOT NULL
                );
                """
            )
            cursor = conn.execute("SELECT version FROM schema_migrations ORDER BY version ASC;")
            applied_versions = {row[0] for row in cursor.fetchall()}

            migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
            known_versions = {}
            for mf in migration_files:
                match = re.match(r"^(\d{4})_", mf.name)
                if match:
                    v_int = int(match.group(1))
                    known_versions[v_int] = mf

            max_known = max(known_versions.keys()) if known_versions else 0
            for v in applied_versions:
                if v > max_known:
                    raise RuntimeError(
                        f"Database contains migration version {v} newer than maximum known migration {max_known}"
                    )

            now_iso = datetime.now(timezone.utc).isoformat()
            for v_int in sorted(known_versions.keys()):
                if v_int not in applied_versions:
                    sql_text = known_versions[v_int].read_text(encoding="utf-8")
                    conn.executescript(sql_text)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at_utc) VALUES (?, ?);",
                        (v_int, now_iso),
                    )
                    logger.info("Applied database migration %04d (%s)", v_int, known_versions[v_int].name)

    # -------------------------------------------------------------------------
    # Call Lifecycle & Status Tracking
    # -------------------------------------------------------------------------
    def record_call_started_sync(
        self,
        call_id: str,
        channel: str,
        language: str,
        job_id: str = "",
        room_hash: str = "",
        caller_number: str = "",
        started_at_utc: str | None = None,
    ) -> None:
        started_at_utc = started_at_utc or datetime.now(timezone.utc).isoformat()
        caller_hash = compute_caller_hash(caller_number, self.pii_key) if caller_number else ""
        caller_mask = mask_caller_number(caller_number) if caller_number else ""

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO calls (
                    call_id, job_id, room_hash, channel, language_initial, language_final,
                    status, started_at_utc, caller_hash, caller_mask
                ) VALUES (?, ?, ?, ?, ?, ?, 'started', ?, ?, ?);
                """,
                (
                    call_id,
                    job_id,
                    room_hash,
                    channel,
                    language,
                    language,
                    started_at_utc,
                    caller_hash,
                    caller_mask,
                ),
            )

    def record_call_ended_sync(
        self,
        call_id: str,
        status: str,
        ended_at_utc: str,
        duration_s: float,
        end_reason: str,
        language_final: str,
        turns: int,
        tool_calls: int,
        no_results_count: int,
        language_switches: int,
        language_switch_failures: int,
        transfer_attempted: int,
        transfer_succeeded: int,
        transfer_department: str,
        room_delete_failures: int,
        errors: int,
        usage: dict[str, Any],
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE calls SET
                    status = ?,
                    ended_at_utc = ?,
                    duration_s = ?,
                    end_reason = ?,
                    language_final = ?,
                    turns = ?,
                    tool_calls = ?,
                    no_results_count = ?,
                    language_switches = ?,
                    language_switch_failures = ?,
                    transfer_attempted = ?,
                    transfer_succeeded = ?,
                    transfer_department = ?,
                    room_delete_failures = ?,
                    errors = ?,
                    usage_json = ?
                WHERE call_id = ?;
                """,
                (
                    status,
                    ended_at_utc,
                    duration_s,
                    end_reason,
                    language_final,
                    turns,
                    tool_calls,
                    no_results_count,
                    language_switches,
                    language_switch_failures,
                    transfer_attempted,
                    transfer_succeeded,
                    transfer_department,
                    room_delete_failures,
                    errors,
                    json.dumps(usage, ensure_ascii=False),
                    call_id,
                ),
            )

    # -------------------------------------------------------------------------
    # Turn Metrics
    # -------------------------------------------------------------------------
    def record_turn_metrics_sync(
        self,
        call_id: str,
        channel: str,
        language: str,
        eou_delay_ms: float = 0.0,
        llm_ttft_ms: float = 0.0,
        tts_ttfb_ms: float = 0.0,
        total_ms: float = 0.0,
        retrieval_ms: float = 0.0,
        tool_calls: int = 0,
        ts_utc: str | None = None,
    ) -> None:
        ts_utc = ts_utc or datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO turn_metrics (
                    call_id, ts_utc, channel, language, eou_delay_ms, llm_ttft_ms,
                    tts_ttfb_ms, total_ms, retrieval_ms, tool_calls
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    call_id,
                    ts_utc,
                    channel,
                    language,
                    eou_delay_ms,
                    llm_ttft_ms,
                    tts_ttfb_ms,
                    total_ms,
                    retrieval_ms,
                    tool_calls,
                ),
            )

    # -------------------------------------------------------------------------
    # Knowledge Gaps
    # -------------------------------------------------------------------------
    def record_knowledge_gap_sync(
        self,
        query: str = "",
        language: str = "en",
        channel: str = "web",
        reason: str = "no_results",
        top_score: float = 0.0,
        nearest_topic: str = "",
        timestamp_utc: str | None = None,
        call_id: str | None = None,
        knowledge_version: str | None = None,
    ) -> tuple[str, str] | None:
        timestamp_utc = timestamp_utc or datetime.now(timezone.utc).isoformat()
        if call_id is not None:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO knowledge_gaps_legacy (
                        call_id, language, knowledge_version, recorded_at
                    ) VALUES (?, ?, ?, ?);
                    """,
                    (call_id, language, knowledge_version or "", timestamp_utc),
                )
            return None

        redacted_q, q_hash = redact_query_text(query, self.settings.pii_extra_patterns)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_gaps (
                    query_hash, query_redacted, first_seen_utc, last_seen_utc, count,
                    language, channel, reason, top_score, nearest_topic, status
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, 'open')
                ON CONFLICT(query_hash) DO UPDATE SET
                    count = count + 1,
                    last_seen_utc = excluded.last_seen_utc,
                    reason = excluded.reason,
                    top_score = excluded.top_score,
                    nearest_topic = excluded.nearest_topic;
                """,
                (
                    q_hash,
                    redacted_q,
                    timestamp_utc,
                    timestamp_utc,
                    language,
                    channel,
                    reason,
                    top_score,
                    nearest_topic,
                ),
            )
        return redacted_q, q_hash

    def update_gap_status(self, gap_id: int, status: str, note: str | None = None) -> bool:
        if status not in ("open", "resolved", "ignored"):
            raise ValueError(f"Invalid gap status '{status}'")
        now_iso = datetime.now(timezone.utc).isoformat() if status != "open" else None
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE knowledge_gaps SET
                    status = ?,
                    resolved_note = ?,
                    resolved_at_utc = ?
                WHERE id = ?;
                """,
                (status, note, now_iso, gap_id),
            )
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Legacy / Backwards-Compatible Helpers
    # -------------------------------------------------------------------------
    def record_call_sync(
        self,
        call_id: str,
        started_at: float,
        duration: float,
        channel: str,
        language: str,
        status: str,
        turns: int = 0,
        tool_calls: int = 0,
        no_results: int = 0,
        errors: int = 0,
        timeouts: int = 0,
        knowledge_version: str = "",
        prompt_version: str = "",
        usage: dict[str, Any] | None = None,
    ) -> None:
        usage = usage or {}
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
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

    async def record_call(self, **kwargs: Any) -> None:
        await asyncio.to_thread(self.record_call_sync, **kwargs)

    def record_stage_metrics_sync(
        self,
        call_id: str,
        channel: str,
        language: str,
        stage: str,
        measurements: dict[str, Any],
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

    async def record_stage_metrics(self, **kwargs: Any) -> None:
        await asyncio.to_thread(self.record_stage_metrics_sync, **kwargs)

    async def record_knowledge_gap(self, **kwargs: Any) -> None:
        await asyncio.to_thread(self.record_knowledge_gap_sync, **kwargs)

    def query_recent_calls(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM call_logs ORDER BY ended_at DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
            if rows:
                return [dict(r) for r in rows]
            cur2 = conn.execute("SELECT * FROM calls ORDER BY started_at_utc DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur2.fetchall()]

    def query_knowledge_gaps(self, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            if status:
                cur = conn.execute(
                    "SELECT * FROM knowledge_gaps WHERE status = ? ORDER BY count DESC, last_seen_utc DESC LIMIT ?",
                    (status, limit),
                )
                return [dict(r) for r in cur.fetchall()]
            cur_legacy = conn.execute("SELECT * FROM knowledge_gaps_legacy ORDER BY recorded_at DESC LIMIT ?", (limit,))
            legacy_rows = cur_legacy.fetchall()
            if legacy_rows:
                return [dict(r) for r in legacy_rows]
            cur = conn.execute(
                "SELECT * FROM knowledge_gaps ORDER BY count DESC, last_seen_utc DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
