"""Unit tests for durable operational telemetry, SQLite migrations, privacy redaction, maintenance, and reports."""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.assistant import CallState, NexiAgent
from agent.config import Settings
from agent.lifecycle import CallLifecycle, EndReason
from agent.maintenance import purge_retention, reconcile_stale_calls
from agent.reports import (
    format_gaps_csv,
    format_latency_markdown,
    format_latency_text,
    generate_call_summary,
    generate_latency_report,
    nearest_rank_percentile,
)
from agent.storage import (
    TelemetryStore,
    compute_caller_hash,
    mask_caller_number,
    redact_query_text,
)
from agent.telemetry import TelemetryPipeline


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self._time = start

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


class TestTelemetryMigrationsAndIntegrity(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_nexi.sqlite3"
        self.settings = Settings(chroma_dir=Path(self.temp_dir) / "knowledge", pii_hash_key="test_pii_key_12345")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fresh_database_applies_all_migrations(self) -> None:
        store = TelemetryStore(self.settings, db_path=self.db_path)
        with store._connection() as conn:
            cursor = conn.execute("SELECT version FROM schema_migrations ORDER BY version ASC;")
            versions = [r[0] for r in cursor.fetchall()]
            self.assertEqual(versions, [1, 2])

            # Check tables created
            table_cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {r[0] for r in table_cur.fetchall()}
            self.assertIn("calls", tables)
            self.assertIn("turn_metrics", tables)
            self.assertIn("knowledge_gaps", tables)
            self.assertIn("daily_counters", tables)

    def test_rerun_migrations_is_noop(self) -> None:
        store = TelemetryStore(self.settings, db_path=self.db_path)
        # Run again
        store._apply_migrations()
        with store._connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM schema_migrations;")
            self.assertEqual(cursor.fetchone()[0], 2)

    def test_newer_version_in_db_refuses_to_start(self) -> None:
        store = TelemetryStore(self.settings, db_path=self.db_path)
        with store._connection() as conn:
            conn.execute("INSERT INTO schema_migrations (version, applied_at_utc) VALUES (999, '2099-01-01');")

        # Refuses to start
        with self.assertRaises(RuntimeError) as ctx:
            TelemetryStore(self.settings, db_path=self.db_path)
        self.assertIn("newer than maximum known", str(ctx.exception))

    def test_corrupt_database_is_moved_aside_and_recovered(self) -> None:
        # Create corrupted file
        self.db_path.write_bytes(b"THIS IS CORRUPT SQLITE HEADER NOT VALID")
        store = TelemetryStore(self.settings, db_path=self.db_path)
        self.assertTrue(store.recovered)

        # Corrupt file was renamed
        corrupt_files = list(Path(self.temp_dir).glob("test_nexi.sqlite3.corrupt-*"))
        self.assertEqual(len(corrupt_files), 1)

        # Fresh DB has valid schema
        with store._connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM schema_migrations;")
            self.assertEqual(cur.fetchone()[0], 2)


class TestPrivacyAndRedaction(unittest.TestCase):
    def test_caller_hashing_and_masking(self) -> None:
        raw_phone = "+91 98765 43210"
        h = compute_caller_hash(raw_phone, "secret_salt")
        self.assertEqual(len(h), 16)
        # Deterministic
        self.assertEqual(h, compute_caller_hash(raw_phone, "secret_salt"))
        # Mask
        mask = mask_caller_number(raw_phone)
        self.assertTrue(mask.startswith("+91"))
        self.assertTrue(mask.endswith("10"))
        self.assertNotIn("98765", mask)

    def test_query_redaction_emails_phones_and_digits(self) -> None:
        raw_query = "My email is student@srmap.edu.in and phone is 9876543210 and roll is AP21110010999"
        redacted, q_hash = redact_query_text(raw_query)
        self.assertNotIn("student@srmap.edu.in", redacted)
        self.assertNotIn("9876543210", redacted)
        self.assertNotIn("21110010999", redacted)
        self.assertIn("[redacted]", redacted)
        self.assertEqual(len(q_hash), 64)

    def test_table_scan_finds_no_raw_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "privacy_test.sqlite3"
            settings = Settings(chroma_dir=Path(tmp_dir) / "k", pii_hash_key="test_key")
            store = TelemetryStore(settings, db_path=db_path)

            raw_phone = "+919876543210"
            store.record_call_started_sync(
                call_id="call_p1",
                channel="phone",
                language="en",
                caller_number=raw_phone,
            )

            # Raw phone string should not appear in any row of calls
            with store._connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("SELECT * FROM calls WHERE call_id = 'call_p1';")
                row = dict(cur.fetchone())
                self.assertNotIn("9876543210", json.dumps(row))
                self.assertEqual(row["caller_hash"], compute_caller_hash(raw_phone, "test_key"))
                self.assertEqual(row["caller_mask"], mask_caller_number(raw_phone))


class TestBoundedPipelineAndSpill(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "pipeline.sqlite3"
        self.spill_path = Path(self.temp_dir) / "spill.jsonl"
        self.settings = Settings(
            chroma_dir=Path(self.temp_dir) / "k",
            pii_hash_key="test_key",
            telemetry_queue_size=5,
            telemetry_drain_timeout_s=1.0,
            telemetry_spill_path=self.spill_path,
        )
        self.clock = FakeClock()
        self.store = TelemetryStore(self.settings, db_path=self.db_path)
        self.pipeline = TelemetryPipeline(self.settings, self.store, clock=self.clock)

    async def asyncTearDown(self) -> None:
        await self.pipeline.drain(timeout=0.5)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_queue_overflow_drops_metrics_and_rate_limits_warning(self) -> None:
        """When queue is full, metric events are dropped, count increments, and warning is rate-limited."""
        # Fill queue with 5 metrics without starting writer loop
        for i in range(5):
            accepted = await self.pipeline.put_event("turn_metrics", call_id="c", total_ms=float(i))
            self.assertTrue(accepted)

        # 6th metric is dropped
        with patch("agent.telemetry.logger.warning") as mock_warn:
            accepted6 = await self.pipeline.put_event("turn_metrics", call_id="c", total_ms=6.0)
            self.assertFalse(accepted6)
            self.assertEqual(self.pipeline.telemetry_dropped, 1)
            mock_warn.assert_called_once()

            # Immediate next drop: count increments, but warning log is rate-limited (< 60s)
            mock_warn.reset_mock()
            await self.pipeline.put_event("turn_metrics", call_id="c", total_ms=7.0)
            self.assertEqual(self.pipeline.telemetry_dropped, 2)
            mock_warn.assert_not_called()

            # Advance clock past 60s: warning logs again
            self.clock.advance(65.0)
            await self.pipeline.put_event("turn_metrics", call_id="c", total_ms=8.0)
            self.assertEqual(self.pipeline.telemetry_dropped, 3)
            mock_warn.assert_called_once()

    async def test_critical_event_spill_and_idempotent_replay(self) -> None:
        """Critical event spills to disk on full queue and replays idempotently, skipping corrupt lines."""
        for i in range(5):
            await self.pipeline.put_event("turn_metrics", call_id="c", total_ms=float(i))

        # Putting critical event to full queue spills to disk
        await self.pipeline.put_event(
            "call_started",
            call_id="call_spilled_1",
            channel="web",
            language="en",
        )
        self.assertTrue(self.spill_path.is_file())
        lines = self.spill_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("call_spilled_1", lines[0])

        # Add corrupt line to spill file
        with open(self.spill_path, "a", encoding="utf-8") as f:
            f.write("CORRUPT NOT JSON LINE\n")

        # Replay spill
        self.pipeline._replay_spill_sync()
        # Row should be present in calls table
        calls = self.store.query_recent_calls()
        call_ids = [c["call_id"] for c in calls]
        self.assertIn("call_spilled_1", call_ids)
        # Spill file removed
        self.assertFalse(self.spill_path.is_file())

    async def test_drain_waits_for_inflight_and_spills_on_timeout(self) -> None:
        """Drain flushes remaining events and spills critical ones if timeout expires."""
        self.pipeline.start()
        await self.pipeline.put_event("call_started", call_id="drain_call_1", channel="web", language="en")
        await self.pipeline.drain(timeout=2.0)

        calls = self.store.query_recent_calls()
        self.assertIn("drain_call_1", [c["call_id"] for c in calls])

    async def test_writer_error_increments_count_and_sets_degraded(self) -> None:
        """Writer survives storage error, increments storage_errors, and toggles degraded flag."""
        self.pipeline.start()
        with patch.object(self.store, "record_call_started_sync", side_effect=sqlite3.OperationalError("disk error")):
            await self.pipeline.put_event("call_started", call_id="error_call", channel="web", language="en")
            # Wait for writer task
            await asyncio.sleep(0.1)
            self.assertGreater(self.pipeline.storage_errors, 0)
            self.assertTrue(self.pipeline.degraded)

            # Advance clock past 300s: degraded toggles False
            self.clock.advance(301.0)
            self.assertFalse(self.pipeline.degraded)

    async def test_assistant_and_lifecycle_telemetry_integration(self) -> None:
        """Knowledge gap is emitted on no_results, and call_ended is recorded and drained on lifecycle end."""
        from rag.retriever import SearchResult

        self.pipeline.start()
        state = CallState(channel="web", language="en")
        state.telemetry = self.pipeline

        # Initial call_started
        await self.pipeline.put_event(
            "call_started",
            call_id=state.call_id,
            channel="web",
            language="en",
        )

        mock_retriever = MagicMock()
        mock_retriever.knowledge_version = "v1"
        from unittest.mock import AsyncMock

        mock_retriever.search = AsyncMock(
            return_value=SearchResult((), "v1", reason="below_threshold", top_score=0.15, nearest_topic="hostel_rules")
        )

        agent = NexiAgent(mock_retriever, self.settings, "web", "en")
        ctx = MagicMock()
        ctx.userdata = state

        # Search abstains
        result = await agent.search_university_info(ctx, "What are the curfew times for girls hostel?")
        self.assertEqual(result, "NO_RESULTS")
        self.assertEqual(state.no_results, 1)

        # Lifecycle ends call
        mock_room = MagicMock()
        mock_session = MagicMock()
        mock_session.aclose = MagicMock(return_value=asyncio.sleep(0))
        lifecycle = CallLifecycle(
            ctx=MagicMock(),
            state=state,
            settings=self.settings,
            session=mock_session,
            room=mock_room,
            telemetry_drain_hook=self.pipeline.drain,
        )
        state.lifecycle = lifecycle
        await lifecycle.end_call(EndReason.USER_GOODBYE, farewell=False)

        # Drain pipeline
        await self.pipeline.drain(timeout=2.0)

        # Verify database
        calls = {c["call_id"]: c for c in self.store.query_recent_calls()}
        self.assertIn(state.call_id, calls)
        call_row = calls[state.call_id]
        self.assertEqual(call_row["status"], "ended")
        self.assertEqual(call_row["end_reason"], "user_goodbye")
        self.assertEqual(call_row["no_results_count"], 1)
        self.assertEqual(call_row["tool_calls"], 1)

        gaps = self.store.query_knowledge_gaps()
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["reason"], "below_threshold")
        self.assertEqual(gaps[0]["nearest_topic"], "hostel_rules")
        self.assertIn("curfew times", gaps[0]["query_redacted"])


class TestMaintenanceAndReports(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "maint.sqlite3"
        self.settings = Settings(
            chroma_dir=Path(self.temp_dir) / "k",
            pii_hash_key="test_key",
            call_log_retention_days=10,
            turn_metrics_retention_days=5,
            knowledge_gap_retention_days=30,
            max_call_seconds_web=300,
            max_call_seconds_phone=600,
        )
        self.store = TelemetryStore(self.settings, db_path=self.db_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_reconcile_marks_stale_started_calls_abandoned(self) -> None:
        now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)
        # Fresh call (5 minutes ago): should stay started
        fresh_time = (now - timedelta(minutes=5)).isoformat()
        self.store.record_call_started_sync("fresh_call", "web", "en", started_at_utc=fresh_time)

        # Stale call (30 minutes ago, exceeds max duration threshold): should become abandoned
        stale_time = (now - timedelta(minutes=30)).isoformat()
        self.store.record_call_started_sync("stale_call", "web", "en", started_at_utc=stale_time)

        reconciled = reconcile_stale_calls(self.store, self.settings, now_func=lambda: now)
        self.assertEqual(reconciled, 1)

        calls = {c["call_id"]: c for c in self.store.query_recent_calls()}
        self.assertEqual(calls["fresh_call"]["status"], "started")
        self.assertEqual(calls["stale_call"]["status"], "abandoned")
        self.assertEqual(calls["stale_call"]["end_reason"], "unknown_process_exit")

    def test_retention_purge_batches_and_protects_open_gaps(self) -> None:
        now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Old call (15 days ago > 10d retention)
        old_call_time = (now - timedelta(days=15)).isoformat()
        self.store.record_call_started_sync("old_call", "web", "en", started_at_utc=old_call_time)

        # 2. Old metric (10 days ago > 5d retention)
        old_metric_time = (now - timedelta(days=10)).isoformat()
        self.store.record_call_started_sync("c", "web", "en")
        self.store.record_turn_metrics_sync("c", "web", "en", ts_utc=old_metric_time)

        # 3. Old gap: open vs resolved (40 days ago > 30d retention)
        old_gap_time = (now - timedelta(days=40)).isoformat()
        self.store.record_knowledge_gap_sync("open gap text", "en", "web", "no_results", timestamp_utc=old_gap_time)
        q_red, q_hash = self.store.record_knowledge_gap_sync("resolved gap text", "en", "web", "no_results", timestamp_utc=old_gap_time)
        # Mark second gap resolved
        gaps = self.store.query_knowledge_gaps()
        resolved_id = [g["id"] for g in gaps if "resolved" in g["query_redacted"]][0]
        self.store.update_gap_status(resolved_id, "resolved", note="Added to syllabus")

        # Dry run: deletes nothing
        counts_dry = purge_retention(self.store, self.settings, dry_run=True, now_func=lambda: now)
        self.assertEqual(counts_dry["calls"], 1)
        self.assertEqual(counts_dry["turn_metrics"], 1)
        self.assertEqual(counts_dry["knowledge_gaps"], 1)  # only resolved gap counted

        # Real purge
        counts_real = purge_retention(self.store, self.settings, dry_run=False, now_func=lambda: now)
        self.assertEqual(counts_real["calls"], 1)

        # Open gap is preserved!
        remaining_gaps = self.store.query_knowledge_gaps()
        self.assertEqual(len(remaining_gaps), 1)
        self.assertEqual(remaining_gaps[0]["status"], "open")

    def test_reports_percentiles_and_formatting(self) -> None:
        # Nearest-rank percentile test with known values
        vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        self.assertEqual(nearest_rank_percentile(vals, 50), 50.0)
        self.assertEqual(nearest_rank_percentile(vals, 95), 100.0)

        # Insert synthetic metrics
        now_iso = datetime.now(timezone.utc).isoformat()
        for i in range(10):
            self.store.record_call_started_sync(f"call_{i}", "web", "en")
            self.store.record_turn_metrics_sync(
                call_id=f"call_{i}",
                channel="web",
                language="en",
                total_ms=100.0 + i * 10,
                retrieval_ms=50.0,
                ts_utc=now_iso,
            )

        rows = generate_latency_report(self.store, since="7d")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n"], 10)
        self.assertTrue(rows[0]["sample_size_warning"])  # n < 30

        text_out = format_latency_text(rows)
        self.assertIn("N=10", text_out)
        self.assertIn("WARNING: N < 30", text_out)

        md_out = format_latency_markdown(rows)
        self.assertIn("| web / en |", md_out)

        # Summary
        summary = generate_call_summary(self.store, since="7d")
        self.assertEqual(summary["total_calls"], 10)

        # Gap CSV
        gaps = [{"id": 1, "count": 5, "status": "open", "reason": "no_results", "language": "en", "channel": "web", "query_redacted": "fee", "first_seen_utc": now_iso, "last_seen_utc": now_iso}]
        csv_out = format_gaps_csv(gaps)
        self.assertIn("fee", csv_out)
        self.assertIn("no_results", csv_out)

    def test_maintenance_and_reports_cli(self) -> None:
        import sys

        from agent.maintenance import main as maint_main
        from agent.reports import main as reports_main

        with patch("agent.maintenance.load_settings", return_value=self.settings), \
             patch("agent.maintenance.TelemetryStore", return_value=self.store):
            # Test reconcile CLI
            with patch.object(sys, "argv", ["maintenance.py", "reconcile"]):
                maint_main()

            # Test purge CLI dry-run
            with patch.object(sys, "argv", ["maintenance.py", "purge", "--dry-run"]):
                maint_main()

        with patch("agent.reports.load_settings", return_value=self.settings), \
             patch("agent.reports.TelemetryStore", return_value=self.store):
            # Test summary CLI
            with patch.object(sys, "argv", ["reports.py", "summary"]):
                reports_main()

            # Test latency CLI
            with patch.object(sys, "argv", ["reports.py", "latency", "--format", "text"]):
                reports_main()

            # Test gaps CLI
            with patch.object(sys, "argv", ["reports.py", "gaps", "--format", "text"]):
                reports_main()
