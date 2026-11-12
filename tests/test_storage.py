"""Unit tests for Phase 6 SQLite durable telemetry store."""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from agent.config import Settings
from agent.storage import TelemetryStore


class StorageTests(unittest.TestCase):
    def test_database_initialization_and_call_logging(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            settings = Settings(chroma_dir=tmp / "knowledge")
            store = TelemetryStore(settings)

            self.assertTrue(store.db_path.exists())

            # Record call sync with real epoch timestamp
            t0 = time.time()
            store.record_call_sync(
                call_id="call-test-123",
                started_at=t0 - 15.0,
                duration=15.0,
                channel="phone",
                language="te",
                status="closed",
                turns=3,
                tool_calls=2,
                no_results=0,
                errors=0,
                timeouts=0,
                knowledge_version="ver-test",
                prompt_version="phase4-v1",
                usage={"llm_tokens": 150, "tts_characters": 80},
            )

            # Record stage metrics
            store.record_stage_metrics_sync(
                call_id="call-test-123",
                channel="phone",
                language="te",
                stage="llm_inference",
                measurements={"ttft": 0.42, "total_duration": 0.85},
            )

            # Record knowledge gap
            store.record_knowledge_gap_sync(
                call_id="call-test-123",
                language="te",
                knowledge_version="ver-test",
            )

            # Query recent calls
            recent = store.query_recent_calls(limit=5)
            self.assertEqual(len(recent), 1)
            row = recent[0]
            self.assertEqual(row["call_id"], "call-test-123")
            self.assertEqual(row["channel"], "phone")
            self.assertEqual(row["language"], "te")
            self.assertEqual(row["turns"], 3)
            self.assertEqual(row["duration_seconds"], 15.0)
            # Ensure timestamp is year 2026+ and not epoch 1970
            self.assertTrue(row["started_at"].startswith("202"))
            self.assertTrue(row["ended_at"].startswith("202"))

            # Query knowledge gaps
            gaps = store.query_knowledge_gaps(limit=5)
            self.assertEqual(len(gaps), 1)
            self.assertEqual(gaps[0]["call_id"], "call-test-123")
            self.assertEqual(gaps[0]["language"], "te")
            self.assertTrue(gaps[0]["recorded_at"].startswith("202"))

    def test_monotonic_timestamp_defensive_handling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            settings = Settings(chroma_dir=tmp / "knowledge")
            store = TelemetryStore(settings)

            # Passing monotonic time (e.g. 1234.5 seconds since boot)
            store.record_call_sync(
                call_id="call-mono-1",
                started_at=1234.5,
                duration=10.0,
                channel="web",
                language="en",
                status="closed",
                turns=1,
                tool_calls=1,
                no_results=0,
                errors=0,
                timeouts=0,
                knowledge_version="ver-test",
                prompt_version="phase4-v1",
                usage={},
            )

            recent = store.query_recent_calls(limit=1)
            self.assertEqual(len(recent), 1)
            # Defensive check should convert to plausible current year date, NOT 1970
            self.assertTrue(recent[0]["started_at"].startswith("202"))

    def test_async_recording(self):
        async def run_async_test():
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                settings = Settings(chroma_dir=tmp / "knowledge")
                store = TelemetryStore(settings)

                await store.record_call(
                    call_id="call-async-1",
                    started_at=time.time() - 5.0,
                    duration=5.0,
                    channel="web",
                    language="hi",
                    status="closed",
                    turns=2,
                    tool_calls=1,
                    no_results=1,
                    errors=0,
                    timeouts=0,
                    knowledge_version="ver-test",
                    prompt_version="phase4-v1",
                    usage={},
                )
                await store.record_stage_metrics(
                    call_id="call-async-1",
                    channel="web",
                    language="hi",
                    stage="tts",
                    measurements={"ttfb": 0.22},
                )
                await store.record_knowledge_gap(
                    call_id="call-async-1",
                    language="hi",
                    knowledge_version="ver-test",
                )

                calls = store.query_recent_calls()
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["call_id"], "call-async-1")
                gaps = store.query_knowledge_gaps()
                self.assertEqual(len(gaps), 1)

        asyncio.run(run_async_test())


if __name__ == "__main__":
    unittest.main()
