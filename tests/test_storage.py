"""Unit tests for Phase 6 SQLite durable telemetry store."""

import tempfile
import unittest
from pathlib import Path
from time import monotonic

from agent.config import Settings
from agent.storage import TelemetryStore


class StorageTests(unittest.TestCase):
    def test_database_initialization_and_call_logging(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            settings = Settings(chroma_dir=tmp / "knowledge")
            store = TelemetryStore(settings)

            self.assertTrue(store.db_path.exists())

            # Record call sync
            t0 = monotonic()
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

            # Query recent calls
            recent = store.query_recent_calls(limit=5)
            self.assertEqual(len(recent), 1)
            row = recent[0]
            self.assertEqual(row["call_id"], "call-test-123")
            self.assertEqual(row["channel"], "phone")
            self.assertEqual(row["language"], "te")
            self.assertEqual(row["turns"], 3)
            self.assertEqual(row["duration_seconds"], 15.0)


if __name__ == "__main__":
    unittest.main()
