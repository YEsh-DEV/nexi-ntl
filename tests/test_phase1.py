"""Offline regression checks saved for later. Not executed during implementation.

Run after installing dependencies: python -m unittest discover -s tests
"""

import asyncio
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch

from agent.config import ConfigurationError, Settings, load_settings
from agent.prompts import build_prompt
from rag.ingest import json_records
from rag.manifest import COLLECTION, SCHEMA_VERSION, model_fingerprint, read_active
from rag.retriever import RetrievalBusy, Retriever, SearchResult


class PhaseOneChecks(unittest.TestCase):
    def test_configuration_and_prompt_boundaries(self):
        with patch.dict(
            "os.environ", {"NEXI_ENV": "production", "SEARCH_TIMEOUT": "nan"}, clear=True
        ):
            with self.assertRaises(ConfigurationError):
                load_settings()
        with self.assertRaises(ConfigurationError):
            Settings().validate_runtime(console=True)
        settings = Settings(
            deepgram_api_key="private", google_api_key="private", sarvam_api_key="private"
        )
        settings.validate_runtime(console=True)
        self.assertNotIn("private", repr(settings))
        with self.assertRaises(ConfigurationError):
            settings.validate_runtime(console=False)
        prompt = build_prompt("phone", "Asia/Kolkata", today=date(2026, 11, 11))
        self.assertIn("2026-11-11", prompt)
        self.assertIn("50 words", prompt)
        self.assertIn("NO_RESULTS", prompt)

    def test_missing_and_incompatible_index_never_creates_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "knowledge"
            settings = replace(Settings(), chroma_dir=root)
            with self.assertRaises(RuntimeError):
                read_active(settings)
            self.assertFalse(root.exists())
            version = "a" * 32
            artifact = root / version
            artifact.mkdir(parents=True)
            (root / "CURRENT").write_text(version)
            (artifact / "chroma.sqlite3").touch()
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "version": version,
                "collection": COLLECTION,
                "chunks": 1,
                "embedding": model_fingerprint(settings),
                "quality_status": "passed",
                "partial": False,
            }
            (artifact / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(read_active(settings)[0], artifact)
            with self.assertRaises(RuntimeError):
                read_active(replace(settings, embed_revision="different"))
            (root / "CURRENT").write_text("../outside")
            with self.assertRaises(RuntimeError):
                read_active(settings)

    def test_bootstrap_keeps_nested_records_and_zero_values(self):
        data = {
            "name": "CSE",
            "faculty": [
                {"name": "Example", "email": "x@example.invalid", "fee": 0, "active": False}
            ],
        }
        records = list(json_records(data))
        self.assertEqual(len(records), 2)
        self.assertIn("faculty", records[1][0])
        self.assertIn("x@example.invalid", records[1][1])
        self.assertIn('"fee": 0', records[1][1])
        self.assertIn('"active": false', records[1][1])

    def test_timeout_does_not_free_running_inference_slot(self):
        async def scenario():
            retriever = Retriever.__new__(Retriever)  # No model loading or provider calls.
            retriever.settings = replace(Settings(), search_timeout=0.01)
            retriever._executor = ThreadPoolExecutor(max_workers=1)
            retriever._pending = None
            retriever._closed = False
            release = threading.Event()

            def slow_search(query):
                release.wait()
                return SearchResult((), "test")

            retriever._search_sync = slow_search
            try:
                with self.assertRaises(TimeoutError):
                    await retriever.search("hostel")
                with self.assertRaises(RetrievalBusy):
                    await retriever.search("fees")
            finally:
                release.set()
                if retriever._pending is not None:
                    await asyncio.wrap_future(retriever._pending)
                retriever.close()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
