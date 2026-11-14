"""Unit tests for WP7: runbook commands existence, maintenance loop, capacity reporting, and lock checks."""

from __future__ import annotations

import csv
import importlib
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.config import Settings
from agent.maintenance import run_loop
from scripts.capacity_report import compute_capacity_summary, format_text_report, read_samples_csv
from scripts.check_lock import validate_lock_content

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestRunbookCommandExistence(unittest.TestCase):
    def test_runbook_commands_exist(self) -> None:
        """Verify that every python module or script referenced in docs/runbooks/*.md actually exists."""
        runbooks_dir = REPO_ROOT / "docs" / "runbooks"
        self.assertTrue(runbooks_dir.is_dir(), "docs/runbooks directory must exist")

        md_files = list(runbooks_dir.glob("*.md"))
        self.assertGreaterEqual(len(md_files), 10, "Must have at least 10 runbooks in docs/runbooks")

        # Regex patterns for command references
        py_module_pattern = re.compile(r"python\s+-m\s+([a-zA-Z0-9_\.]+)")
        script_pattern = re.compile(r"python\s+(scripts/[a-zA-Z0-9_\.]+\.py)")

        modules_checked = set()
        scripts_checked = set()

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")

            # Check python -m <module>
            for match in py_module_pattern.finditer(content):
                mod_name = match.group(1).strip()
                modules_checked.add(mod_name)
                # Attempt importing top-level or sub-module
                try:
                    importlib.import_module(mod_name)
                except ImportError as exc:
                    self.fail(f"Runbook {md_file.name} references non-existent python module '{mod_name}': {exc}")

            # Check python scripts/<script.py>
            for match in script_pattern.finditer(content):
                script_rel = match.group(1).strip()
                scripts_checked.add(script_rel)
                script_path = REPO_ROOT / script_rel
                self.assertTrue(
                    script_path.is_file(),
                    f"Runbook {md_file.name} references non-existent script file '{script_rel}'",
                )

        self.assertIn("agent.main", modules_checked)
        self.assertIn("agent.token_service", modules_checked)
        self.assertIn("agent.maintenance", modules_checked)
        self.assertIn("scripts/capacity_sample.py", scripts_checked)
        self.assertIn("scripts/capacity_report.py", scripts_checked)


class TestMaintenanceLoop(unittest.TestCase):
    def test_run_loop_scheduling(self) -> None:
        """Verify that maintenance loop executes reconcile and purge on schedule with fake clock."""
        storage = MagicMock()
        settings = Settings()

        current_time = 0.0

        def fake_clock() -> float:
            return current_time

        def fake_sleep(duration: float) -> None:
            nonlocal current_time
            current_time += duration

        reconcile_mock = MagicMock(return_value=0)
        purge_mock = MagicMock(return_value={"calls": 0, "turn_metrics": 0, "knowledge_gaps": 0})

        with patch("agent.maintenance.reconcile_stale_calls", reconcile_mock), \
             patch("agent.maintenance.purge_retention", purge_mock):
            # Run loop for 5 iterations
            # Iteration 1 at t=0: runs reconcile & purge, sleeps 300s
            # Iteration 2 at t=300: runs reconcile, sleeps 300s
            # Iteration 3 at t=600: runs reconcile, sleeps 300s
            run_loop(
                storage=storage,
                settings=settings,
                reconcile_interval_s=300.0,
                purge_interval_s=86400.0,
                clock_fn=fake_clock,
                sleep_fn=fake_sleep,
                max_iterations=4,
            )

            # Reconcile called at t=0, t=300, t=600, t=900 -> 4 times
            self.assertEqual(reconcile_mock.call_count, 4)
            # Purge called at t=0 -> 1 time (next purge at 86400)
            self.assertEqual(purge_mock.call_count, 1)


class TestCapacityReportFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "samples.csv"

        # Generate synthetic CSV samples fixture
        rows = []
        # 10 idle samples (active_rooms=0, rss ~150MB, cpu ~2%)
        for i in range(10):
            rows.append({
                "timestamp_utc": f"2026-11-14T01:00:{i:02d}Z",
                "pid": 1234,
                "cpu_pct": 2.0,
                "rss_mb": 150.0,
                "active_rooms": 0,
            })
        # 5 samples with 1 active call (active_rooms=1, rss ~220MB, cpu ~35%)
        for i in range(5):
            rows.append({
                "timestamp_utc": f"2026-11-14T01:05:{i:02d}Z",
                "pid": 1234,
                "cpu_pct": 35.0,
                "rss_mb": 220.0,
                "active_rooms": 1,
            })
        # 35 samples with 5 active calls (active_rooms=5, rss ~450MB, cpu ~80%)
        for i in range(35):
            rows.append({
                "timestamp_utc": f"2026-11-14T01:10:{i:02d}Z",
                "pid": 1234,
                "cpu_pct": 80.0,
                "rss_mb": 450.0,
                "active_rooms": 5,
            })

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "pid", "cpu_pct", "rss_mb", "active_rooms"])
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_capacity_report_calculations(self) -> None:
        samples = read_samples_csv(self.csv_path)
        self.assertEqual(len(samples), 50)

        summary = compute_capacity_summary(samples)
        self.assertAlmostEqual(summary["idle_rss_mb"], 150.0)

        concurrency = summary["concurrency_levels"]
        self.assertIn(0, concurrency)
        self.assertIn(1, concurrency)
        self.assertIn(5, concurrency)

        # Concurrency 1: peak 220, est mb/call = (220 - 150) / 1 = 70.0
        c1 = concurrency[1]
        self.assertEqual(c1["sample_count"], 5)
        self.assertIsNotNone(c1["warning"])  # n < 30 warning
        self.assertEqual(c1["peak_rss_mb"], 220.0)
        self.assertAlmostEqual(c1["estimated_mb_per_call"], 70.0)
        self.assertEqual(c1["cpu_p95"], 35.0)

        # Concurrency 5: peak 450, est mb/call = (450 - 150) / 5 = 60.0
        c5 = concurrency[5]
        self.assertEqual(c5["sample_count"], 35)
        self.assertIsNone(c5["warning"])  # n >= 30, no warning
        self.assertEqual(c5["peak_rss_mb"], 450.0)
        self.assertAlmostEqual(c5["estimated_mb_per_call"], 60.0)
        self.assertEqual(c5["cpu_p95"], 80.0)

        # Text output formatting
        text_out = format_text_report(summary)
        self.assertIn("Nexi Capacity & Resource Report", text_out)
        self.assertIn("Baseline Idle Memory: 150.00 MB", text_out)
        self.assertIn("Sample size N < 30", text_out)


class TestLockfileValidator(unittest.TestCase):
    def test_clean_lock_passes(self) -> None:
        clean = "numpy==1.26.4\ntorch==2.8.0+cpu\nrequests==2.31.0\n"
        violations = validate_lock_content(clean)
        self.assertEqual(violations, [])

    def test_forbidden_packages_detected(self) -> None:
        dirty = "nvidia-cublas-cu12==12.1.3.1\ntriton==3.0.0\ntorch==2.8.0+cu121\n"
        violations = validate_lock_content(dirty)
        self.assertEqual(len(violations), 3)
        self.assertTrue(any("nvidia-" in v for v in violations))
        self.assertTrue(any("triton" in v for v in violations))
        self.assertTrue(any("CUDA build tag" in v for v in violations))


class TestDockerComposeValidation(unittest.TestCase):
    def test_compose_config_valid(self) -> None:
        import subprocess

        docker_path = shutil.which("docker")
        if not docker_path:
            self.skipTest("Docker binary not available in local environment")

        res = subprocess.run(
            ["docker", "compose", "config"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(
            res.returncode,
            0,
            f"docker compose config failed: {res.stderr}",
        )
        self.assertIn("services:", res.stdout)
        self.assertIn("agent:", res.stdout)
        self.assertIn("token:", res.stdout)
        self.assertIn("maintenance:", res.stdout)


if __name__ == "__main__":
    unittest.main()
