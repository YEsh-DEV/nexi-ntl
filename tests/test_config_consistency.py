"""Unit tests for configuration consistency, lock verification, and dispatch safety."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent.config import ENV_VARS, ConfigurationError, Settings
from scripts.check_lock import validate_lock_content
from scripts.gen_settings_table import generate_markdown_table

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestConfigConsistency(unittest.TestCase):
    def test_agent_name_fail_fast(self) -> None:
        """Empty AGENT_NAME must fail fast unless ALLOW_AUTO_DISPATCH is enabled."""
        s_invalid = Settings(
            agent_name="",
            allow_auto_dispatch=False,
            google_api_key="mock",
            deepgram_api_key="mock",
            sarvam_api_key="mock",
        )
        with self.assertRaises(ConfigurationError) as ctx:
            s_invalid.validate_runtime(console=True)
        self.assertIn("AGENT_NAME cannot be empty", str(ctx.exception))

        s_valid = Settings(
            agent_name="",
            allow_auto_dispatch=True,
            google_api_key="mock",
            deepgram_api_key="mock",
            sarvam_api_key="mock",
        )
        with self.assertLogs("agent.config", level="WARNING") as cm:
            s_valid.validate_runtime(console=True)
        self.assertTrue(any("ALLOW_AUTO_DISPATCH is enabled" in msg for msg in cm.output))

    def test_token_and_telephony_agent_name_match(self) -> None:
        """The dispatch rule template and token service dispatch must match configured agent_name."""
        dispatch_rule_file = REPO_ROOT / "telephony" / "dispatch_rule.json"
        self.assertTrue(dispatch_rule_file.exists())
        rule_data = json.loads(dispatch_rule_file.read_text(encoding="utf-8"))
        rule_agent = rule_data["dispatch"]["agent_name"]

        default_settings = Settings()
        self.assertEqual(rule_agent, default_settings.agent_name)
        self.assertEqual(rule_agent, "nexi")

    def test_all_env_vars_present_in_dotenv_example(self) -> None:
        """Every registered environment variable must be documented in .env.example."""
        dotenv_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        missing = [var for var in ENV_VARS if f"{var}=" not in dotenv_example and f"#{var}=" not in dotenv_example]
        self.assertEqual(
            missing,
            [],
            f"The following variables from agent.config.ENV_VARS are missing in .env.example: {missing}",
        )

    def test_readme_settings_table_consistency(self) -> None:
        """The settings table in README.md must match the generated table from ENV_VARS."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        start_marker = "<!-- SETTINGS_TABLE_START -->"
        end_marker = "<!-- SETTINGS_TABLE_END -->"
        self.assertIn(start_marker, readme, "README.md missing <!-- SETTINGS_TABLE_START -->")
        self.assertIn(end_marker, readme, "README.md missing <!-- SETTINGS_TABLE_END -->")

        start_idx = readme.index(start_marker) + len(start_marker)
        end_idx = readme.index(end_marker)
        table_in_readme = readme[start_idx:end_idx].strip()
        expected_table = generate_markdown_table().strip()
        self.assertEqual(
            table_in_readme,
            expected_table,
            "Settings table in README.md does not match agent.config.ENV_VARS registry output.",
        )

    def test_check_lock_logic_sample_text(self) -> None:
        """check_lock logic must catch nvidia, triton, and cuda torch builds while passing clean text."""
        clean_sample = """
        # --extra-index-url https://download.pytorch.org/whl/cpu
        aiohttp==3.14.5
        chromadb==1.0.20
        torch==2.8.0+cpu
        transformers==4.55.4
        """
        self.assertEqual(validate_lock_content(clean_sample), [])

        dirty_sample = """
        nvidia-cublas-cu12==12.8.4.1
        triton==3.0.0
        torch==2.8.0+cu124
        """
        violations = validate_lock_content(dirty_sample)
        self.assertEqual(len(violations), 3)
        self.assertTrue(any("nvidia-" in v for v in violations))
        self.assertTrue(any("triton" in v for v in violations))
        self.assertTrue(any("CUDA build tag" in v for v in violations))

    def test_smoke_providers_mock_run(self) -> None:
        """smoke_providers.py runs cleanly when providers are skipped."""
        import asyncio

        from scripts.smoke_providers import run_smoke

        code = asyncio.run(run_smoke(skip=["gemini", "sarvam", "deepgram", "livekit"]))
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
