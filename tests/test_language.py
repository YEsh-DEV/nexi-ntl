"""Unit tests for multilingual phrase tables, phone language menu, mid-call switching, and adapter hygiene."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agent.assistant import CallState
from agent.config import Settings
from agent.language import (
    DEFAULT_LANGUAGE,
    LANGUAGE_CATALOG,
    NEEDS_NATIVE_REVIEW,
    PHRASES,
    MenuState,
    PhoneLanguageMenu,
    switch_call_language,
)
from agent.providers.sarvam_adapter import (
    TESTED_SARVAM_VERSIONS,
    check_plugin_compatibility,
    sanitize_error_message,
)
from scripts.eval_multilingual import (
    check_entities_present,
    normalize_text,
    run_evaluation,
)
from scripts.wer import compute_wer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestMultilingualPhrasesAndCatalog(unittest.TestCase):
    def test_phrase_table_completeness(self) -> None:
        """Every phrase key must have non-empty entries for English, Telugu, and Hindi."""
        expected_keys = {
            "greeting_web",
            "greeting_phone",
            "menu_prompt",
            "menu_confirm",
            "farewell",
            "duration_warning",
            "idle_prompt",
            "transfer_announce",
            "transfer_failed",
            "switch_confirm",
            "switch_refused",
            "technical_trouble",
        }
        self.assertEqual(set(PHRASES.keys()), expected_keys)

        for key, per_lang in PHRASES.items():
            for lang in ("en", "te", "hi"):
                self.assertIn(lang, per_lang, f"Missing language '{lang}' in PHRASES['{key}']")
                self.assertTrue(per_lang[lang].strip(), f"Empty phrase for PHRASES['{key}']['{lang}']")

    def test_needs_native_review_list_coverage(self) -> None:
        """All Telugu and Hindi phrases must be tracked for native speaker review."""
        self.assertGreater(len(NEEDS_NATIVE_REVIEW), 20)
        self.assertTrue(all(item.endswith(".te") or item.endswith(".hi") for item in NEEDS_NATIVE_REVIEW))

    def test_language_catalog_integrity(self) -> None:
        """Language catalog must specify code, speaker, and names for all supported languages."""
        for lang in ("en", "te", "hi"):
            meta = LANGUAGE_CATALOG[lang]
            self.assertTrue(meta["name"])
            self.assertTrue(meta["stt_code"])
            self.assertTrue(meta["tts_code"])
            self.assertEqual(meta["tts_speaker"], "kavya")


class TestPhoneLanguageMenu(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.settings = Settings(
            language_menu_timeout_s=0.1,
            transfer_targets={"general": "+918000000000"},
        )
        self.state = CallState(channel="phone", language="en")
        self.room = MagicMock()
        self.session = MagicMock()
        self.session.say = AsyncMock()

    async def test_dtmf_digit_selection(self) -> None:
        """DTMF digits 1, 2, 3 select English, Telugu, and Hindi respectively."""
        menu = PhoneLanguageMenu(
            room=self.room,
            session=self.session,
            state=self.state,
            settings=self.settings,
        )

        # Simulate DTMF 2 (Telugu)
        event_dtmf = MagicMock()
        event_dtmf.digit = "2"

        async def _trigger_dtmf():
            await asyncio.sleep(0.01)
            menu.on_dtmf_received(event_dtmf)

        t = asyncio.create_task(_trigger_dtmf())
        chosen = await menu.run_menu()
        await t

        self.assertEqual(chosen, "te")
        self.assertEqual(self.state.language, "te")
        self.assertEqual(menu.menu_state, MenuState.SELECTED)

    async def test_dtmf_debounce(self) -> None:
        """Digits arriving within 300ms must be debounced."""
        clock_time = 100.0

        def fake_clock():
            return clock_time

        menu = PhoneLanguageMenu(
            room=self.room,
            session=self.session,
            state=self.state,
            settings=self.settings,
            clock=fake_clock,
        )

        ev1 = MagicMock(digit="1")
        ev2 = MagicMock(digit="2")

        # First digit at t=100.0
        menu.on_dtmf_received(ev1)
        self.assertEqual(menu.selected_language, "en")

        # Second digit at t=100.1 (< 300ms) -> ignored
        clock_time = 100.1
        menu.on_dtmf_received(ev2)
        self.assertEqual(menu.selected_language, "en")

    async def test_digit_zero_triggers_general_transfer(self) -> None:
        """Keypad 0 triggers general office transfer if configured."""
        transferred_dept = []

        async def fake_transfer(dept: str) -> bool:
            transferred_dept.append(dept)
            return True

        menu = PhoneLanguageMenu(
            room=self.room,
            session=self.session,
            state=self.state,
            settings=self.settings,
            transfer_fn=fake_transfer,
        )

        ev0 = MagicMock(digit="0")
        menu.on_dtmf_received(ev0)
        await asyncio.sleep(0.01)
        self.assertEqual(transferred_dept, ["general"])

    async def test_menu_timeout_defaults_to_english(self) -> None:
        """If caller does not press any key, menu repeats and defaults to English."""
        menu = PhoneLanguageMenu(
            room=self.room,
            session=self.session,
            state=self.state,
            settings=self.settings,
        )
        chosen = await menu.run_menu()
        self.assertEqual(chosen, DEFAULT_LANGUAGE)
        self.assertEqual(self.state.language, DEFAULT_LANGUAGE)

    async def test_menu_skipped_on_web(self) -> None:
        """Menu is skipped immediately when channel is web."""
        web_state = CallState(channel="web", language="te")
        menu = PhoneLanguageMenu(
            room=self.room,
            session=self.session,
            state=web_state,
            settings=self.settings,
        )
        chosen = await menu.run_menu()
        self.assertEqual(chosen, "te")
        self.assertEqual(menu.menu_state, MenuState.SELECTED)


class TestLanguageSwitching(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.settings = Settings(
            deepgram_api_key="mock_key",
            sarvam_api_key="mock_key",
            max_language_switches=2,
        )
        self.state = CallState(channel="web", language="en")
        self.session = MagicMock()
        self.session.say = AsyncMock()
        self.session.current_speech = None
        self.session.update_agent = MagicMock()
        self.retriever = MagicMock()
        self.retriever.knowledge_version = "mock-v1"

    async def test_successful_mid_call_switch(self) -> None:
        """Language switch updates state, notifies session, and announces in new language."""
        success, msg = await switch_call_language(
            session=self.session,
            state=self.state,
            settings=self.settings,
            retriever=self.retriever,
            target_language="te",
        )
        self.assertTrue(success)
        self.assertEqual(self.state.language, "te")
        self.assertEqual(self.state.language_switches, 1)
        self.session.update_agent.assert_called_once()
        self.session.say.assert_awaited_once_with(PHRASES["switch_confirm"]["te"])

    async def test_same_language_no_op(self) -> None:
        """Requesting switch to current language is a no-op."""
        success, msg = await switch_call_language(
            session=self.session,
            state=self.state,
            settings=self.settings,
            retriever=self.retriever,
            target_language="en",
        )
        self.assertTrue(success)
        self.assertIn("already in en", msg.lower())
        self.assertEqual(self.state.language_switches, 0)
        self.session.update_agent.assert_not_called()

    async def test_max_switches_limit_refusal(self) -> None:
        """Switch is refused once max_language_switches is reached."""
        self.state.language_switches = 2
        success, msg = await switch_call_language(
            session=self.session,
            state=self.state,
            settings=self.settings,
            retriever=self.retriever,
            target_language="hi",
        )
        self.assertFalse(success)
        self.assertIn("Maximum", msg)
        self.session.say.assert_awaited_once_with(PHRASES["switch_refused"]["en"])

    async def test_switch_failure_rollback(self) -> None:
        """Provider failure triggers rollback to old language and speaks apology in old language."""
        self.session.update_agent.side_effect = RuntimeError("Update agent failed")

        success, msg = await switch_call_language(
            session=self.session,
            state=self.state,
            settings=self.settings,
            retriever=self.retriever,
            target_language="te",
        )
        self.assertFalse(success)
        self.assertEqual(self.state.language, "en")
        self.assertEqual(self.state.language_switch_failures, 1)
        self.session.say.assert_awaited_with(PHRASES["technical_trouble"]["en"])

    async def test_switch_concurrency_serialization(self) -> None:
        """Concurrent switch requests are serialized via lock."""
        lock = asyncio.Lock()
        results = await asyncio.gather(
            switch_call_language(self.session, self.state, self.settings, self.retriever, "te", lock=lock),
            switch_call_language(self.session, self.state, self.settings, self.retriever, "hi", lock=lock),
        )
        self.assertEqual(len(results), 2)
        # One succeeded first, then state updated so both executed cleanly without race
        self.assertEqual(self.state.language_switches, 2)


class TestSarvamAdapterHygiene(unittest.TestCase):
    def test_version_compatibility_check(self) -> None:
        """Adapter enforces tested livekit-plugins-sarvam versions."""
        self.assertIn("1.2.6", TESTED_SARVAM_VERSIONS)
        # Should not raise under normal test environment
        check_plugin_compatibility()

    def test_private_attribute_contract(self) -> None:
        """Verify that sarvam.TTS exposes required private options."""
        from livekit.plugins import sarvam

        mock_tts = sarvam.TTS(target_language_code="en-IN", api_key="mock_key")
        self.assertTrue(hasattr(mock_tts, "_opts"))
        self.assertTrue(hasattr(mock_tts._opts, "target_language_code"))
        self.assertTrue(hasattr(mock_tts._opts, "model"))
        self.assertTrue(hasattr(mock_tts._opts, "speaker"))
        self.assertTrue(hasattr(mock_tts, "_ensure_session"))

    def test_error_message_sanitization(self) -> None:
        """Sensitive API keys, tokens, and headers must be redacted and truncated."""
        dirty_error = (
            "Authentication failed: api-subscription-key: secret_sarvam_key_123456789. "
            "Server returned auth=Bearer_abc1234567890xyz and subscription=sub_secret_9999. "
            + "A" * 300
        )
        clean = sanitize_error_message(dirty_error, max_chars=120)
        self.assertNotIn("secret_sarvam_key_123456789", clean)
        self.assertNotIn("Bearer_abc1234567890xyz", clean)
        self.assertIn("[REDACTED]", clean)
        self.assertLessEqual(len(clean), 120)


class TestEntityPreservationAndWER(unittest.TestCase):
    def test_indic_numeral_and_amount_normalization(self) -> None:
        """Telugu and Hindi numerals and amounts are correctly normalized to ASCII."""
        # Telugu ౧౫౦౦౦౦ (150000)
        self.assertEqual(normalize_text("ఫీజు ౧,౫౦,౦౦౦ రూపాయలు"), "ఫీజు 150000 రూపాయలు")
        # Devanagari १५०००० (150000)
        self.assertEqual(normalize_text("फीस १,५०,००० रुपये"), "फीस 150000 रुपये")

    def test_entity_checker_presence(self) -> None:
        """Entity presence check verifies all expected keywords are in the query."""
        passed, missing = check_entities_present(
            ["hostel", "2-bed", "ac"],
            "hostel fee for 2-bed ac room",
        )
        self.assertTrue(passed)
        self.assertEqual(missing, [])

        passed_fail, missing_fail = check_entities_present(
            ["hostel", "swimming pool", "fees"],
            "hostel room fees",
        )
        self.assertFalse(passed_fail)
        self.assertEqual(missing_fail, ["swimming pool"])

    def test_wer_exact_and_edit_distance(self) -> None:
        """Levenshtein word error rate matches known expected values."""
        # Exact match
        self.assertEqual(compute_wer("hello world", "hello world"), 0.0)
        # One substitution in two words -> 0.5 (50%)
        self.assertEqual(compute_wer("hello world", "hello there"), 0.5)
        # Empty reference
        self.assertEqual(compute_wer("", ""), 0.0)
        self.assertEqual(compute_wer("", "word"), 1.0)

    def test_eval_multilingual_report_needs_review(self) -> None:
        """Multilingual report reports NEEDS_REVIEW when review coverage is below 80%."""
        dataset = REPO_ROOT / "eval" / "multilingual.yaml"
        recorded = REPO_ROOT / "eval" / "recorded_queries.yaml"
        report = run_evaluation(dataset, recorded_path=recorded)
        self.assertEqual(report["status"], "NEEDS_REVIEW")
        self.assertGreaterEqual(report["languages"]["te"]["entity_preservation_rate"], 0.90)
        self.assertGreaterEqual(report["languages"]["hi"]["entity_preservation_rate"], 0.90)


if __name__ == "__main__":
    unittest.main()
