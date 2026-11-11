"""Unit tests for Phase 3/4 multilingual support and voice lifecycle utilities."""

import unittest
from datetime import date

from agent.main import mask_phone_number, read_channel
from agent.prompts import FAREWELLS, GREETINGS, IDLE_PROMPTS, build_prompt


class MultilingualAndLifecycleTests(unittest.TestCase):
    def test_multilingual_prompt_generation(self):
        # English prompt
        en_prompt = build_prompt("web", "Asia/Kolkata", "en", today=date(2026, 11, 12))
        self.assertIn("Reply in English", en_prompt)
        self.assertIn("2026-11-12", en_prompt)
        self.assertIn("80 words", en_prompt)

        # Telugu prompt
        te_prompt = build_prompt("phone", "Asia/Kolkata", "te", today=date(2026, 11, 12))
        self.assertIn("spoken Telugu (తెలుగు)", te_prompt)
        self.assertIn("translate the query to a concise English search phrase", te_prompt)
        self.assertIn("50 words", te_prompt)

        # Hindi prompt
        hi_prompt = build_prompt("web", "Asia/Kolkata", "hi", today=date(2026, 11, 12))
        self.assertIn("spoken Hindi (हिन्दी)", hi_prompt)
        self.assertIn("translate the query to a concise English search phrase", hi_prompt)

        # Invalid parameters
        with self.assertRaises(ValueError):
            build_prompt("invalid_channel", "Asia/Kolkata", "en")  # type: ignore
        with self.assertRaises(ValueError):
            build_prompt("web", "Asia/Kolkata", "es")  # type: ignore

    def test_read_channel_and_language_metadata(self):
        # Normal web and phone
        ch, lang = read_channel('{"channel": "phone", "language": "te"}')
        self.assertEqual(ch, "phone")
        self.assertEqual(lang, "te")

        ch, lang = read_channel('{"channel": "web", "language": "hi"}')
        self.assertEqual(ch, "web")
        self.assertEqual(lang, "hi")

        # Fallbacks for empty, missing, or invalid values
        ch, lang = read_channel("{}")
        self.assertEqual(ch, "web")
        self.assertEqual(lang, "en")

        ch, lang = read_channel('{"channel": "unknown", "language": "fr"}')
        self.assertEqual(ch, "web")
        self.assertEqual(lang, "en")

        # Invalid JSON payloads
        with self.assertRaises(ValueError):
            read_channel("not a json string")
        with self.assertRaises(ValueError):
            read_channel("[1, 2, 3]")

    def test_phone_number_masking(self):
        self.assertEqual(mask_phone_number("+919876543210"), "+91****3210")
        self.assertEqual(mask_phone_number("12345"), "***")
        self.assertEqual(mask_phone_number(""), "***")

    def test_greetings_and_lifecycle_phrases_completeness(self):
        for lang in ("en", "te", "hi"):
            self.assertIn(lang, GREETINGS)
            self.assertIn("web", GREETINGS[lang])
            self.assertIn("phone", GREETINGS[lang])
            self.assertTrue(len(GREETINGS[lang]["web"]) > 5)
            self.assertTrue(len(GREETINGS[lang]["phone"]) > 5)

            self.assertIn(lang, IDLE_PROMPTS)
            self.assertTrue(len(IDLE_PROMPTS[lang]) > 5)

            self.assertIn(lang, FAREWELLS)
            self.assertTrue(len(FAREWELLS[lang]) > 5)


if __name__ == "__main__":
    unittest.main()
