"""Unit tests for token generation service and health endpoints."""

import json
import unittest
from datetime import datetime, timezone

import jwt

from agent.config import Settings
from agent.token_service import (
    check_health,
    generate_token,
)


class TokenServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            livekit_url="wss://test.livekit.cloud",
            livekit_api_key="API12345678",
            livekit_api_secret="SECRET12345678abcdef0123456789abcdef",
            agent_name="nexi",
        )

    def test_token_generation_defaults(self):
        res = generate_token(self.settings, channel="web", language="en")
        self.assertEqual(res["channel"], "web")
        self.assertEqual(res["language"], "en")
        self.assertEqual(res["url"], "wss://test.livekit.cloud")
        self.assertTrue(str(res["room"]).startswith("nexi-web-"))
        self.assertTrue(str(res["identity"]).startswith("student-"))
        self.assertTrue(len(str(res["token"])) > 100)

        expires_dt = datetime.fromisoformat(str(res["expires_at"]))
        self.assertGreater(expires_dt, datetime.now(timezone.utc))

        # Decode JWT without verification to verify claims and restrictions
        decoded = jwt.decode(str(res["token"]), options={"verify_signature": False})
        self.assertEqual(decoded["sub"], res["identity"])
        self.assertEqual(decoded["video"]["room"], res["room"])
        self.assertTrue(decoded["video"]["roomJoin"])
        self.assertTrue(decoded["video"]["canPublish"])
        self.assertTrue(decoded["video"]["canSubscribe"])
        # Published sources must be restricted to MICROPHONE (value 2)
        self.assertEqual(decoded["video"]["canPublishSources"], [2])
        # Room agent dispatch must target nexi
        agents = decoded["roomConfig"]["agents"]
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["agentName"], "nexi")
        self.assertEqual(
            json.loads(agents[0]["metadata"]),
            {"channel": "web", "language": "en"},
        )

    def test_multilingual_tokens(self):
        for lang in ("te", "hi"):
            res = generate_token(self.settings, channel="web", language=lang)
            self.assertEqual(res["language"], lang)
            decoded = jwt.decode(str(res["token"]), options={"verify_signature": False})
            agents = decoded["roomConfig"]["agents"]
            meta = json.loads(agents[0]["metadata"])
            self.assertEqual(meta["language"], lang)

    def test_missing_credentials_fails(self):
        empty_settings = Settings(
            livekit_url="",
            livekit_api_key="",
            livekit_api_secret="",
            agent_name="nexi",
        )
        with self.assertRaises(ValueError):
            generate_token(empty_settings, channel="web", language="en")

    def test_token_generation_parameter_validation(self):
        with self.assertRaises(ValueError):
            generate_token(self.settings, channel="invalid_channel")  # type: ignore
        with self.assertRaises(ValueError):
            generate_token(self.settings, language="invalid_lang")  # type: ignore

    def test_health_check_endpoint(self):
        health = check_health(self.settings)
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["chunks"], 817)
        self.assertTrue(len(str(health["knowledge_version"])) > 10)
        self.assertTrue(health["livekit_configured"])
        self.assertEqual(health["agent_name"], "nexi")

    def test_web_index_file_exists(self):
        from agent.token_service import ROOT

        index_file = ROOT / "web" / "index.html"
        self.assertTrue(index_file.exists())
        self.assertIn("Nexi Voice Assistant", index_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
