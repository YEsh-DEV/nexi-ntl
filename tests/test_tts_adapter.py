"""Unit tests for model-aware Sarvam TTS adapter."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from livekit.agents import APIConnectOptions

from agent.tts_adapter import ModelAwareChunkedStream, NexiSarvamTTS


class TestSarvamTTSAdapter(unittest.TestCase):
    def test_bulbul_v3_omits_unsupported_parameters(self):
        async def run():
            tts = NexiSarvamTTS(
                target_language_code="te-IN",
                model="bulbul:v3",
                speaker="kavya",
                api_key="test-api-key",
                speech_sample_rate=24000,
            )

            captured_payload = {}

            # Create mock response
            mock_res = AsyncMock()
            mock_res.status = 200
            mock_res.json = AsyncMock(
                return_value={"request_id": "req-1", "audios": ["AAAA"]}
            )

            # Mock aiohttp session
            mock_session = MagicMock()

            class MockPostContext:
                async def __aenter__(self):
                    return mock_res

                async def __aexit__(self, exc_type, exc, tb):
                    pass

            def mock_post(url, json=None, headers=None, timeout=None):
                captured_payload.update(json)
                return MockPostContext()

            mock_session.post = mock_post
            tts._session = mock_session

            stream = ModelAwareChunkedStream(
                tts=tts,
                input_text="నమస్కారం",
                conn_options=APIConnectOptions(timeout=5.0),
            )

            mock_emitter = MagicMock()
            await stream._run(mock_emitter)

            # Assert v3 payload contains required fields
            self.assertEqual(captured_payload["model"], "bulbul:v3")
            self.assertEqual(captured_payload["speaker"], "kavya")
            self.assertEqual(captured_payload["target_language_code"], "te-IN")
            self.assertEqual(captured_payload["text"], "నమస్కారం")
            self.assertEqual(captured_payload["speech_sample_rate"], 24000)

            # Assert v3 explicitly excludes pitch, loudness, and enable_preprocessing
            self.assertNotIn("pitch", captured_payload)
            self.assertNotIn("loudness", captured_payload)
            self.assertNotIn("enable_preprocessing", captured_payload)

        asyncio.run(run())

    def test_bulbul_v2_retains_parameters(self):
        async def run():
            tts = NexiSarvamTTS(
                target_language_code="en-IN",
                model="bulbul:v2",
                speaker="anushka",
                api_key="test-api-key",
                speech_sample_rate=22050,
            )

            captured_payload = {}
            mock_res = AsyncMock()
            mock_res.status = 200
            mock_res.json = AsyncMock(
                return_value={"request_id": "req-2", "audios": ["AAAA"]}
            )

            mock_session = MagicMock()

            class MockPostContext:
                async def __aenter__(self):
                    return mock_res

                async def __aexit__(self, exc_type, exc, tb):
                    pass

            def mock_post(url, json=None, headers=None, timeout=None):
                captured_payload.update(json)
                return MockPostContext()

            mock_session.post = mock_post
            tts._session = mock_session

            stream = ModelAwareChunkedStream(
                tts=tts,
                input_text="Hello",
                conn_options=APIConnectOptions(timeout=5.0),
            )

            mock_emitter = MagicMock()
            await stream._run(mock_emitter)

            self.assertEqual(captured_payload["model"], "bulbul:v2")
            self.assertIn("pitch", captured_payload)
            self.assertIn("loudness", captured_payload)
            self.assertIn("enable_preprocessing", captured_payload)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
