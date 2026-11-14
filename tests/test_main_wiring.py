"""Unit tests for agent.main entrypoint, helper functions, and session wiring."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from livekit import rtc
from livekit.agents import ToolError

from agent.assistant import (
    CallState,
    NexiAgent,
    _build_transfer_tool,
    build_providers,
    connect_participant,
    record_call_started,
)
from agent.config import Settings
from agent.main import (
    entrypoint,
    mask_phone_number,
    pick_noise_cancellation,
    prewarm,
    read_channel,
)


class TestMainHelpers(unittest.TestCase):
    def test_read_channel_valid_and_fallbacks(self) -> None:
        # Valid web / en
        ch, lang = read_channel('{"channel": "web", "language": "en"}')
        self.assertEqual(ch, "web")
        self.assertEqual(lang, "en")

        # Valid phone / te
        ch2, lang2 = read_channel('{"channel": "phone", "language": "te"}')
        self.assertEqual(ch2, "phone")
        self.assertEqual(lang2, "te")

        # Invalid channels/languages fallback
        ch3, lang3 = read_channel('{"channel": "unknown", "language": "fr"}')
        self.assertEqual(ch3, "web")
        self.assertEqual(lang3, "en")

        # Non-JSON or invalid payload
        with self.assertRaises(ValueError):
            read_channel("not valid json")

        with self.assertRaises(ValueError):
            read_channel("[1, 2, 3]")

        with self.assertRaises(ValueError):
            read_channel("x" * 5000)

    def test_mask_phone_number(self) -> None:
        self.assertEqual(mask_phone_number("+919876543210"), "+91****3210")
        self.assertEqual(mask_phone_number("123"), "***")
        self.assertEqual(mask_phone_number("+1-555-0199"), "+15****0199")

    def test_pick_noise_cancellation(self) -> None:
        bvc_sip = pick_noise_cancellation(rtc.ParticipantKind.PARTICIPANT_KIND_SIP)
        self.assertIsNotNone(bvc_sip)
        bvc_web = pick_noise_cancellation(rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD)
        self.assertIsNotNone(bvc_web)

    def test_prewarm_exception_closes_retriever(self) -> None:
        proc = MagicMock()
        proc.userdata = {}
        mock_retriever = MagicMock()
        with patch("agent.main.load_settings") as mock_settings, \
             patch("agent.main.Retriever", return_value=mock_retriever), \
             patch("agent.main.TelemetryStore", side_effect=RuntimeError("Storage disk failure")):
            mock_settings.return_value = Settings()
            with self.assertRaises(RuntimeError):
                prewarm(proc)
            mock_retriever.close.assert_called_once()


class TestEntrypointWiring(unittest.IsolatedAsyncioTestCase):
    async def test_entrypoint_console_mode(self) -> None:
        settings = Settings(
            livekit_url="wss://test.livekit.cloud",
            livekit_api_key="API123",
            livekit_api_secret="SECRET123",
            google_api_key="MockGoogle",
            deepgram_api_key="MockDeepgram",
            sarvam_api_key="MockSarvam",
        )
        mock_retriever = MagicMock()
        mock_retriever.knowledge_version = "337ce61b42e1426896ba9cce2beaba4a"
        mock_retriever.search = AsyncMock()
        mock_telemetry = MagicMock()
        mock_telemetry.put_event = AsyncMock()

        ctx = MagicMock()
        ctx.proc.userdata = {
            "settings": settings,
            "retriever": mock_retriever,
            "vad": MagicMock(),
            "storage": MagicMock(),
            "telemetry": mock_telemetry,
        }
        ctx.job.metadata = json.dumps({"channel": "web", "language": "en"})
        ctx.room.name = "test-room"

        with patch("agent.main.cli_state.CLI_ARGUMENTS", MagicMock(console=True)), \
             patch("agent.main.build_providers", return_value=(MagicMock(), MagicMock(), MagicMock())), \
             patch("agent.main.silero.VAD.load"), \
             patch("agent.main.MultilingualModel"), \
             patch("agent.main.AgentSession") as mock_session_cls, \
             patch("agent.main.CallLifecycle") as mock_lifecycle_cls:
            mock_session = MagicMock()
            mock_session.start = AsyncMock()
            mock_session.say = AsyncMock()
            mock_session_cls.return_value = mock_session
            mock_lifecycle = MagicMock()
            mock_lifecycle.run_duration_watchdog = AsyncMock()
            mock_lifecycle.run_inactivity_watchdog = AsyncMock()
            mock_lifecycle_cls.return_value = mock_lifecycle

            # Run entrypoint in console mode (does not block on real participant connect)
            task = asyncio.create_task(entrypoint(ctx))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            mock_session.start.assert_called_once()

    async def test_entrypoint_phone_language_menu(self) -> None:
        settings = Settings(
            livekit_url="wss://test.livekit.cloud",
            livekit_api_key="API123",
            livekit_api_secret="SECRET123",
            google_api_key="MockGoogle",
            deepgram_api_key="MockDeepgram",
            sarvam_api_key="MockSarvam",
        )
        mock_retriever = MagicMock()
        mock_retriever.knowledge_version = "v1"
        mock_retriever.search = AsyncMock()
        mock_telemetry = MagicMock()
        mock_telemetry.put_event = AsyncMock()

        ctx = MagicMock()
        ctx.proc.userdata = {
            "settings": settings,
            "retriever": mock_retriever,
            "vad": MagicMock(),
            "storage": MagicMock(),
            "telemetry": mock_telemetry,
        }
        ctx.job.metadata = json.dumps({"channel": "phone"})
        ctx.room.name = "test-phone-room"

        with patch("agent.main.cli_state.CLI_ARGUMENTS", MagicMock(console=False)), \
             patch("agent.main.connect_participant", AsyncMock(return_value=("sip_part_1", "+919876543210"))), \
             patch("agent.main.build_providers", return_value=(MagicMock(), MagicMock(), MagicMock())), \
             patch("agent.main.silero.VAD.load"), \
             patch("agent.main.MultilingualModel"), \
             patch("agent.main.AgentSession") as mock_session_cls, \
             patch("agent.main.CallLifecycle") as mock_lifecycle_cls, \
             patch("agent.language.PhoneLanguageMenu") as mock_menu_cls, \
             patch("agent.language.switch_call_language", AsyncMock(return_value=(True, "Switched"))):
            mock_session = MagicMock()
            mock_session.start = AsyncMock()
            mock_session.say = AsyncMock()
            mock_session_cls.return_value = mock_session
            mock_lifecycle = MagicMock()
            mock_lifecycle_cls.return_value = mock_lifecycle

            mock_menu_instance = MagicMock()
            mock_menu_instance.run_menu = AsyncMock(return_value="te")
            mock_menu_cls.return_value = mock_menu_instance

            task = asyncio.create_task(entrypoint(ctx))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            mock_menu_cls.assert_called_once()
            mock_menu_instance.run_menu.assert_called_once()


class TestAssistantHelpers(unittest.IsolatedAsyncioTestCase):
    def test_build_providers(self) -> None:
        settings = Settings(
            livekit_url="wss://test.livekit.cloud",
            livekit_api_key="API123",
            livekit_api_secret="SECRET123",
            google_api_key="MockGoogle",
            deepgram_api_key="MockDeepgram",
            sarvam_api_key="MockSarvam",
        )
        with patch("livekit.plugins.deepgram.STT"), \
             patch("livekit.plugins.google.LLM"), \
             patch("agent.tts_adapter.NexiSarvamTTS"):
            stt_en, llm_en, tts_en = build_providers(settings, "en", "web")
            self.assertIsNotNone(stt_en)
            stt_te, llm_te, tts_te = build_providers(settings, "te", "phone")
            self.assertIsNotNone(stt_te)
            stt_hi, llm_hi, tts_hi = build_providers(settings, "hi", "web")
            self.assertIsNotNone(stt_hi)

    async def test_connect_participant_console(self) -> None:
        ctx = MagicMock()
        state = CallState(channel="web")
        room_opts = MagicMock()
        pid, phone = await connect_participant(ctx, Settings(), state, room_opts, console=True)
        self.assertEqual(pid, "")
        self.assertEqual(phone, "")

    async def test_connect_participant_sip(self) -> None:
        ctx = MagicMock()
        ctx.connect = AsyncMock()
        part = MagicMock()
        part.kind = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        part.identity = "sip_user_1"
        part.attributes = {"sip.phoneNumber": "+919876543210"}
        ctx.wait_for_participant = AsyncMock(return_value=part)

        state = CallState(channel="web")
        room_opts = MagicMock()
        settings = Settings(noise_cancellation=True)

        pid, phone = await connect_participant(ctx, settings, state, room_opts, console=False)
        self.assertEqual(pid, "sip_user_1")
        self.assertEqual(phone, "+919876543210")
        self.assertEqual(state.channel, "phone")
        self.assertEqual(state.caller_mask, "+91****3210")
        self.assertIsNotNone(room_opts.noise_cancellation)

    async def test_connect_participant_web(self) -> None:
        ctx = MagicMock()
        ctx.connect = AsyncMock()
        part = MagicMock()
        part.kind = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
        part.identity = "web_user_1"
        part.attributes = {}
        ctx.wait_for_participant = AsyncMock(return_value=part)

        state = CallState(channel="phone")
        room_opts = MagicMock()
        settings = Settings(noise_cancellation=True)

        pid, phone = await connect_participant(ctx, settings, state, room_opts, console=False)
        self.assertEqual(pid, "web_user_1")
        self.assertEqual(phone, "")
        self.assertEqual(state.channel, "web")
        self.assertIsNotNone(room_opts.noise_cancellation)

    async def test_record_call_started(self) -> None:
        state = CallState(channel="web")
        # None telemetry returns safely
        record_call_started(None, state, MagicMock(), "")

        # Valid telemetry calls put_event
        mock_telemetry = MagicMock()
        mock_telemetry.put_event = AsyncMock()
        ctx = MagicMock()
        ctx.room.name = "test-room-hash"
        ctx.job.id = "job-1"

        record_call_started(mock_telemetry, state, ctx, "+919876543210")
        await asyncio.sleep(0.01)
        self.assertEqual(state.channel, "web")
        mock_telemetry.put_event.assert_called_once()

    async def test_transfer_tool(self) -> None:
        targets = {"admissions": "+918632343001", "accounts": "+918632343002"}
        tool = _build_transfer_tool(targets)

        # Unrecognized dept
        ctx = MagicMock()
        ctx.userdata = CallState(channel="phone")
        res1 = await tool(ctx, "sports")
        self.assertIn("not recognized", res1)

        # Unattached lifecycle
        res2 = await tool(ctx, "admissions")
        self.assertIn("unattached", res2)

        # Attached lifecycle success
        mock_lifecycle = MagicMock()
        mock_lifecycle.transfer_to_office = AsyncMock(return_value=True)
        ctx.userdata.lifecycle = mock_lifecycle
        res3 = await tool(ctx, "admissions")
        self.assertIn("Call transferred successfully", res3)

        # Attached lifecycle failure
        mock_lifecycle.transfer_to_office = AsyncMock(return_value=False)
        res4 = await tool(ctx, "admissions")
        self.assertIn("could not be completed", res4)

    async def test_nexi_agent_tools(self) -> None:
        settings = Settings(query_max_chars=100)
        retriever = MagicMock()
        retriever.knowledge_version = "v1"
        agent = NexiAgent(retriever, settings, channel="phone", language="en")

        ctx = MagicMock()
        state = CallState(channel="phone", language="en")
        ctx.userdata = state

        # 1. Search invalid / empty
        with self.assertRaises(ToolError):
            await agent.search_university_info(ctx, "")
        with self.assertRaises(ToolError):
            await agent.search_university_info(ctx, "x" * 150)

        # 2. Search success
        retriever.search = AsyncMock(return_value=MagicMock(evidence=True, context="Admissions open."))
        res_found = await agent.search_university_info(ctx, "admission")
        self.assertEqual(res_found, "Admissions open.")

        # 3. Search error
        retriever.search = AsyncMock(side_effect=RuntimeError("Database failed"))
        with self.assertRaises(ToolError):
            await agent.search_university_info(ctx, "admission")
        self.assertEqual(state.errors, 1)

        # 4. End call < 5 seconds
        state.started_monotonic = 100.0
        with patch("agent.assistant.monotonic", return_value=102.0):
            msg = await agent.end_call(ctx)
            self.assertIn("Call started recently", msg)

        # 5. End call > 5 seconds
        mock_lifecycle = MagicMock()
        mock_lifecycle.track_task = MagicMock()
        mock_lifecycle.end_call = AsyncMock()
        state.lifecycle = mock_lifecycle

        with patch("agent.assistant.monotonic", return_value=120.0):
            msg2 = await agent.end_call(ctx)
            self.assertIn("Call ending scheduled", msg2)
            self.assertTrue(state.explicit_exit)

            # Subsequent call
            msg3 = await agent.end_call(ctx)
            self.assertIn("Call termination already scheduled", msg3)

        # 6. Switch language
        res_inv = await agent.switch_language(ctx, "french")
        self.assertIn("Unsupported language", res_inv)

        res_same = await agent.switch_language(ctx, "en")
        self.assertIn("Conversation is already in en", res_same)

        with patch("agent.language.switch_call_language", AsyncMock(return_value=(True, "Switched to Telugu"))):
            res_switch = await agent.switch_language(ctx, "te")
            self.assertEqual(res_switch, "Switched to Telugu")


if __name__ == "__main__":
    unittest.main()
