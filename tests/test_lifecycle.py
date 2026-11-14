"""Unit tests for CallLifecycle, duration limiter, explicit end_call, and office transfer."""

from __future__ import annotations

import asyncio
import re
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent.assistant import CallState, NexiAgent
from agent.config import Settings
from agent.lifecycle import CallLifecycle, EndReason

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.current_time = start

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds

    async def sleep(self, seconds: float) -> None:
        # Cooperative yield: test controls virtual time progression via advance()
        await asyncio.sleep(0)


class FakeContext:
    def __init__(self) -> None:
        self.shutdown_reasons: list[str] = []

    def shutdown(self, reason: str = "") -> None:
        self.shutdown_reasons.append(reason)


class FakeSession:
    def __init__(self, aclose_hang: bool = False) -> None:
        self.spoken: list[str] = []
        self.closed: bool = False
        self.aclose_hang = aclose_hang

    async def say(self, text: str) -> None:
        self.spoken.append(text)

    async def aclose(self) -> None:
        if self.aclose_hang:
            await asyncio.sleep(999.0)
        self.closed = True


class TestCallLifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.clock = FakeClock(100.0)
        self.settings = Settings(
            livekit_url="wss://test.livekit.cloud",
            livekit_api_key="test_key",
            livekit_api_secret="test_secret",
            farewell_timeout_s=0.2,
            session_close_timeout_s=0.2,
            room_delete_timeout_s=0.2,
            telemetry_drain_timeout_s=0.2,
            end_call_total_timeout_s=1.0,
            max_call_seconds_web=60.0,
            max_call_seconds_phone=120.0,
            call_warning_lead_s=10.0,
            transfer_grace_s=5.0,
            transfer_targets={"admissions": "+918000000001", "accounts": "+918000000002"},
        )
        self.ctx = FakeContext()
        self.state = CallState(channel="web", language="en")
        self.session = FakeSession()

    async def test_concurrent_end_call_idempotence(self) -> None:
        """Concurrent end_call calls must execute teardown only once, and first reason wins."""
        lifecycle = CallLifecycle(
            ctx=self.ctx,
            state=self.state,
            settings=self.settings,
            session=self.session,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
        )

        results = await asyncio.gather(
            lifecycle.end_call(EndReason.USER_GOODBYE, farewell=True),
            lifecycle.end_call(EndReason.IDLE_TIMEOUT, farewell=True),
            lifecycle.end_call(EndReason.DURATION_LIMIT, farewell=True),
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(lifecycle.end_reason, EndReason.USER_GOODBYE)
        self.assertEqual(self.ctx.shutdown_reasons, ["user_goodbye"])
        self.assertTrue(self.session.closed)
        self.assertEqual(len(self.session.spoken), 1)

    async def test_end_reasons_and_room_deletion_rules(self) -> None:
        """Phone channel attempts room deletion; web channel skips room deletion."""
        # 1. Web channel
        lifecycle_web = CallLifecycle(
            ctx=self.ctx,
            state=self.state,
            settings=self.settings,
            session=self.session,
            room_name="web-room-123",
            clock=self.clock,
            sleep_fn=self.clock.sleep,
        )
        await lifecycle_web.end_call(EndReason.USER_GOODBYE, farewell=True)
        self.assertEqual(lifecycle_web.step_outcomes.get("room_delete"), "skipped")

        # 2. Phone channel with successful room delete mock
        phone_state = CallState(channel="phone", language="en")
        phone_ctx = FakeContext()
        phone_session = FakeSession()

        with patch("livekit.api.LiveKitAPI") as mock_lk:
            mock_client = AsyncMock()
            mock_lk.return_value = mock_client
            mock_client.room.delete_room = AsyncMock()
            mock_client.aclose = AsyncMock()

            lifecycle_phone = CallLifecycle(
                ctx=phone_ctx,
                state=phone_state,
                settings=self.settings,
                session=phone_session,
                room_name="phone-room-123",
                clock=self.clock,
                sleep_fn=self.clock.sleep,
            )
            await lifecycle_phone.end_call(EndReason.USER_GOODBYE, farewell=True)
            self.assertEqual(lifecycle_phone.step_outcomes.get("room_delete"), "ok")
            mock_client.room.delete_room.assert_awaited_once()

    async def test_room_delete_exception_logged_and_counted(self) -> None:
        """Room deletion exception is counted in room_delete_failures and not swallowed."""
        phone_state = CallState(channel="phone", language="en")
        phone_ctx = FakeContext()
        phone_session = FakeSession()

        with patch("livekit.api.LiveKitAPI") as mock_lk:
            mock_client = AsyncMock()
            mock_lk.return_value = mock_client
            mock_client.room.delete_room.side_effect = RuntimeError("API down")
            mock_client.aclose = AsyncMock()

            lifecycle = CallLifecycle(
                ctx=phone_ctx,
                state=phone_state,
                settings=self.settings,
                session=phone_session,
                room_name="phone-room-123",
                clock=self.clock,
                sleep_fn=self.clock.sleep,
            )
            with self.assertLogs("agent.lifecycle", level="WARNING") as cm:
                await lifecycle.end_call(EndReason.IDLE_TIMEOUT, farewell=True)

            self.assertEqual(phone_state.room_delete_failures, 1)
            self.assertIn("error_RuntimeError", lifecycle.step_outcomes.get("room_delete", ""))
            self.assertTrue(any("Delete room failed" in msg for msg in cm.output))
            # Shutdown still completes
            self.assertEqual(phone_ctx.shutdown_reasons, ["idle_timeout"])

    async def test_hanging_step_timeout_fail_forward(self) -> None:
        """A hanging session close step hits its timeout and subsequent steps still run."""
        hanging_session = FakeSession(aclose_hang=True)
        lifecycle = CallLifecycle(
            ctx=self.ctx,
            state=self.state,
            settings=self.settings,
            session=hanging_session,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
        )
        await lifecycle.end_call(EndReason.FATAL_PROVIDER_ERROR, farewell=False)
        self.assertEqual(lifecycle.step_outcomes.get("session_close"), "timeout")
        self.assertEqual(self.ctx.shutdown_reasons, ["fatal_provider_error"])

    async def test_duration_limiter_warning_and_expiration(self) -> None:
        """Duration limiter emits one warning at limit-lead and terminates at limit."""
        self.state.started_monotonic = self.clock()
        lifecycle = CallLifecycle(
            ctx=self.ctx,
            state=self.state,
            settings=self.settings,
            session=self.session,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
        )
        task = lifecycle.start_duration_limiter()

        # Advance to 49s (< 50s lead time) -> no warning yet
        self.clock.advance(49.0)
        await asyncio.sleep(0.01)
        self.assertEqual(len(self.session.spoken), 0)

        # Advance past 50s (60s limit - 10s lead) -> warning spoken
        self.clock.advance(2.0)
        await asyncio.sleep(0.01)
        self.assertEqual(len(self.session.spoken), 1)

        # Advance to limit 60s -> duration limit reached
        self.clock.advance(10.0)
        await asyncio.sleep(0.01)
        await task

        self.assertEqual(lifecycle.end_reason, EndReason.DURATION_LIMIT)
        self.assertEqual(self.ctx.shutdown_reasons, ["duration_limit"])

    async def test_explicit_end_call_tool(self) -> None:
        """end_call tool ignores within first 5s, executes goodbye after 5s."""
        import time

        agent = NexiAgent(
            retriever=MagicMock(),
            settings=self.settings,
            channel="web",
            language="en",
        )
        lifecycle = CallLifecycle(
            ctx=self.ctx,
            state=self.state,
            settings=self.settings,
            session=self.session,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
        )
        self.state.lifecycle = lifecycle
        context = MagicMock()
        context.userdata = self.state

        # Within 5s: call ignored
        self.state.started_monotonic = time.monotonic()
        res_early = await agent.end_call(context)
        self.assertIn("Call started recently", res_early)
        self.assertFalse(self.state.explicit_exit)

        # Advance past 5s: triggers farewell and scheduled end_call
        self.state.started_monotonic = time.monotonic() - 10.0
        res_ok = await agent.end_call(context)
        self.assertIn("Call ending scheduled", res_ok)
        self.assertTrue(self.state.explicit_exit)

        # Let the scheduled background task complete
        await asyncio.sleep(0.05)
        self.assertEqual(lifecycle.end_reason, EndReason.USER_GOODBYE)

    async def test_office_transfer_flow_and_guards(self) -> None:
        """Transfer succeeds on valid phone department; fails on invalid or error."""
        # 1. Web channel: transfer tool is NOT registered on NexiAgent
        web_agent = NexiAgent(
            retriever=MagicMock(),
            settings=self.settings,
            channel="web",
            language="en",
        )
        tool_names_web = [getattr(t, "__name__", str(t)) for t in (web_agent.tools or [])]
        self.assertNotIn("transfer_to_office", tool_names_web)

        # 2. Phone channel with valid targets: transfer tool is registered
        phone_agent = NexiAgent(
            retriever=MagicMock(),
            settings=self.settings,
            channel="phone",
            language="en",
        )
        tool_names_phone = [getattr(t, "__name__", str(t)) for t in (phone_agent.tools or [])]
        self.assertIn("transfer_to_office", tool_names_phone)

        # Find the transfer tool function
        transfer_tool = next(t for t in phone_agent.tools if getattr(t, "__name__", "") == "transfer_to_office")

        phone_state = CallState(channel="phone", language="en")
        phone_ctx = FakeContext()
        phone_session = FakeSession()

        with patch("livekit.api.LiveKitAPI") as mock_lk:
            mock_client = AsyncMock()
            mock_lk.return_value = mock_client
            mock_client.sip.transfer_sip_participant = AsyncMock()
            mock_client.aclose = AsyncMock()

            lifecycle = CallLifecycle(
                ctx=phone_ctx,
                state=phone_state,
                settings=self.settings,
                session=phone_session,
                participant_identity="caller_123",
                room_name="phone-room-123",
                clock=self.clock,
                sleep_fn=self.clock.sleep,
            )
            phone_state.lifecycle = lifecycle
            ctx_run = MagicMock()
            ctx_run.userdata = phone_state

            # Unknown department is rejected
            res_unknown = await transfer_tool(ctx_run, department="astronomy")
            self.assertIn("not recognized", res_unknown)

            # Valid department succeeds
            res_valid = await transfer_tool(ctx_run, department="admissions")
            self.assertIn("transferred successfully", res_valid)
            self.assertTrue(phone_state.transfer_attempted)
            self.assertTrue(phone_state.transfer_succeeded)
            self.assertEqual(lifecycle.end_reason, EndReason.TRANSFERRED)

    def test_source_scan_no_outside_teardown_calls(self) -> None:
        """Scan agent source files to ensure no file outside lifecycle.py initiates teardown."""
        agent_dir = REPO_ROOT / "agent"
        teardown_patterns = [
            re.compile(r"\bctx\.room\.delete_room\b"),
            re.compile(r"\blivekit_api.*delete_room\b"),
            re.compile(r"\bctx\.shutdown\b"),
            re.compile(r"\bsession\.aclose\b"),
        ]

        violations = []
        for py_file in agent_dir.glob("*.py"):
            if py_file.name == "lifecycle.py":
                continue
            content = py_file.read_text(encoding="utf-8")
            for pattern in teardown_patterns:
                matches = pattern.findall(content)
                if matches:
                    violations.append(f"{py_file.name} matches forbidden teardown: {matches}")

        self.assertEqual(
            violations,
            [],
            f"Files outside lifecycle.py must not initiate direct teardown: {violations}",
        )


if __name__ == "__main__":
    unittest.main()
