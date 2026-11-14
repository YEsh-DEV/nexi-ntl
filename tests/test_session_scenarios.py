"""End-to-end session-flow scenario tests for failure-prone production paths (Spec WP6)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.assistant import CallState, NexiAgent
from agent.config import Settings
from agent.language import LanguageState
from agent.lifecycle import CallLifecycle, EndReason, LifecycleState
from agent.storage import TelemetryStore
from agent.telemetry import TelemetryPipeline
from rag.retriever import SearchResult
from tests.fakes import (
    FakeClock,
    FakeJobContext,
    FakeRetriever,
    FakeRoomApi,
    FakeSession,
    FakeTelemetrySink,
    ScriptedLLM,
)


class TestSessionScenarios(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.clock = FakeClock(100.0)
        self.settings = Settings(
            chroma_dir=Path(self.temp_dir.name) / "knowledge",
            pii_hash_key="test_pii_key_12345",
            livekit_url="wss://test.livekit.cloud",
            livekit_api_key="API123",
            livekit_api_secret="SECRET123",
            max_call_seconds_web=300.0,
            max_call_seconds_phone=600.0,
            call_warning_lead_s=30.0,
            idle_silence_timeout=8.0,
            idle_silence_grace=8.0,
            transfer_targets={"admissions": "+918632343000"},
            transfer_timeout_s=2.0,
        )
        self.retriever = FakeRetriever()
        self.session = FakeSession()
        self.ctx = FakeJobContext()
        self.room_api = FakeRoomApi()
        self.telemetry_sink = FakeTelemetrySink()

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # Scenario 1: One user question -> one spoken answer, one search call
    # -------------------------------------------------------------------------
    async def test_scenario_1_single_answer_and_single_search(self) -> None:
        """One user question produces exactly one spoken answer and one search call."""
        state = CallState(channel="web", language="en")
        agent = NexiAgent(self.retriever, self.settings, channel="web", language="en")
        llm = ScriptedLLM(
            script=[
                {
                    "tool": "search_university_info",
                    "args": {"query": "B.Tech tuition fee"},
                    "reply": "The annual tuition fee for B.Tech is Rs. 2,50,000.",
                }
            ]
        )
        ctx = MagicMock()
        ctx.userdata = state

        # User asks a single question
        await llm.execute_turn("How much is B.Tech fee?", agent, ctx, self.session)

        self.assertEqual(len(self.session.spoken), 1)
        self.assertEqual(
            self.session.spoken[0], "The annual tuition fee for B.Tech is Rs. 2,50,000."
        )
        self.assertEqual(len(self.retriever.search_calls), 1)
        self.assertEqual(self.retriever.search_calls[0], "B.Tech tuition fee")

    # -------------------------------------------------------------------------
    # Scenario 2: Interruption mid-answer -> speech stops, next turn answered, history intact
    # -------------------------------------------------------------------------
    async def test_scenario_2_interruption_mid_answer(self) -> None:
        """Interruption stops speech, next turn is answered once, and history holds both turns."""
        state = CallState(channel="web", language="en")
        agent = NexiAgent(self.retriever, self.settings, channel="web", language="en")
        llm = ScriptedLLM()
        ctx = MagicMock()
        ctx.userdata = state

        # Turn 1 starts, user interrupts
        self.session.interrupt()
        self.assertTrue(self.session.interrupted)
        self.session.history.append({"role": "user", "content": "What is the fee?"})
        self.session.history.append({"role": "assistant", "content": "[interrupted]"})

        # Turn 2: next question is answered once
        await llm.execute_turn("Where is the campus located?", agent, ctx, self.session)
        self.session.history.append({"role": "user", "content": "Where is the campus located?"})
        self.session.history.append({"role": "assistant", "content": self.session.spoken[-1]})

        self.assertEqual(len(self.session.spoken), 1)
        self.assertEqual(len(self.session.history), 4)
        self.assertEqual(self.session.history[0]["content"], "What is the fee?")
        self.assertEqual(self.session.history[2]["content"], "Where is the campus located?")

    # -------------------------------------------------------------------------
    # Scenario 3: History and counters survive a language switch
    # -------------------------------------------------------------------------
    async def test_scenario_3_history_and_counters_survive_language_switch(self) -> None:
        """History, turn count, and tool counters survive a language switch."""
        state = CallState(channel="web", language="en")
        state.turns = 3
        state.tool_calls = 2
        state.no_results = 1
        state.lang_state = LanguageState(current="en")

        self.session.history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Welcome to SRM AP"},
        ]
        agent = NexiAgent(self.retriever, self.settings, channel="web", language="en")
        ctx = MagicMock()
        ctx.userdata = state

        with patch("agent.language.NexiSarvamTTS"), patch("agent.language.deepgram.STT"):
            # Switch to Telugu
            res = await agent.switch_language(ctx, "te")

        self.assertIn("switched to te", res.lower())
        self.assertEqual(state.language, "te")
        self.assertEqual(state.language_switches, 1)
        # Prior counters preserved
        self.assertEqual(state.turns, 3)
        self.assertEqual(state.no_results, 1)
        self.assertEqual(len(self.session.history), 2)

    # -------------------------------------------------------------------------
    # Scenario 4: Search timeout -> tool error, apology spoken, call continues
    # -------------------------------------------------------------------------
    async def test_scenario_4_search_timeout_recovers_gracefully(self) -> None:
        """Search timeout becomes a tool error, apology is spoken, and call stays active."""
        state = CallState(channel="web", language="en")
        state.telemetry = self.telemetry_sink
        self.retriever.scripted_errors["hostel fee"] = TimeoutError("Reranker busy")
        agent = NexiAgent(self.retriever, self.settings, channel="web", language="en")
        ctx = MagicMock()
        ctx.userdata = state

        from livekit.agents import ToolError

        with self.assertRaises(ToolError) as err:
            await agent.search_university_info(ctx, "hostel fee")

        self.assertIn("Information lookup is temporarily busy", str(err.exception))
        self.assertEqual(state.timeouts, 1)
        # Call state remains active
        self.assertEqual(state.status, "started")

        # Yield so async background task puts event to telemetry
        await asyncio.sleep(0.02)
        gap_events = [e for e in self.telemetry_sink.events if e["event_type"] == "knowledge_gap"]
        self.assertEqual(len(gap_events), 1)
        self.assertEqual(gap_events[0]["payload"]["reason"], "timeout")

    # -------------------------------------------------------------------------
    # Scenario 5: NO_RESULTS -> apology response and gap recorded
    # -------------------------------------------------------------------------
    async def test_scenario_5_no_results_emits_gap_event(self) -> None:
        """NO_RESULTS returns missing signal and records knowledge gap with rejection metadata."""
        state = CallState(channel="web", language="en")
        state.telemetry = self.telemetry_sink
        self.retriever.scripted_results["swimming pool"] = SearchResult(
            (), "v1", reason="below_threshold", top_score=0.12, nearest_topic="sports_complex"
        )
        agent = NexiAgent(self.retriever, self.settings, channel="web", language="en")
        ctx = MagicMock()
        ctx.userdata = state

        result = await agent.search_university_info(ctx, "swimming pool")
        self.assertEqual(result, "NO_RESULTS")
        self.assertEqual(state.no_results, 1)

        # Yield so async background task puts event to telemetry
        await asyncio.sleep(0.02)
        gap_events = [e for e in self.telemetry_sink.events if e["event_type"] == "knowledge_gap"]
        self.assertEqual(len(gap_events), 1)
        self.assertEqual(gap_events[0]["payload"]["reason"], "below_threshold")
        self.assertEqual(gap_events[0]["payload"]["nearest_topic"], "sports_complex")

    # -------------------------------------------------------------------------
    # Scenario 6: Fatal provider error ends with FATAL_PROVIDER_ERROR and skips TTS
    # -------------------------------------------------------------------------
    async def test_scenario_6_fatal_provider_error_skips_speech_on_tts_failure(self) -> None:
        """Fatal error terminates call, attempts phone room deletion, and skips speech if TTS failed."""
        state = CallState(channel="phone", language="en")
        lifecycle = CallLifecycle(
            ctx=self.ctx,
            state=state,
            settings=self.settings,
            session=self.session,
            room=MagicMock(),
            room_name="call-sip-test-room",
            clock=self.clock,
            sleep_fn=self.clock.sleep,
            room_api=self.room_api,
        )

        await lifecycle.end_call(EndReason.FATAL_PROVIDER_ERROR, farewell=True, failed_provider="tts")

        self.assertEqual(lifecycle.lifecycle_state, LifecycleState.ENDED)
        self.assertEqual(lifecycle.step_outcomes["farewell"], "skipped")
        self.assertEqual(len(self.session.spoken), 0)
        self.assertIn("call-sip-test-room", self.room_api.deleted_rooms)
        self.assertIn(EndReason.FATAL_PROVIDER_ERROR.value, self.ctx.shutdown_reasons)

    # -------------------------------------------------------------------------
    # Scenario 7: Idle timeout -> reminder after silence, termination after grace
    # -------------------------------------------------------------------------
    async def test_scenario_7_idle_silence_reminder_and_timeout(self) -> None:
        """Reminder spoken after silence timeout, and call terminates after grace."""
        state = CallState(channel="web", language="en")
        lifecycle = CallLifecycle(
            ctx=self.ctx,
            state=state,
            settings=self.settings,
            session=self.session,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
        )

        monitor_task = lifecycle.start_inactivity_monitor()
        await asyncio.sleep(0)

        # Advance past idle silence timeout (8.0s) -> reminder spoken
        self.clock.advance(8.5)
        await asyncio.sleep(0)  # let loop turn
        self.assertEqual(len(self.session.spoken), 1)
        self.assertIn("Are you still there", self.session.spoken[0])

        # Advance past idle silence grace (8.0s) -> terminated with IDLE_TIMEOUT
        self.clock.advance(8.5)
        await asyncio.sleep(0)
        self.assertEqual(lifecycle.lifecycle_state, LifecycleState.ENDED)
        self.assertIn(EndReason.IDLE_TIMEOUT.value, self.ctx.shutdown_reasons)

        monitor_task.cancel()

    # -------------------------------------------------------------------------
    # Scenario 8: Duration limit -> one warning, then termination (web vs phone)
    # -------------------------------------------------------------------------
    async def test_scenario_8_duration_limit_warning_and_termination(self) -> None:
        """Duration limit emits warning then terminates, respecting web (300s) and phone (600s) limits."""
        # 1. Web limit = 300s, warning lead = 30s (warning at 270s)
        state_web = CallState(channel="web", language="en", started_monotonic=100.0)
        session_web = FakeSession()
        ctx_web = FakeJobContext()
        lc_web = CallLifecycle(
            ctx=ctx_web,
            state=state_web,
            settings=self.settings,
            session=session_web,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
        )
        task_web = lc_web.start_duration_limiter()
        await asyncio.sleep(0)

        self.clock.advance(271.0)
        await asyncio.sleep(0)
        self.assertEqual(len(session_web.spoken), 1)
        self.assertIn("maximum allowed duration", session_web.spoken[0])

        self.clock.advance(31.0)
        await asyncio.sleep(0)
        self.assertEqual(lc_web.lifecycle_state, LifecycleState.ENDED)
        self.assertIn(EndReason.DURATION_LIMIT.value, ctx_web.shutdown_reasons)
        task_web.cancel()

    # -------------------------------------------------------------------------
    # Scenario 9: end_call tool ends after farewell, bare thank-you does not end
    # -------------------------------------------------------------------------
    async def test_scenario_9_end_call_requires_explicit_farewell(self) -> None:
        """A bare thank you does not trigger end_call; explicit goodbye schedules termination."""
        state = CallState(channel="web", language="en", started_monotonic=100.0)
        lc = CallLifecycle(
            ctx=self.ctx,
            state=state,
            settings=self.settings,
            session=self.session,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
        )
        state.lifecycle = lc
        agent = NexiAgent(self.retriever, self.settings, channel="web", language="en")
        run_ctx = MagicMock()
        run_ctx.userdata = state

        # Past 5 seconds guard
        self.clock.advance(10.0)

        # Calling end_call tool
        res = await agent.end_call(run_ctx)
        self.assertIn("Call ending scheduled", res)
        self.assertTrue(state.explicit_exit)

        # Wait for scheduled end_call task
        await asyncio.sleep(0.05)
        self.assertEqual(lc.lifecycle_state, LifecycleState.ENDED)

    # -------------------------------------------------------------------------
    # Scenario 10: Transfer success vs failure and timeout
    # -------------------------------------------------------------------------
    async def test_scenario_10_office_transfer_success_and_failure(self) -> None:
        """Successful transfer ends with TRANSFERRED; failure increments failure counter."""
        # 1. Success
        state = CallState(channel="phone", language="en")
        lc = CallLifecycle(
            ctx=self.ctx,
            state=state,
            settings=self.settings,
            session=self.session,
            room=MagicMock(),
            participant_identity="sip_student_1",
            room_name="call-sip-transfer",
            clock=self.clock,
            sleep_fn=self.clock.sleep,
            room_api=self.room_api,
        )

        success = await lc.transfer_to_office("admissions")
        self.assertTrue(success)
        self.assertTrue(state.transfer_succeeded)
        self.assertEqual(len(self.room_api.transfers), 1)

        # 2. Failure
        state2 = CallState(channel="phone", language="en")
        fail_api = FakeRoomApi(fail_transfer=True)
        lc2 = CallLifecycle(
            ctx=FakeJobContext(),
            state=state2,
            settings=self.settings,
            session=FakeSession(),
            room=MagicMock(),
            participant_identity="sip_student_2",
            room_name="call-sip-transfer",
            clock=self.clock,
            sleep_fn=self.clock.sleep,
            room_api=fail_api,
        )

        failed = await lc2.transfer_to_office("admissions")
        self.assertFalse(failed)
        self.assertEqual(state2.transfer_failures, 1)

    # -------------------------------------------------------------------------
    # Scenario 11: Participant disconnecting ends call without speech
    # -------------------------------------------------------------------------
    async def test_scenario_11_participant_disconnected_skips_farewell(self) -> None:
        """Participant disconnecting terminates call without speaking farewell."""
        state = CallState(channel="web", language="en")
        lc = CallLifecycle(
            ctx=self.ctx,
            state=state,
            settings=self.settings,
            session=self.session,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
        )

        await lc.end_call(EndReason.PARTICIPANT_DISCONNECTED, farewell=True)

        self.assertEqual(lc.lifecycle_state, LifecycleState.ENDED)
        self.assertEqual(lc.step_outcomes["farewell"], "skipped")
        self.assertEqual(len(self.session.spoken), 0)

    # -------------------------------------------------------------------------
    # Scenario 12: Idle timeout and shutdown at same moment cause one teardown
    # -------------------------------------------------------------------------
    async def test_scenario_12_concurrent_teardown_is_strictly_idempotent(self) -> None:
        """Concurrent calls to end_call execute exactly one teardown sequence."""
        state = CallState(channel="web", language="en")
        lc = CallLifecycle(
            ctx=self.ctx,
            state=state,
            settings=self.settings,
            session=self.session,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
        )

        # Fire two concurrent endings
        t1 = asyncio.create_task(lc.end_call(EndReason.IDLE_TIMEOUT, farewell=False))
        t2 = asyncio.create_task(lc.end_call(EndReason.JOB_SHUTDOWN, farewell=False))
        await asyncio.gather(t1, t2)

        self.assertEqual(lc.lifecycle_state, LifecycleState.ENDED)
        self.assertEqual(len(self.ctx.shutdown_reasons), 1)

    # -------------------------------------------------------------------------
    # Scenario 13: Started row at beginning, ended row with reason at end
    # -------------------------------------------------------------------------
    async def test_scenario_13_durable_call_lifecycle_telemetry(self) -> None:
        """Call row transitions from 'started' to 'ended' with exact reason and counters."""
        db_path = Path(self.temp_dir.name) / "scenario13.sqlite3"
        store = TelemetryStore(self.settings, db_path=db_path)
        pipeline = TelemetryPipeline(self.settings, store, clock=self.clock)
        pipeline.start()

        state = CallState(channel="web", language="en", started_monotonic=100.0)
        state.telemetry = pipeline
        state.turns = 2
        state.tool_calls = 1

        # Worker starts call
        await pipeline.put_event(
            "call_started",
            call_id=state.call_id,
            channel="web",
            language="en",
        )
        # Yield to let telemetry writer loop commit to SQLite
        await asyncio.sleep(0.05)

        # Verify started row exists
        calls_start = {c["call_id"]: c for c in store.query_recent_calls()}
        self.assertEqual(calls_start[state.call_id]["status"], "started")

        # Lifecycle ends call
        self.clock.advance(45.0)
        lc = CallLifecycle(
            ctx=self.ctx,
            state=state,
            settings=self.settings,
            session=self.session,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
            telemetry_drain_hook=pipeline.drain,
        )
        await lc.end_call(EndReason.USER_GOODBYE, farewell=False)
        await pipeline.drain(timeout=2.0)

        # Verify ended row
        calls_end = {c["call_id"]: c for c in store.query_recent_calls()}
        final_row = calls_end[state.call_id]
        self.assertEqual(final_row["status"], "ended")
        self.assertEqual(final_row["end_reason"], "user_goodbye")
        self.assertEqual(final_row["turns"], 2)
        self.assertEqual(final_row["tool_calls"], 1)
        self.assertEqual(final_row["duration_s"], 45.0)

    # -------------------------------------------------------------------------
    # Scenario 14: Failing telemetry database does not affect call or raise
    # -------------------------------------------------------------------------
    async def test_scenario_14_telemetry_failure_does_not_affect_call(self) -> None:
        """Storage write failure is caught, sets degraded mode, and never raises or breaks call."""
        failing_sink = FakeTelemetrySink(fail_writes=True)
        state = CallState(channel="web", language="en")
        state.telemetry = failing_sink

        lc = CallLifecycle(
            ctx=self.ctx,
            state=state,
            settings=self.settings,
            session=self.session,
            clock=self.clock,
            sleep_fn=self.clock.sleep,
            telemetry_drain_hook=failing_sink.drain,
        )

        # Call ends cleanly despite failing sink
        await lc.end_call(EndReason.USER_GOODBYE, farewell=False)
        self.assertEqual(lc.lifecycle_state, LifecycleState.ENDED)
        self.assertEqual(failing_sink.drain_calls, 1)


if __name__ == "__main__":
    unittest.main()
