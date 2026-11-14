"""Centralized, bounded, and observable call lifecycle controller.

Every call termination, timeout, farewell, duration limit, and office transfer
routes strictly through this module. No other module closes sessions or shuts down jobs.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import math
import time
from enum import Enum
from typing import Any, Callable, Coroutine

from livekit import api

from agent.config import Settings
from agent.language import PHRASES, Language
from agent.telemetry import event

LOGGER = logging.getLogger("agent.lifecycle")


class EndReason(str, Enum):
    USER_GOODBYE = "user_goodbye"
    IDLE_TIMEOUT = "idle_timeout"
    DURATION_LIMIT = "duration_limit"
    TRANSFERRED = "transferred"
    FATAL_PROVIDER_ERROR = "fatal_provider_error"
    PARTICIPANT_DISCONNECTED = "participant_disconnected"
    JOB_SHUTDOWN = "job_shutdown"
    INTERNAL_ERROR = "internal_error"


class LifecycleState(str, Enum):
    ACTIVE = "active"
    ENDING = "ending"
    ENDED = "ended"


class CallLifecycle:
    def __init__(
        self,
        ctx: Any,
        state: Any,
        settings: Settings,
        session: Any = None,
        room: Any = None,
        participant_identity: str = "",
        room_name: str = "",
        telemetry_drain_hook: Callable[[], Coroutine[Any, Any, None]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
        room_api: Any = None,
    ) -> None:
        self.ctx = ctx
        self.state = state
        self.settings = settings
        self.session = session
        self.room = room
        self.participant_identity = participant_identity
        self.room_name = room_name or (getattr(room, "name", "") if room else "")
        self.telemetry_drain_hook = telemetry_drain_hook
        self.clock = clock
        self.sleep_fn = sleep_fn
        self.room_api = room_api

        self.lifecycle_state = LifecycleState.ACTIVE
        self.end_reason: EndReason | None = None
        self.ended_at_monotonic: float = 0.0
        self.transfer_in_progress: bool = False
        self.step_outcomes: dict[str, str] = {}

        self._lock = asyncio.Lock()
        self._tracked_tasks: set[asyncio.Task[Any]] = set()
        self._agent_busy: bool = False
        self._user_busy: bool = False
        self._last_activity: float = self.clock()

    def track_task(self, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        """Track a background task for coordinated cancellation at teardown."""
        self._tracked_tasks.add(task)
        task.add_done_callback(self._tracked_tasks.discard)
        return task

    def set_agent_busy(self, busy: bool) -> None:
        self._agent_busy = busy
        if busy:
            self._last_activity = self.clock()

    def set_user_busy(self, busy: bool) -> None:
        self._user_busy = busy
        if busy:
            self._last_activity = self.clock()

    def record_activity(self) -> None:
        self._last_activity = self.clock()

    def attach_session_handlers(
        self,
        telemetry: Any = None,
        storage: Any = None,
        usage: Any = None,
        console: bool = False,
    ) -> None:
        """Register session listeners for state, transcript, metrics, and errors."""
        if self.session is None:
            return

        @self.session.on("agent_state_changed")
        def on_agent_state(ev: Any) -> None:
            self.set_agent_busy(getattr(ev, "new_state", "") in ("thinking", "speaking"))

        @self.session.on("user_state_changed")
        def on_user_state(ev: Any) -> None:
            self.set_user_busy(getattr(ev, "new_state", "") == "speaking")

        @self.session.on("user_input_transcribed")
        def on_transcript(ev: Any) -> None:
            self.record_activity()
            if getattr(ev, "is_final", False) and self.state is not None:
                self.state.turns += 1

        active_turns: dict[str, dict[str, float]] = {}

        @self.session.on("metrics_collected")
        def on_metrics(ev: Any) -> None:
            if usage is not None:
                usage.collect(ev.metrics)
                try:
                    summary = usage.get_summary()
                    if self.state is not None:
                        self.state.usage_summary = (
                            dataclasses.asdict(summary)
                            if dataclasses.is_dataclass(summary)
                            else (summary.model_dump() if hasattr(summary, "model_dump") else {})
                        )
                except Exception as exc:
                    LOGGER.debug("Could not serialize usage summary: %s", exc)

            m = ev.metrics
            speech_id = getattr(m, "speech_id", "") or "turn"
            turn_data = active_turns.setdefault(speech_id, {})
            m_type = type(m).__name__
            if "EOU" in m_type or hasattr(m, "end_of_utterance_delay"):
                turn_data["eou_delay_ms"] = (getattr(m, "end_of_utterance_delay", 0.0) or 0.0) * 1000.0
            elif "LLM" in m_type or hasattr(m, "ttft"):
                turn_data["llm_ttft_ms"] = (getattr(m, "ttft", 0.0) or 0.0) * 1000.0
            elif "TTS" in m_type or hasattr(m, "ttfb"):
                turn_data["tts_ttfb_ms"] = (getattr(m, "ttfb", 0.0) or 0.0) * 1000.0

            if "TTS" in m_type or ("llm_ttft_ms" in turn_data and "tts_ttfb_ms" in turn_data):
                eou = turn_data.get("eou_delay_ms", 0.0)
                llm_ttft = turn_data.get("llm_ttft_ms", 0.0)
                tts_ttfb = turn_data.get("tts_ttfb_ms", 0.0)
                total = eou + llm_ttft + tts_ttfb
                if telemetry is not None and self.state is not None:
                    t = asyncio.create_task(
                        telemetry.put_event(
                            "turn_metrics",
                            call_id=self.state.call_id,
                            channel=self.state.channel,
                            language=self.state.language,
                            eou_delay_ms=eou,
                            llm_ttft_ms=llm_ttft,
                            tts_ttfb_ms=tts_ttfb,
                            total_ms=total,
                            retrieval_ms=0.0,
                            tool_calls=self.state.tool_calls,
                        )
                    )
                    self.track_task(t)
                active_turns.pop(speech_id, None)

            if storage is not None and self.state is not None and hasattr(ev.metrics, "model_dump"):
                data = ev.metrics.model_dump()
                measurements = {
                    k: v for k, v in data.items()
                    if isinstance(v, (int, float, bool)) and k != "timestamp" and math.isfinite(v)
                }
                t = asyncio.create_task(
                    storage.record_stage_metrics(
                        call_id=self.state.call_id,
                        channel=self.state.channel,
                        language=self.state.language,
                        stage=getattr(ev.metrics, "type", m_type),
                        measurements=measurements,
                    )
                )
                self.track_task(t)

        @self.session.on("error")
        def on_error(ev: Any) -> None:
            if self.state is not None:
                self.state.errors += 1
            recoverable = bool(getattr(ev.error, "recoverable", False))
            LOGGER.error("Provider error: %s (recoverable: %s)", ev.error, recoverable)
            if not recoverable and self.lifecycle_state == LifecycleState.ACTIVE:
                prov = getattr(ev.error, "provider", "")
                self.track_task(asyncio.create_task(self.end_call(EndReason.FATAL_PROVIDER_ERROR, failed_provider=prov)))

        if self.room is not None:
            @self.room.on("participant_disconnected")
            def on_participant_disconnected(p: Any) -> None:
                if not console and getattr(p, "identity", None) == self.participant_identity:
                    self.track_task(asyncio.create_task(self.end_call(EndReason.PARTICIPANT_DISCONNECTED)))

    async def end_call(
        self,
        reason: EndReason,
        farewell: bool | str = False,
        failed_provider: str = "",
    ) -> None:
        """Single centralized teardown path. Concurrency-safe and idempotent."""
        async with self._lock:
            if self.lifecycle_state != LifecycleState.ACTIVE:
                LOGGER.debug(
                    "Call %s is already ending or ended (state=%s); ignoring duplicate reason=%s",
                    getattr(self.state, "call_id", ""),
                    self.lifecycle_state,
                    reason.value,
                )
                return

            self.lifecycle_state = LifecycleState.ENDING
            self.end_reason = reason
            self.ended_at_monotonic = self.clock()
            setattr(self.state, "end_reason", reason.value)
            setattr(self.state, "status", "ended" if not getattr(self.state, "errors", 0) else "error")

            event(
                "call_ending",
                call_id=getattr(self.state, "call_id", ""),
                reason=reason.value,
                channel=getattr(self.state, "channel", ""),
                language=getattr(self.state, "language", ""),
            )

            # Cancel all tracked auxiliary tasks immediately
            for task in list(self._tracked_tasks):
                if not task.done() and task != asyncio.current_task():
                    task.cancel()

            # Execute ordered, bounded teardown within END_CALL_TOTAL_TIMEOUT_S
            try:
                async with asyncio.timeout(self.settings.end_call_total_timeout_s):
                    await self._execute_teardown_steps(reason, farewell, failed_provider)
            except TimeoutError:
                LOGGER.warning(
                    "Call %s teardown exceeded total bound of %ss; skipping to immediate shutdown",
                    getattr(self.state, "call_id", ""),
                    self.settings.end_call_total_timeout_s,
                )
                self.step_outcomes["total_bound"] = "timeout"
            except Exception as exc:
                LOGGER.warning(
                    "Unexpected error during teardown of call %s: %s",
                    getattr(self.state, "call_id", ""),
                    type(exc).__name__,
                    exc_info=True,
                )
                self.step_outcomes["teardown_error"] = type(exc).__name__
            finally:
                # Guaranteed final shutdown of the framework job context
                self._shutdown_context(reason)
                self.lifecycle_state = LifecycleState.ENDED
                duration = round(self.clock() - getattr(self.state, "started_monotonic", self.clock()), 3)
                event(
                    "call_ended",
                    call_id=getattr(self.state, "call_id", ""),
                    reason=reason.value,
                    duration_s=duration,
                    step_outcomes=self.step_outcomes,
                )

    async def _execute_teardown_steps(
        self,
        reason: EndReason,
        farewell: bool | str,
        failed_provider: str,
    ) -> None:
        lang: Language = getattr(self.state, "language", "en")

        # Step 1: Speak farewell if requested and allowed
        allows_farewell = reason in {
            EndReason.USER_GOODBYE,
            EndReason.IDLE_TIMEOUT,
            EndReason.DURATION_LIMIT,
        }
        skip_speech = (
            reason == EndReason.PARTICIPANT_DISCONNECTED
            or (reason == EndReason.FATAL_PROVIDER_ERROR and failed_provider.lower() == "tts")
        )

        if farewell and allows_farewell and not skip_speech and self.session is not None:
            farewell_text = (
                farewell
                if isinstance(farewell, str)
                else PHRASES["farewell"].get(lang, PHRASES["farewell"]["en"])
            )
            try:
                async with asyncio.timeout(self.settings.farewell_timeout_s):
                    say_coro = getattr(self.session, "say", None)
                    if callable(say_coro):
                        await say_coro(farewell_text)
                self.step_outcomes["farewell"] = "ok"
            except TimeoutError:
                LOGGER.warning("Farewell speech timed out for call %s", getattr(self.state, "call_id", ""))
                self.step_outcomes["farewell"] = "timeout"
            except Exception as exc:
                LOGGER.warning(
                    "Farewell speech failed for call %s: %s",
                    getattr(self.state, "call_id", ""),
                    type(exc).__name__,
                )
                self.step_outcomes["farewell"] = f"error_{type(exc).__name__}"
        else:
            self.step_outcomes["farewell"] = "skipped"

        # Step 2: Close agent session
        if self.session is not None:
            try:
                async with asyncio.timeout(self.settings.session_close_timeout_s):
                    aclose_fn = getattr(self.session, "aclose", None)
                    if callable(aclose_fn):
                        await aclose_fn()
                self.step_outcomes["session_close"] = "ok"
            except TimeoutError:
                LOGGER.warning("Session close timed out for call %s", getattr(self.state, "call_id", ""))
                self.step_outcomes["session_close"] = "timeout"
            except Exception as exc:
                LOGGER.warning(
                    "Session close failed for call %s: %s",
                    getattr(self.state, "call_id", ""),
                    type(exc).__name__,
                )
                self.step_outcomes["session_close"] = f"error_{type(exc).__name__}"

        # Step 3: Room deletion (Phone channel only)
        channel = getattr(self.state, "channel", "web")
        if channel == "phone" and (self.room_api is not None or (self.settings.livekit_url and self.settings.livekit_api_key)):
            room_to_delete = self.room_name or (getattr(self.room, "name", "") if self.room else "")
            if room_to_delete:
                room_hash = hashlib.sha256(room_to_delete.encode()).hexdigest()[:12]
                try:
                    async with asyncio.timeout(self.settings.room_delete_timeout_s):
                        lk_api = self.room_api
                        should_close = False
                        if lk_api is None:
                            lk_api = api.LiveKitAPI(
                                self.settings.livekit_url,
                                self.settings.livekit_api_key,
                                self.settings.livekit_api_secret,
                            )
                            should_close = True
                        try:
                            await lk_api.room.delete_room(api.DeleteRoomRequest(room=room_to_delete))
                        finally:
                            if should_close:
                                await lk_api.aclose()
                    self.step_outcomes["room_delete"] = "ok"
                except TimeoutError:
                    setattr(self.state, "room_delete_failures", getattr(self.state, "room_delete_failures", 0) + 1)
                    LOGGER.warning("Delete room timed out for room %s (call %s)", room_hash, getattr(self.state, "call_id", ""))
                    self.step_outcomes["room_delete"] = "timeout"
                except Exception as exc:
                    setattr(self.state, "room_delete_failures", getattr(self.state, "room_delete_failures", 0) + 1)
                    LOGGER.warning(
                        "Delete room failed for room %s (call %s): %s",
                        room_hash,
                        getattr(self.state, "call_id", ""),
                        type(exc).__name__,
                    )
                    self.step_outcomes["room_delete"] = f"error_{type(exc).__name__}"
            else:
                self.step_outcomes["room_delete"] = "skipped_empty_name"
        else:
            self.step_outcomes["room_delete"] = "skipped"

        # Record call_ended to telemetry before drain
        telemetry = getattr(self.state, "telemetry", None)
        if telemetry is not None:
            duration = round(self.clock() - getattr(self.state, "started_monotonic", self.clock()), 3)
            try:
                await telemetry.put_event(
                    "call_ended",
                    call_id=getattr(self.state, "call_id", ""),
                    status="ended",
                    duration_s=duration,
                    end_reason=reason.value,
                    language_final=getattr(self.state, "language", "en"),
                    turns=getattr(self.state, "turns", 0),
                    tool_calls=getattr(self.state, "tool_calls", 0),
                    no_results_count=getattr(self.state, "no_results", 0),
                    language_switches=getattr(self.state, "language_switches", 0),
                    language_switch_failures=getattr(self.state, "language_switch_failures", 0),
                    transfer_attempted=1 if getattr(self.state, "transfer_attempted", False) else 0,
                    transfer_succeeded=1 if getattr(self.state, "transfer_succeeded", False) else 0,
                    transfer_department=getattr(self.state, "transfer_department", ""),
                    room_delete_failures=getattr(self.state, "room_delete_failures", 0),
                    errors=getattr(self.state, "errors", 0),
                    usage=getattr(self.state, "usage_summary", {}),
                )
            except Exception as exc:
                LOGGER.warning("Failed putting call_ended to telemetry: %s", exc)

        # Step 4: Drain telemetry
        if self.telemetry_drain_hook is not None:
            try:
                async with asyncio.timeout(self.settings.telemetry_drain_timeout_s):
                    await self.telemetry_drain_hook()
                self.step_outcomes["telemetry_drain"] = "ok"
            except TimeoutError:
                LOGGER.warning("Telemetry drain timed out for call %s", getattr(self.state, "call_id", ""))
                self.step_outcomes["telemetry_drain"] = "timeout"
            except Exception as exc:
                LOGGER.warning(
                    "Telemetry drain error for call %s: %s",
                    getattr(self.state, "call_id", ""),
                    type(exc).__name__,
                )
                self.step_outcomes["telemetry_drain"] = f"error_{type(exc).__name__}"
        else:
            self.step_outcomes["telemetry_drain"] = "skipped"

    def _shutdown_context(self, reason: EndReason) -> None:
        """Call job context shutdown safely."""
        if self.ctx is not None:
            shutdown_fn = getattr(self.ctx, "shutdown", None)
            if callable(shutdown_fn):
                try:
                    shutdown_fn(reason=reason.value)
                    self.step_outcomes["job_shutdown"] = "ok"
                except Exception as exc:
                    LOGGER.warning("Context shutdown failed: %s", type(exc).__name__)
                    self.step_outcomes["job_shutdown"] = f"error_{type(exc).__name__}"

    def start_duration_limiter(self) -> asyncio.Task[Any]:
        """Start the background hard duration watchdog."""
        channel = getattr(self.state, "channel", "web")
        limit = (
            self.settings.max_call_seconds_phone
            if channel == "phone"
            else self.settings.max_call_seconds_web
        )
        lead = self.settings.call_warning_lead_s

        async def _limiter_loop() -> None:
            warned = False
            start_m = getattr(self.state, "started_monotonic", self.clock())
            while self.lifecycle_state == LifecycleState.ACTIVE:
                await self.sleep_fn(1.0)
                if self.lifecycle_state != LifecycleState.ACTIVE:
                    break
                elapsed = self.clock() - start_m

                if not warned and elapsed >= (limit - lead):
                    warned = True
                    event("duration_warning_triggered", call_id=getattr(self.state, "call_id", ""), elapsed=round(elapsed, 1))
                    if self.session is not None and not self.transfer_in_progress:
                        lang: Language = getattr(self.state, "language", "en")
                        warning_msg = PHRASES["duration_warning"].get(lang, PHRASES["duration_warning"]["en"])
                        try:
                            # Wait until current speech finishes then speak warning
                            while self._agent_busy and self.lifecycle_state == LifecycleState.ACTIVE:
                                await self.sleep_fn(0.5)
                            if self.lifecycle_state == LifecycleState.ACTIVE:
                                say_fn = getattr(self.session, "say", None)
                                if callable(say_fn):
                                    await say_fn(warning_msg)
                        except Exception as exc:
                            LOGGER.warning("Failed to speak duration warning: %s", type(exc).__name__)

                if elapsed >= limit:
                    # If transfer is in progress, allow extra grace period
                    if self.transfer_in_progress:
                        grace_start = self.clock()
                        while (
                            self.transfer_in_progress
                            and (self.clock() - grace_start) < self.settings.transfer_grace_s
                        ):
                            await self.sleep_fn(0.5)

                    event("duration_limit_exceeded", call_id=getattr(self.state, "call_id", ""), elapsed=round(elapsed, 1))
                    await self.end_call(EndReason.DURATION_LIMIT, farewell=True)
                    break

        limiter_task = asyncio.create_task(_limiter_loop())
        return self.track_task(limiter_task)

    def start_inactivity_monitor(self) -> asyncio.Task[Any]:
        """Start silence monitoring: prompt on first silence, disconnect on second silence."""
        async def _inactivity_loop() -> None:
            idle_prompted = False
            while self.lifecycle_state == LifecycleState.ACTIVE:
                await self.sleep_fn(1.0)
                if self.lifecycle_state != LifecycleState.ACTIVE:
                    break
                if self._agent_busy or self._user_busy or self.transfer_in_progress:
                    self._last_activity = self.clock()
                    continue

                elapsed = self.clock() - self._last_activity
                lang: Language = getattr(self.state, "language", "en")

                if not idle_prompted and elapsed >= self.settings.idle_silence_timeout:
                    idle_prompted = True
                    event("idle_prompt_triggered", call_id=getattr(self.state, "call_id", ""), elapsed=round(elapsed, 1))
                    if self.session is not None:
                        prompt_msg = PHRASES["idle_prompt"].get(lang, PHRASES["idle_prompt"]["en"])
                        try:
                            say_fn = getattr(self.session, "say", None)
                            if callable(say_fn):
                                await say_fn(prompt_msg)
                        except Exception as exc:
                            LOGGER.warning("Failed to speak idle prompt: %s", type(exc).__name__)
                    # Grace window starts strictly AFTER the idle prompt completes speaking
                    self._last_activity = self.clock()

                elif idle_prompted and elapsed >= self.settings.idle_silence_grace:
                    event("idle_timeout_closing", call_id=getattr(self.state, "call_id", ""), elapsed=round(elapsed, 1))
                    await self.end_call(EndReason.IDLE_TIMEOUT, farewell=True)
                    break

        inactivity_task = asyncio.create_task(_inactivity_loop())
        return self.track_task(inactivity_task)

    async def transfer_to_office(self, department: str) -> bool:
        """Execute allowlisted SIP REFER transfer for inbound phone callers."""
        channel = getattr(self.state, "channel", "web")
        if channel != "phone":
            LOGGER.info("Transfer requested on non-phone channel '%s'; rejecting", channel)
            return False

        dep_key = department.strip().lower()
        target_number = self.settings.transfer_targets.get(dep_key)
        if not target_number:
            LOGGER.warning("Unknown department '%s' requested for transfer", dep_key)
            return False

        lang: Language = getattr(self.state, "language", "en")
        setattr(self.state, "transfer_attempted", True)
        setattr(self.state, "transfer_department", dep_key)
        self.transfer_in_progress = True

        # Announce transfer to caller
        if self.session is not None:
            announce_msg = PHRASES["transfer_announce"].get(lang, PHRASES["transfer_announce"]["en"])
            try:
                say_fn = getattr(self.session, "say", None)
                if callable(say_fn):
                    await say_fn(announce_msg)
            except Exception as exc:
                LOGGER.warning("Transfer announcement failed: %s", type(exc).__name__)

        # Call LiveKit SIP transfer API
        success = False
        try:
            async with asyncio.timeout(self.settings.transfer_timeout_s):
                lk_api = self.room_api
                should_close = False
                if lk_api is None:
                    lk_api = api.LiveKitAPI(
                        self.settings.livekit_url,
                        self.settings.livekit_api_key,
                        self.settings.livekit_api_secret,
                    )
                    should_close = True
                try:
                    transfer_req = api.TransferSIPParticipantRequest(
                        participant_identity=self.participant_identity,
                        room_name=self.room_name,
                        transfer_to=f"tel:{target_number}",
                    )
                    await lk_api.sip.transfer_sip_participant(transfer_req)
                    success = True
                finally:
                    if should_close:
                        await lk_api.aclose()
        except TimeoutError:
            LOGGER.warning("SIP transfer request timed out for department '%s'", dep_key)
        except Exception as exc:
            LOGGER.warning(
                "SIP transfer failed for department '%s': %s",
                dep_key,
                type(exc).__name__,
            )

        if success:
            setattr(self.state, "transfer_succeeded", True)
            self.transfer_in_progress = False
            event(
                "call_transferred",
                call_id=getattr(self.state, "call_id", ""),
                department=dep_key,
            )
            # End call cleanly with TRANSFERRED without deleting room before participant is moved
            await self.end_call(EndReason.TRANSFERRED, farewell=False)
            return True

        # Failure path: stay in call, count failure, speak apology, continue helping
        self.transfer_in_progress = False
        setattr(self.state, "transfer_failures", getattr(self.state, "transfer_failures", 0) + 1)
        event(
            "transfer_failed",
            call_id=getattr(self.state, "call_id", ""),
            department=dep_key,
        )
        if self.session is not None and self.lifecycle_state == LifecycleState.ACTIVE:
            fail_msg = PHRASES["transfer_failed"].get(lang, PHRASES["transfer_failed"]["en"])
            try:
                say_fn = getattr(self.session, "say", None)
                if callable(say_fn):
                    await say_fn(fail_msg)
            except Exception as exc:
                LOGGER.warning("Transfer failure apology failed: %s", type(exc).__name__)
        return False
