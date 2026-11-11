"""Pinned LiveKit 1.2.6 entrypoint. No custom transcript-to-reply path."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from dataclasses import asdict
from time import monotonic

from livekit import rtc
from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.cli import cli as cli_state
from livekit.plugins import deepgram, google, noise_cancellation, sarvam, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent.assistant import CallState, NexiAgent
from agent.config import ConfigurationError, load_settings
from agent.prompts import GREETINGS, PROMPT_VERSION, Channel
from agent.telemetry import configure_logging, event
from rag.retriever import Retriever


def read_channel(metadata: str) -> Channel:
    """Dispatch is validated; actual participant type determines the room profile."""
    if len(metadata) > 4096:
        raise ValueError("Job metadata is too large")
    try:
        payload = json.loads(metadata or "{}")
    except json.JSONDecodeError:
        raise ValueError("Job metadata must be a JSON object") from None
    if not isinstance(payload, dict):
        raise ValueError("Job metadata must be a JSON object")
    channel = payload.get("channel", "web")
    if channel not in ("web", "phone") or payload.get("language", "en") != "en":
        raise ValueError("Phase 1 supports web/phone profiles in English only")
    return channel


def prewarm(proc: JobProcess) -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    retriever = Retriever(settings)
    try:
        proc.userdata.update(settings=settings, retriever=retriever, vad=silero.VAD.load())
    except BaseException:
        retriever.close()
        raise
    event("prewarm_ready", knowledge_version=retriever.knowledge_version)


async def entrypoint(ctx: JobContext) -> None:
    settings = ctx.proc.userdata["settings"]
    configure_logging(settings.log_level)
    retriever = ctx.proc.userdata["retriever"]
    vad = ctx.proc.userdata["vad"]
    console = bool(cli_state.CLI_ARGUMENTS and cli_state.CLI_ARGUMENTS.console)
    state = CallState(channel="web")
    usage = metrics.UsageCollector()
    session = None
    providers = []
    tasks: set[asyncio.Task] = set()
    closed = False
    fatal_reported = False

    async def cleanup(reason: str = "startup_failed") -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            async with asyncio.timeout(settings.shutdown_timeout):
                if session is not None:
                    await session.aclose()
        except Exception as exc:
            event(
                "session_close_error",
                level=logging.WARNING,
                call_id=state.call_id,
                error_type=type(exc).__name__,
            )
        try:
            async with asyncio.timeout(settings.shutdown_timeout):
                await asyncio.gather(*(p.aclose() for p in providers), return_exceptions=True)
                await vad.aclose()
        except Exception as exc:
            event(
                "provider_close_error",
                level=logging.WARNING,
                call_id=state.call_id,
                error_type=type(exc).__name__,
            )
        finally:
            retriever.close()
            # The framework's reason may contain external content: log our own status only.
            event(
                "call_summary",
                call_id=state.call_id,
                channel=state.channel,
                language="en",
                status="error" if state.errors else "closed",
                turns=state.turns,
                tool_calls=state.tool_calls,
                no_results=state.no_results,
                errors=state.errors,
                timeouts=state.timeouts,
                duration=round(monotonic() - state.started_at, 3),
                knowledge_version=retriever.knowledge_version,
                prompt_version=PROMPT_VERSION,
                usage=asdict(usage.get_summary()),
            )

    ctx.add_shutdown_callback(cleanup)
    try:
        settings.validate_runtime(console=console)
        requested_channel = read_channel(ctx.job.metadata or "")
        room_options = RoomInputOptions()
        if not console:
            await asyncio.wait_for(ctx.connect(), settings.participant_timeout)
            participant = await asyncio.wait_for(
                ctx.wait_for_participant(),
                settings.participant_timeout,
            )
            state.channel = (
                "phone" if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else "web"
            )
            if requested_channel != state.channel:
                event("channel_metadata_mismatch", call_id=state.call_id, channel=state.channel)
            room_options.participant_identity = participant.identity
            if settings.noise_cancellation:
                room_options.noise_cancellation = (
                    noise_cancellation.BVCTelephony()
                    if state.channel == "phone"
                    else noise_cancellation.BVC()
                )

        stt = deepgram.STT(
            model=settings.stt_model,
            language=settings.stt_language,
            api_key=settings.deepgram_api_key,
            interim_results=True,
            punctuate=True,
        )
        providers.append(stt)
        llm = google.LLM(
            model=settings.llm_model,
            api_key=settings.google_api_key,
            vertexai=False,
            temperature=0.2,
        )
        providers.append(llm)
        tts = sarvam.TTS(
            target_language_code="en-IN",
            model=settings.tts_model,
            speaker=settings.tts_speaker,
            api_key=settings.sarvam_api_key,
            speech_sample_rate=(
                settings.phone_sample_rate if state.channel == "phone" else settings.web_sample_rate
            ),
        )
        providers.append(tts)
        session = AgentSession(
            userdata=state,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
            turn_detection=MultilingualModel(),
            max_tool_steps=3,
        )

        @session.on("metrics_collected")
        def on_metrics(ev):
            usage.collect(ev.metrics)
            data = ev.metrics.model_dump()
            # Exclude provider metadata; only declared numeric stage measurements are logged.
            measurements = {
                key: value
                for key, value in data.items()
                if isinstance(value, (int, float, bool))
                and key != "timestamp"
                and math.isfinite(value)
            }
            event(
                "stage_metrics",
                call_id=state.call_id,
                channel=state.channel,
                language="en",
                stage=ev.metrics.type,
                **measurements,
            )

        @session.on("user_input_transcribed")
        def on_transcript(ev):
            if ev.is_final:
                state.turns += 1  # Observe only: never generate a reply here.

        async def report_failure() -> None:
            try:
                await asyncio.wait_for(
                    session.say(
                        "Sorry, I'm having technical trouble. Please contact the university office."
                    ),
                    settings.speech_timeout,
                )
            except Exception:
                event("failure_notice_unavailable", call_id=state.call_id)
            finally:
                ctx.shutdown(reason="provider_failure")

        @session.on("error")
        def on_error(ev):
            nonlocal fatal_reported
            state.errors += 1
            recoverable = bool(getattr(ev.error, "recoverable", False))
            event(
                "provider_error",
                level=logging.ERROR,
                call_id=state.call_id,
                error_type=type(ev.error).__name__,
                recoverable=recoverable,
            )
            if not recoverable and not fatal_reported and not closed:
                fatal_reported = True
                task = asyncio.create_task(report_failure())
                tasks.add(task)
                task.add_done_callback(tasks.discard)

        @session.on("close")
        def on_close(ev):
            if not closed:
                ctx.shutdown(reason="session_closed")

        await session.start(
            room=ctx.room,
            agent=NexiAgent(retriever, settings, state.channel),
            room_input_options=room_options,
        )
        event(
            "call_ready",
            call_id=state.call_id,
            channel=state.channel,
            language="en",
            knowledge_version=retriever.knowledge_version,
        )
        try:
            await asyncio.wait_for(session.say(GREETINGS[state.channel]), settings.speech_timeout)
        except TimeoutError:
            state.timeouts += 1
            session.interrupt()
            event("greeting_timeout", call_id=state.call_id)
        # Returning keeps the framework-owned job alive; shutdown callback owns cleanup.
    except BaseException:
        state.errors += 1
        await cleanup()
        raise


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    # Download/help commands need no provider keys. No connection occurs at module import.
    if any(command in sys.argv[1:] for command in ("console", "dev", "start", "connect")):
        settings.validate_runtime(console="console" in sys.argv[1:])
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=settings.agent_name,
            ws_url=settings.livekit_url or "ws://localhost:7880",
            api_key=settings.livekit_api_key or None,
            api_secret=settings.livekit_api_secret or None,
            initialize_process_timeout=settings.prewarm_timeout,
            shutdown_process_timeout=settings.shutdown_timeout * 3,
            num_idle_processes=settings.idle_processes,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from None
