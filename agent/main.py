"""LiveKit entrypoint: configures session, attaches lifecycle, and starts worker."""

from __future__ import annotations

import asyncio
import json
import sys

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
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent.assistant import (
    CallState,
    NexiAgent,
    build_providers,
    connect_participant,
    mask_phone_number,
    pick_noise_cancellation,
    read_channel,
    record_call_started,
)
from agent.config import ConfigurationError, Settings, load_settings
from agent.language import PHRASES
from agent.lifecycle import CallLifecycle, EndReason
from agent.maintenance import reconcile_stale_calls
from agent.storage import TelemetryStore
from agent.telemetry import TelemetryPipeline, configure_logging, event
from rag.retriever import Retriever

__all__ = ["entrypoint", "main", "mask_phone_number", "pick_noise_cancellation", "prewarm", "read_channel"]


def prewarm(proc: JobProcess) -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    retriever = Retriever(settings)
    try:
        storage = TelemetryStore(settings)
        reconcile_stale_calls(storage, settings)
        telemetry = TelemetryPipeline(settings, storage)
        proc.userdata.update(
            settings=settings,
            retriever=retriever,
            vad=silero.VAD.load(),
            storage=storage,
            telemetry=telemetry,
        )
    except BaseException:
        retriever.close()
        raise
    event("prewarm_ready", knowledge_version=retriever.knowledge_version)


async def entrypoint(ctx: JobContext) -> None:
    settings: Settings = ctx.proc.userdata["settings"]
    configure_logging(settings.log_level)
    retriever: Retriever = ctx.proc.userdata["retriever"]
    vad = ctx.proc.userdata["vad"]
    storage: TelemetryStore | None = ctx.proc.userdata.get("storage")
    telemetry: TelemetryPipeline | None = ctx.proc.userdata.get("telemetry")
    if telemetry is not None:
        telemetry.start()
    console = bool(cli_state.CLI_ARGUMENTS and cli_state.CLI_ARGUMENTS.console)

    ch, lang = read_channel(ctx.job.metadata or "")
    state = CallState(channel=ch, language=lang, telemetry=telemetry)
    usage = metrics.UsageCollector()
    room_opts = RoomInputOptions()
    part_id, phone_num = await connect_participant(ctx, settings, state, room_opts, console)

    raw_meta = json.loads(ctx.job.metadata or "{}") if ctx.job.metadata else {}
    if state.channel == "phone" and "language" not in raw_meta and not console:
        from agent.language import PhoneLanguageMenu
        state.language = await PhoneLanguageMenu(ctx.room, None, state, settings).run_menu()

    record_call_started(telemetry, state, ctx, phone_num)

    stt, llm, tts = build_providers(settings, state.language, state.channel)
    session = AgentSession(userdata=state, stt=stt, llm=llm, tts=tts, vad=vad, turn_detection=MultilingualModel(), max_tool_steps=3)
    lifecycle = CallLifecycle(
        ctx=ctx,
        state=state,
        settings=settings,
        session=session,
        room=ctx.room,
        participant_identity=part_id,
        room_name=getattr(ctx.room, "name", ""),
        telemetry_drain_hook=telemetry.drain if telemetry is not None else None,
    )
    state.lifecycle = lifecycle
    ctx.add_shutdown_callback(lambda reason: asyncio.create_task(lifecycle.end_call(EndReason.JOB_SHUTDOWN)))
    lifecycle.attach_session_handlers(telemetry=telemetry, storage=storage, usage=usage, console=console)

    await session.start(room=ctx.room, agent=NexiAgent(retriever, settings, state.channel, language=state.language), room_input_options=room_opts)
    event("call_ready", call_id=state.call_id, channel=state.channel, language=state.language, knowledge_version=retriever.knowledge_version)

    greet_key = "greeting_phone" if state.channel == "phone" else "greeting_web"
    try:
        await asyncio.wait_for(session.say(PHRASES[greet_key][state.language]), settings.speech_timeout)
    except TimeoutError:
        state.timeouts += 1
        session.interrupt()
        event("greeting_timeout", call_id=state.call_id)

    lifecycle.record_activity()
    lifecycle.start_duration_limiter()
    lifecycle.start_inactivity_monitor()


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    if any(cmd in sys.argv[1:] for cmd in ("console", "dev", "start", "connect")):
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
