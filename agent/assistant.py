"""One agent and function tools; LiveKit owns the conversation loop."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from time import monotonic, time
from typing import Any
from uuid import uuid4

from livekit.agents import Agent, RunContext, ToolError, function_tool
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import noise_cancellation

from agent.config import Settings
from agent.prompts import Channel, Language, build_prompt
from agent.telemetry import event
from rag.retriever import RetrievalBusy, Retriever


def read_channel(metadata: str) -> tuple[Channel, Language]:
    """Parse dispatch metadata for channel and language."""
    import json
    if len(metadata) > 4096:
        raise ValueError("Job metadata is too large")
    try:
        payload = json.loads(metadata or "{}")
    except json.JSONDecodeError:
        raise ValueError("Job metadata must be a JSON object") from None
    if not isinstance(payload, dict):
        raise ValueError("Job metadata must be a JSON object")
    channel: Channel = payload.get("channel", "web")
    if channel not in ("web", "phone"):
        channel = "web"
    language: Language = payload.get("language", "en")
    if language not in ("en", "te", "hi"):
        language = "en"
    return channel, language


def mask_phone_number(raw: str) -> str:
    """Mask phone number preserving leading and trailing digits."""
    import re
    digits = re.sub(r"[^\d+]", "", raw)
    return digits[:3] + "****" + digits[-4:] if len(digits) > 6 else "***"


def pick_noise_cancellation(participant_kind: Any):
    """Select noise cancellation filter by participant kind."""
    if hasattr(participant_kind, "name") and "SIP" in participant_kind.name:
        return noise_cancellation.BVCTelephony()
    return noise_cancellation.BVC()


def build_providers(settings: Settings, language: Language, channel: Channel) -> tuple[Any, Any, Any]:
    """Instantiate speech and language models configured for requested language and channel."""
    from livekit.plugins import deepgram, google

    from agent.tts_adapter import NexiSarvamTTS

    stt_lang = {"en": settings.stt_language, "te": "te", "hi": "hi"}.get(language, settings.stt_language)
    stt = deepgram.STT(model=settings.stt_model, language=stt_lang, api_key=settings.deepgram_api_key, interim_results=True, punctuate=True)
    llm = google.LLM(model=settings.llm_model, api_key=settings.google_api_key, vertexai=False, temperature=0.2)
    tts_lang = "te-IN" if language == "te" else ("hi-IN" if language == "hi" else "en-IN")
    rate = settings.phone_sample_rate if channel == "phone" else settings.web_sample_rate
    tts = NexiSarvamTTS(target_language_code=tts_lang, model=settings.tts_model, speaker=settings.tts_speaker, api_key=settings.sarvam_api_key, speech_sample_rate=rate)
    return stt, llm, tts


async def connect_participant(ctx: Any, settings: Settings, state: CallState, room_opts: Any, console: bool) -> tuple[str, str]:
    """Connect to room, wait for participant, and configure channel / noise cancellation."""
    if console:
        return "", ""
    from livekit import rtc
    await asyncio.wait_for(ctx.connect(), settings.participant_timeout)
    part = await asyncio.wait_for(ctx.wait_for_participant(), settings.participant_timeout)
    part_id = part.identity
    room_opts.participant_identity = part.identity
    phone_num = ""
    if part.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        state.channel = "phone"
        phone_num = part.attributes.get("sip.phoneNumber", "")
        if phone_num:
            state.caller_mask = mask_phone_number(phone_num)
            event("sip_caller_connected", call_id=state.call_id, caller=state.caller_mask)
    else:
        state.channel = "web"
    if settings.noise_cancellation:
        room_opts.noise_cancellation = pick_noise_cancellation(part.kind)
    return part_id, phone_num


def record_call_started(telemetry: Any, state: CallState, ctx: Any, phone_num: str) -> None:
    """Record initial call_started telemetry event asynchronously."""
    if telemetry is None:
        return
    import hashlib
    r_name = getattr(ctx.room, "name", "")
    r_hash = hashlib.sha256(r_name.encode()).hexdigest()[:12] if r_name else ""
    asyncio.create_task(
        telemetry.put_event(
            "call_started",
            call_id=state.call_id,
            channel=state.channel,
            language=state.language,
            job_id=getattr(ctx.job, "id", ""),
            room_hash=r_hash,
            caller_number=phone_num if state.channel == "phone" else "",
        )
    )


@dataclass
class CallState:
    channel: Channel
    language: Language = "en"
    call_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: float = field(default_factory=time)
    started_monotonic: float = field(default_factory=monotonic)
    turns: int = 0
    tool_calls: int = 0
    no_results: int = 0
    timeouts: int = 0
    errors: int = 0
    status: str = "started"
    end_reason: str = ""
    transfer_attempted: bool = False
    transfer_succeeded: bool = False
    transfer_department: str = ""
    transfer_failures: int = 0
    room_delete_failures: int = 0
    language_switches: int = 0
    language_switch_failures: int = 0
    caller_mask: str = ""
    caller_hash: str = ""
    explicit_exit: bool = False
    lifecycle: Any = None
    telemetry: Any = None
    usage_summary: dict[str, Any] = field(default_factory=dict)


def _build_transfer_tool(targets: dict[str, str]):
    target_keys = tuple(sorted(targets.keys()))
    dept_list = ", ".join(target_keys)

    @function_tool
    async def transfer_to_office(context: RunContext[CallState], department: str) -> str:
        """Transfer the student's phone call to an allowlisted university office department.

        Args:
            department: The university department key to transfer to (e.g. admissions, accounts).
        """
        state = context.userdata
        dept_clean = department.strip().lower()
        if dept_clean not in targets:
            return f"Department '{department}' is not recognized. Configured options: {dept_list}."

        if state.lifecycle is not None:
            success = await state.lifecycle.transfer_to_office(dept_clean)
            if success:
                return "Call transferred successfully."
            return "Office transfer could not be completed. Offer to assist the caller directly."
        return "Transfer system is currently unattached."

    return transfer_to_office


class NexiAgent(Agent):
    def __init__(
        self,
        retriever: Retriever,
        settings: Settings,
        channel: Channel,
        language: Language = "en",
        stt: Any = NOT_GIVEN,
        tts: Any = NOT_GIVEN,
    ) -> None:
        extra_tools = []
        if channel == "phone" and settings.transfer_targets:
            extra_tools.append(_build_transfer_tool(settings.transfer_targets))

        kwargs: dict[str, Any] = {}
        if stt is not NOT_GIVEN and stt is not None:
            kwargs["stt"] = stt
        if tts is not NOT_GIVEN and tts is not None:
            kwargs["tts"] = tts

        super().__init__(
            instructions=build_prompt(channel, settings.timezone, language=language),
            tools=extra_tools if extra_tools else None,
            **kwargs,
        )
        self._retriever = retriever
        self._settings = settings
        self._language = language

    @function_tool
    async def search_university_info(self, context: RunContext[CallState], query: str) -> str:
        """Look up public SRM AP facts: fees, hostels, dates, faculty, admissions and policies.

        Args:
            query: A short standalone English question, resolving follow-ups from this call.
        """
        query = query.strip()
        if not query or len(query) > self._settings.query_max_chars:
            raise ToolError("Ask the student to repeat the question.")
        state = context.userdata
        state.tool_calls += 1
        start = monotonic()
        outcome = "error"
        try:
            result = await self._retriever.search(query)
            outcome = "found" if result.evidence else "no_results"
            if not result.evidence:
                state.no_results += 1
                if state.telemetry is not None:
                    asyncio.create_task(
                        state.telemetry.put_event(
                            "knowledge_gap",
                            query=query,
                            language=state.language,
                            channel=state.channel,
                            reason=result.reason or "no_results",
                            top_score=result.top_score,
                            nearest_topic=result.nearest_topic,
                        )
                    )
                return "NO_RESULTS"
            return result.context
        except (TimeoutError, RetrievalBusy):
            outcome = "timeout_or_busy"
            state.timeouts += 1
            if state.telemetry is not None:
                asyncio.create_task(
                    state.telemetry.put_event(
                        "knowledge_gap",
                        query=query,
                        language=state.language,
                        channel=state.channel,
                        reason="timeout",
                        top_score=0.0,
                        nearest_topic="",
                    )
                )
            raise ToolError(
                "Information lookup is temporarily busy. Ask the student to try again."
            ) from None
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            state.errors += 1
            event(
                "retrieval_error",
                level=logging.ERROR,
                call_id=state.call_id,
                error_type=type(exc).__name__,
            )
            raise ToolError(
                "Information lookup is temporarily unavailable. Apologise briefly."
            ) from None
        finally:
            event(
                "retrieval",
                call_id=state.call_id,
                channel=state.channel,
                language=state.language,
                outcome=outcome,
                duration=round(monotonic() - start, 3),
                knowledge_version=self._retriever.knowledge_version,
            )

    @function_tool
    async def end_call(self, context: RunContext[CallState]) -> str:
        """Call this tool ONLY when the student explicitly and clearly says goodbye or is finished (e.g. 'Goodbye', 'That is all', 'No more questions'). Never call this for a bare thank you followed by more speech."""
        state = context.userdata
        # Guard 1: ignore within the first 5 seconds of the call
        if (monotonic() - state.started_monotonic) < 5.0:
            return "Call started recently, continuing conversation."
        # Guard 2: ignore second call while already ending
        if state.explicit_exit:
            return "Call termination already scheduled."

        state.explicit_exit = True
        if state.lifecycle is not None:
            from agent.lifecycle import EndReason

            state.lifecycle.track_task(
                asyncio.create_task(
                    state.lifecycle.end_call(EndReason.USER_GOODBYE, farewell=True)
                )
            )
        return "Call ending scheduled. Say a brief polite farewell and nothing more."

    @function_tool
    async def switch_language(self, context: RunContext[CallState], language: str) -> str:
        """Switch the conversation language between English ('en'), Telugu ('te'), and Hindi ('hi').

        Args:
            language: The target language code to switch into ('en', 'te', or 'hi').
        """
        state = context.userdata
        lang_clean = language.strip().lower()
        if lang_clean not in ("en", "te", "hi"):
            return f"Unsupported language '{language}'. Supported options are English ('en'), Telugu ('te'), and Hindi ('hi')."
        if lang_clean == state.language:
            return f"Conversation is already in {language}."

        from agent.language import switch_call_language

        session = getattr(state.lifecycle, "session", None)
        success, msg = await switch_call_language(
            session=session,
            state=state,
            settings=self._settings,
            retriever=self._retriever,
            target_language=lang_clean,
        )
        return msg
