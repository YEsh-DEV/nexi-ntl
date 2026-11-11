"""One agent and one knowledge tool; LiveKit owns the conversation loop."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from livekit.agents import Agent, RunContext, ToolError, function_tool

from agent.config import Settings
from agent.prompts import Channel, Language, build_prompt
from agent.telemetry import event
from rag.retriever import RetrievalBusy, Retriever


@dataclass
class CallState:
    channel: Channel
    language: Language = "en"
    call_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: float = field(default_factory=monotonic)
    turns: int = 0
    tool_calls: int = 0
    no_results: int = 0
    timeouts: int = 0
    errors: int = 0


class NexiAgent(Agent):
    def __init__(
        self,
        retriever: Retriever,
        settings: Settings,
        channel: Channel,
        language: Language = "en",
    ) -> None:
        super().__init__(instructions=build_prompt(channel, settings.timezone, language=language))
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
                return "NO_RESULTS"
            return result.context
        except (TimeoutError, RetrievalBusy):
            outcome = "timeout_or_busy"
            state.timeouts += 1
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
