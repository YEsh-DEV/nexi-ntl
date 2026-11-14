"""Fakes and test doubles for session-flow scenarios without network, audio, or external APIs."""

from __future__ import annotations

import asyncio
from typing import Any, Callable
from unittest.mock import MagicMock

from rag.retriever import Evidence, SearchResult


class FakeClock:
    """Controllable monotonic virtual clock with cooperative time-advancing sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self.current_time = start

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds

    async def sleep(self, seconds: float) -> None:
        self.current_time += seconds
        # Cooperative yield to let concurrent asyncio tasks make progress
        await asyncio.sleep(0)


class FakeSession:
    """Double for LiveKit AgentSession recording spoken phrases, aclose, interruptions, and history."""

    def __init__(
        self,
        say_hang: bool = False,
        aclose_hang: bool = False,
        agent: Any = None,
    ) -> None:
        self.spoken: list[str] = []
        self.closed: bool = False
        self.say_hang = say_hang
        self.aclose_hang = aclose_hang
        self.interrupted = False
        self.current_agent = agent
        self.history: list[dict[str, str]] = []
        self._listeners: dict[str, list[Callable[[Any], None]]] = {}

    def on(self, event_name: str) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
        def decorator(fn: Callable[[Any], None]) -> Callable[[Any], None]:
            self._listeners.setdefault(event_name, []).append(fn)
            return fn

        return decorator

    def emit(self, event_name: str, event_obj: Any) -> None:
        for fn in self._listeners.get(event_name, []):
            fn(event_obj)

    async def say(self, text: str) -> None:
        if self.say_hang:
            await asyncio.sleep(999.0)
        self.spoken.append(text)

    def interrupt(self) -> None:
        self.interrupted = True

    async def aclose(self) -> None:
        if self.aclose_hang:
            await asyncio.sleep(999.0)
        self.closed = True

    def update_agent(self, new_agent: Any) -> None:
        self.current_agent = new_agent


class FakeJobContext:
    """Double for LiveKit JobContext."""

    def __init__(self) -> None:
        self.shutdown_reasons: list[str] = []
        self.shutdown_callbacks: list[Callable[[str], Any]] = []

    def shutdown(self, reason: str = "") -> None:
        self.shutdown_reasons.append(reason)
        for cb in self.shutdown_callbacks:
            res = cb(reason)
            if asyncio.iscoroutine(res):
                asyncio.create_task(res)

    def add_shutdown_callback(self, cb: Callable[[str], Any]) -> None:
        self.shutdown_callbacks.append(cb)


class FakeRoomApi:
    """Double for LiveKit server API room deletion and SIP participant transfer."""

    def __init__(
        self,
        fail_delete: bool = False,
        fail_transfer: bool = False,
        hang_transfer: bool = False,
    ) -> None:
        self.deleted_rooms: list[str] = []
        self.transfers: list[dict[str, str]] = []
        self.fail_delete = fail_delete
        self.fail_transfer = fail_transfer
        self.hang_transfer = hang_transfer

        self.room = MagicMock()
        self.room.delete_room = self._delete_room
        self.sip = MagicMock()
        self.sip.transfer_sip_participant = self._transfer_sip

    async def _delete_room(self, req: Any) -> None:
        if self.fail_delete:
            raise RuntimeError("Room delete failed on server")
        self.deleted_rooms.append(getattr(req, "room", str(req)))

    async def _transfer_sip(self, req: Any) -> None:
        if self.hang_transfer:
            await asyncio.sleep(999.0)
        if self.fail_transfer:
            raise RuntimeError("SIP REFER transfer failed on server")
        self.transfers.append(
            {
                "participant": getattr(req, "participant_identity", ""),
                "room": getattr(req, "room_name", ""),
                "target": getattr(req, "transfer_to", ""),
            }
        )

    async def aclose(self) -> None:
        pass


class FakeRetriever:
    """Double for knowledge base retriever with scripted results and failure injection."""

    def __init__(
        self,
        default_result: SearchResult | None = None,
        knowledge_version: str = "v1_test",
    ) -> None:
        self.knowledge_version = knowledge_version
        self.scripted_results: dict[str, SearchResult] = {}
        self.scripted_errors: dict[str, Exception] = {}
        self.search_calls: list[str] = []
        self.default_result = default_result or SearchResult(
            (
                Evidence(
                    chunk_id="c1",
                    text="SRM University-AP B.Tech annual tuition fee is Rs. 2,50,000.",
                    source="fees.pdf",
                    page_or_path="1",
                    score=0.92,
                ),
            ),
            knowledge_version=knowledge_version,
        )

    async def search(self, query: str) -> SearchResult:
        self.search_calls.append(query)
        clean = query.strip()
        if clean in self.scripted_errors:
            raise self.scripted_errors[clean]
        if clean in self.scripted_results:
            return self.scripted_results[clean]
        return self.default_result


class FakeTelemetrySink:
    """In-memory event sink for verifying operational telemetry without SQLite."""

    def __init__(self, fail_writes: bool = False) -> None:
        self.events: list[dict[str, Any]] = []
        self.fail_writes = fail_writes
        self.drain_calls = 0

    async def put_event(self, event_type: str, **payload: Any) -> bool:
        if self.fail_writes:
            raise RuntimeError("Simulated telemetry sink failure")
        self.events.append({"event_type": event_type, "payload": payload})
        return True

    async def drain(self, timeout: float = 5.0) -> None:
        self.drain_calls += 1


class ScriptedLLM:
    """Simulates conversational model responses and tool executions deterministically."""

    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        self.script = list(script or [])
        self.turns_handled: list[str] = []

    async def execute_turn(
        self,
        user_text: str,
        agent: Any,
        run_context: Any,
        session: FakeSession,
    ) -> str:
        self.turns_handled.append(user_text)
        if not self.script:
            # Default fallback: answer directly
            reply = f"Nexi answering: {user_text}"
            await session.say(reply)
            return reply

        step = self.script.pop(0)
        tool_name = step.get("tool")
        if tool_name:
            tool_args = step.get("args", {})
            tool_fn = getattr(agent, tool_name, None)
            if callable(tool_fn):
                tool_output = await tool_fn(run_context, **tool_args)
                reply = step.get("reply", f"Tool executed: {tool_output}")
                await session.say(reply)
                return reply

        reply = step.get("reply", "Understood.")
        await session.say(reply)
        return reply
