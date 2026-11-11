"""Offline scenario tests for the Nexi voice assistant.

Covers Scenario A (search tool invocations), Scenario B (channel-specific
instructions), Scenario C (channel dispatch parsing), and Scenario D (noise
cancellation selection) without requiring live LiveKit rooms, provider API keys,
or network access.

Run:
    .venv/bin/python -m unittest discover -s tests
"""

from __future__ import annotations

import asyncio
import unittest
from datetime import date

from livekit import rtc
from livekit.agents import ToolError
from livekit.plugins import noise_cancellation

from agent.assistant import CallState, NexiAgent
from agent.config import Settings
from agent.main import pick_noise_cancellation, read_channel
from agent.prompts import build_prompt

OFFLINE_SETTINGS = Settings(
    deepgram_api_key="offline",
    google_api_key="offline",
    sarvam_api_key="offline",
)


class _FakeEvidence:
    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeSearchResult:
    """Stand-in for rag.retriever.SearchResult."""

    def __init__(self, texts=(), knowledge_version="kv-test") -> None:
        self.evidence = tuple(_FakeEvidence(t) for t in texts)
        self.knowledge_version = knowledge_version

    @property
    def context(self) -> str:
        return "\n\n".join(evidence.text for evidence in self.evidence)


class _FakeRetriever:
    """Stand-in for rag.retriever.Retriever whose .search() we control."""

    def __init__(
        self,
        result=None,
        side_effect=None,
        knowledge_version="kv-test",
    ) -> None:
        self.knowledge_version = knowledge_version
        self._result = result
        self._side_effect = side_effect
        self.calls: list[str] = []

    async def search(self, query: str) -> _FakeSearchResult:
        self.calls.append(query)
        if self._side_effect is not None:
            raise self._side_effect
        return self._result


class _FakeContext:
    """Minimal stand-in for livekit.agents.RunContext exposing .userdata."""

    def __init__(self, state: CallState) -> None:
        self.userdata = state


def _run(coro):
    return asyncio.run(coro)


class SearchToolScenarioTests(unittest.TestCase):
    """Scenario A: NexiAgent.search_university_info behaviour."""

    def _agent(self, retriever):
        return NexiAgent(retriever, OFFLINE_SETTINGS, "web")

    def _ctx(self):
        return _FakeContext(CallState(channel="web"))

    def test_returns_evidence_text_when_found(self):
        retriever = _FakeRetriever(
            _FakeSearchResult(["Hostel fees are refundable only under special circumstances."])
        )
        agent = self._agent(retriever)
        result = _run(agent.search_university_info(self._ctx(), "hostel refund rules"))
        self.assertEqual(
            result, "Hostel fees are refundable only under special circumstances."
        )
        self.assertEqual(retriever.calls, ["hostel refund rules"])

    def test_returns_no_results_when_evidence_empty(self):
        retriever = _FakeRetriever(_FakeSearchResult(()))
        agent = self._agent(retriever)
        result = _run(agent.search_university_info(self._ctx(), "internship policy"))
        self.assertEqual(result, "NO_RESULTS")

    def test_returns_no_results_when_evidence_none(self):
        # `if not result.evidence` also covers a None evidence value.
        result = _FakeSearchResult(())
        result.evidence = None  # type: ignore[assignment]
        retriever = _FakeRetriever(result)
        agent = self._agent(retriever)
        outcome = _run(agent.search_university_info(self._ctx(), "anything"))
        self.assertEqual(outcome, "NO_RESULTS")

    def test_empty_or_whitespace_query_raises_tool_error(self):
        agent = self._agent(_FakeRetriever())
        for query in ("", "   ", "\t\n"):
            with self.assertRaises(ToolError) as caught:
                _run(agent.search_university_info(self._ctx(), query))
            self.assertEqual(
                str(caught.exception), "Ask the student to repeat the question."
            )

    def test_timeout_raises_tool_error_suggesting_retry(self):
        retriever = _FakeRetriever(side_effect=TimeoutError("search took too long"))
        agent = self._agent(retriever)
        with self.assertRaises(ToolError) as caught:
            _run(agent.search_university_info(self._ctx(), "attendance rule"))
        self.assertIn("try again", str(caught.exception))

    def test_retrieval_busy_raises_tool_error_suggesting_retry(self):
        from rag.retriever import RetrievalBusy

        retriever = _FakeRetriever(side_effect=RetrievalBusy("worker occupied"))
        agent = self._agent(retriever)
        with self.assertRaises(ToolError) as caught:
            _run(agent.search_university_info(self._ctx(), "calendar date"))
        self.assertIn("try again", str(caught.exception))


class ChannelPromptScenarioTests(unittest.TestCase):
    """Scenario B: channel-specific instructions from build_prompt."""

    def test_web_prompt_has_80_word_limit(self):
        prompt = build_prompt("web", "Asia/Kolkata")
        self.assertIn("under 80 words", prompt)
        self.assertIn("Offer more detail if the student asks.", prompt)
        self.assertNotIn("Read email addresses and phone numbers slowly", prompt)

    def test_phone_prompt_has_50_word_limit_and_slow_grouping(self):
        prompt = build_prompt("phone", "Asia/Kolkata")
        self.assertIn("under 50 words", prompt)
        self.assertIn("Read email addresses and phone numbers slowly in small groups.", prompt)
        self.assertIn("Offer to repeat them.", prompt)

    def test_date_injection_uses_todays_date(self):
        prompt = build_prompt("web", "Asia/Kolkata", today=date(2026, 11, 12))
        self.assertIn("Today's university-local date is 2026-11-12 (Asia/Kolkata).", prompt)

    def test_unsupported_channel_raises(self):
        with self.assertRaises(ValueError):
            build_prompt("sip", "Asia/Kolkata")  # type: ignore[arg-type]


class ChannelDispatchScenarioTests(unittest.TestCase):
    """Scenario C: read_channel dispatch metadata parsing."""

    def test_parses_channel_from_metadata(self):
        self.assertEqual(read_channel('{"channel": "phone"}'), ("phone", "en"))
        self.assertEqual(read_channel('{"channel": "web"}'), ("web", "en"))

    def test_language_is_parsed_when_present(self):
        self.assertEqual(read_channel('{"channel": "phone", "language": "te"}'), ("phone", "te"))
        self.assertEqual(read_channel('{"channel": "web", "language": "hi"}'), ("web", "hi"))

    def test_missing_key_falls_back_to_web(self):
        self.assertEqual(read_channel("{}"), ("web", "en"))

    def test_empty_string_falls_back_to_web(self):
        self.assertEqual(read_channel(""), ("web", "en"))

    def test_invalid_channel_falls_back_to_web(self):
        self.assertEqual(read_channel('{"channel": "sip"}'), ("web", "en"))

    def test_malformed_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            read_channel("not a json string")

    def test_non_object_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            read_channel("[1, 2, 3]")


class NoiseCancellationScenarioTests(unittest.TestCase):
    """Scenario D: noise cancellation selection by participant kind."""

    def test_sip_participant_gets_bvctelephony(self):
        profile = pick_noise_cancellation(rtc.ParticipantKind.PARTICIPANT_KIND_SIP)
        self.assertIs(type(profile), type(noise_cancellation.BVCTelephony()))

    def test_standard_web_participant_gets_bvc(self):
        profile = pick_noise_cancellation(rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD)
        self.assertIs(type(profile), type(noise_cancellation.BVC()))


if __name__ == "__main__":
    unittest.main()
