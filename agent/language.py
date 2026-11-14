"""Single source of truth for multilingual phrases, speaker catalog, and language state."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Literal

from livekit.plugins import deepgram

from agent.config import Settings
from agent.providers.sarvam_adapter import NexiSarvamTTS
from agent.telemetry import event

LOGGER = logging.getLogger("agent.language")

Language = Literal["en", "te", "hi"]
DEFAULT_LANGUAGE: Language = "en"

# Catalog of verified models and speakers
LANGUAGE_CATALOG: dict[Language, dict[str, str]] = {
    "en": {
        "name": "English",
        "stt_code": "en-US",
        "tts_code": "en-IN",
        "tts_speaker": "kavya",
    },
    "te": {
        "name": "Telugu",
        "stt_code": "te",
        "tts_code": "te-IN",
        "tts_speaker": "kavya",
    },
    "hi": {
        "name": "Hindi",
        "stt_code": "hi",
        "tts_code": "hi-IN",
        "tts_speaker": "kavya",
    },
}

# Native phrase table for all system announcements and agent turns
PHRASES: dict[str, dict[Language, str]] = {
    "greeting_web": {
        "en": "Hello! I am Nexi, your SRM University-AP voice assistant. How can I help you today?",
        "te": "నమస్కారం! నేను నెక్సి, ఎస్ఆర్ఎం యూనివర్సిటీ-ఏపీ వాయిస్ అసిస్టెంట్. నేను మీకు ఎలా సహాయపడగలను?",
        "hi": "नमस्ते! मैं नेक्सी हूँ, एसआरएम यूनिवर्सिटी-एपी का वॉइस असिस्टेंट। मैं आज आपकी क्या सहायता कर सकता हूँ?",
    },
    "greeting_phone": {
        "en": "Hello, thank you for calling SRM University-AP. I am Nexi, an AI assistant. How can I help you?",
        "te": "నమస్కారం, ఎస్ఆర్ఎం యూనివర్సిటీ-ఏపీ కి కాల్ చేసినందుకు ధన్యవాదాలు. నేను నెక్సిని. నేను మీకు ఎలా సహాయపడగలను?",
        "hi": "नमस्ते, एसआरएम यूनिवर्सिटी-एपी में कॉल करने के लिए धन्यवाद। मैं नेक्सी हूँ। मैं आपकी क्या मदद कर सकता हूँ?",
    },
    "menu_prompt": {
        "en": "For English press 1. For Telugu press 2. For Hindi press 3.",
        "te": "ఇంగ్లీష్ కోసం 1, తెలుగు కోసం 2, హిందీ కోసం 3 నొక్కండి.",
        "hi": "अंग्रेजी के लिए 1, तेलुगु के लिए 2, हिंदी के लिए 3 दबाएं।",
    },
    "menu_confirm": {
        "en": "You have selected English. How can I help you?",
        "te": "మీరు తెలుగును ఎంచుకున్నారు. నేను మీకు ఎలా సహాయపడగలను?",
        "hi": "आपने हिंदी चुनी है। मैं आपकी क्या मदद कर सकता हूँ?",
    },
    "farewell": {
        "en": "Thank you for reaching out to SRM University-AP. Have a wonderful day! Goodbye.",
        "te": "ఎస్ఆర్ఎం యూనివర్సిటీ-ఏపీని సంప్రదించినందుకు ధన్యవాదాలు. మంచి రోజు కావాలని కోరుకుంటున్నాను! వీడ్కోలు.",
        "hi": "एसआरएम यूनिवर्सिटी-एपी से संपर्क करने के लिए धन्यवाद। आपका दिन शुभ हो! अलविदा।",
    },
    "duration_warning": {
        "en": "Notice: This call will reach its maximum allowed duration in thirty seconds.",
        "te": "గమనిక: ఈ కాల్ ముప్పై సెకన్లలో గరిష్ట సమయ పరిమితిని చేరుకుంటుంది.",
        "hi": "सूचना: यह कॉल तीस सेकंड में अपनी अधिकतम समय सीमा तक पहुँच जाएगी।",
    },
    "idle_prompt": {
        "en": "Are you still there? Please let me know if you have any questions.",
        "te": "మీరు ఇంకా లైన్ లో ఉన్నారా? మీకు ఏవైనా ప్రశ్నలు ఉంటే దయచేసి చెప్పండి.",
        "hi": "क्या आप अभी भी सुन रहे हैं? कृपया मुझे बताएं यदि आपका कोई प्रश्न है।",
    },
    "transfer_announce": {
        "en": "Transferring your call to the requested university office now. Please stay on the line.",
        "te": "మీ కాల్‌ను సంబంధిత విశ్వవిద్యాలయ కార్యాలయానికి బదిలీ చేస్తున్నాను. దయచేసి లైన్‌లో ఉండండి.",
        "hi": "आपकी कॉल को विश्वविद्यालय कार्यालय में स्थानांतरित किया जा रहा है। कृपया लाइन पर बने रहें।",
    },
    "transfer_failed": {
        "en": "I'm sorry, the office transfer could not be completed right now. How else may I assist you?",
        "te": "క్షమించండి, ప్రస్తుతం కార్యాలయానికి కాల్ బదిలీ సాధ్యం కాలేదు. నేను మీకు ఇతర విధాలుగా ఎలా సహాయపడగలను?",
        "hi": "क्षमा करें, इस समय कार्यालय स्थानांतरण पूरा नहीं हो सका। मैं आपकी और क्या सहायता कर सकता हूँ?",
    },
    "switch_confirm": {
        "en": "Switched to English. How can I assist you now?",
        "te": "తెలుగు భాషకు మార్చబడింది. నేను ఇప్పుడు మీకు ఎలా సహాయపడగలను?",
        "hi": "हिंदी भाषा में बदल दिया गया है। अब मैं आपकी क्या सहायता कर सकता हूँ?",
    },
    "switch_refused": {
        "en": "You have reached the maximum number of language switches for this call.",
        "te": "ఈ కాల్ కోసం మీరు గరిష్ట భాషా మార్పుల పరిమితిని చేరుకున్నారు.",
        "hi": "आप इस कॉल के लिए अधिकतम भाषा परिवर्तन सीमा तक पहुँच चुके हैं।",
    },
    "technical_trouble": {
        "en": "Sorry, I am having technical difficulty. Please contact the university office directly.",
        "te": "క్షమించండి, సాంకేతిక సమస్య ఎదురైంది. దయచేసి విశ్వవిద్యాలయ కార్యాలయాన్ని నేరుగా సంప్రదించండి.",
        "hi": "क्षमा करें, मुझे तकनीकी समस्या आ रही है। कृपया सीधे विश्वविद्यालय कार्यालय से संपर्क करें।",
    },
}

# Telugu and Hindi strings requiring human native-speaker review per acceptance A2.2
NEEDS_NATIVE_REVIEW: list[str] = [
    f"{key}.{lang}"
    for key, per_lang in PHRASES.items()
    for lang in ("te", "hi")
    if lang in per_lang
]


@dataclass
class LanguageState:
    current: Language = DEFAULT_LANGUAGE
    switch_count: int = 0
    switches: list[tuple[Language, Language, float]] = field(default_factory=list)


class MenuState(str, Enum):
    MENU = "menu"
    SELECTED = "selected"


class PhoneLanguageMenu:
    """State machine for inbound telephony language selection via DTMF or speech."""

    def __init__(
        self,
        room: Any,
        session: Any,
        state: Any,
        settings: Settings,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
        transfer_fn: Callable[[str], Coroutine[Any, Any, bool]] | None = None,
    ) -> None:
        self.room = room
        self.session = session
        self.state = state
        self.settings = settings
        self.clock = clock
        self.sleep_fn = sleep_fn
        self.transfer_fn = transfer_fn

        self.menu_state = MenuState.MENU
        self.selected_language: Language = DEFAULT_LANGUAGE
        self._last_digit_time: float = 0.0
        self._selection_future: asyncio.Future[Language] | None = None

    def on_dtmf_received(self, ev: Any) -> None:
        """Handle incoming SIP DTMF digit with 300ms debouncing."""
        digit = getattr(ev, "digit", "")
        now = self.clock()
        if (now - self._last_digit_time) < 0.3:
            return  # Debounce repeat within 300ms
        self._last_digit_time = now

        if self.menu_state != MenuState.MENU:
            return

        if digit == "1":
            self._apply_selection("en")
        elif digit == "2":
            self._apply_selection("te")
        elif digit == "3":
            self._apply_selection("hi")
        elif digit == "0" and "general" in self.settings.transfer_targets:
            if self.transfer_fn is not None:
                asyncio.create_task(self.transfer_fn("general"))

    def _apply_selection(self, lang: Language) -> None:
        self.menu_state = MenuState.SELECTED
        self.selected_language = lang
        setattr(self.state, "language", lang)
        if self._selection_future is not None and not self._selection_future.done():
            self._selection_future.set_result(lang)

    async def run_menu(self) -> Language:
        """Execute the language selection menu loop."""
        channel = getattr(self.state, "channel", "web")
        if channel != "phone":
            self.menu_state = MenuState.SELECTED
            return getattr(self.state, "language", DEFAULT_LANGUAGE)

        loop = asyncio.get_running_loop()
        self._selection_future = loop.create_future()

        # Register DTMF listener on room if room supports events
        if self.room is not None and hasattr(self.room, "on"):
            self.room.on("sip_dtmf_received", self.on_dtmf_received)

        menu_text = PHRASES["menu_prompt"]["en"]
        timeout_s = self.settings.language_menu_timeout_s

        try:
            # First attempt
            if self.session is not None and hasattr(self.session, "say"):
                await self.session.say(menu_text)

            done, _ = await asyncio.wait([self._selection_future], timeout=timeout_s)
            if self._selection_future in done:
                return self._selection_future.result()

            # Second attempt: repeat once
            if self.menu_state == MenuState.MENU:
                if self.session is not None and hasattr(self.session, "say"):
                    await self.session.say(menu_text)
                done, _ = await asyncio.wait([self._selection_future], timeout=timeout_s)
                if self._selection_future in done:
                    return self._selection_future.result()
                else:
                    # Timeout reached after repeat: default to English
                    self._apply_selection(DEFAULT_LANGUAGE)
                    return DEFAULT_LANGUAGE

            return self.selected_language
        finally:
            self.menu_state = MenuState.SELECTED
            if self.room is not None and hasattr(self.room, "off"):
                self.room.off("sip_dtmf_received", self.on_dtmf_received)
            # Confirm selection in chosen language
            confirm_text = PHRASES["menu_confirm"].get(self.selected_language, PHRASES["menu_confirm"]["en"])
            if self.session is not None and hasattr(self.session, "say"):
                try:
                    await self.session.say(confirm_text)
                except Exception as exc:
                    LOGGER.debug("Non-fatal speech confirmation error in language menu: %s", exc)


GLOBAL_SWITCH_LOCK = asyncio.Lock()


async def switch_call_language(
    session: Any,
    state: Any,
    settings: Settings,
    retriever: Any,
    target_language: str,
    lock: asyncio.Lock | None = None,
) -> tuple[bool, str]:
    """Execute mid-call serialized language handover while preserving session history."""
    if target_language not in ("en", "te", "hi"):
        return False, f"Language '{target_language}' not supported. Allowed: en, te, hi."

    current_lang: Language = getattr(state, "language", DEFAULT_LANGUAGE)
    if target_language == current_lang:
        return True, f"Conversation is already in {target_language}."

    switches = getattr(state, "language_switches", 0)
    if switches >= settings.max_language_switches:
        refuse_msg = PHRASES["switch_refused"].get(current_lang, PHRASES["switch_refused"]["en"])
        if session is not None and hasattr(session, "say"):
            await session.say(refuse_msg)
        return False, "Maximum allowed language switches reached for this call."

    chosen_lock = lock or GLOBAL_SWITCH_LOCK
    async with chosen_lock:
        old_lang = current_lang
        target_typed: Language = target_language  # type: ignore

        try:
            # 1. Wait for any active speech to finish
            if session is not None and hasattr(session, "current_speech") and session.current_speech is not None:
                await session.current_speech.wait_for_complete()

            # 2. Re-configure STT and TTS providers for the target language
            stt_lang = {"en": settings.stt_language, "te": "te", "hi": "hi"}.get(
                target_typed, settings.stt_language
            )
            stt = deepgram.STT(
                model=settings.stt_model,
                language=stt_lang,
                api_key=settings.deepgram_api_key,
                interim_results=True,
                punctuate=True,
            )
            tts_lang = "te-IN" if target_typed == "te" else ("hi-IN" if target_typed == "hi" else "en-IN")
            rate = settings.phone_sample_rate if getattr(state, "channel", "web") == "phone" else settings.web_sample_rate
            tts = NexiSarvamTTS(
                target_language_code=tts_lang,
                model=settings.tts_model,
                speaker=settings.tts_speaker,
                api_key=settings.sarvam_api_key,
                speech_sample_rate=rate,
            )

            # 3. Create replacement agent subclass with new instructions and preserved chat context
            from agent.assistant import NexiAgent

            new_agent = NexiAgent(
                retriever=retriever,
                settings=settings,
                channel=getattr(state, "channel", "web"),
                language=target_typed,
                stt=stt,
                tts=tts,
            )

            # 4. Perform LiveKit agent handover
            if session is not None:
                if hasattr(session, "_stt"):
                    session._stt = stt
                if hasattr(session, "_tts"):
                    session._tts = tts
                if hasattr(session, "update_agent"):
                    session.update_agent(new_agent)

            setattr(state, "language", target_typed)
            setattr(state, "language_switches", switches + 1)
            event(
                "language_switched",
                call_id=getattr(state, "call_id", ""),
                from_lang=old_lang,
                to_lang=target_typed,
                switch_count=switches + 1,
            )

            # 5. Announce successful switch in the NEW language
            confirm_msg = PHRASES["switch_confirm"].get(target_typed, PHRASES["switch_confirm"]["en"])
            if session is not None and hasattr(session, "say"):
                await session.say(confirm_msg)

            return True, f"Successfully switched to {target_typed}."

        except Exception as exc:
            setattr(
                state,
                "language_switch_failures",
                getattr(state, "language_switch_failures", 0) + 1,
            )
            LOGGER.warning(
                "Language switch failed from %s to %s: %s",
                old_lang,
                target_language,
                type(exc).__name__,
                exc_info=True,
            )
            # Rollback to original language
            setattr(state, "language", old_lang)
            error_msg = PHRASES["technical_trouble"].get(old_lang, PHRASES["technical_trouble"]["en"])
            if session is not None and hasattr(session, "say"):
                try:
                    await session.say(error_msg)
                except Exception as say_exc:
                    LOGGER.debug("Failed speaking error message after language switch error: %s", say_exc)
            return False, f"Language switch error: {type(exc).__name__}"
