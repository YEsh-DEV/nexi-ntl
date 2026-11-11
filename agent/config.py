"""Explicit settings loading; importing this module does not load secrets or models."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parent.parent


class ConfigurationError(ValueError):
    """A configuration error whose message contains names, never secret values."""


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    agent_name: str = ""
    timezone: str = "Asia/Kolkata"
    log_level: str = "INFO"
    pdf_dir: Path = ROOT / "uni_pdfs"
    json_dir: Path = ROOT / "data/json"
    chroma_dir: Path = ROOT / "data/knowledge"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    rerank_min_score: float = 0.0
    search_timeout: float = 6.0
    retrieval_k: int = 12
    retrieval_top_n: int = 4
    context_max_chars: int = 4000
    query_max_chars: int = 1000
    inference_threads: int = 1
    stt_model: str = "nova-3"
    stt_language: str = "en-US"
    llm_model: str = "gemini-2.5-flash"
    tts_model: str = "bulbul:v2"
    tts_speaker: str = "anushka"
    web_sample_rate: int = 24000
    phone_sample_rate: int = 8000
    noise_cancellation: bool = False
    prewarm_timeout: float = 120.0
    participant_timeout: float = 30.0
    speech_timeout: float = 15.0
    shutdown_timeout: float = 10.0
    idle_processes: int = 1
    livekit_url: str = field(default="", repr=False)
    livekit_api_key: str = field(default="", repr=False)
    livekit_api_secret: str = field(default="", repr=False)
    deepgram_api_key: str = field(default="", repr=False)
    google_api_key: str = field(default="", repr=False)
    sarvam_api_key: str = field(default="", repr=False)

    def validate_runtime(self, *, console: bool) -> None:
        required = ["deepgram_api_key", "google_api_key", "sarvam_api_key"]
        if not console:
            required += ["livekit_url", "livekit_api_key", "livekit_api_secret"]
        missing = [name.upper() for name in required if not getattr(self, name)]
        if missing:
            raise ConfigurationError("Missing required settings: " + ", ".join(missing))
        if not console:
            url = urlsplit(self.livekit_url)
            if url.scheme not in {"ws", "wss"} or not url.hostname or url.username or url.password:
                raise ConfigurationError("LIVEKIT_URL must be a ws:// or wss:// server URL")
            if self.environment == "production" and url.scheme != "wss":
                raise ConfigurationError("Production LIVEKIT_URL must use wss://")


def load_settings() -> Settings:
    """Read local dotenv in development, with process environment taking precedence."""
    if os.environ.get("NEXI_ENV", "development") != "production":
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env.local", override=False)

    defaults = Settings()
    values: dict[str, object] = {}
    for name in Settings.__dataclass_fields__:
        key = "NEXI_ENV" if name == "environment" else name.upper()
        raw = os.environ.get(key)
        if raw is None and name == "sarvam_api_key":
            raw = os.environ.get("sarvam_api_key")
        if raw is None:
            continue
        raw = raw.strip()
        default = getattr(defaults, name)
        try:
            if isinstance(default, bool):
                if raw.lower() not in {"true", "false", "1", "0"}:
                    raise ValueError
                value = raw.lower() in {"true", "1"}
            elif isinstance(default, int):
                value = int(raw)
            elif isinstance(default, float):
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError
            elif isinstance(default, Path):
                if not raw:
                    raise ValueError
                path = Path(raw).expanduser()
                value = (path if path.is_absolute() else ROOT / path).resolve()
            else:
                value = raw
        except ValueError:
            raise ConfigurationError(f"Invalid setting: {key}") from None
        values[name] = value

    settings = Settings(**values)
    if settings.environment not in {"development", "production"}:
        raise ConfigurationError("NEXI_ENV must be development or production")
    if settings.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigurationError("LOG_LEVEL must be DEBUG, INFO, WARNING, or ERROR")
    try:
        ZoneInfo(settings.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ConfigurationError("TIMEZONE must name an installed IANA timezone") from None
    for name in (
        "search_timeout",
        "retrieval_k",
        "retrieval_top_n",
        "context_max_chars",
        "query_max_chars",
        "inference_threads",
        "prewarm_timeout",
        "participant_timeout",
        "speech_timeout",
        "shutdown_timeout",
    ):
        if getattr(settings, name) <= 0:
            raise ConfigurationError(f"{name.upper()} must be positive")
    if not 1 <= settings.retrieval_top_n <= settings.retrieval_k <= 50:
        raise ConfigurationError("Require 1 <= RETRIEVAL_TOP_N <= RETRIEVAL_K <= 50")
    if not 1000 <= settings.context_max_chars <= 16000:
        raise ConfigurationError("CONTEXT_MAX_CHARS must be between 1000 and 16000")
    if settings.query_max_chars > 4000 or settings.inference_threads > 8:
        raise ConfigurationError("QUERY_MAX_CHARS <= 4000 and INFERENCE_THREADS <= 8 required")
    if not 0 <= settings.idle_processes <= 4:
        raise ConfigurationError("IDLE_PROCESSES must be between 0 and 4")
    for name in (
        "embed_model",
        "embed_revision",
        "rerank_model",
        "rerank_revision",
        "stt_model",
        "stt_language",
        "llm_model",
        "tts_model",
        "tts_speaker",
    ):
        if not getattr(settings, name):
            raise ConfigurationError(f"{name.upper()} cannot be empty")
    for rate in (settings.web_sample_rate, settings.phone_sample_rate):
        if rate not in {8000, 16000, 22050, 24000}:
            raise ConfigurationError("TTS sample rate must be 8000, 16000, 22050, or 24000")
    return settings
