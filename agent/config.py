"""Explicit settings loading; importing this module does not load secrets or models."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger("agent.config")


class ConfigurationError(ValueError):
    """A configuration error whose message contains names, never secret values."""


# Master registry of every environment variable read by Nexi.
# Required for consistency tests with .env.example and README.
ENV_VARS: dict[str, dict[str, object]] = {
    # LiveKit Connection & Dispatch
    "LIVEKIT_URL": {
        "description": "LiveKit Cloud or self-hosted server URL (ws:// or wss://)",
        "default": "",
        "required": True,
    },
    "LIVEKIT_API_KEY": {
        "description": "API Key for LiveKit server authentication",
        "default": "",
        "required": True,
    },
    "LIVEKIT_API_SECRET": {
        "description": "API Secret for LiveKit server authentication",
        "default": "",
        "required": True,
    },
    "AGENT_NAME": {
        "description": "Named dispatch agent identifier matching SIP rules and tokens",
        "default": "nexi",
        "required": True,
    },
    "ALLOW_AUTO_DISPATCH": {
        "description": "Allow running without named dispatch (development only)",
        "default": "0",
        "required": False,
    },
    # AI Providers
    "GOOGLE_API_KEY": {
        "description": "Google Gemini API key for LLM responses",
        "default": "",
        "required": True,
    },
    "DEEPGRAM_API_KEY": {
        "description": "Deepgram API key for English speech-to-text",
        "default": "",
        "required": True,
    },
    "SARVAM_API_KEY": {
        "description": "Sarvam AI API key for Bulbul TTS and Indian language STT",
        "default": "",
        "required": True,
    },
    "ALLOW_UNTESTED_PROVIDER_VERSIONS": {
        "description": "Bypass provider plugin version check (development only)",
        "default": "0",
        "required": False,
    },
    # Agent Runtime & Models
    "ENVIRONMENT": {
        "description": "Environment tier ('development' or 'production')",
        "default": "development",
        "required": False,
    },
    "NEXI_ENV": {
        "description": "Alias for ENVIRONMENT",
        "default": "development",
        "required": False,
    },
    "LOG_LEVEL": {
        "description": "Logging verbosity (DEBUG, INFO, WARNING, ERROR)",
        "default": "INFO",
        "required": False,
    },
    "TIMEZONE": {
        "description": "Campus IANA timezone for date/time prompts",
        "default": "Asia/Kolkata",
        "required": False,
    },
    "LLM_MODEL": {
        "description": "Google Gemini model name",
        "default": "gemini-3.5-flash",
        "required": False,
    },
    "STT_MODEL": {
        "description": "Deepgram STT model name",
        "default": "nova-3",
        "required": False,
    },
    "STT_LANGUAGE": {
        "description": "Default STT language code for English",
        "default": "en-US",
        "required": False,
    },
    "TTS_MODEL": {
        "description": "Sarvam TTS model name (e.g. bulbul:v3)",
        "default": "bulbul:v3",
        "required": False,
    },
    "TTS_SPEAKER": {
        "description": "Sarvam TTS speaker voice name",
        "default": "kavya",
        "required": False,
    },
    "DEFAULT_LANGUAGE": {
        "description": "Default spoken language ('en', 'te', 'hi')",
        "default": "en",
        "required": False,
    },
    "WEB_SAMPLE_RATE": {
        "description": "Audio sample rate for web clients in Hz",
        "default": "24000",
        "required": False,
    },
    "PHONE_SAMPLE_RATE": {
        "description": "Audio sample rate for telephony SIP callers in Hz",
        "default": "8000",
        "required": False,
    },
    "NOISE_CANCELLATION": {
        "description": "Enable noise cancellation filter (BVC/BVCTelephony)",
        "default": "false",
        "required": False,
    },
    "IDLE_PROCESSES": {
        "description": "Pre-warmed idle worker processes",
        "default": "1",
        "required": False,
    },
    # Call Lifecycle & Timeouts
    "MAX_CALL_SECONDS_WEB": {
        "description": "Hard maximum call duration for web sessions in seconds",
        "default": "300.0",
        "required": False,
    },
    "MAX_CALL_SECONDS_PHONE": {
        "description": "Hard maximum call duration for phone calls in seconds",
        "default": "600.0",
        "required": False,
    },
    "CALL_WARNING_LEAD_S": {
        "description": "Lead time in seconds to announce call duration expiration",
        "default": "30.0",
        "required": False,
    },
    "IDLE_SILENCE_TIMEOUT": {
        "description": "Seconds of silence before prompting caller for activity",
        "default": "8.0",
        "required": False,
    },
    "IDLE_SILENCE_GRACE": {
        "description": "Seconds of silence after prompt before disconnecting call",
        "default": "8.0",
        "required": False,
    },
    "PREWARM_TIMEOUT": {
        "description": "Timeout in seconds for prewarming models and index",
        "default": "120.0",
        "required": False,
    },
    "PARTICIPANT_TIMEOUT": {
        "description": "Timeout in seconds waiting for participant to connect",
        "default": "30.0",
        "required": False,
    },
    "SPEECH_TIMEOUT": {
        "description": "Timeout in seconds for TTS generation and playback",
        "default": "15.0",
        "required": False,
    },
    "SHUTDOWN_TIMEOUT": {
        "description": "Timeout in seconds for process shutdown grace period",
        "default": "10.0",
        "required": False,
    },
    "FAREWELL_TIMEOUT_S": {
        "description": "Timeout in seconds for speaking farewell message",
        "default": "8.0",
        "required": False,
    },
    "SESSION_CLOSE_TIMEOUT_S": {
        "description": "Timeout in seconds for closing LiveKit AgentSession",
        "default": "5.0",
        "required": False,
    },
    "ROOM_DELETE_TIMEOUT_S": {
        "description": "Timeout in seconds for LiveKit API delete_room request",
        "default": "5.0",
        "required": False,
    },
    "TELEMETRY_DRAIN_TIMEOUT_S": {
        "description": "Timeout in seconds to drain queued telemetry events",
        "default": "5.0",
        "required": False,
    },
    "END_CALL_TOTAL_TIMEOUT_S": {
        "description": "Hard total timeout in seconds for complete teardown sequence",
        "default": "25.0",
        "required": False,
    },
    # Telephony & Office Transfer
    "TRANSFER_TARGETS": {
        "description": "JSON object mapping department names to allowlisted E.164 phone numbers",
        "default": "",
        "required": False,
    },
    "TRANSFER_TIMEOUT_S": {
        "description": "Timeout in seconds for SIP REFER transfer request",
        "default": "10.0",
        "required": False,
    },
    "TRANSFER_GRACE_S": {
        "description": "Extra grace seconds allowed if transfer is active at duration limit",
        "default": "15.0",
        "required": False,
    },
    "LANGUAGE_MENU_TIMEOUT_S": {
        "description": "Inbound phone language menu selection timeout in seconds",
        "default": "8.0",
        "required": False,
    },
    "MAX_LANGUAGE_SWITCHES": {
        "description": "Maximum allowed language switches per call",
        "default": "5",
        "required": False,
    },
    # Knowledge & Retrieval
    "PDF_DIR": {
        "description": "Directory holding university source PDFs",
        "default": "uni_pdfs",
        "required": False,
    },
    "JSON_DIR": {
        "description": "Directory holding university structured JSON records",
        "default": "data/json",
        "required": False,
    },
    "CHROMA_DIR": {
        "description": "Directory holding versioned Chroma knowledge artifacts",
        "default": "data/knowledge",
        "required": False,
    },
    "EMBED_MODEL": {
        "description": "Sentence Transformer model name",
        "default": "sentence-transformers/all-MiniLM-L6-v2",
        "required": False,
    },
    "EMBED_REVISION": {
        "description": "Pinned HuggingFace revision commit for embedding model",
        "default": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "required": False,
    },
    "RERANK_MODEL": {
        "description": "Cross-Encoder reranker model name",
        "default": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "required": False,
    },
    "RERANK_REVISION": {
        "description": "Pinned HuggingFace revision commit for reranker model",
        "default": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "required": False,
    },
    "RERANK_MIN_SCORE": {
        "description": "Relevance threshold for accepting reranked evidence",
        "default": "0.0",
        "required": False,
    },
    "SEARCH_TIMEOUT": {
        "description": "Timeout in seconds for retrieval queries",
        "default": "6.0",
        "required": False,
    },
    "RETRIEVAL_K": {
        "description": "Candidate chunk count for dense and BM25 retrieval",
        "default": "12",
        "required": False,
    },
    "RETRIEVAL_TOP_N": {
        "description": "Final context chunk count passed to LLM",
        "default": "4",
        "required": False,
    },
    "CONTEXT_MAX_CHARS": {
        "description": "Maximum character budget for retrieved context",
        "default": "4000",
        "required": False,
    },
    "QUERY_MAX_CHARS": {
        "description": "Maximum character length of student query",
        "default": "1000",
        "required": False,
    },
    "INFERENCE_THREADS": {
        "description": "CPU thread limit for local neural inference",
        "default": "1",
        "required": False,
    },
    "ALLOW_UNVERIFIED_KNOWLEDGE": {
        "description": "Allow loading unverified knowledge index (development only)",
        "default": "0",
        "required": False,
    },
    # Operational Telemetry & Privacy
    "TELEMETRY_QUEUE_SIZE": {
        "description": "Maximum in-memory bounded queue size for telemetry events",
        "default": "1000",
        "required": False,
    },
    "TELEMETRY_SPILL_PATH": {
        "description": "File path for critical event spillover when queue is saturated",
        "default": "",
        "required": False,
    },
    "PII_HASH_KEY": {
        "description": "Secret key for HMAC-SHA256 caller number hashing (required in prod)",
        "default": "",
        "required": False,
    },
    "PII_EXTRA_PATTERNS": {
        "description": "JSON list of custom regex patterns to redact from questions",
        "default": "",
        "required": False,
    },
    "CALL_LOG_RETENTION_DAYS": {
        "description": "Retention window in days for call_logs table",
        "default": "90",
        "required": False,
    },
    "TURN_METRICS_RETENTION_DAYS": {
        "description": "Retention window in days for stage_metrics table",
        "default": "30",
        "required": False,
    },
    "KNOWLEDGE_GAP_RETENTION_DAYS": {
        "description": "Retention window in days for resolved knowledge gaps",
        "default": "180",
        "required": False,
    },
    # Token Service & Admission Control
    "HOST": {
        "description": "Bind host IP for Token and Health HTTP API",
        "default": "127.0.0.1",
        "required": False,
    },
    "PORT": {
        "description": "Bind port for Token and Health HTTP API",
        "default": "8080",
        "required": False,
    },
    "ALLOWED_ORIGINS": {
        "description": "Comma-separated list of allowed CORS origin URLs",
        "default": "http://localhost:8080,http://127.0.0.1:8080,http://localhost:3000,http://127.0.0.1:3000",
        "required": False,
    },
    "TOKEN_TTL_S": {
        "description": "TTL duration for issued LiveKit room tokens in seconds",
        "default": "600.0",
        "required": False,
    },
    "TOKEN_RATE_PER_IP": {
        "description": "Maximum token generation requests per client IP per window",
        "default": "10",
        "required": False,
    },
    "TOKEN_RATE_WINDOW_S": {
        "description": "Sliding rate limit window in seconds",
        "default": "600.0",
        "required": False,
    },
    "TOKEN_RATE_GLOBAL_PER_MIN": {
        "description": "Maximum global token generation requests per minute",
        "default": "60",
        "required": False,
    },
    "TRUSTED_PROXIES": {
        "description": "Comma-separated list of trusted reverse proxy IP addresses",
        "default": "",
        "required": False,
    },
    "MAX_CONCURRENT_CALLS": {
        "description": "Maximum concurrent active call sessions allowed",
        "default": "5",
        "required": False,
    },
    "MAX_TOKENS_PER_DAY": {
        "description": "Maximum tokens allowed to be issued per UTC day",
        "default": "300",
        "required": False,
    },
    "ADMISSION_FAIL_OPEN": {
        "description": "Allow token issuance when LiveKit capacity check fails (dev only)",
        "default": "false",
        "required": False,
    },
    "WORKER_HEALTH_URL": {
        "description": "URL to probe LiveKit agent worker health",
        "default": "http://127.0.0.1:8081/healthz",
        "required": False,
    },
    "REQUIRE_CAPTCHA": {
        "description": "Require captcha verification proof before token issuance",
        "default": "false",
        "required": False,
    },
}


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    agent_name: str = "nexi"
    allow_auto_dispatch: bool = False
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
    allow_unverified_knowledge: bool = False
    stt_model: str = "nova-3"
    stt_language: str = "en-US"
    llm_model: str = "gemini-3.5-flash"
    tts_model: str = "bulbul:v3"
    tts_speaker: str = "kavya"
    default_language: str = "en"
    allow_untested_provider_versions: bool = False
    max_call_seconds_web: float = 300.0
    max_call_seconds_phone: float = 600.0
    call_warning_lead_s: float = 30.0
    idle_silence_timeout: float = 8.0
    idle_silence_grace: float = 8.0
    web_sample_rate: int = 24000
    phone_sample_rate: int = 8000
    noise_cancellation: bool = False
    prewarm_timeout: float = 120.0
    participant_timeout: float = 30.0
    speech_timeout: float = 15.0
    shutdown_timeout: float = 10.0
    idle_processes: int = 1
    farewell_timeout_s: float = 8.0
    session_close_timeout_s: float = 5.0
    room_delete_timeout_s: float = 5.0
    telemetry_drain_timeout_s: float = 5.0
    end_call_total_timeout_s: float = 25.0
    transfer_targets: dict[str, str] = field(default_factory=dict)
    transfer_timeout_s: float = 10.0
    transfer_grace_s: float = 15.0
    language_menu_timeout_s: float = 8.0
    max_language_switches: int = 5
    telemetry_queue_size: int = 1000
    telemetry_spill_path: Path | None = None
    pii_hash_key: str = field(default="", repr=False)
    pii_extra_patterns: list[str] = field(default_factory=list)
    call_log_retention_days: int = 90
    turn_metrics_retention_days: int = 30
    knowledge_gap_retention_days: int = 180
    host: str = "127.0.0.1"
    port: int = 8080
    allowed_origins: tuple[str, ...] = (
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )
    token_ttl_s: float = 600.0
    token_rate_per_ip: int = 10
    token_rate_window_s: float = 600.0
    token_rate_global_per_min: int = 60
    trusted_proxies: tuple[str, ...] = ()
    max_concurrent_calls: int = 5
    max_tokens_per_day: int = 300
    admission_fail_open: bool = False
    worker_health_url: str = "http://127.0.0.1:8081/healthz"
    require_captcha: bool = False
    livekit_url: str = field(default="", repr=False)
    livekit_api_key: str = field(default="", repr=False)
    livekit_api_secret: str = field(default="", repr=False)
    deepgram_api_key: str = field(default="", repr=False)
    google_api_key: str = field(default="", repr=False)
    sarvam_api_key: str = field(default="", repr=False)

    def validate_runtime(self, *, console: bool) -> None:
        if not self.agent_name:
            if not self.allow_auto_dispatch:
                raise ConfigurationError(
                    "AGENT_NAME cannot be empty. Named dispatch is required for LiveKit "
                    "telephony routing and token issuance. Set AGENT_NAME=nexi or set "
                    "ALLOW_AUTO_DISPATCH=1 for dev auto-dispatch."
                )
            LOGGER.warning(
                "ALLOW_AUTO_DISPATCH is enabled: running without named dispatch (development only)"
            )

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

        if not (60.0 <= self.token_ttl_s <= 3600.0):
            raise ConfigurationError("TOKEN_TTL_S must be between 60 and 3600 seconds")

        if self.environment == "production":
            if not self.pii_hash_key:
                raise ConfigurationError("PII_HASH_KEY is required in production environment")
            if "*" in self.allowed_origins:
                raise ConfigurationError("ALLOWED_ORIGINS cannot contain wildcards in production")


def _parse_bool(raw: str) -> bool:
    low = raw.strip().lower()
    if low in {"true", "1", "yes", "on"}:
        return True
    if low in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw}")


def load_settings() -> Settings:
    """Read local dotenv in development, with process environment taking precedence."""
    env_name = os.environ.get("ENVIRONMENT") or os.environ.get("NEXI_ENV", "development")
    if env_name != "production":
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env.local", override=False)

    defaults = Settings()
    values: dict[str, object] = {}

    for name in Settings.__dataclass_fields__:
        # Handle special key mappings
        if name == "environment":
            key = "ENVIRONMENT" if "ENVIRONMENT" in os.environ else "NEXI_ENV"
        else:
            key = name.upper()

        raw = os.environ.get(key)
        if raw is None and name == "sarvam_api_key":
            raw = os.environ.get("sarvam_api_key")
        if raw is None:
            continue
        raw = raw.strip()
        default = getattr(defaults, name)

        try:
            if isinstance(default, bool):
                values[name] = _parse_bool(raw)
            elif isinstance(default, int):
                values[name] = int(raw)
            elif isinstance(default, float):
                val_f = float(raw)
                if not math.isfinite(val_f):
                    raise ValueError
                values[name] = val_f
            elif isinstance(default, Path):
                if not raw:
                    raise ValueError
                path = Path(raw).expanduser()
                values[name] = (path if path.is_absolute() else ROOT / path).resolve()
            elif name == "telemetry_spill_path":
                if raw:
                    path = Path(raw).expanduser()
                    values[name] = (path if path.is_absolute() else ROOT / path).resolve()
            elif name == "transfer_targets":
                if raw:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("TRANSFER_TARGETS must be a JSON object")
                    target_map: dict[str, str] = {}
                    for k, v in parsed.items():
                        num = str(v).strip()
                        if not re.match(r"^\+[1-9]\d{7,14}$", num):
                            raise ConfigurationError(
                                f"TRANSFER_TARGETS number for '{k}' must match E.164: '{num}'"
                            )
                        target_map[str(k).strip().lower()] = num
                    values[name] = target_map
            elif name == "pii_extra_patterns":
                if raw:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, list):
                        raise ValueError("PII_EXTRA_PATTERNS must be a JSON list of strings")
                    values[name] = [str(item) for item in parsed]
            elif name in {"allowed_origins", "trusted_proxies"}:
                values[name] = tuple(part.strip() for part in raw.split(",") if part.strip())
            else:
                values[name] = raw
        except (ValueError, json.JSONDecodeError):
            raise ConfigurationError(f"Invalid setting: {key}") from None

    settings = Settings(**values)

    if settings.environment not in {"development", "production"}:
        raise ConfigurationError("ENVIRONMENT/NEXI_ENV must be development or production")
    if settings.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigurationError("LOG_LEVEL must be DEBUG, INFO, WARNING, or ERROR")
    try:
        ZoneInfo(settings.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ConfigurationError("TIMEZONE must name an installed IANA timezone") from None

    # Bounds validation
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
        "idle_silence_timeout",
        "idle_silence_grace",
        "max_call_seconds_web",
        "max_call_seconds_phone",
        "farewell_timeout_s",
        "session_close_timeout_s",
        "room_delete_timeout_s",
        "telemetry_drain_timeout_s",
        "end_call_total_timeout_s",
        "token_ttl_s",
        "token_rate_window_s",
    ):
        if getattr(settings, name) <= 0:
            raise ConfigurationError(f"{name.upper()} must be positive")

    if not 30 <= settings.max_call_seconds_web <= 3600:
        raise ConfigurationError("MAX_CALL_SECONDS_WEB must be between 30 and 3600")
    if not 30 <= settings.max_call_seconds_phone <= 3600:
        raise ConfigurationError("MAX_CALL_SECONDS_PHONE must be between 30 and 3600")
    if settings.call_warning_lead_s >= min(settings.max_call_seconds_web, settings.max_call_seconds_phone):
        raise ConfigurationError("CALL_WARNING_LEAD_S must be smaller than the minimum max call limit")
    if not 60 <= settings.token_ttl_s <= 3600:
        raise ConfigurationError("TOKEN_TTL_S must be between 60 and 3600")
    if not 1 <= settings.retrieval_top_n <= settings.retrieval_k <= 50:
        raise ConfigurationError("Require 1 <= RETRIEVAL_TOP_N <= RETRIEVAL_K <= 50")
    if not 1000 <= settings.context_max_chars <= 16000:
        raise ConfigurationError("CONTEXT_MAX_CHARS must be between 1000 and 16000")
    if settings.query_max_chars > 4000 or settings.inference_threads > 8:
        raise ConfigurationError("QUERY_MAX_CHARS <= 4000 and INFERENCE_THREADS <= 8 required")
    if not 0 <= settings.idle_processes <= 4:
        raise ConfigurationError("IDLE_PROCESSES must be between 0 and 4")
    if not 1 <= settings.max_language_switches <= 20:
        raise ConfigurationError("MAX_LANGUAGE_SWITCHES must be between 1 and 20")

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
    if settings.default_language not in {"en", "te", "hi"}:
        raise ConfigurationError("DEFAULT_LANGUAGE must be en, te, or hi")

    return settings
