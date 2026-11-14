"""Token generation, rate limiting, admission control, and health API service."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from uuid import uuid4

from livekit import api

from agent.admission import AdmissionController
from agent.config import ConfigurationError, Settings, load_settings
from agent.ratelimit import RateLimiter
from agent.readiness import ReadinessChecker
from agent.storage import TelemetryStore, compute_caller_hash
from rag.manifest import read_active

ROOT = Path(__file__).resolve().parent.parent
MAX_BODY_BYTES = 16 * 1024  # 16 KB bound on JSON request body
LOGGER = logging.getLogger("nexi.token_service")


def generate_token(
    settings: Settings,
    channel: str = "web",
    language: str = "en",
    identity: str | None = None,
    name: str | None = None,
) -> dict[str, object]:
    """Create an ephemeral LiveKit room token authorizing agent dispatch."""
    if not settings.livekit_url or not settings.livekit_api_key or not settings.livekit_api_secret:
        raise ValueError("LiveKit credentials not configured")
    if channel not in ("web", "phone"):
        raise ValueError("Channel must be 'web' or 'phone'")
    if language not in ("en", "te", "hi"):
        raise ValueError("Language must be 'en', 'te', or 'hi'")

    user_identity = identity or f"student-{uuid4().hex[:8]}"
    display_name = name or (
        "Student" if language == "en" else "విద్యార్థి" if language == "te" else "छात्र"
    )
    room_name = f"web-{uuid4().hex[:12]}"
    dispatch_agent = settings.agent_name or "nexi"

    dispatch_meta = json.dumps({"channel": channel, "language": language}, ensure_ascii=False)
    room_config = api.RoomConfiguration(
        name=room_name,
        empty_timeout=300,
        departure_timeout=30,
        max_participants=2,
        agents=[api.RoomAgentDispatch(agent_name=dispatch_agent, metadata=dispatch_meta)],
    )

    ttl = timedelta(seconds=settings.token_ttl_s)
    # Restrict grants strictly: room join, publish microphone only, data channel, subscribe
    grants = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_publish_sources=[api.TrackSource.MICROPHONE],
        can_publish_data=True,
        can_subscribe=True,
    )
    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(user_identity)
        .with_name(display_name)
        .with_grants(grants)
        .with_room_config(room_config)
        .with_ttl(ttl)
        .to_jwt()
    )

    expires_at = (datetime.now(timezone.utc) + ttl).isoformat()
    return {
        "token": token,
        "url": settings.livekit_url,
        "room": room_name,
        "identity": user_identity,
        "expires_at": expires_at,
        "channel": channel,
        "language": language,
    }


def check_health(settings: Settings) -> dict[str, object]:
    """Check active knowledge base and runtime configuration status."""
    try:
        path, manifest = read_active(settings)
        knowledge_status = "verified"
    except RuntimeError:
        root = settings.chroma_dir.resolve()
        from rag.manifest import read_active_version, resolve_artifact_path

        version = read_active_version(root)
        path = resolve_artifact_path(root, version)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        knowledge_status = "unverified"

    livekit_configured = bool(
        settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret
    )
    is_healthy = livekit_configured and (
        knowledge_status == "verified" or settings.environment == "development"
    )
    return {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "knowledge_version": manifest.get("version", "unknown"),
        "chunks": manifest.get("chunks", 0),
        "knowledge_status": knowledge_status,
        "environment": settings.environment,
        "agent_name": settings.agent_name or "nexi",
        "livekit_configured": livekit_configured,
        "ai_configured": bool(
            settings.google_api_key and settings.deepgram_api_key and settings.sarvam_api_key
        ),
    }


class TokenApiHandler(BaseHTTPRequestHandler):
    """Hardened HTTP handler for token issuance, rate limits, admission, and readiness."""

    settings: Settings = load_settings()
    storage: TelemetryStore = TelemetryStore(settings)
    rate_limiter: RateLimiter = RateLimiter(settings)
    admission: AdmissionController = AdmissionController(settings, storage)
    readiness: ReadinessChecker = ReadinessChecker(settings, storage)

    def _get_request_id(self) -> str:
        return self.headers.get("X-Request-Id") or uuid4().hex

    def _get_cors_origin(self) -> str | None:
        origin = self.headers.get("Origin", "").strip()
        if origin and origin in self.settings.allowed_origins:
            return origin
        return None

    def _send_json(
        self,
        status: int | HTTPStatus,
        data: dict[str, object],
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-Id", self._get_request_id())
        self.send_header("Vary", "Origin")

        allowed_origin = self._get_cors_origin()
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Request-Id")

        self.send_header("Cache-Control", "no-store")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("X-Request-Id", self._get_request_id())
        self.send_header("Vary", "Origin")
        allowed_origin = self._get_cors_origin()
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Request-Id")
            self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self) -> None:
        self.request.settimeout(5.0)
        req_id = self._get_request_id()

        if self.path in ("/healthz", "/healthz/"):
            data = asyncio.run(self.readiness.check_liveness())
            self._send_json(HTTPStatus.OK, data)
        elif self.path in ("/readyz", "/readyz/"):
            code, data = asyncio.run(self.readiness.check_readiness())
            self._send_json(code, data)
        elif self.path in ("/api/health", "/health"):
            try:
                self._send_json(HTTPStatus.OK, check_health(self.settings))
            except Exception as exc:
                LOGGER.warning("Health check failed: %s", exc)
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"status": "unhealthy", "error": "Health check failed"},
                )
        elif self.path in ("/", "/index.html"):
            index_path = ROOT / "web" / "index.html"
            if index_path.exists():
                content = index_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("X-Request-Id", req_id)
                self.end_headers()
                self.wfile.write(content)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Index file not found"})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        self.request.settimeout(5.0)
        req_id = self._get_request_id()

        # 1. Rate limiting check
        peer_ip = self.client_address[0] if self.client_address else "127.0.0.1"
        xff = self.headers.get("X-Forwarded-For")
        allowed, retry_after, client_ip = self.rate_limiter.check_rate_limit(peer_ip, xff)
        client_hash = compute_caller_hash(client_ip, self.settings.pii_hash_key)

        if not allowed:
            LOGGER.warning(
                "Rate limit exceeded for client %s (req_id: %s); retry after %ds",
                client_hash,
                req_id,
                retry_after,
            )
            self._send_json(
                HTTPStatus.TOO_MANY_REQUESTS,
                {"error": "rate_limited", "retry_after_s": retry_after},
                headers={"Retry-After": str(retry_after)},
            )
            return

        # 2. Token issuance route
        if self.path in ("/api/token", "/token", "/api/token/", "/token/"):
            # Enforce Content-Type
            ct = self.headers.get("Content-Type", "").strip()
            if not ct.startswith("application/json"):
                self._send_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"error": "Content-Type must be application/json"},
                )
                return

            raw_len = self.headers.get("Content-Length")
            if not raw_len:
                payload: dict[str, object] = {}
            else:
                try:
                    length = int(raw_len)
                except ValueError:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length header"}
                    )
                    return

                if length > MAX_BODY_BYTES:
                    self._send_json(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        {"error": f"Payload exceeds limit of {MAX_BODY_BYTES} bytes"},
                    )
                    return

                raw_data = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw_data)
                except json.JSONDecodeError:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST, {"error": "Malformed JSON in request body"}
                    )
                    return

                if not isinstance(payload, dict):
                    self._send_json(
                        HTTPStatus.BAD_REQUEST, {"error": "Request body must be a JSON object"}
                    )
                    return

            language = str(payload.get("language", "en")).strip().lower()
            if language not in ("en", "te", "hi"):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Language must be 'en', 'te', or 'hi'"},
                )
                return

            # 3. Daily quota check
            budget_ok, err_reason, b_status, b_retry = self.admission.check_and_consume_daily_budget()
            if not budget_ok:
                LOGGER.warning("Daily token budget exhausted (req_id: %s)", req_id)
                self._send_json(
                    b_status,
                    {"error": err_reason},
                    headers={"Retry-After": str(b_retry)},
                )
                return

            # 4. Admission capacity check
            cap_ok, cap_err, cap_status, cap_retry = asyncio.run(self.admission.check_capacity())
            if not cap_ok:
                LOGGER.warning("Admission capacity check denied: %s (req_id: %s)", cap_err, req_id)
                self._send_json(
                    cap_status,
                    {"error": cap_err},
                    headers={"Retry-After": str(cap_retry)},
                )
                return

            # 5. Generate token (channel is fixed to 'web')
            try:
                result = generate_token(self.settings, channel="web", language=language)
                self._send_json(HTTPStatus.OK, result)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                LOGGER.error("Failed generating token (req_id: %s): %s", req_id, exc)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "Failed to generate room token"},
                )
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})


def serve_token_api(port: int = 8080, host: str | None = None) -> None:
    settings = load_settings()
    if settings.environment == "production" and "*" in settings.allowed_origins:
        raise ConfigurationError("ALLOWED_ORIGINS cannot contain wildcards in production")

    listen_host = host or settings.host or "127.0.0.1"
    server = HTTPServer((listen_host, port), TokenApiHandler)
    LOGGER.info("Nexi Token API server running on http://%s:%d", listen_host, port)
    server.serve_forever()


if __name__ == "__main__":
    serve_token_api()
