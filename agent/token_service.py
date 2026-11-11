"""Token generation and health API service for web and mobile clients."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from uuid import uuid4

from livekit import api

from agent.config import Settings, load_settings
from rag.manifest import read_active

ROOT = Path(__file__).resolve().parent.parent


def generate_token(
    settings: Settings,
    channel: str = "web",
    language: str = "en",
    identity: str | None = None,
    name: str | None = None,
) -> dict[str, object]:
    """Create an ephemeral LiveKit room token authorizing agent dispatch."""
    if channel not in ("web", "phone"):
        raise ValueError("Channel must be 'web' or 'phone'")
    if language not in ("en", "te", "hi"):
        raise ValueError("Language must be 'en', 'te', or 'hi'")

    user_identity = identity or f"student-{uuid4().hex[:8]}"
    display_name = name or ("Student" if language == "en" else "విద్యార్థి" if language == "te" else "छात्र")
    room_name = f"nexi-{channel}-{uuid4().hex[:12]}"
    dispatch_agent = settings.agent_name or "nexi"

    dispatch_meta = json.dumps({"channel": channel, "language": language}, ensure_ascii=False)
    room_config = api.RoomConfiguration(
        agents=[api.RoomAgentDispatch(agent_name=dispatch_agent, metadata=dispatch_meta)]
    )

    ttl = timedelta(minutes=15)
    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(user_identity)
        .with_name(display_name)
        .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True))
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
    path, manifest = read_active(settings)
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "knowledge_version": manifest["version"],
        "chunks": manifest["chunks"],
        "environment": settings.environment,
        "agent_name": settings.agent_name or "automatic-dispatch",
        "livekit_configured": bool(settings.livekit_url and settings.livekit_api_key),
    }


class TokenApiHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP handler for /api/token and /api/health."""

    settings: Settings = load_settings()

    def _send_json(self, status: HTTPStatus, data: dict[str, object]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/api/health", "/health"):
            try:
                self._send_json(HTTPStatus.OK, check_health(self.settings))
            except Exception as exc:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"status": "unhealthy", "error": str(exc)})
        elif self.path in ("/", "/index.html"):
            index_path = ROOT / "web" / "index.html"
            if index_path.exists():
                content = index_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Index file not found"})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path in ("/api/token", "/token"):
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or "{}") if length > 0 else {}
                channel = str(payload.get("channel", "web"))
                language = str(payload.get("language", "en"))
                result = generate_token(self.settings, channel=channel, language=language)
                self._send_json(HTTPStatus.OK, result)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Token generation failed", "detail": str(exc)})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})


def serve_token_api(port: int = 8080) -> None:
    server = HTTPServer(("0.0.0.0", port), TokenApiHandler)
    print(f"Nexi Token API server running on http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve_token_api()
