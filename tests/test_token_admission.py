"""Unit tests for token service admission control, rate limiting, request hardening, and readiness."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent.admission import AdmissionController
from agent.config import ROOT, ConfigurationError, Settings
from agent.ratelimit import RateLimiter, resolve_client_ip
from agent.readiness import ReadinessChecker
from agent.storage import TelemetryStore
from agent.token_service import TokenApiHandler


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


class FakeLiveKitRoom:
    def __init__(self, name: str, num_participants: int = 1) -> None:
        self.name = name
        self.num_participants = num_participants


class FakeLiveKitApi:
    def __init__(self, rooms: list[FakeLiveKitRoom] | None = None, fail: bool = False) -> None:
        self.rooms = rooms if rooms is not None else []
        self.fail = fail
        self.call_count = 0
        self.room = MagicMock()
        self.room.list_rooms = AsyncMock(side_effect=self._list_rooms)

    async def _list_rooms(self, req: object) -> object:
        self.call_count += 1
        if self.fail:
            raise RuntimeError("LiveKit room service unreachable")
        resp = MagicMock()
        resp.rooms = self.rooms
        return resp

    async def aclose(self) -> None:
        pass


class TestRateLimiter(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock(100.0)
        self.settings = Settings(
            token_rate_per_ip=3,
            token_rate_window_s=60.0,
            token_rate_global_per_min=5,
            trusted_proxies=("10.0.0.1", "10.0.0.2", "192.168.1.0/24"),
        )
        self.limiter = RateLimiter(self.settings, clock=self.clock, max_clients=10)

    def test_per_ip_limit_and_sliding_window(self) -> None:
        # First 3 requests from IP 1.2.3.4 allowed
        for _ in range(3):
            allowed, retry, ip = self.limiter.check_rate_limit("1.2.3.4")
            self.assertTrue(allowed)
            self.assertEqual(retry, 0)
            self.assertEqual(ip, "1.2.3.4")

        # 4th request within 60s denied
        allowed, retry, ip = self.limiter.check_rate_limit("1.2.3.4")
        self.assertFalse(allowed)
        self.assertGreaterEqual(retry, 1)

        # Other IP is unaffected (per-IP isolation)
        allowed_other, _, _ = self.limiter.check_rate_limit("5.6.7.8")
        self.assertTrue(allowed_other)

        # Advance clock past window: 1.2.3.4 allowed again
        self.clock.advance(61.0)
        allowed_after, _, _ = self.limiter.check_rate_limit("1.2.3.4")
        self.assertTrue(allowed_after)

    def test_global_rate_limit(self) -> None:
        # 5 distinct IPs use up global quota (limit = 5)
        for i in range(5):
            allowed, _, _ = self.limiter.check_rate_limit(f"192.0.2.{i}")
            self.assertTrue(allowed)

        # 6th request from a brand new IP denied by global limiter
        allowed6, retry6, _ = self.limiter.check_rate_limit("192.0.2.99")
        self.assertFalse(allowed6)
        self.assertGreaterEqual(retry6, 1)

    def test_lru_client_eviction(self) -> None:
        # max_clients is 10
        limiter = RateLimiter(self.settings, clock=self.clock, max_clients=3)
        limiter.check_rate_limit("10.0.0.10")
        limiter.check_rate_limit("10.0.0.11")
        limiter.check_rate_limit("10.0.0.12")
        self.assertEqual(len(limiter._client_windows), 3)

        # 4th client evicts the least recently used (10.0.0.10)
        limiter.check_rate_limit("10.0.0.13")
        self.assertEqual(len(limiter._client_windows), 3)
        self.assertNotIn("10.0.0.10", limiter._client_windows)
        self.assertIn("10.0.0.13", limiter._client_windows)

    def test_x_forwarded_for_trusted_proxies(self) -> None:
        # 1. Peer NOT in trusted proxies: ignore header completely
        ip1 = resolve_client_ip(
            peer_ip="203.0.113.5",
            x_forwarded_for="198.51.100.1, 10.0.0.1",
            trusted_proxies=self.settings.trusted_proxies,
        )
        self.assertEqual(ip1, "203.0.113.5")

        # 2. Peer IS trusted proxy: take right-most non-trusted address
        ip2 = resolve_client_ip(
            peer_ip="10.0.0.1",
            x_forwarded_for="203.0.113.44, 10.0.0.2",
            trusted_proxies=self.settings.trusted_proxies,
        )
        self.assertEqual(ip2, "203.0.113.44")

        # 3. CIDR trusted proxy match (192.168.1.50 in 192.168.1.0/24)
        ip3 = resolve_client_ip(
            peer_ip="192.168.1.50",
            x_forwarded_for="198.51.100.77",
            trusted_proxies=self.settings.trusted_proxies,
        )
        self.assertEqual(ip3, "198.51.100.77")


class TestAdmissionAndBudget(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "admission.sqlite3"
        self.clock = FakeClock(100.0)
        self.settings = Settings(
            chroma_dir=Path(self.temp_dir) / "k",
            pii_hash_key="test_key",
            max_concurrent_calls=2,
            max_tokens_per_day=3,
            admission_fail_open=False,
        )
        self.store = TelemetryStore(self.settings, db_path=self.db_path)

    async def asyncTearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_capacity_under_and_at_capacity(self) -> None:
        rooms = [
            FakeLiveKitRoom("web-1234", num_participants=1),
            FakeLiveKitRoom("call-sip-99", num_participants=1),
        ]
        fake_api = FakeLiveKitApi(rooms=rooms)
        adm = AdmissionController(
            self.settings, self.store, clock=self.clock, livekit_api=fake_api
        )

        # 2 active rooms == max_concurrent_calls (2) -> at_capacity
        allowed, err, status, retry = await adm.check_capacity()
        self.assertFalse(allowed)
        self.assertEqual(err, "at_capacity")
        self.assertEqual(status, 503)

        # Empty room with 0 participants does not count
        rooms.pop()
        rooms.append(FakeLiveKitRoom("call-sip-empty", num_participants=0))
        # Advance clock to bust 5s cache
        self.clock.advance(6.0)
        allowed2, err2, status2, _ = await adm.check_capacity()
        self.assertTrue(allowed2)
        self.assertEqual(status2, 200)

    async def test_capacity_caching_for_5_seconds(self) -> None:
        fake_api = FakeLiveKitApi(rooms=[])
        adm = AdmissionController(
            self.settings, self.store, clock=self.clock, livekit_api=fake_api
        )

        await adm.check_capacity()
        self.assertEqual(fake_api.call_count, 1)

        # Within 5s: cached, call_count unchanged
        self.clock.advance(2.0)
        await adm.check_capacity()
        self.assertEqual(fake_api.call_count, 1)

        # After 5s: queries again
        self.clock.advance(4.0)
        await adm.check_capacity()
        self.assertEqual(fake_api.call_count, 2)

    async def test_capacity_fail_closed_and_fail_open(self) -> None:
        fake_api = FakeLiveKitApi(fail=True)

        # 1. Fail closed (default)
        adm_closed = AdmissionController(
            self.settings, self.store, clock=self.clock, livekit_api=fake_api
        )
        allowed, err, status, _ = await adm_closed.check_capacity()
        self.assertFalse(allowed)
        self.assertEqual(err, "capacity_unknown")
        self.assertEqual(status, 503)

        # 2. Fail open (development override)
        dev_settings = Settings(
            chroma_dir=Path(self.temp_dir) / "k",
            pii_hash_key="test_key",
            admission_fail_open=True,
        )
        self.clock.advance(6.0)
        adm_open = AdmissionController(
            dev_settings, self.store, clock=self.clock, livekit_api=fake_api
        )
        allowed_open, _, status_open, _ = await adm_open.check_capacity()
        self.assertTrue(allowed_open)
        self.assertEqual(status_open, 200)

    def test_daily_budget_exhaustion_and_rollover(self) -> None:
        current_time = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)
        adm = AdmissionController(
            self.settings,
            self.store,
            now_func=lambda: current_time,
        )

        # max_tokens_per_day = 3
        for _ in range(3):
            allowed, _, status, _ = adm.check_and_consume_daily_budget()
            self.assertTrue(allowed)
            self.assertEqual(status, 200)

        # 4th token on same UTC day is denied
        allowed4, err4, status4, retry4 = adm.check_and_consume_daily_budget()
        self.assertFalse(allowed4)
        self.assertEqual(err4, "budget_exhausted")
        self.assertEqual(status4, 503)
        self.assertEqual(retry4, 12 * 3600)  # 12 hours until midnight

        # Rollover to next UTC day allows requests again
        current_time = datetime(2026, 11, 15, 0, 1, 0, tzinfo=timezone.utc)
        allowed_next, _, status_next, _ = adm.check_and_consume_daily_budget()
        self.assertTrue(allowed_next)
        self.assertEqual(status_next, 200)


class TestReadiness(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "ready.sqlite3"
        self.clock = FakeClock(100.0)
        self.settings = Settings(
            chroma_dir=Path(self.temp_dir) / "k",
            pii_hash_key="test_key",
            livekit_url="wss://test.livekit.cloud",
            livekit_api_key="API123",
            livekit_api_secret="SECRET123",
            google_api_key="AIzaSyMock",
            deepgram_api_key="MockDeepgram",
            sarvam_api_key="MockSarvam",
            allow_unverified_knowledge=True,
            environment="development",
        )
        self.store = TelemetryStore(self.settings, db_path=self.db_path)
        self.fake_api = FakeLiveKitApi(rooms=[])
        self.checker = ReadinessChecker(
            self.settings, self.store, clock=self.clock, livekit_api=self.fake_api
        )

    async def asyncTearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_liveness_endpoint(self) -> None:
        res = await self.checker.check_liveness()
        self.assertEqual(res["status"], "ok")

    async def test_readiness_pass_and_caching(self) -> None:
        with patch.object(self.checker, "_check_worker", return_value={"status": "ok", "reason": "worker_http_ok"}):
            code, res = await self.checker.check_readiness()
            self.assertEqual(code, 200)
            self.assertIn(res["status"], ("ok", "degraded"))
            self.assertEqual(self.fake_api.call_count, 1)

            # Within 5s: cached
            self.clock.advance(2.0)
            code2, _ = await self.checker.check_readiness()
            self.assertEqual(code2, 200)
            self.assertEqual(self.fake_api.call_count, 1)

    async def test_readiness_fails_when_livekit_unreachable(self) -> None:
        self.fake_api.fail = True
        self.clock.advance(6.0)
        with patch.object(self.checker, "_check_worker", return_value={"status": "ok", "reason": "worker_http_ok"}):
            code, res = await self.checker.check_readiness()
            self.assertEqual(code, 503)
            self.assertEqual(res["status"], "fail")
            self.assertEqual(res["checks"]["livekit"]["status"], "fail")

    async def test_readiness_fails_when_provider_key_missing(self) -> None:
        missing_settings = Settings(
            chroma_dir=Path(self.temp_dir) / "k",
            pii_hash_key="test_key",
            livekit_url="wss://test.livekit.cloud",
            livekit_api_key="API123",
            livekit_api_secret="SECRET123",
            google_api_key="",
            deepgram_api_key="MockDeepgram",
            sarvam_api_key="MockSarvam",
            allow_unverified_knowledge=True,
            environment="development",
        )
        checker = ReadinessChecker(
            missing_settings, self.store, clock=self.clock, livekit_api=self.fake_api
        )
        with patch.object(checker, "_check_worker", return_value={"status": "ok", "reason": "worker_http_ok"}):
            code, res = await checker.check_readiness()
            self.assertEqual(code, 503)
            self.assertEqual(res["status"], "fail")
            self.assertEqual(res["checks"]["providers"]["status"], "fail")

    def test_readiness_worker_heartbeat_fallback(self) -> None:
        heartbeat_file = self.store.db_dir / "worker_heartbeat.json"
        heartbeat_file.write_text(json.dumps({"timestamp_monotonic": self.clock()}))
        res = self.checker._check_worker()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["reason"], "worker_heartbeat_fresh")

    def test_readiness_database_states(self) -> None:
        # Default writable
        res1 = self.checker._check_database()
        self.assertEqual(res1["status"], "ok")

        # Recovered
        self.store.recovered = True
        res2 = self.checker._check_database()
        self.assertEqual(res2["status"], "degraded")
        self.store.recovered = False

        # Degraded
        self.store.degraded = True
        res3 = self.checker._check_database()
        self.assertEqual(res3["status"], "degraded")


class DummySocket:
    def __init__(self, peer_ip: str = "127.0.0.1") -> None:
        self.peer_ip = peer_ip

    def getpeername(self) -> tuple[str, int]:
        return self.peer_ip, 12345

    def settimeout(self, t: float) -> None:
        pass


class TestTokenApiHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "handler.sqlite3"
        self.settings = Settings(
            chroma_dir=ROOT / "data/knowledge",
            allow_unverified_knowledge=True,
            pii_hash_key="test_key",
            livekit_url="wss://test.livekit.cloud",
            livekit_api_key="API123",
            livekit_api_secret="SECRET123",
            google_api_key="MockGoogle",
            deepgram_api_key="MockDeepgram",
            sarvam_api_key="MockSarvam",
            allowed_origins=("http://localhost:3000", "http://example.com"),
            token_ttl_s=300.0,
            max_concurrent_calls=5,
            max_tokens_per_day=50,
        )
        self.store = TelemetryStore(self.settings, db_path=self.db_path)
        self.fake_api = FakeLiveKitApi(rooms=[])

        # Configure handler class
        TokenApiHandler.settings = self.settings
        TokenApiHandler.storage = self.store
        TokenApiHandler.rate_limiter = RateLimiter(self.settings)
        TokenApiHandler.admission = AdmissionController(
            self.settings, self.store, livekit_api=self.fake_api
        )
        TokenApiHandler.readiness = ReadinessChecker(
            self.settings, self.store, livekit_api=self.fake_api
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _execute_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes = b"",
        peer_ip: str = "127.0.0.1",
    ) -> tuple[int, dict[str, str], bytes]:
        req_lines = [f"{method} {path} HTTP/1.1"]
        for k, v in headers.items():
            req_lines.append(f"{k}: {v}")
        if "Content-Length" not in headers and body:
            req_lines.append(f"Content-Length: {len(body)}")
        req_lines.append("\r\n")
        raw_request = "\r\n".join(req_lines).encode("utf-8") + body

        rfile = io.BytesIO(raw_request)
        wfile = io.BytesIO()

        handler = TokenApiHandler.__new__(TokenApiHandler)
        handler.rfile = rfile
        handler.wfile = wfile
        handler.connection = DummySocket(peer_ip)
        handler.client_address = (peer_ip, 12345)
        handler.server = MagicMock()
        handler.request = handler.connection
        handler.requestline = f"{method} {path} HTTP/1.1"
        handler.command = method
        handler.path = path
        handler.request_version = "HTTP/1.1"
        handler.headers = handler.parse_headers() if hasattr(handler, "parse_headers") else MagicMock()

        # Parse headers cleanly using email parser skipping request line
        from email.parser import BytesParser
        header_bytes = raw_request.split(b"\r\n", 1)[1].split(b"\r\n\r\n", 1)[0] + b"\r\n\r\n"
        msg = BytesParser().parsebytes(header_bytes)
        handler.headers = msg
        handler.rfile = io.BytesIO(body)

        if method == "POST":
            handler.do_POST()
        elif method == "GET":
            handler.do_GET()
        elif method == "OPTIONS":
            handler.do_OPTIONS()

        wfile.seek(0)
        raw_response = wfile.read()
        lines = raw_response.split(b"\r\n\r\n", 1)
        header_part = lines[0].decode("utf-8", errors="replace")
        body_part = lines[1] if len(lines) > 1 else b""

        status_line = header_part.splitlines()[0]
        status_code = int(status_line.split()[1])

        resp_headers = {}
        for hline in header_part.splitlines()[1:]:
            if ":" in hline:
                hk, hv = hline.split(":", 1)
                resp_headers[hk.strip()] = hv.strip()

        return status_code, resp_headers, body_part

    def test_post_token_content_type_validation(self) -> None:
        # 1. Non-JSON Content-Type -> 415
        code, _, body = self._execute_request(
            "POST",
            "/api/token",
            {"Content-Type": "text/plain"},
            b"not json",
        )
        self.assertEqual(code, 415)
        self.assertIn("application/json", json.loads(body)["error"])

    def test_post_token_payload_size_limit(self) -> None:
        # Payload > 16KB -> 413
        large_body = b"x" * (17 * 1024)
        code, _, body = self._execute_request(
            "POST",
            "/api/token",
            {"Content-Type": "application/json", "Content-Length": str(len(large_body))},
            large_body,
        )
        self.assertEqual(code, 413)

    def test_post_token_malformed_json_and_validation(self) -> None:
        # Malformed JSON -> 400
        code, _, _ = self._execute_request(
            "POST",
            "/api/token",
            {"Content-Type": "application/json"},
            b"{bad json",
        )
        self.assertEqual(code, 400)

        # Invalid language -> 400
        code2, _, body2 = self._execute_request(
            "POST",
            "/api/token",
            {"Content-Type": "application/json"},
            json.dumps({"language": "spanish"}).encode(),
        )
        self.assertEqual(code2, 400)
        self.assertIn("Language must be", json.loads(body2)["error"])

    def test_post_token_success_and_headers(self) -> None:
        code, headers, body = self._execute_request(
            "POST",
            "/api/token",
            {"Content-Type": "application/json", "Origin": "http://localhost:3000"},
            json.dumps({"language": "te"}).encode(),
        )
        self.assertEqual(code, 200)
        self.assertIn("X-Request-Id", headers)
        self.assertEqual(headers.get("Vary"), "Origin")
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://localhost:3000")

        data = json.loads(body)
        self.assertEqual(data["channel"], "web")
        self.assertEqual(data["language"], "te")
        self.assertTrue(data["room"].startswith("web-"))

    def test_cors_preflight_and_wildcard_rejection(self) -> None:
        code, headers, _ = self._execute_request(
            "OPTIONS",
            "/api/token",
            {"Origin": "http://localhost:3000"},
        )
        self.assertEqual(code, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://localhost:3000")
        self.assertIn("POST", headers.get("Access-Control-Allow-Methods", ""))

        # Production with wildcard fails configuration
        prod_settings = Settings(
            environment="production",
            allowed_origins=("*",),
            pii_hash_key="test_key",
        )
        with self.assertRaises(ConfigurationError):
            prod_settings.validate_runtime(console=False)

    def test_get_healthz_and_readyz_routes(self) -> None:
        with patch.object(TokenApiHandler.readiness, "_check_worker", return_value={"status": "ok", "reason": "worker_http_ok"}):
            # /healthz
            code1, headers1, body1 = self._execute_request("GET", "/healthz", {})
            self.assertEqual(code1, 200)
            self.assertEqual(json.loads(body1)["status"], "ok")

            # /readyz
            code2, headers2, body2 = self._execute_request("GET", "/readyz", {})
            self.assertEqual(code2, 200)
            self.assertIn(json.loads(body2)["status"], ("ok", "degraded"))

            # /api/health
            code3, _, body3 = self._execute_request("GET", "/api/health", {})
            self.assertEqual(code3, 200)

            # Not found
            code4, _, _ = self._execute_request("GET", "/unknown/route", {})
            self.assertEqual(code4, 404)

            # POST not found
            code5, _, _ = self._execute_request("POST", "/unknown/post", {"Content-Type": "application/json"}, b"{}")
            self.assertEqual(code5, 404)


if __name__ == "__main__":
    unittest.main()
