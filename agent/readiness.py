"""Deep readiness and liveness inspection for token and agent services."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Callable

from agent.config import Settings
from agent.storage import TelemetryStore
from rag.manifest import read_active

LOGGER = logging.getLogger("nexi.readiness")


class ReadinessChecker:
    """Evaluates liveness and deep readiness of all operational subsystems."""

    def __init__(
        self,
        settings: Settings,
        storage: TelemetryStore,
        clock: Callable[[], float] | None = None,
        livekit_api: Any = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.clock = clock or monotonic
        self._livekit_api = livekit_api

        self._cache_time: float = -100.0
        self._cached_result: dict[str, Any] = {}
        self._cache_lock = asyncio.Lock()

    async def check_liveness(self) -> dict[str, Any]:
        """Liveness check for /healthz."""
        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def check_readiness(self) -> tuple[int, dict[str, Any]]:
        """Deep readiness check for /readyz (cached for 5 seconds)."""
        now = self.clock()
        async with self._cache_lock:
            if (now - self._cache_time) < 5.0 and self._cached_result:
                status_code = 200 if self._cached_result.get("status") in ("ok", "degraded") else 503
                return status_code, self._cached_result

            checks: dict[str, dict[str, str]] = {}
            has_fail = False
            has_degraded = False

            # 1. LiveKit API reachable
            lk_status = await self._check_livekit()
            checks["livekit"] = lk_status
            if lk_status["status"] == "fail":
                has_fail = True
            elif lk_status["status"] == "degraded":
                has_degraded = True

            # 2. Agent worker reachable
            worker_status = await asyncio.to_thread(self._check_worker)
            checks["worker"] = worker_status
            if worker_status["status"] == "fail":
                has_fail = True
            elif worker_status["status"] == "degraded":
                has_degraded = True

            # 3. Knowledge Base
            kb_status = self._check_knowledge()
            checks["knowledge"] = kb_status
            if kb_status["status"] == "fail":
                has_fail = True
            elif kb_status["status"] == "degraded":
                has_degraded = True

            # 4. Telemetry database
            db_status = self._check_database()
            checks["database"] = db_status
            if db_status["status"] == "fail":
                has_fail = True
            elif db_status["status"] == "degraded":
                has_degraded = True

            # 5. Provider configuration
            prov_status = self._check_provider_config()
            checks["providers"] = prov_status
            if prov_status["status"] == "fail":
                has_fail = True

            overall_status = "fail" if has_fail else ("degraded" if has_degraded else "ok")
            result = {
                "status": overall_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": checks,
            }
            self._cached_result = result
            self._cache_time = now

            status_code = 200 if overall_status in ("ok", "degraded") else 503
            return status_code, result

    async def _check_livekit(self) -> dict[str, str]:
        if not (self.settings.livekit_url and self.settings.livekit_api_key and self.settings.livekit_api_secret):
            return {"status": "fail", "reason": "livekit_unconfigured"}

        lk = self._livekit_api
        should_close = False
        if lk is None:
            from livekit import api

            lk = api.LiveKitAPI(
                self.settings.livekit_url,
                self.settings.livekit_api_key,
                self.settings.livekit_api_secret,
            )
            should_close = True

        try:
            from livekit.api import room_service

            async with asyncio.timeout(2.0):
                await lk.room.list_rooms(room_service.ListRoomsRequest())
            return {"status": "ok", "reason": "reachable"}
        except Exception as exc:
            return {"status": "fail", "reason": f"unreachable_{type(exc).__name__}"}
        finally:
            if should_close:
                try:
                    await lk.aclose()
                except Exception as exc:
                    LOGGER.debug("Error closing temporary LiveKitAPI: %s", exc)

    def _check_worker(self) -> dict[str, str]:
        # Try HTTP GET to worker_health_url with 1.5s timeout
        url = self.settings.worker_health_url or "http://127.0.0.1:8081/"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nexi-readiness"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status in (200, 204):
                    return {"status": "ok", "reason": "worker_http_ok"}
        except Exception as exc:
            LOGGER.debug("Worker HTTP health check unavailable: %s", exc)

        # Check worker heartbeat file if present and fresh (< 15s)
        heartbeat_file = self.storage.db_dir / "worker_heartbeat.json"
        if heartbeat_file.is_file():
            try:
                data = json.loads(heartbeat_file.read_text(encoding="utf-8"))
                ts = data.get("timestamp_monotonic", 0.0)
                if (self.clock() - ts) < 15.0:
                    return {"status": "ok", "reason": "worker_heartbeat_fresh"}
            except Exception as exc:
                LOGGER.debug("Error reading worker heartbeat file: %s", exc)

        return {"status": "fail", "reason": "worker_unreachable"}

    def _check_knowledge(self) -> dict[str, str]:
        try:
            path, manifest = read_active(self.settings)
            quality_status = manifest.get("quality_status", "")
            if quality_status == "passed":
                return {"status": "ok", "version": manifest.get("version", "unknown")}
            elif self.settings.allow_unverified_knowledge or self.settings.environment == "development":
                return {
                    "status": "degraded",
                    "version": manifest.get("version", "unknown"),
                    "reason": "unverified_knowledge",
                }
            else:
                return {"status": "fail", "reason": f"knowledge_status_{quality_status}"}
        except Exception as exc:
            if self.settings.allow_unverified_knowledge or self.settings.environment == "development":
                return {"status": "degraded", "reason": f"dev_unverified_{type(exc).__name__}"}
            return {"status": "fail", "reason": f"knowledge_error_{type(exc).__name__}"}

    def _check_database(self) -> dict[str, str]:
        try:
            with self.storage._connection() as conn:
                conn.execute("BEGIN IMMEDIATE;")
                conn.rollback()

            if self.storage.degraded:
                return {"status": "degraded", "reason": "storage_degraded"}
            if self.storage.recovered:
                return {"status": "degraded", "reason": "database_recovered_from_corruption"}
            return {"status": "ok", "reason": "writable"}
        except Exception as exc:
            return {"status": "fail", "reason": f"database_error_{type(exc).__name__}"}

    def _check_provider_config(self) -> dict[str, str]:
        missing = []
        if not self.settings.google_api_key:
            missing.append("google_api_key")
        if not self.settings.deepgram_api_key:
            missing.append("deepgram_api_key")
        if not self.settings.sarvam_api_key:
            missing.append("sarvam_api_key")

        if missing:
            return {"status": "fail", "reason": f"missing_{'_'.join(missing)}"}
        return {"status": "ok", "reason": "configured"}
