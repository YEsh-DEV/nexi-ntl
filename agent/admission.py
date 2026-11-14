"""Admission control: active call capacity via LiveKit rooms and daily token budget."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any, Callable

from agent.config import Settings
from agent.storage import TelemetryStore

LOGGER = logging.getLogger("nexi.admission")


class AdmissionController:
    """Controls token issuance capacity and daily consumption quotas."""

    def __init__(
        self,
        settings: Settings,
        storage: TelemetryStore,
        clock: Callable[[], float] | None = None,
        now_func: Callable[[], datetime] | None = None,
        livekit_api: Any = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.clock = clock or monotonic
        self.now_func = now_func or (lambda: datetime.now(timezone.utc))
        self._livekit_api = livekit_api

        # Capacity cache: (timestamp, count, is_error)
        self._cache_time: float = -100.0
        self._cached_count: int = 0
        self._cached_error: bool = False
        self._cache_lock = asyncio.Lock()

    def _get_livekit_api(self) -> Any:
        if self._livekit_api is not None:
            return self._livekit_api
        if not (self.settings.livekit_url and self.settings.livekit_api_key and self.settings.livekit_api_secret):
            return None
        from livekit import api

        return api.LiveKitAPI(
            self.settings.livekit_url,
            self.settings.livekit_api_key,
            self.settings.livekit_api_secret,
        )

    async def _query_active_rooms_count(self) -> tuple[int, bool]:
        """Query LiveKit rooms API with 2 second timeout.

        Returns (active_room_count, is_error).
        """
        lk = self._get_livekit_api()
        if lk is None:
            return 0, True

        should_close = self._livekit_api is None
        try:
            from livekit.api import room_service

            async with asyncio.timeout(2.0):
                resp = await lk.room.list_rooms(room_service.ListRoomsRequest())
                active = sum(
                    1
                    for r in resp.rooms
                    if (
                        r.name.startswith("web-")
                        or r.name.startswith("call-")
                        or r.name.startswith("nexi-")
                    )
                    and r.num_participants >= 1
                )
                return active, False
        except Exception as exc:
            LOGGER.warning("Failed querying LiveKit room list: %s", exc)
            return 0, True
        finally:
            if should_close:
                try:
                    await lk.aclose()
                except Exception as exc:
                    LOGGER.debug("Error closing temporary LiveKitAPI: %s", exc)

    async def check_capacity(self) -> tuple[bool, str, int, int]:
        """Check active call capacity.

        Returns:
            (allowed: bool, error_reason: str, http_status: int, retry_after_s: int)
        """
        now = self.clock()
        async with self._cache_lock:
            if (now - self._cache_time) >= 5.0:
                count, err = await self._query_active_rooms_count()
                self._cached_count = count
                self._cached_error = err
                self._cache_time = now
            else:
                count = self._cached_count
                err = self._cached_error

        if err:
            if self.settings.admission_fail_open:
                LOGGER.warning("Admission capacity check failed; failing open (development mode)")
                return True, "", 200, 0
            return False, "capacity_unknown", 503, 5

        if count >= self.settings.max_concurrent_calls:
            return False, "at_capacity", 503, 10

        return True, "", 200, 0

    def check_and_consume_daily_budget(self) -> tuple[bool, str, int, int]:
        """Check and atomically consume daily token budget from daily_counters.

        Returns:
            (allowed: bool, error_reason: str, http_status: int, retry_after_s: int)
        """
        now = self.now_func()
        day_utc = now.strftime("%Y-%m-%d")

        # Atomic increment in daily_counters
        with self.storage._connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO daily_counters (day_utc, kind, count)
                VALUES (?, 'tokens_issued', 1)
                ON CONFLICT(day_utc, kind) DO UPDATE SET count = count + 1
                RETURNING count;
                """,
                (day_utc,),
            )
            row = cur.fetchone()
            current_count = row[0] if row else 1

        if current_count > self.settings.max_tokens_per_day:
            tomorrow = datetime(now.year, now.month, now.day, tzinfo=timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            retry_after_s = max(1, int((tomorrow - now).total_seconds()))
            return False, "budget_exhausted", 503, retry_after_s

        return True, "", 200, 0
