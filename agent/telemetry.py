"""Bounded durable operational telemetry pipeline with spill, drain, and structured logging."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable
from uuid import uuid4

from agent.config import Settings
from agent.storage import TelemetryStore

log = logging.getLogger("nexi")
logger = logging.getLogger(__name__)

CRITICAL_EVENTS = {"call_started", "call_ended", "knowledge_gap"}
METRIC_EVENTS = {"turn_metrics", "stage_metrics", "usage"}


@dataclass
class TelemetryEvent:
    event_type: str
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid4().hex)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_critical(self) -> bool:
        return self.event_type in CRITICAL_EVENTS


class TelemetryPipeline:
    def __init__(
        self,
        settings: Settings,
        storage: TelemetryStore,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.clock = clock or monotonic

        self.queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue(maxsize=settings.telemetry_queue_size)
        self.spill_path: Path = settings.telemetry_spill_path or (storage.db_dir / "telemetry_spill.jsonl")

        self.telemetry_dropped = 0
        self.storage_errors = 0
        self._last_error_time: float = 0.0
        self._last_drop_warning_time: float = -1000.0

        self._closing = False
        self._in_flight = False
        self._writer_task: asyncio.Task[None] | None = None

        # Replay any lingering spill lines on creation
        self._replay_spill_sync()

    def start(self) -> None:
        if self._writer_task is None or self._writer_task.done():
            self._writer_task = asyncio.create_task(self._writer_loop(), name="telemetry-writer")

    @property
    def degraded(self) -> bool:
        if self.storage.degraded:
            return True
        # True if a storage error occurred in the last 5 minutes (300 seconds)
        return (self.clock() - self._last_error_time) < 300.0

    def _replay_spill_sync(self) -> None:
        if not self.spill_path.is_file():
            return

        lines_committed = 0
        remaining_lines: list[str] = []
        try:
            with open(self.spill_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                        ev_type = data.get("event_type")
                        payload = data.get("payload", {})
                        self._write_event_sync(ev_type, payload)
                        lines_committed += 1
                    except json.JSONDecodeError:
                        logger.warning("Corrupt JSON line in telemetry spill file, skipping: %s", line_str[:100])
                    except Exception as exc:
                        logger.error("Failed replaying spill line: %s", exc)
                        remaining_lines.append(line_str)
        except Exception as exc:
            logger.error("Failed opening spill file for replay: %s", exc)
            return

        if not remaining_lines:
            try:
                self.spill_path.unlink(missing_ok=True)
                if lines_committed > 0:
                    logger.info("Successfully replayed and cleared %d spilled telemetry events", lines_committed)
            except Exception as exc:
                logger.error("Failed unlinking spill file: %s", exc)
        else:
            with open(self.spill_path, "w", encoding="utf-8") as f:
                for line_str in remaining_lines:
                    f.write(line_str + "\n")

    def _spill_event_sync(self, event: TelemetryEvent) -> None:
        try:
            self.spill_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.spill_path, "a", encoding="utf-8") as f:
                rec = {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "payload": event.payload,
                    "created_at_utc": event.created_at_utc,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error("Failed spilling critical event %s: %s", event.event_type, exc)

    async def put_event(self, event_type: str, **payload: Any) -> bool:
        if self._closing:
            logger.warning("Telemetry pipeline is draining/closed; rejecting new event %s", event_type)
            return False

        ev = TelemetryEvent(event_type=event_type, payload=payload)
        if not ev.is_critical:
            # Metric event: non-blocking put; drop on queue full
            try:
                self.queue.put_nowait(ev)
                return True
            except asyncio.QueueFull:
                self.telemetry_dropped += 1
                now = self.clock()
                if now - self._last_drop_warning_time >= 60.0:
                    self._last_drop_warning_time = now
                    logger.warning(
                        "WARNING: Telemetry queue full; dropped %d metric events so far.",
                        self.telemetry_dropped,
                    )
                return False
        else:
            # Critical event: wait up to 1 second for queue space
            try:
                await asyncio.wait_for(self.queue.put(ev), timeout=1.0)
                return True
            except (asyncio.TimeoutError, asyncio.QueueFull):
                # Queue full past timeout: spill critical event directly to disk
                logger.warning("Telemetry queue full for critical event %s; spilling to disk.", event_type)
                await asyncio.to_thread(self._spill_event_sync, ev)
                return True

    def _write_event_sync(self, event_type: str, p: dict[str, Any]) -> None:
        if event_type == "call_started":
            self.storage.record_call_started_sync(
                call_id=p["call_id"],
                channel=p.get("channel", "web"),
                language=p.get("language", "en"),
                job_id=p.get("job_id", ""),
                room_hash=p.get("room_hash", ""),
                caller_number=p.get("caller_number", ""),
                started_at_utc=p.get("started_at_utc"),
            )
        elif event_type == "call_ended":
            self.storage.record_call_ended_sync(
                call_id=p["call_id"],
                status=p.get("status", "ended"),
                ended_at_utc=p.get("ended_at_utc", datetime.now(timezone.utc).isoformat()),
                duration_s=p.get("duration_s", 0.0),
                end_reason=p.get("end_reason", "unknown"),
                language_final=p.get("language_final", "en"),
                turns=p.get("turns", 0),
                tool_calls=p.get("tool_calls", 0),
                no_results_count=p.get("no_results_count", 0),
                language_switches=p.get("language_switches", 0),
                language_switch_failures=p.get("language_switch_failures", 0),
                transfer_attempted=p.get("transfer_attempted", 0),
                transfer_succeeded=p.get("transfer_succeeded", 0),
                transfer_department=p.get("transfer_department", ""),
                room_delete_failures=p.get("room_delete_failures", 0),
                errors=p.get("errors", 0),
                usage=p.get("usage", {}),
            )
        elif event_type == "turn_metrics":
            self.storage.record_turn_metrics_sync(
                call_id=p["call_id"],
                channel=p.get("channel", "web"),
                language=p.get("language", "en"),
                eou_delay_ms=p.get("eou_delay_ms", 0.0),
                llm_ttft_ms=p.get("llm_ttft_ms", 0.0),
                tts_ttfb_ms=p.get("tts_ttfb_ms", 0.0),
                total_ms=p.get("total_ms", 0.0),
                retrieval_ms=p.get("retrieval_ms", 0.0),
                tool_calls=p.get("tool_calls", 0),
                ts_utc=p.get("ts_utc"),
            )
        elif event_type == "knowledge_gap":
            self.storage.record_knowledge_gap_sync(
                query=p["query"],
                language=p.get("language", "en"),
                channel=p.get("channel", "web"),
                reason=p.get("reason", "no_results"),
                top_score=p.get("top_score", 0.0),
                nearest_topic=p.get("nearest_topic", ""),
                timestamp_utc=p.get("timestamp_utc"),
            )

    async def _writer_loop(self) -> None:
        while not self._closing or not self.queue.empty():
            try:
                try:
                    event = await asyncio.wait_for(self.queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

                self._in_flight = True
                try:
                    await asyncio.to_thread(self._write_event_sync, event.event_type, event.payload)
                except Exception as exc:
                    self.storage_errors += 1
                    self._last_error_time = self.clock()
                    logger.error("Storage write error for event %s: %s", event.event_type, exc)
                    if event.is_critical:
                        await asyncio.to_thread(self._spill_event_sync, event)
                finally:
                    self._in_flight = False
                    self.queue.task_done()
            except asyncio.CancelledError:
                # Do not drop in-flight writes when cancelled
                break
            except Exception as exc:
                logger.error("Unexpected error in telemetry writer loop: %s", exc)

    async def drain(self, timeout: float | None = None) -> None:
        """Drain queued events within timeout, spill critical ones on expiration."""
        timeout = timeout if timeout is not None else self.settings.telemetry_drain_timeout_s
        self._closing = True

        async def _drain_work() -> None:
            await self.queue.join()
            while self._in_flight:
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(_drain_work(), timeout=timeout)
        except asyncio.TimeoutError:
            rem = self.queue.qsize()
            logger.warning("WARNING: Telemetry drain timed out with %d events remaining.", rem)
            # Spill remaining critical events
            while not self.queue.empty():
                try:
                    ev = self.queue.get_nowait()
                    if ev.is_critical:
                        await asyncio.to_thread(self._spill_event_sync, ev)
                    self.queue.task_done()
                except Exception:
                    break
        finally:
            if self._writer_task is not None and not self._writer_task.done():
                self._writer_task.cancel()
                try:
                    await self._writer_task
                except asyncio.CancelledError:
                    pass


# -------------------------------------------------------------------------
# Logging Configuration and Structured Event Helper
# -------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage() if record.name == "nexi" else "external_log",
        }
        if record.name == "nexi":
            payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    log.setLevel(level)
    for name in ("httpx", "httpcore", "google", "chromadb", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def event(name: str, *, level: int = logging.INFO, **fields: object) -> None:
    """Emit application-owned event name with non-sensitive scalar metrics only."""
    log.log(level, name, extra={"fields": fields})
