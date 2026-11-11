"""Small, content-free structured event logger; durable reporting is Phase 6."""

import json
import logging
from datetime import datetime, timezone

log = logging.getLogger("nexi")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Provider messages/tracebacks can contain transcripts, keys, or request bodies.
        # Keep only their origin and severity. Nexi events use curated fields below.
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
    """Call with application-owned event names and non-sensitive scalar metrics only."""
    log.log(level, name, extra={"fields": fields})
