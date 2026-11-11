"""Minimal artifact contract shared by offline bootstrap and runtime."""

import json
import re
from pathlib import Path

from agent.config import Settings

COLLECTION = "nexi"
SCHEMA_VERSION = 1


def model_fingerprint(settings: Settings) -> dict[str, object]:
    return {
        "embed_model": settings.embed_model,
        "embed_revision": settings.embed_revision,
        "normalize_embeddings": True,
        "distance": "cosine",
    }


def read_active(settings: Settings) -> tuple[Path, dict]:
    """Never create a missing database or silently accept an incompatible artifact."""
    root = settings.chroma_dir.resolve()
    try:
        version = (root / "CURRENT").read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[a-f0-9]{32}", version):
            raise ValueError("Invalid knowledge version")
        path = (root / version).resolve()
        if path.parent != root or not (path / "chroma.sqlite3").is_file():
            raise ValueError("Knowledge database missing or outside configured directory")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Invalid knowledge manifest")
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("version") != version:
            raise ValueError("Unsupported knowledge manifest version")
        if manifest.get("embedding") != model_fingerprint(settings):
            raise ValueError("Knowledge embedding configuration differs; rebuild offline")
        if manifest.get("collection") != COLLECTION:
            raise ValueError("Knowledge collection mismatch")
        count = manifest.get("chunks")
        if type(count) is not int or count <= 0:
            raise ValueError("Knowledge artifact is empty")
        return path, manifest
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "No valid knowledge artifact. Run python -m rag.ingest before starting the agent."
        ) from exc
