"""Artifact contract and manifest validation for offline bootstrap and runtime."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from agent.config import Settings

logger = logging.getLogger(__name__)

COLLECTION = "nexi"
SCHEMA_VERSION = 1
VALID_QUALITY_STATUSES = {"unevaluated", "needs_review", "passed", "failed", "phase1_unevaluated"}


def model_fingerprint(settings: Settings) -> dict[str, object]:
    return {
        "embed_model": settings.embed_model,
        "embed_revision": settings.embed_revision,
        "normalize_embeddings": True,
        "distance": "cosine",
    }


def resolve_artifact_path(root: Path, version: str) -> Path:
    """Find artifact directory under artifacts/<version> or root/<version>."""
    candidate1 = root / "artifacts" / version
    if candidate1.is_dir() and (candidate1 / "chroma.sqlite3").is_file():
        return candidate1
    candidate2 = root / version
    if candidate2.is_dir() and (candidate2 / "chroma.sqlite3").is_file():
        return candidate2
    return candidate2


def read_active_version(root: Path) -> str:
    """Read active version from ACTIVE or legacy CURRENT file."""
    active_file = root / "ACTIVE"
    if active_file.is_file():
        return active_file.read_text(encoding="utf-8").strip()
    current_file = root / "CURRENT"
    if current_file.is_file():
        return current_file.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("Neither ACTIVE nor CURRENT pointer exists in knowledge directory")


def read_active(settings: Settings) -> tuple[Path, dict[str, Any]]:
    """Never create a missing database or silently accept an incompatible/unverified artifact."""
    root = settings.chroma_dir.resolve()
    try:
        version = read_active_version(root)
        if not re.fullmatch(r"[a-f0-9]{32}", version):
            raise ValueError(f"Invalid knowledge version format: {version}")

        path = resolve_artifact_path(root, version)
        if not (path / "chroma.sqlite3").is_file():
            raise ValueError(f"Knowledge database missing at {path}")

        manifest_file = path / "manifest.json"
        if not manifest_file.is_file():
            raise ValueError(f"Knowledge manifest missing at {manifest_file}")

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Invalid knowledge manifest JSON")

        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("version") != version:
            raise ValueError("Unsupported knowledge manifest version or schema version")

        if manifest.get("embedding") != model_fingerprint(settings):
            raise ValueError("Knowledge embedding configuration differs; rebuild offline")

        if manifest.get("collection") != COLLECTION:
            raise ValueError("Knowledge collection mismatch")

        count = manifest.get("chunks")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Knowledge artifact is empty")

        # Quality status and verification gates
        quality_status = manifest.get("quality_status", "unevaluated")
        is_partial = bool(manifest.get("partial", False))

        if is_partial or quality_status != "passed":
            allow_unverified = getattr(settings, "allow_unverified_knowledge", False) or os.getenv("ALLOW_UNVERIFIED_KNOWLEDGE") == "1"
            if not allow_unverified:
                raise RuntimeError(
                    f"Knowledge artifact '{version}' is unverified (status: {quality_status}, partial: {is_partial}). "
                    f"Refusing to load in production. Set ALLOW_UNVERIFIED_KNOWLEDGE=1 for development override."
                )
            logger.warning(
                "WARNING: Loading unverified knowledge artifact '%s' (status: %s, partial: %s) due to ALLOW_UNVERIFIED_KNOWLEDGE=1",
                version,
                quality_status,
                is_partial,
            )

        return path, manifest
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "No valid knowledge artifact. Run python -m rag.ingest before starting the agent."
        ) from exc
