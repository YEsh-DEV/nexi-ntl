"""Offline ingestion, atomic activation, rollback, and version pruning for knowledge artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agent.config import Settings, load_settings
from rag.chunking import chunk_json_node, chunk_text_and_tables, derive_dates_and_topic
from rag.manifest import COLLECTION, SCHEMA_VERSION, model_fingerprint, resolve_artifact_path

logger = logging.getLogger(__name__)


def json_records(node: object, path: str = "root"):
    """Keep scalar siblings together without discarding nested children."""
    if isinstance(node, dict):
        facts = {
            key: value
            for key, value in node.items()
            if value is not None and not isinstance(value, (dict, list))
        }
        if facts:
            yield path, json.dumps(facts, ensure_ascii=False)
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                yield from json_records(value, f"{path} > {key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            label = (value.get("name") or value.get("title")) if isinstance(value, dict) else None
            yield from json_records(value, f"{path} > {label or index}")
    elif node is not None and str(node).strip():
        yield path, str(node)


def load_history(root: Path) -> list[str]:
    history_file = root / "history.json"
    if history_file.is_file():
        try:
            data = json.loads(history_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(v) for v in data]
        except Exception as hist_exc:
            logger.debug("Non-fatal history file parse error: %s", hist_exc)
    return []


def save_history(root: Path, history: list[str]) -> None:
    history_file = root / "history.json"
    temp_file = root / f".history-{uuid4().hex}.tmp"
    temp_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
    os.replace(temp_file, history_file)


def prune_old_versions(root: Path, active_version: str, keep_count: int = 3) -> list[str]:
    """Retain the last `keep_count` versions plus active version, delete older versions."""
    history = load_history(root)
    keep_set = set(history[-keep_count:])
    keep_set.add(active_version)

    deleted = []
    # Check root direct version directories
    for p in root.iterdir():
        if p.is_dir() and len(p.name) == 32 and p.name not in ("staging", "artifacts"):
            if p.name not in keep_set:
                shutil.rmtree(p, ignore_errors=True)
                deleted.append(p.name)

    # Check artifacts/<version>
    artifacts_dir = root / "artifacts"
    if artifacts_dir.is_dir():
        for p in artifacts_dir.iterdir():
            if p.is_dir() and len(p.name) == 32:
                if p.name not in keep_set:
                    shutil.rmtree(p, ignore_errors=True)
                    deleted.append(p.name)

    return deleted


def build(settings: Settings, *, allow_empty_sources: bool = False) -> Path:
    """Build knowledge index into staging/<version>. Never touches ACTIVE."""
    import chromadb
    import torch
    from chromadb.config import Settings as ChromaSettings
    from pypdf import PdfReader
    from sentence_transformers import CrossEncoder, SentenceTransformer

    files = sorted(settings.pdf_dir.rglob("*.pdf")) + sorted(settings.json_dir.rglob("*.json"))
    if not files:
        raise RuntimeError("No PDF or JSON sources found; check PDF_DIR and JSON_DIR")

    documents: list[str] = []
    metadatas: list[dict[str, object]] = []
    ids: list[str] = []
    coverage: list[dict[str, object]] = []

    for source in files:
        is_pdf = source.suffix.lower() == ".pdf"
        root = settings.pdf_dir if is_pdf else settings.json_dir
        source_id = ("pdf/" if is_pdf else "json/") + source.relative_to(root).as_posix()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        empty_pages = []
        before = len(ids)

        eff_date, val_until, doc_ver, topic_key = derive_dates_and_topic(source)

        if is_pdf:
            reader = PdfReader(source)
            for page_num, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                if not raw_text.strip():
                    empty_pages.append(page_num)
                else:
                    chunks = chunk_text_and_tables(raw_text, page_num)
                    for chunk_idx, (loc, c_text) in enumerate(chunks):
                        identity = json.dumps([source_id, digest, loc, chunk_idx, c_text], ensure_ascii=False)
                        cid = hashlib.sha256(identity.encode()).hexdigest()
                        ids.append(cid)
                        documents.append(f"Reference: {source.name} | {loc}\n{c_text}")
                        meta: dict[str, object] = {
                            "source": source_id,
                            "source_type": "pdf",
                            "source_hash": digest,
                            "language": "en",
                            "path": loc,
                            "page": page_num,
                            "topic_key": topic_key,
                        }
                        if eff_date:
                            meta["effective_date"] = eff_date
                        if val_until:
                            meta["valid_until"] = val_until
                        if doc_ver:
                            meta["doc_version"] = doc_ver
                        metadatas.append(meta)
        else:
            data = json.loads(source.read_text(encoding="utf-8-sig"))
            records = chunk_json_node(data)
            for chunk_idx, (path_loc, ctx_prefix, c_text, scalars) in enumerate(records):
                s_eff, s_val, s_ver, s_top = derive_dates_and_topic(source, record=scalars)
                identity = json.dumps([source_id, digest, path_loc, chunk_idx, c_text], ensure_ascii=False)
                cid = hashlib.sha256(identity.encode()).hexdigest()
                ids.append(cid)
                doc_str = f"Reference: {source.name} | {path_loc}"
                if ctx_prefix:
                    doc_str += f"\nContext: {ctx_prefix}"
                doc_str += f"\n{c_text}"
                documents.append(doc_str)
                meta = {
                    "source": source_id,
                    "source_type": "json",
                    "source_hash": digest,
                    "language": "en",
                    "path": path_loc,
                    "topic_key": s_top or topic_key,
                }
                if s_eff or eff_date:
                    meta["effective_date"] = s_eff or eff_date
                if s_val or val_until:
                    meta["valid_until"] = s_val or val_until
                if s_ver or doc_ver:
                    meta["doc_version"] = s_ver or doc_ver
                metadatas.append(meta)

        coverage.append({
            "source": source_id,
            "sha256": digest,
            "chunks": len(ids) - before,
            "empty_pages": empty_pages,
            "effective_date": eff_date,
            "topic_key": topic_key,
        })

    incomplete = [row for row in coverage if not row["chunks"] or row["empty_pages"]]
    if incomplete and not allow_empty_sources:
        raise RuntimeError(
            "Sources contain empty text/pages. Obtain OCR/replacements, or explicitly use "
            "--allow-empty-sources for a partial development corpus. No index was activated."
        )
    if not documents:
        raise RuntimeError("No searchable text found. No index was activated.")

    torch.set_num_threads(settings.inference_threads)
    embedding = SentenceTransformer(
        settings.embed_model,
        revision=settings.embed_revision,
        device="cpu",
    )
    CrossEncoder(settings.rerank_model, revision=settings.rerank_revision, device="cpu")

    version = uuid4().hex
    staging_dir = settings.chroma_dir / "staging" / version
    staging_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(staging_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.create_collection(
        COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    for start in range(0, len(documents), 128):
        batch = documents[start : start + 128]
        vectors = embedding.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        collection.add(
            ids=ids[start : start + 128],
            documents=batch,
            metadatas=metadatas[start : start + 128],
            embeddings=vectors.tolist(),
        )

    if collection.count() != len(ids):
        raise RuntimeError("Index count mismatch during staging build.")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "collection": COLLECTION,
        "embedding": model_fingerprint(settings),
        "chunks": len(ids),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sources": coverage,
        "partial": bool(incomplete),
        "quality_status": "unevaluated",
        "coverage_approval": None,
        "rerank_min_score": settings.rerank_min_score,
        "latency_waiver": None,
    }
    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return staging_dir


def activate(settings: Settings, version: str) -> Path:
    """Atomically activate version after verifying approval, passed gates, and complete sources."""
    root = settings.chroma_dir.resolve()
    staging_path = root / "staging" / version
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    target_path = artifacts_dir / version

    if staging_path.is_dir():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        shutil.move(str(staging_path), str(target_path))
    else:
        target_path = resolve_artifact_path(root, version)

    if not target_path.is_dir():
        raise FileNotFoundError(f"Artifact directory for version '{version}' not found")

    manifest_file = target_path / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest missing at {manifest_file}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    # Verification checks
    if manifest.get("partial", True):
        raise RuntimeError(f"Cannot activate partial artifact '{version}'")

    quality_status = manifest.get("quality_status")
    if quality_status != "passed":
        raise RuntimeError(
            f"Cannot activate artifact '{version}' with quality_status '{quality_status}' (must be 'passed')"
        )

    approval = manifest.get("coverage_approval")
    if not approval or not isinstance(approval, dict):
        raise RuntimeError(f"Coverage approval is missing for artifact '{version}'")

    coverage_report = target_path / "coverage_report.json"
    if not coverage_report.is_file():
        raise RuntimeError(f"Coverage report missing at {coverage_report}")

    curr_sha256 = hashlib.sha256(coverage_report.read_bytes()).hexdigest()
    if approval.get("report_sha256") != curr_sha256:
        raise RuntimeError(
            f"Coverage approval hash mismatch: approved {approval.get('report_sha256')} vs actual {curr_sha256}"
        )

    # Atomic swap: write to temp file in root, then os.replace to ACTIVE
    temp_active = root / f".ACTIVE-{version}-{uuid4().hex}.tmp"
    temp_active.write_text(version + "\n", encoding="utf-8")
    os.replace(temp_active, root / "ACTIVE")

    # Also update CURRENT for legacy compatibility
    temp_current = root / f".CURRENT-{version}-{uuid4().hex}.tmp"
    temp_current.write_text(version + "\n", encoding="utf-8")
    os.replace(temp_current, root / "CURRENT")

    # History update
    history = load_history(root)
    if not history or history[-1] != version:
        history.append(version)
        save_history(root, history)

    # Prune keeping last 3 versions
    prune_old_versions(root, active_version=version, keep_count=3)

    return target_path


def rollback(settings: Settings) -> str:
    """Rollback ACTIVE to previous version from history."""
    root = settings.chroma_dir.resolve()
    history = load_history(root)
    if len(history) < 2:
        raise RuntimeError("Cannot rollback: fewer than 2 versions recorded in history")

    # The current version is history[-1], previous is history[-2]
    prev_version = history[-2]
    prev_path = resolve_artifact_path(root, prev_version)
    if not prev_path.is_dir():
        raise FileNotFoundError(f"Previous version directory not found at {prev_path}")

    # Atomic swap
    temp_active = root / f".ACTIVE-{prev_version}-{uuid4().hex}.tmp"
    temp_active.write_text(prev_version + "\n", encoding="utf-8")
    os.replace(temp_active, root / "ACTIVE")

    temp_current = root / f".CURRENT-{prev_version}-{uuid4().hex}.tmp"
    temp_current.write_text(prev_version + "\n", encoding="utf-8")
    os.replace(temp_current, root / "CURRENT")

    # Update history
    history.append(prev_version)
    save_history(root, history)

    return prev_version


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge base ingestion and lifecycle")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build", help="Build into staging")
    build_p.add_argument("--allow-empty-sources", action="store_true", help="Allow partial sources")

    activate_p = subparsers.add_parser("activate", help="Atomically activate version")
    activate_p.add_argument("version", help="Version to activate")

    subparsers.add_parser("rollback", help="Rollback to previous version")

    args = parser.parse_args()
    settings = load_settings()

    if args.command == "build":
        target = build(settings, allow_empty_sources=args.allow_empty_sources)
        print(f"Staged unevaluated artifact: {target.name}")
    elif args.command == "activate":
        target = activate(settings, args.version)
        print(f"Successfully activated artifact: {target.name}")
    elif args.command == "rollback":
        rolled = rollback(settings)
        print(f"Successfully rolled back to version: {rolled}")


if __name__ == "__main__":
    main()
