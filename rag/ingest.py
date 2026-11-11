"""Explicit Phase 1 offline bootstrap, never imported by the running agent.

Phase 2 adds reviewed OCR, complete record/temporal semantics, hybrid indexing and eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agent.config import Settings, load_settings
from rag.manifest import COLLECTION, SCHEMA_VERSION, model_fingerprint


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


def build(settings: Settings, *, allow_empty_sources: bool) -> Path:
    # Imports and model downloads are restricted to this explicitly invoked offline command.
    import chromadb
    import torch
    from chromadb.config import Settings as ChromaSettings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader
    from sentence_transformers import CrossEncoder, SentenceTransformer

    files = sorted(settings.pdf_dir.rglob("*.pdf")) + sorted(settings.json_dir.rglob("*.json"))
    if not files:
        raise RuntimeError("No PDF or JSON sources found; check PDF_DIR and JSON_DIR")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents, metadatas, ids, coverage = [], [], [], []
    for source in files:
        is_pdf = source.suffix.lower() == ".pdf"
        root = settings.pdf_dir if is_pdf else settings.json_dir
        source_id = ("pdf/" if is_pdf else "json/") + source.relative_to(root).as_posix()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        empty_pages = []
        if is_pdf:
            reader = PdfReader(source)
            records = []
            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    empty_pages.append(number)
                else:
                    records.append((str(number), text))
        else:
            records = list(json_records(json.loads(source.read_text(encoding="utf-8-sig"))))
        before = len(ids)
        for location, text in records:
            for index, chunk in enumerate(splitter.split_text(text)):
                if not chunk.strip():
                    continue
                # Same source content produces the same IDs, independently of build timestamp.
                identity = json.dumps(
                    [source_id, digest, location, index, chunk], ensure_ascii=False
                )
                ids.append(hashlib.sha256(identity.encode()).hexdigest())
                label = f"page {location}" if is_pdf else location
                documents.append(f"Reference: {source.name} | {label}\n{chunk}")
                metadata = {
                    "source": source_id,
                    "source_type": "pdf" if is_pdf else "json",
                    "source_hash": digest,
                    "language": "en",
                    "path": location,
                }
                if is_pdf:
                    metadata["page"] = int(location)
                metadatas.append(metadata)
        coverage.append(
            {
                "source": source_id,
                "sha256": digest,
                "chunks": len(ids) - before,
                "empty_pages": empty_pages,
            }
        )

    incomplete = [row for row in coverage if not row["chunks"] or row["empty_pages"]]
    for row in coverage:
        print(json.dumps(row, ensure_ascii=False))
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
    # Cache the exact reranker before runtime's local_files_only load.
    CrossEncoder(settings.rerank_model, revision=settings.rerank_revision, device="cpu")
    version = uuid4().hex
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    target = settings.chroma_dir / version
    target.mkdir()  # Unique version: never append to or delete the active database.
    client = chromadb.PersistentClient(
        path=str(target),
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
        raise RuntimeError("Index count mismatch. Previous active version was preserved.")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "collection": COLLECTION,
        "embedding": model_fingerprint(settings),
        "chunks": len(ids),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sources": coverage,
        "partial": bool(incomplete),
        "quality_status": "phase1_unevaluated",
        "chunker": {"version": "bootstrap-v1", "size": 800, "overlap": 100},
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary = settings.chroma_dir / f".CURRENT-{version}"
    temporary.write_text(version + "\n", encoding="utf-8")
    os.replace(temporary, settings.chroma_dir / "CURRENT")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-empty-sources",
        action="store_true",
        help="Explicitly publish partial development data; omissions enter the manifest",
    )
    args = parser.parse_args()
    target = build(load_settings(), allow_empty_sources=args.allow_empty_sources)
    print(
        f"Activated unevaluated Phase 1 artifact: {target.name}. Stop/restart workers to load it."
    )


if __name__ == "__main__":
    main()
