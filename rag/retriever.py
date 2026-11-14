"""Phase 2 hybrid retrieval (dense + BM25 with RRF), cross-encoder reranking, and freshness filtering."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from agent.config import Settings
from rag.manifest import COLLECTION, read_active

logger = logging.getLogger(__name__)


class BM25Index:
    """Zero-dependency Okapi BM25 implementation (k1=1.5, b=0.75) using Python stdlib."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(corpus)
        self.doc_counts = [Counter(re.findall(r"\w+", doc.lower())) for doc in corpus]
        self.doc_lens = [sum(counts.values()) for counts in self.doc_counts]
        self.avgdl = sum(self.doc_lens) / (self.n or 1)
        df: dict[str, int] = Counter()
        for counts in self.doc_counts:
            for term in counts:
                df[term] += 1
        self.idf = {
            term: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def top_k(self, query: str, k: int = 12) -> list[int]:
        q_terms = [t for t in re.findall(r"\w+", query.lower()) if t in self.idf]
        if not q_terms:
            return []
        scores = [0.0] * self.n
        for term in q_terms:
            idf = self.idf[term]
            for i, counts in enumerate(self.doc_counts):
                tf = counts[term]
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                    scores[i] += idf * (tf * (self.k1 + 1)) / denom
        indexed = [(score, i) for i, score in enumerate(scores) if score > 0]
        indexed.sort(key=lambda item: (-item[0], item[1]))
        return [i for _, i in indexed[:k]]


def rrf(dense_ids: list[str], bm25_ids: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion with rank bias k=60."""
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    text: str
    source: str
    page_or_path: str
    score: float
    effective_date: str | None = None
    valid_until: str | None = None
    topic_key: str | None = None


@dataclass(frozen=True)
class SearchResult:
    evidence: tuple[Evidence, ...]
    knowledge_version: str
    prefix_note: str = ""
    reason: str = ""
    top_score: float = 0.0
    nearest_topic: str = ""

    @property
    def context(self) -> str:
        body = "\n\n".join(item.text for item in self.evidence)
        if self.prefix_note and body:
            return f"{self.prefix_note}\n\n{body}"
        return body


class RetrievalBusy(RuntimeError):
    """A previous timed-out search still occupies the inference worker."""


def filter_freshness(
    candidates: list[tuple[tuple[str, str, dict[str, Any]], float]],
    today_str: str,
) -> tuple[list[tuple[tuple[str, str, dict[str, Any]], float]], str]:
    """Filter candidates by validity, effective date, and topic key.

    1. Drop chunks whose valid_until < today, unless nothing else remains.
    2. Among chunks sharing topic_key with different effective_date <= today, keep only latest.
    3. Tie-break deterministically: (-effective_date, -doc_version, chunk_id).
    4. If only expired chunks remain, prefix note with warning.
    """
    if not candidates:
        return [], ""

    prefix_note = ""
    # 1. Check expiration
    non_expired = []
    for item in candidates:
        (_, _, meta), _ = item
        valid_until = meta.get("valid_until")
        if valid_until and str(valid_until) < today_str:
            continue
        non_expired.append(item)

    if non_expired:
        pool = non_expired
    else:
        # Only expired chunks remain
        pool = candidates
        dates = [item[0][2].get("effective_date") for item in pool if item[0][2].get("effective_date")]
        eff_str = str(dates[0]) if dates else "previous years"
        prefix_note = f"NOTE: this information may be out of date (effective {eff_str})."

    # 2. Topic key and effective date filtering
    # Group by topic_key
    latest_by_topic: dict[str, str] = {}
    for item in pool:
        (_, _, meta), _ = item
        topic = meta.get("topic_key")
        eff = meta.get("effective_date")
        if topic and eff and str(eff) <= today_str:
            eff_str = str(eff)
            if topic not in latest_by_topic or eff_str > latest_by_topic[topic]:
                latest_by_topic[topic] = eff_str

    filtered = []
    for item in pool:
        (_, _, meta), _ = item
        topic = meta.get("topic_key")
        eff = meta.get("effective_date")
        if topic and eff and topic in latest_by_topic:
            # Keep only the chunk matching latest effective date <= today for this topic
            if str(eff) != latest_by_topic[topic]:
                continue
        elif eff and str(eff) > today_str:
            continue
        filtered.append(item)

    # 3. Deterministic sort: score desc, effective_date desc, doc_version desc, chunk_id asc
    def _sort_key(item: tuple[tuple[str, str, dict[str, Any]], float]) -> tuple[Any, ...]:
        (chunk_id, _, meta), score = item
        eff = str(meta.get("effective_date") or "")
        ver = str(meta.get("doc_version") or "")
        return (-score, -len(eff), eff, ver, chunk_id)

    filtered.sort(key=lambda item: (-item[1], _sort_key(item)))
    return filtered, prefix_note


class Retriever:
    def __init__(
        self,
        settings: Settings,
        today_func: Callable[[], date] | None = None,
    ) -> None:
        # Validate on-disk state before Chroma can create anything.
        path, manifest = read_active(settings)
        import chromadb
        import torch
        from chromadb.config import Settings as ChromaSettings
        from sentence_transformers import CrossEncoder, SentenceTransformer

        torch.set_num_threads(settings.inference_threads)
        self.settings = settings
        self.today_func = today_func or date.today
        self.manifest = manifest
        self.knowledge_version = manifest["version"]

        # Rerank threshold from manifest with development override notice
        self.rerank_min_score = manifest.get("rerank_min_score", settings.rerank_min_score)
        env_override = os.getenv("RERANK_MIN_SCORE")
        if env_override is not None:
            logger.warning(
                "WARNING: RERANK_MIN_SCORE environment override active (%s). Manifest specified %s.",
                env_override,
                self.rerank_min_score,
            )
            try:
                self.rerank_min_score = float(env_override)
            except ValueError:
                pass

        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_collection(COLLECTION, embedding_function=None)
        self._count = self._collection.count()
        if self._count != manifest["chunks"]:
            raise RuntimeError("Knowledge count differs from its manifest; rebuild offline")

        # In-memory corpus and lookup cache for hybrid BM25 and fast candidate resolution
        raw = self._collection.get(include=["documents", "metadatas"])
        self._all_ids: list[str] = raw["ids"]
        self._all_docs: list[str] = raw["documents"]
        self._lookup: dict[str, tuple[str, dict[str, Any]]] = {
            cid: (doc, meta or {})
            for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"], strict=True)
        }
        self._bm25 = BM25Index(self._all_docs)

        # LRU cache for query embeddings keyed by normalized query (maxsize 512)
        self._embed_cache: dict[str, list[float]] = {}
        self._embed_cache_maxsize = 512

        # Download using the offline bootstrap first. Calls never fetch model weights.
        self._embeddings = SentenceTransformer(
            settings.embed_model,
            revision=settings.embed_revision,
            device="cpu",
            local_files_only=True,
        )
        self._reranker = CrossEncoder(
            settings.rerank_model,
            revision=settings.rerank_revision,
            device="cpu",
            local_files_only=True,
            activation_fn=torch.nn.Identity(),
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nexi-retrieval")
        self._pending: Future | None = None
        self._closed = False

    def _get_query_embedding(self, query: str) -> list[float]:
        q_norm = " ".join(query.lower().split())
        if q_norm in self._embed_cache:
            return self._embed_cache[q_norm]

        vector = self._embeddings.encode(query, normalize_embeddings=True).tolist()
        if len(self._embed_cache) >= self._embed_cache_maxsize:
            # Evict first element (LRU FIFO)
            first_key = next(iter(self._embed_cache))
            del self._embed_cache[first_key]
        self._embed_cache[q_norm] = vector
        return vector

    def _search_sync(self, query: str) -> SearchResult:
        # 1. Dense retrieval
        vector = self._get_query_embedding(query)
        found = self._collection.query(
            query_embeddings=[vector],
            n_results=min(self.settings.retrieval_k, self._count),
            include=["documents", "metadatas"],
        )
        dense_ids = found["ids"][0] if found["ids"] else []

        # 2. BM25 keyword retrieval
        bm25_indices = self._bm25.top_k(query, k=self.settings.retrieval_k)
        bm25_ids = [self._all_ids[idx] for idx in bm25_indices]

        # 3. Reciprocal Rank Fusion (top 16)
        merged_ids = rrf(dense_ids, bm25_ids, k=60)[:16]
        candidates = [
            (cid, self._lookup[cid][0], self._lookup[cid][1])
            for cid in merged_ids
            if cid in self._lookup and self._lookup[cid][0].strip()
        ]
        if not candidates:
            return SearchResult((), self.knowledge_version, reason="no_results", top_score=0.0, nearest_topic="")

        # 4. Cross-encoder reranking
        scores = self._reranker.predict([(query, row[1]) for row in candidates])
        best_idx = max(range(len(candidates)), key=lambda i: scores[i])
        top_score = float(scores[best_idx])
        best_meta = candidates[best_idx][2]
        nearest_topic = str(best_meta.get("topic_key") or best_meta.get("source") or "")

        accepted = []
        for row, score in zip(candidates, scores, strict=True):
            score = float(score)
            if math.isfinite(score) and score >= self.rerank_min_score:
                accepted.append((row, score))
        accepted.sort(key=lambda item: (-item[1], item[0][0]))

        if not accepted:
            return SearchResult(
                (),
                self.knowledge_version,
                reason="below_threshold",
                top_score=top_score,
                nearest_topic=nearest_topic,
            )

        # 5. Freshness and conflict filtering
        today_str = self.today_func().isoformat()
        filtered_candidates, prefix_note = filter_freshness(accepted, today_str)
        if not filtered_candidates:
            return SearchResult(
                (),
                self.knowledge_version,
                prefix_note=prefix_note,
                reason="stale_only",
                top_score=top_score,
                nearest_topic=nearest_topic,
            )

        # 6. Pack evidence chunks within context_max_chars budget
        evidence = []
        used = 0
        for (chunk_id, text, meta), score in filtered_candidates:
            cost = len(text) + (2 if evidence else 0)
            if used + cost > self.settings.context_max_chars:
                continue  # Preserve complete chunks, including conditions and exceptions.
            evidence.append(
                Evidence(
                    chunk_id,
                    text,
                    str(meta.get("source", "unknown")),
                    str(meta.get("page", meta.get("path", ""))),
                    score,
                    effective_date=meta.get("effective_date"),
                    valid_until=meta.get("valid_until"),
                    topic_key=meta.get("topic_key"),
                )
            )
            used += cost
            if len(evidence) == self.settings.retrieval_top_n:
                break

        reason = "" if evidence else "no_results"
        return SearchResult(
            tuple(evidence),
            self.knowledge_version,
            prefix_note=prefix_note,
            reason=reason,
            top_score=top_score,
            nearest_topic=nearest_topic,
        )

    async def search(self, query: str) -> SearchResult:
        query = query.strip()
        if not query or len(query) > self.settings.query_max_chars:
            raise ValueError("Search query must be nonempty and within the configured limit")
        if self._closed:
            raise RuntimeError("Retriever is closed")
        # ponytail: one active search per job process; replace only if measured throughput needs it.
        if self._pending is not None and not self._pending.done():
            raise RetrievalBusy("Retrieval is still processing a previous request")
        self._pending = self._executor.submit(self._search_sync, query)
        wrapped = asyncio.wrap_future(self._pending)
        # Consume failures even when caller cancellation/timeout detaches from the work.
        wrapped.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        return await asyncio.wait_for(asyncio.shield(wrapped), self.settings.search_timeout)

    def close(self) -> None:
        self._closed = True
        # Running native inference cannot be cancelled. The job process owns its lifetime.
        self._executor.shutdown(wait=False, cancel_futures=True)
