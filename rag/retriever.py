"""Phase 2 hybrid retrieval (dense + BM25 with RRF) and cross-encoder reranking."""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from agent.config import Settings
from rag.manifest import COLLECTION, read_active


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


@dataclass(frozen=True)
class SearchResult:
    evidence: tuple[Evidence, ...]
    knowledge_version: str

    @property
    def context(self) -> str:
        return "\n\n".join(item.text for item in self.evidence)


class RetrievalBusy(RuntimeError):
    """A previous timed-out search still occupies the inference worker."""


class Retriever:
    def __init__(self, settings: Settings) -> None:
        # Validate on-disk state before Chroma can create anything.
        path, manifest = read_active(settings)
        import chromadb
        import torch
        from chromadb.config import Settings as ChromaSettings
        from sentence_transformers import CrossEncoder, SentenceTransformer

        torch.set_num_threads(settings.inference_threads)
        self.settings = settings
        self.knowledge_version = manifest["version"]
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
        self._lookup: dict[str, tuple[str, dict]] = {
            cid: (doc, meta or {})
            for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"], strict=True)
        }
        self._bm25 = BM25Index(self._all_docs)

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

    def _search_sync(self, query: str) -> SearchResult:
        # 1. Dense retrieval
        vector = self._embeddings.encode(query, normalize_embeddings=True).tolist()
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
            return SearchResult((), self.knowledge_version)

        # 4. Cross-encoder reranking
        scores = self._reranker.predict([(query, row[1]) for row in candidates])
        accepted = []
        for row, score in zip(candidates, scores, strict=True):
            score = float(score)
            if math.isfinite(score) and score >= self.settings.rerank_min_score:
                accepted.append((row, score))
        accepted.sort(key=lambda item: (-item[1], item[0][0]))

        # 5. Pack evidence chunks within context_max_chars budget
        evidence = []
        used = 0
        for (chunk_id, text, meta), score in accepted:
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
                )
            )
            used += cost
            if len(evidence) == self.settings.retrieval_top_n:
                break
        return SearchResult(tuple(evidence), self.knowledge_version)

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
