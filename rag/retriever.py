"""Phase 1 dense retrieval with provisional rejection; hybrid evaluation is Phase 2."""

from __future__ import annotations

import asyncio
import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from agent.config import Settings
from rag.manifest import COLLECTION, read_active


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
        vector = self._embeddings.encode(query, normalize_embeddings=True).tolist()
        found = self._collection.query(
            query_embeddings=[vector],
            n_results=min(self.settings.retrieval_k, self._count),
            include=["documents", "metadatas"],
        )
        ids = found["ids"][0]
        documents = (found["documents"] or [[]])[0]
        metadata = (found["metadatas"] or [[]])[0]
        candidates = [
            (chunk_id, text, meta or {})
            for chunk_id, text, meta in zip(ids, documents, metadata, strict=True)
            if text and text.strip()
        ]
        if not candidates:
            return SearchResult((), self.knowledge_version)
        # Rerank even a single candidate; do not bypass acceptance after a provider/model error.
        scores = self._reranker.predict([(query, row[1]) for row in candidates])
        accepted = []
        for row, score in zip(candidates, scores, strict=True):
            score = float(score)
            if math.isfinite(score) and score >= self.settings.rerank_min_score:
                accepted.append((row, score))
        accepted.sort(key=lambda item: (-item[1], item[0][0]))
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
