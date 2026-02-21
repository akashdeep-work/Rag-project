from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import math
import time
from sentence_transformers import CrossEncoder
from AiModel.models import get_optimal_device


@dataclass
class SearchCandidate:
    """Represents a candidate returned by the ANN index."""

    chunk_id: str
    vector_score: float
    metadata: Dict[str, Any]
    lexical_score: float | None = None
    score: float | None = None


class Reranker:
    """Interface for re-ranking strategies."""

    def rerank(self, query: str, candidates: List[SearchCandidate], top_k: int) -> List[SearchCandidate]:
        raise NotImplementedError


class LinearReranker(Reranker):
    """Combine vector scores and metadata to produce a stable ordering.

    The weights are intentionally simple so additional strategies can be plugged
    in later without changing the Indexer contract.
    """

    def __init__(
        self,
        vector_weight: float = 0.7,
        recency_weight: float = 0.2,
        file_type_weight: float = 0.1,
    ) -> None:
        self.vector_weight = vector_weight
        self.recency_weight = recency_weight
        self.file_type_weight = file_type_weight

    def _normalize_score(self, score: float) -> float:
        # FAISS returns either distance or similarity; higher is better for cosine
        return float(score)

    def _recency_score(self, metadata: Dict[str, Any]) -> float:
        modified = metadata.get("modified_at")
        if not modified:
            return 0.0
        # Apply exponential decay over ~30 days
        days_ago = max((time.time() - float(modified)) / 86400.0, 0.0)
        return math.exp(-days_ago / 30.0)

    def _file_type_score(self, metadata: Dict[str, Any]) -> float:
        file_type = (metadata.get("file_type") or "text").lower()
        if file_type == "media":
            return 0.8
        if file_type == "text":
            return 1.0
        return 0.5

    def rerank(self, query: str, candidates: List[SearchCandidate], top_k: int) -> List[SearchCandidate]:
        scored: List[SearchCandidate] = []
        for cand in candidates:
            vector_component = self._normalize_score(cand.vector_score)
            recency_component = self._recency_score(cand.metadata)
            file_type_component = self._file_type_score(cand.metadata)

            final_score = (
                vector_component * self.vector_weight
                + recency_component * self.recency_weight
                + file_type_component * self.file_type_weight
            )
            scored.append(SearchCandidate(
                chunk_id=cand.chunk_id,
                vector_score=cand.vector_score,
                metadata=cand.metadata,
                score=final_score,
            ))

        scored.sort(key=lambda c: c.score if c.score is not None else -float("inf"), reverse=True)
        return scored[:top_k]

class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        device: str = get_optimal_device(),
    ):
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: List[SearchCandidate],
        top_k: int,
    ) -> List[SearchCandidate]:
        if not candidates:
            return []

        pairs = [
            (query, c.metadata.get("text", ""))
            for c in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        for c, s in zip(candidates, scores):
            c.score = float(s)

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]
