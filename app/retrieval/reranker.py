"""Jina AI cross-encoder reranker integration with documented degraded fallback mode.

Reranks hybrid search candidates (e.g. Top 20) down to the most relevant
passages (Top 5) before sending to the LLM context window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from app.config import get_settings
from app.retrieval.hybrid_search import HybridSearchResult
from app.utils.logging import get_logger

logger = get_logger("app.retrieval.reranker")


class RerankedResult(BaseModel):
    """Result model representing a reranked document passage."""

    chunk_id: str = Field(description="Unique chunk identifier")
    document: str = Field(description="Original document filename")
    page: int = Field(description="1-indexed page or slide number")
    section: str = Field(description="Section heading")
    text: str = Field(description="Passage text content")
    score: float = Field(description="Final relevance score (from reranker or hybrid fusion)")
    reranker: str = Field(description="Reranker identifier (e.g. 'jina', 'none (degraded)')")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class BaseReranker(ABC):
    """Abstract interface for passage reranking services."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[HybridSearchResult],
        top_k: int = 5,
    ) -> List[RerankedResult]:
        """Rerank candidates according to semantic query relevance."""
        pass


class JinaReranker(BaseReranker):
    """Production cross-encoder reranker using Jina AI REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.jina_api_key
        self.model = model or getattr(settings, "jina_reranker_model", "jina-reranker-v2-base-multilingual")

        if not self.api_key:
            raise ValueError(
                "Jina API key is not configured. Please set 'JINA_API_KEY' in your environment or .env file."
            )

    def rerank(
        self,
        query: str,
        candidates: List[HybridSearchResult],
        top_k: int = 5,
    ) -> List[RerankedResult]:
        """Call Jina Reranker endpoint to score and sort candidates."""
        if not candidates:
            return []

        clean_query = query.strip()
        doc_texts = [c.text for c in candidates]
        n_results = min(top_k, len(candidates))

        url = "https://api.jina.ai/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "query": clean_query,
            "documents": doc_texts,
            "top_n": n_results,
        }

        logger.info(f"Calling Jina Reranker ('{self.model}') on {len(candidates)} candidates for: '{clean_query[:50]}...'")

        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()

            reranked_items: List[RerankedResult] = []
            for item in data.get("results", []):
                idx = item["index"]
                orig = candidates[idx]
                rel_score = float(item["relevance_score"])
                reranked_items.append(
                    RerankedResult(
                        chunk_id=orig.chunk_id,
                        document=orig.document,
                        page=orig.page,
                        section=orig.section,
                        text=orig.text,
                        score=round(rel_score, 4),
                        reranker=f"jina ({self.model})",
                        metadata=orig.metadata,
                    )
                )

            logger.info(f"Jina successfully reranked {len(candidates)} down to {len(reranked_items)} passages.")
            return reranked_items

        except Exception as exc:
            logger.error(f"Jina reranker call failed: {exc}. Falling back to degraded ranking.")
            # Fallback to top_k of candidates without pretending Jina worked
            return [
                RerankedResult(
                    chunk_id=c.chunk_id,
                    document=c.document,
                    page=c.page,
                    section=c.section,
                    text=c.text,
                    score=c.score,
                    reranker="none (jina call failed, degraded)",
                    metadata=c.metadata,
                )
                for c in candidates[:top_k]
            ]


class DegradedReranker(BaseReranker):
    """Documented fallback reranker when Jina credentials are not configured.

    Does not call an external service and clearly reports that Jina was not used.
    """

    def rerank(
        self,
        query: str,
        candidates: List[HybridSearchResult],
        top_k: int = 5,
    ) -> List[RerankedResult]:
        logger.info("Jina credentials not configured; using documented degraded mode (no reranking applied).")
        return [
            RerankedResult(
                chunk_id=c.chunk_id,
                document=c.document,
                page=c.page,
                section=c.section,
                text=c.text,
                score=c.score,
                reranker="none (degraded fallback)",
                metadata=c.metadata,
            )
            for c in candidates[:top_k]
        ]


class MockReranker(BaseReranker):
    """Deterministic mock reranker for unit testing."""

    def rerank(
        self,
        query: str,
        candidates: List[HybridSearchResult],
        top_k: int = 5,
    ) -> List[RerankedResult]:
        # Sort by candidate score descending and take top_k
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        return [
            RerankedResult(
                chunk_id=c.chunk_id,
                document=c.document,
                page=c.page,
                section=c.section,
                text=c.text,
                score=round(c.score * 1.1, 4),
                reranker="mock",
                metadata=c.metadata,
            )
            for c in sorted_candidates[:top_k]
        ]


def get_reranker(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    force_degraded: bool = False,
) -> BaseReranker:
    """Factory retrieving JinaReranker if configured, or DegradedReranker otherwise."""
    if force_degraded:
        return DegradedReranker()

    settings = get_settings()
    key = api_key or settings.jina_api_key

    if not key:
        logger.info("No JINA_API_KEY detected. Initializing DegradedReranker (documented degraded mode).")
        return DegradedReranker()

    return JinaReranker(api_key=key, model=model)
