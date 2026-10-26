"""Semantic cache for HR Policy RAG Assistant.

Stores previous answers and embeddings. Uses cosine similarity to identify
semantically identical questions, while strictly invalidating stale policy versions.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("app.cache.semantic_cache")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if len(vec1) != len(vec2) or not vec1:
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCacheEntry(BaseModel):
    """Data model representing a cached question-answer pair."""

    question: str = Field(description="Original user question")
    embedding: List[float] = Field(description="Vector embedding of the question")
    answer: str = Field(description="Generated grounded answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Verified source citations")
    policy_version: str = Field(description="Policy version when answer was generated")
    timestamp: float = Field(default_factory=time.time, description="Creation timestamp (epoch seconds)")


class SemanticCache:
    """In-memory semantic cache with policy version invalidation and cosine matching."""

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        default_policy_version: Optional[str] = None,
        max_entries: int = 1000,
    ) -> None:
        settings = get_settings()
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(settings, "cache_similarity_threshold", 0.90)
        )
        self.default_policy_version = (
            default_policy_version
            if default_policy_version is not None
            else getattr(settings, "current_policy_version", "2026")
        )
        self.max_entries = max_entries
        self.entries: List[SemanticCacheEntry] = []

    def lookup(
        self,
        query_vector: List[float],
        current_version: Optional[str] = None,
    ) -> Optional[SemanticCacheEntry]:
        """Lookup query embedding against cached entries.

        Rules:
            1. Calculate cosine similarity against all cached entries.
            2. Invalidate / skip any entry where policy_version does not match current_version.
            3. Return the highest scoring match exceeding similarity_threshold.
        """
        if not self.entries or not query_vector:
            return None

        active_version = current_version or self.default_policy_version
        best_entry: Optional[SemanticCacheEntry] = None
        best_sim = -1.0

        for entry in self.entries:
            # Check version freshness
            if entry.policy_version != active_version:
                logger.debug(
                    f"Skipping stale cache entry (entry_version='{entry.policy_version}', "
                    f"active_version='{active_version}')"
                )
                continue

            sim = cosine_similarity(query_vector, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self.similarity_threshold:
            logger.info(
                f"Semantic cache HIT (similarity={best_sim:.4f} >= {self.similarity_threshold:.2f}, "
                f"cached_q='{best_entry.question[:50]}...')"
            )
            return best_entry

        if best_entry is not None:
            logger.debug(
                f"Semantic cache MISS (best similarity={best_sim:.4f} < {self.similarity_threshold:.2f})"
            )
        else:
            logger.debug("Semantic cache MISS (no fresh version entries found)")

        return None

    def store(
        self,
        question: str,
        embedding: List[float],
        answer: str,
        sources: List[Dict[str, Any]],
        policy_version: Optional[str] = None,
    ) -> SemanticCacheEntry:
        """Store a new question-answer pair in the cache."""
        version = policy_version or self.default_policy_version
        entry = SemanticCacheEntry(
            question=question.strip(),
            embedding=embedding,
            answer=answer,
            sources=sources,
            policy_version=version,
            timestamp=time.time(),
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
        logger.info(
            f"Stored query in semantic cache (total_entries={len(self.entries)}, "
            f"version='{version}', q='{question[:50]}...')"
        )
        return entry

    def clear(self) -> None:
        """Clear all cache entries."""
        self.entries.clear()
        logger.info("Semantic cache cleared.")

    def size(self) -> int:
        """Return count of cached entries."""
        return len(self.entries)


# Global singleton instance
_GLOBAL_CACHE: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Return singleton SemanticCache instance."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = SemanticCache()
    return _GLOBAL_CACHE
