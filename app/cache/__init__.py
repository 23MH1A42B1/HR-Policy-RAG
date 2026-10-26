"""Semantic cache module for HR Policy RAG Assistant."""

from app.cache.semantic_cache import (
    SemanticCache,
    SemanticCacheEntry,
    cosine_similarity,
    get_semantic_cache,
)

__all__ = [
    "SemanticCache",
    "SemanticCacheEntry",
    "cosine_similarity",
    "get_semantic_cache",
]
