"""Embeddings module for HR Policy RAG Assistant."""

from app.embeddings.embedder import (
    EmbeddingService,
    FastEmbedService,
    MockEmbeddingService,
    get_embedding_service,
)

__all__ = [
    "EmbeddingService",
    "FastEmbedService",
    "MockEmbeddingService",
    "get_embedding_service",
]
