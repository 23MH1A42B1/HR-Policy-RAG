"""Embeddings service abstraction and implementations.

Decouples the application from any specific embedding provider.
Supports local fastembed ONNX models as well as lightweight test mocks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math
from typing import List, Optional
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("app.embeddings.embedder")


class EmbeddingService(ABC):
    """Abstract base class defining the embedding interface."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding for a single text string / query."""
        pass

    def embed_query(self, text: str) -> List[float]:
        """Generate vector embedding for a search query (alias for embed_text)."""
        return self.embed_text(text)

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of document passages."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        pass


class FastEmbedService(EmbeddingService):
    """Local, CPU-optimized embedding service powered by FastEmbed (ONNX)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None
        # Standard dimension mappings for known FastEmbed models
        self._dimension_map = {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
        }
        self._dimension = self._dimension_map.get(model_name, 384)

    def _get_model(self):
        """Lazy loader for FastEmbed model to avoid startup delays."""
        if self._model is None:
            logger.info(f"Initializing FastEmbed model: '{self.model_name}'...")
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as exc:
                logger.error(f"Failed to load FastEmbed model '{self.model_name}': {exc}")
                raise RuntimeError(
                    f"Could not load FastEmbed model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """Embed a single query text."""
        model = self._get_model()
        clean = text.strip() or " "
        embeddings = list(model.embed([clean]))
        return [float(v) for v in embeddings[0]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document passages."""
        if not texts:
            return []
        model = self._get_model()
        clean_texts = [t.strip() or " " for t in texts]
        embeddings = list(model.embed(clean_texts))
        return [[float(v) for v in vector] for vector in embeddings]


class MockEmbeddingService(EmbeddingService):
    """Deterministic, lightweight embedding service for testing without model downloads."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> List[float]:
        """Generate a deterministic pseudo-random unit vector based on text hash."""
        vec = []
        for i in range(self._dimension):
            h = hashlib.sha256(f"{text}_{i}".encode("utf-8")).digest()
            val = (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)
        # Normalize to unit length (cosine similarity compatibility)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]

    def embed_text(self, text: str) -> List[float]:
        return self._generate_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]


def get_embedding_service(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    dimension: Optional[int] = None,
) -> EmbeddingService:
    """Factory to retrieve the configured embedding service."""
    settings = get_settings()
    active_provider = (provider or settings.embedding_provider).lower()
    active_model = model_name or settings.embedding_model
    active_dim = dimension or settings.embedding_dimension

    if active_provider == "fastembed":
        return FastEmbedService(model_name=active_model)
    elif active_provider == "mock":
        return MockEmbeddingService(dimension=active_dim)
    else:
        logger.warning(
            f"Unknown embedding provider '{active_provider}'. Falling back to FastEmbed."
        )
        return FastEmbedService(model_name=active_model)
