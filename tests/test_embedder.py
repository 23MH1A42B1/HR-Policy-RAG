"""Tests for embedding services abstraction and implementations."""

from app.embeddings.embedder import (
    FastEmbedService,
    MockEmbeddingService,
    get_embedding_service,
)


def test_mock_embedding_service_dimensions():
    """Verify MockEmbeddingService produces vectors of requested dimensionality."""
    dim = 256
    service = MockEmbeddingService(dimension=dim)
    assert service.dimension == dim

    vec = service.embed_text("Annual leave policy guidelines")
    assert len(vec) == dim
    assert isinstance(vec[0], float)


def test_mock_embedding_service_deterministic():
    """Verify identical text produces identical embedding vectors."""
    service = MockEmbeddingService(dimension=128)
    v1 = service.embed_text("Remote work setup reimbursement")
    v2 = service.embed_text("Remote work setup reimbursement")
    v3 = service.embed_text("Different query text")

    assert v1 == v2
    assert v1 != v3


def test_mock_embedding_service_batch():
    """Verify embed_documents batch processing returns list of vectors."""
    service = MockEmbeddingService(dimension=64)
    texts = ["Paragraph 1", "Paragraph 2", "Paragraph 3"]
    vectors = service.embed_documents(texts)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 64


def test_fastembed_service_dimension_and_factory():
    """Verify FastEmbedService dimension mapping and factory resolution."""
    service = FastEmbedService(model_name="BAAI/bge-small-en-v1.5")
    assert service.dimension == 384

    resolved = get_embedding_service(provider="mock", dimension=128)
    assert isinstance(resolved, MockEmbeddingService)
    assert resolved.dimension == 128
