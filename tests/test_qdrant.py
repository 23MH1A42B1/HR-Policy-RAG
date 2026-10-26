"""Tests for Qdrant vector store, dense search, and error handling."""

from unittest import mock
import pytest
from app.ingestion.metadata import DocumentChunk
from app.retrieval.qdrant_store import (
    DenseSearchResult,
    QdrantConfigurationError,
    QdrantStore,
)


def test_missing_qdrant_url_raises_configuration_error():
    """Verify that attempting to initialize QdrantStore without URL raises clean error."""
    with mock.patch.dict("os.environ", {"QDRANT_URL": ""}, clear=True):
        with pytest.raises(QdrantConfigurationError) as excinfo:
            QdrantStore(url="")
        assert "Qdrant URL is not configured" in str(excinfo.value)


def test_qdrant_store_lifecycle_in_memory():
    """Verify collection creation, upsert, search, get, and delete in in-memory Qdrant."""
    store = QdrantStore(url=":memory:", collection_name="test_hr_policies")

    # 1. Health check
    assert store.health_check() is True

    # 2. Collection creation
    created = store.create_collection(vector_size=4, distance="Cosine")
    assert created is True
    # Creating again without recreate should return False
    assert store.create_collection(vector_size=4, distance="Cosine") is False

    # 3. Create test chunks
    chunk1 = DocumentChunk(
        chunk_id="doc1_p1_c001",
        text="Employees receive 20 days of paid annual leave.",
        document="leave_policy.pdf",
        page=1,
        section="Annual Leave",
        policy_type="leave",
        version="2026",
        metadata={"char_count": 47},
    )
    chunk2 = DocumentChunk(
        chunk_id="doc2_p1_c001",
        text="Core collaboration hours are 10:00 AM to 4:00 PM.",
        document="remote_work_policy.docx",
        page=1,
        section="Core Hours",
        policy_type="remote_work",
        version="2026",
        metadata={"char_count": 49},
    )

    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    # 4. Upsert documents
    upserted = store.upsert_documents([chunk1, chunk2], embeddings)
    assert upserted == 2

    # 5. Search dense (query similar to chunk 1)
    results = store.search_dense(query_vector=[0.9, 0.1, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    top_result = results[0]
    assert isinstance(top_result, DenseSearchResult)
    assert top_result.document == "leave_policy.pdf"
    assert top_result.page == 1
    assert top_result.section == "Annual Leave"
    assert "20 days" in top_result.text
    assert top_result.score > 0.8
    assert top_result.metadata.get("char_count") == 47

    # 6. Search dense with policy_type filter
    filtered_results = store.search_dense(
        query_vector=[0.9, 0.1, 0.0, 0.0],
        top_k=2,
        policy_type="remote_work",
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].document == "remote_work_policy.docx"

    # 7. Get document by chunk_id
    doc = store.get_document("doc1_p1_c001")
    assert doc is not None
    assert doc.document == "leave_policy.pdf"
    assert store.get_document("non_existent_id") is None

    # 8. Delete document
    store.delete_document("leave_policy.pdf")
    remaining = store.search_dense(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(remaining) == 1
    assert remaining[0].document == "remote_work_policy.docx"


def test_qdrant_store_local_disk_path(tmp_path):
    """Verify QdrantStore initializes and creates collections in embedded local path mode."""
    db_path = str(tmp_path / "test_qdrant_db")
    store = QdrantStore(url=db_path, collection_name="test_embedded")
    assert store.health_check() is True

    created = store.create_collection(vector_size=8)
    assert created is True
    assert store.client.collection_exists("test_embedded") is True


def test_upsert_mismatched_lengths_raises_error():
    """Verify upsert_documents validates chunk and embedding counts match."""
    store = QdrantStore(url=":memory:", collection_name="test_mismatch")
    store.create_collection(vector_size=4)

    chunk = DocumentChunk(
        chunk_id="test_01",
        text="Sample text",
        document="test.pdf",
        page=1,
        policy_type="general",
    )

    with pytest.raises(ValueError) as excinfo:
        store.upsert_documents([chunk], [])
    assert "Mismatch" in str(excinfo.value)
