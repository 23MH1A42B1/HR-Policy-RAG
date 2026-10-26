"""Qdrant vector store management and dense retrieval.

Manages collection lifecycle, document upserts with metadata payloads,
document deletion, dense vector similarity search, and health checks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.ingestion.metadata import DocumentChunk
from app.utils.logging import get_logger

logger = get_logger("app.retrieval.qdrant_store")


class QdrantConfigurationError(ValueError):
    """Raised when Qdrant connection credentials or endpoint URL are not properly configured."""
    pass


class DenseSearchResult(BaseModel):
    """Result model returned by dense vector search."""

    document: str = Field(description="Original document filename")
    page: int = Field(description="1-indexed page or slide number")
    section: str = Field(description="Section heading or topic")
    text: str = Field(description="Passage text content")
    score: float = Field(description="Vector similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")


_SHARED_QDRANT_CLIENTS: Dict[str, QdrantClient] = {}


class QdrantStore:
    """Manages Qdrant vector database operations for HR policy retrieval."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        client: Optional[QdrantClient] = None,
    ) -> None:
        settings = get_settings()
        self.url = url if url is not None else settings.qdrant_url
        self.api_key = api_key if api_key is not None else settings.qdrant_api_key
        self.collection_name = collection_name if collection_name is not None else settings.qdrant_collection_name

        if client is not None:
            self.client = client
        else:
            if not self.url:
                raise QdrantConfigurationError(
                    "Qdrant URL is not configured. Please set 'QDRANT_URL' in your environment or .env file."
                )

            logger.info(f"Connecting to Qdrant at: {self.url} (collection: '{self.collection_name}')")
            try:
                if self.url == ":memory:":
                    self.client = QdrantClient(":memory:")
                elif self.url in _SHARED_QDRANT_CLIENTS:
                    self.client = _SHARED_QDRANT_CLIENTS[self.url]
                elif self.url.startswith("http://") or self.url.startswith("https://"):
                    self.client = QdrantClient(url=self.url, api_key=self.api_key or None)
                    _SHARED_QDRANT_CLIENTS[self.url] = self.client
                else:
                    # Embedded local disk persistence (e.g. data/qdrant_db)
                    self.client = QdrantClient(path=self.url)
                    _SHARED_QDRANT_CLIENTS[self.url] = self.client
            except Exception as exc:
                logger.error(f"Failed to connect to Qdrant at {self.url}: {exc}")
                raise

    def health_check(self) -> bool:
        """Verify that the Qdrant instance is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.warning(f"Qdrant health check failed: {exc}")
            return False

    def create_collection(
        self,
        vector_size: int = 384,
        distance: str = "Cosine",
        recreate: bool = False,
    ) -> bool:
        """Create the target collection if it does not already exist.

        Args:
            vector_size: Dimensionality of vector embeddings.
            distance: Distance metric ('Cosine', 'Euclid', 'Dot').
            recreate: If True, delete and recreate existing collection.

        Returns:
            True if a new collection was created, False if it already existed.
        """
        distance_enum = getattr(models.Distance, distance.upper(), models.Distance.COSINE)

        existing_collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing_collections:
            if recreate:
                logger.info(f"Recreating existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
                return False

        logger.info(f"Creating collection '{self.collection_name}' (vector size: {vector_size}, distance: {distance})")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance_enum),
        )
        return True

    def upsert_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 64,
    ) -> int:
        """Upsert document chunks and their dense embeddings into Qdrant.

        Args:
            chunks: List of DocumentChunk models.
            embeddings: List of embedding vectors matching chunks.
            batch_size: Number of points to upsert per network call.

        Returns:
            Total count of points upserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        if not chunks:
            return 0

        total_upserted = 0
        points: List[models.PointStruct] = []

        for chunk, vector in zip(chunks, embeddings):
            # Derive a deterministic UUID string from chunk_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            company = getattr(chunk, "company", "XYZ Technologies Private Limited")
            policy_id = getattr(chunk, "policy_id", "GENERAL")
            status = getattr(chunk, "status", "active")
            version = getattr(chunk, "version", "2.0")
            effective_date = getattr(chunk, "effective_date", "2026-04-01")
            expiry_date = getattr(chunk, "expiry_date", None)
            access_level = getattr(chunk, "access_level", "employee")
            document_type = getattr(chunk, "document_type", "hr_policy")
            doc_name = getattr(chunk, "document_name", None) or chunk.document

            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document": chunk.document,
                "document_name": doc_name,
                "company": company,
                "policy_id": policy_id,
                "document_type": document_type,
                "version": version,
                "status": status,
                "effective_date": effective_date,
                "expiry_date": expiry_date,
                "access_level": access_level,
                "page": chunk.page,
                "section": chunk.section,
                "policy_type": chunk.policy_type,
                "metadata": chunk.metadata,
            }
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            total_upserted += len(batch)

        logger.info(f"Successfully upserted {total_upserted} chunk vector(s) into '{self.collection_name}'.")
        return total_upserted

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 5,
        policy_type: Optional[str] = None,
        status: Optional[str] = None,
        policy_id: Optional[str] = None,
    ) -> List[DenseSearchResult]:
        """Perform dense vector similarity search with metadata filtering.

        Args:
            query_vector: Vector representation of the search query.
            top_k: Number of nearest neighbors to retrieve.
            policy_type: Optional policy category filter (e.g. 'leave').
            status: Optional lifecycle filter ('active' or 'archived').
            policy_id: Optional exact policy code filter (e.g. 'HR-LEAVE-001').

        Returns:
            List of DenseSearchResult models sorted by relevance score.
        """
        filter_conditions = []
        if policy_type:
            filter_conditions.append(
                models.FieldCondition(
                    key="policy_type",
                    match=models.MatchValue(value=policy_type.lower()),
                )
            )
        if status:
            filter_conditions.append(
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value=status.lower()),
                )
            )
        if policy_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="policy_id",
                    match=models.MatchValue(value=policy_id.upper()),
                )
            )

        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

        # Standard query_points API in modern qdrant-client
        query_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        results: List[DenseSearchResult] = []
        for scored_point in query_response.points:
            payload = scored_point.payload or {}
            meta = dict(payload.get("metadata", {}))
            meta["chunk_id"] = payload.get("chunk_id", str(scored_point.id))
            meta["policy_type"] = payload.get("policy_type", "")
            meta["version"] = payload.get("version", "")
            meta["policy_id"] = payload.get("policy_id", "")
            meta["status"] = payload.get("status", "active")
            meta["company"] = payload.get("company", "XYZ Technologies Private Limited")
            meta["effective_date"] = payload.get("effective_date", "2026-04-01")
            meta["expiry_date"] = payload.get("expiry_date", None)
            meta["access_level"] = payload.get("access_level", "employee")
            meta["document_name"] = payload.get("document_name", payload.get("document", "unknown"))

            results.append(
                DenseSearchResult(
                    document=payload.get("document", "unknown"),
                    page=payload.get("page", 1),
                    section=payload.get("section", "General"),
                    text=payload.get("text", ""),
                    score=float(scored_point.score),
                    metadata=meta,
                )
            )

        return results

    def get_document(self, chunk_id: str) -> Optional[DenseSearchResult]:
        """Retrieve a single document chunk by its unique chunk_id."""
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchValue(value=chunk_id),
                )
            ]
        )
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=1,
            with_payload=True,
        )

        if not records:
            return None

        record = records[0]
        payload = record.payload or {}
        return DenseSearchResult(
            document=payload.get("document", "unknown"),
            page=payload.get("page", 1),
            section=payload.get("section", "General"),
            text=payload.get("text", ""),
            score=1.0,
            metadata=payload.get("metadata", {}),
        )

    def delete_document(self, document_name: str) -> bool:
        """Delete all chunk vectors belonging to a specific document name."""
        logger.info(f"Deleting document chunks for: '{document_name}' from '{self.collection_name}'")
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document",
                            match=models.MatchValue(value=document_name),
                        )
                    ]
                )
            ),
        )
        return True


_global_qdrant_store: Optional[QdrantStore] = None


def get_qdrant_store() -> QdrantStore:
    """Retrieve or initialize singleton QdrantStore instance."""
    global _global_qdrant_store
    if _global_qdrant_store is None:
        _global_qdrant_store = QdrantStore()
    return _global_qdrant_store

