"""Admin document ingestion pipeline for HR Policy RAG Assistant.

Implements the end-to-end admin lifecycle:
GCS -> Parser -> Chunker -> Embeddings -> Qdrant -> Available for RAG
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

from app.embeddings.embedder import get_embedding_service
from app.ingestion.chunker import DocumentChunker
from app.ingestion.loader import DocumentLoader, ExtractedPage
from app.ingestion.metadata import DocumentChunk
from app.retrieval.bm25_search import get_bm25_index
from app.retrieval.qdrant_store import get_qdrant_store
from app.storage.gcs_storage import (
    VALID_POLICY_CATEGORIES,
    GCSStorageService,
    StoredPolicyMetadata,
    get_gcs_storage,
)
from app.utils.logging import get_logger

logger = get_logger("app.ingestion.admin_pipeline")


class AdminPolicyPipeline:
    """Orchestrates production policy storage and dynamic ingestion into Qdrant & BM25."""

    def __init__(
        self,
        gcs_storage: Optional[GCSStorageService] = None,
        qdrant_store=None,
        embedding_service=None,
    ) -> None:
        self.gcs_storage = gcs_storage or get_gcs_storage()
        self.qdrant_store = qdrant_store or get_qdrant_store()
        self.embedding_service = embedding_service or get_embedding_service()
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()

    def upload_and_index_policy(
        self,
        file_bytes: bytes,
        filename: str,
        category: str,
        version: str = "2026",
    ) -> Dict[str, Any]:
        """Upload raw document to GCS, parse, chunk, embed, and index into Qdrant + BM25."""
        clean_filename = Path(filename).name
        clean_category = category.lower().strip()
        if clean_category not in VALID_POLICY_CATEGORIES:
            clean_category = "general"

        destination_path = f"policies/{clean_category}/{clean_filename}"
        logger.info(f"Admin initiating upload for '{clean_filename}' -> '{destination_path}'")

        # 1. Upload to GCS
        gcs_uri = self.gcs_storage.upload(file_bytes, destination_path)

        # 2. Parse document via temporary file
        suffix = Path(clean_filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            pages: List[ExtractedPage] = self.loader.load_file(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # Update page document names to the authoritative filename
        for p in pages:
            p.document = clean_filename

        # 3. Chunk pages
        chunks: List[DocumentChunk] = self.chunker.chunk_document_pages(pages)
        for c in chunks:
            c.policy_type = clean_category
            c.version = version

        # 4. Generate embeddings
        texts = [c.text for c in chunks]
        embeddings: List[List[float]] = []
        if texts:
            embeddings = self.embedding_service.embed_documents(texts)

        # 5. Upsert into Qdrant
        upserted_count = 0
        if chunks and embeddings:
            try:
                self.qdrant_store.create_collection(vector_size=len(embeddings[0]))
            except Exception as exc:
                logger.debug(f"Collection check/creation notice: {exc}")
            upserted_count = self.qdrant_store.upsert_documents(chunks, embeddings)

        # 6. Index into BM25
        bm25_index = get_bm25_index()
        bm25_index.add_chunks(chunks)

        # 7. Persist copy to data/raw for local offline reference
        raw_target = Path("data/raw") / clean_filename
        try:
            raw_target.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_target, "wb") as f_raw:
                f_raw.write(file_bytes)
        except Exception as e_raw:
            logger.debug(f"Notice saving copy to data/raw: {e_raw}")

        # 8. Reset active agent singleton and clear cache so queries immediately use new document
        try:
            from app.agent.agent import reset_agent
            reset_agent()
            from app.cache.semantic_cache import get_semantic_cache
            get_semantic_cache().clear()
        except Exception as exc:
            logger.debug(f"Cache/agent reset notice: {exc}")

        logger.info(
            f"Successfully processed policy '{clean_filename}': "
            f"{len(pages)} pages, {len(chunks)} chunks indexed into Qdrant & BM25."
        )

        return {
            "status": "success",
            "document": clean_filename,
            "category": clean_category,
            "gcs_uri": gcs_uri,
            "pages_extracted": len(pages),
            "chunks_indexed": len(chunks),
            "qdrant_points_upserted": upserted_count,
        }

    def delete_policy(self, policy_path: str) -> Dict[str, Any]:
        """Delete document from GCS, purge vectors from Qdrant, and remove from BM25."""
        filename = Path(policy_path).name
        logger.info(f"Admin deleting policy: path='{policy_path}', filename='{filename}'")

        # 1. Delete from GCS
        gcs_deleted = self.gcs_storage.delete(policy_path)

        # 2. Purge from Qdrant
        qdrant_deleted = self.qdrant_store.delete_document(filename)

        # 3. Remove from BM25
        bm25_removed = get_bm25_index().remove_document(filename)

        # 4. Reset agent singleton and clear cache
        try:
            from app.agent.agent import reset_agent
            reset_agent()
            from app.cache.semantic_cache import get_semantic_cache
            get_semantic_cache().clear()
        except Exception as exc:
            logger.debug(f"Reset agent/cache notice on delete: {exc}")

        return {
            "status": "deleted",
            "path": policy_path,
            "document": filename,
            "gcs_deleted": gcs_deleted,
            "qdrant_purged": qdrant_deleted,
            "bm25_chunks_removed": bm25_removed,
        }

    def reindex_policy(self, policy_path: str, version: str = "2026") -> Dict[str, Any]:
        """Download document from GCS and re-run parsing, chunking, and indexing."""
        logger.info(f"Admin re-indexing policy: '{policy_path}'")
        file_bytes = self.gcs_storage.download(policy_path)
        filename = Path(policy_path).name
        category = self.gcs_storage._extract_category(policy_path)

        # Purge existing Qdrant points and BM25 chunks first
        self.qdrant_store.delete_document(filename)
        get_bm25_index().remove_document(filename)

        # Re-run upload and index flow
        result = self.upload_and_index_policy(
            file_bytes=file_bytes,
            filename=filename,
            category=category,
            version=version,
        )
        result["reindexed"] = True
        return result

    def list_policies(self, prefix: str = "policies/") -> List[StoredPolicyMetadata]:
        """List all policies stored in GCS under the given prefix."""
        return self.gcs_storage.list(prefix=prefix)

    def get_ingestion_status(self) -> Dict[str, Any]:
        """Retrieve aggregated ingestion metrics across GCS, Qdrant, and BM25."""
        policies = self.list_policies()
        category_counts: Dict[str, int] = {}
        for p in policies:
            category_counts[p.category] = category_counts.get(p.category, 0) + 1

        qdrant_health = self.qdrant_store.health_check()
        bm25_chunks = len(get_bm25_index().chunks)

        return {
            "storage_status": self.gcs_storage.integration_status,
            "bucket_name": self.gcs_storage.bucket_name,
            "total_documents_gcs": len(policies),
            "category_breakdown": category_counts,
            "qdrant_collection": self.qdrant_store.collection_name,
            "qdrant_health": qdrant_health,
            "bm25_indexed_chunks": bm25_chunks,
        }


# Global singleton pipeline
_GLOBAL_ADMIN_PIPELINE: Optional[AdminPolicyPipeline] = None


def get_admin_pipeline() -> AdminPolicyPipeline:
    """Return singleton AdminPolicyPipeline instance."""
    global _GLOBAL_ADMIN_PIPELINE
    if _GLOBAL_ADMIN_PIPELINE is None:
        _GLOBAL_ADMIN_PIPELINE = AdminPolicyPipeline()
    return _GLOBAL_ADMIN_PIPELINE
