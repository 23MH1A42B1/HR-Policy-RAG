"""Hybrid search combining dense vector retrieval and BM25 lexical retrieval.

Implements Reciprocal Rank Fusion (RRF) and Weighted Linear Score Fusion
to balance semantic search with exact keyword matching.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from app.config import get_settings
from app.embeddings.embedder import EmbeddingService, get_embedding_service
from app.retrieval.bm25_search import BM25Result, BM25SearchIndex, get_bm25_index
from app.retrieval.qdrant_store import DenseSearchResult, QdrantStore
from app.utils.logging import get_logger

logger = get_logger("app.retrieval.hybrid_search")

FusionStrategy = Literal["rrf", "weighted"]


class HybridSearchResult(BaseModel):
    """Unified result model representing a fused search candidate."""

    chunk_id: str = Field(description="Unique chunk identifier")
    document: str = Field(description="Document name")
    page: int = Field(description="1-indexed page or slide number")
    section: str = Field(description="Section heading")
    text: str = Field(description="Text passage content")
    score: float = Field(description="Fused hybrid relevance score")
    dense_rank: Optional[int] = Field(default=None, description="Rank in dense retrieval (1-indexed)")
    bm25_rank: Optional[int] = Field(default=None, description="Rank in BM25 retrieval (1-indexed)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


def reciprocal_rank_fusion(
    dense_results: List[DenseSearchResult],
    bm25_results: List[BM25Result],
    rrf_k: int = 60,
    top_k: int = 20,
) -> List[HybridSearchResult]:
    """Combine dense and BM25 results using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = sum_{m in models} 1.0 / (rrf_k + rank_m(d))
    """
    scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    bm25_ranks: Dict[str, int] = {}
    doc_data: Dict[str, Any] = {}

    # Helper to generate unique identifier key for chunk deduplication
    def get_key(doc: str, page: int, section: str, text: str) -> str:
        snippet = text[:60].strip()
        return f"{doc.lower()}_p{page}_{section.lower()}_{snippet}"

    # 1. Process Dense Results
    for rank, item in enumerate(dense_results, start=1):
        key = get_key(item.document, item.page, item.section, item.text)
        scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))
        dense_ranks[key] = rank
        if key not in doc_data:
            doc_data[key] = {
                "chunk_id": item.metadata.get("chunk_id", key),
                "document": item.document,
                "page": item.page,
                "section": item.section,
                "text": item.text,
                "metadata": item.metadata,
            }

    # 2. Process BM25 Results
    for rank, item in enumerate(bm25_results, start=1):
        key = get_key(item.document, item.page, item.section, item.text)
        scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))
        bm25_ranks[key] = rank
        if key not in doc_data:
            doc_data[key] = {
                "chunk_id": item.chunk_id,
                "document": item.document,
                "page": item.page,
                "section": item.section,
                "text": item.text,
                "metadata": item.metadata,
            }

    # 3. Sort by fused RRF score descending
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    results: List[HybridSearchResult] = []
    for key in sorted_keys[:top_k]:
        info = doc_data[key]
        results.append(
            HybridSearchResult(
                chunk_id=info["chunk_id"],
                document=info["document"],
                page=info["page"],
                section=info["section"],
                text=info["text"],
                score=round(scores[key], 6),
                dense_rank=dense_ranks.get(key),
                bm25_rank=bm25_ranks.get(key),
                metadata=info["metadata"],
            )
        )

    return results


def weighted_linear_fusion(
    dense_results: List[DenseSearchResult],
    bm25_results: List[BM25Result],
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
    top_k: int = 20,
) -> List[HybridSearchResult]:
    """Combine dense and BM25 scores using min-max normalized linear weighting."""
    scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    bm25_ranks: Dict[str, int] = {}
    doc_data: Dict[str, Any] = {}

    def get_key(doc: str, page: int, section: str, text: str) -> str:
        return f"{doc.lower()}_p{page}_{section.lower()}_{text[:60].strip()}"

    # Normalize dense scores
    d_max = max((r.score for r in dense_results), default=1.0)
    d_min = min((r.score for r in dense_results), default=0.0)
    d_range = (d_max - d_min) or 1.0

    for rank, item in enumerate(dense_results, start=1):
        key = get_key(item.document, item.page, item.section, item.text)
        norm_score = (item.score - d_min) / d_range
        scores[key] = scores.get(key, 0.0) + (dense_weight * norm_score)
        dense_ranks[key] = rank
        if key not in doc_data:
            doc_data[key] = {
                "chunk_id": item.metadata.get("chunk_id", key),
                "document": item.document,
                "page": item.page,
                "section": item.section,
                "text": item.text,
                "metadata": item.metadata,
            }

    # Normalize BM25 scores
    b_max = max((r.score for r in bm25_results), default=1.0)
    b_min = min((r.score for r in bm25_results), default=0.0)
    b_range = (b_max - b_min) or 1.0

    for rank, item in enumerate(bm25_results, start=1):
        key = get_key(item.document, item.page, item.section, item.text)
        norm_score = (item.score - b_min) / b_range
        scores[key] = scores.get(key, 0.0) + (bm25_weight * norm_score)
        bm25_ranks[key] = rank
        if key not in doc_data:
            doc_data[key] = {
                "chunk_id": item.chunk_id,
                "document": item.document,
                "page": item.page,
                "section": item.section,
                "text": item.text,
                "metadata": item.metadata,
            }

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    results: List[HybridSearchResult] = []
    for key in sorted_keys[:top_k]:
        info = doc_data[key]
        results.append(
            HybridSearchResult(
                chunk_id=info["chunk_id"],
                document=info["document"],
                page=info["page"],
                section=info["section"],
                text=info["text"],
                score=round(scores[key], 6),
                dense_rank=dense_ranks.get(key),
                bm25_rank=bm25_ranks.get(key),
                metadata=info["metadata"],
            )
        )

    return results


class HybridRetriever:
    """Orchestrates parallel dense and lexical retrieval followed by result fusion."""

    def __init__(
        self,
        store: Optional[QdrantStore] = None,
        bm25_index: Optional[BM25SearchIndex] = None,
        embedder: Optional[EmbeddingService] = None,
        fusion_strategy: Optional[FusionStrategy] = None,
        rrf_k: Optional[int] = None,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self.store = store
        self.bm25_index = bm25_index or get_bm25_index()
        self.embedder = embedder or get_embedding_service()
        self.fusion_strategy = fusion_strategy or getattr(settings, "fusion_strategy", "rrf")
        self.rrf_k = rrf_k or getattr(settings, "rrf_k", 60)
        self.dense_weight = dense_weight or getattr(settings, "dense_weight", 0.6)
        self.bm25_weight = bm25_weight or getattr(settings, "bm25_weight", 0.4)

    def retrieve_raw(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        top_candidates: int = 20,
        policy_type: Optional[str] = None,
        status: Optional[str] = None,
        policy_id: Optional[str] = None,
    ) -> tuple[List[DenseSearchResult], List[BM25Result]]:
        """Perform dense and lexical retrieval independently with metadata filtering."""
        clean_query = query.strip()
        if not clean_query:
            return [], []

        # 1. BM25 Lexical Retrieval
        bm25_results: List[BM25Result] = []
        try:
            bm25_results = self.bm25_index.search(
                query=clean_query,
                top_k=top_candidates,
                policy_type=policy_type,
                status=status,
                policy_id=policy_id,
            )
            logger.info(f"BM25 returned {len(bm25_results)} candidate(s).")
        except Exception as exc:
            logger.warning(f"BM25 retrieval encountered an issue: {exc}")

        # 2. Dense Vector Retrieval
        dense_results: List[DenseSearchResult] = []
        if self.store is not None and self.store.health_check():
            try:
                vec = query_vector or self.embedder.embed_text(clean_query)
                dense_results = self.store.search_dense(
                    query_vector=vec,
                    top_k=top_candidates,
                    policy_type=policy_type,
                    status=status,
                    policy_id=policy_id,
                )
                logger.info(f"Dense search returned {len(dense_results)} candidate(s).")
            except Exception as exc:
                logger.warning(f"Dense vector retrieval encountered an issue: {exc}")
        else:
            logger.info("Qdrant store not available; proceeding with BM25 results only.")

        return dense_results, bm25_results

    def fuse_results(
        self,
        dense_results: List[DenseSearchResult],
        bm25_results: List[BM25Result],
        top_candidates: int = 20,
        status: Optional[str] = None,
    ) -> List[HybridSearchResult]:
        """Combine dense and BM25 search results according to the configured fusion strategy."""
        if self.fusion_strategy == "weighted":
            fused = weighted_linear_fusion(
                dense_results=dense_results,
                bm25_results=bm25_results,
                dense_weight=self.dense_weight,
                bm25_weight=self.bm25_weight,
                top_k=top_candidates,
            )
        else:
            fused = reciprocal_rank_fusion(
                dense_results=dense_results,
                bm25_results=bm25_results,
                rrf_k=self.rrf_k,
                top_k=top_candidates,
            )

        # Enforce status filter on fused results if specified
        if status:
            fused = [
                r for r in fused
                if (r.metadata.get("status") or "active").lower() == status.lower()
            ]

        logger.info(f"Hybrid fusion produced {len(fused)} top candidate(s).")
        return fused

    def retrieve_hybrid(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        top_candidates: int = 20,
        policy_type: Optional[str] = None,
        status: Optional[str] = None,
        policy_id: Optional[str] = None,
    ) -> List[HybridSearchResult]:
        """Execute hybrid search combining dense vector search and BM25 lexical search."""
        clean_query = query.strip()
        if not clean_query:
            return []

        logger.info(f"Executing hybrid search for: '{clean_query[:60]}...' (target candidates: {top_candidates}, status: {status})")
        dense_results, bm25_results = self.retrieve_raw(
            query=clean_query,
            query_vector=query_vector,
            top_candidates=top_candidates,
            policy_type=policy_type,
            status=status,
            policy_id=policy_id,
        )
        return self.fuse_results(dense_results, bm25_results, top_candidates=top_candidates, status=status)
