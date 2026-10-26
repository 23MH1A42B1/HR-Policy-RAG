"""Controlled policy retrieval tools for HR Agent.

Restricts tool execution exclusively to pre-approved policy search and reranking operations.
Arbitrary tool, code, or command execution is strictly prevented.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field

from app.embeddings.embedder import EmbeddingService, get_embedding_service
from app.retrieval.bm25_search import BM25SearchIndex, get_bm25_index
from app.retrieval.hybrid_search import HybridRetriever, HybridSearchResult
from app.retrieval.qdrant_store import QdrantConfigurationError, QdrantStore, get_qdrant_store
from app.retrieval.reranker import BaseReranker, RerankedResult, get_reranker
from app.utils.logging import get_logger

logger = get_logger("app.agent.tools")


class PolicySearchResult(BaseModel):
    """Container for controlled policy search results."""

    passages: List[RerankedResult] = Field(description="Top reranked policy passages")
    candidates_count: int = Field(description="Total hybrid candidates retrieved before reranking")
    reranker_used: str = Field(description="Reranker identifier")
    query_used: str = Field(description="Search query executed")


class PolicySearchTool:
    """Controlled policy retrieval tool executing Dense + BM25 search followed by Jina reranking."""

    name: str = "search_hr_policies"
    description: str = (
        "Search authoritative company HR policy documents (leave, remote work, expenses, attendance, conduct) "
        "using hybrid retrieval and cross-encoder reranking."
    )

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[BaseReranker] = None,
        store: Optional[QdrantStore] = None,
        bm25_index: Optional[BM25SearchIndex] = None,
        embedder: Optional[EmbeddingService] = None,
    ) -> None:
        self.store = store
        self.bm25_index = bm25_index
        self.embedder = embedder

        if retriever is not None:
            self.retriever = retriever
        else:
            # Build retriever safely
            active_store: Optional[QdrantStore] = self.store
            if active_store is None:
                try:
                    active_store = get_qdrant_store()
                except (QdrantConfigurationError, Exception) as exc:
                    logger.warning(f"Qdrant store initialization note: {exc}. Proceeding with BM25.")
                    active_store = None
            active_bm25 = self.bm25_index or get_bm25_index()
            active_embedder = self.embedder or get_embedding_service()
            self.retriever = HybridRetriever(
                store=active_store,
                bm25_index=active_bm25,
                embedder=active_embedder,
            )

        self.reranker = reranker or get_reranker()

    def execute(
        self,
        query: str,
        policy_type: Optional[str] = None,
        status: Optional[str] = None,
        policy_id: Optional[str] = None,
        top_k: int = 5,
        candidate_k: int = 20,
        trace: Optional[Any] = None,
    ) -> PolicySearchResult:
        """Execute strictly controlled policy search.

        Args:
            query: Policy question or search terms.
            policy_type: Optional category filter (e.g. 'leave', 'remote', 'expense', 'attendance').
            status: Optional lifecycle filter ('active' or 'archived').
            policy_id: Optional specific policy code filter (e.g. 'HR-LEAVE-001').
            top_k: Final top passages to return after reranking (default: 5).
            candidate_k: Hybrid candidate pool size (default: 20).
            trace: Optional RAGPipelineTrace for step-by-step latency profiling.
        """
        clean_q = query.strip()
        if not clean_q:
            return PolicySearchResult(passages=[], candidates_count=0, reranker_used="none", query_used=query)

        logger.info(f"Tool [{self.name}] executing search for: '{clean_q[:60]}...' (status={status}, policy_id={policy_id})")

        # 1. Retrieval (Dense + BM25)
        if trace is not None and hasattr(trace, "measure_step"):
            with trace.measure_step("retrieval") as step_meta:
                dense_results, bm25_results = self.retriever.retrieve_raw(
                    query=clean_q,
                    top_candidates=candidate_k,
                    policy_type=policy_type,
                    status=status,
                    policy_id=policy_id,
                )
                step_meta["dense_candidates"] = len(dense_results)
                step_meta["bm25_candidates"] = len(bm25_results)

            with trace.measure_step("fusion") as step_meta:
                candidates: List[HybridSearchResult] = self.retriever.fuse_results(
                    dense_results=dense_results,
                    bm25_results=bm25_results,
                    top_candidates=candidate_k,
                    status=status,
                )
                step_meta["fusion_strategy"] = getattr(self.retriever, "fusion_strategy", "rrf")
                step_meta["fused_candidates"] = len(candidates)
                trace.retrieval_scores = [round(c.score, 4) for c in candidates[:top_k]]
        else:
            candidates = self.retriever.retrieve_hybrid(
                query=clean_q,
                top_candidates=candidate_k,
                policy_type=policy_type,
                status=status,
                policy_id=policy_id,
            )

        if not candidates:
            return PolicySearchResult(passages=[], candidates_count=0, reranker_used="none", query_used=clean_q)

        # 2. Cross-Encoder Reranking
        rerank_step = trace.measure_step("reranking") if (trace is not None and hasattr(trace, "measure_step")) else None
        try:
            if rerank_step:
                with rerank_step as step_meta:
                    reranked: List[RerankedResult] = self.reranker.rerank(
                        query=clean_q,
                        candidates=candidates,
                        top_k=top_k,
                    )
                    step_meta["reranker"] = reranked[0].reranker if reranked else "none"
                    step_meta["top_k"] = len(reranked)
                    trace.reranker_scores = [round(r.score, 4) for r in reranked]
            else:
                reranked = self.reranker.rerank(
                    query=clean_q,
                    candidates=candidates,
                    top_k=top_k,
                )
        except Exception as exc:
            logger.warning(f"Reranking encountered error: {exc}. Using top candidate fallback.")
            reranked = [
                RerankedResult(
                    chunk_id=c.chunk_id,
                    document=c.document,
                    page=c.page,
                    section=c.section,
                    text=c.text,
                    score=c.score,
                    reranker="none (tool exception fallback)",
                    metadata=c.metadata,
                )
                for c in candidates[:top_k]
            ]
            if trace is not None and hasattr(trace, "errors"):
                trace.errors.append(f"[reranking] {exc}")

        reranker_desc = reranked[0].reranker if reranked else "none"

        return PolicySearchResult(
            passages=reranked,
            candidates_count=len(candidates),
            reranker_used=reranker_desc,
            query_used=clean_q,
        )
