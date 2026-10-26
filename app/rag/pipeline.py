"""Core RAG pipeline orchestrating hybrid retrieval, Jina reranking, prompt assembly, and generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from app.config import get_settings
from app.embeddings.embedder import EmbeddingService, get_embedding_service
from app.rag.citations import SourceCitation, extract_citations_from_results
from app.rag.generator import LLMConfigurationError, LLMService, get_llm_service
from app.rag.prompt import build_system_prompt, build_user_prompt
from app.retrieval.bm25_search import BM25SearchIndex, get_bm25_index
from app.retrieval.hybrid_search import HybridRetriever, HybridSearchResult
from app.retrieval.qdrant_store import (
    QdrantConfigurationError,
    QdrantStore,
    get_qdrant_store,
)
from app.retrieval.reranker import BaseReranker, RerankedResult, get_reranker
from app.utils.logging import get_logger

logger = get_logger("app.rag.pipeline")


def answer_question(
    query: str,
    session_id: Optional[str] = None,
    top_k: Optional[int] = None,
    candidate_k: Optional[int] = None,
    policy_type: Optional[str] = None,
    retriever: Optional[HybridRetriever] = None,
    reranker: Optional[BaseReranker] = None,
    store: Optional[QdrantStore] = None,
    bm25_index: Optional[BM25SearchIndex] = None,
    embedder: Optional[EmbeddingService] = None,
    llm: Optional[LLMService] = None,
) -> Dict[str, Any]:
    """Execute the production hybrid RAG pipeline for an employee query.

    Pipeline:
        Query -> Dense Search + BM25 Search -> Result Fusion (Top 20) -> Jina Reranker -> Top 5 -> LLM

    Args:
        query: Employee's question about HR policies.
        session_id: Optional conversation/session identifier.
        top_k: Final number of reranked passages to send to LLM (default: 5).
        candidate_k: Number of hybrid candidates to retrieve before reranking (default: 20).
        policy_type: Optional policy category filter (e.g. 'leave').
        retriever: Optional injected HybridRetriever.
        reranker: Optional injected BaseReranker (Jina or Degraded fallback).
        store: Optional injected QdrantStore.
        bm25_index: Optional injected BM25SearchIndex.
        embedder: Optional injected EmbeddingService.
        llm: Optional injected LLMService.

    Returns:
        Structured response dictionary with:
            - "answer": Grounded answer string
            - "sources": List of citation dictionaries {source, page, section, snippet}
            - "retrieval_results": Reranked passages passed to LLM
            - "hybrid_candidates_count": Number of hybrid candidates prior to reranking
            - "reranker_used": Identifier of reranker used (e.g. 'jina (...)', 'none (degraded fallback)')
    """
    settings = get_settings()
    final_k = top_k or getattr(settings, "rerank_top_k", 5)
    candidates_k = candidate_k or getattr(settings, "hybrid_top_k", 20)
    clean_query = query.strip()

    if not clean_query:
        return {
            "answer": "Please ask a question about company HR policies.",
            "sources": [],
            "retrieval_results": [],
            "hybrid_candidates_count": 0,
            "reranker_used": "none",
        }

    logger.info(f"Hybrid RAG query received (session={session_id}): '{clean_query[:75]}...'")

    # 1. Initialize or resolve Hybrid Retriever
    if retriever is None:
        active_store: Optional[QdrantStore] = store
        if active_store is None:
            try:
                active_store = get_qdrant_store()
            except QdrantConfigurationError as cfg_err:
                logger.warning(f"Qdrant not configured: {cfg_err}. Proceeding with BM25-only hybrid retrieval.")
                active_store = None
            except Exception as exc:
                logger.warning(f"Failed to connect to Qdrant: {exc}. Proceeding with BM25-only hybrid retrieval.")
                active_store = None
            active_bm25 = bm25_index or get_bm25_index()
        else:
            # When caller explicitly injects a custom store (e.g. in unit tests),
            # do not pollute with global disk index unless explicitly provided
            active_bm25 = bm25_index or BM25SearchIndex()

        active_embedder = embedder or get_embedding_service()
        retriever = HybridRetriever(
            store=active_store,
            bm25_index=active_bm25,
            embedder=active_embedder,
        )

    # 2. Retrieve Top Candidates via Dense + BM25 Fusion (Top 20)
    try:
        candidates: List[HybridSearchResult] = retriever.retrieve_hybrid(
            query=clean_query,
            top_candidates=candidates_k,
            policy_type=policy_type,
        )
    except Exception as exc:
        logger.error(f"Hybrid retrieval failed: {exc}")
        return {
            "answer": f"Retrieval failed while searching policy documents: {exc}",
            "sources": [],
            "retrieval_results": [],
            "hybrid_candidates_count": 0,
            "reranker_used": "none",
        }

    if not candidates:
        logger.info("No documents matched the query in dense or lexical indices.")
        return {
            "answer": (
                "The provided company HR policy documents do not contain information regarding this topic. "
                "Please reach out to the People Operations / HR team directly for assistance."
            ),
            "sources": [],
            "retrieval_results": [],
            "hybrid_candidates_count": 0,
            "reranker_used": "none",
        }

    # 3. Rerank candidates down to top_k (Top 5)
    active_reranker = reranker or get_reranker()
    try:
        reranked_results: List[RerankedResult] = active_reranker.rerank(
            query=clean_query,
            candidates=candidates,
            top_k=final_k,
        )
    except Exception as exc:
        logger.error(f"Reranking stage failed: {exc}. Falling back to top {final_k} candidates.")
        reranked_results = [
            RerankedResult(
                chunk_id=c.chunk_id,
                document=c.document,
                page=c.page,
                section=c.section,
                text=c.text,
                score=c.score,
                reranker="none (reranker exception fallback)",
                metadata=c.metadata,
            )
            for c in candidates[:final_k]
        ]

    if not reranked_results:
        return {
            "answer": (
                "The provided company HR policy documents do not contain information regarding this topic. "
                "Please reach out to the People Operations / HR team directly for assistance."
            ),
            "sources": [],
            "retrieval_results": [],
            "hybrid_candidates_count": len(candidates),
            "reranker_used": "none",
        }

    # 4. Extract citations from top reranked results
    citations: List[SourceCitation] = extract_citations_from_results(reranked_results)
    sources_payload = [c.to_ui_dict() for c in citations]

    # 5. Build prompt with ONLY the top reranked passages (never all 20) and generate answer via LLM
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(clean_query, reranked_results)

    model_used = "unknown"
    fallback_used = False
    llm_latency = 0.0

    try:
        active_llm = llm or get_llm_service()
        answer_text = active_llm.generate(prompt=user_prompt, system_prompt=system_prompt)

        # Inspect routing diagnostics from LiteLLM / LLMService
        last_resp = getattr(active_llm, "last_response", None)
        if last_resp is not None:
            model_used = last_resp.model_used
            fallback_used = last_resp.fallback_used
            llm_latency = last_resp.latency
            settings = get_settings()
            if fallback_used and getattr(settings, "google_cloud_project", None):
                answer_text += f"\n\n*Note: Response generated using backup model ({model_used}) due to primary service unavailability.*"
        else:
            model_used = getattr(active_llm, "model_name", "unknown")

    except LLMConfigurationError as llm_cfg_err:
        logger.warning(f"LLM not configured: {llm_cfg_err}")
        top_doc = reranked_results[0]
        answer_text = (
            f"**Policy Excerpt Found** (Top Match: {top_doc.document}, Section: {top_doc.section}, Page: {top_doc.page}):\n\n"
            f"> {top_doc.text}\n\n"
            f"*Note: LLM synthesis is inactive because credentials are not configured: {llm_cfg_err}*"
        )
    except Exception as exc:
        logger.error(f"LLM generation failed: {exc}")
        answer_text = f"An error occurred during response generation: {exc}"

    reranker_desc = reranked_results[0].reranker if reranked_results else "none"

    return {
        "answer": answer_text,
        "sources": sources_payload,
        "retrieval_results": [r.model_dump() for r in reranked_results],
        "hybrid_candidates_count": len(candidates),
        "reranker_used": reranker_desc,
        "model_used": model_used,
        "fallback_used": fallback_used,
        "llm_latency": llm_latency,
    }
