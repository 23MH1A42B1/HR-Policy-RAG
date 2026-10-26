"""Comprehensive test suite for Hybrid Search, BM25, Result Fusion, and Jina Reranker."""

from unittest import mock
import httpx
from app.embeddings.embedder import MockEmbeddingService
from app.ingestion.metadata import DocumentChunk
from app.rag.generator import MockLLMService
from app.rag.pipeline import answer_question
from app.retrieval.bm25_search import BM25Result, BM25SearchIndex, tokenize_code_and_text
from app.retrieval.hybrid_search import (
    HybridRetriever,
    HybridSearchResult,
    reciprocal_rank_fusion,
    weighted_linear_fusion,
)
from app.retrieval.qdrant_store import DenseSearchResult, QdrantStore
from app.retrieval.reranker import (
    DegradedReranker,
    JinaReranker,
    MockReranker,
    get_reranker,
)


# =====================================================================
# 1. BM25 & Tokenization Tests
# =====================================================================

def test_bm25_tokenization_preserves_exact_codes_and_terms():
    """Verify tokenizer preserves hyphenated policy IDs, employee IDs, dates, and amounts."""
    text = "Refer to HR-LEAVE-2026 section 4.2 for employee EMP-9821 on 2026-03-15 with $750 reimbursement."
    tokens = tokenize_code_and_text(text)

    # Must preserve exact domain identifiers
    assert "hr-leave-2026" in tokens
    assert "emp-9821" in tokens
    assert "2026-03-15" in tokens
    assert "750" in tokens or "$750" in tokens
    assert "reimbursement" in tokens


def test_bm25_search_exact_code_matching():
    """Verify BM25 retrieves exact matches for policy codes and specific numbers."""
    chunks = [
        DocumentChunk(
            chunk_id="chunk_01",
            text="General guidelines for company vacation and standard holiday schedule.",
            document="leave_policy.pdf",
            page=1,
            section="Overview",
            policy_type="leave",
        ),
        DocumentChunk(
            chunk_id="chunk_02",
            text="Special maternity leave policy under code HR-LEAVE-2026 provides 16 weeks paid.",
            document="leave_policy.pdf",
            page=3,
            section="Maternity",
            policy_type="leave",
        ),
        DocumentChunk(
            chunk_id="chunk_03",
            text="Remote employee equipment stipend for employee EMP-4019 is capped at $500.",
            document="remote_policy.docx",
            page=2,
            section="Stipends",
            policy_type="remote",
        ),
    ]

    index = BM25SearchIndex()
    index.build_index(chunks)

    # 1. Exact query for HR-LEAVE-2026
    results = index.search("HR-LEAVE-2026", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "chunk_02"
    assert "HR-LEAVE-2026" in results[0].text

    # 2. Exact query for employee ID EMP-4019
    results_emp = index.search("EMP-4019", top_k=2)
    assert len(results_emp) > 0
    assert results_emp[0].chunk_id == "chunk_03"
    assert "EMP-4019" in results_emp[0].text


def test_bm25_policy_type_filtering():
    """Verify BM25 results can be filtered by policy_type."""
    chunks = [
        DocumentChunk(
            chunk_id="chunk_leave",
            text="Expense allowance while on maternity leave.",
            document="leave_policy.pdf",
            page=1,
            section="Leave",
            policy_type="leave",
        ),
        DocumentChunk(
            chunk_id="chunk_expense",
            text="Expense allowance while traveling on company business.",
            document="expense_policy.pdf",
            page=1,
            section="Travel",
            policy_type="expense",
        ),
    ]
    index = BM25SearchIndex()
    index.build_index(chunks)

    # Search with filter
    filtered_results = index.search("expense allowance", top_k=5, policy_type="expense")
    assert len(filtered_results) == 1
    assert filtered_results[0].chunk_id == "chunk_expense"


# =====================================================================
# 2. Result Fusion Tests (RRF & Weighted)
# =====================================================================

def test_reciprocal_rank_fusion_scoring_and_deduplication():
    """Verify RRF formula scores and merges dense and BM25 results."""
    dense = [
        DenseSearchResult(
            document="docA.pdf",
            page=1,
            section="Sec1",
            text="Text passage A",
            score=0.95,
            metadata={"chunk_id": "c_A"},
        ),
        DenseSearchResult(
            document="docB.pdf",
            page=2,
            section="Sec2",
            text="Text passage B",
            score=0.85,
            metadata={"chunk_id": "c_B"},
        ),
    ]
    bm25 = [
        BM25Result(
            chunk_id="c_B",
            document="docB.pdf",
            page=2,
            section="Sec2",
            text="Text passage B",
            score=4.2,
        ),
        BM25Result(
            chunk_id="c_C",
            document="docC.pdf",
            page=3,
            section="Sec3",
            text="Text passage C",
            score=3.8,
        ),
    ]

    fused = reciprocal_rank_fusion(dense, bm25, rrf_k=60, top_k=10)

    # c_B appears in both dense (rank 2) and bm25 (rank 1), so should score highest:
    # 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = ~0.03252
    assert len(fused) == 3
    assert fused[0].chunk_id == "c_B"
    assert fused[0].dense_rank == 2
    assert fused[0].bm25_rank == 1
    assert fused[0].score > fused[1].score


def test_weighted_linear_fusion_scoring():
    """Verify min-max normalized weighted fusion correctly combines scores."""
    dense = [
        DenseSearchResult(
            document="docA.pdf",
            page=1,
            section="Sec1",
            text="Text A",
            score=0.9,
            metadata={"chunk_id": "cA"},
        ),
        DenseSearchResult(
            document="docB.pdf",
            page=2,
            section="Sec2",
            text="Text B",
            score=0.5,
            metadata={"chunk_id": "cB"},
        ),
    ]
    bm25 = [
        BM25Result(
            chunk_id="cB",
            document="docB.pdf",
            page=2,
            section="Sec2",
            text="Text B",
            score=10.0,
        ),
        BM25Result(
            chunk_id="cA",
            document="docA.pdf",
            page=1,
            section="Sec1",
            text="Text A",
            score=2.0,
        ),
    ]

    fused = weighted_linear_fusion(
        dense, bm25, dense_weight=0.5, bm25_weight=0.5, top_k=5
    )
    assert len(fused) == 2
    # Both items exist and are scored
    assert all(r.score >= 0.0 for r in fused)


# =====================================================================
# 3. Jina Reranker & Fallback Tests
# =====================================================================

def test_degraded_reranker_when_jina_not_configured():
    """Verify DegradedReranker returns top_k candidates and explicitly tags degraded status."""
    candidates = [
        HybridSearchResult(
            chunk_id=f"c_{i}",
            document="policy.pdf",
            page=i,
            section=f"Section {i}",
            text=f"Passage text {i}",
            score=0.03 - (i * 0.001),
        )
        for i in range(10)
    ]

    reranker = DegradedReranker()
    results = reranker.rerank(query="leave rules", candidates=candidates, top_k=5)

    assert len(results) == 5
    # Must NOT pretend Jina was used
    for r in results:
        assert r.reranker == "none (degraded fallback)"


def test_get_reranker_factory_returns_degraded_without_credentials():
    """Verify get_reranker() automatically selects DegradedReranker when JINA_API_KEY is unset."""
    with mock.patch.dict("os.environ", {"JINA_API_KEY": ""}, clear=True):
        reranker = get_reranker()
        assert isinstance(reranker, DegradedReranker)


def test_jina_reranker_successful_api_call():
    """Verify JinaReranker parses HTTP response, scores, and reorders candidates."""
    candidates = [
        HybridSearchResult(
            chunk_id="c_1",
            document="leave.pdf",
            page=1,
            section="A",
            text="Irrelevant text about parking permits.",
            score=0.03,
        ),
        HybridSearchResult(
            chunk_id="c_2",
            document="leave.pdf",
            page=2,
            section="B",
            text="Employees receive 20 days annual leave per year.",
            score=0.02,
        ),
    ]

    # Mock Jina API response
    mock_api_response = {
        "model": "jina-reranker-v2-base-multilingual",
        "results": [
            {"index": 1, "relevance_score": 0.9421},
            {"index": 0, "relevance_score": 0.1205},
        ],
    }

    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_api_response
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client.post", return_value=mock_resp):
        reranker = JinaReranker(api_key="test_jina_key")
        reranked = reranker.rerank(query="annual leave days", candidates=candidates, top_k=2)

        assert len(reranked) == 2
        # c_2 should be ranked first due to higher relevance score 0.9421
        assert reranked[0].chunk_id == "c_2"
        assert reranked[0].score == 0.9421
        assert "jina" in reranked[0].reranker
        assert reranked[1].chunk_id == "c_1"
        assert reranked[1].score == 0.1205


def test_jina_reranker_network_failure_falls_back_gracefully():
    """Verify JinaReranker degrades cleanly without crashing if network call fails."""
    candidates = [
        HybridSearchResult(
            chunk_id="c_1",
            document="leave.pdf",
            page=1,
            section="A",
            text="Text passage.",
            score=0.03,
        ),
    ]

    with mock.patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused")):
        reranker = JinaReranker(api_key="test_jina_key")
        reranked = reranker.rerank(query="annual leave", candidates=candidates, top_k=1)

        assert len(reranked) == 1
        assert "none (jina call failed, degraded)" in reranked[0].reranker


# =====================================================================
# 4. End-to-End Hybrid RAG Pipeline Integration
# =====================================================================

def test_hybrid_rag_pipeline_top20_candidates_reranked_to_top5():
    """Verify full hybrid pipeline retrieves candidate pool (Top 20) and reranks down to Top 5 for LLM."""
    # 1. Prepare 10 sample chunks in store and BM25 index
    chunks = []
    vectors = []
    for i in range(1, 11):
        c = DocumentChunk(
            chunk_id=f"chunk_{i:02d}",
            text=f"Policy clause #{i}: Specific HR guideline covering employee regulation {i}.",
            document="general_policy.pdf",
            page=i,
            section=f"Section {i}",
            policy_type="general",
        )
        chunks.append(c)
        # Vector pointing in direction of index i
        vectors.append([1.0 if j == (i % 4) else 0.0 for j in range(4)])

    store = QdrantStore(url=":memory:", collection_name="test_hybrid_rag")
    store.create_collection(vector_size=4)
    store.upsert_documents(chunks, vectors)

    bm25 = BM25SearchIndex()
    bm25.build_index(chunks)

    embedder = MockEmbeddingService(dimension=4)
    embedder.embed_text = lambda q: [1.0, 0.0, 0.0, 0.0]

    retriever = HybridRetriever(
        store=store,
        bm25_index=bm25,
        embedder=embedder,
    )

    reranker = MockReranker()
    llm = MockLLMService(custom_response="Synthesized answer based on Top 5 policy passages.")

    result = answer_question(
        query="employee regulation 3",
        top_k=5,
        candidate_k=10,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
    )

    assert "answer" in result
    assert "Synthesized answer" in result["answer"]
    assert result["hybrid_candidates_count"] > 0
    # Must only send top_k (5) documents to LLM / in results, not all 10 candidates
    assert len(result["retrieval_results"]) == 5
    assert result["reranker_used"] == "mock"
