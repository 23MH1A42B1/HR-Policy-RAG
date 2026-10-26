"""Test suite for Semantic Cache, Conversation Memory, HR Scope Checker, and Guarded Agent."""

from unittest import mock
import pytest

from app.agent.agent import HRPolicyAgent
from app.agent.memory import SessionMemory
from app.agent.scope_checker import HRScopeChecker, OUT_OF_SCOPE_RESPONSE
from app.agent.tools import PolicySearchResult, PolicySearchTool
from app.cache.semantic_cache import SemanticCache, cosine_similarity
from app.embeddings.embedder import MockEmbeddingService
from app.rag.generator import MockLLMService
from app.retrieval.reranker import RerankedResult


# =====================================================================
# 1. Semantic Cache Tests (Hit, Miss, Stale Invalidation, Similarity)
# =====================================================================

def test_cosine_similarity_edge_cases():
    """Verify cosine similarity calculation for identical, orthogonal, and zero vectors."""
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([], []) == pytest.approx(0.0)


def test_semantic_cache_hit_and_miss():
    """Verify exact or high-similarity query hits the cache, while distant query misses."""
    cache = SemanticCache(similarity_threshold=0.85, default_policy_version="2026")

    # Store entry with vector [1.0, 0.0, 0.0]
    cache.store(
        question="How many vacation days do I get?",
        embedding=[1.0, 0.0, 0.0],
        answer="Employees receive 20 days paid leave.",
        sources=[{"source": "Leave Policy", "page": 1}],
        policy_version="2026",
    )

    # 1. High similarity query vector [0.99, 0.05, 0.0] -> HIT
    hit = cache.lookup(query_vector=[0.99, 0.05, 0.0], current_version="2026")
    assert hit is not None
    assert hit.answer == "Employees receive 20 days paid leave."
    assert hit.question == "How many vacation days do I get?"

    # 2. Low similarity query vector [0.0, 1.0, 0.0] -> MISS
    miss = cache.lookup(query_vector=[0.0, 1.0, 0.0], current_version="2026")
    assert miss is None


def test_stale_cache_version_invalidation():
    """Verify cached entries with an outdated policy version are rejected even if query vector matches."""
    cache = SemanticCache(similarity_threshold=0.85, default_policy_version="2026")

    # Store entry from previous policy revision 2024
    cache.store(
        question="What is the carryover policy?",
        embedding=[1.0, 0.0, 0.0],
        answer="Old rule: carryover up to 10 days.",
        sources=[],
        policy_version="2024",
    )

    # Active system policy is now 2026 -> Must NOT return stale 2024 answer
    result = cache.lookup(query_vector=[1.0, 0.0, 0.0], current_version="2026")
    assert result is None, "Stale cache entry was incorrectly returned!"


# =====================================================================
# 2. Conversation Memory & Follow-up Tests
# =====================================================================

def test_session_memory_turn_limit_sliding_window():
    """Verify session memory retains up to max_turns and trims older turns."""
    mem = SessionMemory(session_id="test_sess", max_turns=3)

    for i in range(1, 6):
        mem.add_turn(f"Question {i}", f"Answer {i}")

    turns = mem.get_history()
    assert len(turns) == 3
    # Should only retain Question 3, 4, 5
    assert turns[0].user_query == "Question 3"
    assert turns[-1].user_query == "Question 5"


def test_follow_up_question_contextualization():
    """Verify follow-up queries with pronoun or ellipsis are contextualized with prior turn."""
    mem = SessionMemory(session_id="test_sess", max_turns=3)
    mem.add_turn("How many annual leave days do full-time employees get?", "Full-time employees receive 20 days.")

    # Follow-up: "What about carryover?"
    contextualized = mem.contextualize_query("What about carryover?")
    assert "carryover" in contextualized.lower()
    assert "annual leave" in contextualized.lower()

    # Normal standalone question should not be mangled
    standalone = mem.contextualize_query("What is the reimbursement limit for client dinner expenses?")
    assert "reimbursement limit" in standalone


# =====================================================================
# 3. HR Scope Checker Tests
# =====================================================================

def test_scope_checker_accepts_valid_hr_questions():
    """Verify normal HR policy questions across all domains are in-scope."""
    checker = HRScopeChecker()

    valid_questions = [
        "How many annual leave days do full-time employees receive?",
        "Can I carry over unused PTO into next year?",
        "What is the daily meal allowance for travel?",
        "Am I eligible for a home office equipment stipend under remote work policy?",
        "What are the core working hours and attendance expectations?",
        "What health and dental insurance benefits are provided?",
        "How does the maternity and parental leave policy work?",
        "Refer to HR-POL-2026-01 section 2 for employee EMP-1029.",
        "What is the procedure for reporting workplace harassment?",
    ]

    for q in valid_questions:
        res = checker.check_scope(q)
        assert res.is_in_scope is True, f"Incorrectly rejected valid HR query: '{q}' (reason: {res.reason})"


def test_scope_checker_rejects_out_of_scope_questions():
    """Verify unrelated questions (coding, cooking, creative fiction, trading) are declined."""
    checker = HRScopeChecker()

    unrelated_questions = [
        "Write a Python script to sort a list of numbers",
        "How do I bake chocolate chip cookies from scratch?",
        "Write a poem about the sunrise and ocean waves",
        "Should I invest in Bitcoin or Ethereum this year?",
        "Who is the president of France?",
        "Solve for x in the equation 2x + 5 = 15",
    ]

    for q in unrelated_questions:
        res = checker.check_scope(q)
        assert res.is_in_scope is False, f"Incorrectly accepted out-of-scope query: '{q}'"
        assert res.decline_message == OUT_OF_SCOPE_RESPONSE


# =====================================================================
# 4. Guarded HR Agent Flow Tests
# =====================================================================

def test_agent_deflects_out_of_scope_inquiry():
    """Verify agent immediately deflects out-of-scope question without calling search or LLM."""
    mock_tool = mock.MagicMock(spec=PolicySearchTool)
    mock_llm = mock.MagicMock(spec=MockLLMService)

    agent = HRPolicyAgent(
        search_tool=mock_tool,
        llm=mock_llm,
    )

    response = agent.process_query("Write a python function to scrape a website")
    assert response.in_scope is False
    assert response.answer == OUT_OF_SCOPE_RESPONSE
    assert response.sources == []
    # Search and LLM must never be called
    mock_tool.execute.assert_not_called()
    mock_llm.generate.assert_not_called()


def test_agent_serves_from_semantic_cache():
    """Verify agent returns cached answer on semantic cache hit without executing search tool."""
    mock_tool = mock.MagicMock(spec=PolicySearchTool)
    mock_llm = mock.MagicMock(spec=MockLLMService)
    cache = SemanticCache(similarity_threshold=0.85, default_policy_version="2026")
    embedder = MockEmbeddingService(dimension=4)
    # Fixed vector for deterministic matching
    embedder.embed_text = lambda q: [1.0, 0.0, 0.0, 0.0]

    # Pre-populate cache
    cache.store(
        question="How many leave days?",
        embedding=[1.0, 0.0, 0.0, 0.0],
        answer="Employees receive 20 days of paid annual leave.",
        sources=[{"source": "Leave Policy", "page": 1}],
        policy_version="2026",
    )

    agent = HRPolicyAgent(
        search_tool=mock_tool,
        cache=cache,
        embedder=embedder,
        llm=mock_llm,
    )

    response = agent.process_query("How many leave days do we get?", session_id="test_cache_sess")
    assert response.cache_hit is True
    assert response.model_used == "semantic-cache"
    assert "20 days of paid annual leave" in response.answer
    assert len(response.sources) == 1

    # Search tool and LLM should NOT have been invoked
    mock_tool.execute.assert_not_called()
    mock_llm.generate.assert_not_called()


def test_agent_end_to_end_search_and_generation_flow():
    """Verify full agent flow: scope check -> cache miss -> search tool -> rerank -> LLM grounded answer."""
    # 1. Setup mock search tool returning policy passages
    mock_tool = mock.MagicMock(spec=PolicySearchTool)
    passage = RerankedResult(
        chunk_id="chunk_01",
        document="expense_policy.pptx",
        page=3,
        section="Meal Allowances",
        text="Lunch allowance is up to $30 per day for traveling team members.",
        score=0.92,
        reranker="mock",
    )
    mock_tool.execute.return_value = PolicySearchResult(
        passages=[passage],
        candidates_count=10,
        reranker_used="mock",
        query_used="What is the lunch allowance?",
    )

    # 2. Setup mock LLM
    llm = MockLLMService(custom_response="The daily lunch allowance for traveling team members is up to $30 (Expense Policy, Slide 3).")
    cache = SemanticCache()
    embedder = MockEmbeddingService(dimension=4)

    agent = HRPolicyAgent(
        search_tool=mock_tool,
        cache=cache,
        embedder=embedder,
        llm=llm,
    )

    response = agent.process_query("What is the lunch allowance for business travel?", session_id="sess_travel")

    assert response.in_scope is True
    assert response.cache_hit is False
    assert "$30" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0]["source"] == "Expense Policy"
    assert response.sources[0]["page"] == 3

    # Verify tool was called
    mock_tool.execute.assert_called_once()

    # Verify response was subsequently persisted in the semantic cache
    assert cache.size() == 1
