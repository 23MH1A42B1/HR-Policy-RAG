"""Comprehensive test suite for RAG evaluation, metrics, and LangSmith observability.

Validates:
1. Evaluation dataset format and coverage
2. Metric calculations (Recall, Precision, Context Relevance, Answer Relevance, Faithfulness)
3. Groundedness logic and sentence-level thresholding
4. Citation accuracy evaluation
5. LangSmith configuration, local fallback, and secret sanitization
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.evaluation.metrics import EvaluationMetrics, RAGEvaluator
from app.observability.langsmith_tracer import LangSmithService


@pytest.fixture
def evaluator() -> RAGEvaluator:
    return RAGEvaluator()


# ==============================================================================
# 1. Evaluation Dataset Tests
# ==============================================================================


def test_evaluation_dataset_file_structure():
    """Verify data/evaluation/questions.json exists, contains valid schema and items."""
    dataset_path = Path("data/evaluation/questions.json")
    assert dataset_path.exists(), "data/evaluation/questions.json must exist."

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 10, f"Dataset should contain at least 10 items, found {len(data)}"

    raw_dir = Path("data/raw")
    existing_raw_files = {p.name.lower() for p in raw_dir.glob("*.*")}

    for idx, item in enumerate(data):
        assert "question" in item, f"Item {idx} missing 'question' key"
        assert "expected_source" in item, f"Item {idx} missing 'expected_source' key"
        assert isinstance(item["question"], str) and len(item["question"].strip()) > 5
        assert isinstance(item["expected_source"], str)

        # Expected source must correspond to an existing policy file in data/raw
        expected_src = item["expected_source"].lower()
        assert expected_src in existing_raw_files, (
            f"Expected source '{item['expected_source']}' in item {idx} not found in {existing_raw_files}"
        )


# ==============================================================================
# 2. Metric Calculations Tests
# ==============================================================================


def test_retrieval_recall_and_precision(evaluator: RAGEvaluator):
    """Test retrieval recall and precision calculation."""
    # Case 1: Expected source is present in retrieved documents
    retrieved = ["leave_policy.pdf", "remote_work_policy.docx", "leave_policy.pdf"]
    recall = evaluator.calculate_retrieval_recall("leave_policy.pdf", retrieved)
    assert recall == 1.0

    # Precision: 2 out of 3 match leave_policy.pdf
    precision = evaluator.calculate_retrieval_precision("leave_policy.pdf", retrieved)
    assert abs(precision - (2.0 / 3.0)) < 1e-4

    # Case 2: Expected source is absent
    recall_zero = evaluator.calculate_retrieval_recall("expense_policy.pptx", retrieved)
    assert recall_zero == 0.0
    precision_zero = evaluator.calculate_retrieval_precision("expense_policy.pptx", retrieved)
    assert precision_zero == 0.0

    # Case 3: Empty retrieval
    assert evaluator.calculate_retrieval_recall("leave_policy.pdf", []) == 0.0
    assert evaluator.calculate_retrieval_precision("leave_policy.pdf", []) == 0.0


def test_context_relevance(evaluator: RAGEvaluator):
    """Test context relevance calculation based on question token grounding."""
    q = "What is the annual leave allowance for employees?"

    # Highly relevant context
    relevant_passages = [
        "Employees are entitled to 20 business days of paid annual leave per calendar year.",
        "Annual leave must be scheduled in advance with managerial approval.",
    ]
    score_high = evaluator.calculate_context_relevance(q, relevant_passages)
    assert score_high > 0.4

    # Irrelevant context
    irrelevant_passages = [
        "Meal allowances are reimbursed up to $65 per day on business travel.",
        "Hotel expenses require itemized VAT receipts for accounting reconciliation.",
    ]
    score_low = evaluator.calculate_context_relevance(q, irrelevant_passages)
    assert score_low < score_high

    # Empty context
    assert evaluator.calculate_context_relevance(q, []) == 0.0


def test_answer_relevance(evaluator: RAGEvaluator):
    """Test answer relevance calculation."""
    q = "What is the policy on carrying over unused leave?"

    relevant_ans = "A maximum of 5 unused annual leave days may be carried over and must be used before March 31st."
    irrelevant_ans = "All international flights must be booked in economy class using the corporate travel portal."

    score_rel = evaluator.calculate_answer_relevance(q, relevant_ans)
    score_irrel = evaluator.calculate_answer_relevance(q, irrelevant_ans)

    assert score_rel > score_irrel
    assert evaluator.calculate_answer_relevance("", "Some answer") == 0.0
    assert evaluator.calculate_answer_relevance(q, "") == 0.0


def test_faithfulness_calculation(evaluator: RAGEvaluator):
    """Test faithfulness calculation (words in answer supported by context)."""
    context = [
        "Full-time regular employees are entitled to 20 business days of paid annual leave.",
        "Sick leave entitlement is 10 days per year with medical certificate required after 3 days.",
    ]

    # Faithful answer: all concepts come directly from context
    faithful_answer = "Full-time regular employees receive 20 days of paid annual leave."
    faith_score = evaluator.calculate_faithfulness(faithful_answer, context)
    assert faith_score >= 0.85

    # Hallucinated answer: mentions unsubstantiated claims
    hallucinated_answer = "Employees get unlimited stock options and an annual bonus of 50000 dollars."
    halluc_score = evaluator.calculate_faithfulness(hallucinated_answer, context)
    assert halluc_score < faith_score

    # Empty context / answer edge cases
    assert evaluator.calculate_faithfulness("", context) == 0.0
    assert evaluator.calculate_faithfulness(faithful_answer, []) == 0.0


# ==============================================================================
# 3. Groundedness Logic Tests
# ==============================================================================


def test_groundedness_logic(evaluator: RAGEvaluator):
    """Test sentence-level groundedness ratio calculation."""
    context = [
        "Eligible employees receive up to $500 home office equipment stipend for ergonomic chairs.",
        "Internet and utilities are subsidized up to $50 per month.",
    ]

    # Sentence 1 is grounded, Sentence 2 is grounded -> 1.0
    fully_grounded = (
        "Eligible employees receive a $500 home office equipment stipend. "
        "Internet is subsidized up to $50 per month."
    )
    score_full = evaluator.calculate_groundedness(fully_grounded, context)
    assert score_full == 1.0

    # Sentence 1 is grounded, Sentence 2 is completely ungrounded -> 0.5
    partially_grounded = (
        "Eligible employees receive a $500 home office equipment stipend. "
        "Every worker receives a free Tesla electric vehicle upon completion of probation."
    )
    score_part = evaluator.calculate_groundedness(partially_grounded, context)
    assert 0.3 <= score_part <= 0.7

    # Edge cases
    assert evaluator.calculate_groundedness("", context) == 0.0
    assert evaluator.calculate_groundedness("Any text", []) == 0.0


# ==============================================================================
# 4. Citation Evaluation Tests
# ==============================================================================


def test_citation_accuracy(evaluator: RAGEvaluator):
    """Test verification of cited documents against retrieved document pool."""
    retrieved = ["leave_policy.pdf", "remote_work_policy.docx"]

    # Accurate citations (matches filename or clean display title)
    accurate_citations = ["leave_policy.pdf", "Remote Work Policy"]
    acc_score = evaluator.calculate_citation_accuracy(accurate_citations, retrieved)
    assert acc_score == 1.0

    # Inaccurate / fabricated citations
    fabricated_citations = ["non_existent_policy.pdf", "secret_board_memo.docx"]
    fab_score = evaluator.calculate_citation_accuracy(fabricated_citations, retrieved)
    assert fab_score == 0.0

    # Mixed citations (1 accurate, 1 fabricated) -> 0.5
    mixed_citations = ["leave_policy.pdf", "fabricated_policy.pdf"]
    mixed_score = evaluator.calculate_citation_accuracy(mixed_citations, retrieved)
    assert mixed_score == 0.5

    # Edge cases
    assert evaluator.calculate_citation_accuracy([], retrieved) == 1.0
    assert evaluator.calculate_citation_accuracy(["leave_policy.pdf"], []) == 0.0


# ==============================================================================
# 5. LangSmith Configuration and Fallback Tests
# ==============================================================================


def test_langsmith_service_local_mode():
    """Verify LangSmithService operates in local mode when API key is missing."""
    service = LangSmithService(api_key=None)

    assert not service.is_configured()
    assert "local_tracing" in service.integration_status

    trace = service.start_trace(question="What is the bereavement policy?", session_id="test_sess")
    assert trace.question == "What is the bereavement policy?"
    assert trace.session_id == "test_sess"
    assert len(trace.steps) == 0

    # Measure step
    with trace.measure_step("input_security", is_safe=True) as meta:
        meta["checked"] = True

    with trace.measure_step("cache", cache_hit=False) as meta:
        meta["similarity"] = 0.0

    assert len(trace.steps) == 2
    assert trace.steps[0].name == "input_security"
    assert trace.steps[0].metadata["checked"] is True
    assert trace.steps[0].latency_ms >= 0.0

    # Finalize trace
    trace.final_answer = "Bereavement leave is 5 days for immediate family members."
    finalized = service.finish_trace(trace)

    assert finalized.total_latency_ms >= 0.0
    assert finalized.final_answer == "Bereavement leave is 5 days for immediate family members."

    # Verify local trace buffer
    recent = service.get_recent_traces()
    assert len(recent) > 0
    assert any(t.trace_id == finalized.trace_id for t in recent)


def test_langsmith_secret_scrubbing_in_traces():
    """Verify that credentials and secrets are stripped and never recorded in traces."""
    service = LangSmithService(api_key=None)

    # Input containing simulated leaked secret
    question_with_secret = "How do I use my API key AIzaSyDUMMYSECRETKEY1234567890 to access policy?"
    trace = service.start_trace(question=question_with_secret)

    # Question in trace must not contain the raw API key
    assert "AIzaSyDUMMYSECRETKEY1234567890" not in trace.question

    # Output containing simulated Groq secret
    trace.final_answer = "Your Groq key is gsk_abcdef12345678901234567890 for API access."
    finalized = service.finish_trace(trace)

    assert "gsk_abcdef12345678901234567890" not in finalized.final_answer
    assert "[REDACTED_SECRET]" in finalized.final_answer


def test_evaluator_composite_summary(evaluator: RAGEvaluator):
    """Test compute_summary aggregates metrics correctly."""
    sample1 = EvaluationMetrics(
        retrieval_recall=1.0,
        retrieval_precision=0.8,
        context_relevance=0.7,
        answer_relevance=0.9,
        faithfulness=0.85,
        groundedness=1.0,
        citation_accuracy=1.0,
    )
    sample2 = EvaluationMetrics(
        retrieval_recall=0.0,
        retrieval_precision=0.0,
        context_relevance=0.3,
        answer_relevance=0.5,
        faithfulness=0.4,
        groundedness=0.5,
        citation_accuracy=0.0,
    )

    summary = evaluator.compute_summary([sample1, sample2])
    assert summary["sample_count"] == 2
    assert summary["mean_retrieval_recall"] == 0.5
    assert summary["mean_retrieval_precision"] == 0.4
    assert 0.0 <= summary["composite_score"] <= 1.0
