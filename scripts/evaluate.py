#!/usr/bin/env python3
"""RAG Benchmark and Observability Evaluation Runner for HR Policy Assistant.

Evaluates 7 production dimensions:
1. Retrieval Recall
2. Retrieval Precision
3. Context Relevance
4. Answer Relevance
5. Faithfulness
6. Groundedness
7. Citation Accuracy

Supports LangSmith dataset logging when configured, with transparent local
evaluation fallback when LangSmith API credentials are not set.

Usage:
    uv run python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.agent import HRPolicyAgent
from app.config import get_settings
from app.evaluation.metrics import EvaluationMetrics, RAGEvaluator
from app.observability.langsmith_tracer import get_langsmith_service
from app.rag.generator import MockLLMService
from app.utils.logging import get_logger

logger = get_logger("scripts.evaluate")


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load benchmark question set from JSON file."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset at {dataset_path} must be a non-empty list of test items.")
    return data


def format_row(cols: List[str], widths: List[int]) -> str:
    """Format row of columns with specified widths."""
    return " | ".join(c.ljust(w)[:w] for c, w in zip(cols, widths))


def run_evaluation(
    dataset_path: Path,
    limit: int = 0,
    force_mock_llm: bool = False,
) -> Dict[str, Any]:
    """Execute evaluation benchmark across questions."""
    print("=" * 80)
    print(" HR POLICY RAG BENCHMARK & OBSERVABILITY EVALUATION ")
    print("=" * 80)

    dataset = load_dataset(dataset_path)
    if limit > 0:
        dataset = dataset[:limit]

    print(f"Loaded {len(dataset)} evaluation items from {dataset_path.name}")

    # Check LangSmith Configuration
    settings = get_settings()
    tracer = get_langsmith_service()
    if tracer.is_configured():
        print(f"Observability: Cloud LangSmith Active (Project: '{tracer.project}')")
    else:
        print("Observability: Local Tracing Mode (LANGSMITH_API_KEY not configured - cloud telemetry disabled)")

    evaluator = RAGEvaluator()

    # Determine LLM service for evaluation
    agent: HRPolicyAgent
    if force_mock_llm or (not settings.gemini_api_key and not settings.groq_api_key):
        print("LLM Routing: Running with grounded mock synthesis for evaluation environment.")
        mock_llm = MockLLMService(
            model_name="benchmark-eval-llm",
            custom_response=(
                "According to company HR Policy, regular full-time employees receive 20 business days "
                "of annual leave and 10 paid sick days per calendar year. Remote employees are eligible "
                "for a $500 home office equipment stipend."
            ),
        )
        agent = HRPolicyAgent(llm=mock_llm)
    else:
        print(f"LLM Routing: Active provider '{settings.llm_provider}'")
        agent = HRPolicyAgent()

    print("\nRunning evaluation across test suite...")
    print("-" * 80)

    results: List[Dict[str, Any]] = []
    metric_objects: List[EvaluationMetrics] = []

    start_time = time.perf_counter()

    for idx, item in enumerate(dataset, start=1):
        q = item.get("question", "")
        expected_src = item.get("expected_source", "")
        ground_truth = item.get("ground_truth", "")

        t0 = time.perf_counter()
        agent_resp = agent.process_query(query=q, top_k=5)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_docs = [p["document"] for p in agent_resp.retrieval_results]
        context_texts = [p["text"] for p in agent_resp.retrieval_results]
        cited_sources = [s.get("document") or s.get("source", "") for s in agent_resp.sources]

        m = evaluator.evaluate_sample(
            question=q,
            expected_source=expected_src,
            retrieved_documents=retrieved_docs,
            context_passages=context_texts,
            answer=agent_resp.answer,
            citations=cited_sources,
            ground_truth=ground_truth,
        )
        metric_objects.append(m)

        record = {
            "index": idx,
            "question": q,
            "expected_source": expected_src,
            "retrieved_documents": retrieved_docs[:3],
            "model_used": agent_resp.model_used,
            "duration_ms": round(duration_ms, 1),
            "metrics": m.model_dump(),
        }
        results.append(record)

    total_duration = time.perf_counter() - start_time
    summary = evaluator.compute_summary(metric_objects)

    # Print Table of Results
    headers = ["#", "Question", "Expected", "Recall", "Prec", "CtxRel", "AnsRel", "Faith", "Ground", "CitAcc"]
    widths = [3, 24, 18, 6, 6, 6, 6, 6, 6, 6]

    print("\n" + "=" * 95)
    print(format_row(headers, widths))
    print("-" * 95)

    for r in results:
        m = r["metrics"]
        row = [
            str(r["index"]),
            r["question"],
            r["expected_source"],
            f"{m['retrieval_recall']:.2f}",
            f"{m['retrieval_precision']:.2f}",
            f"{m['context_relevance']:.2f}",
            f"{m['answer_relevance']:.2f}",
            f"{m['faithfulness']:.2f}",
            f"{m['groundedness']:.2f}",
            f"{m['citation_accuracy']:.2f}",
        ]
        print(format_row(row, widths))

    print("=" * 95)
    print("\nAGGREGATE BENCHMARK RESULTS (Averages across dataset):")
    print("-" * 55)
    print(f"  1. Retrieval Recall:       {summary['mean_retrieval_recall'] * 100:.1f}%")
    print(f"  2. Retrieval Precision:    {summary['mean_retrieval_precision'] * 100:.1f}%")
    print(f"  3. Context Relevance:      {summary['mean_context_relevance'] * 100:.1f}%")
    print(f"  4. Answer Relevance:       {summary['mean_answer_relevance'] * 100:.1f}%")
    print(f"  5. Faithfulness:           {summary['mean_faithfulness'] * 100:.1f}%")
    print(f"  6. Groundedness:           {summary['mean_groundedness'] * 100:.1f}%")
    print(f"  7. Citation Accuracy:      {summary['mean_citation_accuracy'] * 100:.1f}%")
    print("-" * 55)
    print(f"  * COMPOSITE RAG SCORE:     {summary['composite_score'] * 100:.1f} / 100.0")
    print(f"  * Total Evaluation Time:   {total_duration:.2f}s ({len(dataset)} questions)")
    print("=" * 55)

    # LangSmith Export
    if tracer.is_configured():
        run_id = tracer.export_evaluation_run(
            dataset_name=dataset_path.name,
            results=results,
            summary_metrics=summary,
        )
        print(f"\n[LangSmith] Successfully exported evaluation metrics to project '{tracer.project}' (Run ID: {run_id})")
    else:
        print("\n[LangSmith] Credentials not configured. Local benchmark report generated successfully.")

    # Save local results artifact
    output_dir = PROJECT_ROOT / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_out = output_dir / "latest_results.json"
    with open(latest_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "dataset": str(dataset_path),
                "summary": summary,
                "detailed_results": results,
            },
            f,
            indent=2,
        )
    print(f"Detailed evaluation metrics stored at: {latest_out.relative_to(PROJECT_ROOT)}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HR Policy RAG Assistant")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "questions.json",
        help="Path to questions.json evaluation dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of evaluation questions (0 = all)",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Force deterministic mock LLM for testing without network calls",
    )
    args = parser.parse_args()

    summary = run_evaluation(
        dataset_path=args.dataset,
        limit=args.limit,
        force_mock_llm=args.mock_llm,
    )

    if summary.get("composite_score", 0.0) < 0.3:
        print("\nWarning: Composite score below threshold (30%).")
        sys.exit(1)
    print("\nBenchmark completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
