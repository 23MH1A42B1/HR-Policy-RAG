"""CLI script to test the end-to-end RAG pipeline from terminal.

Usage:
    uv run python scripts/ask_policy.py --query "How many annual leave days do employees get?"
"""

from __future__ import annotations

import argparse
import sys
from app.rag.pipeline import answer_question
from app.utils.logging import get_logger, setup_logging

logger = get_logger("scripts.ask_policy")


def parse_args() -> argparse.Namespace:
    """Parse question arguments."""
    parser = argparse.ArgumentParser(
        description="Query HR policies using the end-to-end RAG pipeline.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Employee question regarding company HR policies",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of policy excerpts to retrieve (default: 3)",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default=None,
        help="Optional policy domain filter (e.g. leave, remote_work, expense, attendance)",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Execute query through the Guarded HR Agent (scope check, semantic cache, conversation memory)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup_logging()
    args = parse_args()

    mode_str = "Guarded HR Agent" if args.agent else "Hybrid RAG Pipeline"
    print("\n==================================================")
    print("         HR POLICY RAG ASSISTANT - QUERY")
    print("==================================================")
    print(f"Mode:        {mode_str}")
    print(f"Question:    \"{args.query}\"")
    print(f"Top K:       {args.top_k}")
    print(f"Filter:      {args.policy_type or '(None)'}")
    print("==================================================\n")

    if args.agent:
        from app.agent.agent import get_agent

        agent = get_agent()
        agent_resp = agent.process_query(
            query=args.query,
            policy_type=args.policy_type,
            top_k=args.top_k,
        )
        response = {
            "answer": agent_resp.answer,
            "sources": agent_resp.sources,
            "retrieval_results": agent_resp.retrieval_results,
            "hybrid_candidates_count": len(agent_resp.retrieval_results),
            "reranker_used": agent_resp.reranker_used,
            "model_used": agent_resp.model_used,
            "fallback_used": agent_resp.fallback_used,
            "cache_hit": agent_resp.cache_hit,
            "in_scope": agent_resp.in_scope,
        }
    else:
        response = answer_question(
            query=args.query,
            top_k=args.top_k,
            policy_type=args.policy_type,
        )

    print("--------------------------------------------------")
    print("ANSWER:")
    print("--------------------------------------------------")
    print(response["answer"])
    print("\n--------------------------------------------------")
    print("VERIFIED SOURCES:")
    print("--------------------------------------------------")
    sources = response.get("sources", [])
    if sources:
        for idx, s in enumerate(sources, start=1):
            page_str = f"Page/Slide {s.get('page')}" if s.get("page") is not None else "N/A"
            print(f"[{idx}] 📄 {s.get('source')} | {page_str} | Section: {s.get('section')}")
            if s.get("snippet"):
                print(f"    Excerpt: \"{s.get('snippet')}\"")
    else:
        print("(No sources cited)")

    print("\n--------------------------------------------------")
    print("HYBRID RETRIEVAL & RERANKING SUMMARY:")
    print("--------------------------------------------------")
    candidates_count = response.get("hybrid_candidates_count", 0)
    reranker_used = response.get("reranker_used", "none")
    retrieval_records = response.get("retrieval_results", [])
    model_used = response.get("model_used", "unknown")
    fallback_used = response.get("fallback_used", False)
    llm_latency = response.get("llm_latency", 0.0)

    if "cache_hit" in response:
        print(f"Semantic Cache Hit:         {response['cache_hit']}")
    if "in_scope" in response:
        print(f"HR Scope Check:             {'In Scope' if response['in_scope'] else 'Out of Scope'}")

    print(f"Hybrid Candidates Retrieved: {candidates_count}")
    print(f"Reranker:                   {reranker_used}")
    print(f"Final Passages to LLM:      {len(retrieval_records)}")
    print(f"LLM Model Used:             {model_used}")
    print(f"Fallback Route Activated:   {fallback_used}")
    if llm_latency > 0:
        print(f"LLM Latency:                {llm_latency:.3f}s")
    for idx, r in enumerate(retrieval_records, start=1):
        score_val = r.get("score", 0.0)
        print(f"  • Top #{idx}: {r.get('document')} (Page {r.get('page')}, Section: {r.get('section')}) [Score: {score_val:.4f}]")
    print("==================================================\n")


if __name__ == "__main__":
    main()
