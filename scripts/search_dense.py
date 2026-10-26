"""CLI script to test dense vector retrieval from Qdrant.

Usage:
    uv run python scripts/search_dense.py --query "How many annual leave days do employees get?" [--top-k 3] [--policy-type leave]
"""

from __future__ import annotations

import argparse
import sys
from app.config import get_settings
from app.embeddings.embedder import get_embedding_service
from app.retrieval.qdrant_store import QdrantConfigurationError, QdrantStore
from app.utils.logging import get_logger, setup_logging

logger = get_logger("scripts.search_dense")


def parse_args() -> argparse.Namespace:
    """Parse search CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Search HR policies using dense vector similarity.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language question or query string",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top matching passages to retrieve (default: 3)",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default=None,
        help="Optional policy filter (e.g. 'leave', 'remote_work', 'expense', 'attendance')",
    )
    parser.add_argument(
        "--embed-provider",
        type=str,
        default=None,
        help="Embedding provider (fastembed, mock)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint."""
    setup_logging()
    args = parse_args()
    settings = get_settings()

    print("\n==================================================")
    print("           DENSE VECTOR POLICY SEARCH")
    print("==================================================")
    print(f"Query:        \"{args.query}\"")
    print(f"Top K:        {args.top_k}")
    print(f"Filter:       {args.policy_type or '(None - Global Search)'}")
    print(f"Collection:   {settings.qdrant_collection_name}")
    print("==================================================\n")

    try:
        store = QdrantStore()
    except QdrantConfigurationError as exc:
        print(f"[Configuration Error]: {exc}")
        print("Please configure QDRANT_URL in your .env file or environment.")
        sys.exit(1)
    except Exception as exc:
        print(f"[Connection Error]: Unable to connect to Qdrant ({exc})")
        sys.exit(1)

    if not store.health_check():
        print(f"[Health Check Failed]: Qdrant at '{settings.qdrant_url}' is not reachable.")
        sys.exit(1)

    embedder = get_embedding_service(provider=args.embed_provider)
    print(f"Generating query embedding ({embedder.__class__.__name__})...")
    query_vector = embedder.embed_text(args.query)

    print("Searching Qdrant collection...\n")
    results = store.search_dense(
        query_vector=query_vector,
        top_k=args.top_k,
        policy_type=args.policy_type,
    )

    if not results:
        print("No matching policy passages found.")
        return

    print(f"Found {len(results)} relevant policy passage(s):\n")
    for idx, r in enumerate(results, start=1):
        print(f"--- Result #{idx} (Similarity Score: {r.score:.4f}) ---")
        print(f"Document: {r.document} | Page/Slide: {r.page} | Section: {r.section}")
        print(f"Text:\n{r.text}")
        if r.metadata:
            print(f"Metadata: {r.metadata}")
        print()


if __name__ == "__main__":
    main()
