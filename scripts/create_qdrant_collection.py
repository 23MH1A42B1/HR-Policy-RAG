"""CLI script to initialize or recreate the Qdrant collection for HR policies.

Usage:
    uv run python scripts/create_qdrant_collection.py [--recreate] [--vector-size 384]
"""

from __future__ import annotations

import argparse
import sys
from app.config import get_settings
from app.retrieval.qdrant_store import QdrantConfigurationError, QdrantStore
from app.utils.logging import get_logger, setup_logging

logger = get_logger("scripts.create_qdrant_collection")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Initialize Qdrant collection for HR Policy Assistant.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection if it already exists (WARNING: drops existing data)",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=settings.embedding_dimension,
        help=f"Embedding vector dimensionality (default: {settings.embedding_dimension})",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=settings.qdrant_collection_name,
        help=f"Collection name (default: {settings.qdrant_collection_name})",
    )
    return parser.parse_args()


def main() -> None:
    """Execution entry point."""
    setup_logging()
    args = parse_args()
    settings = get_settings()

    print("\n==================================================")
    print("      QDRANT COLLECTION INITIALIZATION")
    print("==================================================")
    print(f"Target URL:        {settings.qdrant_url or '(not configured)'}")
    print(f"Collection Name:   {args.collection_name}")
    print(f"Vector Dimension:  {args.vector_size}")
    print(f"Recreate:          {args.recreate}")
    print("==================================================\n")

    try:
        store = QdrantStore(collection_name=args.collection_name)
    except QdrantConfigurationError as exc:
        print(f"[Configuration Error]: {exc}")
        print("Please configure QDRANT_URL in your .env file or environment.")
        sys.exit(1)
    except Exception as exc:
        print(f"[Connection Error]: Unable to connect to Qdrant ({exc})")
        sys.exit(1)

    if not store.health_check():
        print(f"[Health Check Failed]: Qdrant endpoint at '{settings.qdrant_url}' is not reachable.")
        sys.exit(1)

    try:
        created = store.create_collection(
            vector_size=args.vector_size,
            recreate=args.recreate,
        )
        if created:
            print(f"✅ Successfully created collection '{args.collection_name}' in Qdrant!")
        else:
            print(f"ℹ️  Collection '{args.collection_name}' already exists. Use --recreate to overwrite.")
    except Exception as exc:
        logger.exception(f"Failed to create collection: {exc}")
        print(f"[Error]: Failed creating collection in Qdrant: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
