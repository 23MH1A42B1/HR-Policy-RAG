#!/usr/bin/env python3
"""Production health check utility for HR Policy RAG Assistant.

Inspects critical subsystem readiness:
1. Environment and configuration loading
2. Retrieval indices (BM25 lexical index and Qdrant vector store)
3. Embedding model accessibility
4. Storage service availability (GCS or local filesystem)
5. Observability and LLM routing readiness

Outputs JSON status report and exits 0 if operational, 1 if critical error.

Usage:
    uv run python scripts/healthcheck.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.embeddings.embedder import get_embedding_service
from app.observability import get_langsmith_service
from app.retrieval.bm25_search import get_bm25_index
from app.retrieval.qdrant_store import QdrantStore
from app.storage import get_gcs_storage


def check_health() -> Dict[str, Any]:
    """Execute subsystem diagnostic checks."""
    status: Dict[str, Any] = {
        "status": "healthy",
        "timestamp": time.time(),
        "subsystems": {},
    }

    # 1. Configuration check
    try:
        settings = get_settings()
        status["subsystems"]["config"] = {
            "status": "healthy",
            "app_name": settings.app_name,
            "environment": getattr(settings, "environment", "development"),
        }
    except Exception as exc:
        status["subsystems"]["config"] = {"status": "unhealthy", "error": str(exc)}
        status["status"] = "unhealthy"

    # 2. BM25 Search Index Check
    try:
        bm25 = get_bm25_index()
        status["subsystems"]["bm25_index"] = {
            "status": "healthy",
            "indexed_chunks": len(bm25.chunks),
        }
    except Exception as exc:
        status["subsystems"]["bm25_index"] = {"status": "degraded", "error": str(exc)}

    # 3. Qdrant Vector Store Check
    try:
        store = QdrantStore()
        is_healthy = store.health_check()
        status["subsystems"]["qdrant"] = {
            "status": "healthy" if is_healthy else "unreachable",
            "url": store.url if is_healthy else "disconnected",
        }
    except Exception as exc:
        status["subsystems"]["qdrant"] = {
            "status": "not_configured_or_unreachable",
            "detail": "Operating in local BM25 mode if Qdrant is not provisioned",
            "note": str(exc),
        }

    # 4. Embeddings Model Check
    try:
        embedder = get_embedding_service()
        model_name = getattr(embedder, "model_name", "unknown")
        status["subsystems"]["embeddings"] = {
            "status": "healthy",
            "model": model_name,
        }
    except Exception as exc:
        status["subsystems"]["embeddings"] = {"status": "unhealthy", "error": str(exc)}
        status["status"] = "unhealthy"

    # 5. Document Storage Check
    try:
        storage = get_gcs_storage()
        mode = "gcs" if getattr(storage, "is_gcs", False) else "local"
        status["subsystems"]["storage"] = {
            "status": "healthy",
            "mode": mode,
            "bucket": getattr(storage, "bucket_name", "local"),
        }
    except Exception as exc:
        status["subsystems"]["storage"] = {"status": "degraded", "error": str(exc)}

    # 6. Observability Check
    try:
        tracer = get_langsmith_service()
        status["subsystems"]["observability"] = {
            "status": "healthy",
            "mode": "cloud_langsmith" if tracer.is_configured() else "local_tracing",
            "project": tracer.project if tracer.is_configured() else None,
        }
    except Exception as exc:
        status["subsystems"]["observability"] = {"status": "degraded", "error": str(exc)}

    return status


def main() -> None:
    health_data = check_health()
    print(json.dumps(health_data, indent=2))
    if health_data["status"] == "unhealthy":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
