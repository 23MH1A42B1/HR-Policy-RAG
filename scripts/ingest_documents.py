"""CLI script to ingest, clean, chunk, and persist HR policy documents.

Usage:
    uv run python scripts/ingest_documents.py [--raw-dir data/raw] [--out-dir data/processed]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

from app.config import get_settings
from app.embeddings.embedder import get_embedding_service
from app.ingestion.chunker import DocumentChunker
from app.ingestion.loader import DocumentLoader, ExtractedPage
from app.ingestion.metadata import DocumentChunk
from app.retrieval.qdrant_store import QdrantConfigurationError, QdrantStore
from app.utils.logging import get_logger, setup_logging

logger = get_logger("scripts.ingest_documents")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Ingest HR policy documents (PDF, DOCX, PPTX), embed, and index into Qdrant.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(settings.raw_data_dir),
        help="Directory containing raw policy files (default: data/raw)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(settings.processed_data_dir),
        help="Directory to save processed chunks (default: data/processed)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Target character length per chunk (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Character overlap between chunks (default: {settings.chunk_overlap})",
    )
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip upserting to Qdrant vector database (extract and chunk only)",
    )
    parser.add_argument(
        "--embed-provider",
        type=str,
        default=settings.embedding_provider,
        help=f"Embedding provider (fastembed, mock) (default: {settings.embedding_provider})",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=True,
        help="Recreate Qdrant collection to replace previous index (default: True)",
    )
    return parser.parse_args()


def ingest_documents(
    raw_dir: Path,
    out_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    skip_qdrant: bool = False,
    embed_provider: Optional[str] = None,
    recreate: bool = True,
) -> List[DocumentChunk]:
    """Execute the full document ingestion pipeline: Parse -> Chunk -> Embed -> Qdrant."""
    logger.info(f"Starting document ingestion from '{raw_dir}' into '{out_dir}'...")

    if not raw_dir.exists():
        logger.error(f"Raw documents directory '{raw_dir}' does not exist.")
        raise FileNotFoundError(f"Raw documents directory '{raw_dir}' does not exist.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Loader and Chunker
    loader = DocumentLoader()
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 2. Discover supported files
    supported_files = [
        f for f in raw_dir.iterdir()
        if f.is_file() and f.suffix.lower() in loader.supported_extensions
    ]

    if not supported_files:
        logger.warning(f"No supported policy documents found in {raw_dir}.")
        print(f"\n[Ingestion Notice] No supported documents found in {raw_dir}.")
        print("Please place .pdf, .docx, or .pptx files into that directory.\n")
        return []

    print("\n==================================================")
    print("       HR POLICY RAG - DOCUMENT INGESTION")
    print("==================================================")
    print(f"Source Directory:     {raw_dir}")
    print(f"Target Directory:     {out_dir}")
    print(f"Discovered Files:     {len(supported_files)}")
    print(f"Chunk Config:         Size={chunk_size} chars | Overlap={chunk_overlap} chars")
    print(f"Embedding Provider:   {embed_provider or 'default'}\n")

    all_chunks: List[DocumentChunk] = []
    total_pages_or_slides = 0
    doc_stats: Dict[str, dict] = {}
    active_documents = set()
    archived_documents = set()
    failed_files = []

    # 3. Process each document through the pipeline
    for file_path in supported_files:
        doc_name = file_path.name
        print(f"Processing: {doc_name}...", end=" ", flush=True)

        try:
            pages: List[ExtractedPage] = loader.load_file(file_path)
            total_pages_or_slides += len(pages)

            chunks: List[DocumentChunk] = chunker.chunk_document_pages(pages)
            all_chunks.extend(chunks)

            doc_status = chunks[0].status if chunks else "active"
            if doc_status == "archived":
                archived_documents.add(doc_name)
            else:
                active_documents.add(doc_name)

            doc_stats[doc_name] = {
                "pages": len(pages),
                "chunks": len(chunks),
                "policy_id": chunks[0].policy_id if chunks else "unknown",
                "policy_type": chunks[0].policy_type if chunks else "unknown",
                "version": chunks[0].version if chunks else "unknown",
                "status": doc_status,
            }
            print(f"OK ({len(pages)} pages -> {len(chunks)} chunks [{doc_status}])")

        except Exception as exc:
            logger.error(f"Failed to process {doc_name}: {exc}")
            print(f"FAILED ({exc})")
            failed_files.append(doc_name)
            doc_stats[doc_name] = {"error": str(exc)}

    # 4. Save processed output to JSON and JSONL
    json_path = out_dir / "chunks.json"
    jsonl_path = out_dir / "chunks.jsonl"
    manifest_path = out_dir / "ingestion_manifest.json"

    chunks_data = [chunk.model_dump() for chunk in all_chunks]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in chunks_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 5. Generate embeddings
    embedder = get_embedding_service(provider=embed_provider)
    print(f"\nGenerating embeddings ({embedder.__class__.__name__}, dim={embedder.dimension})...", end=" ", flush=True)
    chunk_texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_documents(chunk_texts)
    print(f"OK ({len(embeddings)} vectors)")

    # 6. Index into Qdrant
    qdrant_status = "Skipped (--skip-qdrant)"
    upserted_count = 0

    if not skip_qdrant:
        try:
            store = QdrantStore()
            if store.health_check():
                store.create_collection(vector_size=embedder.dimension, recreate=recreate)
                upserted_count = store.upsert_documents(all_chunks, embeddings)
                qdrant_status = f"Indexed ({upserted_count} vectors)"
            else:
                qdrant_status = "Endpoint unreachable (Check Qdrant server)"
                logger.warning("Qdrant health check failed. Skipping vector upsert.")
        except QdrantConfigurationError as cfg_err:
            qdrant_status = f"Not configured ({cfg_err})"
            logger.info(f"Qdrant not configured: {cfg_err}")
        except Exception as q_exc:
            qdrant_status = f"Connection failed ({q_exc})"
            logger.warning(f"Could not connect to Qdrant: {q_exc}")

    # Save manifest
    manifest = {
        "documents_found": len(supported_files),
        "documents_processed": len(supported_files) - len(failed_files),
        "failed_documents": len(failed_files),
        "active_policies": len(active_documents),
        "archived_policies": len(archived_documents),
        "pages_processed": total_pages_or_slides,
        "chunks_created": len(all_chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_provider": embedder.__class__.__name__,
        "embedding_dimension": embedder.dimension,
        "qdrant_status": qdrant_status,
        "vectors_upserted": upserted_count,
        "document_breakdown": doc_stats,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 7. Print summary statistics
    avg_chunk_len = (
        sum(len(c.text) for c in all_chunks) // len(all_chunks) if all_chunks else 0
    )

    print("\n==================================================")
    print("INGESTION VALIDATION")
    print("==================================================")
    print(f"Documents found:                   {len(supported_files)}")
    print(f"Documents successfully processed:  {len(supported_files) - len(failed_files)}")
    print(f"Pages processed:                   {total_pages_or_slides}")
    print(f"Chunks created:                    {len(all_chunks)}")
    print(f"Active policies:                   {len(active_documents)}")
    print(f"Archived policies:                 {len(archived_documents)}")
    print(f"Failed documents:                  {len(failed_files)}")
    print(f"Vectors upserted to Qdrant:        {upserted_count}")
    print(f"Artifacts saved to:                {out_dir}")
    print("==================================================\n")

    return all_chunks


def main() -> None:
    """CLI execution entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup_logging()
    args = parse_args()
    try:
        ingest_documents(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            skip_qdrant=args.skip_qdrant,
            embed_provider=args.embed_provider,
            recreate=args.recreate,
        )
    except Exception as exc:
        logger.exception(f"Ingestion pipeline failed: {exc}")
        sys.exit(1)



if __name__ == "__main__":
    main()
