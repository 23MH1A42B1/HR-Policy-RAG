"""BM25 lexical search for exact keyword, policy code, date, and identifier matching."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from rank_bm25 import BM25Plus

from app.ingestion.metadata import DocumentChunk
from app.utils.logging import get_logger

logger = get_logger("app.retrieval.bm25_search")

# Regex to capture exact policy terms, hyphenated codes (HR-POL-2026-01), dates, and currencies
TOKEN_PATTERN = re.compile(
    r"\b[a-zA-Z0-9]+(?:[-_/\.][a-zA-Z0-9]+)*\b|\$[0-9]+(?:\.[0-9]+)?",
    re.IGNORECASE,
)


def _inflect_variants(token: str) -> List[str]:
    """Generate basic singular/plural variants for English vocabulary."""
    variants: List[str] = []
    if token == "leaves":
        variants.append("leave")
    elif token.endswith("ies") and len(token) > 4:
        variants.append(token[:-3] + "y")  # policies -> policy
    elif token.endswith("es") and len(token) > 3 and token[-3] in ("s", "x", "z", "ch", "sh"):
        variants.append(token[:-2])
    elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        variants.append(token[:-1])  # days -> day, hours -> hour
    return variants


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text while strictly preserving exact codes, policy IDs, dates, and terms.

    For hyphenated terms like 'HR-POL-2026-01', retains both the full composite token
    and individual constituent tokens for flexible keyword matching.
    """
    if not text:
        return []

    tokens: List[str] = []
    matches = TOKEN_PATTERN.findall(text.lower())

    for match in matches:
        tokens.append(match)
        # Add basic inflected variants (e.g. leaves -> leave, policies -> policy)
        for v in _inflect_variants(match):
            tokens.append(v)
        # If match contains separators, also add decomposed sub-tokens
        if any(sep in match for sep in ("-", "_", "/", ".")):
            parts = re.split(r"[-_/\.]", match)
            for part in parts:
                if part and part != match:
                    tokens.append(part)
                    for v in _inflect_variants(part):
                        tokens.append(v)

    return tokens


# Alias for backward compatibility / explicit testing naming
tokenize_code_and_text = tokenize_for_bm25


class BM25Result(BaseModel):
    """Result model returned by BM25 lexical search."""

    chunk_id: str = Field(description="Unique chunk identifier")
    document: str = Field(description="Original document filename")
    page: int = Field(description="1-indexed page or slide number")
    section: str = Field(description="Section heading or topic")
    text: str = Field(description="Passage text content")
    score: float = Field(description="BM25 lexical relevance score")
    policy_type: str = Field(default="general", description="Policy category")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class BM25SearchIndex:
    """Manages inverted index and BM25 scoring across document chunks."""

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None) -> None:
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Plus] = None
        self.corpus_tokens: List[List[str]] = []

        if chunks:
            self.build_index(chunks)

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build the BM25 index from a list of DocumentChunk instances."""
        if not chunks:
            logger.warning("Empty chunk list provided to BM25SearchIndex.")
            self.chunks = []
            self.bm25 = None
            self.corpus_tokens = []
            return

        self.chunks = list(chunks)
        logger.info(f"Tokenizing {len(self.chunks)} document chunk(s) for BM25...")

        self.corpus_tokens = []
        for c in self.chunks:
            # Combine section and text for richer keyword indexing
            full_content = f"{c.document} {c.section} {c.text}"
            self.corpus_tokens.append(tokenize_for_bm25(full_content))

        self.bm25 = BM25Plus(self.corpus_tokens)
        logger.info(f"BM25 index successfully constructed across {len(self.chunks)} chunks.")

    def search(
        self,
        query: str,
        top_k: int = 20,
        policy_type: Optional[str] = None,
        status: Optional[str] = None,
        policy_id: Optional[str] = None,
    ) -> List[BM25Result]:
        """Execute BM25 lexical search for exact terms and keywords with metadata filtering.

        Args:
            query: User's search query.
            top_k: Number of top candidate results to return.
            policy_type: Optional policy filter (e.g. 'leave').
            status: Optional lifecycle filter ('active' or 'archived').
            policy_id: Optional policy ID filter (e.g. 'HR-LEAVE-001').

        Returns:
            List of BM25Result models sorted in descending order of score.
        """
        if self.bm25 is None or not self.chunks:
            logger.warning("BM25 search called on uninitialized index.")
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        query_set = set(query_tokens)

        # Pair each chunk with its score and filter
        scored_items = []
        for idx, score in enumerate(raw_scores):
            if score <= 0.0:
                continue

            # Ensure chunk contains at least one query token
            if not (query_set & set(self.corpus_tokens[idx])):
                continue

            chunk = self.chunks[idx]
            if policy_type and chunk.policy_type.lower() != policy_type.lower():
                continue

            chunk_status = getattr(chunk, "status", None) or chunk.metadata.get("status", "active")
            if status and chunk_status.lower() != status.lower():
                continue

            chunk_policy_id = getattr(chunk, "policy_id", None) or chunk.metadata.get("policy_id", "")
            if policy_id and chunk_policy_id.upper() != policy_id.upper():
                continue

            scored_items.append((score, chunk))

        # Sort descending by BM25 score
        scored_items.sort(key=lambda x: x[0], reverse=True)

        results: List[BM25Result] = []
        for score, chunk in scored_items[:top_k]:
            results.append(
                BM25Result(
                    chunk_id=chunk.chunk_id,
                    document=chunk.document,
                    page=chunk.page,
                    section=chunk.section,
                    text=chunk.text,
                    score=float(score),
                    policy_type=chunk.policy_type,
                    metadata=chunk.metadata,
                )
            )

        return results

    def _rebuild_index(self) -> None:
        """Reconstruct BM25Plus model across current chunks."""
        self.corpus_tokens = []
        for c in self.chunks:
            full_content = f"{c.document} {c.section} {c.text}"
            self.corpus_tokens.append(tokenize_for_bm25(full_content))
        if self.corpus_tokens:
            self.bm25 = BM25Plus(self.corpus_tokens)
        else:
            self.bm25 = None

    def add_chunks(self, new_chunks: List[DocumentChunk]) -> None:
        """Add new chunks to the BM25 index and refresh the model."""
        if not new_chunks:
            return
        existing_ids = {c.chunk_id for c in self.chunks}
        filtered = [c for c in new_chunks if c.chunk_id not in existing_ids]
        self.chunks.extend(filtered)
        self._rebuild_index()
        logger.info(f"Added {len(filtered)} chunks to BM25 index (total: {len(self.chunks)})")

    def remove_document(self, document_name: str) -> int:
        """Remove all chunks belonging to document_name and refresh the model."""
        orig_len = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.document != document_name]
        removed_count = orig_len - len(self.chunks)
        self._rebuild_index()
        logger.info(f"Removed {removed_count} chunks from BM25 index for document '{document_name}'")
        return removed_count

    def save(self, filepath: str | Path) -> None:
        """Persist chunks to JSON so index can be rehydrated quickly."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [c.model_dump() for c in self.chunks]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved BM25 source data ({len(self.chunks)} chunks) to {path}")

    @classmethod
    def load(cls, filepath: str | Path) -> BM25SearchIndex:
        """Load chunks from disk and initialize index."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"BM25 source file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        chunks = [DocumentChunk(**item) for item in raw_data]
        return cls(chunks=chunks)


_global_bm25_index: Optional[BM25SearchIndex] = None


def get_bm25_index(chunks_path: Optional[str | Path] = None) -> BM25SearchIndex:
    """Retrieve or initialize singleton BM25SearchIndex from processed chunks."""
    global _global_bm25_index
    if _global_bm25_index is not None:
        return _global_bm25_index

    default_path = Path(chunks_path or "data/processed/chunks.json")
    if default_path.exists():
        try:
            _global_bm25_index = BM25SearchIndex.load(default_path)
            return _global_bm25_index
        except Exception as exc:
            logger.warning(f"Failed loading BM25 index from {default_path}: {exc}")

    _global_bm25_index = BM25SearchIndex()
    return _global_bm25_index
