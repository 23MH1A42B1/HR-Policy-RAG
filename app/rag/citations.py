"""Citations module for extracting, validating, and formatting document citations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Data model representing a verified source citation."""

    policy_id: Optional[str] = Field(default=None, description="Policy code identifier (e.g. 'HR-LEAVE-001')")
    version: Optional[str] = Field(default=None, description="Policy revision version (e.g. '2.0')")
    document: str = Field(description="Policy document name (e.g. 'Leave Policy' or '02_XYZ_Leave_Policy.pdf')")
    page: Optional[int] = Field(default=None, description="Verified page or slide number")
    section: Optional[str] = Field(default=None, description="Verified section heading")
    snippet: Optional[str] = Field(default=None, description="Relevant excerpt text")
    score: Optional[float] = Field(default=None, description="Retrieval similarity score")

    def to_ui_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching Streamlit UI Citation model."""
        source_label = f"[{self.policy_id} v{self.version}] {self.document}" if self.policy_id else self.document
        return {
            "source": source_label,
            "document": self.document,
            "policy_id": self.policy_id,
            "version": self.version,
            "page": self.page,
            "section": self.section,
            "snippet": self.snippet,
        }


def format_document_display_name(filename: str) -> str:
    """Format file names into human-readable policy titles."""
    stem = Path(filename).stem
    # Strip leading number prefixes like '02_XYZ_'
    clean_stem = stem
    if clean_stem.startswith(("01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_")):
        clean_stem = clean_stem[3:]
    if clean_stem.startswith("XYZ_"):
        clean_stem = clean_stem[4:]
    if clean_stem.startswith("ARCHIVED_XYZ_"):
        clean_stem = "Archived " + clean_stem[13:]

    words = clean_stem.replace("_", " ").replace("-", " ").split()
    return " ".join(w.capitalize() for w in words)


def extract_citations_from_results(
    results: Sequence[Any],
    max_citations: int = 4,
) -> List[SourceCitation]:
    """Extract and deduplicate citations from retrieval search results."""
    seen_keys = set()
    citations: List[SourceCitation] = []

    for r in results:
        meta = getattr(r, "metadata", {}) or {}
        policy_id = meta.get("policy_id") or getattr(r, "policy_id", None)
        version = meta.get("version") or getattr(r, "version", None)

        key = (r.document.lower(), r.page, (r.section or "").lower(), policy_id, version)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        clean_doc = format_document_display_name(r.document)
        snippet_text = r.text[:220].strip()
        if len(r.text) > 220:
            snippet_text += "..."

        citations.append(
            SourceCitation(
                policy_id=policy_id,
                version=version,
                document=clean_doc,
                page=r.page,
                section=r.section,
                snippet=snippet_text,
                score=r.score,
            )
        )

        if len(citations) >= max_citations:
            break

    return citations


def format_citations_markdown(citations: List[SourceCitation]) -> str:
    """Format citations into a standardized markdown representation."""
    if not citations:
        return ""

    lines = ["**Sources & Citations:**\n"]
    for c in citations:
        prefix = f"**{c.policy_id} (v{c.version})** — " if c.policy_id else ""
        doc_line = f"📄 {prefix}{c.document}"
        page_line = f"Page {c.page}" if c.page is not None else ""
        section_line = f"Section: {c.section}" if c.section else ""

        citation_block = [doc_line]
        if page_line:
            citation_block.append(page_line)
        if section_line:
            citation_block.append(section_line)

        lines.append(" | ".join(citation_block))

    return "\n\n".join(lines).strip()
