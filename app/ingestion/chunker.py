"""Configurable, section-aware document chunker for policy documents.

Splits cleaned text respecting section boundaries and sentence structure
to avoid cutting sentences mid-thought, while preserving metadata.
"""

from __future__ import annotations

import re
from typing import List
from app.ingestion.cleaner import clean_extracted_text
from app.ingestion.loader import ExtractedPage
from app.ingestion.metadata import (
    DocumentChunk,
    extract_policy_metadata,
    extract_policy_type,
    extract_version,
    generate_chunk_id,
)
from app.utils.logging import get_logger

logger = get_logger("app.ingestion.chunker")

# Regex to detect section headers (numbered, markdown, formal articles, or special headings)
SECTION_HEADER_REGEX = re.compile(
    r"(?:^|\n)\s*((?:[0-9]{1,2}(?:\.[0-9]{1,2})*\.\s+[A-Za-z][^\n]{2,70})|"
    r"(?:#{1,3}\s+[^\n]{2,70})|"
    r"(?:(?:Section|Article|Part)\s+[0-9IVXLCDM]+[:\.\s][^\n]{2,70})|"
    r"(?:ARCHIVED\s+[—\-]\s+NOT\s+CURRENT)|"
    r"(?:Previous\s+Entitlements)|"
    r"(?:Replacement)|"
    r"(?:RAG\s+Metadata\s+Requirement)|"
    r"(?:Policy\s+governance))",
    re.IGNORECASE,
)

# Sentence boundary regex (protects common abbreviations like e.g., i.e., vs., Dr., Mr.)
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
KNOWN_ABBREVIATIONS = {"e.g.", "i.e.", "vs.", "etc.", "mr.", "ms.", "mrs.", "dr.", "rev.", "inc.", "corp.", "ltd.", "al."}


class DocumentChunker:
    """Chunks documents into semantically coherent passages with overlap and metadata."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        min_chunk_length: int = 15,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while respecting common abbreviations."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sentences: List[str] = []

        for p in paragraphs:
            # Check for bullet point lines
            lines = p.split("\n")
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                # If bullet point or numbered item, treat line as a unit
                if re.match(r"^[-*•]\s+|^\d+\.\s+", line_str):
                    sentences.append(line_str)
                else:
                    raw_splits = SENTENCE_SPLIT_REGEX.split(line_str)
                    merged: List[str] = []
                    for s in raw_splits:
                        s_clean = s.strip()
                        if not s_clean:
                            continue
                        # If previous piece ended with known abbreviation, merge with current
                        if merged and any(merged[-1].lower().endswith(abbr) for abbr in KNOWN_ABBREVIATIONS):
                            merged[-1] = f"{merged[-1]} {s_clean}"
                        else:
                            merged.append(s_clean)
                    sentences.extend(merged)

        return sentences

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split an exceptionally long sentence by word boundaries."""
        words = sentence.split(" ")
        chunks = []
        current = []
        curr_len = 0

        for w in words:
            if curr_len + len(w) + 1 > self.chunk_size and current:
                chunks.append(" ".join(current))
                current = [w]
                curr_len = len(w)
            else:
                current.append(w)
                curr_len += len(w) + 1

        if current:
            chunks.append(" ".join(current))
        return chunks

    def chunk_section_text(self, text: str) -> List[str]:
        """Chunk a section's text into overlapping blocks with sentence preservation."""
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        # Expand any single sentences that exceed chunk_size
        expanded_sentences: List[str] = []
        for s in sentences:
            if len(s) > self.chunk_size:
                expanded_sentences.extend(self._split_long_sentence(s))
            else:
                expanded_sentences.append(s)

        chunks: List[str] = []
        current_chunk_sentences: List[str] = []
        current_length = 0

        for sent in expanded_sentences:
            sent_len = len(sent)
            # If adding this sentence exceeds chunk size and we already have content
            if current_length + sent_len + 1 > self.chunk_size and current_chunk_sentences:
                chunk_str = " ".join(current_chunk_sentences).strip()
                if len(chunk_str) >= self.min_chunk_length:
                    chunks.append(chunk_str)

                # Determine overlap sentences to carry forward
                overlap_sentences: List[str] = []
                overlap_len = 0
                for rev_sent in reversed(current_chunk_sentences):
                    if overlap_len + len(rev_sent) + 1 <= self.chunk_overlap:
                        overlap_sentences.insert(0, rev_sent)
                        overlap_len += len(rev_sent) + 1
                    else:
                        break

                current_chunk_sentences = list(overlap_sentences)
                current_length = sum(len(s) + 1 for s in current_chunk_sentences)

            current_chunk_sentences.append(sent)
            current_length += sent_len + 1

        # Append trailing chunk
        if current_chunk_sentences:
            final_chunk = " ".join(current_chunk_sentences).strip()
            if len(final_chunk) >= self.min_chunk_length:
                # Avoid duplicate if it matches the last chunk
                if not chunks or final_chunk != chunks[-1]:
                    chunks.append(final_chunk)
            elif not chunks:
                # Always keep if it is the only content
                chunks.append(final_chunk)

        return chunks

    def detect_sections_in_text(self, text: str, default_section: str = "General") -> List[tuple[str, str]]:
        """Split text into (section_name, section_text) tuples."""
        matches = list(SECTION_HEADER_REGEX.finditer(text))
        if not matches:
            return [(default_section, text)]

        sections: List[tuple[str, str]] = []
        # Any text before the first heading belongs to default section
        first_start = matches[0].start()
        if first_start > 0:
            intro_text = text[:first_start].strip()
            if intro_text:
                sections.append((default_section, intro_text))

        for idx, match in enumerate(matches):
            raw_title = match.group(1).strip() if match.lastindex else match.group(0).strip()
            # Clean leading Markdown hashes
            clean_title = re.sub(r"^#{1,3}\s*", "", raw_title).strip()
            start_pos = match.end()
            end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body_text = text[start_pos:end_pos].strip()

            sections.append((clean_title, body_text))

        return sections

    def chunk_page(self, page: ExtractedPage) -> List[DocumentChunk]:
        """Process and chunk a single ExtractedPage into DocumentChunk instances."""
        cleaned_text = clean_extracted_text(page.text)
        if not cleaned_text:
            return []

        doc_name = page.document
        meta_dict = extract_policy_metadata(page.document_path or doc_name, cleaned_text)
        policy_type = meta_dict.get("policy_type", "general")
        version = meta_dict.get("version", "1.0")
        company = meta_dict.get("company", "XYZ Technologies Private Limited")
        policy_id = meta_dict.get("policy_id", "GENERAL")
        status = meta_dict.get("status", "active")
        effective_date = meta_dict.get("effective_date", "2026-04-01")
        expiry_date = meta_dict.get("expiry_date", None)
        access_level = meta_dict.get("access_level", "employee")
        document_type = meta_dict.get("document_type", "hr_policy")

        base_section = page.section or "General"

        # Detect any sub-sections within the page text
        detected_sections = self.detect_sections_in_text(cleaned_text, default_section=base_section)

        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for section_name, section_body in detected_sections:
            if not section_body:
                continue

            raw_chunks = self.chunk_section_text(section_body)
            for raw_chunk in raw_chunks:
                chunk_id = generate_chunk_id(
                    document=doc_name,
                    page=page.page,
                    section=section_name,
                    chunk_index=chunk_idx,
                    text=raw_chunk,
                )

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=raw_chunk,
                    document=doc_name,
                    document_name=doc_name,
                    company=company,
                    policy_id=policy_id,
                    document_type=document_type,
                    version=version,
                    status=status,
                    effective_date=effective_date,
                    expiry_date=expiry_date,
                    policy_type=policy_type,
                    access_level=access_level,
                    page=page.page,
                    section=section_name,
                    metadata={
                        "company": company,
                        "policy_id": policy_id,
                        "document_name": doc_name,
                        "document": doc_name,
                        "document_type": document_type,
                        "version": version,
                        "status": status,
                        "effective_date": effective_date,
                        "expiry_date": expiry_date,
                        "policy_type": policy_type,
                        "access_level": access_level,
                        "page": page.page,
                        "section": section_name,
                        "chunk_id": chunk_id,
                        "char_count": len(raw_chunk),
                        "word_count": len(raw_chunk.split()),
                        "source_path": page.document_path,
                    },
                )
                chunks.append(chunk)
                chunk_idx += 1

        return chunks

    def chunk_document_pages(self, pages: List[ExtractedPage]) -> List[DocumentChunk]:
        """Chunk a list of extracted pages from a document."""
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page)
            all_chunks.extend(page_chunks)
        return all_chunks
