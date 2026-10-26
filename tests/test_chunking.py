"""Tests for text cleaning, section detection, metadata extraction, and chunking."""

from app.ingestion.chunker import DocumentChunker
from app.ingestion.cleaner import clean_extracted_text
from app.ingestion.loader import ExtractedPage
from app.ingestion.metadata import (
    DocumentChunk,
    extract_policy_type,
    extract_version,
    generate_chunk_id,
)


def test_cleaner_removes_excess_whitespace_and_newlines():
    """Verify clean_extracted_text cleans redundant spaces and blank lines."""
    raw = "Line 1   with   excess   spaces.\n\n\n\n\nLine 2 after huge gap.\r\nLine 3."
    cleaned = clean_extracted_text(raw)
    assert "Line 1 with excess spaces." in cleaned
    assert "\n\n\n" not in cleaned
    assert "Line 2 after huge gap.\nLine 3." in cleaned


def test_cleaner_dehyphenates_linebreaks():
    """Verify words broken across line wraps are rejoined."""
    raw = "Employees receive full compen-\nsation and addi-\ntional leave."
    cleaned = clean_extracted_text(raw)
    assert "compensation" in cleaned
    assert "additional" in cleaned


def test_cleaner_normalizes_unicode_artifacts():
    """Verify non-breaking space and control chars are normalized."""
    raw = "Word1\xa0Word2\x0cWord3\u200bWord4"
    cleaned = clean_extracted_text(raw)
    assert "Word1 Word2" in cleaned
    assert "\x0c" not in cleaned
    assert "\u200b" not in cleaned


def test_extract_policy_type_and_version():
    """Verify policy domain and version extraction from filename and content."""
    assert extract_policy_type("leave_policy_v2.pdf") == "leave"
    assert extract_policy_type("remote_work_handbook.docx") == "remote_work"
    assert extract_policy_type("travel_expenses.pptx") == "expense"
    assert extract_policy_type("attendance_rules.pdf") == "attendance"
    assert extract_policy_type("unknown_document.pdf") == "general"

    # Version extraction
    assert extract_version("leave_policy_2026.pdf") == "2026"
    assert extract_version("remote_work_policy_v2.4.docx") == "2.4"
    assert extract_version("annual_summary.pdf", text="Approved for fiscal year 2025") == "2025"


def test_generate_chunk_id_is_deterministic():
    """Verify chunk IDs are reproducible given identical inputs."""
    id1 = generate_chunk_id("leave_policy.pdf", page=1, section="Annual Leave", chunk_index=0, text="Sample text")
    id2 = generate_chunk_id("leave_policy.pdf", page=1, section="Annual Leave", chunk_index=0, text="Sample text")
    id3 = generate_chunk_id("leave_policy.pdf", page=1, section="Annual Leave", chunk_index=1, text="Different text")

    assert id1 == id2
    assert id1 != id3
    assert "leave_policy_p1_c000" in id1


def test_chunker_respects_chunk_size():
    """Verify that chunks do not exceed the configured maximum chunk size."""
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=40)
    text = (
        "First sentence of the policy. Second sentence detailing rules and conditions. "
        "Third sentence with additional criteria and requirements. "
        "Fourth sentence outlining reporting obligations. Fifth sentence on compliance."
    )
    chunks = chunker.chunk_section_text(text)

    assert len(chunks) >= 2
    for c in chunks:
        # Check within reasonable sentence boundary tolerance
        assert len(c) <= 220


def test_chunker_avoids_splitting_sentences_blindly():
    """Verify that sentence boundaries are respected."""
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)
    text = "Sentence one is clear. Sentence two is very informative. Sentence three concludes the section."
    chunks = chunker.chunk_section_text(text)

    for c in chunks:
        # Sentences should end with appropriate terminal punctuation
        assert c.endswith((".", "!", "?")) or c == chunks[-1]


def test_chunker_section_detection():
    """Verify section detection extracts section headings accurately."""
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=50)
    text = (
        "Introductory guidance for employees.\n\n"
        "1. Annual Leave Entitlement\n"
        "Employees get 20 days.\n\n"
        "2. Sick Leave\n"
        "Employees get 10 days."
    )
    sections = chunker.detect_sections_in_text(text, default_section="Overview")

    assert len(sections) == 3
    assert sections[0][0] == "Overview"
    assert "1. Annual Leave Entitlement" in sections[1][0]
    assert "2. Sick Leave" in sections[2][0]


def test_chunk_page_full_pipeline():
    """Verify chunk_page produces fully populated DocumentChunk models."""
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    page = ExtractedPage(
        document="leave_policy.pdf",
        document_path="data/raw/leave_policy.pdf",
        page=1,
        text=(
            "1. Annual Leave Entitlement\n"
            "Full-time employees receive 20 days of paid annual leave per calendar year. "
            "Accrual occurs at 1.67 days per month."
        ),
        section=None,
    )
    chunks = chunker.chunk_page(page)

    assert len(chunks) >= 1
    chunk = chunks[0]
    assert isinstance(chunk, DocumentChunk)
    assert chunk.document == "leave_policy.pdf"
    assert chunk.page == 1
    assert chunk.policy_type == "leave"
    assert "1. Annual Leave" in chunk.section
    assert "char_count" in chunk.metadata
    assert chunk.chunk_id.startswith("leave_policy_p1_c000")
