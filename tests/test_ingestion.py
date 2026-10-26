"""Tests for document loading and parsing (PDF, DOCX, PPTX)."""

from pathlib import Path
import pytest
from app.ingestion.docx_loader import DOCXLoader
from app.ingestion.loader import DocumentLoader, UnsupportedFileTypeError
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.pptx_loader import PPTXLoader

RAW_DATA_DIR = Path("data/raw")


def test_pdf_loader_valid():
    """Verify PDFLoader extracts text and page numbers from sample PDF."""
    pdf_path = RAW_DATA_DIR / "02_XYZ_Leave_Policy.pdf"
    if not pdf_path.exists():
        pdf_path = RAW_DATA_DIR / "leave_policy.pdf"
    assert pdf_path.exists(), f"Sample {pdf_path.name} must exist"

    loader = PDFLoader(pdf_path)
    pages = loader.load()

    assert len(pages) >= 1
    for p in pages:
        assert p["document"] == pdf_path.name
        assert p["page"] >= 1
        assert isinstance(p["text"], str)

    combined = " ".join(p["text"] for p in pages)
    assert "Annual Leave" in combined
    assert "XYZ Technologies" in combined or "Acme" in combined


def test_docx_loader_valid():
    """Verify DOCXLoader extracts headings, paragraphs, and tables."""
    docx_path = Path("data/raw_demo_backup/remote_work_policy.docx")
    if not docx_path.exists():
        docx_path = RAW_DATA_DIR / "remote_work_policy.docx"
    assert docx_path.exists(), "Sample remote_work_policy.docx must exist"

    loader = DOCXLoader(docx_path)
    pages = loader.load()

    assert len(pages) >= 1
    page1 = pages[0]
    assert page1["document"] == "remote_work_policy.docx"
    assert page1["page"] >= 1

    combined = " ".join(p["text"] for p in pages)
    assert "Remote" in combined
    assert "Ergonomic" in combined or "Stipend" in combined
    # Table data extraction verification
    assert "[Table Data]:" in combined or "Hybrid" in combined


def test_pptx_loader_valid():
    """Verify PPTXLoader extracts slide titles, bullets, and notes."""
    pptx_path = Path("data/raw_demo_backup/expense_policy.pptx")
    if not pptx_path.exists():
        pptx_path = RAW_DATA_DIR / "expense_policy.pptx"
    assert pptx_path.exists(), "Sample expense_policy.pptx must exist"

    loader = PPTXLoader(pptx_path)
    slides = loader.load()

    assert len(slides) == 5
    for s in slides:
        assert s["document"] == "expense_policy.pptx"
        assert s["page"] in [1, 2, 3, 4, 5]
        assert s["section"] != ""

    combined = " ".join(s["text"] for s in slides)
    assert "Corporate Travel" in combined or "Per Diem" in combined
    assert "[Presenter Notes]:" in combined


def test_unsupported_file_type_raises_error(tmp_path: Path):
    """Verify DocumentLoader raises UnsupportedFileTypeError for non-supported extensions."""
    unsupported_file = tmp_path / "notes.txt"
    unsupported_file.write_text("Some text notes", encoding="utf-8")

    loader = DocumentLoader()
    with pytest.raises(UnsupportedFileTypeError) as excinfo:
        loader.validate_file(unsupported_file)

    assert "Unsupported file format '.txt'" in str(excinfo.value)


def test_empty_file_raises_error(tmp_path: Path):
    """Verify DocumentLoader rejects 0-byte empty files."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    loader = DocumentLoader()
    with pytest.raises(ValueError) as excinfo:
        loader.validate_file(empty_file)

    assert "empty (0 bytes)" in str(excinfo.value)


def test_missing_file_raises_not_found():
    """Verify DocumentLoader raises FileNotFoundError for missing paths."""
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.validate_file("data/raw/non_existent_policy.pdf")


def test_document_loader_load_directory():
    """Verify load_directory discovers and loads all valid policy documents."""
    loader = DocumentLoader()
    pages = loader.load_directory(RAW_DATA_DIR)

    assert len(pages) > 0
    docs = {p.document for p in pages}
    assert any("Leave_Policy" in d for d in docs)
    assert any("Employee_Handbook" in d or "Handbook" in d for d in docs)
