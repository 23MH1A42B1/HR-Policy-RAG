"""Unified document loader and validator supporting PDF, DOCX, and PPTX."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set
from pydantic import BaseModel, Field

from app.ingestion.docx_loader import DOCXLoader
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.pptx_loader import PPTXLoader
from app.utils.logging import get_logger

logger = get_logger("app.ingestion.loader")

SUPPORTED_EXTENSIONS: Set[str] = {".pdf", ".docx", ".pptx"}


class UnsupportedFileTypeError(ValueError):
    """Raised when an unsupported document format is provided for ingestion."""
    pass


class ExtractedPage(BaseModel):
    """Represents raw extracted content from a document page, logical section, or slide."""

    document: str = Field(description="Filename of the source document")
    document_path: str = Field(description="Full path to source document")
    page: int = Field(default=1, ge=1, description="1-indexed page or slide number")
    text: str = Field(default="", description="Extracted raw text content")
    section: Optional[str] = Field(default=None, description="Identified section heading if any")


class DocumentLoader:
    """Validates files and orchestrates multi-format extraction."""

    def __init__(self, supported_extensions: Optional[Set[str]] = None) -> None:
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS

    def validate_file(self, file_path: str | Path) -> Path:
        """Validate that a file exists, is non-empty, and has a supported extension."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")

        if path.stat().st_size == 0:
            raise ValueError(f"File is empty (0 bytes): {path.name}")

        ext = path.suffix.lower()
        if ext not in self.supported_extensions:
            raise UnsupportedFileTypeError(
                f"Unsupported file format '{ext}' for file '{path.name}'. "
                f"Supported formats: {sorted(list(self.supported_extensions))}"
            )

        return path

    def load_file(self, file_path: str | Path) -> List[ExtractedPage]:
        """Validate and load a single document into ExtractedPage instances."""
        path = self.validate_file(file_path)
        ext = path.suffix.lower()

        logger.info(f"Loading document: {path.name} (type: {ext})")
        raw_items: List[dict] = []

        try:
            if ext == ".pdf":
                raw_items = PDFLoader(path).load()
            elif ext == ".docx":
                raw_items = DOCXLoader(path).load()
            elif ext == ".pptx":
                raw_items = PPTXLoader(path).load()
        except Exception as exc:
            logger.error(f"Failed loading document '{path.name}': {exc}")
            raise

        extracted_pages = [
            ExtractedPage(
                document=item.get("document", path.name),
                document_path=item.get("document_path", str(path)),
                page=item.get("page", 1),
                text=item.get("text", ""),
                section=item.get("section"),
            )
            for item in raw_items
        ]

        logger.info(f"Loaded {len(extracted_pages)} extracted page(s)/slide(s) from {path.name}")
        return extracted_pages

    def load_directory(self, directory_path: str | Path) -> List[ExtractedPage]:
        """Scan a directory, discover all supported files, and load them."""
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        all_pages: List[ExtractedPage] = []
        found_files = [
            p for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in self.supported_extensions
        ]

        logger.info(f"Discovered {len(found_files)} supported file(s) in {dir_path}")

        for file_path in sorted(found_files):
            try:
                pages = self.load_file(file_path)
                all_pages.extend(pages)
            except Exception as exc:
                logger.warning(f"Skipping failed file {file_path.name}: {exc}")

        return all_pages
