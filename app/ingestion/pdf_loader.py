"""PDF document loader and text extractor using pypdf."""

from __future__ import annotations

from pathlib import Path
from typing import List
from pypdf import PdfReader
from app.utils.logging import get_logger

logger = get_logger("app.ingestion.pdf_loader")


class PDFLoader:
    """Extracts text page-by-page from PDF documents."""

    def __init__(self, file_path: str | Path) -> None:
        self.path = Path(file_path)

    def load(self) -> List[dict]:
        """Extract pages from the target PDF file.

        Returns:
            List of dicts with:
                - document: filename
                - page: 1-indexed page number
                - text: raw extracted text
        """
        if not self.path.exists():
            logger.error(f"PDF file not found: {self.path}")
            raise FileNotFoundError(f"PDF file not found: {self.path}")

        extracted_pages: List[dict] = []
        try:
            reader = PdfReader(str(self.path))

            if reader.is_encrypted:
                try:
                    # Attempt decrypt with empty password for unprotected encrypted files
                    reader.decrypt("")
                except Exception as exc:
                    logger.warning(f"Encrypted PDF cannot be read without credentials: {self.path.name} ({exc})")
                    return []

            total_pages = len(reader.pages)
            logger.info(f"Extracting {total_pages} page(s) from PDF: {self.path.name}")

            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                    extracted_pages.append({
                        "document": self.path.name,
                        "document_path": str(self.path),
                        "page": page_idx,
                        "text": page_text,
                    })
                except Exception as page_exc:
                    logger.warning(
                        f"Failed extracting text on page {page_idx} of {self.path.name}: {page_exc}"
                    )
                    extracted_pages.append({
                        "document": self.path.name,
                        "document_path": str(self.path),
                        "page": page_idx,
                        "text": "",
                    })

        except Exception as exc:
            logger.error(f"Failed to open or parse PDF '{self.path.name}': {exc}")
            raise ValueError(f"Corrupted or invalid PDF file '{self.path.name}': {exc}") from exc

        return extracted_pages
