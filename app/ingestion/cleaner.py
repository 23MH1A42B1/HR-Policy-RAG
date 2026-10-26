"""Text cleaning module for document extraction artifacts.

Removes redundant whitespace, hyphenation splits, and OCR/PDF artifacts
while preserving semantic layout, bullet points, and section formatting.
"""

from __future__ import annotations

import re


def clean_extracted_text(raw_text: str) -> str:
    """Clean and normalize raw extracted document text.

    Args:
        raw_text: Unprocessed text from PDF, DOCX, or PPTX parser.

    Returns:
        Normalized, clean text ready for chunking.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text

    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Normalize unicode whitespace and remove control/form-feed artifacts
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\ufeff", "")  # byte order mark
    text = text.replace("\x0c", "\n")  # form-feed (page separator)
    text = text.replace("\x0b", " ")   # vertical tab

    # 3. De-hyphenate words broken across line wraps (e.g. "compen-\nsation" -> "compensation")
    # Only applies to alphabetic word stems
    text = re.sub(r"([a-zA-Z]{2,})-\n\s*([a-zA-Z]{2,})", r"\1\2", text)

    # 4. Clean trailing spaces and excessive inline whitespace
    lines = []
    for line in text.split("\n"):
        # Replace multiple horizontal spaces/tabs with single space
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)

    text = "\n".join(lines)

    # 5. Collapse three or more consecutive blank lines into double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Normalize smart quotes and dashes for consistency
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", " - ")

    return text.strip()
