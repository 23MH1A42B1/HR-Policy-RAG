"""Metadata definitions and extraction utilities for policy document chunks.

Authoritative Company: XYZ Technologies Private Limited
Handles full metadata schema including policy_id, version, status (active/archived),
effective_date, expiry_date, access_level, and policy_type.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, model_validator


# Authoritative catalog mapping for XYZ Technologies policy documents
XYZ_POLICY_CATALOG: Dict[str, Dict[str, Any]] = {
    "01_XYZ_Employee_Handbook.pdf": {
        "policy_id": "HR-HANDBOOK-001",
        "version": "2.0",
        "status": "active",
        "policy_type": "handbook",
        "effective_date": "2026-04-01",
        "expiry_date": None,
    },
    "02_XYZ_Leave_Policy.pdf": {
        "policy_id": "HR-LEAVE-001",
        "version": "2.0",
        "status": "active",
        "policy_type": "leave",
        "effective_date": "2026-04-01",
        "expiry_date": None,
    },
    "03_XYZ_Remote_Work_Policy.pdf": {
        "policy_id": "HR-WFH-001",
        "version": "1.0",
        "status": "active",
        "policy_type": "remote_work",
        "effective_date": "2026-04-01",
        "expiry_date": None,
    },
    "04_XYZ_Attendance_Policy.pdf": {
        "policy_id": "HR-ATT-001",
        "version": "1.0",
        "status": "active",
        "policy_type": "attendance",
        "effective_date": "2026-04-01",
        "expiry_date": None,
    },
    "05_XYZ_Travel_Expense_Policy.pdf": {
        "policy_id": "FIN-EXP-001",
        "version": "1.0",
        "status": "active",
        "policy_type": "expense",
        "effective_date": "2026-04-01",
        "expiry_date": None,
    },
    "06_XYZ_Code_of_Conduct.pdf": {
        "policy_id": "HR-CONDUCT-001",
        "version": "1.0",
        "status": "active",
        "policy_type": "conduct",
        "effective_date": "2026-04-01",
        "expiry_date": None,
    },
    "07_XYZ_Benefits_Policy.pdf": {
        "policy_id": "HR-BEN-001",
        "version": "1.0",
        "status": "active",
        "policy_type": "benefits",
        "effective_date": "2026-04-01",
        "expiry_date": None,
    },
    "08_XYZ_IT_Security_Policy.pdf": {
        "policy_id": "IT-SEC-001",
        "version": "1.0",
        "status": "active",
        "policy_type": "security",
        "effective_date": "2026-04-01",
        "expiry_date": None,
    },
    "09_XYZ_Performance_Policy.pdf": {
        "policy_id": "HR-PERF-001",
        "version": "1.0",
        "status": "active",
        "policy_type": "performance",
        "effective_date": "2026-04-01",
        "expiry_date": None,
    },
    "10_XYZ_Separation_Policy.pdf": {
        "policy_id": "HR-EXIT-001",
        "version": "1.0",
        "status": "active",
        "policy_type": "separation",
        "effective_date": "2026-04-01",
        "expiry_date": None,
    },
    "ARCHIVED_XYZ_Leave_Policy_v1.0.pdf": {
        "policy_id": "HR-LEAVE-001",
        "version": "1.0",
        "status": "archived",
        "policy_type": "leave",
        "effective_date": "2025-04-01",
        "expiry_date": "2026-03-31",
    },
}


class DocumentChunk(BaseModel):
    """Pydantic model representing an ingested and chunked policy passage."""

    chunk_id: str = Field(
        ...,
        description="Unique deterministic identifier for the chunk",
    )
    text: str = Field(
        ...,
        description="Cleaned passage text content",
    )
    document: str = Field(
        ...,
        description="Original document file name, e.g. '02_XYZ_Leave_Policy.pdf'",
    )
    document_name: str = Field(
        default="",
        description="Canonical document name matching document",
    )
    company: str = Field(
        default="XYZ Technologies Private Limited",
        description="Corporate entity name",
    )
    policy_id: str = Field(
        default="GENERAL",
        description="Official policy identifier code, e.g. HR-LEAVE-001",
    )
    document_type: str = Field(
        default="hr_policy",
        description="Document type classification",
    )
    version: str = Field(
        default="2.0",
        description="Policy version string, e.g. '2.0' or '1.0'",
    )
    status: str = Field(
        default="active",
        description="Lifecycle status: 'active' or 'archived'",
    )
    effective_date: Optional[str] = Field(
        default="2026-04-01",
        description="Effective start date",
    )
    expiry_date: Optional[str] = Field(
        default=None,
        description="Expiry/sunset date if archived",
    )
    policy_type: str = Field(
        default="general",
        description="Categorized policy domain, e.g. 'leave', 'remote_work', 'expense'",
    )
    access_level: str = Field(
        default="employee",
        description="Access entitlement level",
    )
    page: int = Field(
        default=1,
        ge=1,
        description="1-indexed document page or presentation slide number",
    )
    section: str = Field(
        default="General",
        description="Detected section or topic heading",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supplementary metadata (e.g. char_count, word_count, file_type)",
    )

    @model_validator(mode="after")
    def populate_defaults_and_sync_metadata(self) -> DocumentChunk:
        """Ensure document_name and nested metadata dictionary stay fully in sync."""
        if not self.document_name:
            self.document_name = self.document

        # Guarantee all required schema fields are in self.metadata for consumers
        self.metadata["company"] = self.company
        self.metadata["policy_id"] = self.policy_id
        self.metadata["document_name"] = self.document_name
        self.metadata["document"] = self.document
        self.metadata["document_type"] = self.document_type
        self.metadata["version"] = self.version
        self.metadata["status"] = self.status
        self.metadata["effective_date"] = self.effective_date
        self.metadata["expiry_date"] = self.expiry_date
        self.metadata["policy_type"] = self.policy_type
        self.metadata["access_level"] = self.access_level
        self.metadata["page"] = self.page
        self.metadata["section"] = self.section
        self.metadata["chunk_id"] = self.chunk_id
        return self


# Recognized policy categories based on filename or text hints
POLICY_TYPE_MAPPINGS: Dict[str, list[str]] = {
    "leave": ["leave", "pto", "vacation", "holiday", "maternity", "paternity", "parental", "sick", "time_off"],
    "remote_work": ["remote", "work_from_home", "wfh", "telecommute", "hybrid_work", "hybrid"],
    "attendance": ["attendance", "punctuality", "hours", "overtime", "shift", "tardiness", "working_hours"],
    "expense": ["expense", "reimbursement", "travel", "per_diem", "corporate_card", "allowance", "hotel"],
    "conduct": ["conduct", "ethics", "harassment", "disciplinary", "whistleblower", "conflict_of_interest", "behaviour"],
    "benefits": ["benefit", "healthcare", "insurance", "401k", "pension", "wellness", "medical", "learning"],
    "security": ["security", "acceptable_use", "mfa", "password", "infosec", "cyber", "device"],
    "performance": ["performance", "career", "review", "pip", "promotion", "goals", "rating"],
    "separation": ["separation", "exit", "resignation", "termination", "handover", "settlement"],
    "handbook": ["handbook", "employee_handbook"],
}


def extract_policy_metadata(document_path: str, text: Optional[str] = None) -> Dict[str, Any]:
    """Extract authoritative metadata for a policy document.

    Checks predefined catalog first, then dynamically inspects document header patterns.
    """
    filename = Path(document_path).name

    # 1. Exact catalog match
    if filename in XYZ_POLICY_CATALOG:
        cat = dict(XYZ_POLICY_CATALOG[filename])
        cat["company"] = "XYZ Technologies Private Limited"
        cat["document_name"] = filename
        cat["document_type"] = "hr_policy"
        cat["access_level"] = "employee"
        return cat

    # 2. Dynamic extraction from text header
    header_text = text[:1500] if text else filename

    # Policy ID
    policy_id_m = re.search(r"Policy ID:\s*([A-Z0-9\-]+)", header_text, re.IGNORECASE)
    policy_id = policy_id_m.group(1).upper() if policy_id_m else "GENERAL"

    # Version
    ver_m = re.search(r"Version:\s*([0-9]+(?:\.[0-9]+)*)", header_text, re.IGNORECASE)
    version = ver_m.group(1) if ver_m else extract_version(document_path, text)

    # Status
    is_archived = (
        "archived" in filename.lower()
        or "archived" in header_text.lower()
        or "not current" in header_text.lower()
    )
    status = "archived" if is_archived else "active"

    # Dates
    eff_m = re.search(r"Effective Date:\s*([0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}|[0-9]{4}-[0-9]{2}-[0-9]{2})", header_text, re.IGNORECASE)
    effective_date = eff_m.group(1) if eff_m else "2026-04-01"

    expiry_date = "2026-03-31" if status == "archived" else None

    # Policy Type
    policy_type = extract_policy_type(document_path, text)

    return {
        "company": "XYZ Technologies Private Limited",
        "policy_id": policy_id,
        "document_name": filename,
        "document_type": "hr_policy",
        "version": version,
        "status": status,
        "effective_date": effective_date,
        "expiry_date": expiry_date,
        "policy_type": policy_type,
        "access_level": "employee",
    }


def extract_policy_type(document_path: str, text: Optional[str] = None) -> str:
    """Infer the policy domain from filename or document content."""
    filename = Path(document_path).stem.lower().replace("-", "_")

    # 1. Match against filename keywords
    for policy_type, keywords in POLICY_TYPE_MAPPINGS.items():
        if any(keyword in filename for keyword in keywords):
            return policy_type

    # 2. Match against opening content if provided
    if text:
        content_sample = text[:1500].lower()
        for policy_type, keywords in POLICY_TYPE_MAPPINGS.items():
            if any(re.search(rf"\b{re.escape(k)}\b", content_sample) for k in keywords):
                return policy_type

    return "general"


def extract_version(document_path: str, text: Optional[str] = None) -> str:
    """Infer the policy version or effective year from filename or text content."""
    source_str = f"{Path(document_path).name} {text[:1000] if text else ''}"

    # Search for explicit version tag e.g. "v2.1", "version 1.0", "Rev 2026", "_v2.4"
    ver_match = re.search(r"(?:^|[\b_\s])(?:v|version|rev)\.?\s*([0-9]+(?:\.[0-9]+)*)", source_str, re.IGNORECASE)
    if ver_match:
        return ver_match.group(1)

    # Search for year 2020 - 2035
    year_match = re.search(r"\b(202[0-9]|203[0-5])\b", source_str)
    if year_match:
        return year_match.group(1)

    return "2026"


def generate_chunk_id(document: str, page: int, section: str, chunk_index: int, text: str) -> str:
    """Generate a deterministic, unique chunk ID combining doc, page, index, and content hash."""
    clean_doc = re.sub(r"[^a-zA-Z0-9_]", "_", Path(document).stem)
    content_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:8]
    return f"{clean_doc}_p{page}_c{chunk_index:03d}_{content_hash}"
