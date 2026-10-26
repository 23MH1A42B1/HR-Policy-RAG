"""Reusable UI components for HR Policy RAG Assistant.

Implements enterprise styling: clean white background, purple accents,
rounded modern cards, and responsive layout.
"""

from __future__ import annotations

from typing import List, Optional
import streamlit as st
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Data model representing a source citation for RAG retrieval."""

    source: str = Field(default="", description="Policy document name or formatted label")
    document_name: Optional[str] = Field(default=None, description="Clean document display title")
    policy_id: Optional[str] = Field(default=None, description="Policy code identifier (e.g. 'HR-LEAVE-001')")
    version: Optional[str] = Field(default=None, description="Policy version string (e.g. '2.0')")
    page: Optional[int] = Field(default=None, description="Page number where content is found")
    section: Optional[str] = Field(default=None, description="Section heading, e.g. 'Annual Leave'")
    snippet: Optional[str] = Field(default=None, description="Relevant excerpt text from document")

    def model_post_init(self, __context) -> None:
        """Ensure source label is populated if document_name is given."""
        if not self.source and self.document_name:
            self.source = self.document_name


def inject_custom_styles(is_authenticated: bool = True) -> None:
    """Inject modern enterprise CSS styling for Streamlit."""
    sidebar_hide_rule = ""
    if not is_authenticated:
        sidebar_hide_rule = """
        <style>
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        </style>
        """

    base_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #1E293B;
        }

        /* App background and root cleanup */
        .stApp {
            background-color: #FFFFFF;
        }

        /* App container space cleanup */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 1050px !important;
        }

        /* Hide Streamlit Community Cloud top-right header buttons (Fork, GitHub, Menu, Deploy) */
        #MainMenu { visibility: hidden !important; display: none !important; }
        footer { visibility: hidden !important; display: none !important; }
        [data-testid="stToolbar"] { visibility: hidden !important; display: none !important; }
        [data-testid="stDecoration"] { visibility: hidden !important; display: none !important; }
        [data-testid="stStatusWidget"] { visibility: hidden !important; display: none !important; }
        .stAppDeployButton { visibility: hidden !important; display: none !important; }
        [data-testid="manage-app-button"] { visibility: hidden !important; display: none !important; }

        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0px !important;
        }

        header[data-testid="stHeader"] [data-testid="stToolbarActions"],
        header[data-testid="stHeader"] .stAppDeployButton,
        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] a {
            display: none !important;
            visibility: hidden !important;
        }

        /* Ensure sidebar open/collapse toggle arrow remains visible */
        [data-testid="collapsedControl"],
        [data-testid="collapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] button {
            display: flex !important;
            visibility: visible !important;
        }

        /* Header styling */
        .hr-header-container {
            padding: 0.2rem 0 0.65rem 0;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid #E2E8F0;
        }

        .hr-header-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1E1B4B;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }

        .hr-header-badge {
            font-size: 0.725rem;
            font-weight: 600;
            background: #EEF2FF;
            color: #4F46E5;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            border: 1px solid #C7D2FE;
            vertical-align: middle;
        }

        .hr-header-subtitle {
            font-size: 0.85rem;
            color: #64748B;
            font-weight: 400;
            margin: 0;
        }

        /* Custom Cards */
        .hr-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease-in-out;
        }

        .hr-card:hover {
            border-color: #C7D2FE;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08);
        }

        /* Citation Card */
        .citation-card {
            background: #FAFAFD;
            border: 1px solid #E0E7FF;
            border-left: 4px solid #6366F1;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-top: 0.65rem;
            font-size: 0.875rem;
        }

        .citation-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            color: #4338CA;
            margin-bottom: 0.35rem;
        }

        .citation-meta {
            display: flex;
            gap: 0.85rem;
            color: #475569;
            font-size: 0.8rem;
        }

        .citation-badge {
            background: #EEF2FF;
            color: #3730A3;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            font-weight: 500;
        }

        .citation-snippet {
            margin-top: 0.5rem;
            padding-top: 0.5rem;
            border-top: 1px dashed #CBD5E1;
            color: #334155;
            font-style: italic;
        }

        /* Status card */
        .status-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            margin-bottom: 0.65rem;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-top: 5px;
            flex-shrink: 0;
        }

        .dot-operational { background-color: #10B981; }
        .dot-pending { background-color: #F59E0B; }
        .dot-info { background-color: #6366F1; }

        .status-title {
            font-weight: 600;
            color: #1E293B;
            font-size: 0.925rem;
            margin-bottom: 0.2rem;
        }

        .status-desc {
            font-size: 0.825rem;
            color: #64748B;
            line-height: 1.4;
        }

        /* Empty state */
        .empty-state-box {
            text-align: center;
            padding: 1.75rem 1.25rem;
            background: #F8FAFC;
            border: 1px dashed #CBD5E1;
            border-radius: 14px;
            margin: 0.75rem 0 1rem 0;
        }

        .empty-state-icon {
            font-size: 2rem;
            color: #6366F1;
            margin-bottom: 0.5rem;
        }

        .empty-state-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: #1E293B;
            margin-bottom: 0.35rem;
        }

        .empty-state-desc {
            font-size: 0.875rem;
            color: #64748B;
            max-width: 520px;
            margin: 0 auto 0.75rem auto;
            line-height: 1.45;
        }

        /* Modern button styling with word wrap */
        div[data-testid="stButton"] > button {
            border-radius: 9px !important;
            font-size: 0.84rem !important;
            font-weight: 500 !important;
            transition: all 0.15s ease-in-out !important;
            white-space: normal !important;
            word-break: break-word !important;
            text-align: left !important;
            line-height: 1.35 !important;
            padding: 0.55rem 0.85rem !important;
            min-height: 42px !important;
            border: 1px solid #E2E8F0 !important;
            background: #FFFFFF !important;
            color: #1E293B !important;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: #6366F1 !important;
            color: #4F46E5 !important;
            background: #F8FAFC !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.08) !important;
        }

        /* Streamlit chat override aesthetics */
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.85rem 1.15rem;
            margin-bottom: 0.65rem;
            border: 1px solid #F1F5F9;
            line-height: 1.55;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            background-color: #FFFFFF;
            border: 1px solid #EEF2FF;
            box-shadow: 0 1px 3px rgba(99, 102, 241, 0.04);
        }

        /* Sidebar compactness */
        [data-testid="stSidebar"] {
            background-color: #FAFAFC;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }

        /* ChatGPT / Claude style feedback toolbar */
        [data-testid="stFeedback"] {
            margin-top: 0.4rem !important;
            margin-bottom: 0.2rem !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 0.35rem !important;
        }

        [data-testid="stFeedback"] button {
            border: none !important;
            background: transparent !important;
            padding: 4px 6px !important;
            border-radius: 6px !important;
            color: #64748B !important;
            min-height: 28px !important;
            height: 28px !important;
            box-shadow: none !important;
            transition: all 0.15s ease-in-out !important;
        }

        [data-testid="stFeedback"] button:hover {
            background-color: #F1F5F9 !important;
            color: #1E293B !important;
            box-shadow: none !important;
            transform: none !important;
        }
    </style>
    """
    if sidebar_hide_rule:
        st.markdown(sidebar_hide_rule, unsafe_allow_html=True)
    st.markdown(base_css, unsafe_allow_html=True)


def render_header(
    title: str = "HR Policy Assistant",
    subtitle: str = "Ask questions about company HR policies",
    badge: str = "Phase 1: Foundation",
) -> None:
    """Render the application header with title, subtitle, and badge."""
    st.markdown(
        f"""
        <div class="hr-header-container">
            <div class="hr-header-title">
                <span>{title}</span>
                <span class="hr-header-badge">{badge}</span>
            </div>
            <p class="hr-header-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_citation(source: Any) -> Dict[str, Any]:
    """Format raw source or citation model into clean, user-facing dictionary.

    Excludes all internal chunk IDs, similarity scores, vector payloads, and filesystem paths.
    """
    if isinstance(source, dict):
        doc = source.get("document_name") or source.get("document") or source.get("source") or "HR Policy"
        # Clean doc name from filename if raw
        from app.rag.citations import format_document_display_name
        clean_doc = format_document_display_name(doc)
        return {
            "document_name": clean_doc,
            "policy_id": source.get("policy_id") or "HR-POLICY",
            "version": source.get("version") or "1.0",
            "section": source.get("section") or "General",
            "page": source.get("page"),
            "snippet": source.get("snippet"),
        }

    from app.rag.citations import format_document_display_name
    raw_doc = getattr(source, "document_name", None) or getattr(source, "document", None) or getattr(source, "source", "HR Policy")
    clean_doc = format_document_display_name(str(raw_doc))
    return {
        "document_name": clean_doc,
        "policy_id": getattr(source, "policy_id", None) or "HR-POLICY",
        "version": getattr(source, "version", None) or "1.0",
        "section": getattr(source, "section", None) or "General",
        "page": getattr(source, "page", None),
        "snippet": getattr(source, "snippet", None),
    }


def render_source_expander(citations: List[Any]) -> None:
    """Render a compact, expandable citation panel showing only user-friendly source facts."""
    if not citations:
        return

    clean_citations = [format_citation(c) for c in citations]
    count = len(clean_citations)

    with st.expander(f"Sources ({count})", expanded=False):
        for idx, c in enumerate(clean_citations, start=1):
            doc = c["document_name"]
            pid = c["policy_id"]
            ver = c["version"]
            sec = c["section"]
            p = c["page"]
            page_label = f"Page: {p}" if p is not None else "Page: Not specified"

            st.markdown(
                f"""
                <div style="
                    background: #F8FAFC;
                    border: 1px solid #E2E8F0;
                    border-left: 3px solid #6366F1;
                    border-radius: 8px;
                    padding: 0.75rem 1rem;
                    margin-bottom: 0.65rem;
                ">
                    <div style="font-weight: 600; font-size: 0.95rem; color: #1E1B4B; display: flex; align-items: center; gap: 0.4rem;">
                        <span>§</span> <span>{doc}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #4F46E5; font-weight: 600; margin: 0.2rem 0 0.4rem 0;">
                        <code>{pid}</code> • Version {ver}
                    </div>
                    <div style="font-size: 0.825rem; color: #475569; display: flex; flex-wrap: wrap; gap: 1.25rem;">
                        <span><strong>Section:</strong> {sec}</span>
                        <span><strong>{page_label}</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_citation_card(citation: Citation) -> None:
    """Render a source citation card formatted for document references."""
    page_text = f"<span class='citation-badge'>Page {citation.page}</span>" if citation.page is not None else ""
    section_text = (
        f"<span class='citation-badge'>Section: {citation.section}</span>" if citation.section else ""
    )
    snippet_html = (
        f"<div class='citation-snippet'>\"{citation.snippet}\"</div>" if citation.snippet else ""
    )

    st.markdown(
        f"""
        <div class="citation-card">
            <div class="citation-header">
                § Source: {citation.source}
            </div>
            <div class="citation-meta">
                {page_text}
                {section_text}
            </div>
            {snippet_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(
    title: str,
    status_type: str = "operational",
    description: str = "",
    tag: Optional[str] = None,
) -> None:
    """Render a system status card.

    status_type can be 'operational', 'pending', or 'info'.
    """
    dot_class = {
        "operational": "dot-operational",
        "pending": "dot-pending",
        "info": "dot-info",
    }.get(status_type, "dot-info")

    tag_html = f"<span class='hr-header-badge' style='float: right;'>{tag}</span>" if tag else ""

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-dot {dot_class}"></div>
            <div style="flex-grow: 1;">
                <div class="status-title">{title} {tag_html}</div>
                <div class="status-desc">{description}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(
    title: str = "Welcome to HR Policy Assistant",
    description: str = "Ask any question regarding company leave, health insurance, remote work, or employee benefits.",
    suggestions: Optional[List[str]] = None,
) -> None:
    """Render an empty state card with helpful suggested starter questions."""
    suggestions_html = ""
    if suggestions:
        chips = "".join(f"<span class='prompt-chip'>→ {s}</span>" for s in suggestions)
        suggestions_html = f"<div style='margin-top: 1rem;'>{chips}</div>"

    st.markdown(
        f"""
        <div class="empty-state-box">
            <div class="empty-state-icon">§</div>
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-desc">{description}</div>
            {suggestions_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_indicator(text: str = "Searching policy documents...") -> None:
    """Render a clean loading indicator placeholder."""
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; color: #6366F1; padding: 0.5rem 0;">
            <span style="animation: spin 1s linear infinite;">⟳</span>
            <span style="font-size: 0.9rem; font-weight: 500;">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
