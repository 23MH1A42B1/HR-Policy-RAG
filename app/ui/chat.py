"""Chat interface component for XYZ Technologies HR Policy RAG Assistant.

Manages conversational session state, follow-up context, clean answer rendering,
compact source citations expander, and admin-only developer retrieval diagnostics.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
import streamlit as st

from app.auth.user import UserInfo
from app.ui.components import format_citation, render_empty_state, render_source_expander
from app.ui.feedback import generate_followup_suggestions, record_feedback
from app.rag.multilingual import STARTER_QUESTIONS, WELCOME_MESSAGES, detect_language
from app.utils.logging import get_logger

logger = get_logger("app.ui.chat")

# Clean initial greeting with starter follow-ups
INITIAL_ASSISTANT_MESSAGE: Dict[str, Any] = {
    "id": "init_welcome",
    "role": "assistant",
    "content": "Hello! I'm your HR Policy Assistant. Ask me anything about company HR policies.",
    "citations": [],
    "suggestions": [
        "How many annual leave days do employees get?",
        "Can I work from home under the remote work policy?",
        "What are the core collaboration hours?",
    ],
}


def initialize_chat_state() -> None:
    """Initialize session state keys for messages and conversation metadata."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [dict(INITIAL_ASSISTANT_MESSAGE)]

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())


def get_conversation_history() -> List[Dict[str, Any]]:
    """Return the current conversation message history from session state."""
    return st.session_state.get("messages", [])


def handle_user_query(query: str, language: str = "English") -> Dict[str, Any]:
    """Execute shared RAG Service to evaluate scope, cache, and grounded retrieval."""
    logger.info(f"Processing chat query (lang={language}): '{query[:60]}...'")
    session_id = st.session_state.get("session_id", "web_default_session")
    auth_user = st.session_state.get("authenticated_user")
    user_id = auth_user.email if auth_user and hasattr(auth_user, "email") else "web_user"

    try:
        from app.api.rag_service import answer_question

        result = answer_question(
            query=query,
            user_id=user_id,
            session_id=session_id,
            source="web",
            language=language,
        )

        formatted_citations = result.get("sources", [])
        meta = result.get("metadata", {})

        # Dynamically generate intelligent follow-up suggestions
        suggestions = generate_followup_suggestions(
            query=query,
            answer=result.get("answer", ""),
            citations=formatted_citations,
        )

        return {
            "id": str(uuid.uuid4())[:8],
            "role": "assistant",
            "content": result.get("answer", ""),
            "citations": formatted_citations,
            "suggestions": suggestions,
            "retrieval_results": meta.get("retrieval_results", []),
            "model_used": meta.get("model", "none"),
            "cache_hit": meta.get("cache_hit", False),
            "reranker_used": meta.get("reranker_used", "none"),
        }
    except Exception as exc:
        logger.error(f"Error executing HR Agent via RAG service in chat UI: {exc}", exc_info=True)
        return {
            "id": str(uuid.uuid4())[:8],
            "role": "assistant",
            "content": "HR policy search is temporarily unavailable. Please try again later.",
            "citations": [],
            "suggestions": [
                "How many annual leave days do employees get?",
                "Can I work from home under the remote work policy?",
            ],
            "retrieval_results": [],
            "model_used": "none",
            "cache_hit": False,
            "reranker_used": "none",
        }


def render_chat_view(user: Optional[UserInfo] = None) -> None:
    """Render the chat interface with message stream, citations, follow-up chips, and input box."""
    initialize_chat_state()
    messages: List[Dict[str, Any]] = st.session_state["messages"]
    is_admin = bool(user and user.is_admin)

    # Check for queued prompt from interactive chip clicks
    pending_prompt = st.session_state.pop("pending_prompt", None)

    # Multilingual selection toolbar
    c_sub, c_lang = st.columns([3, 1], vertical_alignment="center")
    with c_sub:
        st.markdown(
            "<div style='font-size: 0.825rem; color: #64748B; font-weight: 500; margin-bottom: 0.4rem;'>"
            "Instant answers from official XYZ Technologies HR policy documents."
            "</div>",
            unsafe_allow_html=True,
        )
    with c_lang:
        lang_choice = st.selectbox(
            "Language",
            options=["English", "Hindi (हिंदी)", "Telugu (తెలుగు)", "Tamil (தமிழ்)"],
            index=st.session_state.get("lang_index", 0),
            key="lang_select_box",
            label_visibility="collapsed",
            help="Select response language for policy answers.",
        )

    # Map selected choice to language code
    if "hindi" in lang_choice.lower():
        current_lang = "Hindi"
        st.session_state["lang_index"] = 1
    elif "telugu" in lang_choice.lower():
        current_lang = "Telugu"
        st.session_state["lang_index"] = 2
    elif "tamil" in lang_choice.lower():
        current_lang = "Tamil"
        st.session_state["lang_index"] = 3
    else:
        current_lang = "English"
        st.session_state["lang_index"] = 0

    # Identify the latest assistant message to display contextual follow-up chips
    latest_assistant_idx = max(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        default=-1,
    )

    # Display chat message history
    for idx, msg in enumerate(messages):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        citations: List[Dict[str, Any]] = msg.get("citations", [])
        retrieval_results: List[Dict[str, Any]] = msg.get("retrieval_results", [])
        suggestions: List[str] = msg.get("suggestions", [])
        msg_id = msg.get("id", f"msg_{idx}")

        with st.chat_message(role):
            if msg_id == "init_welcome":
                welcome_text = WELCOME_MESSAGES.get(current_lang, content)
                st.markdown(welcome_text)

                # Show starter questions in a neat 2-column grid when starting a conversation
                if len(messages) <= 1:
                    st.markdown(
                        f"<div style='font-size: 0.825rem; font-weight: 600; color: #4F46E5; margin-top: 0.65rem; margin-bottom: 0.45rem; display: flex; align-items: center; gap: 0.35rem;'>"
                        f"<span>✦</span> <span>Popular Questions ({current_lang}):</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    starter_qs = STARTER_QUESTIONS.get(current_lang, STARTER_QUESTIONS["English"])
                    sq_col1, sq_col2 = st.columns(2)
                    for s_idx, sq in enumerate(starter_qs):
                        with (sq_col1 if s_idx % 2 == 0 else sq_col2):
                            if st.button(f"→ {sq}", key=f"start_q_{current_lang}_{s_idx}", use_container_width=True):
                                st.session_state["pending_prompt"] = sq
                                st.rerun()
            else:
                st.markdown(content)

            # 1. Clean, user-friendly source citations expander
            if citations:
                render_source_expander(citations)

            # 2. ChatGPT / Claude Style Feedback (Outline Thumbs Up & Thumbs Down)
            if role == "assistant" and msg_id != "init_welcome":
                user_email = user.email if user else "employee@company.com"
                prev_query = messages[idx - 1]["content"] if idx > 0 and messages[idx - 1].get("role") == "user" else ""
                fb_key = f"feedback_{msg_id}"
                rec_key = f"recorded_{msg_id}"

                fb_score = st.feedback("thumbs", key=fb_key)
                if fb_score is not None and not st.session_state.get(rec_key):
                    rating_str = "helpful" if fb_score == 1 else "unhelpful"
                    record_feedback(
                        message_id=msg_id,
                        query=prev_query,
                        answer=content,
                        rating=rating_str,
                        user_email=user_email,
                    )
                    st.session_state[rec_key] = True
                    toast_msg = "Thanks for your feedback!" if fb_score == 1 else "Feedback recorded. We will improve!"
                    st.toast(toast_msg)

            # 3. Interactive Suggested Follow-up Question Chips (on latest assistant response)
            if role == "assistant" and idx == latest_assistant_idx and msg_id != "init_welcome" and suggestions:
                st.markdown(
                    "<div style='margin-top: 0.65rem; margin-bottom: 0.4rem; font-size: 0.825rem; font-weight: 600; color: #4F46E5; display: flex; align-items: center; gap: 0.35rem;'><span>✦</span> <span>Suggested Follow-ups:</span></div>",
                    unsafe_allow_html=True,
                )
                num_cols = min(len(suggestions), 2)
                chip_cols = st.columns(num_cols)
                for c_i, chip in enumerate(suggestions):
                    with chip_cols[c_i % num_cols]:
                        if st.button(f"→ {chip}", key=f"chip_{msg_id}_{c_i}", use_container_width=True):
                            st.session_state["pending_prompt"] = chip
                            st.rerun()

            # 3. Developer Retrieval Details: STRICTLY ADMIN-ONLY (Requirement 15)
            if is_admin and retrieval_results:
                with st.expander("Diagnostics: Developer Retrieval Details", expanded=False):
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("LLM Route", msg.get("model_used", "none"))
                    col_m2.metric("Reranker", msg.get("reranker_used", "none"))
                    col_m3.metric("Cache Hit", "Yes" if msg.get("cache_hit") else "No")

                    st.markdown("##### Supporting Chunks & Relevance")
                    for r_idx, r in enumerate(retrieval_results, start=1):
                        score = r.get("score", 0.0)
                        doc_name = r.get("document", "Unknown")
                        section = r.get("section", "General")
                        page = r.get("page", 1)
                        chunk_id = r.get("chunk_id", "N/A")

                        st.markdown(
                            f"""
                            **Match #{r_idx}** — Document: `{doc_name}` | Section: `{section}` | Page: `{page}`  
                            • Score: `{score:.4f}` | Chunk ID: `{chunk_id}`  
                            ```
                            {r.get('text', '')[:160]}...
                            ```
                            """
                        )

    # Chat input box
    user_prompt = st.chat_input("Ask a question about HR policies...")
    active_prompt = pending_prompt or user_prompt

    if active_prompt:
        clean_prompt = active_prompt.strip()
        if clean_prompt:
            # 1. Append and render user message
            st.session_state["messages"].append({"role": "user", "content": clean_prompt})
            with st.chat_message("user"):
                st.markdown(clean_prompt)

            # 2. Generate and append assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching policies..."):
                    response_data = handle_user_query(clean_prompt, language=current_lang)
                    st.markdown(response_data["content"])

                    if response_data.get("citations"):
                        render_source_expander(response_data["citations"])

                    if is_admin and response_data.get("retrieval_results"):
                        with st.expander("🛠️ Developer Retrieval Details", expanded=False):
                            col_m1, col_m2 = st.columns(2)
                            col_m1.metric("LLM Route", response_data.get("model_used", "none"))
                            col_m2.metric("Cache Hit", "Yes" if response_data.get("cache_hit") else "No")

            st.session_state["messages"].append(response_data)
            st.rerun()
