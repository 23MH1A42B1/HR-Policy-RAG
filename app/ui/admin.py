"""Admin Console view for HR Policy document management and GCS ingestion.

Allows authorized administrators to upload, re-index, list, and delete policy documents.
Enforces strict RBAC to prevent unauthorized employees from accessing management controls.
"""

import datetime
import json
from typing import Optional
import streamlit as st

from app.auth.user import UserInfo
from app.ingestion.admin_pipeline import get_admin_pipeline
from app.storage.gcs_storage import VALID_POLICY_CATEGORIES
from app.ui.feedback import get_feedback_metrics, get_topic_analytics
from app.utils.logging import get_logger

logger = get_logger("app.ui.admin")


def render_admin_view(user: Optional[UserInfo] = None) -> None:
    """Render the Enterprise Admin Policy Management Console."""
    if user is None:
        from app.auth.google_auth import get_authenticated_user
        user = get_authenticated_user() or st.session_state.get("authenticated_user")

    # Strict RBAC enforcement
    if not user or not user.is_admin:
        st.error(
            "🔒 **Access Denied**: You do not have administrator permissions to access this console. "
            "Only authorized HR administrators may manage company policy documents.",
            icon="🚫",
        )
        return

    st.subheader("🛠️ HR Policy Administration Console")
    st.caption("Manage production policy documents in Google Cloud Storage and maintain search indices.")

    pipeline = get_admin_pipeline()
    status = pipeline.get_ingestion_status()

    # Top summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GCS Documents", status.get("total_documents_gcs", 0))
    col2.metric("BM25 Chunks", status.get("bm25_indexed_chunks", 0))
    col3.metric("Storage Bucket", f"gs://{status.get('bucket_name', 'company-hr-policies')}")
    col4.metric("Storage Status", status.get("storage_status", "local_storage").split()[0])

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    tab_analytics, tab_upload, tab_manage, tab_status = st.tabs([
        "📈 Query Analytics & Feedback",
        "📤 Upload & Index Policy",
        "📂 Policy Repository (GCS)",
        "📊 Ingestion Diagnostics",
    ])

    # ==============================================================
    # TAB 1: Admin Insights & Query Analytics Dashboard
    # ==============================================================
    with tab_analytics:
        st.markdown("#### 📈 Employee Inquiry & Feedback Analytics")
        st.caption("Real-time visibility into HR policy search trends, employee satisfaction, and feedback ratings.")

        fb_metrics = get_feedback_metrics()
        topic_data = get_topic_analytics()

        # Top Performance Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Satisfaction Rate", f"{fb_metrics['satisfaction_rate']}%")
        m_col2.metric("Total Rated Answers", fb_metrics["total_feedback"])
        m_col3.metric("Helpful Votes (👍)", fb_metrics["helpful_count"])
        m_col4.metric("Unhelpful Votes (👎)", fb_metrics["unhelpful_count"])

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        # Policy Topics Breakdown & Sentiment
        top_col1, top_col2 = st.columns([1.6, 1])
        with top_col1:
            st.markdown("##### 📊 Inquiries by Policy Category")
            total_topics = sum(topic_data.values())
            if total_topics > 0:
                for cat_name, cat_count in topic_data.items():
                    pct = int((cat_count / total_topics) * 100) if total_topics > 0 else 0
                    st.write(f"**{cat_name}** — `{cat_count}` queries ({pct}%)")
                    st.progress(pct / 100.0)
            else:
                st.info("No policy inquiries have been logged yet. Employee queries will appear here.", icon="ℹ️")

        with top_col2:
            st.markdown("##### 💡 Quality & Resolution Rate")
            sat_rate = fb_metrics["satisfaction_rate"]
            st.markdown(
                f"""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.25rem;">
                    <div style="font-size: 0.85rem; color: #64748B; font-weight: 600;">EMPLOYEE SENTIMENT</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: {'#059669' if sat_rate >= 80 else '#E11D48'};">
                        {sat_rate}%
                    </div>
                    <div style="font-size: 0.825rem; color: #475569; margin-top: 0.25rem;">
                        {'High employee satisfaction across company policies.' if sat_rate >= 80 else 'Attention recommended: review unhelpful logs.'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='margin: 1.5rem 0; border: 0; border-top: 1px solid #E2E8F0;' />", unsafe_allow_html=True)

        # Recent Query Feedback Logs Table
        st.markdown("##### 📋 Recent Employee Inquiries & Ratings")
        recent_items = fb_metrics.get("recent_entries", [])
        if not recent_items:
            st.info("No employee query feedback logs recorded yet.", icon="ℹ️")
        else:
            table_rows = []
            for item in recent_items:
                ts = item.get("timestamp", 0)
                time_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "Recent"
                rating_badge = "👍 Helpful" if item.get("rating") == "helpful" else "👎 Unhelpful"
                table_rows.append({
                    "Timestamp": time_str,
                    "Employee": item.get("user_email", "anonymous"),
                    "Rating": rating_badge,
                    "Question": item.get("query", ""),
                    "Answer Snippet": item.get("answer_preview", "")[:120] + "...",
                })

            import pandas as pd
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Export Button
            export_json = json.dumps(recent_items, indent=2)
            st.download_button(
                label="📥 Export Query Feedback Logs (JSON)",
                data=export_json,
                file_name="hr_policy_query_feedback.json",
                mime="application/json",
            )

    # ==============================================================
    # TAB 2: Upload & Index Policy
    # ==============================================================
    with tab_upload:
        st.markdown("#### 📤 Automated Policy Ingestion (Drag-and-Drop)")
        st.write(
            "Drag and drop official HR policy documents (PDF, DOCX, PPTX). "
            "Documents will be automatically parsed, chunked, embedded with FastEmbed, and indexed into Qdrant & BM25."
        )

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            category_options = ["auto-detect"] + list(VALID_POLICY_CATEGORIES)
            selected_category_mode = st.selectbox(
                "Policy Category",
                options=category_options,
                format_func=lambda x: "🤖 Auto-Detect from Filename" if x == "auto-detect" else f"📁 policies/{x}/",
            )
        with c_col2:
            policy_version = st.text_input("Policy Version Tag", value="2026")

        uploaded_files = st.file_uploader(
            "Drag & Drop HR Policy Files Here (PDF, DOCX, PPTX)",
            type=["pdf", "docx", "pptx"],
            accept_multiple_files=True,
            help="Drag one or multiple policy files into this drop zone to ingest them simultaneously.",
        )

        def infer_category(filename: str) -> str:
            lower_name = filename.lower()
            if any(k in lower_name for k in ["leave", "pto", "vacation"]):
                return "leave"
            elif any(k in lower_name for k in ["remote", "wfh", "hybrid", "workplace"]):
                return "workplace"
            elif any(k in lower_name for k in ["travel", "expense", "reimbursement"]):
                return "expenses"
            elif any(k in lower_name for k in ["conduct", "ethics", "discipline"]):
                return "conduct"
            elif any(k in lower_name for k in ["benefit", "insurance", "health", "learning"]):
                return "benefits"
            elif any(k in lower_name for k in ["attendance", "hours"]):
                return "attendance"
            elif any(k in lower_name for k in ["security", "it", "password"]):
                return "security"
            return "general"

        if uploaded_files:
            st.markdown(f"##### 📄 Queued Documents ({len(uploaded_files)} file{'s' if len(uploaded_files) > 1 else ''})")
            for uf in uploaded_files:
                eff_cat = infer_category(uf.name) if selected_category_mode == "auto-detect" else selected_category_mode
                st.markdown(
                    f"• **{uf.name}** (`{uf.size / 1024:.1f} KB`) → Assigned category: `policies/{eff_cat}/`"
                )

            if st.button("⚡ Ingest & Index All Documents", type="primary", use_container_width=True):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                total_files = len(uploaded_files)
                total_extracted_pages = 0
                total_indexed_chunks = 0
                success_list = []
                error_list = []

                for idx, uf in enumerate(uploaded_files):
                    status_text.info(f"Ingesting ({idx + 1}/{total_files}): `{uf.name}`...")
                    effective_cat = infer_category(uf.name) if selected_category_mode == "auto-detect" else selected_category_mode
                    try:
                        file_bytes = uf.read()
                        res = pipeline.upload_and_index_policy(
                            file_bytes=file_bytes,
                            filename=uf.name,
                            category=effective_cat,
                            version=policy_version,
                        )
                        total_extracted_pages += res.get("pages_extracted", 0)
                        total_indexed_chunks += res.get("chunks_indexed", 0)
                        success_list.append(uf.name)
                    except Exception as exc:
                        logger.error(f"Failed to ingest '{uf.name}': {exc}")
                        error_list.append((uf.name, str(exc)))

                    progress_bar.progress((idx + 1) / total_files)

                status_text.empty()

                if success_list:
                    st.success(
                        f"🎉 **Successfully Ingested {len(success_list)} Policy Document(s)!**\n\n"
                        f"• **{total_extracted_pages}** total page(s) extracted\n"
                        f"• **{total_indexed_chunks}** chunks embedded and indexed into Qdrant & BM25\n"
                        f"• Vector database and search index refreshed. Documents are now immediately searchable in chat!",
                        icon="✅",
                    )
                if error_list:
                    for err_name, err_msg in error_list:
                        st.error(f"Failed to process '{err_name}': {err_msg}", icon="⚠️")

    # ==============================================================
    # TAB 2: Policy Repository & Management
    # ==============================================================
    with tab_manage:
        st.markdown("#### Stored Policy Documents in Google Cloud Storage")
        policies = pipeline.list_policies()

        if not policies:
            st.info("No policy documents found in GCS storage. Upload a policy above to begin.", icon="ℹ️")
        else:
            for p in policies:
                with st.expander(f"📄 **{p.filename}** ({p.category.capitalize()}) — {p.size_bytes / 1024:.1f} KB", expanded=False):
                    col_p1, col_p2 = st.columns([3, 1])
                    with col_p1:
                        st.markdown(
                            f"""
                            **Path:** `{p.path}`<br/>
                            **GCS URI:** `{p.gcs_uri}`<br/>
                            **Last Updated:** `{p.updated_at}`
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_p2:
                        reindex_key = f"reindex_{p.path}"
                        delete_key = f"delete_{p.path}"

                        if st.button("🔄 Re-index", key=reindex_key, use_container_width=True):
                            with st.spinner("Re-indexing..."):
                                try:
                                    re_res = pipeline.reindex_policy(p.path)
                                    st.success(f"Re-indexed {re_res['chunks_indexed']} chunks!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Re-indexing failed: {e}")

                        if st.button("🗑️ Delete", key=delete_key, type="secondary", use_container_width=True):
                            with st.spinner("Deleting..."):
                                try:
                                    del_res = pipeline.delete_policy(p.path)
                                    st.warning(f"Deleted '{del_res['document']}' from GCS & search indices.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Deletion failed: {e}")

    # ==============================================================
    # TAB 3: Diagnostics & Status
    # ==============================================================
    with tab_status:
        st.markdown("#### Architecture Diagnostics")
        st.json(status)
