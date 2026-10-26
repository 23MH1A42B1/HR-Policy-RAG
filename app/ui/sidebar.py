"""Sidebar component for authenticated users of XYZ Technologies HR Policy Assistant.

Rendered strictly AFTER authentication and authorization have succeeded.
Provides corporate profile display, role-restricted navigation, conversation reset, and sign-out.
"""

from __future__ import annotations

from typing import Literal
import uuid
import streamlit as st

from app.auth.google_auth import logout_user
from app.auth.user import UserInfo

NavPage = Literal["Chat", "Policy Documents", "System Status", "Admin Console"]


def render_sidebar(user: UserInfo) -> NavPage:
    """Render the authenticated sidebar with profile, navigation, conversation, and account controls."""
    with st.sidebar:
        # 1. Company Brand Header
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.25rem;">
                <div style="
                    background: #4F46E5;
                    color: white;
                    width: 40px;
                    height: 40px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.25rem;
                    font-weight: 800;
                    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
                ">
                    HR
                </div>
                <div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #6366F1; letter-spacing: 0.05em; text-transform: uppercase;">XYZ Technologies</div>
                    <div style="font-weight: 700; font-size: 1.05rem; color: #1E1B4B; line-height: 1.2;">HR Policy Assistant</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Enterprise HR Portal</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="border-top: 1px solid #E2E8F0; margin: 0.65rem 0;"></div>', unsafe_allow_html=True)

        # 2. User Profile Card (Never "Guest User")
        role_badge = (
            '<span style="background: #FEF3C7; color: #92400E; font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 9999px;">Admin</span>'
            if user.is_admin
            else '<span style="background: #E0E7FF; color: #3730A3; font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 9999px;">Employee</span>'
        )

        initials = "".join(part[0].upper() for part in user.name.split()[:2]) or "HR"
        avatar_html = (
            f'<img src="{user.picture}" style="width: 34px; height: 34px; border-radius: 50%; object-fit: cover;" />'
            if user.picture
            else f'<div style="width: 34px; height: 34px; border-radius: 50%; background: #4F46E5; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 700;">{initials}</div>'
        )

        st.markdown(
            f"""
            <div style="
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
            ">
                <div style="display: flex; align-items: center; gap: 0.65rem;">
                    {avatar_html}
                    <div style="overflow: hidden; flex: 1;">
                        <div style="font-weight: 600; font-size: 0.875rem; color: #0F172A; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">
                            {user.name}
                        </div>
                        <div style="font-size: 0.75rem; color: #64748B; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">
                            {user.email}
                        </div>
                    </div>
                </div>
                <div style="margin-top: 0.45rem; display: flex; justify-content: space-between; align-items: center;">
                    {role_badge}
                    <span style="font-size: 0.68rem; color: #10B981; font-weight: 600;">● Active</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. Navigation (Role-aware)
        st.caption("NAVIGATION")

        nav_options = ["• Chat", "• Policy Documents"]
        if user.is_admin:
            nav_options.append("• System Status")
            nav_options.append("• Admin Console")

        selected_label = st.radio(
            "Navigation Menu",
            options=nav_options,
            index=0,
            label_visibility="collapsed",
        )

        # Normalize page selection
        if "Chat" in selected_label:
            selected_page: NavPage = "Chat"
        elif "Policy Documents" in selected_label:
            selected_page: NavPage = "Policy Documents"
        elif "System Status" in selected_label:
            selected_page: NavPage = "System Status"
        else:
            selected_page: NavPage = "Admin Console"

        st.markdown('<div style="border-top: 1px solid #E2E8F0; margin: 0.65rem 0;"></div>', unsafe_allow_html=True)

        # 4. Conversation Controls
        st.caption("CONVERSATION")
        if st.button("+ New Conversation", use_container_width=True, type="secondary"):
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "Hello! I'm your HR Policy Assistant. Ask me anything about company HR policies.",
                    "citations": [],
                }
            ]
            st.session_state["session_id"] = str(uuid.uuid4())
            st.rerun()

        st.markdown('<div style="border-top: 1px solid #E2E8F0; margin: 0.65rem 0;"></div>', unsafe_allow_html=True)

        # 5. Account & Sign Out
        st.caption("ACCOUNT")
        if st.button("→ Sign Out", use_container_width=True, type="secondary"):
            logout_user()

        # Security footer
        st.markdown(
            """
            <div style="
                background: #EEF2FF;
                border: 1px solid #C7D2FE;
                border-radius: 8px;
                padding: 0.65rem 0.75rem;
                margin-top: 0.85rem;
                font-size: 0.75rem;
                color: #3730A3;
                line-height: 1.4;
            ">
                <strong>● Enterprise Security</strong><br/>
                • Model Armor Input Guard<br/>
                • Prompt Injection Defense<br/>
                • Grounded Source Citations<br/>
                • Role-Based Access Control
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page
