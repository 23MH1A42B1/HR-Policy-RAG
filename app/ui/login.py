"""Enterprise Login and Authentication Gateway for XYZ Technologies HR Policy Assistant.

Enforces corporate authentication and authorization before any protected application logic
or sidebar components are rendered.
"""

from __future__ import annotations

import streamlit as st

from app.auth.authorizer import UNAUTHORIZED_MESSAGE, get_authorizer
from app.auth.google_auth import logout_user
from app.auth.user import UserInfo
from app.utils.logging import get_logger

logger = get_logger("app.ui.login")


def render_login_page() -> None:
    """Render the enterprise login screen with clean centered login card.

    Strictly renders only the login card and stops execution immediately.
    """
    col_l, col_center, col_r = st.columns([1, 1.8, 1])

    with col_center:
        st.markdown(
            """
            <div style="
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 2.5rem 2rem 2rem 2rem;
                margin-top: 3.5rem;
                box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
                text-align: center;
            ">
                <div style="
                    background: #4F46E5;
                    color: white;
                    width: 58px;
                    height: 58px;
                    border-radius: 14px;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.8rem;
                    font-weight: 800;
                    margin-bottom: 1.25rem;
                    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
                ">
                    HR
                </div>
                <hr style="border: none; border-top: 1px solid #E2E8F0; margin: 0 0 1.5rem 0;" />
                <h2 style="color: #1E1B4B; font-size: 1.6rem; font-weight: 700; margin: 0 0 0.35rem 0;">HR Policy Assistant</h2>
                <div style="color: #6366F1; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.85rem;">Enterprise HR Policy Portal</div>
                <p style="color: #64748B; font-size: 0.9rem; line-height: 1.5; margin-bottom: 1.75rem;">
                    Secure access to XYZ Technologies<br/>HR policies and employee information.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Option 1: Login with Google SSO
        login_clicked = st.button("Continue with Google", type="primary", use_container_width=True, icon="🌐")

        if login_clicked:
            try:
                st.login()
            except Exception as exc:
                logger.info(f"st.login() call exception (OIDC config status): {exc}")
                st.error("Google OIDC authentication is not configured in `.streamlit/secrets.toml`.", icon="⚙️")
                with st.expander("🛠️ How to configure Google OIDC authentication", expanded=True):
                    st.markdown(
                        """
                        To enable corporate Google login, create `.streamlit/secrets.toml`:
                        ```toml
                        [auth]
                        redirect_uri = "http://localhost:8501/oauth2callback"
                        cookie_secret = "your-secure-random-32-byte-secret"
                        client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
                        client_secret = "GOCSPX-YOUR_CLIENT_SECRET"
                        server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
                        ```
                        *Reference: See [secrets.toml.example](file:///c:/pdf/HR%20Policy%20Rag/.streamlit/secrets.toml.example)*
                        """
                    )

        # Visual Divider
        st.markdown(
            """
            <div style="display: flex; align-items: center; margin: 1.25rem 0 1rem 0; color: #94A3B8; font-size: 0.775rem;">
                <div style="flex: 1; border-top: 1px solid #E2E8F0;"></div>
                <span style="padding: 0 0.65rem; font-weight: 600; letter-spacing: 0.04em;">OR ENTER WORK EMAIL</span>
                <div style="flex: 1; border-top: 1px solid #E2E8F0;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Option 2: Direct Corporate Email Entry
        with st.form(key="work_email_login_form", border=False):
            work_email = st.text_input(
                "Corporate Email Address",
                placeholder="e.g. murali@xyz.com or nadipenamurali13@gmail.com",
                label_visibility="collapsed",
                key="login_work_email_field",
            )
            email_login_btn = st.form_submit_button("Continue with Email", use_container_width=True, icon="✉️")

            if email_login_btn:
                clean_email = work_email.strip().lower()
                if not clean_email or "@" not in clean_email or "." not in clean_email.split("@")[-1]:
                    st.error("Please enter a valid corporate email address.", icon="⚠️")
                else:
                    authorizer = get_authorizer()
                    auth_res = authorizer.check_authorization(clean_email)
                    if not auth_res.is_authorized:
                        st.error(
                            auth_res.rejection_reason or "Your account is not authorized to access XYZ Technologies HR Policy Assistant.",
                            icon="⛔",
                        )
                    else:
                        display_name = clean_email.split("@")[0].replace(".", " ").title()
                        st.session_state["_authenticated_user"] = UserInfo(
                            email=clean_email,
                            name=display_name,
                            user_id=f"usr_{clean_email.split('@')[0]}",
                            role=auth_res.role or "employee",
                            is_admin=auth_res.is_admin,
                        )
                        st.rerun()

        # Security footer
        st.markdown(
            """
            <div style="
                text-align: center;
                margin-top: 1.5rem;
                padding-top: 1.25rem;
                border-top: 1px solid #E2E8F0;
                color: #64748B;
                font-size: 0.825rem;
                line-height: 1.45;
            ">
                <span style="font-weight: 600; color: #334155;">🔒 Secure employee access</span><br/>
                Authorized employees only
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Immediately halt execution to protect all application content
    st.stop()


def render_unauthorized_page(user: UserInfo) -> None:
    """Render the access denied screen when user is authenticated but not authorized.

    Does not display chat, policy documents, or internal system details.
    """
    col_l, col_center, col_r = st.columns([1, 1.8, 1])

    with col_center:
        st.markdown(
            f"""
            <div style="
                background: #FFFFFF;
                border: 1px solid #FECACA;
                border-radius: 16px;
                padding: 2.5rem 2rem;
                margin-top: 4rem;
                box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.08);
                text-align: center;
            ">
                <div style="
                    background: #FEE2E2;
                    color: #DC2626;
                    width: 60px;
                    height: 60px;
                    border-radius: 14px;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 2rem;
                    margin-bottom: 1.25rem;
                ">
                    🚫
                </div>
                <h3 style="color: #991B1B; font-size: 1.35rem; font-weight: 700; margin-bottom: 0.5rem;">Access Denied</h3>
                <p style="color: #4B5563; font-size: 0.95rem; line-height: 1.55; margin-bottom: 1.25rem;">
                    {UNAUTHORIZED_MESSAGE}
                </p>
                <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 0.75rem; font-size: 0.85rem; color: #374151; margin-bottom: 1.5rem;">
                    Authenticated Account: <strong>{user.email}</strong>
                </div>
                <div style="font-size: 0.8rem; color: #6B7280; line-height: 1.4;">
                    If you believe this is an error, please contact your HR Administrator or People Operations to request portal access.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", use_container_width=True, type="secondary"):
            logout_user()

    # Halt execution to protect all application content
    st.stop()


# Backward compatibility alias
render_login_screen = render_login_page
