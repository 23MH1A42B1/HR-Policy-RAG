"""Google OAuth / OIDC Authentication service for HR Policy RAG Assistant.

Handles OAuth2 code exchange, OIDC ID token verification, and user profile extraction.
Supports local simulated authentication for developer environments and automated test suites.
"""

from __future__ import annotations

import urllib.parse
from typing import Optional
import httpx

from app.auth.authorizer import Authorizer, get_authorizer
from app.auth.user import UserInfo
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("app.auth.google_auth")

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"


class GoogleAuthService:
    """Enterprise Google OAuth 2.0 and OIDC authentication provider."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        authorizer: Optional[Authorizer] = None,
    ) -> None:
        settings = get_settings()
        self.client_id = client_id or getattr(settings, "google_client_id", None)
        self.client_secret = client_secret or getattr(settings, "google_client_secret", None)
        self.redirect_uri = redirect_uri or getattr(settings, "google_redirect_uri", "http://localhost:8501")
        self.authorizer = authorizer or get_authorizer()

    def is_oauth_configured(self) -> bool:
        """Check if Google OAuth client ID and secret are configured."""
        return bool(self.client_id and self.client_secret)

    def get_authorization_url(self, state: str = "security_state_hr_rag") -> str:
        """Construct standard Google OAuth2 authorization redirect URL."""
        if not self.client_id:
            raise ValueError("Google Client ID is not configured.")

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "scope": "openid email profile",
            "redirect_uri": self.redirect_uri,
            "state": state,
            "access_type": "online",
            "prompt": "select_account",
        }
        query_string = urllib.parse.urlencode(params)
        return f"{GOOGLE_AUTH_ENDPOINT}?{query_string}"

    def exchange_code_for_token(self, code: str) -> Optional[UserInfo]:
        """Exchange authorization code for tokens and verify authenticated user."""
        if not self.is_oauth_configured():
            logger.warning("Attempted OAuth token exchange while client credentials are not configured.")
            return None

        payload = {
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(GOOGLE_TOKEN_ENDPOINT, data=payload)
                resp.raise_for_status()
                token_data = resp.json()

            id_token = token_data.get("id_token")
            if not id_token:
                logger.error("No id_token received in Google OAuth token exchange response.")
                return None

            return self.verify_id_token(id_token)

        except Exception as exc:
            logger.error(f"Google OAuth code exchange failed: {exc}")
            return None

    def verify_id_token(self, id_token_str: str) -> Optional[UserInfo]:
        """Verify Google OIDC ID token using google-auth library."""
        try:
            from google.oauth2 import id_token
            from google.auth.transport import requests as google_requests

            request = google_requests.Request()
            id_info = id_token.verify_oauth2_token(
                id_token_str,
                request,
                self.client_id,
            )

            email = id_info.get("email", "")
            name = id_info.get("name", email.split("@")[0].capitalize())
            picture = id_info.get("picture")
            user_id = id_info.get("sub", email)

            # Evaluate authorization rules
            auth_res = self.authorizer.check_authorization(email)
            if not auth_res.is_authorized:
                logger.warning(f"Authenticated user '{email}' rejected: not authorized.")
                raise PermissionError(auth_res.rejection_reason)

            return UserInfo(
                email=email,
                name=name,
                picture=picture,
                user_id=user_id,
                role=auth_res.role or "employee",
                is_admin=auth_res.is_admin,
            )

        except PermissionError:
            raise
        except Exception as exc:
            logger.error(f"Failed to verify Google ID token: {exc}")
            return None

    def authenticate_mock(
        self,
        email: str,
        name: Optional[str] = None,
        picture: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> UserInfo:
        """Simulate authentication for local developer environments or test automation.

        Enforces identical authorization rules as live Google OAuth authentication.
        """
        clean_email = email.strip().lower()
        display_name = name or clean_email.split("@")[0].replace(".", " ").title()
        uid = user_id or f"usr_{clean_email.split('@')[0]}"

        auth_res = self.authorizer.check_authorization(clean_email)
        if not auth_res.is_authorized:
            raise PermissionError(auth_res.rejection_reason)

        return UserInfo(
            email=clean_email,
            name=display_name,
            picture=picture or f"https://api.dicebear.com/7.x/initials/svg?seed={display_name}",
            user_id=uid,
            role=auth_res.role or "employee",
            is_admin=auth_res.is_admin,
        )


# Global singleton service
_GLOBAL_GOOGLE_AUTH: Optional[GoogleAuthService] = None


def get_google_auth_service() -> GoogleAuthService:
    """Return singleton GoogleAuthService instance."""
    global _GLOBAL_GOOGLE_AUTH
    if _GLOBAL_GOOGLE_AUTH is None:
        _GLOBAL_GOOGLE_AUTH = GoogleAuthService()
    return _GLOBAL_GOOGLE_AUTH


def is_authenticated() -> bool:
    """Check if current user is authenticated via Streamlit native OIDC, corporate email, or test harness."""
    try:
        import streamlit as st
        # 1. Corporate email or session authenticated user
        if "_authenticated_user" in st.session_state and st.session_state["_authenticated_user"]:
            return True
        # 2. Streamlit Native OIDC
        if hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
            return True
        # 3. Test harness session override for unit tests
        if "_test_authenticated_user" in st.session_state:
            return True
    except Exception:
        pass
    return False


def get_authenticated_user() -> Optional[UserInfo]:
    """Retrieve verified authenticated user and evaluate role-based authorization."""
    try:
        import streamlit as st
        authorizer = get_authorizer()

        # 1. Corporate email or manual session authenticated user
        if "_authenticated_user" in st.session_state and st.session_state["_authenticated_user"]:
            user_obj = st.session_state["_authenticated_user"]
            if isinstance(user_obj, UserInfo):
                return user_obj
            elif hasattr(user_obj, "email"):
                safe_user = UserInfo(
                    email=getattr(user_obj, "email"),
                    name=getattr(user_obj, "name", "Employee"),
                    user_id=getattr(user_obj, "user_id", "usr"),
                    role=getattr(user_obj, "role", "employee"),
                    is_admin=getattr(user_obj, "is_admin", False),
                )
                st.session_state["_authenticated_user"] = safe_user
                return safe_user

        # 2. Test harness override for automated tests
        if "_test_authenticated_user" in st.session_state:
            return st.session_state["_test_authenticated_user"]

        # 3. Streamlit Native OIDC
        if hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
            email = getattr(st.user, "email", "") or ""
            name = getattr(st.user, "name", "") or (email.split("@")[0].replace(".", " ").title() if email else "Employee")
            picture = getattr(st.user, "picture", None)
            sub = getattr(st.user, "sub", None) or getattr(st.user, "id", None) or (f"usr_{email.split('@')[0]}" if email else "usr_anon")

            auth_res = authorizer.check_authorization(email)
            return UserInfo(
                email=email,
                name=name,
                picture=picture,
                user_id=str(sub),
                role=auth_res.role or "employee",
                is_admin=auth_res.is_admin,
            )
    except Exception as exc:
        logger.warning(f"Error resolving authenticated user: {exc}")

    return None


def logout_user() -> None:
    """Sign out the current user, clear session state, and reset UI."""
    try:
        import streamlit as st
        st.session_state.clear()
        if hasattr(st, "logout"):
            st.logout()
        st.rerun()
    except Exception as exc:
        logger.warning(f"Error during logout: {exc}")
