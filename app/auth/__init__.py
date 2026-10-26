"""Authentication and authorization package for HR Policy RAG Assistant."""

from app.auth.authorizer import (
    UNAUTHORIZED_MESSAGE,
    AuthorizationResult,
    Authorizer,
    get_authorizer,
)
from app.auth.google_auth import (
    GoogleAuthService,
    get_authenticated_user,
    get_google_auth_service,
    is_authenticated,
    logout_user,
)
from app.auth.user import UserInfo, UserRole

__all__ = [
    "UserInfo",
    "UserRole",
    "Authorizer",
    "AuthorizationResult",
    "UNAUTHORIZED_MESSAGE",
    "get_authorizer",
    "GoogleAuthService",
    "get_google_auth_service",
    "get_authenticated_user",
    "is_authenticated",
    "logout_user",
]
