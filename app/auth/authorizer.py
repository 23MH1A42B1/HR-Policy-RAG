"""Authorization and Role-Based Access Control (RBAC) service.

Enforces corporate email domain allow-lists, employee allow-lists, and admin privileges.
"""

from __future__ import annotations

from typing import List, Optional, Set
from pydantic import BaseModel, Field

from app.auth.user import UserInfo, UserRole
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("app.auth.authorizer")

UNAUTHORIZED_MESSAGE = "Your account is authenticated, but you are not authorized to access XYZ Technologies HR Policy Assistant."


class AuthorizationResult(BaseModel):
    """Result of authorization evaluation for a given email/user."""

    is_authorized: bool = Field(description="Whether the user is permitted access")
    role: Optional[UserRole] = Field(default=None, description="Assigned role if authorized")
    is_admin: bool = Field(default=False, description="Whether user holds administrative permissions")
    rejection_reason: Optional[str] = Field(default=None, description="Error reason if unauthorized")


class Authorizer:
    """Enforces enterprise employee and administrator authorization rules."""

    def __init__(
        self,
        allowed_domains: Optional[List[str]] = None,
        allowed_users: Optional[List[str]] = None,
        admin_users: Optional[List[str]] = None,
    ) -> None:
        settings = get_settings()

        # Parse allowed email domains
        if allowed_domains is not None:
            self.allowed_domains: Set[str] = {d.strip().lower() for d in allowed_domains if d.strip()}
        else:
            raw_domains = getattr(settings, "allowed_email_domains", "")
            self.allowed_domains = {
                d.strip().lower().lstrip("@") for d in raw_domains.split(",") if d.strip()
            }

        # Parse allowed individual users
        if allowed_users is not None:
            self.allowed_users: Set[str] = {u.strip().lower() for u in allowed_users if u.strip()}
        else:
            raw_users = getattr(settings, "allowed_users", "")
            self.allowed_users = {
                u.strip().lower() for u in raw_users.split(",") if u.strip()
            }

        # Parse administrator users
        if admin_users is not None:
            self.admin_users: Set[str] = {a.strip().lower() for a in admin_users if a.strip()}
        else:
            raw_admins = getattr(settings, "admin_users", "")
            self.admin_users = {
                a.strip().lower() for a in raw_admins.split(",") if a.strip()
            }

    def check_authorization(self, email: str) -> AuthorizationResult:
        """Evaluate if an email is authorized to access the application and determine its role."""
        if not email or "@" not in email:
            return AuthorizationResult(
                is_authorized=False,
                rejection_reason=UNAUTHORIZED_MESSAGE,
            )

        clean_email = email.strip().lower()
        domain = clean_email.split("@")[-1]

        # 1. Admin Allow-List Check
        if clean_email in self.admin_users:
            logger.info(f"User '{clean_email}' authorized with ADMIN role")
            return AuthorizationResult(
                is_authorized=True,
                role="admin",
                is_admin=True,
            )

        # 2. Individual Employee Allow-List Check
        if clean_email in self.allowed_users:
            logger.info(f"User '{clean_email}' authorized with EMPLOYEE role via user allow-list")
            return AuthorizationResult(
                is_authorized=True,
                role="employee",
                is_admin=False,
            )

        # 3. Domain Allow-List Check
        if domain in self.allowed_domains:
            logger.info(f"User '{clean_email}' authorized with EMPLOYEE role via domain '{domain}'")
            return AuthorizationResult(
                is_authorized=True,
                role="employee",
                is_admin=False,
            )

        # 4. Unauthorized rejection
        logger.warning(f"Unauthorized access attempt for email: '{clean_email}' (domain: '{domain}')")
        return AuthorizationResult(
            is_authorized=False,
            rejection_reason=UNAUTHORIZED_MESSAGE,
        )

    def can_perform_admin_action(self, user: UserInfo) -> bool:
        """Verify whether an authenticated user is permitted to perform administrative operations."""
        if not user:
            return False
        return bool(user.is_admin or user.role == "admin")

    def authorize_user(self, user: UserInfo) -> UserInfo:
        """Verify user authorization and return updated UserInfo with evaluated role."""
        res = self.check_authorization(user.email)
        if not res.is_authorized:
            raise PermissionError(res.rejection_reason or UNAUTHORIZED_MESSAGE)

        return user.model_copy(update={
            "role": res.role or "employee",
            "is_admin": res.is_admin,
        })


# Global singleton authorizer
_GLOBAL_AUTHORIZER: Optional[Authorizer] = None


def get_authorizer() -> Authorizer:
    """Return singleton Authorizer instance."""
    global _GLOBAL_AUTHORIZER
    if _GLOBAL_AUTHORIZER is None:
        _GLOBAL_AUTHORIZER = Authorizer()
    return _GLOBAL_AUTHORIZER
