"""User authentication data models for HR Policy RAG Assistant.

Defines authenticated identity models and role representations.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

UserRole = Literal["employee", "admin"]


class UserInfo(BaseModel):
    """Authenticated user model with Google OAuth attributes and role authorization."""

    email: str = Field(description="User primary email address")
    name: str = Field(description="User display name")
    picture: Optional[str] = Field(default=None, description="Profile picture URL")
    user_id: str = Field(description="Unique user identifier (sub claim or system ID)")
    role: UserRole = Field(default="employee", description="Assigned role: 'employee' or 'admin'")
    is_admin: bool = Field(default=False, description="Flag indicating administrator permissions")

    def model_post_init(self, __context) -> None:
        """Synchronize is_admin attribute with role."""
        if self.role == "admin":
            self.is_admin = True
