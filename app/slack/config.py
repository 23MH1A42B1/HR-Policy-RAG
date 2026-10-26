"""Configuration management for XYZ Technologies Slack integration.

Safely loads Slack bot credentials, authorization allow-lists, and rate-limiting limits
from environment variables without hardcoding secrets.
"""

from __future__ import annotations

import os
from typing import List, Optional
from pydantic import BaseModel, Field


class SlackSettings(BaseModel):
    """Slack configuration model."""

    bot_token: Optional[str] = Field(default=None, description="Slack Bot User OAuth Token (xoxb-...)")
    app_token: Optional[str] = Field(default=None, description="Slack App-Level Token for Socket Mode (xapp-...)")
    allowed_users: List[str] = Field(default_factory=list, description="Allowed Slack User IDs (e.g. U12345)")
    allowed_email_domains: List[str] = Field(default_factory=list, description="Allowed employee email domains (e.g. xyz.com)")
    admin_users: List[str] = Field(default_factory=list, description="Slack Admin User IDs")
    rate_limit_requests: int = Field(default=10, description="Max allowed requests per user within time window")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")

    @property
    def is_configured(self) -> bool:
        """Return True if both Bot Token and App Token are present."""
        return bool(self.bot_token and self.app_token)


def get_slack_settings() -> SlackSettings:
    """Load Slack configuration from environment variables."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        pass

    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")

    # Clean whitespace
    bot_token = bot_token.strip() if bot_token else None
    app_token = app_token.strip() if app_token else None

    # Handle case where user copied app token without leading 'xapp-1-'
    if app_token and not app_token.startswith("xapp-"):
        if app_token.startswith("A0"):
            app_token = f"xapp-1-{app_token}"
        else:
            app_token = f"xapp-{app_token}"

    # Parse allowed users (comma-separated list of Slack user IDs)
    raw_allowed = os.getenv("SLACK_ALLOWED_USERS", "")
    allowed_users = [u.strip() for u in raw_allowed.split(",") if u.strip()]

    # Parse allowed email domains
    raw_domains = os.getenv("SLACK_ALLOWED_EMAIL_DOMAINS", "xyz.com")
    allowed_domains = [d.strip().lower() for d in raw_domains.split(",") if d.strip()]

    # Parse admin users
    raw_admins = os.getenv("SLACK_ADMIN_USERS", "")
    admin_users = [a.strip() for a in raw_admins.split(",") if a.strip()]

    # Rate limiting
    try:
        rate_requests = int(os.getenv("SLACK_RATE_LIMIT_REQUESTS", "10"))
    except ValueError:
        rate_requests = 10

    try:
        rate_window = int(os.getenv("SLACK_RATE_LIMIT_WINDOW", "60"))
    except ValueError:
        rate_window = 60

    return SlackSettings(
        bot_token=bot_token,
        app_token=app_token,
        allowed_users=allowed_users,
        allowed_email_domains=allowed_domains,
        admin_users=admin_users,
        rate_limit_requests=rate_requests,
        rate_limit_window=rate_window,
    )
