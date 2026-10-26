"""Slack Bot Integration package for XYZ Technologies HR Policy Assistant."""

from __future__ import annotations

from app.slack.app import create_slack_app
from app.slack.config import SlackSettings, get_slack_settings
from app.slack.handlers import SlackEventHandler

__all__ = [
    "create_slack_app",
    "get_slack_settings",
    "SlackSettings",
    "SlackEventHandler",
]
