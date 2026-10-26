"""Slack Bolt Application setup and routing for XYZ Technologies HR Policy Bot."""

from __future__ import annotations

from typing import Optional
from slack_bolt import App

from app.slack.config import SlackSettings, get_slack_settings
from app.slack.handlers import SlackEventHandler
from app.utils.logging import get_logger

logger = get_logger("app.slack.app")


def create_slack_app(
    settings: Optional[SlackSettings] = None,
    handler: Optional[SlackEventHandler] = None,
) -> App:
    """Instantiate and configure the Slack Bolt App."""
    active_settings = settings or get_slack_settings()
    active_handler = handler or SlackEventHandler()

    # If running in tests or unconfigured mode, use a dummy token to allow initialization
    token = active_settings.bot_token or "xoxb-dummy-test-token"

    app = App(token=token)

    # Register Direct Message listener
    @app.event("message")
    def on_message(event, say, client):
        active_handler.handle_message(event=event, say=say, client=client)

    # Register App Mention listener
    @app.event("app_mention")
    def on_app_mention(event, say, client):
        active_handler.handle_app_mention(event=event, say=say, client=client)

    # Register /ask-hr Slash Command
    @app.command("/ask-hr")
    def on_ask_hr(ack, command, respond, client):
        active_handler.handle_slash_command(ack=ack, command=command, respond=respond, client=client)

    logger.info("Configured Slack Bolt application event listeners: message, app_mention, /ask-hr.")
    return app
