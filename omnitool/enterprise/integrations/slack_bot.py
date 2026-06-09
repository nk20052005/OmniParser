"""
Slack Bot Integration.

Connects Omnitool Enterprise to Slack using Socket Mode.
All incoming messages flow through the conversation engine.
"""

import asyncio
import logging
import re
from typing import Any

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from ..config import get_config
from ..engine.conversation import ConversationEngine

logger = logging.getLogger(__name__)


class SlackBot:
    """
    Slack bot that bridges Slack messages to the conversation engine.

    Uses Socket Mode for real-time message handling without
    needing a public-facing webhook URL.
    """

    def __init__(self, conversation_engine: ConversationEngine):
        config = get_config()
        self._engine = conversation_engine

        self._app = AsyncApp(
            token=config.slack.bot_token,
            signing_secret=config.slack.signing_secret,
        )

        # Register event handlers
        self._app.event("app_mention")(self._handle_mention)
        self._app.event("message")(self._handle_direct_message)
        self._app.action(re.compile(r"^approve_action_.*"))(self._handle_approval)
        self._app.action(re.compile(r"^reject_action_.*"))(self._handle_rejection)

        self._socket_handler = AsyncSocketModeHandler(
            self._app, config.slack.app_token
        )

    async def start(self):
        """Start the Slack bot in Socket Mode."""
        logger.info("Starting Slack bot in Socket Mode...")
        await self._socket_handler.start_async()

    async def stop(self):
        """Stop the Slack bot."""
        logger.info("Stopping Slack bot...")
        await self._socket_handler.close_async()

    async def _handle_mention(self, event: dict, say: Any):
        """Handle @mention messages in channels."""
        text = event.get("text", "")
        user_id = event.get("user", "unknown")
        channel = event.get("channel", "unknown")
        thread_ts = event.get("thread_ts") or event.get("ts", "")

        # Strip the bot mention from the message
        text = re.sub(r"<@\w+>\s*", "", text).strip()

        if not text:
            await say(
                text="Hi! I'm Omnitool. How can I help? You can ask me to manage VMs, "
                     "create incidents, send emails, check alerts, and much more.",
                thread_ts=thread_ts,
            )
            return

        response = await self._engine.process_message(
            user_message=text,
            user_id=user_id,
            channel_id=channel,
        )

        # Build the Slack response
        blocks = self._build_response_blocks(response)

        await say(
            text=response.message,
            blocks=blocks,
            thread_ts=thread_ts,
        )

    async def _handle_direct_message(self, event: dict, say: Any):
        """Handle direct messages to the bot."""
        # Skip bot's own messages
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        # Only handle DMs (channel type 'im')
        channel_type = event.get("channel_type", "")
        if channel_type != "im":
            return

        text = event.get("text", "").strip()
        user_id = event.get("user", "unknown")
        channel = event.get("channel", "unknown")

        if not text:
            return

        response = await self._engine.process_message(
            user_message=text,
            user_id=user_id,
            channel_id=channel,
        )

        blocks = self._build_response_blocks(response)
        await say(text=response.message, blocks=blocks)

    async def _handle_approval(self, ack: Any, body: dict, say: Any):
        """Handle approval button click."""
        await ack()
        user_id = body.get("user", {}).get("id", "unknown")
        channel = body.get("channel", {}).get("id", "default")

        response = await self._engine.process_message(
            user_message="yes, confirmed",
            user_id=user_id,
            channel_id=channel,
        )
        await say(text=response.message)

    async def _handle_rejection(self, ack: Any, body: dict, say: Any):
        """Handle rejection button click."""
        await ack()
        user_id = body.get("user", {}).get("id", "unknown")
        channel = body.get("channel", {}).get("id", "default")

        response = await self._engine.process_message(
            user_message="no, cancel",
            user_id=user_id,
            channel_id=channel,
        )
        await say(text=response.message)

    def _build_response_blocks(self, response: Any) -> list[dict]:
        """Build Slack Block Kit blocks for the response."""
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": response.message},
            }
        ]

        # Add approval buttons if awaiting approval
        if response.awaiting_approval:
            conv_id = response.conversation_id
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "action_id": f"approve_action_{conv_id}",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "style": "danger",
                        "action_id": f"reject_action_{conv_id}",
                    },
                ],
            })

        # Add context about tool used
        if response.tool_called:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Tool: `{response.tool_called}` | "
                                f"Intent: `{response.intent.value if response.intent else 'N/A'}`",
                    }
                ],
            })

        return blocks
