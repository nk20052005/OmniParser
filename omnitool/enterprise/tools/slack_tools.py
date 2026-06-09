"""
Slack operation tools.

Uses the Slack SDK for sending messages, replying in threads, and broadcasting.
"""

import logging
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class SlackSendTool(BaseTool):
    name = "slack_send"
    description = "Send a message to a Slack channel"
    required_params = ["channel", "message"]

    def __init__(self, bot_token: str = ""):
        self._bot_token = bot_token

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            from slack_sdk.web.async_client import AsyncWebClient

            client = AsyncWebClient(token=self._bot_token)
            response = await client.chat_postMessage(
                channel=params["channel"],
                text=params["message"],
            )
            return ToolResult(
                success=True,
                data={
                    "channel": params["channel"],
                    "ts": response.get("ts", ""),
                },
                message=f"Message sent to #{params['channel']}.",
            )
        except Exception as e:
            logger.exception("Failed to send Slack message")
            return ToolResult(success=False, error=str(e))


class SlackThreadReplyTool(BaseTool):
    name = "slack_thread_reply"
    description = "Reply in a Slack thread"
    required_params = ["channel", "thread_ts", "message"]

    def __init__(self, bot_token: str = ""):
        self._bot_token = bot_token

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            from slack_sdk.web.async_client import AsyncWebClient

            client = AsyncWebClient(token=self._bot_token)
            response = await client.chat_postMessage(
                channel=params["channel"],
                text=params["message"],
                thread_ts=params["thread_ts"],
            )
            return ToolResult(
                success=True,
                data={"channel": params["channel"], "ts": response.get("ts", "")},
                message="Thread reply sent.",
            )
        except Exception as e:
            logger.exception("Failed to reply in thread")
            return ToolResult(success=False, error=str(e))


class SlackBroadcastTool(BaseTool):
    name = "slack_broadcast"
    description = "Broadcast a message to multiple Slack channels"
    required_params = ["channels", "message"]

    def __init__(self, bot_token: str = ""):
        self._bot_token = bot_token

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            from slack_sdk.web.async_client import AsyncWebClient

            client = AsyncWebClient(token=self._bot_token)
            channels = params["channels"]
            if isinstance(channels, str):
                channels = [c.strip() for c in channels.split(",")]

            results = []
            for channel in channels:
                try:
                    resp = await client.chat_postMessage(
                        channel=channel, text=params["message"]
                    )
                    results.append({"channel": channel, "success": True, "ts": resp.get("ts", "")})
                except Exception as ch_err:
                    results.append({"channel": channel, "success": False, "error": str(ch_err)})

            success_count = sum(1 for r in results if r["success"])
            return ToolResult(
                success=success_count > 0,
                data={"results": results, "total": len(channels), "success_count": success_count},
                message=f"Broadcast sent to {success_count}/{len(channels)} channels.",
            )
        except Exception as e:
            logger.exception("Failed to broadcast")
            return ToolResult(success=False, error=str(e))


def create_slack_tools(bot_token: str = "") -> list[BaseTool]:
    """Factory function to create all Slack tools."""
    return [
        SlackSendTool(bot_token),
        SlackThreadReplyTool(bot_token),
        SlackBroadcastTool(bot_token),
    ]
