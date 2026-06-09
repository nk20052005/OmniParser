"""
Omnitool Enterprise - Main Entry Point.

Bootstraps all components and starts the platform.
Supports two modes:
  1. Gradio (in-VM interface)
  2. Slack (external interface)
  3. Both (default)
"""

import argparse
import asyncio
import logging
import signal
import sys
import threading

from .config import get_config, EnterpriseConfig
from .engine.gemma_client import GemmaClient
from .engine.conversation import ConversationEngine
from .memory.store import MemoryStore
from .approval.manager import ApprovalManager
from .tools.registry import ToolRegistry
from .tools.azure_vm import create_azure_vm_tools
from .tools.azure_monitor import create_azure_monitor_tools
from .tools.servicenow import create_servicenow_tools
from .tools.email_tools import create_email_tools
from .tools.slack_tools import create_slack_tools
from .tools.analytics import create_analytics_tools
from .agents.router import AgentRouter

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def create_tool_registry(config: EnterpriseConfig, gemma: GemmaClient) -> ToolRegistry:
    """Create and populate the tool registry with all available tools."""
    registry = ToolRegistry()

    # Azure VM tools
    registry.register_all(
        create_azure_vm_tools(config.azure.subscription_id, config.azure.resource_group)
    )

    # Azure Monitor tools
    registry.register_all(
        create_azure_monitor_tools(config.azure.subscription_id)
    )

    # ServiceNow tools
    registry.register_all(
        create_servicenow_tools(
            config.servicenow.instance_url,
            config.servicenow.username,
            config.servicenow.password,
        )
    )

    # Email tools
    registry.register_all(
        create_email_tools(
            config.email.smtp_server,
            config.email.smtp_port,
            config.email.imap_server,
            config.email.username,
            config.email.password,
        )
    )

    # Slack tools
    registry.register_all(
        create_slack_tools(config.slack.bot_token)
    )

    # Analytics tools (log analysis, alert analysis, RCA, dashboard, runbook)
    registry.register_all(
        create_analytics_tools(config.azure.subscription_id, gemma)
    )

    logger.info("Tool registry initialized with %d tools", len(registry.list_tools()))
    return registry


def build_engine(config: EnterpriseConfig) -> ConversationEngine:
    """Build the complete conversation engine with all dependencies."""
    gemma = GemmaClient(config.gemma)
    memory = MemoryStore(config.memory.db_path)
    approval = ApprovalManager()
    registry = create_tool_registry(config, gemma)

    engine = ConversationEngine(
        gemma=gemma,
        tool_registry=registry,
        memory=memory,
        approval_manager=approval,
    )

    logger.info("Conversation engine initialized")
    return engine


def run_gradio(engine: ConversationEngine, port: int = 7860, share: bool = False):
    """Start the Gradio web interface."""
    from .interfaces.gradio_app import create_gradio_app

    app = create_gradio_app(engine)
    logger.info("Starting Gradio interface on port %d", port)
    app.launch(server_port=port, share=share, server_name="0.0.0.0")


async def run_slack(engine: ConversationEngine):
    """Start the Slack bot."""
    from .integrations.slack_bot import SlackBot

    bot = SlackBot(engine)
    logger.info("Starting Slack bot")
    await bot.start()


def run_both(engine: ConversationEngine, gradio_port: int = 7860):
    """Run both Gradio and Slack interfaces concurrently."""
    # Start Slack bot in a background thread with its own event loop
    def _run_slack_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_slack(engine))
        except Exception:
            logger.exception("Slack bot failed")

    slack_thread = threading.Thread(target=_run_slack_thread, daemon=True)
    slack_thread.start()
    logger.info("Slack bot started in background thread")

    # Run Gradio in the main thread (it blocks)
    run_gradio(engine, port=gradio_port)


def main():
    parser = argparse.ArgumentParser(
        description="Omnitool Enterprise - Autonomous Operations Platform"
    )
    parser.add_argument(
        "--mode",
        choices=["gradio", "slack", "both"],
        default="both",
        help="Interface mode: gradio (in-VM), slack (external), or both",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio server port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    config = get_config()
    setup_logging(args.debug or config.debug)

    logger.info("=== Omnitool Enterprise Starting ===")
    logger.info("Mode: %s", args.mode)

    engine = build_engine(config)

    if args.mode == "gradio":
        run_gradio(engine, port=args.port, share=args.share)
    elif args.mode == "slack":
        asyncio.run(run_slack(engine))
    else:
        run_both(engine, gradio_port=args.port)


if __name__ == "__main__":
    main()
