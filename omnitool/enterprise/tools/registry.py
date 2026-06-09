"""
Tool Registry - maps intents to tools and manages tool execution.
"""

import logging
from typing import Any, Optional

from ..engine.intent import IntentCategory
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# Mapping from intent to tool name
INTENT_TO_TOOL: dict[str, str] = {
    "vm_start": "azure_vm_start",
    "vm_stop": "azure_vm_stop",
    "vm_restart": "azure_vm_restart",
    "vm_resize": "azure_vm_resize",
    "vm_deallocate": "azure_vm_deallocate",
    "vm_status": "azure_vm_status",
    "vm_list": "azure_vm_list",
    "vm_metrics": "azure_vm_metrics",

    "incident_create": "snow_create_incident",
    "incident_update": "snow_update_incident",
    "incident_close": "snow_close_incident",
    "incident_assign": "snow_assign_incident",
    "incident_query": "snow_query_incidents",

    "ritm_create": "snow_create_ritm",
    "service_request_create": "snow_create_request",
    "ticket_query": "snow_query",

    "email_read": "email_read",
    "email_send": "email_send",
    "email_reply": "email_reply",
    "email_forward": "email_forward",
    "email_search": "email_search",

    "slack_send": "slack_send",
    "slack_thread_reply": "slack_thread_reply",
    "slack_broadcast": "slack_broadcast",

    "monitor_alerts": "azure_monitor_alerts",
    "monitor_metrics": "azure_monitor_metrics",
    "resource_health": "azure_resource_health",
    "cost_analysis": "azure_cost_analysis",

    "log_analysis": "analyze_logs",
    "alert_analysis": "analyze_alert",
    "rca_generate": "generate_rca",

    "dashboard_generate": "generate_dashboard",
    "runbook_execute": "run_runbook",
}


class ToolRegistry:
    """Registry of all available tools. Maps intents to tools and handles execution."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def register_all(self, tools: list[BaseTool]):
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)

    def get_tool_for_intent(self, intent: IntentCategory) -> str:
        """Get the tool name that handles a given intent."""
        return INTENT_TO_TOOL.get(intent.value, "unknown")

    def list_tools(self) -> list[dict[str, Any]]:
        """List all registered tools with their metadata."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "required_params": t.required_params,
                "optional_params": t.optional_params,
            }
            for t in self._tools.values()
        ]

    async def execute(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name with given parameters."""
        tool = self._tools.get(tool_name)
        if not tool:
            logger.warning("Tool not found: %s", tool_name)
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' is not registered or available.",
            ).to_dict()

        # Validate parameters
        valid, error_msg = tool.validate_params(params)
        if not valid:
            return ToolResult(success=False, error=error_msg).to_dict()

        try:
            result = await tool.execute(params)
            return result.to_dict()
        except Exception as e:
            logger.exception("Tool execution failed: %s", tool_name)
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
            ).to_dict()
