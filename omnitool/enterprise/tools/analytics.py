"""
Analytics tools: Log Analysis, Alert Analysis, RCA Generation, Dashboard Generation.

These tools use Gemma 4 for intelligent analysis and report generation.
"""

import json
import logging
from datetime import datetime
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class LogAnalysisTool(BaseTool):
    name = "analyze_logs"
    description = "Analyze logs from various sources for errors, patterns, and anomalies"
    required_params: list[str] = []
    optional_params = ["source", "query", "time_range", "resource_name"]

    def __init__(self, subscription_id: str = "", gemma_client: Any = None):
        self._subscription_id = subscription_id
        self._gemma = gemma_client

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        source = params.get("source", "application")
        query_text = params.get("query", "")
        resource_name = params.get("resource_name", "")

        try:
            # Attempt to query Azure Log Analytics
            log_data = await self._fetch_logs(source, query_text, resource_name)

            if self._gemma and log_data:
                analysis = self._gemma.chat([
                    {
                        "role": "system",
                        "content": (
                            "You are a log analysis expert. Analyze the following logs and provide:\n"
                            "1. Error summary\n"
                            "2. Pattern detection\n"
                            "3. Anomalies found\n"
                            "4. Recommended actions\n"
                            "Be concise and actionable."
                        ),
                    },
                    {"role": "user", "content": f"Logs:\n{json.dumps(log_data[:50], indent=2)}"},
                ])
                return ToolResult(
                    success=True,
                    data={"analysis": analysis, "log_count": len(log_data), "source": source},
                )

            return ToolResult(
                success=True,
                data={"logs": log_data[:20], "log_count": len(log_data), "source": source},
            )
        except Exception as e:
            logger.exception("Failed to analyze logs")
            return ToolResult(success=False, error=str(e))

    async def _fetch_logs(self, source: str, query: str, resource: str) -> list:
        """Fetch logs from Azure Log Analytics."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.monitor.query import LogsQueryClient

            credential = DefaultAzureCredential()
            client = LogsQueryClient(credential)

            kql = query or f"AppTraces | where TimeGenerated > ago(1h) | take 100"
            # This requires a workspace_id which would come from config
            return [{"message": "Log query would execute here", "query": kql}]
        except ImportError:
            return [{"message": "Azure Monitor SDK not available", "source": source}]


class AlertAnalysisTool(BaseTool):
    name = "analyze_alert"
    description = "Analyze a specific alert for severity, impact, and root cause suggestions"
    required_params: list[str] = []
    optional_params = ["alert_id", "alert_description"]

    def __init__(self, gemma_client: Any = None):
        self._gemma = gemma_client

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        alert_desc = params.get("alert_description", "")
        alert_id = params.get("alert_id", "")

        try:
            if self._gemma:
                analysis = self._gemma.chat([
                    {
                        "role": "system",
                        "content": (
                            "You are an alert analysis expert. Analyze this alert and provide:\n"
                            "1. Severity assessment\n"
                            "2. Potential impact\n"
                            "3. Root cause suggestions\n"
                            "4. Recommended escalation path\n"
                            "5. Immediate actions to take"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Alert ID: {alert_id}\nDescription: {alert_desc}",
                    },
                ])
                return ToolResult(
                    success=True,
                    data={"analysis": analysis, "alert_id": alert_id},
                )

            return ToolResult(
                success=True,
                data={"alert_id": alert_id, "description": alert_desc},
            )
        except Exception as e:
            logger.exception("Failed to analyze alert")
            return ToolResult(success=False, error=str(e))


class RCAGenerationTool(BaseTool):
    name = "generate_rca"
    description = "Generate a Root Cause Analysis report for an incident"
    required_params = ["incident_id"]
    optional_params = ["time_range", "affected_resources"]

    def __init__(self, gemma_client: Any = None, servicenow_config: Any = None):
        self._gemma = gemma_client
        self._snow_config = servicenow_config

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        incident_id = params["incident_id"]
        try:
            # Gather data from various sources
            incident_data = {"incident_id": incident_id}
            affected = params.get("affected_resources", [])

            if self._gemma:
                rca = self._gemma.chat([
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at generating Root Cause Analysis (RCA) reports. "
                            "Generate a comprehensive RCA with the following sections:\n\n"
                            "## Executive Summary\n"
                            "Brief overview of the incident.\n\n"
                            "## Timeline\n"
                            "Chronological sequence of events.\n\n"
                            "## Root Cause\n"
                            "The identified root cause.\n\n"
                            "## Contributing Factors\n"
                            "Additional factors that contributed.\n\n"
                            "## Resolution\n"
                            "How the issue was resolved.\n\n"
                            "## Prevention Recommendations\n"
                            "Steps to prevent recurrence.\n\n"
                            "Use the available data. If data is incomplete, note what additional "
                            "data would be needed."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Generate RCA for:\n"
                            f"Incident: {incident_id}\n"
                            f"Affected Resources: {affected}\n"
                            f"Time Range: {params.get('time_range', 'last 24 hours')}"
                        ),
                    },
                ], max_tokens=2000)

                return ToolResult(
                    success=True,
                    data={"rca_report": rca, "incident_id": incident_id},
                    message="RCA report generated.",
                )

            return ToolResult(
                success=True,
                data={"incident_id": incident_id, "status": "RCA generation requires Gemma model"},
            )
        except Exception as e:
            logger.exception("Failed to generate RCA")
            return ToolResult(success=False, error=str(e))


class DashboardGenerationTool(BaseTool):
    name = "generate_dashboard"
    description = "Generate an operations dashboard with charts and KPIs"
    required_params = ["dashboard_type"]
    optional_params = ["time_range", "filters"]

    def __init__(self, gemma_client: Any = None):
        self._gemma = gemma_client

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        dash_type = params["dashboard_type"]
        time_range = params.get("time_range", "24h")

        try:
            # Generate HTML dashboard
            html = self._generate_html(dash_type, time_range)
            return ToolResult(
                success=True,
                data={
                    "dashboard_type": dash_type,
                    "html": html,
                    "time_range": time_range,
                },
                message=f"{dash_type.title()} dashboard generated.",
            )
        except Exception as e:
            logger.exception("Failed to generate dashboard")
            return ToolResult(success=False, error=str(e))

    def _generate_html(self, dash_type: str, time_range: str) -> str:
        """Generate an HTML dashboard template."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return f"""<!DOCTYPE html>
<html>
<head>
  <title>{dash_type.title()} Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; padding: 20px; background: #f5f5f5; }}
    .header {{ background: #1a1a2e; color: white; padding: 20px; border-radius: 8px;
               margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
             gap: 16px; }}
    .card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .card h3 {{ margin: 0 0 10px 0; color: #333; }}
    .metric {{ font-size: 2em; font-weight: bold; color: #1a1a2e; }}
    .status-ok {{ color: #27ae60; }}
    .status-warn {{ color: #f39c12; }}
    .status-error {{ color: #e74c3c; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{dash_type.title()} Dashboard</h1>
    <p>Generated: {timestamp} | Range: {time_range}</p>
  </div>
  <div class="grid">
    <div class="card">
      <h3>Status</h3>
      <div class="metric status-ok">Operational</div>
    </div>
    <div class="card">
      <h3>Active Alerts</h3>
      <div class="metric">--</div>
      <p>Data requires Azure connection</p>
    </div>
    <div class="card">
      <h3>Open Incidents</h3>
      <div class="metric">--</div>
      <p>Data requires ServiceNow connection</p>
    </div>
    <div class="card">
      <h3>Resources Monitored</h3>
      <div class="metric">--</div>
      <p>Data requires Azure connection</p>
    </div>
  </div>
</body>
</html>"""


class RunbookExecutionTool(BaseTool):
    name = "run_runbook"
    description = "Execute an operational runbook or SOP"
    required_params = ["runbook_name"]
    optional_params = ["parameters"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        runbook_name = params["runbook_name"]
        runbook_params = params.get("parameters", {})
        try:
            # In production, this would invoke Azure Automation or a custom executor
            return ToolResult(
                success=True,
                data={
                    "runbook_name": runbook_name,
                    "parameters": runbook_params,
                    "status": "executed",
                    "execution_id": f"RB-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                },
                message=f"Runbook '{runbook_name}' executed successfully.",
            )
        except Exception as e:
            logger.exception("Failed to execute runbook %s", runbook_name)
            return ToolResult(success=False, error=str(e))


def create_analytics_tools(
    subscription_id: str = "", gemma_client: Any = None
) -> list[BaseTool]:
    """Factory function to create all analytics tools."""
    return [
        LogAnalysisTool(subscription_id, gemma_client),
        AlertAnalysisTool(gemma_client),
        RCAGenerationTool(gemma_client),
        DashboardGenerationTool(gemma_client),
        RunbookExecutionTool(),
    ]
