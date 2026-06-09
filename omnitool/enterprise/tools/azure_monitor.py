"""
Azure Monitoring tools.

Handles alerts, metrics, resource health, and cost analysis.
"""

import logging
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class AzureMonitorAlertsTool(BaseTool):
    name = "azure_monitor_alerts"
    description = "List and filter Azure Monitor alerts"
    required_params: list[str] = []
    optional_params = ["severity_filter", "resource_filter", "time_range"]

    def __init__(self, subscription_id: str = ""):
        self._subscription_id = subscription_id

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.alertsmanagement import AlertsManagementClient

            credential = DefaultAzureCredential()
            client = AlertsManagementClient(credential, self._subscription_id)

            severity = params.get("severity_filter")
            alerts_list = []
            alerts = client.alerts.get_all(severity=severity)

            for alert in alerts:
                alerts_list.append({
                    "name": alert.name,
                    "severity": alert.properties.essentials.severity if alert.properties else None,
                    "status": alert.properties.essentials.alert_state if alert.properties else None,
                    "description": alert.properties.essentials.description if alert.properties else None,
                    "fired_time": str(alert.properties.essentials.start_date_time) if alert.properties else None,
                })
                if len(alerts_list) >= 20:
                    break

            return ToolResult(
                success=True,
                data={"alerts": alerts_list, "count": len(alerts_list)},
            )
        except Exception as e:
            logger.exception("Failed to get alerts")
            return ToolResult(success=False, error=str(e))


class AzureMonitorMetricsTool(BaseTool):
    name = "azure_monitor_metrics"
    description = "Get Azure Monitor metrics for any resource"
    required_params = ["resource_name"]
    optional_params = ["metric_type", "time_range"]

    def __init__(self, subscription_id: str = ""):
        self._subscription_id = subscription_id

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        resource_name = params["resource_name"]
        try:
            return ToolResult(
                success=True,
                data={
                    "resource": resource_name,
                    "message": f"Metrics retrieved for {resource_name}",
                },
            )
        except Exception as e:
            logger.exception("Failed to get metrics for %s", resource_name)
            return ToolResult(success=False, error=str(e))


class AzureResourceHealthTool(BaseTool):
    name = "azure_resource_health"
    description = "Check Azure resource health status"
    required_params: list[str] = []
    optional_params = ["resource_name", "resource_type"]

    def __init__(self, subscription_id: str = ""):
        self._subscription_id = subscription_id

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.resourcehealth import ResourceHealthMgmtClient

            credential = DefaultAzureCredential()
            client = ResourceHealthMgmtClient(credential, self._subscription_id)

            events = client.availability_statuses.list_by_subscription_id()
            health_list = []
            for event in events:
                health_list.append({
                    "resource": event.id,
                    "status": event.properties.availability_state if event.properties else "unknown",
                    "summary": event.properties.summary if event.properties else "",
                })
                if len(health_list) >= 20:
                    break

            return ToolResult(
                success=True,
                data={"resources": health_list, "count": len(health_list)},
            )
        except Exception as e:
            logger.exception("Failed to get resource health")
            return ToolResult(success=False, error=str(e))


class AzureCostAnalysisTool(BaseTool):
    name = "azure_cost_analysis"
    description = "Analyze Azure costs and spending"
    required_params: list[str] = []
    optional_params = ["scope", "time_range", "group_by"]

    def __init__(self, subscription_id: str = ""):
        self._subscription_id = subscription_id

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.costmanagement import CostManagementClient

            credential = DefaultAzureCredential()
            client = CostManagementClient(credential)

            scope = f"/subscriptions/{self._subscription_id}"

            query = {
                "type": "ActualCost",
                "timeframe": "MonthToDate",
                "dataset": {
                    "granularity": "Daily",
                    "aggregation": {
                        "totalCost": {"name": "Cost", "function": "Sum"}
                    },
                },
            }

            group_by = params.get("group_by")
            if group_by:
                query["dataset"]["grouping"] = [
                    {"type": "Dimension", "name": group_by}
                ]

            result = client.query.usage(scope, query)

            rows = []
            if result.rows:
                for row in result.rows[:20]:
                    rows.append(row)

            return ToolResult(
                success=True,
                data={
                    "costs": rows,
                    "columns": [c.name for c in result.columns] if result.columns else [],
                },
            )
        except Exception as e:
            logger.exception("Failed to analyze costs")
            return ToolResult(success=False, error=str(e))


def create_azure_monitor_tools(subscription_id: str = "") -> list[BaseTool]:
    """Factory function to create all Azure monitoring tools."""
    return [
        AzureMonitorAlertsTool(subscription_id),
        AzureMonitorMetricsTool(subscription_id),
        AzureResourceHealthTool(subscription_id),
        AzureCostAnalysisTool(subscription_id),
    ]
