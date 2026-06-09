"""
ServiceNow integration tools.

Handles incident, RITM, and service request operations via the ServiceNow REST API.
"""

import logging
from typing import Any

import requests

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class _ServiceNowBase(BaseTool):
    """Base class for ServiceNow tools with shared connection logic."""

    def __init__(self, instance_url: str = "", username: str = "", password: str = ""):
        self._instance_url = instance_url.rstrip("/")
        self._auth = (username, password) if username else None

    def _api_url(self, table: str) -> str:
        return f"{self._instance_url}/api/now/table/{table}"

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "Accept": "application/json"}

    def _get(self, table: str, params: dict | None = None) -> dict:
        resp = requests.get(
            self._api_url(table), auth=self._auth, headers=self._headers(),
            params=params, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, table: str, data: dict) -> dict:
        resp = requests.post(
            self._api_url(table), auth=self._auth, headers=self._headers(),
            json=data, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _patch(self, table: str, sys_id: str, data: dict) -> dict:
        resp = requests.patch(
            f"{self._api_url(table)}/{sys_id}",
            auth=self._auth, headers=self._headers(),
            json=data, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


class SNCreateIncidentTool(_ServiceNowBase):
    name = "snow_create_incident"
    description = "Create an incident in ServiceNow"
    required_params = ["short_description", "severity"]
    optional_params = ["category", "assignment_group", "description"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            severity_map = {"1": "1", "2": "2", "3": "3", "4": "4"}
            sev = str(params.get("severity", "3"))
            sev = severity_map.get(sev, sev)

            payload = {
                "short_description": params["short_description"],
                "severity": sev,
                "impact": sev,
                "urgency": sev,
            }
            if params.get("category"):
                payload["category"] = params["category"]
            if params.get("assignment_group"):
                payload["assignment_group"] = params["assignment_group"]
            if params.get("description"):
                payload["description"] = params["description"]

            result = self._post("incident", payload)
            inc = result.get("result", {})
            return ToolResult(
                success=True,
                data={
                    "incident_number": inc.get("number", ""),
                    "sys_id": inc.get("sys_id", ""),
                    "state": inc.get("state", ""),
                    "short_description": inc.get("short_description", ""),
                },
                message=f"Incident {inc.get('number', '')} created successfully.",
            )
        except Exception as e:
            logger.exception("Failed to create incident")
            return ToolResult(success=False, error=str(e))


class SNUpdateIncidentTool(_ServiceNowBase):
    name = "snow_update_incident"
    description = "Update an existing incident in ServiceNow"
    required_params = ["incident_id"]
    optional_params = ["work_notes", "state"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            incident_id = params["incident_id"]
            # Look up sys_id from incident number
            query_result = self._get("incident", {"number": incident_id, "sysparm_limit": "1"})
            records = query_result.get("result", [])
            if not records:
                return ToolResult(success=False, error=f"Incident {incident_id} not found")

            sys_id = records[0]["sys_id"]
            update_data: dict[str, Any] = {}
            if params.get("work_notes"):
                update_data["work_notes"] = params["work_notes"]
            if params.get("state"):
                update_data["state"] = params["state"]

            result = self._patch("incident", sys_id, update_data)
            inc = result.get("result", {})
            return ToolResult(
                success=True,
                data={"incident_number": incident_id, "updated_fields": list(update_data.keys())},
                message=f"Incident {incident_id} updated.",
            )
        except Exception as e:
            logger.exception("Failed to update incident")
            return ToolResult(success=False, error=str(e))


class SNCloseIncidentTool(_ServiceNowBase):
    name = "snow_close_incident"
    description = "Close/resolve an incident in ServiceNow"
    required_params = ["incident_id", "close_notes"]
    optional_params = ["resolution_code"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            incident_id = params["incident_id"]
            query_result = self._get("incident", {"number": incident_id, "sysparm_limit": "1"})
            records = query_result.get("result", [])
            if not records:
                return ToolResult(success=False, error=f"Incident {incident_id} not found")

            sys_id = records[0]["sys_id"]
            update_data = {
                "state": "6",  # Resolved
                "close_notes": params["close_notes"],
                "close_code": params.get("resolution_code", "Solved (Permanently)"),
            }
            self._patch("incident", sys_id, update_data)
            return ToolResult(
                success=True,
                data={"incident_number": incident_id, "state": "resolved"},
                message=f"Incident {incident_id} has been resolved.",
            )
        except Exception as e:
            logger.exception("Failed to close incident")
            return ToolResult(success=False, error=str(e))


class SNAssignIncidentTool(_ServiceNowBase):
    name = "snow_assign_incident"
    description = "Assign an incident to a person or group"
    required_params = ["incident_id", "assignee"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            incident_id = params["incident_id"]
            query_result = self._get("incident", {"number": incident_id, "sysparm_limit": "1"})
            records = query_result.get("result", [])
            if not records:
                return ToolResult(success=False, error=f"Incident {incident_id} not found")

            sys_id = records[0]["sys_id"]
            self._patch("incident", sys_id, {"assigned_to": params["assignee"]})
            return ToolResult(
                success=True,
                data={"incident_number": incident_id, "assigned_to": params["assignee"]},
                message=f"Incident {incident_id} assigned to {params['assignee']}.",
            )
        except Exception as e:
            logger.exception("Failed to assign incident")
            return ToolResult(success=False, error=str(e))


class SNQueryIncidentsTool(_ServiceNowBase):
    name = "snow_query_incidents"
    description = "Search and list incidents in ServiceNow"
    required_params: list[str] = []
    optional_params = ["query_text", "state_filter", "severity_filter", "time_range"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            query_parts = []
            state = params.get("state_filter", "open")
            if state == "open":
                query_parts.append("stateNOT IN6,7,8")  # Not resolved/closed/cancelled
            elif state == "resolved":
                query_parts.append("state=6")
            elif state == "closed":
                query_parts.append("state=7")

            if params.get("severity_filter"):
                query_parts.append(f"severity={params['severity_filter']}")

            if params.get("query_text"):
                query_parts.append(f"short_descriptionLIKE{params['query_text']}")

            sysparm_query = "^".join(query_parts) if query_parts else ""

            result = self._get("incident", {
                "sysparm_query": sysparm_query,
                "sysparm_limit": "20",
                "sysparm_fields": "number,short_description,severity,state,assigned_to,sys_created_on",
                "sysparm_display_value": "true",
            })

            incidents = result.get("result", [])
            return ToolResult(
                success=True,
                data={"incidents": incidents, "count": len(incidents)},
            )
        except Exception as e:
            logger.exception("Failed to query incidents")
            return ToolResult(success=False, error=str(e))


class SNCreateRITMTool(_ServiceNowBase):
    name = "snow_create_ritm"
    description = "Create a Requested Item (RITM) in ServiceNow"
    required_params = ["item_name", "description"]
    optional_params = ["quantity"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            payload = {
                "short_description": params["item_name"],
                "description": params["description"],
                "quantity": str(params.get("quantity", 1)),
            }
            result = self._post("sc_req_item", payload)
            ritm = result.get("result", {})
            return ToolResult(
                success=True,
                data={
                    "ritm_number": ritm.get("number", ""),
                    "sys_id": ritm.get("sys_id", ""),
                },
                message=f"RITM {ritm.get('number', '')} created.",
            )
        except Exception as e:
            logger.exception("Failed to create RITM")
            return ToolResult(success=False, error=str(e))


class SNCreateRequestTool(_ServiceNowBase):
    name = "snow_create_request"
    description = "Create a service request in ServiceNow"
    required_params = ["request_type", "description"]
    optional_params = ["priority"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            payload = {
                "short_description": params["request_type"],
                "description": params["description"],
                "priority": params.get("priority", "3"),
            }
            result = self._post("sc_request", payload)
            req = result.get("result", {})
            return ToolResult(
                success=True,
                data={
                    "request_number": req.get("number", ""),
                    "sys_id": req.get("sys_id", ""),
                },
                message=f"Service request {req.get('number', '')} created.",
            )
        except Exception as e:
            logger.exception("Failed to create service request")
            return ToolResult(success=False, error=str(e))


class SNQueryTool(_ServiceNowBase):
    name = "snow_query"
    description = "General query against ServiceNow tables"
    required_params: list[str] = []
    optional_params = ["query_text", "state_filter", "severity_filter"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        # Delegates to incident query as default
        tool = SNQueryIncidentsTool(self._instance_url, *self._auth if self._auth else ("", ""))
        return await tool.execute(params)


def create_servicenow_tools(
    instance_url: str = "", username: str = "", password: str = ""
) -> list[BaseTool]:
    """Factory function to create all ServiceNow tools."""
    return [
        SNCreateIncidentTool(instance_url, username, password),
        SNUpdateIncidentTool(instance_url, username, password),
        SNCloseIncidentTool(instance_url, username, password),
        SNAssignIncidentTool(instance_url, username, password),
        SNQueryIncidentsTool(instance_url, username, password),
        SNCreateRITMTool(instance_url, username, password),
        SNCreateRequestTool(instance_url, username, password),
        SNQueryTool(instance_url, username, password),
    ]
