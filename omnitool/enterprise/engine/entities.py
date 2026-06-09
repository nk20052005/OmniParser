"""
Entity Extraction using Gemma 4.

Extracts structured entities from user messages without regex or keyword matching.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .gemma_client import GemmaClient
from .intent import IntentCategory

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    """Entities extracted from a user message."""
    entities: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    confidence: float = 0.0


# Maps intent categories to the entities we should look for
INTENT_ENTITY_SCHEMAS: dict[str, list[dict[str, str]]] = {
    "vm_start": [
        {"name": "vm_name", "type": "string", "description": "Name of the virtual machine"},
        {"name": "resource_group", "type": "string", "description": "Azure resource group"},
        {"name": "subscription", "type": "string", "description": "Azure subscription name or ID"},
    ],
    "vm_stop": [
        {"name": "vm_name", "type": "string", "description": "Name of the virtual machine"},
        {"name": "resource_group", "type": "string", "description": "Azure resource group"},
        {"name": "force", "type": "boolean", "description": "Whether to force stop"},
    ],
    "vm_restart": [
        {"name": "vm_name", "type": "string", "description": "Name of the virtual machine"},
        {"name": "resource_group", "type": "string", "description": "Azure resource group"},
    ],
    "vm_resize": [
        {"name": "vm_name", "type": "string", "description": "Name of the virtual machine"},
        {"name": "new_size", "type": "string", "description": "New VM size/SKU"},
        {"name": "resource_group", "type": "string", "description": "Azure resource group"},
    ],
    "vm_deallocate": [
        {"name": "vm_name", "type": "string", "description": "Name of the virtual machine"},
        {"name": "resource_group", "type": "string", "description": "Azure resource group"},
    ],
    "vm_status": [
        {"name": "vm_name", "type": "string", "description": "Name of the virtual machine"},
        {"name": "resource_group", "type": "string", "description": "Azure resource group"},
    ],
    "vm_list": [
        {"name": "resource_group", "type": "string", "description": "Azure resource group to filter by"},
        {"name": "status_filter", "type": "string", "description": "Filter by status (running, stopped, etc.)"},
    ],
    "vm_metrics": [
        {"name": "vm_name", "type": "string", "description": "Name of the virtual machine"},
        {"name": "metric_type", "type": "string", "description": "Type of metric (cpu, memory, disk, network)"},
        {"name": "time_range", "type": "string", "description": "Time range for metrics"},
    ],
    "incident_create": [
        {"name": "short_description", "type": "string", "description": "Brief description of the incident"},
        {"name": "severity", "type": "string", "description": "Severity level (1-4 or Sev1-Sev4)"},
        {"name": "category", "type": "string", "description": "Incident category"},
        {"name": "assignment_group", "type": "string", "description": "Team to assign to"},
        {"name": "description", "type": "string", "description": "Detailed description"},
    ],
    "incident_update": [
        {"name": "incident_id", "type": "string", "description": "Incident number (e.g., INC0012345)"},
        {"name": "work_notes", "type": "string", "description": "Work notes to add"},
        {"name": "state", "type": "string", "description": "New state"},
    ],
    "incident_close": [
        {"name": "incident_id", "type": "string", "description": "Incident number"},
        {"name": "close_notes", "type": "string", "description": "Closure notes"},
        {"name": "resolution_code", "type": "string", "description": "Resolution code"},
    ],
    "incident_assign": [
        {"name": "incident_id", "type": "string", "description": "Incident number"},
        {"name": "assignee", "type": "string", "description": "Person or group to assign to"},
    ],
    "incident_query": [
        {"name": "query_text", "type": "string", "description": "Search text or filter criteria"},
        {"name": "state_filter", "type": "string", "description": "Filter by state (open, resolved, etc.)"},
        {"name": "severity_filter", "type": "string", "description": "Filter by severity"},
        {"name": "time_range", "type": "string", "description": "Time range filter"},
    ],
    "ritm_create": [
        {"name": "item_name", "type": "string", "description": "Name of the requested item"},
        {"name": "description", "type": "string", "description": "Description of the request"},
        {"name": "quantity", "type": "integer", "description": "Quantity requested"},
    ],
    "service_request_create": [
        {"name": "request_type", "type": "string", "description": "Type of service request"},
        {"name": "description", "type": "string", "description": "Description of the request"},
        {"name": "priority", "type": "string", "description": "Priority level"},
    ],
    "email_read": [
        {"name": "folder", "type": "string", "description": "Email folder (inbox, sent, etc.)"},
        {"name": "filter", "type": "string", "description": "Filter (unread, from specific sender, etc.)"},
        {"name": "count", "type": "integer", "description": "Number of emails to read"},
    ],
    "email_send": [
        {"name": "recipient", "type": "string", "description": "Email recipient(s)"},
        {"name": "subject", "type": "string", "description": "Email subject"},
        {"name": "body", "type": "string", "description": "Email body content"},
        {"name": "cc", "type": "string", "description": "CC recipients"},
    ],
    "email_reply": [
        {"name": "email_id", "type": "string", "description": "ID of email to reply to"},
        {"name": "body", "type": "string", "description": "Reply body"},
    ],
    "email_forward": [
        {"name": "email_id", "type": "string", "description": "ID of email to forward"},
        {"name": "recipient", "type": "string", "description": "Forward recipient"},
        {"name": "body", "type": "string", "description": "Additional message"},
    ],
    "email_search": [
        {"name": "query", "type": "string", "description": "Search query"},
        {"name": "folder", "type": "string", "description": "Folder to search in"},
    ],
    "slack_send": [
        {"name": "channel", "type": "string", "description": "Slack channel name"},
        {"name": "message", "type": "string", "description": "Message content"},
    ],
    "slack_thread_reply": [
        {"name": "channel", "type": "string", "description": "Slack channel name"},
        {"name": "thread_ts", "type": "string", "description": "Thread timestamp"},
        {"name": "message", "type": "string", "description": "Reply message"},
    ],
    "slack_broadcast": [
        {"name": "channels", "type": "list", "description": "List of channels to broadcast to"},
        {"name": "message", "type": "string", "description": "Broadcast message"},
    ],
    "monitor_alerts": [
        {"name": "severity_filter", "type": "string", "description": "Filter by severity"},
        {"name": "resource_filter", "type": "string", "description": "Filter by resource"},
        {"name": "time_range", "type": "string", "description": "Time range"},
    ],
    "monitor_metrics": [
        {"name": "resource_name", "type": "string", "description": "Resource to monitor"},
        {"name": "metric_type", "type": "string", "description": "Type of metric"},
        {"name": "time_range", "type": "string", "description": "Time range"},
    ],
    "resource_health": [
        {"name": "resource_name", "type": "string", "description": "Resource to check"},
        {"name": "resource_type", "type": "string", "description": "Type of resource"},
    ],
    "cost_analysis": [
        {"name": "scope", "type": "string", "description": "Scope (subscription, resource group, resource)"},
        {"name": "time_range", "type": "string", "description": "Time range for analysis"},
        {"name": "group_by", "type": "string", "description": "Group results by (resource, type, tag)"},
    ],
    "log_analysis": [
        {"name": "source", "type": "string", "description": "Log source (application, system, container)"},
        {"name": "query", "type": "string", "description": "Search query or filter"},
        {"name": "time_range", "type": "string", "description": "Time range"},
        {"name": "resource_name", "type": "string", "description": "Resource to query logs for"},
    ],
    "alert_analysis": [
        {"name": "alert_id", "type": "string", "description": "Alert ID to analyze"},
        {"name": "alert_description", "type": "string", "description": "Description of the alert"},
    ],
    "rca_generate": [
        {"name": "incident_id", "type": "string", "description": "Incident to generate RCA for"},
        {"name": "time_range", "type": "string", "description": "Time window of the incident"},
        {"name": "affected_resources", "type": "list", "description": "List of affected resources"},
    ],
    "dashboard_generate": [
        {"name": "dashboard_type", "type": "string", "description": "Type (operations, incident, vm, cost, executive)"},
        {"name": "time_range", "type": "string", "description": "Time range for dashboard data"},
        {"name": "filters", "type": "string", "description": "Any filters to apply"},
    ],
    "runbook_execute": [
        {"name": "runbook_name", "type": "string", "description": "Name of the runbook to execute"},
        {"name": "parameters", "type": "dict", "description": "Parameters for the runbook"},
    ],
}


ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are an entity extraction engine for an enterprise IT operations platform.

Given a user message and the detected intent, extract all relevant entities from the message.

Rules:
- Extract ONLY entities that are explicitly mentioned or clearly implied in the message
- Do NOT invent or hallucinate entity values
- If an entity is not present in the message, do not include it
- Normalize values where appropriate (e.g., "sev 2" -> "2", "production web server" -> keep as-is for VM name resolution)
- For severity, normalize to numbers: Sev1/Critical/P1 -> "1", Sev2/High/P2 -> "2", etc.
- For boolean values, use true/false

Respond with JSON:
{
  "entities": {
    "<entity_name>": "<extracted_value>",
    ...
  },
  "confidence": <0.0-1.0>
}

Only include entities that were found in the message."""


class EntityExtractor:
    """Extracts entities from user messages using Gemma 4."""

    def __init__(self, gemma: GemmaClient):
        self._gemma = gemma

    def extract(
        self,
        user_message: str,
        intent: IntentCategory,
        conversation_history: Optional[list[dict]] = None,
    ) -> ExtractedEntities:
        """Extract entities from a user message given the detected intent."""
        schema = INTENT_ENTITY_SCHEMAS.get(intent.value, [])
        if not schema:
            return ExtractedEntities(raw_text=user_message, confidence=1.0)

        schema_desc = "\n".join(
            f"- {e['name']} ({e['type']}): {e['description']}" for e in schema
        )

        messages = [{"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT}]

        if conversation_history:
            recent = conversation_history[-6:]
            for turn in recent:
                messages.append(turn)

        messages.append({
            "role": "user",
            "content": (
                f"Intent: {intent.value}\n\n"
                f"Expected entities:\n{schema_desc}\n\n"
                f"User message: {user_message}\n\n"
                f"Extract all entities found in the message."
            ),
        })

        result = self._gemma.chat_json(messages, temperature=0.1)

        return ExtractedEntities(
            entities=result.get("entities", {}),
            raw_text=user_message,
            confidence=float(result.get("confidence", 0.5)),
        )
