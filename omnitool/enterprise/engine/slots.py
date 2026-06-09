"""
Slot Filling Engine.

Validates that all required parameters for an action are present.
Generates follow-up questions for missing information.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .gemma_client import GemmaClient
from .intent import IntentCategory

logger = logging.getLogger(__name__)


@dataclass
class SlotDefinition:
    """Definition of a single slot (parameter) for an action."""
    name: str
    description: str
    required: bool = True
    slot_type: str = "string"
    default: Any = None
    examples: list[str] = field(default_factory=list)


@dataclass
class SlotValidationResult:
    """Result of slot validation."""
    is_complete: bool
    filled_slots: dict[str, Any]
    missing_required: list[SlotDefinition]
    missing_optional: list[SlotDefinition]
    follow_up_question: Optional[str] = None


# Slot definitions for each intent
SLOT_DEFINITIONS: dict[str, list[SlotDefinition]] = {
    "vm_start": [
        SlotDefinition("vm_name", "Name of the virtual machine to start", required=True,
                       examples=["PROD-WEB-01", "dev-api-server"]),
        SlotDefinition("resource_group", "Azure resource group containing the VM", required=False),
        SlotDefinition("subscription", "Azure subscription", required=False),
    ],
    "vm_stop": [
        SlotDefinition("vm_name", "Name of the virtual machine to stop", required=True),
        SlotDefinition("resource_group", "Azure resource group", required=False),
        SlotDefinition("force", "Force stop without graceful shutdown", required=False,
                       slot_type="boolean", default=False),
    ],
    "vm_restart": [
        SlotDefinition("vm_name", "Name of the virtual machine to restart", required=True),
        SlotDefinition("resource_group", "Azure resource group", required=False),
    ],
    "vm_resize": [
        SlotDefinition("vm_name", "Name of the virtual machine to resize", required=True),
        SlotDefinition("new_size", "New VM size/SKU (e.g., Standard_D4s_v3)", required=True),
        SlotDefinition("resource_group", "Azure resource group", required=False),
    ],
    "vm_deallocate": [
        SlotDefinition("vm_name", "Name of the virtual machine to deallocate", required=True),
        SlotDefinition("resource_group", "Azure resource group", required=False),
    ],
    "vm_status": [
        SlotDefinition("vm_name", "Name of the virtual machine to check", required=True),
        SlotDefinition("resource_group", "Azure resource group", required=False),
    ],
    "vm_list": [
        SlotDefinition("resource_group", "Filter by resource group", required=False),
        SlotDefinition("status_filter", "Filter by VM status", required=False),
    ],
    "vm_metrics": [
        SlotDefinition("vm_name", "Name of the virtual machine", required=True),
        SlotDefinition("metric_type", "Metric type (cpu, memory, disk, network)", required=False, default="cpu"),
        SlotDefinition("time_range", "Time range (e.g., last 1 hour, last 24 hours)", required=False, default="1h"),
    ],
    "incident_create": [
        SlotDefinition("short_description", "Brief summary of the incident", required=True),
        SlotDefinition("severity", "Severity level (1=Critical, 2=High, 3=Medium, 4=Low)", required=True,
                       examples=["1", "2", "3", "4"]),
        SlotDefinition("category", "Incident category", required=False),
        SlotDefinition("assignment_group", "Team to assign the incident to", required=False),
        SlotDefinition("description", "Detailed description", required=False),
    ],
    "incident_update": [
        SlotDefinition("incident_id", "Incident number (e.g., INC0012345)", required=True),
        SlotDefinition("work_notes", "Work notes to add", required=False),
        SlotDefinition("state", "New state for the incident", required=False),
    ],
    "incident_close": [
        SlotDefinition("incident_id", "Incident number to close", required=True),
        SlotDefinition("close_notes", "Closure notes explaining resolution", required=True),
        SlotDefinition("resolution_code", "Resolution code", required=False, default="Resolved"),
    ],
    "incident_assign": [
        SlotDefinition("incident_id", "Incident number to assign", required=True),
        SlotDefinition("assignee", "Person or group to assign to", required=True),
    ],
    "incident_query": [
        SlotDefinition("query_text", "Search criteria", required=False),
        SlotDefinition("state_filter", "Filter by state", required=False, default="open"),
        SlotDefinition("severity_filter", "Filter by severity", required=False),
    ],
    "ritm_create": [
        SlotDefinition("item_name", "Name of the requested item", required=True),
        SlotDefinition("description", "Description of the request", required=True),
        SlotDefinition("quantity", "Quantity", required=False, slot_type="integer", default=1),
    ],
    "service_request_create": [
        SlotDefinition("request_type", "Type of service request", required=True),
        SlotDefinition("description", "Description of the request", required=True),
        SlotDefinition("priority", "Priority level", required=False, default="3"),
    ],
    "email_read": [
        SlotDefinition("folder", "Email folder", required=False, default="inbox"),
        SlotDefinition("filter", "Filter criteria", required=False, default="unread"),
        SlotDefinition("count", "Number of emails", required=False, slot_type="integer", default=10),
    ],
    "email_send": [
        SlotDefinition("recipient", "Email recipient(s)", required=True),
        SlotDefinition("subject", "Email subject", required=True),
        SlotDefinition("body", "Email body content", required=True),
        SlotDefinition("cc", "CC recipients", required=False),
    ],
    "email_reply": [
        SlotDefinition("email_id", "Email to reply to", required=True),
        SlotDefinition("body", "Reply message", required=True),
    ],
    "email_forward": [
        SlotDefinition("email_id", "Email to forward", required=True),
        SlotDefinition("recipient", "Forward to", required=True),
        SlotDefinition("body", "Additional message", required=False),
    ],
    "email_search": [
        SlotDefinition("query", "Search query", required=True),
        SlotDefinition("folder", "Folder to search", required=False),
    ],
    "slack_send": [
        SlotDefinition("channel", "Slack channel", required=True),
        SlotDefinition("message", "Message content", required=True),
    ],
    "slack_thread_reply": [
        SlotDefinition("channel", "Slack channel", required=True),
        SlotDefinition("thread_ts", "Thread timestamp", required=True),
        SlotDefinition("message", "Reply message", required=True),
    ],
    "slack_broadcast": [
        SlotDefinition("channels", "Channels to broadcast to", required=True),
        SlotDefinition("message", "Broadcast message", required=True),
    ],
    "monitor_alerts": [
        SlotDefinition("severity_filter", "Severity filter", required=False),
        SlotDefinition("resource_filter", "Resource filter", required=False),
        SlotDefinition("time_range", "Time range", required=False, default="24h"),
    ],
    "monitor_metrics": [
        SlotDefinition("resource_name", "Resource name", required=True),
        SlotDefinition("metric_type", "Metric type", required=False, default="cpu"),
        SlotDefinition("time_range", "Time range", required=False, default="1h"),
    ],
    "resource_health": [
        SlotDefinition("resource_name", "Resource to check", required=False),
        SlotDefinition("resource_type", "Type of resource", required=False),
    ],
    "cost_analysis": [
        SlotDefinition("scope", "Analysis scope", required=False, default="subscription"),
        SlotDefinition("time_range", "Time range", required=False, default="30d"),
        SlotDefinition("group_by", "Group by field", required=False),
    ],
    "log_analysis": [
        SlotDefinition("source", "Log source", required=False, default="application"),
        SlotDefinition("query", "Search query", required=False),
        SlotDefinition("time_range", "Time range", required=False, default="1h"),
        SlotDefinition("resource_name", "Resource name", required=False),
    ],
    "alert_analysis": [
        SlotDefinition("alert_id", "Alert ID", required=False),
        SlotDefinition("alert_description", "Alert description", required=False),
    ],
    "rca_generate": [
        SlotDefinition("incident_id", "Incident to generate RCA for", required=True),
        SlotDefinition("time_range", "Incident time window", required=False),
        SlotDefinition("affected_resources", "Affected resources", required=False),
    ],
    "dashboard_generate": [
        SlotDefinition("dashboard_type", "Dashboard type", required=True,
                       examples=["operations", "incident", "vm", "cost", "executive"]),
        SlotDefinition("time_range", "Time range", required=False, default="24h"),
        SlotDefinition("filters", "Dashboard filters", required=False),
    ],
    "runbook_execute": [
        SlotDefinition("runbook_name", "Runbook name", required=True),
        SlotDefinition("parameters", "Runbook parameters", required=False, slot_type="dict"),
    ],
}


FOLLOW_UP_SYSTEM_PROMPT = """You are a conversational assistant for an enterprise IT operations platform.

The user has made a request, but some required information is missing.
Generate a natural, conversational follow-up question to collect the missing information.

Rules:
- Ask about ONE missing parameter at a time (the most important one first)
- Be concise and specific
- If you can suggest likely values based on context, mention them
- Never be robotic - be natural and helpful
- Do not explain that parameters are missing - just ask naturally

Examples:
- "Which VM would you like me to start?"
- "What severity should this incident be? (1=Critical, 2=High, 3=Medium, 4=Low)"
- "Who should I send this email to?"
- "What should the subject line be?"
"""


class SlotFiller:
    """Validates required parameters and generates follow-up questions."""

    def __init__(self, gemma: GemmaClient):
        self._gemma = gemma

    def validate(
        self,
        intent: IntentCategory,
        entities: dict[str, Any],
        conversation_history: Optional[list[dict]] = None,
    ) -> SlotValidationResult:
        """Validate that all required slots are filled for an intent."""
        slot_defs = SLOT_DEFINITIONS.get(intent.value, [])
        if not slot_defs:
            return SlotValidationResult(
                is_complete=True,
                filled_slots=entities,
                missing_required=[],
                missing_optional=[],
            )

        filled = {}
        missing_required = []
        missing_optional = []

        for slot in slot_defs:
            value = entities.get(slot.name)
            if value is not None and value != "":
                filled[slot.name] = value
            elif slot.default is not None:
                filled[slot.name] = slot.default
            elif slot.required:
                missing_required.append(slot)
            else:
                missing_optional.append(slot)

        is_complete = len(missing_required) == 0
        follow_up = None

        if not is_complete:
            follow_up = self._generate_follow_up(
                intent, filled, missing_required, conversation_history
            )

        return SlotValidationResult(
            is_complete=is_complete,
            filled_slots=filled,
            missing_required=missing_required,
            missing_optional=missing_optional,
            follow_up_question=follow_up,
        )

    def _generate_follow_up(
        self,
        intent: IntentCategory,
        filled: dict[str, Any],
        missing: list[SlotDefinition],
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """Generate a natural follow-up question for missing parameters."""
        # Ask about the first missing required slot
        slot = missing[0]

        filled_summary = ", ".join(f"{k}={v}" for k, v in filled.items()) if filled else "none"
        examples_str = f" (e.g., {', '.join(slot.examples)})" if slot.examples else ""

        messages = [
            {"role": "system", "content": FOLLOW_UP_SYSTEM_PROMPT},
        ]

        if conversation_history:
            recent = conversation_history[-4:]
            for turn in recent:
                messages.append(turn)

        messages.append({
            "role": "user",
            "content": (
                f"Intent: {intent.value}\n"
                f"Already collected: {filled_summary}\n"
                f"Missing parameter: {slot.name} - {slot.description}{examples_str}\n\n"
                f"Generate a follow-up question to ask the user for this information."
            ),
        })

        return self._gemma.chat(messages, temperature=0.5, max_tokens=150).strip()

    def merge_entities(
        self,
        existing: dict[str, Any],
        new_entities: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge new entities into existing, preferring new values."""
        merged = dict(existing)
        for k, v in new_entities.items():
            if v is not None and v != "":
                merged[k] = v
        return merged
