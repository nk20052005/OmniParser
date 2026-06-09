"""
Intent Detection using Gemma 4.

Classifies user messages into structured intents without keyword matching.
Gemma interprets the semantic meaning, synonyms, and context.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .gemma_client import GemmaClient

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    VM_START = "vm_start"
    VM_STOP = "vm_stop"
    VM_RESTART = "vm_restart"
    VM_RESIZE = "vm_resize"
    VM_DEALLOCATE = "vm_deallocate"
    VM_STATUS = "vm_status"
    VM_LIST = "vm_list"
    VM_METRICS = "vm_metrics"

    INCIDENT_CREATE = "incident_create"
    INCIDENT_UPDATE = "incident_update"
    INCIDENT_CLOSE = "incident_close"
    INCIDENT_ASSIGN = "incident_assign"
    INCIDENT_QUERY = "incident_query"

    RITM_CREATE = "ritm_create"
    SERVICE_REQUEST_CREATE = "service_request_create"
    TICKET_QUERY = "ticket_query"

    EMAIL_READ = "email_read"
    EMAIL_SEND = "email_send"
    EMAIL_REPLY = "email_reply"
    EMAIL_FORWARD = "email_forward"
    EMAIL_SEARCH = "email_search"

    SLACK_SEND = "slack_send"
    SLACK_THREAD_REPLY = "slack_thread_reply"
    SLACK_BROADCAST = "slack_broadcast"

    MONITOR_ALERTS = "monitor_alerts"
    MONITOR_METRICS = "monitor_metrics"
    RESOURCE_HEALTH = "resource_health"
    COST_ANALYSIS = "cost_analysis"

    LOG_ANALYSIS = "log_analysis"
    ALERT_ANALYSIS = "alert_analysis"
    RCA_GENERATE = "rca_generate"

    DASHBOARD_GENERATE = "dashboard_generate"
    RUNBOOK_EXECUTE = "runbook_execute"

    GENERAL_CHAT = "general_chat"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    REJECTION = "rejection"


@dataclass
class IntentResult:
    """Result of intent detection."""
    intent: IntentCategory
    confidence: float
    reasoning: str
    sub_intent: Optional[str] = None


INTENT_DETECTION_SYSTEM_PROMPT = """You are an intent classification engine for an enterprise IT operations platform.

Your job is to understand the SEMANTIC MEANING of the user's message and classify it into one of the defined intents.

DO NOT use keyword matching. Understand what the user MEANS, including:
- Synonyms ("power on" = "start", "bring online" = "start", "boot up" = "start")
- Colloquial language ("kill the machine" = "stop VM", "nuke the server" = "stop VM")
- Implicit intent ("the server is down, can you help?" might be vm_status or incident_create depending on context)
- Follow-up context (if the conversation is about a VM, "do it" might mean the previously discussed action)

Available intents:
- vm_start: Starting/powering on/booting a virtual machine or server
- vm_stop: Stopping/shutting down/powering off/killing a virtual machine
- vm_restart: Restarting/rebooting a virtual machine
- vm_resize: Changing the size/SKU/capacity of a virtual machine
- vm_deallocate: Deallocating a virtual machine (releasing compute)
- vm_status: Checking status/state of a virtual machine
- vm_list: Listing virtual machines
- vm_metrics: Getting performance metrics for a VM (CPU, memory, disk)
- incident_create: Creating/opening/raising/logging an incident or issue
- incident_update: Updating an existing incident (adding notes, changing fields)
- incident_close: Closing/resolving an incident
- incident_assign: Assigning/reassigning an incident to someone
- incident_query: Querying/searching/listing incidents
- ritm_create: Creating a requested item (RITM) in ServiceNow
- service_request_create: Creating a service request
- ticket_query: General ticket search/query
- email_read: Reading/checking emails
- email_send: Sending/composing an email
- email_reply: Replying to an email
- email_forward: Forwarding an email
- email_search: Searching emails
- slack_send: Sending a Slack message
- slack_thread_reply: Replying in a Slack thread
- slack_broadcast: Broadcasting a message to a channel
- monitor_alerts: Checking/listing monitoring alerts
- monitor_metrics: Checking resource metrics (CPU, memory, etc.)
- resource_health: Checking Azure resource health
- cost_analysis: Analyzing Azure costs
- log_analysis: Analyzing logs (application, system, container)
- alert_analysis: Analyzing a specific alert (severity, impact, root cause)
- rca_generate: Generating a root cause analysis
- dashboard_generate: Creating/generating a dashboard or report
- runbook_execute: Executing a runbook or SOP
- general_chat: General conversation, greetings, questions about capabilities
- clarification: User is providing clarification to a previous question
- confirmation: User is confirming a proposed action (yes, go ahead, do it, confirmed)
- rejection: User is rejecting a proposed action (no, cancel, stop, don't)

Respond with JSON:
{
  "intent": "<intent_name>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation of why this intent was chosen>",
  "sub_intent": "<optional more specific sub-intent>"
}"""


class IntentDetector:
    """Detects user intent using Gemma 4 semantic understanding."""

    def __init__(self, gemma: GemmaClient):
        self._gemma = gemma

    def detect(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> IntentResult:
        """Detect the intent of a user message, optionally using conversation history."""
        messages = [{"role": "system", "content": INTENT_DETECTION_SYSTEM_PROMPT}]

        # Include recent conversation context for better intent resolution
        if conversation_history:
            # Only include the last 6 turns for context
            recent = conversation_history[-6:]
            for turn in recent:
                messages.append(turn)

        messages.append({"role": "user", "content": user_message})

        result = self._gemma.chat_json(messages, temperature=0.1)

        intent_str = result.get("intent", "general_chat")
        try:
            intent = IntentCategory(intent_str)
        except ValueError:
            logger.warning("Unknown intent '%s', falling back to general_chat", intent_str)
            intent = IntentCategory.GENERAL_CHAT

        return IntentResult(
            intent=intent,
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", ""),
            sub_intent=result.get("sub_intent"),
        )
