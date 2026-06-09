"""
Multi-Agent Router.

Routes requests to specialized agents based on detected intent domain.
Each agent handles a specific domain (CloudOps, ServiceNow, Email, RCA, Mission Control).
The OmniParser agent is used as a fallback when APIs are unavailable.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..engine.intent import IntentCategory

logger = logging.getLogger(__name__)


class AgentDomain(str, Enum):
    CLOUDOPS = "cloudops"
    SERVICENOW = "servicenow"
    EMAIL = "email"
    SLACK = "slack"
    RCA = "rca"
    MISSION_CONTROL = "mission_control"
    OMNIPARSER = "omniparser"
    GENERAL = "general"


@dataclass
class AgentInfo:
    """Metadata about a specialized agent."""
    domain: AgentDomain
    name: str
    description: str
    handles_intents: set[IntentCategory]


# Define which agent handles which intents
AGENT_DEFINITIONS: list[AgentInfo] = [
    AgentInfo(
        domain=AgentDomain.CLOUDOPS,
        name="CloudOps Agent",
        description="Manages Azure resources, VMs, monitoring, and infrastructure",
        handles_intents={
            IntentCategory.VM_START, IntentCategory.VM_STOP, IntentCategory.VM_RESTART,
            IntentCategory.VM_RESIZE, IntentCategory.VM_DEALLOCATE, IntentCategory.VM_STATUS,
            IntentCategory.VM_LIST, IntentCategory.VM_METRICS,
            IntentCategory.MONITOR_ALERTS, IntentCategory.MONITOR_METRICS,
            IntentCategory.RESOURCE_HEALTH, IntentCategory.COST_ANALYSIS,
        },
    ),
    AgentInfo(
        domain=AgentDomain.SERVICENOW,
        name="ServiceNow Agent",
        description="Handles incidents, RITMs, and service requests",
        handles_intents={
            IntentCategory.INCIDENT_CREATE, IntentCategory.INCIDENT_UPDATE,
            IntentCategory.INCIDENT_CLOSE, IntentCategory.INCIDENT_ASSIGN,
            IntentCategory.INCIDENT_QUERY,
            IntentCategory.RITM_CREATE, IntentCategory.SERVICE_REQUEST_CREATE,
            IntentCategory.TICKET_QUERY,
        },
    ),
    AgentInfo(
        domain=AgentDomain.EMAIL,
        name="Email Agent",
        description="Manages email operations",
        handles_intents={
            IntentCategory.EMAIL_READ, IntentCategory.EMAIL_SEND,
            IntentCategory.EMAIL_REPLY, IntentCategory.EMAIL_FORWARD,
            IntentCategory.EMAIL_SEARCH,
        },
    ),
    AgentInfo(
        domain=AgentDomain.SLACK,
        name="Slack Agent",
        description="Handles Slack messaging operations",
        handles_intents={
            IntentCategory.SLACK_SEND, IntentCategory.SLACK_THREAD_REPLY,
            IntentCategory.SLACK_BROADCAST,
        },
    ),
    AgentInfo(
        domain=AgentDomain.RCA,
        name="RCA Agent",
        description="Handles diagnostics, log analysis, and root cause analysis",
        handles_intents={
            IntentCategory.LOG_ANALYSIS, IntentCategory.ALERT_ANALYSIS,
            IntentCategory.RCA_GENERATE,
        },
    ),
    AgentInfo(
        domain=AgentDomain.MISSION_CONTROL,
        name="Mission Control",
        description="Generates dashboards, reports, and executes runbooks",
        handles_intents={
            IntentCategory.DASHBOARD_GENERATE, IntentCategory.RUNBOOK_EXECUTE,
        },
    ),
]

# Build lookup table
_INTENT_TO_AGENT: dict[IntentCategory, AgentInfo] = {}
for agent_def in AGENT_DEFINITIONS:
    for intent in agent_def.handles_intents:
        _INTENT_TO_AGENT[intent] = agent_def


class AgentRouter:
    """Routes incoming requests to the appropriate specialized agent."""

    def route(self, intent: IntentCategory) -> AgentInfo:
        """Determine which agent should handle a given intent."""
        agent = _INTENT_TO_AGENT.get(intent)
        if agent:
            logger.info("Routing intent %s to %s", intent.value, agent.name)
            return agent

        # Default to general
        logger.info("No specific agent for intent %s, using general", intent.value)
        return AgentInfo(
            domain=AgentDomain.GENERAL,
            name="General Agent",
            description="Handles general conversation and unrecognized intents",
            handles_intents={IntentCategory.GENERAL_CHAT},
        )

    def should_use_omniparser(self, intent: IntentCategory, api_available: bool) -> bool:
        """
        Determine if OmniParser should be used instead of API.
        OmniParser is used only when:
        - API is unavailable
        - Portal access is required
        - Legacy applications need GUI automation
        """
        if api_available:
            return False
        # If API is not available, fall back to OmniParser for applicable intents
        return intent in _INTENT_TO_AGENT

    def get_all_agents(self) -> list[AgentInfo]:
        """Get information about all registered agents."""
        return AGENT_DEFINITIONS

    def get_agent_for_domain(self, domain: AgentDomain) -> Optional[AgentInfo]:
        """Get agent info for a specific domain."""
        for agent in AGENT_DEFINITIONS:
            if agent.domain == domain:
                return agent
        return None
