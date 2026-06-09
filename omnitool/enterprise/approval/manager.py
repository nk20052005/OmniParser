"""
Human Approval Manager.

Determines which actions require explicit human approval before execution.
Manages approval state for dangerous operations.
"""

import logging
import re
from enum import Enum
from typing import Any

from ..engine.intent import IntentCategory

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


# Intents that ALWAYS require approval
ALWAYS_REQUIRE_APPROVAL: set[IntentCategory] = {
    IntentCategory.RUNBOOK_EXECUTE,
}

# Intents that require approval when targeting production resources
PRODUCTION_APPROVAL: set[IntentCategory] = {
    IntentCategory.VM_STOP,
    IntentCategory.VM_RESTART,
    IntentCategory.VM_RESIZE,
    IntentCategory.VM_DEALLOCATE,
    IntentCategory.INCIDENT_CLOSE,
}

# Intents that require approval for mass operations
MASS_OPERATION_APPROVAL: set[IntentCategory] = {
    IntentCategory.EMAIL_SEND,
    IntentCategory.SLACK_BROADCAST,
}

# Patterns that indicate production resources
PRODUCTION_PATTERNS = [
    re.compile(r"prod", re.IGNORECASE),
    re.compile(r"prd", re.IGNORECASE),
    re.compile(r"production", re.IGNORECASE),
    re.compile(r"live", re.IGNORECASE),
    re.compile(r"master", re.IGNORECASE),
]


class ApprovalManager:
    """Manages human approval for dangerous operations."""

    def requires_approval(
        self,
        intent: IntentCategory,
        parameters: dict[str, Any],
    ) -> bool:
        """Determine if an action requires human approval."""
        # Always-approve intents
        if intent in ALWAYS_REQUIRE_APPROVAL:
            logger.info("Action %s always requires approval", intent.value)
            return True

        # Check if targeting production resources
        if intent in PRODUCTION_APPROVAL:
            if self._is_production_resource(parameters):
                logger.info(
                    "Action %s targets production resource, requires approval",
                    intent.value,
                )
                return True

        # Check mass operations
        if intent in MASS_OPERATION_APPROVAL:
            if self._is_mass_operation(parameters):
                logger.info(
                    "Action %s is a mass operation, requires approval",
                    intent.value,
                )
                return True

        return False

    def _is_production_resource(self, parameters: dict[str, Any]) -> bool:
        """Check if any parameter values indicate a production resource."""
        for value in parameters.values():
            if isinstance(value, str):
                for pattern in PRODUCTION_PATTERNS:
                    if pattern.search(value):
                        return True
        return False

    def _is_mass_operation(self, parameters: dict[str, Any]) -> bool:
        """Check if the operation is a mass/bulk operation."""
        # Multiple recipients in email
        recipients = parameters.get("recipient", "")
        if isinstance(recipients, str) and recipients.count(",") > 2:
            return True

        # Multiple channels in broadcast
        channels = parameters.get("channels", [])
        if isinstance(channels, list) and len(channels) > 1:
            return True

        return False
