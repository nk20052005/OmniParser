"""
Response Generation using Gemma 4.

Generates natural conversational responses after tool execution.
"""

import logging
from typing import Any, Optional

from .gemma_client import GemmaClient
from .intent import IntentCategory

logger = logging.getLogger(__name__)


RESPONSE_SYSTEM_PROMPT = """You are a friendly, professional enterprise IT operations assistant.

Generate natural conversational responses based on tool execution results.

Rules:
- Be concise but informative
- If the action succeeded, confirm what was done
- If the action failed, explain the failure clearly and suggest next steps
- Never expose internal system details, secrets, or credentials
- Never claim an action succeeded if it failed
- Format data clearly when presenting lists or metrics
- Use markdown formatting for better readability
- If results contain tables or lists, format them appropriately
- Always be honest about what happened"""


class ResponseGenerator:
    """Generates natural language responses using Gemma 4."""

    def __init__(self, gemma: GemmaClient):
        self._gemma = gemma

    def generate(
        self,
        intent: IntentCategory,
        tool_name: str,
        tool_result: dict[str, Any],
        parameters: dict[str, Any],
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """Generate a natural response after tool execution."""
        success = tool_result.get("success", False)
        result_data = tool_result.get("data", {})
        error = tool_result.get("error", "")

        messages = [{"role": "system", "content": RESPONSE_SYSTEM_PROMPT}]

        if conversation_history:
            recent = conversation_history[-4:]
            for turn in recent:
                messages.append(turn)

        messages.append({
            "role": "user",
            "content": (
                f"Action performed: {intent.value}\n"
                f"Tool used: {tool_name}\n"
                f"Parameters: {parameters}\n"
                f"Success: {success}\n"
                f"Result data: {result_data}\n"
                f"Error: {error}\n\n"
                f"Generate a natural response to the user about this result."
            ),
        })

        return self._gemma.chat(messages, temperature=0.5, max_tokens=500).strip()

    def generate_approval_request(
        self,
        intent: IntentCategory,
        parameters: dict[str, Any],
    ) -> str:
        """Generate a confirmation prompt for dangerous actions."""
        messages = [
            {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"I need to ask the user for confirmation before executing a potentially "
                    f"dangerous action.\n\n"
                    f"Action: {intent.value}\n"
                    f"Parameters: {parameters}\n\n"
                    f"Generate a clear confirmation request that explains what will happen "
                    f"and asks for explicit approval. Include any risks."
                ),
            },
        ]
        return self._gemma.chat(messages, temperature=0.3, max_tokens=300).strip()

    def generate_error_response(self, error_message: str) -> str:
        """Generate a user-friendly error response."""
        messages = [
            {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"An error occurred: {error_message}\n\n"
                    f"Generate a helpful response explaining the issue and suggesting next steps."
                ),
            },
        ]
        return self._gemma.chat(messages, temperature=0.3, max_tokens=200).strip()

    def generate_general_response(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """Generate a response for general chat (non-tool) messages."""
        system = (
            "You are a friendly enterprise IT operations assistant. "
            "You help with Azure VM management, ServiceNow incidents, email, "
            "monitoring, log analysis, RCA generation, and more. "
            "If asked what you can do, explain your capabilities. "
            "Be conversational and helpful."
        )
        messages = [{"role": "system", "content": system}]

        if conversation_history:
            recent = conversation_history[-6:]
            for turn in recent:
                messages.append(turn)

        messages.append({"role": "user", "content": user_message})
        return self._gemma.chat(messages, temperature=0.7, max_tokens=500).strip()
