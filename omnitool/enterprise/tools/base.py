"""
Base Tool class for all enterprise tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Standard result from any tool execution."""
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "message": self.message,
        }


class BaseTool(ABC):
    """Abstract base class for all enterprise tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name (e.g., 'azure_vm_start')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        ...

    @property
    @abstractmethod
    def required_params(self) -> list[str]:
        """List of required parameter names."""
        ...

    @property
    def optional_params(self) -> list[str]:
        """List of optional parameter names."""
        return []

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        ...

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, str]:
        """Validate that all required parameters are present."""
        missing = [p for p in self.required_params if p not in params or params[p] is None]
        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"
        return True, ""
