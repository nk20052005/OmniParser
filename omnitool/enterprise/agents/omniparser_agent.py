"""
OmniParser Agent - Fallback for GUI automation when APIs are unavailable.

Integrates with the existing OmniParser server for screenshot analysis
and the host control server for desktop automation.
"""

import base64
import logging
from typing import Any, Optional

import requests

from ..config import get_config

logger = logging.getLogger(__name__)


class OmniParserAgent:
    """
    Agent that uses OmniParser for GUI-based automation.

    Workflow:
    1. Take screenshot
    2. Send to OmniParser for UI element detection
    3. Use Gemma to decide action based on parsed elements
    4. Execute action via host control server

    This is a FALLBACK — APIs are always preferred over GUI automation.
    """

    def __init__(
        self,
        omniparser_url: Optional[str] = None,
        host_control_url: Optional[str] = None,
        gemma_client: Any = None,
    ):
        config = get_config()
        self._omniparser_url = (omniparser_url or config.omniparser.server_url).rstrip("/")
        self._host_control_url = (host_control_url or config.host_control_url).rstrip("/")
        self._gemma = gemma_client

    def is_available(self) -> bool:
        """Check if OmniParser and host control servers are reachable."""
        try:
            resp = requests.get(f"{self._host_control_url}/probe", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    async def take_screenshot(self) -> Optional[str]:
        """Take a screenshot via the host control server. Returns base64 image."""
        try:
            resp = requests.get(f"{self._host_control_url}/screenshot", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("screenshot", data.get("image"))
        except Exception:
            logger.exception("Failed to take screenshot")
            return None

    async def parse_screen(self, screenshot_b64: str) -> dict:
        """Send screenshot to OmniParser for UI element detection."""
        try:
            resp = requests.post(
                f"{self._omniparser_url}/parse/",
                json={"base64_image": screenshot_b64},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.exception("Failed to parse screenshot")
            return {}

    async def execute_action(self, action_code: str) -> dict:
        """Execute a pyautogui action via host control server."""
        try:
            resp = requests.post(
                f"{self._host_control_url}/execute",
                json={"action": action_code},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.exception("Failed to execute action")
            return {"success": False, "error": "Execution failed"}

    async def perform_gui_task(self, task_description: str) -> dict:
        """
        Perform a GUI-based task using the OmniParser pipeline.

        1. Screenshot → 2. Parse → 3. Gemma decides action → 4. Execute
        """
        if not self._gemma:
            return {"success": False, "error": "Gemma client not available for GUI automation"}

        # Step 1: Take screenshot
        screenshot = await self.take_screenshot()
        if not screenshot:
            return {"success": False, "error": "Could not capture screenshot"}

        # Step 2: Parse the screen
        parsed = await self.parse_screen(screenshot)
        elements = parsed.get("parsed_content_list", [])

        if not elements:
            return {"success": False, "error": "No UI elements detected on screen"}

        # Step 3: Ask Gemma what action to take
        elements_desc = "\n".join(
            f"[{i}] {el}" for i, el in enumerate(elements)
        )

        action_plan = self._gemma.chat_json([
            {
                "role": "system",
                "content": (
                    "You are a GUI automation agent. Given a task and a list of UI elements "
                    "detected on screen, decide what action to take.\n\n"
                    "Available actions:\n"
                    "- click: Click on an element by its index\n"
                    "- type: Type text into a field\n"
                    "- key: Press a key combination\n"
                    "- scroll: Scroll up or down\n\n"
                    "Respond with JSON:\n"
                    '{"action": "click|type|key|scroll", "element_index": <int>, '
                    '"value": "<text to type or key to press>", '
                    '"reasoning": "<why this action>"}'
                ),
            },
            {
                "role": "user",
                "content": f"Task: {task_description}\n\nUI Elements:\n{elements_desc}",
            },
        ])

        # Step 4: Execute the action
        action = action_plan.get("action", "")
        element_idx = action_plan.get("element_index", 0)

        if action == "click" and element_idx < len(elements):
            # Extract coordinates from the element
            result = await self.execute_action(
                f"import pyautogui; pyautogui.click({element_idx})"
            )
        elif action == "type":
            value = action_plan.get("value", "")
            result = await self.execute_action(
                f"import pyautogui; pyautogui.typewrite({repr(value)})"
            )
        elif action == "key":
            value = action_plan.get("value", "")
            result = await self.execute_action(
                f"import pyautogui; pyautogui.hotkey({repr(value)})"
            )
        else:
            result = {"success": False, "error": f"Unknown action: {action}"}

        return {
            "success": result.get("success", False),
            "action_taken": action_plan,
            "elements_found": len(elements),
        }
