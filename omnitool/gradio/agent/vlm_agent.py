import json
from collections.abc import Callable
from typing import cast, Callable
import uuid
import os
from PIL import Image, ImageDraw
import base64
from io import BytesIO

from anthropic import APIResponse
from anthropic.types import ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam, BetaUsage

from agent.llm_utils.oaiclient import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
from agent.llm_utils.utils import is_image_path
import time
import re

VALID_NEXT_ACTIONS = {
    "key",
    "type",
    "left_click",
    "right_click",
    "double_click",
    "hover",
    "scroll_up",
    "scroll_down",
    "wait",
    "none",
}


def normalize_next_action(raw_action: str) -> str:
    """Normalize model output to one supported action token."""
    if raw_action is None:
        return "None"

    action = str(raw_action).strip()
    if not action:
        return "None"

    lowered = action.lower().strip()
    if lowered in VALID_NEXT_ACTIONS:
        return "None" if lowered == "none" else lowered

    # Handle common verbose outputs like: "left_click, click the send button"
    first_token = re.split(r"[,;:\\n]|\\s+", lowered, maxsplit=1)[0].strip()
    if first_token in VALID_NEXT_ACTIONS:
        return "None" if first_token == "none" else first_token

    # Fallback: search any known action keyword in the string.
    for candidate in sorted(VALID_NEXT_ACTIONS, key=len, reverse=True):
        if candidate in lowered:
            return "None" if candidate == "none" else candidate

    return "None"


def enforce_safe_next_action(vlm_response_json: dict, parsed_screen: dict) -> dict:
    normalized_action = normalize_next_action(vlm_response_json.get("Next Action", "None"))
    vlm_response_json["Next Action"] = normalized_action

    # Keep Box ID as an integer when present so executor parsing stays stable.
    if "Box ID" in vlm_response_json:
        try:
            vlm_response_json["Box ID"] = int(vlm_response_json["Box ID"])
        except (TypeError, ValueError):
            vlm_response_json.pop("Box ID", None)

    return vlm_response_json

OUTPUT_DIR = "./tmp/outputs"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

def extract_data(input_string, data_type):
    # Regular expression to extract content starting from '```python' until the end if there are no closing backticks
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    # Extract content
    # re.DOTALL allows '.' to match newlines as well
    matches = re.findall(pattern, input_string, re.DOTALL)
    # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
    return matches[0][0].strip() if matches else input_string

class VLMAgent:
    def __init__(
        self,
        model: str, 
        provider: str, 
        api_key: str,
        output_callback: Callable, 
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
    ):
        if model == "omniparser + gpt-4.1-mini":
            self.model = "gpt-4.1-mini"
        elif model == "omniparser + gpt-4o":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1":
            self.model = "deepseek-r1-distill-llama-70b"
        elif model == "omniparser + qwen2.5vl":
            self.model = "qwen2.5-vl-72b-instruct"
        elif model == "omniparser + o1":
            self.model = "o1"
        elif model == "omniparser + o3-mini":
            self.model = "o3-mini"
        else:
            raise ValueError(f"Model {model} not supported")
        

        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0

        self.system = ''
           
    def __call__(self, messages: list, parsed_screen: list[str, list, dict]):
        self.step_count += 1
        image_base64 = parsed_screen['original_screenshot_base64']
        latency_omniparser = parsed_screen['latency']
        self.output_callback(f'-- Step {self.step_count}: --', sender="bot")
        screen_info = str(parsed_screen['screen_info'])
        screenshot_uuid = parsed_screen['screenshot_uuid']
        screen_width, screen_height = parsed_screen['width'], parsed_screen['height']

        boxids_and_labels = parsed_screen["screen_info"]
        system = self._get_system_prompt(boxids_and_labels)

        # drop looping actions msg, byte image etc
        planner_messages = messages
        _trim_messages_to_n_most_recent_turns(planner_messages, turns_to_keep=4)
        _strip_reasoning_from_history(planner_messages)
        _remove_som_images(planner_messages)
        _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)

        if isinstance(planner_messages[-1], dict):
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            # Send only the SOM image — it already has numbered boxes drawn on it
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_som_{screenshot_uuid}.png")

        start = time.time()
        if "gpt" in self.model or "o1" in self.model or "o3-mini" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                provider_base_url=OPENAI_BASE_URL,
                temperature=0,
            )
            print(f"oai token usage: {token_usage}")
            self.total_token_usage += token_usage
            if 'gpt' in self.model:
                self.total_cost += (token_usage * 2.5 / 1000000)  # https://openai.com/api/pricing/
            elif 'o1' in self.model:
                self.total_cost += (token_usage * 15 / 1000000)  # https://openai.com/api/pricing/
            elif 'o3-mini' in self.model:
                self.total_cost += (token_usage * 1.1 / 1000000)  # https://openai.com/api/pricing/
        elif "r1" in self.model:
            vlm_response, token_usage = run_groq_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
            )
            print(f"groq token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.99 / 1000000)
        elif "qwen" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=min(2048, self.max_tokens),
                provider_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0,
            )
            print(f"qwen token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 2.2 / 1000000)  # https://help.aliyun.com/zh/model-studio/getting-started/models?spm=a2c4g.11186623.0.0.74b04823CGnPv7#fe96cfb1a422a
        else:
            raise ValueError(f"Model {self.model} not supported")
        latency_vlm = time.time() - start
        self.output_callback(f"LLM: {latency_vlm:.2f}s, OmniParser: {latency_omniparser:.2f}s", sender="bot")

        print(f"{vlm_response}")
        
        if self.print_usage:
            print(f"Total token so far: {self.total_token_usage}. Total cost so far: $USD{self.total_cost:.5f}")
        
        vlm_response_json = extract_data(vlm_response, "json")
        vlm_response_json = json.loads(vlm_response_json)
        vlm_response_json = enforce_safe_next_action(vlm_response_json, parsed_screen)

        img_to_show_base64 = parsed_screen["som_image_base64"]
        if "Box ID" in vlm_response_json:
            try:
                bbox = parsed_screen["parsed_content_list"][int(vlm_response_json["Box ID"])]["bbox"]
                vlm_response_json["box_centroid_coordinate"] = [int((bbox[0] + bbox[2]) / 2 * screen_width), int((bbox[1] + bbox[3]) / 2 * screen_height)]
                img_to_show_data = base64.b64decode(img_to_show_base64)
                img_to_show = Image.open(BytesIO(img_to_show_data))

                draw = ImageDraw.Draw(img_to_show)
                x, y = vlm_response_json["box_centroid_coordinate"] 
                radius = 10
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
                draw.ellipse((x - radius*3, y - radius*3, x + radius*3, y + radius*3), fill=None, outline='red', width=2)

                buffered = BytesIO()
                img_to_show.save(buffered, format="PNG")
                img_to_show_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except:
                print(f"Error parsing: {vlm_response_json}")
                pass
        self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">', sender="bot")
        self.output_callback(
                    f'<details>'
                    f'  <summary>Parsed Screen elements by OmniParser</summary>'
                    f'  <pre>{screen_info}</pre>'
                    f'</details>',
                    sender="bot"
                )
        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'

        # construct the response so that anthropicExcutor can execute the tool
        response_content = [BetaTextBlock(text=vlm_plan_str, type='text')]
        if 'box_centroid_coordinate' in vlm_response_json:
            move_cursor_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': 'mouse_move', 'coordinate': vlm_response_json["box_centroid_coordinate"]},
                                            name='computer', type='tool_use')
            response_content.append(move_cursor_block)

        if vlm_response_json["Next Action"] == "None":
            print("Task paused/completed.")
        elif vlm_response_json["Next Action"] == "key":
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': 'key', 'text': vlm_response_json.get("value", "")},
                                        name='computer', type='tool_use')
            response_content.append(sim_content_block)
        elif vlm_response_json["Next Action"] == "type":
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': vlm_response_json["Next Action"], 'text': vlm_response_json.get("value", "")},
                                        name='computer', type='tool_use')
            response_content.append(sim_content_block)
        else:
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': vlm_response_json["Next Action"]},
                                            name='computer', type='tool_use')
            response_content.append(sim_content_block)
        response_message = BetaMessage(id=f'toolu_{uuid.uuid4()}', content=response_content, model='', role='assistant', type='message', stop_reason='tool_use', usage=BetaUsage(input_tokens=0, output_tokens=0))
        return response_message, vlm_response_json

    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)

    def _get_system_prompt(self, screen_info: str = ""):
        main_section = f"""You are a Windows desktop automation agent. Use mouse and keyboard to complete the task.
NEVER interact with the OmniParser control window. On step 1, switch away from it first.

Detected UI elements (Box ID: description):{screen_info}

Actions: key | type | left_click | right_click | double_click | hover | scroll_up | scroll_down | wait | None
- key/type require a "value" field. left_click/right_click/double_click/hover require a "Box ID" field.

Respond with ONLY this JSON:
```json
{{
    "Reasoning": "<one sentence: what you see and why this action>",
    "Next Action": "<action token>",
    "Box ID": n,
    "value": "xxx"
}}
```

Think carefully before acting:
1. LOOK at the screenshot first. Describe what you actually see — window title, active app, visible buttons/fields.
2. VERIFY your target element exists on screen before clicking. Cross-check the Box ID description against what you see in the screenshot. If the description says "icon" but you need a button, find the right element.
3. REFLECT on history. What did you do last? Did it work? If the screen hasn't changed, your last action may have failed — try a different approach.
4. PREFER specific elements. When multiple elements could match, pick the one whose bounding box most precisely covers your intended target. Avoid clicking text labels when you mean to click their adjacent button/icon.

Rules: single action per turn; omit Box ID for scroll/wait/key/type; omit value unless action is key or type; say Next Action None when done; avoid repeating the same action twice in a row.
"""
        return main_section

def _strip_reasoning_from_history(messages: list):
    """
    In older assistant turns, replace the full Reasoning text with a compact
    summary so the model still knows what actions were taken without paying
    full token cost for every past explanation.
    Keep only the last assistant turn intact (the model needs full context for
    the immediately preceding step).
    """
    assistant_indices = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    # Leave the last assistant message untouched
    for idx in assistant_indices[:-1]:
        content = messages[idx].get("content", [])
        if not isinstance(content, list):
            continue
        new_content = []
        for block in content:
            if hasattr(block, "text"):  # BetaTextBlock
                import re
                # Keep only the Next Action line(s), drop verbose Reasoning
                compact = re.sub(r"[^\n]*Reasoning[^\n]*\n?", "", block.text, flags=re.IGNORECASE)
                block = type(block)(text=compact.strip(), type=block.type)
            new_content.append(block)
        messages[idx]["content"] = new_content


def _trim_messages_to_n_most_recent_turns(
    messages: list,
    turns_to_keep: int = 10,
):
    """
    Keep the first message (the original task) and the last `turns_to_keep * 2`
    messages (each turn = 1 assistant message + 1 user/tool-result message).
    This prevents unbounded context growth without losing the task or recent history.
    """
    if len(messages) <= 1:
        return
    max_history = turns_to_keep * 2
    if len(messages) - 1 > max_history:
        messages[1:] = messages[-(max_history):]


def _remove_som_images(messages):
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            msg["content"] = [
                cnt for cnt in msg_content 
                if not (isinstance(cnt, str) and 'som' in cnt and is_image_path(cnt))
            ]


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place
    """
    if images_to_keep is None:
        return messages

    total_images = 0
    for msg in messages:
        for cnt in msg.get("content", []):
            if isinstance(cnt, str) and is_image_path(cnt):
                total_images += 1
            elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                for content in cnt.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        total_images += 1

    images_to_remove = total_images - images_to_keep
    
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            new_content = []
            for cnt in msg_content:
                # Remove images from SOM or screenshot as needed
                if isinstance(cnt, str) and is_image_path(cnt):
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                # VLM shouldn't use anthropic screenshot tool so shouldn't have these but in case it does, remove as needed
                elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                    new_tool_result_content = []
                    for tool_result_entry in cnt.get("content", []):
                        if isinstance(tool_result_entry, dict) and tool_result_entry.get("type") == "image":
                            if images_to_remove > 0:
                                images_to_remove -= 1
                                continue
                        new_tool_result_content.append(tool_result_entry)
                    cnt["content"] = new_tool_result_content
                # Append fixed content to current message's content list
                new_content.append(cnt)
            msg["content"] = new_content