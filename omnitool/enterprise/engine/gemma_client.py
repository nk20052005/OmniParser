"""
Gemma 4 LLM client for all NLU operations.

Uses OpenAI-compatible API so Gemma can be served via vLLM, Ollama,
TGI, or any compatible server.
"""

import json
import logging
from typing import Any, Optional

from openai import OpenAI

from ..config import GemmaConfig, get_config

logger = logging.getLogger(__name__)


class GemmaClient:
    """Client for interacting with a Gemma 4 model endpoint."""

    def __init__(self, config: Optional[GemmaConfig] = None):
        cfg = config or get_config().gemma
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self._model = cfg.model
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a chat completion request and return the text response."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature or self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception:
            logger.exception("Gemma API call failed")
            raise

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
    ) -> dict:
        """Send a chat request expecting a JSON response. Parses and returns dict."""
        raw = self.chat(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
