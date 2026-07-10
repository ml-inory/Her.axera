"""OpenAI-compatible LLM client for the SDK."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterator

import requests

logger = logging.getLogger("her_axera_sdk.llm")


@dataclass
class LLMStreamChunk:
    content: str
    finish_reason: str | None = None


class LLMClient:
    """Lightweight OpenAI-compatible chat client."""

    def __init__(
        self,
        api_base: str,
        api_key: str = "",
        model: str = "deepseek-chat",
        timeout: float = 60.0,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Send a chat completion request and return the reply text."""
        payload = self._build_payload(messages, model=model, temperature=temperature, max_tokens=max_tokens, stream=False)

        resp = requests.post(
            f"{self._api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM returned no choices")
        return choices[0].get("message", {}).get("content", "") or ""

    def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Iterator[LLMStreamChunk]:
        """Stream chat completion tokens."""
        payload = self._build_payload(messages, model=model, temperature=temperature, max_tokens=max_tokens, stream=True)

        resp = requests.post(
            f"{self._api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self._timeout,
            stream=True,
        )
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            finish = choices[0].get("finish_reason")
            if content:
                yield LLMStreamChunk(content=content, finish_reason=finish)

    def _build_payload(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> dict:
        return {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
