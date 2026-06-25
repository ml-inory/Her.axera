"""Built-in tool registry for LLM function calling."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, schema: dict[str, Any], handler: Callable[..., str]) -> None:
        self._tools[name] = {"schema": schema, "handler": handler}

    def get_schemas(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": name, **tool["schema"]}}
            for name, tool in self._tools.items()
        ]

    def execute(self, name: str, arguments_json: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            return tool["handler"](**args)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool %s execution failed: %s", name, exc)
            return json.dumps({"error": str(exc)})

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())


# ── Built-in tools ─────────────────────────────────────────────────

def _get_current_time(**_: Any) -> str:
    return json.dumps({"time": datetime.now(timezone.utc).isoformat(), "timezone": "UTC"})


def _get_weather(city: str = "北京", **_: Any) -> str:
    # Mock weather data.
    return json.dumps({"city": city, "temperature": "22°C", "condition": "晴", "humidity": "45%"})


tool_registry = ToolRegistry()

tool_registry.register(
    "get_current_time",
    {
        "description": "获取当前 UTC 时间",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    _get_current_time,
)

tool_registry.register(
    "get_weather",
    {
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名称"}},
            "required": ["city"],
        },
    },
    _get_weather,
)
