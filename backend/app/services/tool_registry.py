"""Built-in tool registry for LLM function calling."""

from __future__ import annotations

import ast
import json
import logging
import math
import operator
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


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_MATH_FUNCS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "min": min,
    "max": max,
    "sum": sum,
}


def _eval_expression(node: ast.AST) -> float:
    """Safe evaluator: numbers, arithmetic operators and whitelisted functions."""
    if isinstance(node, ast.Expression):
        node = node.body
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_expression(node.left), _eval_expression(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_expression(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _MATH_FUNCS
        and not node.keywords
    ):
        return _MATH_FUNCS[node.func.id](*[_eval_expression(arg) for arg in node.args])
    raise ValueError("unsupported expression")


def _calculate(expression: str = "1+1", **_: Any) -> str:
    """Safe arithmetic evaluation for the voice assistant."""
    try:
        value = _eval_expression(ast.parse(expression, mode="eval"))
        if isinstance(value, float) and not math.isfinite(value):
            return json.dumps({"error": "result is not finite"})
        if abs(value) > 1e15:
            return json.dumps({"error": "result is too large"})
        return json.dumps({"expression": expression, "result": value})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"invalid expression: {exc}"})


# In-memory todo list (process-local; resets on restart).
_todos: list[str] = []


def _add_todo(item: str = "", **_: Any) -> str:
    if not item.strip():
        return json.dumps({"error": "item is required"})
    _todos.append(item.strip())
    return json.dumps({"added": item.strip(), "count": len(_todos)})


def _list_todos(**_: Any) -> str:
    return json.dumps({"todos": _todos, "count": len(_todos)})


def _clear_todos(**_: Any) -> str:
    cleared = len(_todos)
    _todos.clear()
    return json.dumps({"cleared": cleared})


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

tool_registry.register(
    "calculate",
    {
        "description": "计算数学表达式，例如 \"(3+5)*2\" 或 \"sqrt(16)\"。支持 + - * / ** % 和 abs/round/sqrt/min/max/sum",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "要计算的数学表达式"}},
            "required": ["expression"],
        },
    },
    _calculate,
)

tool_registry.register(
    "add_todo",
    {
        "description": "添加一条待办事项",
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string", "description": "待办事项内容"}},
            "required": ["item"],
        },
    },
    _add_todo,
)

tool_registry.register(
    "list_todos",
    {
        "description": "列出当前所有待办事项",
        "parameters": {"type": "object", "properties": {}},
    },
    _list_todos,
)

tool_registry.register(
    "clear_todos",
    {
        "description": "清空所有待办事项",
        "parameters": {"type": "object", "properties": {}},
    },
    _clear_todos,
)
