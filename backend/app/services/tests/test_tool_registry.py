from app.services.tool_registry import ToolRegistry, tool_registry


class TestToolRegistry:
    def test_empty_registry(self) -> None:
        reg = ToolRegistry()
        assert reg.names == []
        assert reg.get_schemas() == []

    def test_register_and_list(self) -> None:
        reg = ToolRegistry()
        reg.register("echo", {"description": "echoes"}, lambda **kw: kw.get("msg", ""))
        assert reg.names == ["echo"]
        assert len(reg.get_schemas()) == 1
        schema = reg.get_schemas()[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert schema["function"]["description"] == "echoes"

    def test_execute_known_tool(self) -> None:
        reg = ToolRegistry()
        reg.register("add", {"description": "adds"}, lambda a, b: str(a + b))
        import json
        result = json.loads(reg.execute("add", '{"a": 1, "b": 2}'))
        assert result == 3

    def test_execute_unknown_tool(self) -> None:
        reg = ToolRegistry()
        import json
        result = json.loads(reg.execute("nonexistent", "{}"))
        assert "error" in result

    def test_execute_invalid_args(self) -> None:
        reg = ToolRegistry()
        reg.register("boom", {"description": "fails"}, lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        result = reg.execute("boom", "{}")
        assert "error" in result

    def test_register_multiple(self) -> None:
        reg = ToolRegistry()
        reg.register("t1", {"description": "one"}, lambda: "")
        reg.register("t2", {"description": "two"}, lambda: "")
        assert reg.names == ["t1", "t2"]
        assert len(reg.get_schemas()) == 2


class TestBuiltinTools:
    def test_get_current_time(self) -> None:
        import json
        result = json.loads(tool_registry.execute("get_current_time", "{}"))
        assert "time" in result
        assert "timezone" in result
        assert result["timezone"] == "UTC"

    def test_get_weather(self) -> None:
        import json
        result = json.loads(tool_registry.execute("get_weather", '{"city": "上海"}'))
        assert result["city"] == "上海"
        assert "temperature" in result
        assert "condition" in result

    def test_builtin_tools_registered(self) -> None:
        assert "get_current_time" in tool_registry.names
        assert "get_weather" in tool_registry.names
