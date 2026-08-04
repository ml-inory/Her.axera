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
        assert "calculate" in tool_registry.names
        assert "add_todo" in tool_registry.names
        assert "list_todos" in tool_registry.names


class TestCalculate:
    def test_arithmetic(self) -> None:
        import json

        result = json.loads(tool_registry.execute("calculate", '{"expression": "(3+5)*2"}'))
        assert result["result"] == 16

    def test_power_and_sqrt(self) -> None:
        import json

        assert json.loads(tool_registry.execute("calculate", '{"expression": "2**10"}'))["result"] == 1024
        assert json.loads(tool_registry.execute("calculate", '{"expression": "sqrt(16)"}'))["result"] == 4.0

    def test_invalid_expression(self) -> None:
        import json

        assert "error" in json.loads(tool_registry.execute("calculate", '{"expression": "1+"}'))

    def test_division_by_zero(self) -> None:
        import json

        assert "error" in json.loads(tool_registry.execute("calculate", '{"expression": "1/0"}'))

    def test_injection_rejected(self) -> None:
        import json

        result = json.loads(
            tool_registry.execute(
                "calculate",
                '{"expression": "__import__(\\"os\\").system(\\"echo hi\\")"}',
            )
        )
        assert "error" in result


class TestTodos:
    def setup_method(self) -> None:
        tool_registry.execute("clear_todos", "{}")

    def test_add_and_list(self) -> None:
        import json

        tool_registry.execute("add_todo", '{"item": "买牛奶"}')
        result = json.loads(tool_registry.execute("list_todos", "{}"))
        assert "买牛奶" in result["todos"]
        assert result["count"] == 1

    def test_clear(self) -> None:
        import json

        tool_registry.execute("add_todo", '{"item": "x"}')
        result = json.loads(tool_registry.execute("clear_todos", "{}"))
        assert result["cleared"] == 1


class TestToolLoop:
    def test_dialogue_executes_tools_then_speaks_final_text(self, monkeypatch) -> None:
        import asyncio

        from app.core.config import get_settings
        from app.models.llm import LLMStreamChunk, ToolCall, ToolCallFunction
        from app.services.dialogue_service import dialogue_service
        from app.services.llm_service import llm_service

        state = {"called": False}

        async def fake_chat_stream_detailed(trace_id, request):
            if request.tools and not state["called"]:
                state["called"] = True
                yield LLMStreamChunk(text="让我查一下")
                yield LLMStreamChunk(
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=ToolCallFunction(name="get_weather", arguments='{"city": "上海"}'),
                        )
                    ],
                )
            else:
                yield LLMStreamChunk(text="上海今天晴，22度。")
                yield LLMStreamChunk(finish_reason="stop")

        monkeypatch.setattr(llm_service, "chat_stream_detailed", fake_chat_stream_detailed)
        settings = get_settings()
        original = settings.enable_function_calling
        object.__setattr__(settings, "enable_function_calling", True)
        session_id = "tool-loop-test"
        llm_service.sessions.pop(session_id, None)
        try:
            async def run():
                events = []
                async for event in dialogue_service.stream_text_pipeline(
                    trace_id="tool-trace",
                    text="上海天气怎么样？",
                    session_id=session_id,
                    user_id=None,
                    language="zh-CN",
                    llm_provider="mock_llm",
                    llm_model=None,
                    llm_api_key=None,
                    tts_provider="mock_tts",
                    tts_model=None,
                    voice=None,
                    output_audio_format="wav",
                    sample_rate=24000,
                    system_prompt=None,
                ):
                    events.append(event)
                return events

            events = asyncio.run(run())
        finally:
            object.__setattr__(settings, "enable_function_calling", original)

        types = [event["type"] for event in events]
        assert "tool_call" in types
        tool_events = [event for event in events if event["type"] == "tool_call"]
        assert tool_events[0]["name"] == "get_weather"
        import json

        assert json.loads(tool_events[0]["result"])["city"] == "上海"
        assert "tts_sentence" in types
        llm_event = next(event for event in events if event["type"] == "llm")
        assert "上海今天晴" in llm_event["text"]

        roles = [message.role for message in llm_service.sessions.get(session_id, [])]
        assert roles == ["user", "assistant", "tool", "assistant"]
        llm_service.sessions.pop(session_id, None)

    def test_dialogue_loop_with_calculate(self, monkeypatch) -> None:
        import asyncio

        from app.core.config import get_settings
        from app.models.llm import LLMStreamChunk, ToolCall, ToolCallFunction
        from app.services.dialogue_service import dialogue_service
        from app.services.llm_service import llm_service

        state = {"called": False}

        async def fake_chat_stream_detailed(trace_id, request):
            if request.tools and not state["called"]:
                state["called"] = True
                yield LLMStreamChunk(
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_calc",
                            function=ToolCallFunction(name="calculate", arguments='{"expression": "6*7"}'),
                        )
                    ],
                )
            else:
                yield LLMStreamChunk(text="结果是42。")
                yield LLMStreamChunk(finish_reason="stop")

        monkeypatch.setattr(llm_service, "chat_stream_detailed", fake_chat_stream_detailed)
        settings = get_settings()
        original = settings.enable_function_calling
        object.__setattr__(settings, "enable_function_calling", True)
        session_id = "tool-loop-calc"
        llm_service.sessions.pop(session_id, None)
        try:
            async def run():
                events = []
                async for event in dialogue_service.stream_text_pipeline(
                    trace_id="tool-calc",
                    text="6乘以7等于多少？",
                    session_id=session_id,
                    user_id=None,
                    language="zh-CN",
                    llm_provider="mock_llm",
                    llm_model=None,
                    llm_api_key=None,
                    tts_provider="mock_tts",
                    tts_model=None,
                    voice=None,
                    output_audio_format="wav",
                    sample_rate=24000,
                    system_prompt=None,
                ):
                    events.append(event)
                return events

            events = asyncio.run(run())
        finally:
            object.__setattr__(settings, "enable_function_calling", original)

        tool_events = [event for event in events if event["type"] == "tool_call"]
        assert tool_events and tool_events[0]["name"] == "calculate"
        assert "42" in tool_events[0]["result"]
        llm_event = next(event for event in events if event["type"] == "llm")
        assert "42" in llm_event["text"]
        llm_service.sessions.pop(session_id, None)
