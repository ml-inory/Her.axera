import pytest

from app.models.llm import ChatCompletionRequest, ChatMessage
from app.services.llm_service import llm_service


class TestLLMProviders:
    def test_list_providers_includes_mock_and_deepseek(self) -> None:
        providers = llm_service.list_providers()
        names = [p.name for p in providers]
        assert "mock_llm" in names
        assert "deepseek" in names

    def test_provider_names(self) -> None:
        providers = llm_service.list_providers()
        mock_p = next(p for p in providers if p.name == "mock_llm")
        assert "mock-llm" in mock_p.models


class TestMockChatCompletion:
    def _req(self, text: str) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=text)],
            provider="mock_llm", model="mock-llm",
        )

    def test_mock_chat_basic(self) -> None:
        result = llm_service.chat("t1", self._req("你好"))
        assert result.trace_id == "t1"
        assert result.provider == "mock_llm"
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0

    def test_mock_chat_stream_structure(self) -> None:
        import asyncio
        async def run():
            chunks = []
            async for line in llm_service.chat_stream("t2", self._req("你好")):
                chunks.append(line)
            return chunks
        chunks = asyncio.run(run())
        assert len(chunks) > 0
        # Stream output should be SSE-like json lines
        for chunk in chunks:
            assert isinstance(chunk, str)  # stream returns text lines


class FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200

    def iter_lines(self, decode_unicode: bool = True):
        yield from self._lines

    @property
    def text(self) -> str:
        return ""


class TestDetailedStreamToolCalls:
    def test_aggregates_tool_call_deltas(self, monkeypatch) -> None:
        import asyncio
        import app.services.llm_service as llm_mod
        from app.models.llm import ToolCall, ToolCallFunction

        lines = [
            'data: {"choices":[{"delta":{"role":"assistant","content":"让我查一下"}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"上海\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
        monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: FakeStreamResponse(lines))
        settings = llm_mod.get_settings()
        original_key = settings.deepseek_api_key
        object.__setattr__(settings, "deepseek_api_key", "test-key")
        try:
            async def run():
                request = llm_mod.ChatCompletionRequest(
                    messages=[llm_mod.ChatMessage(role="user", content="上海天气？")],
                    provider="deepseek",
                    model="deepseek-chat",
                    tools=[{"type": "function"}],
                )
                text_parts = []
                tool_calls = []
                finish = None
                async for chunk in llm_mod.llm_service.chat_stream_detailed("t", request):
                    text_parts.append(chunk.text)
                    tool_calls.extend(chunk.tool_calls)
                    if chunk.finish_reason:
                        finish = chunk.finish_reason
                return text_parts, tool_calls, finish

            text_parts, tool_calls, finish = asyncio.run(run())
        finally:
            object.__setattr__(settings, "deepseek_api_key", original_key)

        assert "".join(text_parts) == "让我查一下"
        assert finish == "tool_calls"
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ToolCall)
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].function.arguments == '{"city":"上海"}'


class TestSessionManagement:
    def test_list_sessions(self) -> None:
        sessions = llm_service.list_sessions()
        assert isinstance(sessions, list)

    def test_delete_session_returns_bool(self) -> None:
        assert llm_service.delete_session("nonexistent_xyz") is True  # always True even for nonexistent

    def test_get_session_messages(self) -> None:
        resp = llm_service.get_session_messages("t1", "nonexistent_sess")
        assert resp.trace_id == "t1"
        assert resp.messages == []

    def test_delete_session_messages(self) -> None:
        resp = llm_service.delete_session_messages("t1", "any_session")
        assert resp.trace_id == "t1"
        assert resp.deleted is True


class TestJobManagement:
    def test_create_and_get_job(self) -> None:
        job = llm_service.create_job("t_job")
        assert job.status == "queued"
        fetched = llm_service.get_job("t2", job.job_id)
        assert fetched.job_id == job.job_id

    def test_cancel_job(self) -> None:
        job = llm_service.create_job("t_cxl")
        cancelled = llm_service.cancel_job("t3", job.job_id)
        assert cancelled.status == "cancelled"
