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
