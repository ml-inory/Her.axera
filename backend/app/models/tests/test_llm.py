from app.models.llm import (
    ChatMessage, ChatCompletionRequest, ChatCompletionResponse,
    TokenUsage, SafetyResult, SessionInfo, SessionListResponse,
)


class TestChatMessage:
    def test_basic(self) -> None:
        m = ChatMessage(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"
        assert m.content_type == "text"
        assert m.text_content == "hello"

    def test_multimodal_content(self) -> None:
        content = [{"type": "text", "text": "what is this?"}]
        m = ChatMessage(role="user", content=content, content_type="image")
        assert m.content_type == "image"
        assert m.text_content == "what is this?"

    def test_system_role(self) -> None:
        m = ChatMessage(role="system", content="be concise")
        assert m.role == "system"

    def test_assistant_role(self) -> None:
        m = ChatMessage(role="assistant", content="ok")
        assert m.role == "assistant"


class TestChatCompletionRequest:
    def test_minimal(self) -> None:
        r = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        assert len(r.messages) == 1
        assert r.temperature == 0.7
        assert r.max_tokens == 512

    def test_with_session(self) -> None:
        r = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            session_id="s1", user_id="u1",
        )
        assert r.session_id == "s1"
        assert r.user_id == "u1"


class TestTokenUsage:
    def test_fields(self) -> None:
        u = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 20
        assert u.total_tokens == 30


class TestSafetyResult:
    def test_default(self) -> None:
        s = SafetyResult()
        assert s.blocked is False
        assert s.categories == []

    def test_blocked(self) -> None:
        s = SafetyResult(blocked=True, categories=["hate"])
        assert s.blocked is True
        assert s.categories == ["hate"]


class TestChatCompletionResponse:
    def test_minimal(self) -> None:
        r = ChatCompletionResponse(
            trace_id="t1", provider="mock", model="mock_llm",
            message=ChatMessage(role="assistant", content="ok"),
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            processing_ms=100,
        )
        assert r.trace_id == "t1"
        assert r.provider == "mock"
        assert r.finish_reason == "stop"
        assert r.message.text_content == "ok"


class TestSessionInfo:
    def test_fields(self) -> None:
        s = SessionInfo(
            session_id="s1", title="test session",
            message_count=5, created_at="now", last_active="later",
        )
        assert s.session_id == "s1"
        assert s.title == "test session"
        assert s.message_count == 5


class TestSessionListResponse:
    def test_fields(self) -> None:
        r = SessionListResponse(trace_id="t1", sessions=[])
        assert r.trace_id == "t1"
        assert r.sessions == []
