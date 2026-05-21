from datetime import datetime
from time import perf_counter
from uuid import uuid4

from app.core.config import get_settings
from app.core.errors import AppError
from app.models.common import JobCreatedResponse, ProviderInfo
from app.models.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeleteSessionMessagesResponse,
    LLMJobResponse,
    SafetyResult,
    SessionMessagesResponse,
    TokenUsage,
)


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.providers = {
            "mock_llm": ProviderInfo(
                name="mock_llm",
                type="mock",
                models=["mock-chat-general"],
                languages=["zh-CN", "en-US"],
                features=["chat", "json_output", "tool_calls"],
                metadata={"max_context_tokens": 32768},
            )
        }
        self.sessions: dict[str, list[ChatMessage]] = {}
        self.jobs: dict[str, LLMJobResponse] = {}

    def list_providers(self) -> list[ProviderInfo]:
        return list(self.providers.values())

    def chat(self, trace_id: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        start = perf_counter()
        provider_name = request.provider or self.settings.default_llm_provider
        provider_info = self.providers.get(provider_name)
        if provider_info is None:
            raise AppError(
                "provider_not_found",
                f"LLM provider {provider_name} is not configured",
                status_code=404,
                stage="llm",
            )
        if not request.messages:
            raise AppError("invalid_request", "messages must not be empty", status_code=400, stage="llm")

        selected_model = request.model or provider_info.models[0]
        latest_user_text = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")
        content = f"收到：{latest_user_text}"
        if request.response_format and request.response_format.get("type") == "json_schema":
            content = '{"message":"收到请求","mock":true}'

        message = ChatMessage(role="assistant", content=content, content_type="json" if content.startswith("{") else "text")
        session_id = request.session_id
        if session_id:
            history = self.sessions.setdefault(session_id, [])
            history.extend(request.messages)
            history.append(message)

        prompt_tokens = sum(max(1, len(msg.content) // 2) for msg in request.messages)
        completion_tokens = max(1, len(message.content) // 2)
        return ChatCompletionResponse(
            trace_id=trace_id,
            session_id=session_id,
            provider=provider_name,
            model=selected_model,
            message=message,
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            safety=SafetyResult(blocked=False, categories=[]),
            processing_ms=int((perf_counter() - start) * 1000),
        )

    def create_job(self, trace_id: str) -> JobCreatedResponse:
        job_id = f"job_llm_{uuid4().hex}"
        response = LLMJobResponse(trace_id=trace_id, job_id=job_id, status="queued")
        self.jobs[job_id] = response
        return JobCreatedResponse(
            trace_id=trace_id,
            job_id=job_id,
            status="queued",
            created_at=datetime.now().astimezone().isoformat(),
        )

    def get_job(self, trace_id: str, job_id: str) -> LLMJobResponse:
        job = self.jobs.get(job_id)
        if job is None:
            raise AppError("job_not_found", f"LLM job {job_id} not found", status_code=404, stage="llm")
        return job.model_copy(update={"trace_id": trace_id})

    def cancel_job(self, trace_id: str, job_id: str) -> LLMJobResponse:
        job = self.get_job(trace_id, job_id)
        cancelled = job.model_copy(update={"status": "cancelled", "trace_id": trace_id})
        self.jobs[job_id] = cancelled
        return cancelled

    def get_session_messages(self, trace_id: str, session_id: str) -> SessionMessagesResponse:
        return SessionMessagesResponse(
            trace_id=trace_id,
            session_id=session_id,
            messages=self.sessions.get(session_id, []),
        )

    def delete_session_messages(self, trace_id: str, session_id: str) -> DeleteSessionMessagesResponse:
        self.sessions.pop(session_id, None)
        return DeleteSessionMessagesResponse(trace_id=trace_id, session_id=session_id, deleted=True)


llm_service = LLMService()
