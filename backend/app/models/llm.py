from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.common import JobStatus, ProviderInfo


Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


class ToolCallFunction(BaseModel):
    name: str
    arguments: str = ""


class ToolCall(BaseModel):
    id: str = ""
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: Role
    content: str | list[dict[str, Any]] = ""
    content_type: str = "text"
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text_content(self) -> str:
        """Return plain text content regardless of format."""
        if isinstance(self.content, str):
            return self.content
        return "".join(item.get("text", "") for item in self.content if item.get("type") == "text")


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    session_id: str | None = None
    user_id: str | None = None
    provider: str | None = None
    api_key: str | None = Field(default=None, repr=False)
    model: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stop: list[str] = Field(default_factory=list)
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    safety: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class SafetyResult(BaseModel):
    blocked: bool = False
    categories: list[str] = Field(default_factory=list)


class ChatCompletionResponse(BaseModel):
    trace_id: str
    session_id: str | None = None
    provider: str
    model: str
    message: ChatMessage
    finish_reason: FinishReason
    usage: TokenUsage
    safety: SafetyResult = Field(default_factory=SafetyResult)
    processing_ms: int


class LLMJobResponse(BaseModel):
    trace_id: str
    job_id: str
    status: JobStatus
    result: ChatCompletionResponse | None = None


class LLMProvidersResponse(BaseModel):
    trace_id: str
    providers: list[ProviderInfo]


class SessionMessagesResponse(BaseModel):
    trace_id: str
    session_id: str
    messages: list[ChatMessage]


class DeleteSessionMessagesResponse(BaseModel):
    trace_id: str
    session_id: str
    deleted: bool
