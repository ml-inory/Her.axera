from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.llm import ChatMessage


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stop: str | list[str] | None = None
    stream: bool = False
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    user: str | None = None
    provider: str | None = None
    api_key: str | None = Field(default=None, repr=False)
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpenAISpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = 1.0
    provider: str | None = None
    language: str = "zh-CN"
    sample_rate: int = 24000
