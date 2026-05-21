from typing import Any, Literal

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
ProviderType = Literal["local", "remote", "mock"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    stage: str | None = None
    retryable: bool = False


class ErrorResponse(BaseModel):
    trace_id: str | None = None
    error: ErrorDetail


class ProviderInfo(BaseModel):
    name: str
    type: ProviderType
    models: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    audio_formats: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobCreatedResponse(BaseModel):
    trace_id: str
    job_id: str
    status: JobStatus
    created_at: str
