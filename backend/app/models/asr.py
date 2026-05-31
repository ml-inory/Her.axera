from typing import Any

from pydantic import BaseModel, Field

from app.models.common import JobStatus, ProviderInfo


class ASRSegment(BaseModel):
    index: int
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None
    speaker: str | None = None


class VADSegment(BaseModel):
    index: int
    start_ms: int
    end_ms: int


class ASRResult(BaseModel):
    trace_id: str
    provider: str
    model: str
    language: str
    text: str
    confidence: float
    duration_ms: int
    processing_ms: int
    segments: list[ASRSegment] = Field(default_factory=list)
    words: list[dict[str, Any]] = Field(default_factory=list)
    vad_segments: list[VADSegment] = Field(default_factory=list)
    speech_duration_ms: int | None = None
    vad_processing_ms: int | None = None


class ASRJobResponse(BaseModel):
    trace_id: str
    job_id: str
    status: JobStatus
    result: ASRResult | None = None


class ASRProvidersResponse(BaseModel):
    trace_id: str
    providers: list[ProviderInfo]


class VADDetectionResponse(BaseModel):
    trace_id: str
    segments: list[VADSegment] = Field(default_factory=list)
    speech_duration_ms: int
    processing_ms: int
