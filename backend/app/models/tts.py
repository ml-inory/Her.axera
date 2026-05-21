from typing import Any

from pydantic import BaseModel, Field

from app.models.common import JobStatus, ProviderInfo


class SpeechRequest(BaseModel):
    text: str
    provider: str | None = None
    model: str | None = None
    voice: str | None = None
    language: str = "zh-CN"
    audio_format: str = "wav"
    sample_rate: int = 24000
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    emotion: str | None = None
    return_audio_base64: bool = False
    enable_cache: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpeechResponse(BaseModel):
    trace_id: str
    provider: str
    model: str
    voice: str
    language: str
    audio_url: str | None = None
    audio_base64: str | None = None
    audio_format: str
    sample_rate: int
    duration_ms: int
    processing_ms: int
    cache_hit: bool = False


class SpeechSegmentRequest(BaseModel):
    index: int
    text: str


class SegmentedSpeechRequest(BaseModel):
    segments: list[SpeechSegmentRequest]
    provider: str | None = None
    model: str | None = None
    voice: str | None = None
    language: str = "zh-CN"
    audio_format: str = "wav"
    sample_rate: int = 24000


class SpeechSegmentResponse(BaseModel):
    index: int
    text: str
    audio_url: str | None = None
    audio_base64: str | None = None
    duration_ms: int
    processing_ms: int


class SegmentedSpeechResponse(BaseModel):
    trace_id: str
    provider: str
    voice: str
    segments: list[SpeechSegmentResponse]
    total_duration_ms: int
    processing_ms: int


class TTSJobResponse(BaseModel):
    trace_id: str
    job_id: str
    status: JobStatus
    result: SpeechResponse | None = None


class VoiceInfo(BaseModel):
    name: str
    display_name: str
    language: str
    gender: str
    styles: list[str] = Field(default_factory=list)
    sample_rates: list[int] = Field(default_factory=list)


class TTSProvidersResponse(BaseModel):
    trace_id: str
    providers: list[ProviderInfo]


class VoicesResponse(BaseModel):
    trace_id: str
    voices: list[VoiceInfo]
