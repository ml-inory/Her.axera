"""Wake word models for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WakeWordRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="Wake word name, e.g. 'hey_jarvis'")
    audio_base64: str = Field(..., description="Base64-encoded WAV/PCM audio sample (16kHz mono 16-bit)")
    description: str = Field("", description="Optional description")


class WakeWordInfo(BaseModel):
    name: str
    description: str
    created_at: str
    sample_count: int
    active: bool = True


class WakeWordListResponse(BaseModel):
    trace_id: str
    wake_words: list[WakeWordInfo]


class WakeWordRegisterResponse(BaseModel):
    trace_id: str
    name: str
    status: str  # "registered", "updated"
    sample_count: int


class WakeWordDeleteResponse(BaseModel):
    trace_id: str
    name: str
    deleted: bool
