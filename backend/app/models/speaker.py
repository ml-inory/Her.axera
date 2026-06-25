from pydantic import BaseModel, Field

from app.models.common import ProviderInfo


class SpeakerMatch(BaseModel):
    speaker_id: str
    score: float
    label: str | None = None


class SpeakerIdentifyResponse(BaseModel):
    trace_id: str
    provider: str
    model: str
    speaker_id: str
    confidence: float
    matches: list[SpeakerMatch] = Field(default_factory=list)
    processing_ms: int


class SpeakerProvidersResponse(BaseModel):
    trace_id: str
    providers: list[ProviderInfo]


class SpeakerProfile(BaseModel):
    speaker_id: str
    name: str
    description: str = ""
    created_at: str = ""
    audio_count: int = 0


class SpeakerEnrollResponse(BaseModel):
    trace_id: str
    speaker_id: str
    name: str
    provider: str
    processing_ms: int


class SpeakerListResponse(BaseModel):
    trace_id: str
    speakers: list[SpeakerProfile]


class SpeakerDeleteResponse(BaseModel):
    trace_id: str
    speaker_id: str
    deleted: bool
