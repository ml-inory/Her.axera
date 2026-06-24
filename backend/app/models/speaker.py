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
