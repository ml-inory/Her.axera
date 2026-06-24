from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import bind_trace_id
from app.models.speaker import SpeakerIdentifyResponse, SpeakerProvidersResponse
from app.services.speaker_service import speaker_service

router = APIRouter(prefix="/speakers", tags=["speakers"])


@router.post("/identify", response_model=SpeakerIdentifyResponse)
async def identify_speaker(
    trace_id: Annotated[str, Depends(bind_trace_id)],
    audio: Annotated[UploadFile, File()],
    provider: Annotated[str | None, Form()] = None,
    top_k: Annotated[int, Form()] = 1,
) -> SpeakerIdentifyResponse:
    content = await audio.read()
    return speaker_service.identify(
        trace_id=trace_id,
        audio_content=content,
        filename=audio.filename,
        provider=provider,
        top_k=top_k,
    )


@router.get("/providers", response_model=SpeakerProvidersResponse)
async def list_speaker_providers(trace_id: Annotated[str, Depends(bind_trace_id)]) -> SpeakerProvidersResponse:
    return SpeakerProvidersResponse(trace_id=trace_id, providers=speaker_service.list_providers())
