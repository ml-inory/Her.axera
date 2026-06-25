from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import bind_trace_id
from app.models.speaker import (
    SpeakerDeleteResponse,
    SpeakerEnrollResponse,
    SpeakerIdentifyResponse,
    SpeakerListResponse,
    SpeakerProvidersResponse,
)
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


@router.post("/enroll", response_model=SpeakerEnrollResponse)
async def enroll_speaker(
    trace_id: Annotated[str, Depends(bind_trace_id)],
    audio: Annotated[UploadFile, File()],
    speaker_id: Annotated[str, Form()],
    name: Annotated[str, Form()] = "",
    description: Annotated[str, Form()] = "",
    provider: Annotated[str | None, Form()] = None,
) -> SpeakerEnrollResponse:
    content = await audio.read()
    return speaker_service.enroll(
        trace_id=trace_id,
        audio_content=content,
        filename=audio.filename,
        speaker_id=speaker_id,
        name=name,
        description=description,
        provider=provider,
    )


@router.get("", response_model=SpeakerListResponse)
async def list_speakers(trace_id: Annotated[str, Depends(bind_trace_id)]) -> SpeakerListResponse:
    return SpeakerListResponse(trace_id=trace_id, speakers=speaker_service.list_speakers(trace_id=trace_id))


@router.delete("/{speaker_id}", response_model=SpeakerDeleteResponse)
async def delete_speaker(
    speaker_id: str,
    trace_id: Annotated[str, Depends(bind_trace_id)],
) -> SpeakerDeleteResponse:
    return speaker_service.delete_speaker(trace_id=trace_id, speaker_id=speaker_id)


@router.get("/providers", response_model=SpeakerProvidersResponse)
async def list_speaker_providers(trace_id: Annotated[str, Depends(bind_trace_id)]) -> SpeakerProvidersResponse:
    return SpeakerProvidersResponse(trace_id=trace_id, providers=speaker_service.list_providers())
