from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import bind_trace_id
from app.models.common import JobCreatedResponse
from app.models.tts import (
    SegmentedSpeechRequest,
    SegmentedSpeechResponse,
    SpeechRequest,
    SpeechResponse,
    TTSJobResponse,
    TTSProvidersResponse,
    VoicesResponse,
)
from app.services.tts_service import tts_service

router = APIRouter(prefix="/tts", tags=["tts"])


@router.post("/speech", response_model=SpeechResponse)
async def create_speech(
    request: SpeechRequest,
    trace_id: Annotated[str, Depends(bind_trace_id)],
) -> SpeechResponse:
    return await tts_service.synthesize(trace_id, request)


@router.post("/speech/segments", response_model=SegmentedSpeechResponse)
async def create_segmented_speech(
    request: SegmentedSpeechRequest,
    trace_id: Annotated[str, Depends(bind_trace_id)],
) -> SegmentedSpeechResponse:
    return tts_service.synthesize_segments(trace_id, request)


@router.post("/jobs", response_model=JobCreatedResponse)
async def create_tts_job(trace_id: Annotated[str, Depends(bind_trace_id)]) -> JobCreatedResponse:
    return tts_service.create_job(trace_id)


@router.get("/jobs/{job_id}", response_model=TTSJobResponse)
async def get_tts_job(job_id: str, trace_id: Annotated[str, Depends(bind_trace_id)]) -> TTSJobResponse:
    return tts_service.get_job(trace_id, job_id)


@router.delete("/jobs/{job_id}", response_model=TTSJobResponse)
async def cancel_tts_job(job_id: str, trace_id: Annotated[str, Depends(bind_trace_id)]) -> TTSJobResponse:
    return tts_service.cancel_job(trace_id, job_id)


@router.get("/providers", response_model=TTSProvidersResponse)
async def list_tts_providers(trace_id: Annotated[str, Depends(bind_trace_id)]) -> TTSProvidersResponse:
    return TTSProvidersResponse(trace_id=trace_id, providers=tts_service.list_providers())


@router.get("/voices", response_model=VoicesResponse)
async def list_tts_voices(
    trace_id: Annotated[str, Depends(bind_trace_id)],
    language: Annotated[str | None, Query()] = None,
) -> VoicesResponse:
    return VoicesResponse(trace_id=trace_id, voices=tts_service.list_voices(language))
