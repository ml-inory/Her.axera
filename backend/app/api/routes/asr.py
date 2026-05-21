from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import bind_trace_id
from app.models.asr import ASRJobResponse, ASRProvidersResponse, ASRResult
from app.models.common import JobCreatedResponse
from app.services.asr_service import asr_service

router = APIRouter(prefix="/asr", tags=["asr"])


@router.post("/transcriptions", response_model=ASRResult)
async def create_transcription(
    trace_id: Annotated[str, Depends(bind_trace_id)],
    audio: Annotated[UploadFile, File()],
    provider: Annotated[str | None, Form()] = None,
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    enable_timestamps: Annotated[bool, Form()] = False,
) -> ASRResult:
    content = await audio.read()
    return await asr_service.transcribe(
        trace_id=trace_id,
        audio_content=content,
        filename=audio.filename,
        provider=provider,
        model=model,
        language=language,
        enable_timestamps=enable_timestamps,
    )


@router.post("/jobs", response_model=JobCreatedResponse)
async def create_asr_job(trace_id: Annotated[str, Depends(bind_trace_id)]) -> JobCreatedResponse:
    return asr_service.create_job(trace_id)


@router.get("/jobs/{job_id}", response_model=ASRJobResponse)
async def get_asr_job(job_id: str, trace_id: Annotated[str, Depends(bind_trace_id)]) -> ASRJobResponse:
    return asr_service.get_job(trace_id, job_id)


@router.delete("/jobs/{job_id}", response_model=ASRJobResponse)
async def cancel_asr_job(job_id: str, trace_id: Annotated[str, Depends(bind_trace_id)]) -> ASRJobResponse:
    return asr_service.cancel_job(trace_id, job_id)


@router.get("/providers", response_model=ASRProvidersResponse)
async def list_asr_providers(trace_id: Annotated[str, Depends(bind_trace_id)]) -> ASRProvidersResponse:
    return ASRProvidersResponse(trace_id=trace_id, providers=asr_service.list_providers())
