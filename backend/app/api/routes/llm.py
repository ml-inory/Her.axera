from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import bind_trace_id
from app.models.common import JobCreatedResponse
from app.models.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    DeleteSessionMessagesResponse,
    LLMJobResponse,
    LLMProvidersResponse,
    SessionMessagesResponse,
)
from app.services.llm_service import llm_service

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    trace_id: Annotated[str, Depends(bind_trace_id)],
) -> ChatCompletionResponse:
    return llm_service.chat(trace_id, request)


@router.post("/chat/jobs", response_model=JobCreatedResponse)
async def create_llm_job(trace_id: Annotated[str, Depends(bind_trace_id)]) -> JobCreatedResponse:
    return llm_service.create_job(trace_id)


@router.get("/chat/jobs/{job_id}", response_model=LLMJobResponse)
async def get_llm_job(job_id: str, trace_id: Annotated[str, Depends(bind_trace_id)]) -> LLMJobResponse:
    return llm_service.get_job(trace_id, job_id)


@router.delete("/chat/jobs/{job_id}", response_model=LLMJobResponse)
async def cancel_llm_job(job_id: str, trace_id: Annotated[str, Depends(bind_trace_id)]) -> LLMJobResponse:
    return llm_service.cancel_job(trace_id, job_id)


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str,
    trace_id: Annotated[str, Depends(bind_trace_id)],
) -> SessionMessagesResponse:
    return llm_service.get_session_messages(trace_id, session_id)


@router.delete("/sessions/{session_id}/messages", response_model=DeleteSessionMessagesResponse)
async def delete_session_messages(
    session_id: str,
    trace_id: Annotated[str, Depends(bind_trace_id)],
) -> DeleteSessionMessagesResponse:
    return llm_service.delete_session_messages(trace_id, session_id)


@router.get("/providers", response_model=LLMProvidersResponse)
async def list_llm_providers(trace_id: Annotated[str, Depends(bind_trace_id)]) -> LLMProvidersResponse:
    return LLMProvidersResponse(trace_id=trace_id, providers=llm_service.list_providers())
