"""Model download status & trigger endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.model_download_service import (
    DownloadStatus,
    ModelDownloadState,
    get_model_download_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ModelStatusItem(BaseModel):
    key: str
    display_name: str
    model_type: str
    status: str
    progress_pct: float
    downloaded_bytes: int
    total_bytes: int
    error_message: str


class ModelsStatusResponse(BaseModel):
    models: list[ModelStatusItem]
    all_ready: bool


class DownloadTriggerRequest(BaseModel):
    keys: list[str] = Field(default_factory=list, description="Specific model keys to download; empty = all")
    model_type: str | None = Field(default=None, description="Filter by model_type: 'asr' or 'tts'")


class DownloadTriggerResponse(BaseModel):
    started: list[str]
    already_done: list[str]
    not_found: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _state_to_item(key: str, state: ModelDownloadState) -> ModelStatusItem:
    return ModelStatusItem(
        key=key,
        display_name=state.spec.display_name,
        model_type=state.spec.model_type,
        status=state.status.value,
        progress_pct=state.progress_pct,
        downloaded_bytes=state.downloaded_bytes,
        total_bytes=state.total_bytes,
        error_message=state.error_message,
    )


@router.get("/models/download/status", response_model=ModelsStatusResponse)
def get_model_download_status(model_type: str | None = Query(default=None, description="Filter: 'asr' or 'tts'")) -> ModelsStatusResponse:
    mgr = get_model_download_manager()
    states = mgr.get_all_states()
    if model_type:
        states = {k: v for k, v in states.items() if v.spec.model_type == model_type}

    items = [_state_to_item(k, v) for k, v in states.items()]
    all_ready = all(s.status == DownloadStatus.DOWNLOADED for s in states.values()) if states else True
    return ModelsStatusResponse(models=items, all_ready=all_ready)


@router.post("/models/download", response_model=DownloadTriggerResponse)
def trigger_model_download(body: DownloadTriggerRequest) -> DownloadTriggerResponse:
    mgr = get_model_download_manager()

    # Determine which keys to download
    if body.keys:
        target_keys = body.keys
    elif body.model_type:
        target_keys = [k for k, s in mgr.specs.items() if s.model_type == body.model_type]
    else:
        target_keys = list(mgr.specs.keys())

    started: list[str] = []
    already_done: list[str] = []
    not_found: list[str] = []

    for key in target_keys:
        state = mgr.get_state(key)
        if state is None:
            not_found.append(key)
            continue
        if state.status == DownloadStatus.DOWNLOADED:
            already_done.append(key)
            continue
        if state.status == DownloadStatus.DOWNLOADING:
            already_done.append(key)
            continue
        if mgr.start_download(key):
            started.append(key)
        else:
            already_done.append(key)

    return DownloadTriggerResponse(started=started, already_done=already_done, not_found=not_found)
