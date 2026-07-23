"""Wake word management API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import bind_trace_id
from app.services.wakeword_service import wakeword_service

router = APIRouter(tags=["wakewords"])


class WakeWordRegisterBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    audio_base64: str = Field(...)
    description: str = ""


@router.post("/wakewords")
async def register_wakeword(body: WakeWordRegisterBody, trace_id: str = Depends(bind_trace_id)):
    """Register a custom wake word from an audio sample."""
    if not wakeword_service.available():
        raise HTTPException(status_code=503, detail="Wake word detection is not enabled")
    try:
        result = wakeword_service.register(body.name, body.audio_base64, body.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "trace_id": trace_id,
        "name": body.name,
        "status": result["status"],
        "sample_count": result["sample_count"],
    }


@router.get("/wakewords")
async def list_wakewords(trace_id: str = Depends(bind_trace_id)):
    """List all registered wake words."""
    wake_words = wakeword_service.list_wakewords()
    return {
        "trace_id": trace_id,
        "wake_words": wake_words,
    }


@router.delete("/wakewords/{name}")
async def delete_wakeword(name: str, trace_id: str = Depends(bind_trace_id)):
    """Delete a registered wake word."""
    deleted = wakeword_service.delete(name)
    return {
        "trace_id": trace_id,
        "name": name,
        "deleted": deleted,
    }
