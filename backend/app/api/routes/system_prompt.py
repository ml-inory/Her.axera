"""System prompt management API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import bind_trace_id
from app.core.config import get_settings

router = APIRouter(tags=["system-prompt"])

# In-memory override (resets on restart)
_override: str | None = None


class SystemPromptBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)


@router.get("/system-prompt")
async def get_system_prompt(trace_id: str = Depends(bind_trace_id)):
    """Get the current system prompt (default or override)."""
    settings = get_settings()
    return {
        "trace_id": trace_id,
        "prompt": _override or settings.default_system_prompt,
        "is_default": _override is None,
    }


@router.put("/system-prompt")
async def set_system_prompt(body: SystemPromptBody, trace_id: str = Depends(bind_trace_id)):
    """Override the system prompt at runtime (resets on restart)."""
    global _override
    _override = body.prompt.strip()
    return {
        "trace_id": trace_id,
        "prompt": _override,
        "is_default": False,
    }


@router.delete("/system-prompt")
async def reset_system_prompt(trace_id: str = Depends(bind_trace_id)):
    """Reset to the default system prompt."""
    global _override
    _override = None
    settings = get_settings()
    return {
        "trace_id": trace_id,
        "prompt": settings.default_system_prompt,
        "is_default": True,
    }
