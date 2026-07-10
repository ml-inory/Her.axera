"""Session listing and management."""

from fastapi import APIRouter, HTTPException, Query

from app.services.llm_service import llm_service

router = APIRouter(tags=["sessions"])


@router.get("/sessions")
def list_sessions(user_id: str | None = Query(default=None)):
    sessions = llm_service.list_sessions(user_id=user_id)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = llm_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if not llm_service.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "deleted": True}
