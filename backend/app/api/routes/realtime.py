"""OpenAI Realtime-compatible endpoints (WebSocket + WebRTC).

``/v1/realtime`` is a WebSocket endpoint speaking the core Realtime event
subset; ``POST /v1/realtime/calls`` performs the WebRTC SDP handshake (audio
over RTP media tracks, JSON events over the ``oai-events`` data channel).
Both transports share the :class:`RealtimeDialogueEngine`.
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect

from app.services.realtime_engine import RealtimeDialogueEngine
from app.services.realtime_service import RealtimeConnection

try:
    from aiortc import RTCPeerConnection

    from app.services.webrtc_service import WebRTCDialogueSession, rtc_configuration_from_env

    WEBRTC_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    WEBRTC_AVAILABLE = False

router = APIRouter(tags=["realtime"])

logger = logging.getLogger(__name__)


# Active WebRTC calls, keyed by session id, so they can be hung up via DELETE.
_ACTIVE_CALLS: dict[str, dict[str, Any]] = {}


@router.websocket("/realtime")
async def realtime_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    conn = RealtimeConnection()
    send_lock = asyncio.Lock()

    async def send_event(event: dict) -> None:
        generation = event.get("generation")
        if generation is not None and conn.is_stale(generation):
            return
        async with send_lock:
            await websocket.send_json(event)

    engine = RealtimeDialogueEngine(conn, send_event)
    try:
        await engine.start()
        while True:
            raw = await websocket.receive_json()
            await engine.handle_client_event(raw)
    except WebSocketDisconnect:
        engine.close()
        return


@router.post("/realtime/calls")
async def webrtc_calls_endpoint(request: Request) -> Response:
    """WebRTC SDP handshake: POST an offer, receive the answer."""
    if not WEBRTC_AVAILABLE:
        return Response(
            content="WebRTC support requires aiortc: pip install 'aiortc'",
            status_code=501,
            media_type="text/plain",
        )
    if "application/sdp" not in request.headers.get("content-type", ""):
        return Response(
            content="Content-Type must be application/sdp",
            status_code=415,
            media_type="text/plain",
        )
    offer_sdp = (await request.body()).decode("utf-8")

    conn = RealtimeConnection()
    holder: dict[str, Any] = {}

    async def send_event(event: dict) -> None:
        generation = event.get("generation")
        if generation is not None and conn.is_stale(generation):
            return
        session = holder.get("session")
        if session is not None:
            await session.send_event(event)

    def audio_sink(pcm: bytes, sample_rate: int) -> None:
        session = holder.get("session")
        if session is not None:
            session.write_audio(pcm, sample_rate)

    holder["send"] = send_event
    holder["audio_sink"] = audio_sink
    engine = RealtimeDialogueEngine(conn, send_event, audio_sink=audio_sink)
    holder["engine"] = engine

    released = False

    def release() -> None:
        nonlocal released
        if released:
            return
        released = True
        engine.close()
        _ACTIVE_CALLS.pop(conn.session_id, None)

    pc = RTCPeerConnection(configuration=rtc_configuration_from_env())
    async def on_audio(pcm: bytes) -> None:
        # Inbound RTP is resampled to 16 kHz by the WebRTC session.
        await engine.feed_audio(pcm, 16000)

    session = WebRTCDialogueSession(
        pc,
        on_client_event=engine.handle_client_event,
        on_audio=on_audio,
        on_open=engine.start,
        on_closed=release,
    )
    holder["session"] = session
    session.setup()

    try:
        answer_sdp = await session.negotiate(offer_sdp)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[WebRTC] Negotiation failed: %s", exc)
        await session.close()
        return Response(content="Invalid SDP offer", status_code=400, media_type="text/plain")

    _ACTIVE_CALLS[conn.session_id] = {
        "pc": pc,
        "session": session,
        "engine": engine,
        "conn": conn,
    }
    return Response(
        content=answer_sdp,
        status_code=201,
        media_type="application/sdp",
        headers={"Location": f"/v1/realtime/calls/{conn.session_id}"},
    )


@router.delete("/realtime/calls/{call_id}")
async def webrtc_hangup_endpoint(call_id: str) -> Response:
    entry = _ACTIVE_CALLS.get(call_id)
    if entry is None:
        return Response(content="Unknown call", status_code=404, media_type="text/plain")
    session = entry["session"]
    await session.close()
    return Response(status_code=200)
