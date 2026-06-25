from base64 import b64decode
import asyncio
from io import BytesIO
import wave

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.errors import AppError
from app.core.tracing import new_trace_id
from app.services.dialogue_service import dialogue_service

router = APIRouter(tags=["dialogue-websocket"])


class ConnectionState:
    def __init__(self) -> None:
        self.active_task: asyncio.Task | None = None
        self.active_turn_id: str | None = None
        self.buffers: dict[str, bytearray] = {}
        self.turn_options: dict[str, dict] = {}
        self.send_lock = asyncio.Lock()


def _pcm_to_wav(pcm: bytes, *, sample_rate: int, channels: int = 1) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


@router.websocket("/dialogue/ws")
async def dialogue_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    state = ConnectionState()

    async def send_event(event: dict) -> None:
        async with state.send_lock:
            await websocket.send_json(event)

    async def cancel_active(reason: str, replacement_turn_id: str | None = None) -> None:
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
            await send_event(
                {
                    "type": "interrupted",
                    "turn_id": state.active_turn_id,
                    "replacement_turn_id": replacement_turn_id,
                    "reason": reason,
                }
            )
        state.active_task = None
        state.active_turn_id = None

    async def run_audio_pipeline(request: dict, trace_id: str, audio_content: bytes, turn_id: str) -> None:
        try:
            await send_event({"type": "asr_started", "trace_id": trace_id, "turn_id": turn_id})
            async for event in dialogue_service.stream_audio_pipeline(
                trace_id=trace_id,
                audio_content=audio_content,
                filename=request.get("filename") or f"{turn_id}.wav",
                session_id=request.get("session_id"),
                user_id=request.get("user_id"),
                language=request.get("language") or "zh-CN",
                asr_provider=request.get("asr_provider"),
                asr_model=request.get("asr_model"),
                llm_provider=request.get("llm_provider"),
                llm_model=request.get("llm_model"),
                llm_api_key=request.get("llm_api_key"),
                tts_provider=request.get("tts_provider"),
                tts_model=request.get("tts_model"),
                voice=request.get("voice"),
                output_audio_format=request.get("output_audio_format") or "wav",
                sample_rate=int(request.get("sample_rate") or 24000),
                system_prompt=request.get("system_prompt"),
                speaker_enabled=bool(request.get("speaker_enabled", False)),
                speaker_provider=request.get("speaker_provider"),
            ):
                event["turn_id"] = turn_id
                if event.get("type") == "llm":
                    await send_event({"type": "llm_started", "trace_id": trace_id, "turn_id": turn_id})
                await send_event(event)
        except asyncio.CancelledError:
            return
        except AppError as exc:
            await send_event(
                {
                    "type": "error",
                    "trace_id": trace_id,
                    "turn_id": turn_id,
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                        "stage": exc.stage,
                        "retryable": exc.retryable,
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            await send_event(
                {
                    "type": "error",
                    "trace_id": trace_id,
                    "turn_id": turn_id,
                    "error": {
                        "code": "pipeline_failed",
                        "message": str(exc),
                        "stage": "dialogue",
                        "retryable": True,
                    },
                }
            )

    async def run_text_pipeline(request: dict, trace_id: str, turn_id: str) -> None:
        try:
            await send_event({"type": "llm_started", "trace_id": trace_id, "turn_id": turn_id})
            async for event in dialogue_service.stream_text_pipeline(
                trace_id=trace_id,
                text=str(request.get("text") or ""),
                session_id=request.get("session_id"),
                user_id=request.get("user_id"),
                language=request.get("language") or "zh-CN",
                llm_provider=request.get("llm_provider"),
                llm_model=request.get("llm_model"),
                llm_api_key=request.get("llm_api_key"),
                tts_provider=request.get("tts_provider"),
                tts_model=request.get("tts_model"),
                voice=request.get("voice"),
                output_audio_format=request.get("output_audio_format") or "wav",
                sample_rate=int(request.get("sample_rate") or 24000),
                system_prompt=request.get("system_prompt"),
            ):
                event["turn_id"] = turn_id
                await send_event(event)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            await send_event(
                {
                    "type": "error",
                    "trace_id": trace_id,
                    "turn_id": turn_id,
                    "error": {
                        "code": "pipeline_failed",
                        "message": str(exc),
                        "stage": "dialogue",
                        "retryable": True,
                    },
                }
            )

    try:
        while True:
            request = await websocket.receive_json()
            message_type = request.get("type")
            if message_type not in {"audio", "utterance", "text", "speech_start", "audio_chunk", "speech_end", "abort"}:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": {"code": "invalid_message", "message": "unsupported message type"},
                    }
                )
                continue

            trace_id = str(request.get("trace_id") or new_trace_id())
            try:
                turn_id = str(request.get("turn_id") or new_trace_id("turn"))
                if message_type == "abort":
                    await cancel_active("client_abort")
                    await send_event({"type": "accepted", "trace_id": trace_id, "turn_id": turn_id})
                    continue

                if message_type == "speech_start":
                    await cancel_active("new_speech", replacement_turn_id=turn_id)
                    state.buffers[turn_id] = bytearray()
                    state.turn_options[turn_id] = dict(request)
                    await send_event({"type": "accepted", "trace_id": trace_id, "turn_id": turn_id})
                    await send_event({"type": "speech_started", "trace_id": trace_id, "turn_id": turn_id})
                    continue

                if message_type == "audio_chunk":
                    chunk_turn_id = str(request.get("turn_id") or "")
                    if chunk_turn_id not in state.buffers:
                        await send_event(
                            {
                                "type": "error",
                                "trace_id": trace_id,
                                "turn_id": chunk_turn_id,
                                "error": {
                                    "code": "turn_not_found",
                                    "message": "audio_chunk received before speech_start",
                                    "stage": "dialogue",
                                    "retryable": False,
                                },
                            }
                        )
                        continue
                    state.buffers[chunk_turn_id].extend(b64decode(str(request.get("audio_base64") or ""), validate=True))
                    continue

                if message_type == "speech_end":
                    end_turn_id = str(request.get("turn_id") or "")
                    options = state.turn_options.pop(end_turn_id, dict(request))
                    pcm = bytes(state.buffers.pop(end_turn_id, b""))
                    if not pcm:
                        await send_event(
                            {
                                "type": "error",
                                "trace_id": trace_id,
                                "turn_id": end_turn_id,
                                "error": {
                                    "code": "empty_audio",
                                    "message": "no audio chunks were received",
                                    "stage": "asr",
                                    "retryable": False,
                                },
                            }
                        )
                        continue
                    options.update(request)
                    sample_rate = int(options.get("input_sample_rate") or options.get("sample_rate") or 16000)
                    channels = int(options.get("channels") or 1)
                    audio_content = _pcm_to_wav(pcm, sample_rate=sample_rate, channels=channels)
                    await send_event({"type": "accepted", "trace_id": trace_id, "turn_id": end_turn_id})
                    state.active_turn_id = end_turn_id
                    state.active_task = asyncio.create_task(run_audio_pipeline(options, trace_id, audio_content, end_turn_id))
                    continue

                await cancel_active("new_turn", replacement_turn_id=turn_id)
                await send_event({"type": "accepted", "trace_id": trace_id, "turn_id": turn_id})
                if message_type in {"audio", "utterance"}:
                    audio_content = b64decode(str(request.get("audio_base64") or ""), validate=True)
                    state.active_turn_id = turn_id
                    state.active_task = asyncio.create_task(run_audio_pipeline(request, trace_id, audio_content, turn_id))
                else:
                    state.active_turn_id = turn_id
                    state.active_task = asyncio.create_task(run_text_pipeline(request, trace_id, turn_id))
            except AppError as exc:
                await send_event(
                    {
                        "type": "error",
                        "trace_id": trace_id,
                        "error": {
                            "code": exc.code,
                            "message": exc.message,
                            "stage": exc.stage,
                            "retryable": exc.retryable,
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001
                await send_event(
                    {
                        "type": "error",
                        "trace_id": trace_id,
                        "error": {
                            "code": "pipeline_failed",
                            "message": str(exc),
                            "stage": "dialogue",
                            "retryable": True,
                        },
                    }
                )
    except WebSocketDisconnect:
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
        return
