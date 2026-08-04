from base64 import b64decode
import asyncio
from io import BytesIO
import logging
import wave

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import get_settings
from app.core.cancel_scope import CancelScope
from app.core.errors import AppError
from app.core.tracing import new_trace_id
from app.services.asr_service import asr_service
from app.services.asr_service import asr_service
from app.services.dialogue_service import dialogue_service
from app.services.streaming_vad import StreamingSileroVAD
from app.services.wakeword_service import wakeword_service

router = APIRouter(tags=["dialogue-websocket"])

logger = logging.getLogger(__name__)

# Trigger a partial ASR result after this many milliseconds of audio accumulated.
PARTIAL_ASR_THRESHOLD_MS = 2000

# Barge-in: minimum consecutive speech chunks to trigger interruption during active TTS.
BARGEIN_SPEECH_CHUNKS = 3
BARGEIN_ENERGY_THRESHOLD = 300

def _rms_energy(pcm_chunk: bytes) -> float:
    """Calculate RMS energy of a 16-bit PCM audio chunk."""
    import struct
    if len(pcm_chunk) < 2:
        return 0.0
    n = len(pcm_chunk) // 2
    samples = struct.unpack(f"<{n}h", pcm_chunk[:n * 2])
    return (sum(s * s for s in samples) / n) ** 0.5


class ConnectionState:
    def __init__(self) -> None:
        self.active_task: asyncio.Task | None = None
        self.active_turn_id: str | None = None
        self.cancel_scope = CancelScope()
        self.buffers: dict[str, bytearray] = {}
        self.turn_options: dict[str, dict] = {}
        self.partial_sent: dict[str, bool] = {}
        self.bargein_counter: int = 0  # consecutive speech chunks during active TTS
        # Free talk mode
        self.free_talk: bool = False
        self.free_talk_options: dict = {}
        self.streaming_vad: StreamingSileroVAD | None = None
        self.send_lock = asyncio.Lock()


def _pcm_to_wav(pcm: bytes, *, sample_rate: int, channels: int = 1) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def _prepare_audio(request: dict, audio_content: bytes, turn_id: str) -> tuple[bytes, str]:
    """Normalize one-shot audio/utterance payloads to a WAV file for ASR.

    ``encoding`` may be ``"pcm"`` (raw 16-bit LE PCM16, wrapped server-side)
    or ``"wav"`` (default).  Chunk-based flows (speech_start/audio_chunk) are
    always raw PCM16.
    """
    encoding = str(request.get("encoding") or request.get("transport") or "wav").lower()
    if encoding == "pcm":
        sample_rate = int(request.get("sample_rate") or request.get("input_sample_rate") or 16000)
        channels = int(request.get("channels") or 1)
        return _pcm_to_wav(audio_content, sample_rate=sample_rate, channels=channels), "wav"
    return audio_content, encoding


@router.get("/dialogue/sessions")
async def dialogue_sessions() -> dict[str, object]:
    """Diagnostics: sessions with active dialogue tasks."""
    return {"sessions": dialogue_service.session_diagnostics()}


@router.websocket("/dialogue/ws")
async def dialogue_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    state = ConnectionState()

    async def send_event(event: dict) -> None:
        generation = event.get("generation")
        if generation is not None and state.cancel_scope.is_stale(generation):
            return
        async with state.send_lock:
            await websocket.send_json(event)

    async def cancel_active(reason: str, replacement_turn_id: str | None = None) -> None:
        had_active = state.active_task is not None and not state.active_task.done()
        # Bump the generation on every turn boundary so detached tasks (e.g.
        # partial ASR) from older turns are treated as stale.
        state.cancel_scope.cancel()
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
        if had_active:
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

    async def run_partial_asr(options: dict, trace_id: str, pcm: bytes, turn_id: str) -> None:
        """Run ASR on accumulated audio so far and send a partial result (display only)."""
        generation = state.cancel_scope.generation
        try:
            sample_rate = int(options.get("input_sample_rate") or options.get("sample_rate") or 16000)
            channels = int(options.get("channels") or 1)
            audio_content = _pcm_to_wav(pcm, sample_rate=sample_rate, channels=channels)
            result = await asr_service.transcribe(
                trace_id=trace_id,
                audio_content=audio_content,
                filename=f"{turn_id}_partial.wav",
                provider=options.get("asr_provider"),
                model=options.get("asr_model"),
                language=options.get("language") or "zh-CN",
                enable_timestamps=False,
                enable_vad=False,
            )
            await send_event({
                "type": "asr_partial",
                "generation": generation,
                "trace_id": trace_id,
                "turn_id": turn_id,
                "text": result.text,
                "provider": result.provider,
                "model": result.model,
                "processing_ms": result.processing_ms,
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("Partial ASR failed (non-fatal): %s", exc)

    async def run_audio_pipeline(request: dict, trace_id: str, audio_content: bytes, turn_id: str) -> None:
        generation = state.cancel_scope.generation
        try:
            await send_event({"type": "asr_started", "trace_id": trace_id, "turn_id": turn_id})
            audio_content, _ = _prepare_audio(request, audio_content, turn_id)
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
                output_audio_codec=request.get("output_audio_codec") or "pcm",
            ):
                event["turn_id"] = turn_id
                event["generation"] = generation
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
        generation = state.cancel_scope.generation
        try:
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
                output_audio_codec=request.get("output_audio_codec") or "pcm",
                image_base64=request.get("image_base64"),
            ):
                event["turn_id"] = turn_id
                event["generation"] = generation
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
            if message_type not in {"audio", "utterance", "text", "speech_start", "audio_chunk", "speech_end", "abort", "free_talk_start", "free_talk_end"}:
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
                if message_type == "free_talk_start":
                    state.free_talk = True
                    state.free_talk_options = dict(request.get("options") or request)
                    state.free_talk_options.setdefault("encoding", "pcm")
                    state.streaming_vad = StreamingSileroVAD()
                    await cancel_active("free_talk")
                    await send_event({
                        "type": "free_talk_started",
                        "trace_id": trace_id,
                        "options": state.free_talk_options,
                    })
                    continue

                if message_type == "free_talk_end":
                    # Process any remaining speech before ending.
                    if state.free_talk and state.streaming_vad is not None:
                        flushed = state.streaming_vad.flush()
                        if flushed.utterance_pcm:
                            ft_turn_id = new_trace_id("turn")
                            audio_content = _pcm_to_wav(flushed.utterance_pcm, sample_rate=16000)
                            await send_event({"type": "utterance_detected", "trace_id": trace_id, "turn_id": ft_turn_id})
                            state.active_turn_id = ft_turn_id
                            state.bargein_counter = 0
                            state.active_task = asyncio.create_task(
                                run_audio_pipeline(state.free_talk_options, trace_id, audio_content, ft_turn_id)
                            )
                    state.free_talk = False
                    state.streaming_vad = None
                    await send_event({"type": "free_talk_ended", "trace_id": trace_id})
                    continue

                if message_type == "speech_start":
                    await cancel_active("new_speech", replacement_turn_id=turn_id)
                    state.buffers[turn_id] = bytearray()
                    state.turn_options[turn_id] = dict(request)
                    state.partial_sent[turn_id] = False
                    await send_event({"type": "accepted", "trace_id": trace_id, "turn_id": turn_id})
                    await send_event({"type": "speech_started", "trace_id": trace_id, "turn_id": turn_id})
                    continue


                if message_type == "audio_chunk":
                    # Free talk mode: streaming Silero VAD auto-splits utterances.
                    if state.free_talk:
                        chunk_data = b64decode(str(request.get("audio_base64") or ""), validate=True)
                        input_sr = int(state.free_talk_options.get("input_sample_rate") or 16000)
                        if state.streaming_vad is None:
                            state.streaming_vad = StreamingSileroVAD()
                        result = state.streaming_vad.feed(chunk_data, input_sr)

                        # Barge-in during free talk: cancel the active pipeline
                        # while the user keeps talking (VAD-driven).
                        if result.speech_active and state.active_task and not state.active_task.done():
                            state.bargein_counter += 1
                            if state.bargein_counter >= BARGEIN_SPEECH_CHUNKS:
                                await cancel_active("barge_in")
                                state.bargein_counter = 0
                        else:
                            state.bargein_counter = max(0, state.bargein_counter - 1)

                        if result.utterance_pcm:
                            ft_turn_id = new_trace_id("turn")
                            audio_content = _pcm_to_wav(result.utterance_pcm, sample_rate=16000)
                            await send_event({"type": "utterance_detected", "trace_id": trace_id, "turn_id": ft_turn_id})
                            state.active_turn_id = ft_turn_id
                            state.bargein_counter = 0
                            state.active_task = asyncio.create_task(
                                run_audio_pipeline(state.free_talk_options, trace_id, audio_content, ft_turn_id)
                            )
                        continue

                    # Legacy chunk mode (non-free-talk)
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
                    chunk_data = b64decode(str(request.get("audio_base64") or ""), validate=True)
                    state.buffers[chunk_turn_id].extend(chunk_data)

                    # Wake word detection on incoming chunk.
                    detected = False
                    ww_name = None
                    if wakeword_service.available():
                        options = state.turn_options.get(chunk_turn_id, {})
                        input_sr = int(options.get("input_sample_rate") or 16000)
                        detected, ww_name = wakeword_service.detect(chunk_data, input_sr)
                    if detected:
                            await send_event({"type": "wake_word_detected", "trace_id": trace_id, "turn_id": chunk_turn_id})

                    # Barge-in: detect speech during active TTS pipeline.
                    if state.active_task and not state.active_task.done():
                        energy = _rms_energy(chunk_data)
                        if energy >= BARGEIN_ENERGY_THRESHOLD:
                            state.bargein_counter += 1
                            if state.bargein_counter >= BARGEIN_SPEECH_CHUNKS:
                                # User is speaking — cancel active pipeline and start new turn
                                bargein_turn = new_trace_id("turn")
                                await cancel_active("barge_in", replacement_turn_id=bargein_turn)
                                state.buffers[bargein_turn] = bytearray()
                                state.turn_options[bargein_turn] = dict(state.turn_options.get(chunk_turn_id, {}))
                                state.partial_sent[bargein_turn] = False
                                state.bargein_counter = 0
                                state.buffers[bargein_turn].extend(chunk_data)
                                await send_event({"type": "speech_started", "trace_id": trace_id, "turn_id": bargein_turn})
                                continue
                        else:
                            state.bargein_counter = max(0, state.bargein_counter - 1)

                    # Trigger partial ASR when buffer exceeds threshold.
                    if not state.partial_sent.get(chunk_turn_id, False):
                        options = state.turn_options.get(chunk_turn_id, {})
                        input_sr = int(options.get("input_sample_rate") or options.get("sample_rate") or 16000)
                        buffer_duration_ms = len(state.buffers[chunk_turn_id]) * 1000 // (input_sr * 2)
                        if buffer_duration_ms >= PARTIAL_ASR_THRESHOLD_MS:
                            state.partial_sent[chunk_turn_id] = True
                            partial_pcm = bytes(state.buffers[chunk_turn_id])
                            asyncio.create_task(run_partial_asr(options, trace_id, partial_pcm, chunk_turn_id))
                    continue

                if message_type == "speech_end":
                    end_turn_id = str(request.get("turn_id") or "")
                    options = state.turn_options.pop(end_turn_id, dict(request))
                    state.partial_sent.pop(end_turn_id, None)
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
                    state.bargein_counter = 0
                    state.active_task = asyncio.create_task(run_audio_pipeline(options, trace_id, audio_content, end_turn_id))
                    continue

                await cancel_active("new_turn", replacement_turn_id=turn_id)
                await send_event({"type": "accepted", "trace_id": trace_id, "turn_id": turn_id})
                if message_type in {"audio", "utterance"}:
                    audio_content = b64decode(str(request.get("audio_base64") or ""), validate=True)
                    state.active_turn_id = turn_id
                    state.active_task = asyncio.create_task(run_audio_pipeline(request, trace_id, audio_content, turn_id))
                elif message_type == "text":
                    state.active_turn_id = turn_id
                    state.active_task = asyncio.create_task(run_text_pipeline(request, trace_id, turn_id))
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
                logger.exception("Pipeline failed for turn %s", turn_id if 'turn_id' in dir() else 'unknown')
                try:
                    await send_event(
                        {
                            "type": "error",
                            "trace_id": trace_id,
                            "error": {
                                "code": "pipeline_failed",
                                "message": str(exc)[:300],
                                "stage": "dialogue",
                                "retryable": True,
                            },
                        }
                    )
                except Exception:
                    logger.warning("Could not send error event (websocket likely closed)")
    except WebSocketDisconnect:
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
        return
