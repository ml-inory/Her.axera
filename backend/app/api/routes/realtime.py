"""OpenAI Realtime-compatible WebSocket endpoint.

Serves the core Realtime event subset at ``/v1/realtime`` so that standard
clients (including the official ``openai`` Python/JS SDKs) can talk to the
Her.axera voice pipeline without a custom protocol.  See
``app.services.realtime_service`` for the protocol state machine.
"""

from base64 import b64decode
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.audio_codec import pcm16_to_wav, wav_to_pcm16
from app.core.errors import AppError
from app.core.tracing import new_trace_id
from app.services.asr_service import asr_service
from app.services.dialogue_service import dialogue_service
from app.services.realtime_service import RealtimeConnection

router = APIRouter(tags=["realtime"])

logger = logging.getLogger(__name__)


@router.websocket("/realtime")
async def realtime_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    conn = RealtimeConnection()
    send_lock = asyncio.Lock()
    response_task: asyncio.Task | None = None

    async def send_event(event: dict) -> None:
        generation = event.get("generation")
        if generation is not None and conn.is_stale(generation):
            return
        async with send_lock:
            await websocket.send_json(event)

    async def run_transcription(item_id: str, pcm: bytes) -> None:
        try:
            wav = pcm16_to_wav(pcm, sample_rate=16000)
            result = await asr_service.transcribe(
                trace_id=new_trace_id(),
                audio_content=wav,
                filename=f"{item_id}.wav",
                provider=conn.config.asr_provider,
                model=None,
                language=conn.config.language,
                enable_timestamps=False,
                enable_vad=False,
            )
        except AppError as exc:
            await send_event(
                {
                    "type": "error",
                    "error": {"type": "transcription_failed", "message": exc.message},
                }
            )
            return
        transcript = result.text or ""
        for event in conn.finish_transcription(item_id, transcript):
            await send_event(event)
        if conn.config.server_vad and transcript.strip():
            nonlocal response_task
            response_task = asyncio.create_task(run_response(transcript))

    async def run_response(text: str) -> None:
        generation = conn.cancel_scope.generation
        output_item_id = f"msg_{conn.session_id}_{generation}"
        for event in conn.begin_response():
            event["generation"] = generation
            await send_event(event)
        try:
            trace_id = new_trace_id()
            async for event in dialogue_service.stream_text_pipeline(
                trace_id=trace_id,
                text=text,
                session_id=conn.session_id,
                user_id=None,
                language=conn.config.language,
                llm_provider=conn.config.llm_provider,
                llm_model=None,
                llm_api_key=None,
                tts_provider=conn.config.tts_provider,
                tts_model=None,
                voice=conn.config.voice,
                output_audio_format="pcm",
                sample_rate=conn.config.sample_rate,
                system_prompt=conn.config.system_prompt,
                output_audio_codec="pcm",
            ):
                await _dispatch_pipeline_event(event, output_item_id, generation)
            for event in conn.end_response("completed"):
                event["generation"] = generation
                await send_event(event)
        except asyncio.CancelledError:
            for event in conn.end_response("cancelled"):
                event["generation"] = generation
                await send_event(event)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Realtime response failed")
            await send_event(
                {
                    "type": "error",
                    "error": {"type": "server_error", "message": str(exc)[:300]},
                }
            )
            for event in conn.end_response("failed"):
                event["generation"] = generation
                await send_event(event)

    async def _dispatch_pipeline_event(event: dict, output_item_id: str, generation: int) -> None:
        event_type = event.get("type")
        if event_type == "llm_delta":
            text = str(event.get("text") or "")
            if conn.config.stream_text and text:
                await send_event(
                    {
                        "type": "response.output_text.delta",
                        "generation": generation,
                        "item_id": output_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                    }
                )
                await send_event(
                    {
                        "type": "response.audio_transcript.delta",
                        "generation": generation,
                        "item_id": output_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                    }
                )
        elif event_type == "tts_sentence":
            audio_b64 = str(event.get("audio_base64") or "")
            audio_format = str(event.get("audio_format") or "pcm")
            if not audio_b64:
                return
            raw = b64decode(audio_b64)
            if audio_format == "pcm":
                pcm = raw
            elif audio_format == "wav":
                decoded = wav_to_pcm16(raw)
                pcm = decoded[0] if decoded else b""
            else:
                # mp3/opus need external decoders; text deltas still flow.
                logger.debug("Skipping realtime audio delta for format %s", audio_format)
                return
            if not pcm:
                return
            from base64 import b64encode

            await send_event(
                {
                    "type": "response.output_audio.delta",
                    "generation": generation,
                    "item_id": output_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": b64encode(pcm).decode("ascii"),
                }
            )
        elif event_type == "llm":
            await send_event(
                {
                    "type": "response.output_text.done",
                    "generation": generation,
                    "item_id": output_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": str(event.get("text") or ""),
                }
            )
        elif event_type == "error":
            err = event.get("error") or {}
            await send_event(
                {
                    "type": "error",
                    "error": {"type": "pipeline_error", "message": str(err.get("message") or "")},
                }
            )
        # user_text / llm_started / done are not exposed over Realtime.

    try:
        await send_event(conn.session_created())
        while True:
            raw = await websocket.receive_json()
            event_type = raw.get("type")
            if event_type == "input_audio_buffer.append":
                audio_b64 = raw.get("audio") or ""
                try:
                    audio = b64decode(audio_b64)
                except Exception:  # noqa: BLE001
                    await send_event(
                        {
                            "type": "error",
                            "error": {"type": "invalid_request_error", "message": "audio must be base64"},
                        }
                    )
                    continue
                events, commit = conn.feed_audio(audio, conn.config.input_sample_rate)
                for event in events:
                    await send_event(event)
                if commit:
                    # New speech while responding: barge-in cancels the response.
                    if conn.in_response and conn.config.interrupt_response and response_task:
                        response_task.cancel()
                        conn.cancel_scope.cancel()
                        for event in conn.end_response("cancelled"):
                            await send_event(event)
                        response_task = None
                    asyncio.create_task(run_transcription(commit["item_id"], commit["pcm"]))
            elif event_type == "response.create":
                if conn.in_response and response_task:
                    response_task.cancel()
                    conn.cancel_scope.cancel()
                    for event in conn.end_response("cancelled"):
                        await send_event(event)
                    response_task = None
                text = conn.last_transcript or ""
                if not text.strip():
                    conn.in_response = False
                    await send_event(
                        {
                            "type": "error",
                            "error": {"type": "no_input", "message": "No user audio or text committed yet."},
                        }
                    )
                    continue
                response_task = asyncio.create_task(run_response(text))
            elif event_type == "response.cancel":
                if response_task:
                    response_task.cancel()
                    response_task = None
                for event in conn.handle_event(raw):
                    await send_event(event)
            else:
                for event in conn.handle_event(raw):
                    await send_event(event)
    except WebSocketDisconnect:
        if response_task:
            response_task.cancel()
        return
