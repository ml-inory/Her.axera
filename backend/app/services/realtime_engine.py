"""Shared dialogue engine for Realtime transports (WebSocket and WebRTC).

The engine binds a :class:`RealtimeConnection` (protocol state machine) to an
event transport.  WebSocket and WebRTC routes both instantiate one engine per
connection and delegate client events / inbound audio to it; the engine owns
ASR transcription tasks, response streaming, barge-in cancellation and the
pipeline-event -> Realtime-event mapping.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.core.audio_codec import pcm16_to_wav, wav_to_pcm16
from app.core.errors import AppError
from app.core.tracing import new_trace_id
from app.services.asr_service import asr_service
from app.services.dialogue_service import dialogue_service
from app.services.realtime_service import RealtimeConnection

logger = logging.getLogger(__name__)


SendEvent = Callable[[dict], Awaitable[None]]
AudioSink = Callable[[bytes, int], None]


class RealtimeDialogueEngine:
    """Owns one Realtime conversation over an arbitrary event transport."""

    def __init__(self, conn: RealtimeConnection, send: SendEvent, audio_sink: AudioSink | None = None) -> None:
        self.conn = conn
        self._send = send
        self._audio_sink = audio_sink
        self.response_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self._send(self.conn.session_created())

    def close(self) -> None:
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()

    # ------------------------------------------------------------------
    # Inbound events / audio
    # ------------------------------------------------------------------

    async def handle_client_event(self, raw: dict) -> None:
        event_type = raw.get("type")
        if event_type == "input_audio_buffer.append":
            audio_b64 = raw.get("audio") or ""
            try:
                audio = b64decode(audio_b64)
            except Exception:  # noqa: BLE001
                await self._send(
                    {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "audio must be base64"},
                    }
                )
                return
            await self.feed_audio(audio, self.conn.config.input_sample_rate)
        elif event_type == "response.create":
            await self._create_response(raw)
        elif event_type == "response.cancel":
            if self.response_task:
                self.response_task.cancel()
                self.response_task = None
            for event in self.conn.handle_event(raw):
                await self._send(event)
        else:
            for event in self.conn.handle_event(raw):
                await self._send(event)

    async def feed_audio(self, pcm: bytes, sample_rate: int | None = None) -> None:
        events, commit = self.conn.feed_audio(pcm, sample_rate)
        for event in events:
            await self._send(event)
        if commit:
            if self.conn.in_response and self.conn.config.interrupt_response and self.response_task:
                self.response_task.cancel()
                self.conn.cancel_scope.cancel()
                for event in self.conn.end_response("cancelled"):
                    await self._send(event)
                self.response_task = None
            asyncio.create_task(self._run_transcription(commit["item_id"], commit["pcm"]))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _create_response(self, raw: dict) -> None:
        if self.conn.in_response and self.response_task:
            self.response_task.cancel()
            self.conn.cancel_scope.cancel()
            for event in self.conn.end_response("cancelled"):
                await self._send(event)
            self.response_task = None
        text = self.conn.last_transcript or ""
        if not text.strip():
            self.conn.in_response = False
            await self._send(
                {
                    "type": "error",
                    "error": {"type": "no_input", "message": "No user audio or text committed yet."},
                }
            )
            return
        self.response_task = asyncio.create_task(self._run_response(text))

    async def _run_transcription(self, item_id: str, pcm: bytes) -> None:
        try:
            wav = pcm16_to_wav(pcm, sample_rate=16000)
            result = await asr_service.transcribe(
                trace_id=new_trace_id(),
                audio_content=wav,
                filename=f"{item_id}.wav",
                provider=self.conn.config.asr_provider,
                model=None,
                language=self.conn.config.language,
                enable_timestamps=False,
                enable_vad=False,
            )
        except AppError as exc:
            await self._send(
                {
                    "type": "error",
                    "error": {"type": "transcription_failed", "message": exc.message},
                }
            )
            return
        transcript = result.text or ""
        for event in self.conn.finish_transcription(item_id, transcript):
            await self._send(event)
        if self.conn.config.server_vad and transcript.strip():
            self.response_task = asyncio.create_task(self._run_response(transcript))

    async def _run_response(self, text: str) -> None:
        generation = self.conn.cancel_scope.generation
        output_item_id = f"msg_{self.conn.session_id}_{generation}"
        for event in self.conn.begin_response():
            event["generation"] = generation
            await self._send(event)
        try:
            trace_id = new_trace_id()
            async for event in dialogue_service.stream_text_pipeline(
                trace_id=trace_id,
                text=text,
                session_id=self.conn.session_id,
                user_id=None,
                language=self.conn.config.language,
                llm_provider=self.conn.config.llm_provider,
                llm_model=None,
                llm_api_key=None,
                tts_provider=self.conn.config.tts_provider,
                tts_model=None,
                voice=self.conn.config.voice,
                output_audio_format="pcm",
                sample_rate=self.conn.config.sample_rate,
                system_prompt=self.conn.config.system_prompt,
                output_audio_codec="pcm",
            ):
                await self._dispatch_pipeline_event(event, output_item_id, generation)
            for event in self.conn.end_response("completed"):
                event["generation"] = generation
                await self._send(event)
        except asyncio.CancelledError:
            for event in self.conn.end_response("cancelled"):
                event["generation"] = generation
                await self._send(event)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Realtime response failed")
            await self._send(
                {
                    "type": "error",
                    "error": {"type": "server_error", "message": str(exc)[:300]},
                }
            )
            for event in self.conn.end_response("failed"):
                event["generation"] = generation
                await self._send(event)

    async def _dispatch_pipeline_event(self, event: dict, output_item_id: str, generation: int) -> None:
        event_type = event.get("type")
        if event_type == "llm_delta":
            text = str(event.get("text") or "")
            if self.conn.config.stream_text and text:
                await self._send(
                    {
                        "type": "response.output_text.delta",
                        "generation": generation,
                        "item_id": output_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                    }
                )
                await self._send(
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
                logger.debug("Skipping realtime audio delta for format %s", audio_format)
                return
            if not pcm:
                return
            if self._audio_sink is not None:
                try:
                    self._audio_sink(pcm, int(event.get("sample_rate") or self.conn.config.sample_rate))
                except Exception:  # noqa: BLE001
                    logger.exception("Realtime audio sink failed")
            await self._send(
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
            await self._send(
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
            await self._send(
                {
                    "type": "error",
                    "error": {"type": "pipeline_error", "message": str(err.get("message") or "")},
                }
            )
