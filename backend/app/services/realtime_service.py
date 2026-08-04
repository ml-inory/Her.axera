"""OpenAI Realtime protocol state machine (core event subset).

Implements the minimal event set needed for an ``openai`` SDK realtime client
(or any Realtime-compatible client) to hold a voice conversation:

- client -> server: ``session.update``, ``input_audio_buffer.append``,
  ``input_audio_buffer.commit``, ``response.create``, ``response.cancel``,
  ``conversation.item.create``, ``ping``
- server -> client: ``session.created/updated``, ``input_audio_buffer.*``,
  ``conversation.item.created``, ``conversation.item.input_audio_transcription.completed``,
  ``response.created/done``, ``response.output_text.delta/done``,
  ``response.output_audio.delta/done``, ``error``

Audio input is segmented with the streaming Silero VAD; when server-side VAD
turn detection is enabled, an utterance automatically triggers ASR and a new
response.  Barge-in (speech while responding) cancels the active response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from uuid import uuid4

from app.core.cancel_scope import CancelScope
from app.services.streaming_vad import StreamingSileroVAD


@dataclass
class RealtimeConfig:
    session_id: str | None = None
    instructions: str = ""
    voice: str | None = None
    language: str = "zh-CN"
    sample_rate: int = 24000
    input_sample_rate: int = 24000
    asr_provider: str | None = None
    llm_provider: str | None = None
    tts_provider: str | None = None
    system_prompt: str | None = None
    server_vad: bool = True
    interrupt_response: bool = True
    stream_text: bool = True
    her: dict = field(default_factory=dict)


class RealtimeConnection:
    """Per-connection Realtime protocol state."""

    def __init__(self, config: RealtimeConfig | None = None, vad: StreamingSileroVAD | None = None) -> None:
        self.config = config or RealtimeConfig()
        self.session_id = self.config.session_id or f"rt_{uuid4().hex[:12]}"
        self.vad = vad or StreamingSileroVAD()
        self.cancel_scope = CancelScope()
        self._item_seq = 0
        self._response_seq = 0
        self.in_response = False
        self.last_transcript = ""
        self._input_pcm = bytearray()
        self._speech_started_sent = False

    # ------------------------------------------------------------------
    # Session / config
    # ------------------------------------------------------------------

    def session_payload(self) -> dict:
        cfg = self.config
        return {
            "id": self.session_id,
            "object": "realtime.session",
            "model": "local",
            "instructions": cfg.instructions,
            "voice": cfg.voice or "alloy",
            "turn_detection": {
                "type": "server_vad" if cfg.server_vad else "none",
                "interrupt_response": cfg.interrupt_response,
            },
            "audio": {
                "input_sample_rate": cfg.input_sample_rate,
                "output_sample_rate": cfg.sample_rate,
            },
            "her": cfg.her,
        }

    def session_created(self) -> dict:
        return {"type": "session.created", "session": self.session_payload()}

    # ------------------------------------------------------------------
    # Client events (synchronous protocol handling)
    # ------------------------------------------------------------------

    def handle_event(self, raw: dict) -> list[dict]:
        event_type = raw.get("type")
        if event_type == "session.update":
            return self._apply_session_update(raw.get("session") or {})
        if event_type == "response.create":
            events: list[dict] = []
            if self.in_response:
                events.append(self._response_done("cancelled"))
                self.cancel_scope.cancel()
            self.in_response = True
            events.append(self._response_created())
            return events
        if event_type == "response.cancel":
            if not self.in_response:
                return []
            self.in_response = False
            self.cancel_scope.cancel()
            return [self._response_done("cancelled")]
        if event_type == "input_audio_buffer.commit":
            return self._commit_input()
        if event_type == "conversation.item.create":
            return self._create_text_item(raw)
        if event_type == "ping":
            return [{"type": "pong", "event_id": raw.get("event_id", "")}]
        return [
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Unsupported event type: {event_type}",
                },
            }
        ]

    def feed_audio(self, pcm: bytes, sample_rate: int | None = None) -> tuple[list[dict], dict | None]:
        """Feed PCM16 audio; returns (events, commit) where commit holds
        ``{"item_id": ..., "pcm": bytes}`` when an utterance was segmented."""
        rate = sample_rate or self.config.input_sample_rate
        result = self.vad.feed(pcm, rate)
        events: list[dict] = []
        if result.speech_active and not self._speech_started_sent:
            self._speech_started_sent = True
            events.append(
                {
                    "type": "input_audio_buffer.speech_started",
                    "audio_start_ms": 0,
                    "item_id": self._new_item_id("item"),
                }
            )
        if not result.speech_active and self._speech_started_sent:
            self._speech_started_sent = False
            events.append(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "audio_end_ms": 0,
                    "item_id": self._new_item_id("item"),
                }
            )
        if result.utterance_pcm:
            commit = self._commit_utterance(result.utterance_pcm)
            if commit is not None:
                events.extend(commit["events"])
                return events, {"item_id": commit["item_id"], "pcm": commit["pcm"]}
        return events, None

    def finish_transcription(self, item_id: str, transcript: str) -> list[dict]:
        """Record the ASR result for a committed audio item."""
        self.last_transcript = transcript
        return [
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "content_index": 0,
                "transcript": transcript,
            }
        ]

    def begin_response(self) -> list[dict]:
        if not self.in_response:
            self.in_response = True
        return [self._response_created()]

    def end_response(self, status: str = "completed") -> list[dict]:
        self.in_response = False
        return [self._response_done(status)]

    def is_stale(self, generation: int) -> bool:
        return self.cancel_scope.is_stale(generation)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_session_update(self, session: dict) -> list[dict]:
        cfg = self.config
        if "instructions" in session:
            cfg.instructions = str(session.get("instructions") or "")
        if "voice" in session:
            cfg.voice = str(session.get("voice") or cfg.voice)
        turn_detection = session.get("turn_detection") or {}
        if turn_detection.get("type") == "server_vad":
            cfg.server_vad = True
        elif turn_detection.get("type") in ("none", "disabled"):
            cfg.server_vad = False
        if "interrupt_response" in turn_detection:
            cfg.interrupt_response = bool(turn_detection["interrupt_response"])
        if session.get("her"):
            cfg.her = dict(session["her"])
            for key in ("asr_provider", "llm_provider", "tts_provider"):
                if key in cfg.her:
                    setattr(cfg, key, cfg.her[key])
        for key in ("asr_provider", "llm_provider", "tts_provider", "system_prompt", "language"):
            if key in session:
                setattr(cfg, key, session[key])
        audio = session.get("audio") or {}
        if isinstance(audio, dict):
            if audio.get("input_sample_rate"):
                cfg.input_sample_rate = int(audio["input_sample_rate"])
            if audio.get("output_sample_rate"):
                cfg.sample_rate = int(audio["output_sample_rate"])
        return [{"type": "session.updated", "session": self.session_payload()}]

    def _new_item_id(self, prefix: str) -> str:
        self._item_seq += 1
        return f"{prefix}_{self.session_id}_{self._item_seq}"

    def _response_created(self) -> dict:
        self._response_seq += 1
        return {
            "type": "response.created",
            "response": {
                "id": f"resp_{self.session_id}_{self._response_seq}",
                "object": "response",
                "status": "in_progress",
                "output": [],
            },
        }

    def _response_done(self, status: str) -> dict:
        return {
            "type": "response.done",
            "response": {
                "id": f"resp_{self.session_id}_{self._response_seq}",
                "object": "response",
                "status": status,
                "output": [],
                "usage": {"total_tokens": 0},
            },
        }

    def _commit_input(self) -> list[dict]:
        if not self._input_pcm:
            return []
        pcm = bytes(self._input_pcm)
        self._input_pcm.clear()
        item_id = self._new_item_id("item")
        return [
            {"type": "input_audio_buffer.committed", "item_id": item_id, "previous_item_id": None},
            {
                "type": "conversation.item.created",
                "item": {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": ""}],
                },
            },
        ]

    def _commit_utterance(self, pcm: bytes) -> dict | None:
        if not pcm:
            return None
        item_id = self._new_item_id("item")
        return {
            "item_id": item_id,
            "pcm": pcm,
            "events": [
                {"type": "input_audio_buffer.committed", "item_id": item_id, "previous_item_id": None},
                {
                    "type": "conversation.item.created",
                    "item": {
                        "id": item_id,
                        "object": "realtime.item",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_audio", "audio": ""}],
                    },
                },
            ],
        }

    def _create_text_item(self, raw: dict) -> list[dict]:
        item = raw.get("item") or {}
        content = item.get("content") or []
        text = ""
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_text":
                text += str(part.get("text") or "")
        if text:
            self.last_transcript = text
        return [
            {
                "type": "conversation.item.created",
                "item": {
                    "id": item.get("id") or self._new_item_id("item"),
                    "object": "realtime.item",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        ]

    @staticmethod
    def now_ms() -> int:
        return int(time.time() * 1000)
