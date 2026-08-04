"""WebRTC transport for the OpenAI Realtime API emulation.

Adapted from huggingface/speech-to-speech.  Audio travels over RTP media
tracks (Opus at 48 kHz, resampled to/from 16 kHz pipeline rate); all JSON
events use the same protocol as the WebSocket transport, carried on the
``oai-events`` data channel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from fractions import Fraction

import av
import numpy as np
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

logger = logging.getLogger(__name__)


WEBRTC_SAMPLE_RATE = 48_000
PIPELINE_SAMPLE_RATE = 16_000
AUDIO_PTIME = 0.02
WEBRTC_FRAME_SAMPLES = int(WEBRTC_SAMPLE_RATE * AUDIO_PTIME)
DATA_CHANNEL_LABEL = "oai-events"
ICE_SERVERS_ENV = "HER_AXERA_ICE_SERVERS"
ICE_GATHERING_TIMEOUT_S = 5.0
CONNECT_TIMEOUT_S = 30.0


def rtc_configuration_from_env() -> RTCConfiguration | None:
    """Build an RTCConfiguration from HER_AXERA_ICE_SERVERS (JSON list)."""
    raw = os.environ.get(ICE_SERVERS_ENV)
    if not raw:
        return None
    try:
        servers = [RTCIceServer(**entry) for entry in json.loads(raw)]
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Ignoring invalid %s: %s", ICE_SERVERS_ENV, exc)
        return None
    return RTCConfiguration(iceServers=servers)


class PcmResampler:
    """Stateful mono/s16 resampler around av.AudioResampler."""

    def __init__(self, target_rate: int) -> None:
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
        self._pts = 0

    def resample_frame(self, frame: av.AudioFrame) -> bytes:
        out = bytearray()
        for resampled in self._resampler.resample(frame):
            out += resampled.to_ndarray().tobytes()
        return bytes(out)

    def resample_pcm(self, pcm: bytes, src_rate: int) -> bytes:
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return b""
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = src_rate
        frame.pts = self._pts
        frame.time_base = Fraction(1, src_rate)
        self._pts += samples.shape[0]
        return self.resample_frame(frame)


class PipelineAudioTrack(MediaStreamTrack):
    """Outbound audio track: paced 20 ms 48 kHz frames from a PCM buffer."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        self._start: float | None = None
        self._timestamp = 0

    def write(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)

    def clear(self) -> None:
        del self._buffer[:]

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        if self._start is None:
            self._start = time.time()
            self._timestamp = 0
        else:
            self._timestamp += WEBRTC_FRAME_SAMPLES
            wait = self._start + (self._timestamp / WEBRTC_SAMPLE_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        needed = WEBRTC_FRAME_SAMPLES * 2
        payload = bytes(self._buffer[:needed])
        del self._buffer[: len(payload)]
        if len(payload) < needed:
            payload += b"\x00" * (needed - len(payload))

        samples = np.frombuffer(payload, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, WEBRTC_SAMPLE_RATE)
        return frame


class WebRTCDialogueSession:
    """One WebRTC peer connection bound to a RealtimeDialogueEngine."""

    kind = "webrtc"

    def __init__(
        self,
        pc: RTCPeerConnection,
        *,
        on_client_event: Callable[[dict], Awaitable[None]],
        on_audio: Callable[[bytes], Awaitable[None]],
        on_open: Callable[[], Awaitable[None]],
        on_closed: Callable[[], None],
    ) -> None:
        self._pc = pc
        self._on_client_event = on_client_event
        self._on_audio = on_audio
        self._on_open = on_open
        self._on_closed = on_closed
        self._dc = None
        self._closed = False
        self._track = PipelineAudioTrack()
        self._out_resampler = PcmResampler(WEBRTC_SAMPLE_RATE)
        self._in_resampler = PcmResampler(PIPELINE_SAMPLE_RATE)
        self._dc_messages: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Wire aiortc callbacks. Call before negotiate()."""
        self._pc.addTrack(self._track)

        @self._pc.on("datachannel")
        def on_datachannel(dc) -> None:
            if dc.label != DATA_CHANNEL_LABEL:
                logger.warning("[WebRTC] Ignoring unexpected data channel: %s", dc.label)
                return
            self._dc = dc
            self._spawn(self._consume_dc_messages())

            if dc.readyState == "open":
                self._spawn(self._on_open())
            else:

                @dc.on("open")
                def on_dc_open() -> None:
                    self._spawn(self._on_open())

            @dc.on("message")
            def on_message(msg) -> None:
                if isinstance(msg, str):
                    self._dc_messages.put_nowait(msg)

            @dc.on("close")
            def on_dc_close() -> None:
                self._spawn(self.close())

        @self._pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                self._spawn(self._consume_inbound_audio(track))

        @self._pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            if self._pc.connectionState in ("failed", "closed"):
                await self.close()

    async def negotiate(self, offer_sdp: str) -> str:
        """Apply the offer, create the answer and wait for ICE gathering."""
        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        if self._pc.iceGatheringState != "complete":
            done: asyncio.Event = asyncio.Event()

            @self._pc.on("icegatheringstatechange")
            def on_ice_change() -> None:
                if self._pc.iceGatheringState == "complete":
                    done.set()

            if self._pc.iceGatheringState == "complete":
                done.set()
            try:
                await asyncio.wait_for(done.wait(), timeout=ICE_GATHERING_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning("[WebRTC] ICE gathering timed out, returning partial SDP")

        self._spawn(self._connect_watchdog())
        return self._pc.localDescription.sdp

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        self._track.stop()
        try:
            await self._pc.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[WebRTC] Error closing peer connection: %s", exc)
        self._on_closed()

    # ------------------------------------------------------------------
    # Transport interface
    # ------------------------------------------------------------------

    async def send_event(self, event: dict) -> None:
        dc = self._dc
        if dc is None or dc.readyState != "open":
            return
        try:
            dc.send(json.dumps(event, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            logger.error("[WebRTC] Data channel send error: %s", exc)

    def write_audio(self, pcm: bytes, sample_rate: int) -> None:
        self._track.write(self._out_resampler.resample_pcm(pcm, sample_rate))

    def discard_pending_audio(self) -> None:
        self._track.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.append(task)

    async def _connect_watchdog(self) -> None:
        await asyncio.sleep(CONNECT_TIMEOUT_S)
        if not self._closed and self._pc.connectionState != "connected":
            logger.warning("[WebRTC] Peer not connected after %ss; releasing", CONNECT_TIMEOUT_S)
            await self.close()

    async def _consume_dc_messages(self) -> None:
        while not self._closed:
            msg = await self._dc_messages.get()
            try:
                raw = json.loads(msg)
            except json.JSONDecodeError:
                logger.error("[WebRTC] Invalid JSON on data channel: %r", msg)
                continue
            if not isinstance(raw, dict):
                continue
            try:
                await self._on_client_event(raw)
            except Exception:  # noqa: BLE001
                logger.exception("[WebRTC] Error handling client event")

    async def _consume_inbound_audio(self, track) -> None:
        while not self._closed:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break
            pcm = self._in_resampler.resample_frame(frame)
            if pcm:
                try:
                    await self._on_audio(pcm)
                except Exception:  # noqa: BLE001
                    logger.exception("[WebRTC] Error handling inbound audio")
