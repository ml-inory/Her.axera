"""WebRTC transport tests: unit-level pieces + one end-to-end call.

The e2e test runs a real uvicorn server in a subprocess: aiortc peers on
separate event loops (e.g. inside TestClient's portal) do not reliably
complete ICE, while loopback UDP between two processes does.
"""

import asyncio
from collections import deque
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave

import av
import numpy as np
import pytest
from fractions import Fraction

aiortc = pytest.importorskip("aiortc")
from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription  # noqa: E402
from aiortc.mediastreams import MediaStreamError  # noqa: E402

from app.services.webrtc_service import (  # noqa: E402
    PcmResampler,
    PipelineAudioTrack,
    WEBRTC_FRAME_SAMPLES,
    WEBRTC_SAMPLE_RATE,
    rtc_configuration_from_env,
)


BACKEND_DIR = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech_zh.wav"


class SpeechTrack(AudioStreamTrack):
    """Sends pre-built 48 kHz s16 mono frames, then ends the stream."""

    def __init__(self, frames: list[av.AudioFrame]) -> None:
        super().__init__()
        self._frames = deque(frames)
        self._pts = 0

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if not self._frames:
            raise MediaStreamError
        frame = self._frames.popleft()
        frame.pts = self._pts
        frame.time_base = Fraction(1, WEBRTC_SAMPLE_RATE)
        self._pts += WEBRTC_FRAME_SAMPLES
        return frame


def _fixture_pcm48k() -> bytes:
    with wave.open(str(FIXTURE), "rb") as wav:
        pcm16 = wav.readframes(wav.getnframes())
    pcm16 += bytes(16000 * 2)  # 1s trailing silence
    samples = np.repeat(np.frombuffer(pcm16, dtype=np.int16), 3)
    return samples.astype(np.int16).tobytes()


def _make_frames(pcm48k: bytes) -> list[av.AudioFrame]:
    samples = np.frombuffer(pcm48k, dtype=np.int16)
    usable = len(samples) // WEBRTC_FRAME_SAMPLES * WEBRTC_FRAME_SAMPLES
    frames = []
    for i in range(0, usable, WEBRTC_FRAME_SAMPLES):
        chunk = samples[i : i + WEBRTC_FRAME_SAMPLES]
        frame = av.AudioFrame.from_ndarray(chunk[None, :], format="s16", layout="mono")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frames.append(frame)
    return frames


@pytest.fixture(scope="module")
def webrtc_server():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    tmp_root = tempfile.mkdtemp(prefix="her-webrtc-models-")
    env = os.environ.copy()
    env["HER_AXERA_MODEL_ROOT"] = tmp_root
    env["AX_TTS_MODEL_PATH"] = str(Path(tmp_root) / "tts")
    env["AX_ASR_MODEL_PATH"] = str(Path(tmp_root) / "asr")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=str(BACKEND_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 40
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"{base}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.3)
    else:
        proc.kill()
        raise RuntimeError("WebRTC test server failed to start")
    if proc.poll() is not None:
        output = proc.stdout.read().decode() if proc.stdout else ""
        raise RuntimeError(f"WebRTC test server exited early: {output[-1000:]}")
    yield base
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


class TestPipelineAudioTrack:
    def test_paces_frames_and_pads_with_silence(self) -> None:
        track = PipelineAudioTrack()
        track.write(b"\x01\x00" * WEBRTC_FRAME_SAMPLES)  # exactly one 20ms frame

        async def run() -> list[bytes]:
            payloads = []
            for _ in range(2):
                frame = await track.recv()
                payloads.append(frame.to_ndarray().tobytes())
            return payloads

        payloads = asyncio.run(run())
        assert payloads[0] == b"\x01\x00" * WEBRTC_FRAME_SAMPLES
        # Second frame is silence-padded.
        assert payloads[1] == b"\x00\x00" * WEBRTC_FRAME_SAMPLES

    def test_clear_drops_buffered_audio(self) -> None:
        track = PipelineAudioTrack()
        track.write(b"\x01\x00" * (WEBRTC_FRAME_SAMPLES * 2))
        track.clear()
        assert track.buffered_bytes == 0


class TestPcmResampler:
    def test_downsample_48k_to_16k(self) -> None:
        resampler = PcmResampler(16000)
        pcm = (np.ones(48000, dtype=np.int16) * 100).tobytes()
        out = resampler.resample_pcm(pcm, 48000)
        # av keeps the final partial output frame until flush; allow a small tail.
        assert abs(len(out) - 16000 * 2) < 64


class TestRTCConfig:
    def test_from_env_json(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "HER_AXERA_ICE_SERVERS",
            json.dumps([{"urls": "stun:stun.example.com:3478"}]),
        )
        config = rtc_configuration_from_env()
        assert config is not None
        assert config.iceServers[0].urls == "stun:stun.example.com:3478"

    def test_invalid_env_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("HER_AXERA_ICE_SERVERS", "not-json")
        assert rtc_configuration_from_env() is None


class TestWebRTCEndToEnd:
    def test_full_voice_conversation(self, webrtc_server) -> None:
        events: list[str] = []
        result = {"got_audio": False}

        async def run() -> None:
            pc = RTCPeerConnection()
            dc = pc.createDataChannel("oai-events")
            pc.addTrack(SpeechTrack(_make_frames(_fixture_pcm48k())))
            created = asyncio.Event()

            @dc.on("message")
            def on_message(raw) -> None:
                event = json.loads(raw)
                events.append(event["type"])
                if event["type"] == "session.created":
                    created.set()

            @pc.on("track")
            def on_track(track) -> None:
                async def reader() -> None:
                    while True:
                        try:
                            frame = await track.recv()
                            if np.abs(frame.to_ndarray()).max() > 100:
                                result["got_audio"] = True
                        except Exception:
                            break

                asyncio.ensure_future(reader())

            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            request = urllib.request.Request(
                f"{webrtc_server}/v1/realtime/calls",
                data=offer.sdp.encode(),
                headers={"content-type": "application/sdp"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                answer = response.read().decode()
                assert response.status == 201
            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))
            await asyncio.wait_for(created.wait(), timeout=15)

            dc.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "her": {
                                "asr_provider": "mock_asr",
                                "llm_provider": "mock_llm",
                                "tts_provider": "mock_tts",
                            }
                        },
                    }
                )
            )
            deadline = time.time() + 20
            while "response.done" not in events and time.time() < deadline:
                await asyncio.sleep(0.2)
            await pc.close()

        asyncio.run(run())

        assert "session.created" in events
        assert "input_audio_buffer.speech_started" in events
        assert "conversation.item.input_audio_transcription.completed" in events
        assert "response.output_text.delta" in events
        assert "response.done" in events
        assert result["got_audio"] is True
