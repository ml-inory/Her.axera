#!/usr/bin/env python3
"""Performance tester for Her.axera voice dialogue pipeline.

Usage:
    python tools/performance_tester.py --backend-url http://localhost:8080
    python tools/performance_tester.py --audio-file tests/fixtures/test.wav --iterations 10
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import time
import wave
from base64 import b64encode
from statistics import mean

import requests
import websocket


def _generate_test_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a silent WAV file for testing."""
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return buf.getvalue()


def test_asr_latency(base_url: str, audio: bytes, provider: str, n: int) -> list[float]:
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        resp = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": ("test.wav", audio, "audio/wav")},
            data={"model": provider, "language": "zh-CN"},
            timeout=60,
        )
        elapsed = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        latencies.append(elapsed)
    return latencies


def test_llm_latency(base_url: str, prompt: str, provider: str, n: int) -> list[float]:
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": provider, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        elapsed = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        latencies.append(elapsed)
    return latencies


def test_tts_latency(base_url: str, text: str, provider: str, n: int) -> list[float]:
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        resp = requests.post(
            f"{base_url}/v1/audio/speech",
            json={"model": provider, "input": text, "voice": "alloy"},
            timeout=60,
        )
        elapsed = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        latencies.append(elapsed)
    return latencies


def test_e2e_pipeline(base_url: str, audio: bytes, n: int) -> list[float]:
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        conn = websocket.create_connection(f"{ws_url}/v1/dialogue/ws", timeout=60)
        try:
            conn.send(json.dumps({
                "type": "audio",
                "audio_base64": b64encode(audio).decode("ascii"),
                "filename": "test.wav",
                "session_id": "perf-test",
                "asr_provider": "mock_asr",
                "llm_provider": "mock_llm",
                "tts_provider": "mock_tts",
                "output_audio_format": "wav",
            }))
            while True:
                event = json.loads(conn.recv())
                if event.get("type") in ("done", "error"):
                    break
        finally:
            conn.close()
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
    return latencies


def _format_row(name: str, provider: str, latencies: list[float]) -> str:
    if not latencies:
        return f"  {name:<25} {provider:<15} {'N/A':>8} {'N/A':>8} {'N/A':>8}"
    sorted_l = sorted(latencies)
    return (
        f"  {name:<25} {provider:<15} "
        f"{min(sorted_l):>8.1f} {mean(sorted_l):>8.1f} {max(sorted_l):>8.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Her.axera Performance Tester")
    parser.add_argument("--backend-url", default="http://localhost:8080")
    parser.add_argument("--audio-file", default=None, help="WAV file for ASR/E2E tests")
    parser.add_argument("--asr-provider", default="mock_asr")
    parser.add_argument("--llm-provider", default="mock_llm")
    parser.add_argument("--tts-provider", default="mock_tts")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    if args.audio_file:
        with open(args.audio_file, "rb") as f:
            audio = f.read()
    else:
        audio = _generate_test_wav()

    n = args.iterations
    url = args.backend_url.rstrip("/")

    print(f"\nHer.axera Performance Test — {url} x{n} iterations\n")
    print(f"  {'Test':<25} {'Provider':<15} {'Min(ms)':>8} {'Avg(ms)':>8} {'Max(ms)':>8}")
    print(f"  {'-'*25} {'-'*15} {'-'*8} {'-'*8} {'-'*8}")

    try:
        print(_format_row("ASR Transcription", args.asr_provider, test_asr_latency(url, audio, args.asr_provider, n)))
    except Exception as e:
        print(f"  ASR: FAILED - {e}")

    try:
        print(_format_row("LLM Chat", args.llm_provider, test_llm_latency(url, "你好", args.llm_provider, n)))
    except Exception as e:
        print(f"  LLM: FAILED - {e}")

    try:
        print(_format_row("TTS Speech", args.tts_provider, test_tts_latency(url, "你好世界", args.tts_provider, n)))
    except Exception as e:
        print(f"  TTS: FAILED - {e}")

    try:
        print(_format_row("E2E Pipeline (WS)", "mock_*", test_e2e_pipeline(url, audio, n)))
    except Exception as e:
        print(f"  E2E: FAILED - {e}")

    print()


if __name__ == "__main__":
    main()
