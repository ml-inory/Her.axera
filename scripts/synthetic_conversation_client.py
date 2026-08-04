#!/usr/bin/env python3
"""Synthetic conversation stress client for Her.axera.

Drives the /v1/dialogue/ws endpoint with scripted turns and reports per-stage
latencies (first LLM token, TTS synthesis, end-to-end) so latency regressions
on the AX board can be measured repeatably.

Usage:
    python scripts/synthetic_conversation_client.py --host 127.0.0.1 --port 8080
    python scripts/synthetic_conversation_client.py --audio path/to/speech.wav --turns 5
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import wave

try:
    import websockets
except ImportError:  # pragma: no cover
    sys.stderr.write("websockets is required: pip install websockets\n")
    sys.exit(2)


DEFAULT_TURNS = [
    "你好，今天天气怎么样？",
    "帮我设置一个明天早上七点的闹钟。",
    "讲个笑话吧。",
    "现在几点了？",
    "谢谢，再见。",
]


class TurnReport:
    def __init__(self, index: int, text: str) -> None:
        self.index = index
        self.text = text
        self.accepted_ms: float | None = None
        self.first_llm_ms: float | None = None
        self.tts_ms: float = 0.0
        self.tts_sentences = 0
        self.total_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "turn": self.index,
            "text": self.text,
            "accepted_ms": self.accepted_ms,
            "first_llm_token_ms": self.first_llm_ms,
            "tts_synthesis_ms": round(self.tts_ms, 1),
            "tts_sentences": self.tts_sentences,
            "total_ms": self.total_ms,
        }


async def run_text_turn(ws, report: TurnReport, options: dict) -> None:
    start = time.monotonic()
    await ws.send(
        json.dumps(
            {
                "type": "text",
                "text": report.text,
                "turn_id": f"synth_{report.index}",
                **options,
            }
        )
    )
    async for raw in ws:
        event = json.loads(raw)
        if event["type"] == "accepted":
            report.accepted_ms = (time.monotonic() - start) * 1000
        elif event["type"] == "llm_started":
            report.first_llm_ms = (time.monotonic() - start) * 1000
        elif event["type"] == "tts_sentence":
            report.tts_sentences += 1
            report.tts_ms += float(event.get("processing_ms") or 0)
        elif event["type"] == "done":
            report.total_ms = float(event.get("total_processing_ms") or (time.monotonic() - start) * 1000)
            return


async def run_audio_turn(ws, report: TurnReport, options: dict, wav_bytes: bytes) -> None:
    start = time.monotonic()
    await ws.send(
        json.dumps(
            {
                "type": "audio",
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "filename": "synthetic.wav",
                "turn_id": f"synth_audio_{report.index}",
                **options,
            }
        )
    )
    async for raw in ws:
        event = json.loads(raw)
        if event["type"] == "asr":
            report.accepted_ms = (time.monotonic() - start) * 1000
        elif event["type"] == "llm_started":
            report.first_llm_ms = (time.monotonic() - start) * 1000
        elif event["type"] == "tts_sentence":
            report.tts_sentences += 1
            report.tts_ms += float(event.get("processing_ms") or 0)
        elif event["type"] == "done":
            report.total_ms = float(event.get("total_processing_ms") or (time.monotonic() - start) * 1000)
            return


async def main(args: argparse.Namespace) -> int:
    uri = f"ws://{args.host}:{args.port}/v1/dialogue/ws"
    options = {
        "session_id": args.session or f"synth_{int(time.time())}",
        "llm_provider": args.llm_provider,
        "tts_provider": args.tts_provider,
        "asr_provider": args.asr_provider,
        "output_audio_format": "wav",
    }
    if args.audio:
        with wave.open(args.audio, "rb") as wav:
            assert wav.getnchannels() == 1, "audio must be mono"
            assert wav.getsampwidth() == 2, "audio must be 16-bit PCM"
            wav_bytes = wav.readframes(wav.getnframes())
    else:
        wav_bytes = None

    turns = (args.turns if args.turns and args.turns > 0 else len(DEFAULT_TURNS))
    texts = DEFAULT_TURNS[:turns]
    reports: list[TurnReport] = []

    async with websockets.connect(uri) as ws:
        for index, text in enumerate(texts):
            report = TurnReport(index, text)
            reports.append(report)
            if wav_bytes:
                await run_audio_turn(ws, report, options, wav_bytes)
            else:
                await run_text_turn(ws, report, options)

    print(json.dumps({"reports": [r.to_dict() for r in reports]}, ensure_ascii=False, indent=2))
    failures = [r for r in reports if r.total_ms is None or r.tts_sentences == 0]
    if failures:
        print(f"[FAIL] {len(failures)}/{len(reports)} turns incomplete", file=sys.stderr)
        return 1
    avg_total = sum(r.total_ms or 0 for r in reports) / len(reports)
    avg_first = sum(r.first_llm_ms or 0 for r in reports if r.first_llm_ms) / max(1, len([r for r in reports if r.first_llm_ms]))
    print(f"[OK] turns={len(reports)} avg_total_ms={avg_total:.0f} avg_first_llm_ms={avg_first:.0f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic conversation stress client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--turns", type=int, default=0, help="number of turns (default: all built-in)")
    parser.add_argument("--session", default="", help="session id (default: generated)")
    parser.add_argument("--llm_provider", default="mock_llm")
    parser.add_argument("--tts_provider", default="mock_tts")
    parser.add_argument("--asr_provider", default="mock_asr")
    parser.add_argument("--audio", default="", help="mono 16-bit WAV for audio turns (replaces text)")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    try:
        sys.exit(asyncio.run(main(args)))
    except KeyboardInterrupt:
        sys.exit(130)
