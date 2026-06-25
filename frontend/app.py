from __future__ import annotations

from base64 import b64decode, b64encode
from io import BytesIO
import json
import os
import tempfile
import wave
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr
import numpy as np
import requests
import websocket

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
FREE_SPEAK_SILENCE_MS = int(os.getenv("FREE_SPEAK_SILENCE_MS", "1000"))
FREE_SPEAK_MIN_UTTERANCE_MS = int(os.getenv("FREE_SPEAK_MIN_UTTERANCE_MS", "600"))
FREE_SPEAK_MAX_BUFFER_MS = int(os.getenv("FREE_SPEAK_MAX_BUFFER_MS", "30000"))
FREE_SPEAK_RMS_THRESHOLD = float(os.getenv("FREE_SPEAK_RMS_THRESHOLD", "450"))
ASR_PROVIDER_CHOICES = [
    item.strip()
    for item in os.getenv("ASR_PROVIDER_CHOICES", "mock_asr,wenet_onnx,sensevoice,fireredasr_aed").split(",")
    if item.strip()
]
DEFAULT_ASR_PROVIDER = os.getenv("DEFAULT_ASR_PROVIDER", "mock_asr")
if DEFAULT_ASR_PROVIDER not in ASR_PROVIDER_CHOICES:
    DEFAULT_ASR_PROVIDER = ASR_PROVIDER_CHOICES[0]
LLM_PROVIDER_CHOICES = ["mock_llm", "deepseek"]
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "mock_llm")
if DEFAULT_LLM_PROVIDER not in LLM_PROVIDER_CHOICES:
    DEFAULT_LLM_PROVIDER = "mock_llm"
TTS_PROVIDER_CHOICES = ["edge_tts", "mock_tts", "kokoro", "zipvoice"]
DEFAULT_TTS_PROVIDER = os.getenv("DEFAULT_TTS_PROVIDER", "edge_tts")
if DEFAULT_TTS_PROVIDER not in TTS_PROVIDER_CHOICES:
    DEFAULT_TTS_PROVIDER = "edge_tts"
SPEAKER_PROVIDER_CHOICES = [
    item.strip()
    for item in os.getenv("SPEAKER_PROVIDER_CHOICES", "mock_speaker,3d_speaker").split(",")
    if item.strip()
]
DEFAULT_SPEAKER_PROVIDER = os.getenv("DEFAULT_SPEAKER_PROVIDER", "mock_speaker")
if DEFAULT_SPEAKER_PROVIDER not in SPEAKER_PROVIDER_CHOICES:
    DEFAULT_SPEAKER_PROVIDER = SPEAKER_PROVIDER_CHOICES[0]
VOICE_CHOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunjianNeural",
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "female_default",
    "male_default",
]
ACTIVE_STREAMS: dict[str, websocket.WebSocket] = {}


def _normalize_audio_array(audio_data: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio_data)
    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.ndim == 2 and audio.shape[0] <= 8 < audio.shape[1]:
        audio = audio.T

    if np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(audio)


def _save_numpy_audio(audio: tuple[int, np.ndarray]) -> str:
    sample_rate, audio_data = audio
    pcm = _normalize_audio_array(audio_data)
    path = Path(tempfile.gettempdir()) / f"her_audio_{uuid4().hex}.wav"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    return str(path)


def _materialize_audio(audio: str | tuple[int, np.ndarray]) -> tuple[str, bool]:
    if isinstance(audio, str):
        return audio, False
    return _save_numpy_audio(audio), True


def _api_to_ws_url(base_url: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :]
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :]
    return base_url


def _chunk_rms(pcm: np.ndarray) -> float:
    if pcm.size == 0:
        return 0.0
    mono = pcm.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(mono))))


def _write_audio_chunks(chunks: list[dict[str, Any]]) -> str | None:
    raw_chunks = [b64decode(item["audio_base64"]) for item in chunks if item.get("audio_base64")]
    if not raw_chunks:
        return None

    audio_format = chunks[0].get("audio_format", "wav")
    output_path = Path(tempfile.gettempdir()) / f"her_ws_tts_{uuid4().hex}.{audio_format}"
    if audio_format == "wav":
        params = None
        frames = []
        for raw in raw_chunks:
            with wave.open(BytesIO(raw), "rb") as wav_file:
                if params is None:
                    params = wav_file.getparams()
                frames.append(wav_file.readframes(wav_file.getnframes()))
        if params is None:
            return None
        with wave.open(str(output_path), "wb") as output_file:
            output_file.setparams(params)
            for frame in frames:
                output_file.writeframes(frame)
    else:
        output_path.write_bytes(b"".join(raw_chunks))
    return str(output_path)


class BackendClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def _open_ws(self) -> websocket.WebSocket:
        return websocket.create_connection(f"{_api_to_ws_url(self.base_url)}/v1/dialogue/ws", timeout=REQUEST_TIMEOUT)

    def _recv_until(self, conn: websocket.WebSocket, terminal_types: set[str]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            event = json.loads(conn.recv())
            events.append(event)
            if event.get("type") == "error":
                raise RuntimeError(event.get("error", {}).get("message", "WebSocket pipeline failed"))
            if event.get("type") in terminal_types:
                return events

    def start_speech_ws(
        self,
        *,
        turn_id: str,
        session_id: str,
        language: str,
        asr_provider: str,
        llm_provider: str,
        llm_api_key: str,
        tts_provider: str,
        voice: str,
        input_sample_rate: int,
        speaker_enabled: bool = False,
        speaker_provider: str | None = None,
    ) -> list[dict[str, Any]]:
        self.close_stream(turn_id)
        conn = self._open_ws()
        ACTIVE_STREAMS[turn_id] = conn
        audio_format = "mp3" if tts_provider == "edge_tts" else "wav"
        conn.send(
            json.dumps(
                {
                    "type": "speech_start",
                    "turn_id": turn_id,
                    "session_id": session_id.strip() or "demo-session",
                    "language": language,
                    "asr_provider": asr_provider,
                    "llm_provider": llm_provider,
                    "llm_api_key": llm_api_key.strip() or None,
                    "tts_provider": tts_provider,
                    "voice": voice,
                    "input_sample_rate": input_sample_rate,
                    "output_audio_format": audio_format,
                    "sample_rate": 24000,
                    "channels": 1,
                    "speaker_enabled": speaker_enabled,
                    "speaker_provider": speaker_provider,
                }
            )
        )
        return self._recv_until(conn, {"speech_started"})

    def send_audio_chunk_ws(self, turn_id: str, pcm: np.ndarray) -> None:
        conn = ACTIVE_STREAMS.get(turn_id)
        if conn is None:
            return
        conn.send(
            json.dumps(
                {
                    "type": "audio_chunk",
                    "turn_id": turn_id,
                    "audio_base64": b64encode(pcm.tobytes()).decode("ascii"),
                }
            )
        )

    def end_speech_ws(self, turn_id: str) -> dict[str, Any]:
        conn = ACTIVE_STREAMS.get(turn_id)
        if conn is None:
            return {"events": [], "asr": {}, "speaker": {}, "llm": {}, "tts_chunks": [], "done": {}, "audio_path": None}
        try:
            conn.send(json.dumps({"type": "speech_end", "turn_id": turn_id}))
            events = self._recv_until(conn, {"done"})
            tts_chunks = [event for event in events if event.get("type") == "tts_sentence"]
            return {
                "events": events,
                "asr": next((event for event in events if event.get("type") == "asr"), {}),
                "speaker": next((event for event in events if event.get("type") == "speaker"), {}),
                "llm": next((event for event in events if event.get("type") == "llm"), {}),
                "tts_chunks": tts_chunks,
                "done": next((event for event in events if event.get("type") == "done"), {}),
                "audio_path": _write_audio_chunks(tts_chunks),
            }
        finally:
            self.close_stream(turn_id)

    def close_stream(self, turn_id: str | None) -> None:
        if not turn_id:
            return
        conn = ACTIVE_STREAMS.pop(turn_id, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def dialogue_ws(
        self,
        audio: str | tuple[int, np.ndarray],
        *,
        session_id: str,
        language: str,
        asr_provider: str,
        llm_provider: str,
        llm_api_key: str,
        tts_provider: str,
        voice: str,
        speaker_enabled: bool = False,
        speaker_provider: str | None = None,
    ) -> dict[str, Any]:
        audio_path, should_delete = _materialize_audio(audio)
        try:
            audio_format = "mp3" if tts_provider == "edge_tts" else "wav"
            conn = websocket.create_connection(f"{_api_to_ws_url(self.base_url)}/v1/dialogue/ws", timeout=REQUEST_TIMEOUT)
            try:
                conn.send(
                    json.dumps(
                        {
                            "type": "audio",
                            "audio_base64": b64encode(Path(audio_path).read_bytes()).decode("ascii"),
                            "filename": Path(audio_path).name,
                            "session_id": session_id.strip() or "demo-session",
                            "language": language,
                            "asr_provider": asr_provider,
                            "llm_provider": llm_provider,
                            "llm_api_key": llm_api_key.strip() or None,
                            "tts_provider": tts_provider,
                            "voice": voice,
                            "output_audio_format": audio_format,
                            "sample_rate": 24000,
                            "speaker_enabled": speaker_enabled,
                            "speaker_provider": speaker_provider,
                        }
                    )
                )
                events: list[dict[str, Any]] = []
                tts_chunks: list[dict[str, Any]] = []
                while True:
                    event = json.loads(conn.recv())
                    events.append(event)
                    if event.get("type") == "error":
                        raise RuntimeError(event.get("error", {}).get("message", "WebSocket pipeline failed"))
                    if event.get("type") == "tts_sentence":
                        tts_chunks.append(event)
                    if event.get("type") == "done":
                        break
            finally:
                conn.close()

            return {
                "events": events,
                "asr": next((event for event in events if event.get("type") == "asr"), {}),
                "speaker": next((event for event in events if event.get("type") == "speaker"), {}),
                "llm": next((event for event in events if event.get("type") == "llm"), {}),
                "tts_chunks": tts_chunks,
                "done": next((event for event in events if event.get("type") == "done"), {}),
                "audio_path": _write_audio_chunks(tts_chunks),
            }
        finally:
            if should_delete:
                Path(audio_path).unlink(missing_ok=True)

    def text_ws(
        self,
        text: str,
        *,
        session_id: str,
        language: str,
        llm_provider: str,
        llm_api_key: str,
        tts_provider: str,
        voice: str,
    ) -> dict[str, Any]:
        audio_format = "mp3" if tts_provider == "edge_tts" else "wav"
        conn = websocket.create_connection(f"{_api_to_ws_url(self.base_url)}/v1/dialogue/ws", timeout=REQUEST_TIMEOUT)
        try:
            conn.send(
                json.dumps(
                    {
                        "type": "text",
                        "text": text,
                        "session_id": session_id.strip() or "demo-session",
                        "language": language,
                        "llm_provider": llm_provider,
                        "llm_api_key": llm_api_key.strip() or None,
                        "tts_provider": tts_provider,
                        "voice": voice,
                        "output_audio_format": audio_format,
                        "sample_rate": 24000,
                    }
                )
            )
            events: list[dict[str, Any]] = []
            tts_chunks: list[dict[str, Any]] = []
            while True:
                event = json.loads(conn.recv())
                events.append(event)
                if event.get("type") == "error":
                    raise RuntimeError(event.get("error", {}).get("message", "WebSocket text pipeline failed"))
                if event.get("type") == "tts_sentence":
                    tts_chunks.append(event)
                if event.get("type") == "done":
                    break
        finally:
            conn.close()

        return {
            "events": events,
            "user_text": next((event for event in events if event.get("type") == "user_text"), {}),
            "llm": next((event for event in events if event.get("type") == "llm"), {}),
            "tts_chunks": tts_chunks,
            "done": next((event for event in events if event.get("type") == "done"), {}),
            "audio_path": _write_audio_chunks(tts_chunks),
        }


client = BackendClient(API_BASE_URL)


def send_text(
    text: str,
    history: list[dict[str, str]] | None,
    session_id: str,
    llm_provider: str,
    llm_api_key: str,
    language: str,
    tts_provider: str,
    voice: str,
) -> tuple[list[dict[str, str]], str, str | None, str]:
    history = history or []
    user_text = text.strip()
    if not user_text:
        gr.Warning("请输入文字。")
        return history, text, None, "等待输入。"

    try:
        result = client.text_ws(
            user_text,
            session_id=session_id,
            language=language,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            tts_provider=tts_provider,
            voice=voice,
        )
        assistant_text = result["llm"].get("text", "")
        history.append({"role": "user", "content": result["user_text"].get("text", user_text)})
        if assistant_text:
            history.append({"role": "assistant", "content": assistant_text})
        status_text = "\n".join(
            [
                f"WebSocket：{API_BASE_URL}/v1/dialogue/ws",
                f"输入：文字",
                f"LLM：{result['llm'].get('provider')} / {result['llm'].get('model')}，耗时 {result['llm'].get('processing_ms')} ms",
                f"TTS：{len(result['tts_chunks'])} 句，累计耗时 {sum(int(item.get('processing_ms') or 0) for item in result['tts_chunks'])} ms",
                f"流水线总耗时：{result['done'].get('total_processing_ms')} ms",
            ]
        )
        return history, "", result.get("audio_path"), status_text
    except Exception as exc:  # noqa: BLE001
        return history, text, None, f"WebSocket 文字流水线失败：{exc}"


def reset_free_speak(enabled: bool) -> tuple[dict[str, Any], str, Any]:
    status_text = "自由说话已开启。" if enabled else "自由说话已关闭。"
    return {}, status_text, gr.update(visible=enabled)


def free_speak_stream(
    audio_chunk: tuple[int, np.ndarray] | None,
    enabled: bool,
    state: dict[str, Any] | None,
    history: list[dict[str, str]] | None,
    session_id: str,
    asr_provider: str,
    llm_provider: str,
    llm_api_key: str,
    language: str,
    tts_provider: str,
    voice: str,
    speaker_enabled: bool,
    speaker_provider: str,
) -> tuple[dict[str, Any], Any, Any, Any]:
    state = state or {}
    history = history or []
    if not enabled or audio_chunk is None:
        return state, gr.update(), gr.update(), gr.update()

    sample_rate, chunk = audio_chunk
    chunk_pcm = _normalize_audio_array(chunk)
    buffer = state.get("buffer")
    if buffer is None or state.get("sample_rate") != sample_rate:
        buffer = chunk_pcm
        speech_start_sample = None
        last_speech_sample = None
        active_turn_id = None
    else:
        speech_start_sample = state.get("speech_start_sample")
        last_speech_sample = state.get("last_speech_sample")
        active_turn_id = state.get("active_turn_id")
        buffer = np.concatenate([buffer, chunk_pcm], axis=0)

    max_samples = int(sample_rate * FREE_SPEAK_MAX_BUFFER_MS / 1000)
    if buffer.shape[0] > max_samples:
        dropped = buffer.shape[0] - max_samples
        buffer = buffer[-max_samples:]
        if speech_start_sample is not None:
            speech_start_sample = max(0, speech_start_sample - dropped)
        if last_speech_sample is not None:
            last_speech_sample = max(0, last_speech_sample - dropped)

    chunk_end_sample = buffer.shape[0]
    chunk_start_sample = max(0, chunk_end_sample - chunk_pcm.shape[0])
    rms = _chunk_rms(chunk_pcm)
    just_started_speech = False
    if rms >= FREE_SPEAK_RMS_THRESHOLD:
        if speech_start_sample is None:
            speech_start_sample = chunk_start_sample
            active_turn_id = f"turn_{uuid4().hex}"
            just_started_speech = True
            try:
                client.start_speech_ws(
                    turn_id=active_turn_id,
                    session_id=session_id,
                    language=language,
                    asr_provider=asr_provider,
                    llm_provider=llm_provider,
                    llm_api_key=llm_api_key,
                    tts_provider=tts_provider,
                    voice=voice,
                    input_sample_rate=int(sample_rate),
                    speaker_enabled=speaker_enabled,
                    speaker_provider=speaker_provider if speaker_enabled else None,
                )
            except Exception as exc:  # noqa: BLE001
                state = {
                    "buffer": buffer,
                    "sample_rate": sample_rate,
                    "speech_start_sample": None,
                    "last_speech_sample": None,
                    "active_turn_id": None,
                }
                return state, history, gr.update(), f"WebSocket 建立语音流失败：{exc}"
        last_speech_sample = chunk_end_sample

    if active_turn_id is not None:
        try:
            client.send_audio_chunk_ws(active_turn_id, chunk_pcm)
        except Exception as exc:  # noqa: BLE001
            client.close_stream(active_turn_id)
            state = {
                "buffer": buffer,
                "sample_rate": sample_rate,
                "speech_start_sample": None,
                "last_speech_sample": None,
                "active_turn_id": None,
            }
            return state, history, gr.update(), f"WebSocket 发送音频分片失败：{exc}"

    state = {
        "buffer": buffer,
        "sample_rate": sample_rate,
        "speech_start_sample": speech_start_sample,
        "last_speech_sample": last_speech_sample,
        "active_turn_id": active_turn_id,
    }
    duration_ms = int(buffer.shape[0] * 1000 / sample_rate)
    if speech_start_sample is None or last_speech_sample is None:
        return state, gr.update(), gr.update(), f"自由说话：监听中 {duration_ms} ms，RMS {rms:.0f}"

    speech_duration_ms = int((last_speech_sample - speech_start_sample) * 1000 / sample_rate)
    trailing_silence_ms = int((buffer.shape[0] - last_speech_sample) * 1000 / sample_rate)
    if speech_duration_ms < FREE_SPEAK_MIN_UTTERANCE_MS or trailing_silence_ms < FREE_SPEAK_SILENCE_MS:
        if just_started_speech:
            return state, gr.update(), None, "检测到新语音，已停止当前播报。"
        return state, gr.update(), gr.update(), (
            f"自由说话：采集中 {duration_ms} ms，语音 {speech_duration_ms} ms，静音 {trailing_silence_ms} ms"
        )

    remaining = buffer[last_speech_sample:]
    completed_turn_id = active_turn_id
    state = {
        "buffer": remaining,
        "sample_rate": sample_rate,
        "speech_start_sample": None,
        "last_speech_sample": None,
        "active_turn_id": None,
    }

    try:
        result = client.end_speech_ws(str(completed_turn_id))
        user_text = result["asr"].get("text", "")
        assistant_text = result["llm"].get("text", "")
        if user_text:
            history.append({"role": "user", "content": user_text})
        if assistant_text:
            history.append({"role": "assistant", "content": assistant_text})
        status_lines = [
            f"WebSocket：{API_BASE_URL}/v1/dialogue/ws",
            f"ASR：{result['asr'].get('provider')} / {result['asr'].get('model')}，耗时 {result['asr'].get('processing_ms')} ms",
        ]
        speaker_info = result.get("speaker") or {}
        if speaker_info:
            speaker_name = speaker_info.get("speaker_label") or speaker_info.get("speaker_id", "")
            status_lines.append(
                f"说话人：{speaker_name}，置信度 {speaker_info.get('confidence', 0):.2f}，耗时 {speaker_info.get('processing_ms')} ms"
            )
        status_lines.extend([
            f"LLM：{result['llm'].get('provider')} / {result['llm'].get('model')}，耗时 {result['llm'].get('processing_ms')} ms",
            f"TTS：{len(result['tts_chunks'])} 句，累计耗时 {sum(int(item.get('processing_ms') or 0) for item in result['tts_chunks'])} ms",
            f"流水线总耗时：{result['done'].get('total_processing_ms')} ms",
        ])
        return state, history, result.get("audio_path"), "\n".join(status_lines)
    except Exception as exc:  # noqa: BLE001
        return state, history, gr.update(), f"WebSocket 流水线失败：{exc}"


def clear_dialogue() -> tuple[list[dict[str, str]], str, str | None, str]:
    return [], "", None, "已清空对话。"


def check_backend() -> str:
    try:
        payload = client.health()
        return f"连接成功：{payload}"
    except Exception as exc:  # noqa: BLE001
        return f"连接失败：{exc}"


with gr.Blocks(title="Her 语音对话 Demo") as demo:
    gr.Markdown(
        """
        # Her 语音对话 Demo

        文字输入和自由说话都通过 WebSocket 流水线输出语音回复。
        """
    )

    with gr.Row():
        api_base = gr.Textbox(value=API_BASE_URL, label="后端地址", interactive=False)
        session_id = gr.Textbox(value="demo-session", label="Session ID")
        asr_provider = gr.Dropdown(
            choices=ASR_PROVIDER_CHOICES,
            value=DEFAULT_ASR_PROVIDER,
            label="ASR Provider",
        )
        llm_provider = gr.Dropdown(
            choices=LLM_PROVIDER_CHOICES,
            value=DEFAULT_LLM_PROVIDER,
            label="LLM Provider",
        )
        llm_api_key = gr.Textbox(value="", label="LLM API KEY", type="password")
        language = gr.Dropdown(choices=["zh-CN", "en-US"], value="zh-CN", label="语言")
        tts_provider = gr.Dropdown(choices=TTS_PROVIDER_CHOICES, value=DEFAULT_TTS_PROVIDER, label="TTS Provider")
        voice = gr.Dropdown(choices=VOICE_CHOICES, value="zh-CN-XiaoxiaoNeural", label="音色")
        speaker_enabled = gr.Checkbox(value=False, label="说话人识别")
        speaker_provider = gr.Dropdown(
            choices=SPEAKER_PROVIDER_CHOICES,
            value=DEFAULT_SPEAKER_PROVIDER,
            label="Speaker Provider",
        )
        free_speak_enabled = gr.Checkbox(value=False, label="自由说话")

    chatbot = gr.Chatbot(label="对话", type="messages", height=460)

    text_input = gr.Textbox(label="文字输入", placeholder="输入文字后发送。", lines=3)

    free_audio_input = gr.Audio(
        label="自由说话输入",
        sources=["microphone"],
        type="numpy",
        streaming=True,
        visible=False,
    )

    with gr.Row():
        send_button = gr.Button("发送", variant="primary")
        clear_button = gr.Button("清空")
        health_button = gr.Button("检查后端")

    audio_output = gr.Audio(label="语音回复", type="filepath", autoplay=True)
    status = gr.Textbox(label="状态", value="等待输入。", lines=5)
    free_speak_state = gr.State({})

    send_button.click(
        send_text,
        inputs=[text_input, chatbot, session_id, llm_provider, llm_api_key, language, tts_provider, voice],
        outputs=[chatbot, text_input, audio_output, status],
    )
    text_input.submit(
        send_text,
        inputs=[text_input, chatbot, session_id, llm_provider, llm_api_key, language, tts_provider, voice],
        outputs=[chatbot, text_input, audio_output, status],
    )
    free_speak_enabled.change(
        reset_free_speak,
        inputs=free_speak_enabled,
        outputs=[free_speak_state, status, free_audio_input],
    )
    free_audio_input.stream(
        free_speak_stream,
        inputs=[
            free_audio_input,
            free_speak_enabled,
            free_speak_state,
            chatbot,
            session_id,
            asr_provider,
            llm_provider,
            llm_api_key,
            language,
            tts_provider,
            voice,
            speaker_enabled,
            speaker_provider,
        ],
        outputs=[free_speak_state, chatbot, audio_output, status],
        stream_every=0.5,
        concurrency_limit=1,
    )
    clear_button.click(clear_dialogue, outputs=[chatbot, text_input, audio_output, status])
    health_button.click(check_backend, outputs=status)


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("FRONTEND_HOST", "0.0.0.0"), server_port=int(os.getenv("FRONTEND_PORT", "7860")))
