from __future__ import annotations

import base64
import os
import tempfile
import wave
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr
import numpy as np
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
FREE_SPEAK_SILENCE_MS = int(os.getenv("FREE_SPEAK_SILENCE_MS", "1000"))
FREE_SPEAK_MIN_UTTERANCE_MS = int(os.getenv("FREE_SPEAK_MIN_UTTERANCE_MS", "600"))
FREE_SPEAK_MAX_BUFFER_MS = int(os.getenv("FREE_SPEAK_MAX_BUFFER_MS", "30000"))


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


class BackendClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def transcribe(self, audio: str | tuple[int, np.ndarray], language: str, enable_vad: bool = False) -> dict[str, Any]:
        audio_path, should_delete = _materialize_audio(audio)
        try:
            return self._post_audio(
                endpoint="/v1/asr/transcriptions",
                audio_path=audio_path,
                data={
                    "language": language,
                    "enable_timestamps": "true",
                    "enable_vad": "true" if enable_vad else "false",
                },
            )
        finally:
            if should_delete:
                Path(audio_path).unlink(missing_ok=True)

    def detect_vad(self, audio: str | tuple[int, np.ndarray]) -> dict[str, Any]:
        audio_path, should_delete = _materialize_audio(audio)
        try:
            return self._post_audio(endpoint="/v1/asr/vad/segments", audio_path=audio_path, data={})
        finally:
            if should_delete:
                Path(audio_path).unlink(missing_ok=True)

    def _post_audio(self, *, endpoint: str, audio_path: str, data: dict[str, str]) -> dict[str, Any]:
        with open(audio_path, "rb") as audio_file:
            files = {"audio": (Path(audio_path).name, audio_file, "application/octet-stream")}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                files=files,
                data=data,
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        return response.json()

    def chat(
        self,
        messages: list[dict[str, str]],
        session_id: str | None,
        provider: str | None,
        api_key: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"messages": messages, "max_tokens": 512}
        if session_id:
            payload["session_id"] = session_id
        if provider and provider.strip():
            payload["provider"] = provider.strip()
        if api_key and api_key.strip():
            payload["api_key"] = api_key.strip()
        response = requests.post(
            f"{self.base_url}/v1/llm/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def synthesize(self, text: str, voice: str, language: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/v1/tts/speech",
            json={
                "text": text,
                "voice": voice,
                "language": language,
                "audio_format": "wav",
                "sample_rate": 24000,
                "return_audio_base64": True,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()


client = BackendClient(API_BASE_URL)


def _to_llm_messages(history: list[dict[str, str]], user_text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "你是一个简洁、友好的语音助手。"},
    ]
    for item in history[-10:]:
        if item.get("role") == "user":
            messages.append({"role": "user", "content": item.get("content", "")})
        elif item.get("role") == "assistant":
            messages.append({"role": "assistant", "content": item.get("content", "")})
    messages.append({"role": "user", "content": user_text})
    return messages


def _save_audio_base64(audio_base64: str, audio_format: str) -> str:
    suffix = f".{audio_format or 'wav'}"
    path = Path(tempfile.gettempdir()) / f"her_tts_{uuid4().hex}{suffix}"
    path.write_bytes(base64.b64decode(audio_base64))
    return str(path)


def _format_status(asr_result: dict[str, Any] | None, llm_result: dict[str, Any], tts_result: dict[str, Any]) -> str:
    lines = [f"后端：{API_BASE_URL}"]
    if asr_result:
        lines.append(
            f"ASR：{asr_result.get('provider')} / {asr_result.get('model')}，"
            f"耗时 {asr_result.get('processing_ms')} ms"
        )
        if asr_result.get("vad_segments"):
            lines.append(
                f"VAD：{len(asr_result.get('vad_segments', []))} 段，"
                f"语音 {asr_result.get('speech_duration_ms')} ms，"
                f"耗时 {asr_result.get('vad_processing_ms')} ms"
            )
    lines.append(f"LLM：{llm_result.get('provider')} / {llm_result.get('model')}，耗时 {llm_result.get('processing_ms')} ms")
    lines.append(f"TTS：{tts_result.get('provider')} / {tts_result.get('model')}，耗时 {tts_result.get('processing_ms')} ms")
    lines.append(f"音频时长：{tts_result.get('duration_ms')} ms")
    return "\n".join(lines)


def _respond(
    text: str,
    audio_path: str | None,
    history: list[dict[str, str]] | None,
    session_id: str,
    llm_provider: str,
    llm_api_key: str,
    language: str,
    voice: str,
    enable_vad: bool,
) -> tuple[list[dict[str, str]], str, str | None, str, None]:
    history = history or []
    user_text = text.strip()
    asr_result: dict[str, Any] | None = None

    try:
        if audio_path:
            asr_result = client.transcribe(audio_path, language, enable_vad=enable_vad)
            user_text = asr_result.get("text", "").strip() or user_text

        if not user_text:
            gr.Warning("请输入文字或上传/录制一段语音。")
            return history, "", None, "等待输入。", None

        llm_result = client.chat(
            _to_llm_messages(history, user_text),
            session_id.strip() or None,
            llm_provider,
            llm_api_key,
        )
        assistant_text = llm_result["message"]["content"]
        tts_result = client.synthesize(assistant_text, voice, language)

        output_audio_path = None
        if tts_result.get("audio_base64"):
            output_audio_path = _save_audio_base64(tts_result["audio_base64"], tts_result.get("audio_format", "wav"))

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
        return history, "", output_audio_path, _format_status(asr_result, llm_result, tts_result), None
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        gr.Error(f"后端接口返回错误：{detail}")
    except requests.RequestException as exc:
        gr.Error(f"无法连接后端：{exc}")
    except Exception as exc:  # noqa: BLE001
        gr.Error(f"处理失败：{exc}")

    return history, text, None, "请求失败，请检查后端服务和配置。", audio_path


def respond(
    text: str,
    audio_path: str | None,
    history: list[dict[str, str]] | None,
    session_id: str,
    llm_provider: str,
    llm_api_key: str,
    language: str,
    voice: str,
) -> tuple[list[dict[str, str]], str, str | None, str, None]:
    return _respond(text, audio_path, history, session_id, llm_provider, llm_api_key, language, voice, enable_vad=False)


def reset_free_speak(enabled: bool) -> tuple[dict[str, Any], str, Any, Any]:
    status_text = "自由说话已开启。" if enabled else "自由说话已关闭。"
    return {}, status_text, gr.update(visible=not enabled), gr.update(visible=enabled)


def free_speak_stream(
    audio_chunk: tuple[int, np.ndarray] | None,
    enabled: bool,
    state: dict[str, Any] | None,
    history: list[dict[str, str]] | None,
    session_id: str,
    llm_provider: str,
    llm_api_key: str,
    language: str,
    voice: str,
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
    else:
        buffer = np.concatenate([buffer, chunk_pcm], axis=0)

    max_samples = int(sample_rate * FREE_SPEAK_MAX_BUFFER_MS / 1000)
    if buffer.shape[0] > max_samples:
        buffer = buffer[-max_samples:]

    state = {"buffer": buffer, "sample_rate": sample_rate}
    duration_ms = int(buffer.shape[0] * 1000 / sample_rate)
    if duration_ms < FREE_SPEAK_MIN_UTTERANCE_MS:
        return state, gr.update(), gr.update(), f"自由说话：监听中 {duration_ms} ms"

    try:
        vad_result = client.detect_vad((sample_rate, buffer))
    except requests.RequestException as exc:
        return state, gr.update(), gr.update(), f"VAD 检测失败：{exc}"

    candidate = None
    for segment in vad_result.get("segments", []):
        segment_duration = segment["end_ms"] - segment["start_ms"]
        trailing_silence = duration_ms - segment["end_ms"]
        if segment_duration >= FREE_SPEAK_MIN_UTTERANCE_MS and trailing_silence >= FREE_SPEAK_SILENCE_MS:
            candidate = segment
            break

    if candidate is None:
        return state, gr.update(), gr.update(), f"自由说话：监听中 {duration_ms} ms"

    start_sample = max(0, int(candidate["start_ms"] * sample_rate / 1000))
    end_sample = min(buffer.shape[0], int(candidate["end_ms"] * sample_rate / 1000))
    utterance = buffer[start_sample:end_sample]
    remaining = buffer[end_sample:]
    state = {"buffer": remaining, "sample_rate": sample_rate}

    utterance_path = _save_numpy_audio((sample_rate, utterance))
    try:
        new_history, _, output_audio_path, status_text, _ = _respond(
            "",
            utterance_path,
            history,
            session_id,
            llm_provider,
            llm_api_key,
            language,
            voice,
            enable_vad=False,
        )
    finally:
        Path(utterance_path).unlink(missing_ok=True)

    status_text = f"{status_text}\n自由说话：自动切分 {candidate['start_ms']} - {candidate['end_ms']} ms"
    return state, new_history, output_audio_path, status_text


def clear_dialogue() -> tuple[list[dict[str, str]], str, str | None, str, None]:
    return [], "", None, "已清空对话。", None


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

        支持文字输入和语音输入。前端通过 RESTful API 调用后端的 ASR、LLM、TTS 三个接口。
        """
    )

    with gr.Row():
        api_base = gr.Textbox(value=API_BASE_URL, label="后端地址", interactive=False)
        session_id = gr.Textbox(value="demo-session", label="Session ID")
        llm_provider = gr.Textbox(value=os.getenv("DEFAULT_LLM_PROVIDER", "mock_llm"), label="LLM Provider")
        llm_api_key = gr.Textbox(value="", label="LLM API KEY", type="password")
        language = gr.Dropdown(choices=["zh-CN", "en-US"], value="zh-CN", label="语言")
        voice = gr.Dropdown(choices=["female_default", "male_default"], value="female_default", label="音色")
        free_speak_enabled = gr.Checkbox(value=False, label="自由说话")

    chatbot = gr.Chatbot(label="对话", type="messages", height=460)

    with gr.Row():
        text_input = gr.Textbox(label="文字输入", placeholder="输入文字，或录制/上传语音后点击发送。", lines=3, scale=2)
        audio_input = gr.Audio(label="语音输入", sources=["microphone", "upload"], type="filepath", scale=1)
        free_audio_input = gr.Audio(
            label="自由说话输入",
            sources=["microphone"],
            type="numpy",
            streaming=True,
            visible=False,
            scale=1,
        )

    with gr.Row():
        send_button = gr.Button("发送", variant="primary")
        clear_button = gr.Button("清空")
        health_button = gr.Button("检查后端")

    audio_output = gr.Audio(label="语音回复", type="filepath", autoplay=True)
    status = gr.Textbox(label="状态", value="等待输入。", lines=5)
    free_speak_state = gr.State({})

    send_button.click(
        respond,
        inputs=[text_input, audio_input, chatbot, session_id, llm_provider, llm_api_key, language, voice],
        outputs=[chatbot, text_input, audio_output, status, audio_input],
    )
    text_input.submit(
        respond,
        inputs=[text_input, audio_input, chatbot, session_id, llm_provider, llm_api_key, language, voice],
        outputs=[chatbot, text_input, audio_output, status, audio_input],
    )
    free_speak_enabled.change(
        reset_free_speak,
        inputs=free_speak_enabled,
        outputs=[free_speak_state, status, audio_input, free_audio_input],
    )
    free_audio_input.stream(
        free_speak_stream,
        inputs=[
            free_audio_input,
            free_speak_enabled,
            free_speak_state,
            chatbot,
            session_id,
            llm_provider,
            llm_api_key,
            language,
            voice,
        ],
        outputs=[free_speak_state, chatbot, audio_output, status],
        stream_every=0.5,
        concurrency_limit=1,
    )
    clear_button.click(clear_dialogue, outputs=[chatbot, text_input, audio_output, status, audio_input])
    health_button.click(check_backend, outputs=status)


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("FRONTEND_HOST", "0.0.0.0"), server_port=int(os.getenv("FRONTEND_PORT", "7860")))
