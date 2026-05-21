from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))


class BackendClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def transcribe(self, audio_path: str, language: str) -> dict[str, Any]:
        with open(audio_path, "rb") as audio_file:
            files = {"audio": (Path(audio_path).name, audio_file, "application/octet-stream")}
            data = {"language": language, "enable_timestamps": "true"}
            response = requests.post(
                f"{self.base_url}/v1/asr/transcriptions",
                files=files,
                data=data,
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        return response.json()

    def chat(self, messages: list[dict[str, str]], session_id: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"messages": messages, "max_tokens": 512}
        if session_id:
            payload["session_id"] = session_id
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
    lines.append(f"LLM：{llm_result.get('provider')} / {llm_result.get('model')}，耗时 {llm_result.get('processing_ms')} ms")
    lines.append(f"TTS：{tts_result.get('provider')} / {tts_result.get('model')}，耗时 {tts_result.get('processing_ms')} ms")
    lines.append(f"音频时长：{tts_result.get('duration_ms')} ms")
    return "\n".join(lines)


def respond(
    text: str,
    audio_path: str | None,
    history: list[dict[str, str]] | None,
    session_id: str,
    language: str,
    voice: str,
) -> tuple[list[dict[str, str]], str, str | None, str, None]:
    history = history or []
    user_text = text.strip()
    asr_result: dict[str, Any] | None = None

    try:
        if audio_path:
            asr_result = client.transcribe(audio_path, language)
            user_text = asr_result.get("text", "").strip() or user_text

        if not user_text:
            gr.Warning("请输入文字或上传/录制一段语音。")
            return history, "", None, "等待输入。", None

        llm_result = client.chat(_to_llm_messages(history, user_text), session_id.strip() or None)
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
        language = gr.Dropdown(choices=["zh-CN", "en-US"], value="zh-CN", label="语言")
        voice = gr.Dropdown(choices=["female_default", "male_default"], value="female_default", label="音色")

    chatbot = gr.Chatbot(label="对话", type="messages", height=460)

    with gr.Row():
        text_input = gr.Textbox(label="文字输入", placeholder="输入文字，或录制/上传语音后点击发送。", lines=3, scale=2)
        audio_input = gr.Audio(label="语音输入", sources=["microphone", "upload"], type="filepath", scale=1)

    with gr.Row():
        send_button = gr.Button("发送", variant="primary")
        clear_button = gr.Button("清空")
        health_button = gr.Button("检查后端")

    audio_output = gr.Audio(label="语音回复", type="filepath", autoplay=True)
    status = gr.Textbox(label="状态", value="等待输入。", lines=5)

    send_button.click(
        respond,
        inputs=[text_input, audio_input, chatbot, session_id, language, voice],
        outputs=[chatbot, text_input, audio_output, status, audio_input],
    )
    text_input.submit(
        respond,
        inputs=[text_input, audio_input, chatbot, session_id, language, voice],
        outputs=[chatbot, text_input, audio_output, status, audio_input],
    )
    clear_button.click(clear_dialogue, outputs=[chatbot, text_input, audio_output, status, audio_input])
    health_button.click(check_backend, outputs=status)


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("FRONTEND_HOST", "0.0.0.0"), server_port=int(os.getenv("FRONTEND_PORT", "7860")))
