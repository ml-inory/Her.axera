from base64 import b64decode
import time
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from app.api.deps import bind_trace_id
from app.core.errors import AppError
from app.models.llm import ChatCompletionRequest
from app.models.openai import OpenAIChatCompletionRequest, OpenAISpeechRequest
from app.models.tts import SpeechRequest
from app.services.asr_service import asr_service
from app.services.llm_service import llm_service
from app.services.tts_service import tts_service

router = APIRouter(tags=["openai-compatible"])


AUDIO_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
}

OPENAI_EDGE_VOICE_MAP = {
    "alloy": "zh-CN-XiaoxiaoNeural",
    "echo": "zh-CN-YunxiNeural",
    "fable": "zh-CN-XiaoyiNeural",
    "onyx": "zh-CN-YunjianNeural",
    "nova": "en-US-JennyNeural",
    "shimmer": "en-US-JennyNeural",
}

OPENAI_MOCK_VOICE_MAP = {
    "alloy": "female_default",
    "echo": "male_default",
    "fable": "female_default",
    "onyx": "male_default",
    "nova": "female_default",
    "shimmer": "female_default",
}


def _normalise_stop(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return stop


def _infer_llm_provider(model: str, provider: str | None) -> tuple[str | None, str | None]:
    if provider:
        return provider, None if model == provider else model
    if model in llm_service.providers:
        return model, None
    if model.startswith("ax-llm") or model == "ax_llm":
        return "ax_llm", model
    if model.startswith("deepseek"):
        return "deepseek", model
    if model.startswith("mock"):
        return "mock_llm", model
    return None, model


def _infer_asr_provider(model: str, provider: str | None) -> tuple[str | None, str | None]:
    if provider:
        return provider, None if model == provider else model
    if model in asr_service.providers:
        return model, None
    if model.startswith("ax_asr") or model.startswith("ax-asr"):
        return "ax_asr", model
    if model.startswith("mock") or model in {"whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"}:
        return "mock_asr", model
    return None, model


def _infer_tts_provider(model: str, provider: str | None) -> tuple[str | None, str | None]:
    if provider:
        return provider, None if model == provider else model
    if model in tts_service.providers:
        return model, None
    if model.startswith("ax_tts") or model.startswith("ax-tts"):
        return "ax_tts", model
    if model.startswith("edge"):
        return "edge_tts", model
    if model.startswith("mock"):
        return "mock_tts", model
    if model in {"tts-1", "tts-1-hd", "gpt-4o-mini-tts"}:
        return None, model
    return None, model


def _map_voice(provider: str | None, voice: str) -> str:
    if provider == "mock_tts":
        return OPENAI_MOCK_VOICE_MAP.get(voice, voice)
    return OPENAI_EDGE_VOICE_MAP.get(voice, voice)


def _openai_chat_response(trace_id: str, result) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{trace_id.replace('trc_', '')}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": result.message.role,
                    "content": result.message.content,
                },
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": result.usage.model_dump(),
        "trace_id": trace_id,
        "provider": result.provider,
    }


@router.post("/chat/completions")
async def create_openai_chat_completion(
    request: Annotated[OpenAIChatCompletionRequest, Body()],
    trace_id: Annotated[str, Depends(bind_trace_id)],
) -> dict[str, Any]:
    if request.stream:
        raise AppError(
            "stream_not_supported",
            "OpenAI-compatible stream=true is not implemented yet",
            status_code=400,
            stage="llm",
        )

    provider, model = _infer_llm_provider(request.model, request.provider)
    chat_request = ChatCompletionRequest(
        messages=request.messages,
        session_id=request.session_id,
        user_id=request.user,
        provider=provider,
        api_key=request.api_key,
        model=model,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=_normalise_stop(request.stop),
        response_format=request.response_format,
        tools=request.tools,
        tool_choice=request.tool_choice,
        metadata=request.metadata,
    )
    result = llm_service.chat(trace_id, chat_request)
    return _openai_chat_response(trace_id, result)


@router.post("/audio/transcriptions")
async def create_openai_audio_transcription(
    trace_id: Annotated[str, Depends(bind_trace_id)],
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = "whisper-1",
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
    provider: Annotated[str | None, Form()] = None,
    enable_vad: Annotated[bool, Form()] = False,
):
    del prompt, temperature
    audio_content = await file.read()
    provider_name, selected_model = _infer_asr_provider(model, provider)
    verbose = response_format == "verbose_json"
    result = await asr_service.transcribe(
        trace_id=trace_id,
        audio_content=audio_content,
        filename=file.filename,
        provider=provider_name,
        model=selected_model,
        language=language,
        enable_timestamps=verbose,
        enable_vad=enable_vad,
    )

    if response_format == "text":
        return PlainTextResponse(result.text, headers={"X-Trace-Id": trace_id})
    if response_format in {"srt", "vtt"}:
        raise AppError(
            "response_format_not_supported",
            f"ASR response_format={response_format} is not implemented yet",
            status_code=400,
            stage="asr",
        )
    if verbose:
        return JSONResponse(
            {
                "task": "transcribe",
                "language": result.language,
                "duration": result.duration_ms / 1000,
                "text": result.text,
                "segments": [segment.model_dump() for segment in result.segments],
                "trace_id": trace_id,
                "provider": result.provider,
                "model": result.model,
            }
        )
    return {"text": result.text}


@router.post("/audio/speech")
async def create_openai_audio_speech(
    request: Annotated[OpenAISpeechRequest, Body()],
    trace_id: Annotated[str, Depends(bind_trace_id)],
) -> Response:
    provider, model = _infer_tts_provider(request.model, request.provider)
    speech_request = SpeechRequest(
        text=request.input,
        provider=provider,
        model=model,
        voice=_map_voice(provider, request.voice),
        language=request.language,
        audio_format=request.response_format,
        sample_rate=request.sample_rate,
        speed=request.speed,
        return_audio_base64=True,
    )
    result = await tts_service.synthesize(trace_id, speech_request)
    if not result.audio_base64:
        raise AppError(
            "tts_empty_audio",
            "TTS provider returned no audio content",
            status_code=502,
            stage="tts",
            retryable=True,
        )
    return Response(
        content=b64decode(result.audio_base64),
        media_type=AUDIO_MEDIA_TYPES.get(request.response_format, "application/octet-stream"),
        headers={
            "X-Trace-Id": trace_id,
            "X-Provider": result.provider,
            "X-Model": result.model,
            "X-Audio-Format": result.audio_format,
            "X-Duration-Ms": str(result.duration_ms),
            "X-Processing-Ms": str(result.processing_ms),
        },
    )
