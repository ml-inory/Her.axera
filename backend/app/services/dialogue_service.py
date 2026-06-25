from collections.abc import AsyncIterator
import asyncio
import logging
import re
from time import perf_counter
from uuid import uuid4

from app.core.config import get_settings
from app.models.llm import ChatCompletionRequest, ChatMessage
from app.models.tts import SpeechRequest
from app.services.asr_service import asr_service
from app.services.llm_service import llm_service
from app.services.speaker_service import speaker_service
from app.services.tts_service import tts_service

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = "你是一个简洁、友好的语音助手。优先用简短自然的中文回答。"


class DialogueService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def stream_audio_pipeline(
        self,
        *,
        trace_id: str,
        audio_content: bytes,
        filename: str | None,
        session_id: str | None,
        user_id: str | None,
        language: str | None,
        asr_provider: str | None,
        asr_model: str | None,
        llm_provider: str | None,
        llm_model: str | None,
        llm_api_key: str | None,
        tts_provider: str | None,
        tts_model: str | None,
        voice: str | None,
        output_audio_format: str,
        sample_rate: int,
        system_prompt: str | None,
        speaker_enabled: bool = False,
        speaker_provider: str | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        start = perf_counter()
        session_id = session_id or f"ses_{uuid4().hex}"
        selected_language = language or "zh-CN"

        # Run ASR and speaker identification in parallel when speaker is enabled.
        asr_coro = asr_service.transcribe(
            trace_id=trace_id,
            audio_content=audio_content,
            filename=filename,
            provider=asr_provider,
            model=asr_model,
            language=selected_language,
            enable_timestamps=True,
            enable_vad=False,
        )

        speaker_result = None
        if speaker_enabled:
            loop = asyncio.get_event_loop()
            speaker_future = loop.run_in_executor(
                None,
                lambda: speaker_service.identify(
                    trace_id=trace_id,
                    audio_content=audio_content,
                    filename=filename,
                    provider=speaker_provider,
                    top_k=1,
                ),
            )
            results = await asyncio.gather(asr_coro, speaker_future, return_exceptions=True)
            asr_result_or_exc, speaker_result_or_exc = results
            if isinstance(asr_result_or_exc, Exception):
                raise asr_result_or_exc
            asr_result = asr_result_or_exc
            if isinstance(speaker_result_or_exc, Exception):
                logger.warning("Speaker identification failed (non-fatal): %s", speaker_result_or_exc)
                speaker_result = None
            else:
                speaker_result = speaker_result_or_exc
        else:
            asr_result = await asr_coro

        yield {
            "type": "asr",
            "trace_id": trace_id,
            "session_id": session_id,
            "provider": asr_result.provider,
            "model": asr_result.model,
            "text": asr_result.text,
            "processing_ms": asr_result.processing_ms,
        }

        if speaker_result is not None:
            yield {
                "type": "speaker",
                "trace_id": trace_id,
                "session_id": session_id,
                "provider": speaker_result.provider,
                "model": speaker_result.model,
                "speaker_id": speaker_result.speaker_id,
                "speaker_label": speaker_result.matches[0].label if speaker_result.matches else None,
                "confidence": speaker_result.confidence,
                "processing_ms": speaker_result.processing_ms,
            }

        # Enrich system prompt with speaker identity for personalized responses.
        effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        if speaker_result is not None and speaker_result.confidence >= 0.5:
            speaker_name = (speaker_result.matches[0].label if speaker_result.matches else None) or speaker_result.speaker_id
            effective_system_prompt += f"\n\n当前说话人是{speaker_name}。请根据说话人身份进行个性化回答。"

        history = llm_service.sessions.get(session_id, [])
        user_message = ChatMessage(
            role="user",
            content=asr_result.text,
            metadata={"source": "asr", "asr_provider": asr_result.provider, "asr_model": asr_result.model},
        )
        chat_response = llm_service.chat(
            trace_id,
            ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content=effective_system_prompt),
                    *history[-10:],
                    user_message,
                ],
                session_id=None,
                user_id=user_id,
                provider=llm_provider,
                api_key=llm_api_key,
                model=llm_model,
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
            ),
        )
        llm_service.sessions.setdefault(session_id, []).extend([user_message, chat_response.message])
        yield {
            "type": "llm",
            "trace_id": trace_id,
            "session_id": session_id,
            "provider": chat_response.provider,
            "model": chat_response.model,
            "text": chat_response.message.content,
            "processing_ms": chat_response.processing_ms,
        }

        sentences = split_sentences(chat_response.message.content)
        for index, sentence in enumerate(sentences):
            tts_response = await tts_service.synthesize(
                trace_id,
                SpeechRequest(
                    text=sentence,
                    provider=tts_provider,
                    model=tts_model,
                    voice=voice,
                    language=selected_language,
                    audio_format=output_audio_format,
                    sample_rate=sample_rate,
                    return_audio_base64=True,
                ),
            )
            yield {
                "type": "tts_sentence",
                "trace_id": trace_id,
                "session_id": session_id,
                "index": index,
                "text": sentence,
                "provider": tts_response.provider,
                "model": tts_response.model,
                "voice": tts_response.voice,
                "audio_format": tts_response.audio_format,
                "sample_rate": tts_response.sample_rate,
                "duration_ms": tts_response.duration_ms,
                "processing_ms": tts_response.processing_ms,
                "audio_base64": tts_response.audio_base64 or "",
            }

        yield {
            "type": "done",
            "trace_id": trace_id,
            "session_id": session_id,
            "sentence_count": len(sentences),
            "total_processing_ms": int((perf_counter() - start) * 1000),
        }

    async def stream_text_pipeline(
        self,
        *,
        trace_id: str,
        text: str,
        session_id: str | None,
        user_id: str | None,
        language: str | None,
        llm_provider: str | None,
        llm_model: str | None,
        llm_api_key: str | None,
        tts_provider: str | None,
        tts_model: str | None,
        voice: str | None,
        output_audio_format: str,
        sample_rate: int,
        system_prompt: str | None,
    ) -> AsyncIterator[dict[str, object]]:
        start = perf_counter()
        session_id = session_id or f"ses_{uuid4().hex}"
        selected_language = language or "zh-CN"
        user_text = text.strip()
        if not user_text:
            yield {
                "type": "error",
                "trace_id": trace_id,
                "error": {
                    "code": "invalid_request",
                    "message": "text must not be empty",
                    "stage": "dialogue",
                    "retryable": False,
                },
            }
            return

        yield {
            "type": "user_text",
            "trace_id": trace_id,
            "session_id": session_id,
            "text": user_text,
        }

        history = llm_service.sessions.get(session_id, [])
        user_message = ChatMessage(role="user", content=user_text, metadata={"source": "text"})
        chat_response = llm_service.chat(
            trace_id,
            ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content=system_prompt or DEFAULT_SYSTEM_PROMPT),
                    *history[-10:],
                    user_message,
                ],
                session_id=None,
                user_id=user_id,
                provider=llm_provider,
                api_key=llm_api_key,
                model=llm_model,
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
            ),
        )
        llm_service.sessions.setdefault(session_id, []).extend([user_message, chat_response.message])
        yield {
            "type": "llm",
            "trace_id": trace_id,
            "session_id": session_id,
            "provider": chat_response.provider,
            "model": chat_response.model,
            "text": chat_response.message.content,
            "processing_ms": chat_response.processing_ms,
        }

        sentences = split_sentences(chat_response.message.content)
        for index, sentence in enumerate(sentences):
            tts_response = await tts_service.synthesize(
                trace_id,
                SpeechRequest(
                    text=sentence,
                    provider=tts_provider,
                    model=tts_model,
                    voice=voice,
                    language=selected_language,
                    audio_format=output_audio_format,
                    sample_rate=sample_rate,
                    return_audio_base64=True,
                ),
            )
            yield {
                "type": "tts_sentence",
                "trace_id": trace_id,
                "session_id": session_id,
                "index": index,
                "text": sentence,
                "provider": tts_response.provider,
                "model": tts_response.model,
                "voice": tts_response.voice,
                "audio_format": tts_response.audio_format,
                "sample_rate": tts_response.sample_rate,
                "duration_ms": tts_response.duration_ms,
                "processing_ms": tts_response.processing_ms,
                "audio_base64": tts_response.audio_base64 or "",
            }

        yield {
            "type": "done",
            "trace_id": trace_id,
            "session_id": session_id,
            "sentence_count": len(sentences),
            "total_processing_ms": int((perf_counter() - start) * 1000),
        }


def split_sentences(text: str, max_chars: int = 80) -> list[str]:
    parts = [part.strip() for part in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text) if part.strip()]
    if not parts:
        parts = [text.strip()] if text.strip() else []

    sentences: list[str] = []
    for part in parts:
        while len(part) > max_chars:
            sentences.append(part[:max_chars].strip())
            part = part[max_chars:].strip()
        if part:
            sentences.append(part)
    return sentences


dialogue_service = DialogueService()
