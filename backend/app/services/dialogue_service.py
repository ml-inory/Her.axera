from collections.abc import AsyncIterator
import asyncio
import logging
import re
from time import perf_counter
from uuid import uuid4

from base64 import b64decode, b64encode

from app.core.audio_codec import opus_available, pcm_to_opus
from app.core.config import get_settings
from app.models.llm import ChatCompletionRequest, ChatMessage
from app.models.tts import SpeechRequest
from app.services.asr_service import asr_service
from app.services.llm_service import llm_service
from app.services.speaker_service import speaker_service
from app.services.tts_service import tts_service

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = "你是一个简洁、友好的语音助手。优先用简短自然的中文回答。"

_SENTENCE_END_RE = re.compile(r"[。！？!?；;\n]")
_EMOTION_RE = re.compile(r"^\[emotion:(\w+)\]\s*")
EMOTION_PROMPT_SUFFIX = "\n\n在每次回复开头用 [emotion:XXX] 标签表示你的情绪状态。可选值：happy, sad, surprised, angry, neutral, thinking。标签后直接跟回复正文。"

STREAMING_CHUNK_MIN = 4
STREAMING_CHUNK_MAX = 60


def _parse_emotion(text: str) -> tuple[str | None, str]:
    """Extract [emotion:XXX] tag from text start. Returns (emotion, cleaned_text)."""
    match = _EMOTION_RE.match(text)
    if match:
        return match.group(1), text[match.end():]
    return None, text


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split long text into chunks at max_chars boundaries."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    parts = []
    t = text.strip()
    while len(t) > max_chars:
        parts.append(t[:max_chars].strip())
        t = t[max_chars:].strip()
    if t:
        parts.append(t)
    return parts


def _extract_streaming_chunks(buffer: str) -> tuple[list[str], str]:
    """Extract speakable chunks for low-latency TTS.
    Sentence endings immediately flush, clause markers flush when >4 chars,
    long runs split at 60 chars."""
    chunks: list[str] = []
    last_end = 0
    for match in _SENTENCE_END_RE.finditer(buffer):
        end = match.end()
        part = buffer[last_end:end].strip()
        if part:
            chunks.extend(_split_long(part, STREAMING_CHUNK_MAX))
        last_end = end
    remaining = buffer[last_end:]
    clause_chunks = []
    clause_last = 0
    for match in _CLAUSE_END_RE.finditer(remaining):
        end = match.end()
        part = remaining[clause_last:end].strip()
        if part and len(remaining[:end]) >= STREAMING_CHUNK_MIN:
            clause_chunks.extend(_split_long(part, STREAMING_CHUNK_MAX))
            clause_last = end
    chunks.extend(clause_chunks)
    final_remaining = remaining[clause_last:]
    if len(final_remaining) >= STREAMING_CHUNK_MAX:
        splits = _split_long(final_remaining, STREAMING_CHUNK_MAX)
        if len(splits) > 1:
            chunks.extend(splits[:-1])
            final_remaining = splits[-1]
    return chunks, final_remaining


def _extract_complete_sentences(buffer: str, max_chars: int = 80) -> tuple[list[str], str]:
    """Extract complete sentences from buffer, return (sentences, remaining)."""
    sentences: list[str] = []
    last_end = 0
    for match in _SENTENCE_END_RE.finditer(buffer):
        end = match.end()
        part = buffer[last_end:end].strip()
        if part:
            while len(part) > max_chars:
                sentences.append(part[:max_chars].strip())
                part = part[max_chars:].strip()
            if part:
                sentences.append(part)
        last_end = end
    remaining = buffer[last_end:]
    return sentences, remaining


class DialogueService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def _streaming_llm_tts(
        self,
        *,
        trace_id: str,
        session_id: str,
        user_message: ChatMessage,
        effective_system_prompt: str,
        user_id: str | None,
        llm_provider: str | None,
        llm_model: str | None,
        llm_api_key: str | None,
        tts_provider: str | None,
        tts_model: str | None,
        voice: str | None,
        selected_language: str,
        output_audio_format: str,
        sample_rate: int,
        output_audio_codec: str = "pcm",
    ) -> AsyncIterator[dict[str, object]]:
        """Stream LLM tokens, detect sentence boundaries, synthesize TTS per sentence."""
        use_opus = output_audio_codec == "opus" and opus_available()
        history = llm_service.sessions.get(session_id, [])
        llm_request = ChatCompletionRequest(
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
        )

        llm_start = perf_counter()
        full_content = ""
        buffer = ""
        sentence_index = 0
        first_token_yielded = False

        async for token in llm_service.chat_stream(trace_id, llm_request):
            full_content += token
            buffer += token

            if not first_token_yielded:
                first_token_yielded = True
                yield {
                    "type": "llm_started",
                    "trace_id": trace_id,
                    "session_id": session_id,
                }

            # Stream to TTS: flush clauses and short chunks for low latency
            sentences, buffer = _extract_streaming_chunks(buffer)
            for sentence in sentences:
                yield {
                    "type": "llm_delta",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "text": sentence,
                    "index": sentence_index,
                }
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
                audio_b64 = tts_response.audio_base64 or ""
                audio_fmt = tts_response.audio_format
                if use_opus and audio_b64:
                    try:
                        pcm_data = b64decode(audio_b64)
                        opus_data = pcm_to_opus(pcm_data, sample_rate=tts_response.sample_rate or sample_rate)
                        audio_b64 = b64encode(opus_data).decode("ascii")
                        audio_fmt = "opus"
                    except Exception:  # noqa: BLE001
                        pass  # fallback to original format
                yield {
                    "type": "tts_sentence",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "index": sentence_index,
                    "text": sentence,
                    "provider": tts_response.provider,
                    "model": tts_response.model,
                    "voice": tts_response.voice,
                    "audio_format": audio_fmt,
                    "sample_rate": tts_response.sample_rate,
                    "duration_ms": tts_response.duration_ms,
                    "processing_ms": tts_response.processing_ms,
                    "audio_base64": audio_b64,
                }
                sentence_index += 1

        # Process remaining buffer after LLM finishes.
        remaining = buffer.strip()
        if remaining:
            sentences_final = split_sentences(remaining)
            for sentence in sentences_final:
                yield {
                    "type": "llm_delta",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "text": sentence,
                    "index": sentence_index,
                }
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
                audio_b64 = tts_response.audio_base64 or ""
                audio_fmt = tts_response.audio_format
                if use_opus and audio_b64:
                    try:
                        pcm_data = b64decode(audio_b64)
                        opus_data = pcm_to_opus(pcm_data, sample_rate=tts_response.sample_rate or sample_rate)
                        audio_b64 = b64encode(opus_data).decode("ascii")
                        audio_fmt = "opus"
                    except Exception:  # noqa: BLE001
                        pass
                yield {
                    "type": "tts_sentence",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "index": sentence_index,
                    "text": sentence,
                    "provider": tts_response.provider,
                    "model": tts_response.model,
                    "voice": tts_response.voice,
                    "audio_format": audio_fmt,
                    "sample_rate": tts_response.sample_rate,
                    "duration_ms": tts_response.duration_ms,
                    "processing_ms": tts_response.processing_ms,
                    "audio_base64": audio_b64,
                }
                sentence_index += 1

        llm_processing_ms = int((perf_counter() - llm_start) * 1000)

        # Parse emotion tag if present.
        emotion, cleaned_content = _parse_emotion(full_content)

        # Store session history.
        assistant_message = ChatMessage(role="assistant", content=cleaned_content or full_content)
        llm_service.sessions.setdefault(session_id, []).extend([user_message, assistant_message])
        llm_service._save_sessions()

        llm_event: dict[str, object] = {
            "type": "llm",
            "trace_id": trace_id,
            "session_id": session_id,
            "provider": llm_provider or self.settings.default_llm_provider,
            "model": llm_model or "",
            "text": cleaned_content or full_content,
            "processing_ms": llm_processing_ms,
        }
        if emotion:
            llm_event["emotion"] = emotion
        yield llm_event


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
        output_audio_codec: str = "pcm",
    ) -> AsyncIterator[dict[str, object]]:
        start = perf_counter()
        session_id = session_id or f"ses_{uuid4().hex}"
        llm_service._ensure_meta(session_id)
        llm_service._touch_session(session_id)
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

        # Enrich system prompt with speaker identity and emotion detection.
        effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        if self.settings.enable_emotion_detection:
            effective_system_prompt += EMOTION_PROMPT_SUFFIX
        if speaker_result is not None and speaker_result.confidence >= 0.5:
            speaker_name = (speaker_result.matches[0].label if speaker_result.matches else None) or speaker_result.speaker_id
            effective_system_prompt += f"\n\n当前说话人是{speaker_name}。请根据说话人身份进行个性化回答。"

        user_message = ChatMessage(
            role="user",
            content=asr_result.text,
            metadata={"source": "asr", "asr_provider": asr_result.provider, "asr_model": asr_result.model},
        )

        sentence_count = 0
        async for event in self._streaming_llm_tts(
            trace_id=trace_id,
            session_id=session_id,
            user_message=user_message,
            effective_system_prompt=effective_system_prompt,
            user_id=user_id,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            tts_provider=tts_provider,
            tts_model=tts_model,
            voice=voice,
            selected_language=selected_language,
            output_audio_format=output_audio_format,
            sample_rate=sample_rate,
            output_audio_codec=output_audio_codec,
        ):
            if event.get("type") == "tts_sentence":
                sentence_count += 1
            yield event

        yield {
            "type": "done",
            "trace_id": trace_id,
            "session_id": session_id,
            "sentence_count": sentence_count,
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
        output_audio_codec: str = "pcm",
        image_base64: str | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        start = perf_counter()
        session_id = session_id or f"ses_{uuid4().hex}"
        llm_service._ensure_meta(session_id)
        llm_service._touch_session(session_id)
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

        if image_base64 and self.settings.enable_vision:
            content: str | list = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ]
        else:
            content = user_text
        user_message = ChatMessage(role="user", content=content, metadata={"source": "text"})

        sentence_count = 0
        async for event in self._streaming_llm_tts(
            trace_id=trace_id,
            session_id=session_id,
            user_message=user_message,
            effective_system_prompt=(system_prompt or DEFAULT_SYSTEM_PROMPT) + (EMOTION_PROMPT_SUFFIX if self.settings.enable_emotion_detection else ""),
            user_id=user_id,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            tts_provider=tts_provider,
            tts_model=tts_model,
            voice=voice,
            selected_language=selected_language,
            output_audio_format=output_audio_format,
            sample_rate=sample_rate,
            output_audio_codec=output_audio_codec,
        ):
            if event.get("type") == "tts_sentence":
                sentence_count += 1
            yield event

        yield {
            "type": "done",
            "trace_id": trace_id,
            "session_id": session_id,
            "sentence_count": sentence_count,
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
