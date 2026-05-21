from base64 import b64encode
from io import BytesIO
import math
import wave
from datetime import datetime
from time import perf_counter
from uuid import uuid4

from app.core.config import get_settings
from app.core.errors import AppError
from app.models.common import JobCreatedResponse, ProviderInfo
from app.models.tts import (
    SegmentedSpeechRequest,
    SegmentedSpeechResponse,
    SpeechRequest,
    SpeechResponse,
    SpeechSegmentResponse,
    TTSJobResponse,
    VoiceInfo,
)


def _build_mock_wav(text: str, sample_rate: int, duration_ms: int) -> bytes:
    frame_count = max(1, int(sample_rate * duration_ms / 1000))
    amplitude = 8000
    frequency = 440 + (len(text) % 8) * 40
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            envelope = min(1.0, index / max(1, sample_rate // 20), (frame_count - index) / max(1, sample_rate // 20))
            sample = int(amplitude * envelope * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))
    return buffer.getvalue()


class TTSService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.providers = {
            "mock_tts": ProviderInfo(
                name="mock_tts",
                type="mock",
                models=["mock-tts-general"],
                languages=["zh-CN", "en-US"],
                audio_formats=["wav", "mp3", "pcm", "opus"],
                features=["speed", "pitch", "volume", "emotion", "segments"],
            )
        }
        self.voices = [
            VoiceInfo(
                name="female_default",
                display_name="默认女声",
                language="zh-CN",
                gender="female",
                styles=["neutral", "happy"],
                sample_rates=[16000, 24000],
            ),
            VoiceInfo(
                name="male_default",
                display_name="默认男声",
                language="zh-CN",
                gender="male",
                styles=["neutral"],
                sample_rates=[16000, 24000],
            ),
        ]
        self.jobs: dict[str, TTSJobResponse] = {}

    def list_providers(self) -> list[ProviderInfo]:
        return list(self.providers.values())

    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        if language is None:
            return self.voices
        return [voice for voice in self.voices if voice.language == language]

    def synthesize(self, trace_id: str, request: SpeechRequest) -> SpeechResponse:
        start = perf_counter()
        provider_name = request.provider or self.settings.default_tts_provider
        provider_info = self.providers.get(provider_name)
        if provider_info is None:
            raise AppError(
                "provider_not_found",
                f"TTS provider {provider_name} is not configured",
                status_code=404,
                stage="tts",
            )
        if not request.text.strip():
            raise AppError("invalid_request", "text must not be empty", status_code=400, stage="tts")
        if len(request.text) > self.settings.max_tts_text_length:
            raise AppError("text_too_long", "text exceeds configured length limit", status_code=413, stage="tts")

        selected_model = request.model or provider_info.models[0]
        selected_voice = request.voice or "female_default"
        duration_ms = max(300, len(request.text) * 120)
        if request.audio_format == "wav":
            audio_content = _build_mock_wav(request.text, request.sample_rate, duration_ms)
        else:
            audio_content = f"FAKE_AUDIO:{request.text}".encode("utf-8")
        audio_url = None if request.return_audio_base64 else f"mock://audio/{uuid4().hex}.{request.audio_format}"
        audio_base64 = b64encode(audio_content).decode("ascii") if request.return_audio_base64 else None
        return SpeechResponse(
            trace_id=trace_id,
            provider=provider_name,
            model=selected_model,
            voice=selected_voice,
            language=request.language,
            audio_url=audio_url,
            audio_base64=audio_base64,
            audio_format=request.audio_format,
            sample_rate=request.sample_rate,
            duration_ms=duration_ms,
            processing_ms=int((perf_counter() - start) * 1000),
            cache_hit=False,
        )

    def synthesize_segments(self, trace_id: str, request: SegmentedSpeechRequest) -> SegmentedSpeechResponse:
        start = perf_counter()
        provider_name = request.provider or self.settings.default_tts_provider
        if provider_name not in self.providers:
            raise AppError("provider_not_found", f"TTS provider {provider_name} is not configured", status_code=404, stage="tts")
        voice = request.voice or "female_default"
        segments = [
            SpeechSegmentResponse(
                index=segment.index,
                text=segment.text,
                audio_url=f"mock://audio/segments/{uuid4().hex}.{request.audio_format}",
                duration_ms=max(200, len(segment.text) * 120),
                processing_ms=max(20, len(segment.text) * 8),
            )
            for segment in request.segments
        ]
        return SegmentedSpeechResponse(
            trace_id=trace_id,
            provider=provider_name,
            voice=voice,
            segments=segments,
            total_duration_ms=sum(segment.duration_ms for segment in segments),
            processing_ms=int((perf_counter() - start) * 1000),
        )

    def create_job(self, trace_id: str) -> JobCreatedResponse:
        job_id = f"job_tts_{uuid4().hex}"
        response = TTSJobResponse(trace_id=trace_id, job_id=job_id, status="queued")
        self.jobs[job_id] = response
        return JobCreatedResponse(
            trace_id=trace_id,
            job_id=job_id,
            status="queued",
            created_at=datetime.now().astimezone().isoformat(),
        )

    def get_job(self, trace_id: str, job_id: str) -> TTSJobResponse:
        job = self.jobs.get(job_id)
        if job is None:
            raise AppError("job_not_found", f"TTS job {job_id} not found", status_code=404, stage="tts")
        return job.model_copy(update={"trace_id": trace_id})

    def cancel_job(self, trace_id: str, job_id: str) -> TTSJobResponse:
        job = self.get_job(trace_id, job_id)
        cancelled = job.model_copy(update={"status": "cancelled", "trace_id": trace_id})
        self.jobs[job_id] = cancelled
        return cancelled


tts_service = TTSService()
