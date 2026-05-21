from datetime import datetime
from time import perf_counter
from uuid import uuid4

from app.core.config import get_settings
from app.core.errors import AppError
from app.models.asr import ASRJobResponse, ASRResult, ASRSegment
from app.models.common import JobCreatedResponse, ProviderInfo


class ASRService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.providers = {
            "mock_asr": ProviderInfo(
                name="mock_asr",
                type="mock",
                models=["mock-asr-general"],
                languages=["zh-CN", "en-US"],
                audio_formats=["wav", "mp3", "flac", "pcm", "opus"],
                features=["punctuation", "timestamps", "hotwords"],
            )
        }
        self.jobs: dict[str, ASRJobResponse] = {}

    def list_providers(self) -> list[ProviderInfo]:
        return list(self.providers.values())

    async def transcribe(
        self,
        *,
        trace_id: str,
        audio_content: bytes,
        filename: str | None,
        provider: str | None,
        model: str | None,
        language: str | None,
        enable_timestamps: bool,
    ) -> ASRResult:
        start = perf_counter()
        provider_name = provider or self.settings.default_asr_provider
        provider_info = self.providers.get(provider_name)
        if provider_info is None:
            raise AppError(
                "provider_not_found",
                f"ASR provider {provider_name} is not configured",
                status_code=404,
                stage="asr",
            )
        if not audio_content:
            raise AppError(
                "asr_no_speech",
                "Audio payload is empty",
                status_code=422,
                stage="asr",
            )

        selected_model = model or provider_info.models[0]
        selected_language = language or provider_info.languages[0]
        text = f"这是来自 {filename or 'audio'} 的模拟识别结果。"
        processing_ms = int((perf_counter() - start) * 1000)
        duration_ms = max(1000, min(len(audio_content) // 16, 60000))
        segments = []
        if enable_timestamps:
            segments.append(
                ASRSegment(
                    index=0,
                    start_ms=0,
                    end_ms=duration_ms,
                    text=text,
                    confidence=0.99,
                    speaker="spk_0",
                )
            )

        return ASRResult(
            trace_id=trace_id,
            provider=provider_name,
            model=selected_model,
            language=selected_language,
            text=text,
            confidence=0.99,
            duration_ms=duration_ms,
            processing_ms=processing_ms,
            segments=segments,
        )

    def create_job(self, trace_id: str) -> JobCreatedResponse:
        job_id = f"job_asr_{uuid4().hex}"
        response = ASRJobResponse(trace_id=trace_id, job_id=job_id, status="queued")
        self.jobs[job_id] = response
        return JobCreatedResponse(
            trace_id=trace_id,
            job_id=job_id,
            status="queued",
            created_at=datetime.now().astimezone().isoformat(),
        )

    def get_job(self, trace_id: str, job_id: str) -> ASRJobResponse:
        job = self.jobs.get(job_id)
        if job is None:
            raise AppError("job_not_found", f"ASR job {job_id} not found", status_code=404, stage="asr")
        return job.model_copy(update={"trace_id": trace_id})

    def cancel_job(self, trace_id: str, job_id: str) -> ASRJobResponse:
        job = self.get_job(trace_id, job_id)
        cancelled = job.model_copy(update={"status": "cancelled", "trace_id": trace_id})
        self.jobs[job_id] = cancelled
        return cancelled


asr_service = ASRService()
