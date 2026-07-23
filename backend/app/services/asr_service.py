from datetime import datetime
import os
from pathlib import Path
import tempfile
from time import perf_counter
from uuid import uuid4

from app.core.config import get_settings
from app.core.errors import AppError
from app.models.asr import ASRJobResponse, ASRResult, ASRSegment
from app.models.common import JobCreatedResponse, ProviderInfo
import logging

from app.services.vad_service import vad_service

logger = logging.getLogger(__name__)


class AXASRProvider:
    """ASR provider backed by ax_asr wheel package (ax_asr_api)."""
    name = "ax_asr"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._asr = None

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            type="local",
            models=["ax_asr_sensevoice"],
            languages=["zh", "en", "yue", "ja", "ko", "auto"],
            audio_formats=["wav", "mp3"],
            features=["axengine", "local_model", "ax_asr_api"],
            metadata={
                "source_repo": "https://github.com/AXERA-TECH/ax_asr_api",
                "wheel_version": "0.1.0",
                "model_type": self.settings.ax_asr_model_type,
                "model_path": self.settings.ax_asr_model_path,
            },
        )

    def transcribe(self, audio_content: bytes, filename: str | None, language: str | None) -> tuple[str, dict[str, str | int | bool | None]]:
        if not audio_content:
            raise AppError("asr_no_speech", "Audio payload is empty", status_code=422, stage="asr")

        suffix = Path(filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(prefix="her_axasr_", suffix=suffix, delete=False) as audio_file:
            audio_file.write(audio_content)
            audio_path = audio_file.name

        selected_language = language or self.settings.ax_asr_language

        try:
            text = self.asr.transcribe_file(audio_path, language=selected_language)
        except AppError:
            raise
        except Exception as exc:
            raise AppError(
                "ax_asr_transcription_failed",
                f"AX ASR transcription failed: {exc}",
                status_code=502,
                stage="asr",
                retryable=True,
            ) from exc
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

        return text, {
            "language": selected_language,
            "model_type": self.settings.ax_asr_model_type,
            "model_path": self.settings.ax_asr_model_path,
        }

    # ── Streaming ASR ──────────────────────────────────────────

    def stream_init(self) -> None:
        """Initialize streaming ASR session. Call once per utterance."""
        asr = self.asr
        if hasattr(asr, 'stream_init'):
            asr.stream_init()

    def stream_feed(self, pcm: bytes, sample_rate: int = 16000) -> str | None:
        """Feed a PCM chunk and return partial result if changed."""
        import numpy as np
        asr = self.asr
        if not hasattr(asr, 'stream_feed'):
            return None
        n_samples = len(pcm) // 2
        if n_samples == 0:
            return None
        audio = np.frombuffer(pcm[:n_samples * 2], dtype=np.int16).astype(np.float32) / 32768.0
        asr.stream_feed(audio, sample_rate)
        result = asr.stream_result() if hasattr(asr, 'stream_result') else None
        return result if result else None

    def stream_result(self) -> str:
        """Get current partial streaming result."""
        asr = self.asr
        if hasattr(asr, 'stream_result'):
            return asr.stream_result() or ""
        return ""

    def stream_reset(self) -> None:
        """Reset streaming state for a new utterance."""
        asr = self.asr
        if hasattr(asr, 'stream_reset'):
            asr.stream_reset()

    # ── Internal ───────────────────────────────────────────────

    @property

    @property
    def asr(self):
        if self._asr is None:
            self._asr = self._build_asr()
        return self._asr

    def _build_asr(self):
        try:
            from ax_asr import AX_ASR
        except ImportError as exc:
            raise AppError(
                "ax_asr_not_installed",
                "ax_asr wheel is not installed. Install from: "
                "https://github.com/AXERA-TECH/ax_asr_api/releases",
                status_code=503,
                stage="asr",
                retryable=True,
            ) from exc

        model_path = self.settings.ax_asr_model_path
        if not model_path:
            raise AppError(
                "ax_asr_not_configured",
                "AX_ASR_MODEL_PATH is required for ASR provider ax_asr",
                status_code=503,
                stage="asr",
                retryable=True,
            )

        # Lazy load: check if models exist, trigger download if not
        from pathlib import Path as _Path
        model_dir = _Path(model_path).expanduser().resolve()
        if not model_dir.exists() or not any(model_dir.iterdir()):
            from app.services.model_download_service import get_model_download_manager
            mgr = get_model_download_manager()
            started = mgr.start_download_all(model_type="asr")
            if started:
                raise AppError(
                    "ax_asr_model_not_ready",
                    f"ASR models are being downloaded ({len(started)} model(s) queued). "
                    "Track progress at GET /v1/models/download/status?model_type=asr",
                    status_code=503,
                    stage="asr",
                    retryable=True,
                )
            raise AppError(
                "ax_asr_model_not_found",
                f"Model directory is empty: {model_dir}. "
                "Trigger download at POST /v1/models/download or run download_models.sh",
                status_code=503,
                stage="asr",
                retryable=True,
            )

        try:
            return AX_ASR(self.settings.ax_asr_model_type, str(model_dir))
        except Exception as exc:
            raise AppError(
                "ax_asr_init_failed",
                f"Failed to initialize AX_ASR: {exc}",
                status_code=503,
                stage="asr",
                retryable=True,
            ) from exc


def _apply_noise_reduction(audio_content: bytes, filename: str | None) -> bytes:
    """Apply spectral noise reduction to audio. Requires noisereduce."""
    try:
        import io
        import wave

        import numpy as np
        import noisereduce as nr  # type: ignore[import-untyped]

        buf = io.BytesIO(audio_content)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels)[:, 0]
        reduced = nr.reduce_noise(y=audio, sr=sr)
        pcm = (np.clip(reduced, -1.0, 1.0) * 32767).astype(np.int16)
        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return out.getvalue()
    except ImportError:
        return audio_content
    except Exception:  # noqa: BLE001
        return audio_content


class ASRService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.providers = {}
        self.providers["mock_asr"] = ProviderInfo(
            name="mock_asr",
            type="mock",
            models=["mock-asr"],
            languages=["zh-CN", "en-US"],
            audio_formats=["wav", "pcm", "mp3", "flac", "webm"],
            features=["offline_test", "deterministic"],
        )
        self.ax_asr_provider = AXASRProvider()
        if self._should_register_ax_asr():
            self.providers[self.ax_asr_provider.name] = self.ax_asr_provider.info()
        self.jobs: dict[str, ASRJobResponse] = {}

    def list_providers(self) -> list[ProviderInfo]:
        return list(self.providers.values())


    def _should_register_ax_asr(self) -> bool:
        return (
            self.settings.enable_ax_asr
            or self.settings.default_asr_provider == self.ax_asr_provider.name
            or bool(self.settings.ax_asr_model_path)
        )

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
        enable_vad: bool = False,
    ) -> ASRResult:
        start = perf_counter()
        provider_name = provider or self.settings.default_asr_provider
        provider_info = self.providers.get(provider_name)
        logger.debug("ASR transcribe: provider=%s model=%s language=%s audio_bytes=%d vad=%s",
                     provider_name, model, language, len(audio_content), enable_vad)
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

        vad_segments = []
        speech_duration_ms = None
        vad_processing_ms = None
        transcribe_audio = audio_content
        transcribe_filename = filename

        if self.settings.enable_noise_reduction:
            transcribe_audio = _apply_noise_reduction(transcribe_audio, transcribe_filename)

        if enable_vad:
            vad_result = vad_service.extract_speech(audio_content, filename)
            transcribe_audio = vad_result.audio_content
            transcribe_filename = f"vad_{Path(filename or 'audio.wav').stem}.wav"
            vad_segments = vad_result.segments
            speech_duration_ms = vad_result.speech_duration_ms
            vad_processing_ms = vad_result.processing_ms

        selected_model = model or provider_info.models[0]
        selected_language = language or provider_info.languages[0]


        if provider_name == self.ax_asr_provider.name:
            text, metadata = self.ax_asr_provider.transcribe(transcribe_audio, transcribe_filename, selected_language)
            processing_ms = int((perf_counter() - start) * 1000)
            logger.info("ASR result: provider=%s text_len=%d processing_ms=%d",
                        provider_name, len(text), processing_ms)
            duration_ms = speech_duration_ms or max(1000, min(len(audio_content) // 16, 60000))
            selected_model = model or str(metadata.get("model_type") or provider_info.models[0])
            selected_language = str(metadata.get("language") or selected_language)
            segments = []
            if enable_timestamps:
                segments.append(
                    ASRSegment(
                        index=0,
                        start_ms=0,
                        end_ms=duration_ms,
                        text=text,
                        confidence=None,
                        speaker="spk_0",
                    )
                )
            return ASRResult(
                trace_id=trace_id,
                provider=provider_name,
                model=selected_model,
                language=selected_language,
                text=text,
                confidence=0.0,
                duration_ms=duration_ms,
                processing_ms=processing_ms,
                segments=segments,
                vad_segments=vad_segments,
                speech_duration_ms=speech_duration_ms,
                vad_processing_ms=vad_processing_ms,
            )

        import hashlib
        _MOCK_TEXTS = [
            "今天天气真不错，适合出去走走。",
            "帮我查一下明天上海的天气怎么样。",
            "播放一首轻松的音乐吧。",
            "讲个笑话听听。",
            "你觉得人工智能未来会取代人类工作吗？",
            "附近有什么好吃的推荐吗？",
            "最近睡眠不太好，有什么建议吗？",
            "把客厅的灯调暗一点。",
            "现在几点了？",
            "帮我设置一个明天早上七点的闹钟。",
            "最近有什么好看的电影吗？",
            "我想学做红烧肉，能教我吗？",
            "今天心情不太好，陪我聊聊吧。",
            "世界上最深的海沟在哪里？",
            "帮我翻译一下这段话到英文。",
        ]
        idx = int(hashlib.md5(audio_content[:1024]).hexdigest(), 16) % len(_MOCK_TEXTS)
        text = _MOCK_TEXTS[idx]
        processing_ms = int((perf_counter() - start) * 1000)
        duration_ms = speech_duration_ms or max(1000, min(len(audio_content) // 16, 60000))
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
            vad_segments=vad_segments,
            speech_duration_ms=speech_duration_ms,
            vad_processing_ms=vad_processing_ms,
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
