from datetime import datetime
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from types import ModuleType
from uuid import uuid4

from app.core.config import get_settings
from app.core.errors import AppError
from app.models.asr import ASRJobResponse, ASRResult, ASRSegment
from app.models.common import JobCreatedResponse, ProviderInfo
from app.services.vad_service import vad_service


class WenetONNXProvider:
    name = "wenet_onnx"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._runner = None

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            type="local",
            models=["wenet-onnx-offline", "wenet-onnx-online"],
            languages=["zh-CN", "en-US"],
            audio_formats=["wav", "pcm", "mp3", "flac"],
            features=["onnxruntime", "ctc_greedy_search", "ctc_prefix_beam_search", "attention_rescoring"],
            metadata={
                "source_repo": "https://github.com/ml-inory/wenet.axera",
                "online": self.settings.wenet_online,
                "mode": self.settings.wenet_mode,
            },
        )

    def transcribe(self, audio_content: bytes, filename: str | None) -> tuple[str, dict[str, str | int | bool | None]]:
        if not audio_content:
            raise AppError("asr_no_speech", "Audio payload is empty", status_code=422, stage="asr")

        suffix = Path(filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(prefix="her_wenet_", suffix=suffix, delete=False) as audio_file:
            audio_file.write(audio_content)
            audio_path = audio_file.name

        try:
            text = self.runner.transcribe(
                audio_path,
                online=self.settings.wenet_online,
                mode=self.settings.wenet_mode,
                calib_data_path=self.settings.wenet_calib_data_path,
            )
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                "wenet_transcription_failed",
                f"WeNet ONNX transcription failed: {exc}",
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
            "mode": self.settings.wenet_mode,
            "online": self.settings.wenet_online,
            "onnx_dir": self.settings.wenet_onnx_dir,
        }

    @property
    def runner(self):
        if self._runner is None:
            self._runner = self._build_runner()
        return self._runner

    def _build_runner(self):
        repo_path = self._required_path("WENET_REPO_PATH", self.settings.wenet_repo_path)
        onnx_dir = self._required_path("WENET_ONNX_DIR", self.settings.wenet_onnx_dir)
        config_path = self._required_path("WENET_CONFIG_PATH", self.settings.wenet_config_path)
        vocab_path = self._required_path("WENET_VOCAB_PATH", self.settings.wenet_vocab_path)
        module = self._load_ort_common(repo_path)
        providers = [item.strip() for item in self.settings.wenet_ort_providers.split(",") if item.strip()]
        try:
            return module.WenetONNXRunner(
                str(config_path),
                str(vocab_path),
                onnx_dir=str(onnx_dir),
                offline_seq_len=self.settings.wenet_offline_seq_len,
                decoder_len=self.settings.wenet_decoder_len,
                providers=providers or ["CPUExecutionProvider"],
            )
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                "wenet_runner_init_failed",
                f"Failed to initialize WeNet ONNX runner: {exc}",
                status_code=503,
                stage="asr",
                retryable=True,
            ) from exc

    def _required_path(self, env_name: str, value: str | None) -> Path:
        if not value:
            raise AppError(
                "wenet_not_configured",
                f"{env_name} is required for ASR provider {self.name}",
                status_code=503,
                stage="asr",
                retryable=True,
            )
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise AppError(
                "wenet_path_not_found",
                f"{env_name} does not exist: {path}",
                status_code=503,
                stage="asr",
                retryable=True,
            )
        return path

    def _load_ort_common(self, repo_path: Path) -> ModuleType:
        module_path = repo_path / "ort_common.py"
        if not module_path.exists():
            raise AppError(
                "wenet_repo_invalid",
                f"ort_common.py was not found under WENET_REPO_PATH: {repo_path}",
                status_code=503,
                stage="asr",
                retryable=True,
            )
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        spec = importlib.util.spec_from_file_location("wenet_axera_ort_common", module_path)
        if spec is None or spec.loader is None:
            raise AppError(
                "wenet_repo_invalid",
                f"Failed to load ort_common.py from {module_path}",
                status_code=503,
                stage="asr",
                retryable=True,
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


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
        self.wenet_provider = WenetONNXProvider()
        if self._should_register_wenet():
            self.providers[self.wenet_provider.name] = self.wenet_provider.info()
        self.jobs: dict[str, ASRJobResponse] = {}

    def list_providers(self) -> list[ProviderInfo]:
        return list(self.providers.values())

    def _should_register_wenet(self) -> bool:
        return (
            self.settings.enable_wenet_asr
            or self.settings.default_asr_provider == self.wenet_provider.name
            or any(
                (
                    self.settings.wenet_repo_path,
                    self.settings.wenet_onnx_dir,
                    self.settings.wenet_config_path,
                    self.settings.wenet_vocab_path,
                )
            )
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
        if enable_vad:
            vad_result = vad_service.extract_speech(audio_content, filename)
            transcribe_audio = vad_result.audio_content
            transcribe_filename = f"vad_{Path(filename or 'audio.wav').stem}.wav"
            vad_segments = vad_result.segments
            speech_duration_ms = vad_result.speech_duration_ms
            vad_processing_ms = vad_result.processing_ms

        selected_model = model or provider_info.models[0]
        selected_language = language or provider_info.languages[0]
        if provider_name == self.wenet_provider.name:
            text, _metadata = self.wenet_provider.transcribe(transcribe_audio, transcribe_filename)
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

        text = f"这是来自 {transcribe_filename or 'audio'} 的模拟识别结果。"
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
