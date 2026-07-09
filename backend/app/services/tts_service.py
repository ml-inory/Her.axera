from base64 import b64encode
from io import BytesIO
import math
from pathlib import Path
import shlex
import subprocess
import tempfile
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



def _percentage(value: float, *, neutral: float = 1.0, minimum: int = -50, maximum: int = 100) -> str:
    percent = int(round((value - neutral) * 100))
    percent = min(max(percent, minimum), maximum)
    return f"{percent:+d}%"


def _pitch(value: float) -> str:
    hz = int(round((value - 1.0) * 100))
    hz = min(max(hz, -100), 100)
    return f"{hz:+d}Hz"


def _mock_wav(text: str, sample_rate: int) -> tuple[bytes, int]:
    duration_ms = max(350, min(2200, len(text) * 90))
    frame_count = int(sample_rate * duration_ms / 1000)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            envelope = min(1.0, index / max(1, sample_rate * 0.03), (frame_count - index) / max(1, sample_rate * 0.05))
            value = int(math.sin(2 * math.pi * 440 * index / sample_rate) * 8000 * envelope)
            frames.extend(value.to_bytes(2, "little", signed=True))
        wav_file.writeframes(bytes(frames))
    return buffer.getvalue(), duration_ms


class TTSService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.providers = {
            "mock_tts": ProviderInfo(
                name="mock_tts",
                type="mock",
                models=["mock-tts"],
                languages=["zh-CN", "en-US"],
                audio_formats=["wav"],
                features=["deterministic", "base64_audio"],
            ),
            "edge_tts": ProviderInfo(
                name="edge_tts",
                type="remote",
                models=["edge-tts"],
                languages=["zh-CN", "en-US"],
                audio_formats=["mp3"],
                features=["speed", "pitch", "volume"],
                metadata={"requires_api_key": False},
            ),
            "kokoro": ProviderInfo(
                name="kokoro",
                type="local",
                models=["kokoro.axera"],
                languages=["zh-CN", "en-US", "ja-JP"],
                audio_formats=["wav", "pcm"],
                features=["axengine", "local_model", "voice", "speed"],
                metadata={
                    "source_repo": "https://huggingface.co/AXERA-TECH/kokoro.axera",
                    "repo_path": self.settings.kokoro_repo_path,
                    "model_dir": self.settings.kokoro_model_dir,
                    "command_env": "KOKORO_COMMAND",
                    "enabled": self._should_enable_kokoro(),
                },
            ),
            "zipvoice": ProviderInfo(
                name="zipvoice",
                type="local",
                models=["zipvoice.axera"],
                languages=["zh-CN", "en-US"],
                audio_formats=["wav", "pcm"],
                features=["axengine", "local_model", "voice_clone", "speed"],
                metadata={
                    "source_repo": "https://huggingface.co/AXERA-TECH/ZipVoice.AXERA",
                    "repo_path": self.settings.zipvoice_repo_path,
                    "model_dir": self.settings.zipvoice_model_dir,
                    "command_env": "ZIPVOICE_COMMAND",
                    "enabled": self._should_enable_zipvoice(),
                },
            ),
            "ax_tts": ProviderInfo(
                name="ax_tts",
                type="local",
                models=["ax_tts_kokoro"],
                languages=["zh-CN", "en-US", "ja-JP"],
                audio_formats=["wav", "pcm"],
                features=["axengine", "local_model", "voice", "speed", "ax_tts_api"],
                metadata={
                    "source_repo": "https://github.com/AXERA-TECH/ax_tts_api",
                    "wheel_version": "0.1.2",
                    "model_path": self.settings.ax_tts_model_path,
                    "tts_type": self.settings.ax_tts_type,
                    "enabled": self._should_enable_ax_tts(),
                },
            ),
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
            VoiceInfo(
                name="zh-CN-XiaoxiaoNeural",
                display_name="Edge 晓晓",
                language="zh-CN",
                gender="female",
                styles=["neutral"],
                sample_rates=[24000, 48000],
            ),
            VoiceInfo(
                name="zh-CN-XiaoyiNeural",
                display_name="Edge 晓伊",
                language="zh-CN",
                gender="female",
                styles=["neutral"],
                sample_rates=[24000, 48000],
            ),
            VoiceInfo(
                name="zh-CN-YunxiNeural",
                display_name="Edge 云希",
                language="zh-CN",
                gender="male",
                styles=["neutral"],
                sample_rates=[24000, 48000],
            ),
            VoiceInfo(
                name="zh-CN-YunjianNeural",
                display_name="Edge 云健",
                language="zh-CN",
                gender="male",
                styles=["neutral"],
                sample_rates=[24000, 48000],
            ),
            VoiceInfo(
                name="en-US-JennyNeural",
                display_name="Edge Jenny",
                language="en-US",
                gender="female",
                styles=["neutral"],
                sample_rates=[24000, 48000],
            ),
            VoiceInfo(
                name="en-US-GuyNeural",
                display_name="Edge Guy",
                language="en-US",
                gender="male",
                styles=["neutral"],
                sample_rates=[24000, 48000],
            ),
        ]
        self.jobs: dict[str, TTSJobResponse] = {}

    def list_providers(self) -> list[ProviderInfo]:
        return list(self.providers.values())

    def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        if language is None:
            return self.voices
        return [voice for voice in self.voices if voice.language == language]

    async def synthesize(self, trace_id: str, request: SpeechRequest) -> SpeechResponse:
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
        if provider_name == "mock_tts":
            return self._synthesize_mock_tts(trace_id, request, selected_model, selected_voice, start)
        if provider_name == "edge_tts":
            return await self._synthesize_edge_tts(trace_id, request, selected_model, selected_voice, start)
        if provider_name in {"kokoro", "zipvoice"}:
            return self._synthesize_axera_tts(trace_id, request, provider_name, selected_model, selected_voice, start)
        if provider_name == "ax_tts":
            return await self._synthesize_ax_tts(trace_id, request, selected_model, selected_voice, start)

        raise AppError("provider_not_found", f"TTS provider {provider_name} is not configured", status_code=404, stage="tts")

    def _synthesize_mock_tts(
        self,
        trace_id: str,
        request: SpeechRequest,
        selected_model: str,
        selected_voice: str,
        start: float,
    ) -> SpeechResponse:
        audio_content, duration_ms = _mock_wav(request.text, request.sample_rate)
        return SpeechResponse(
            trace_id=trace_id,
            provider="mock_tts",
            model=selected_model,
            voice=selected_voice,
            language=request.language,
            audio_url=None,
            audio_base64=b64encode(audio_content).decode("ascii") if request.return_audio_base64 else None,
            audio_format="wav",
            sample_rate=request.sample_rate,
            duration_ms=duration_ms,
            processing_ms=int((perf_counter() - start) * 1000),
            cache_hit=False,
        )

    def _should_enable_kokoro(self) -> bool:
        return bool(self.settings.enable_kokoro_tts or self.settings.kokoro_repo_path or self.settings.kokoro_command)

    def _should_enable_zipvoice(self) -> bool:
        return bool(self.settings.enable_zipvoice_tts or self.settings.zipvoice_repo_path or self.settings.zipvoice_command)

    def _should_enable_ax_tts(self) -> bool:
        return bool(self.settings.enable_ax_tts)

    async def _synthesize_edge_tts(
        self,
        trace_id: str,
        request: SpeechRequest,
        selected_model: str,
        selected_voice: str,
        start: float,
    ) -> SpeechResponse:
        try:
            import edge_tts
        except ImportError as exc:
            raise AppError(
                "provider_unavailable",
                "edge-tts is not installed; install backend requirements",
                status_code=500,
                stage="tts",
            ) from exc

        voice_aliases = {
            "female_default": "zh-CN-XiaoxiaoNeural",
            "male_default": "zh-CN-YunxiNeural",
        }
        voice = voice_aliases.get(selected_voice, selected_voice) or self.settings.edge_tts_voice
        output_path = Path(tempfile.gettempdir()) / f"her_edge_tts_{uuid4().hex}.mp3"

        try:
            communicate = edge_tts.Communicate(
                request.text,
                voice,
                rate=_percentage(request.speed),
                volume=_percentage(request.volume),
                pitch=_pitch(request.pitch),
            )
            await communicate.save(str(output_path))
            audio_content = output_path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                "provider_unavailable",
                f"Edge TTS synthesis failed: {exc}",
                status_code=502,
                stage="tts",
                retryable=True,
            ) from exc
        finally:
            output_path.unlink(missing_ok=True)

        audio_base64 = b64encode(audio_content).decode("ascii") if request.return_audio_base64 else None
        return SpeechResponse(
            trace_id=trace_id,
            provider="edge_tts",
            model=selected_model,
            voice=voice,
            language=request.language,
            audio_url=None,
            audio_base64=audio_base64,
            audio_format="mp3",
            sample_rate=request.sample_rate,
            duration_ms=max(300, len(request.text) * 120),
            processing_ms=int((perf_counter() - start) * 1000),
            cache_hit=False,
        )

    def _synthesize_axera_tts(
        self,
        trace_id: str,
        request: SpeechRequest,
        provider_name: str,
        selected_model: str,
        selected_voice: str,
        start: float,
    ) -> SpeechResponse:
        config = self._axera_tts_config(provider_name)
        output_path = Path(tempfile.gettempdir()) / f"her_{provider_name}_{uuid4().hex}.{request.audio_format}"
        command = self._build_axera_tts_command(
            config=config,
            request=request,
            provider_name=provider_name,
            selected_model=selected_model,
            selected_voice=selected_voice,
            output_path=output_path,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=config["repo_path"] or None,
                capture_output=True,
                text=True,
                timeout=int(config["timeout_sec"]),
                check=False,
            )
            if completed.returncode != 0:
                output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
                raise AppError(
                    f"{provider_name}_provider_error",
                    f"{provider_name} exited with code {completed.returncode}: {output[-1200:]}",
                    status_code=502,
                    stage="tts",
                    retryable=True,
                )
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise AppError(
                    f"{provider_name}_empty_audio",
                    f"{provider_name} completed but did not create output audio: {output_path}",
                    status_code=502,
                    stage="tts",
                    retryable=True,
                )
            audio_content = output_path.read_bytes()
        except AppError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise AppError(
                f"{provider_name}_timeout",
                f"{provider_name} synthesis timed out after {config['timeout_sec']}s",
                status_code=504,
                stage="tts",
                retryable=True,
            ) from exc
        except OSError as exc:
            raise AppError(
                f"{provider_name}_invocation_failed",
                f"Failed to execute {provider_name} command: {exc}",
                status_code=503,
                stage="tts",
                retryable=True,
            ) from exc
        finally:
            output_path.unlink(missing_ok=True)

        duration_ms = max(300, len(request.text) * 120)
        return SpeechResponse(
            trace_id=trace_id,
            provider=provider_name,
            model=selected_model,
            voice=selected_voice,
            language=request.language,
            audio_url=None,
            audio_base64=b64encode(audio_content).decode("ascii") if request.return_audio_base64 else None,
            audio_format=request.audio_format,
            sample_rate=request.sample_rate,
            duration_ms=duration_ms,
            processing_ms=int((perf_counter() - start) * 1000),
            cache_hit=False,
        )

    def _axera_tts_config(self, provider_name: str) -> dict[str, str | int | None]:
        if provider_name == "kokoro":
            return {
                "python": self.settings.kokoro_python,
                "repo_path": self.settings.kokoro_repo_path,
                "model_dir": self.settings.kokoro_model_dir,
                "command": self.settings.kokoro_command,
                "timeout_sec": self.settings.kokoro_timeout_sec,
            }
        return {
            "python": self.settings.zipvoice_python,
            "repo_path": self.settings.zipvoice_repo_path,
            "model_dir": self.settings.zipvoice_model_dir,
            "command": self.settings.zipvoice_command,
            "timeout_sec": self.settings.zipvoice_timeout_sec,
        }

    def _build_axera_tts_command(
        self,
        *,
        config: dict[str, str | int | None],
        request: SpeechRequest,
        provider_name: str,
        selected_model: str,
        selected_voice: str,
        output_path: Path,
    ) -> list[str]:
        repo_path = str(config["repo_path"] or "")
        values = {
            "python": str(config["python"] or "python3"),
            "repo_path": repo_path,
            "model_dir": str(config["model_dir"] or ""),
            "model": selected_model,
            "text": request.text,
            "output_path": str(output_path),
            "voice": selected_voice,
            "language": request.language,
            "sample_rate": str(request.sample_rate),
            "audio_format": request.audio_format,
            "speed": str(request.speed),
        }
        command_template = str(config["command"] or "")
        if command_template:
            return shlex.split(command_template.format(**values))

        if not repo_path:
            raise AppError(
                f"{provider_name}_not_configured",
                f"{provider_name} requires {provider_name.upper()}_COMMAND or {provider_name.upper()}_REPO_PATH",
                status_code=503,
                stage="tts",
                retryable=True,
            )
        repo = Path(repo_path).expanduser().resolve()
        candidates = [repo / "python" / "main.py", repo / "main.py", repo / "demo.py"]
        script_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if script_path is None:
            raise AppError(
                f"{provider_name}_repo_invalid",
                f"No default entrypoint was found under {repo}; set {provider_name.upper()}_COMMAND",
                status_code=503,
                stage="tts",
                retryable=True,
            )

        command = [
            str(config["python"] or "python3"),
            str(script_path),
            "--text",
            request.text,
            "--output",
            str(output_path),
            "--voice",
            selected_voice,
            "--language",
            request.language,
            "--sample-rate",
            str(request.sample_rate),
        ]
        if config["model_dir"]:
            command.extend(["--model-dir", str(config["model_dir"])])
        return command

    def synthesize_segments(self, trace_id: str, request: SegmentedSpeechRequest) -> SegmentedSpeechResponse:
        start = perf_counter()
        provider_name = request.provider or self.settings.default_tts_provider
        if provider_name not in self.providers:
            raise AppError("provider_not_found", f"TTS provider {provider_name} is not configured", status_code=404, stage="tts")
        raise AppError("not_implemented", "synthesize_segments is not implemented for real TTS providers", status_code=501, stage="tts")

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


    # ── Voice Clone Management ──────────────────────────────────────

    def _voice_clones_dir(self) -> Path:
        d = Path(__file__).resolve().parent.parent.parent / "data" / "voice_clones"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _voice_clones_metadata_path(self) -> Path:
        return self._voice_clones_dir() / "metadata.json"

    def _load_voice_clones(self) -> dict:
        import json as _json
        p = self._voice_clones_metadata_path()
        if not p.exists():
            return {}
        try:
            return _json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _save_voice_clones(self, data: dict) -> None:
        import json as _json
        self._voice_clones_metadata_path().write_text(
            _json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def upload_voice_clone(self, audio_content: bytes, voice_id: str, name: str, description: str = "") -> dict:
        d = self._voice_clones_dir()
        audio_path = d / f"{voice_id}.wav"
        audio_path.write_bytes(audio_content)
        meta = self._load_voice_clones()
        meta[voice_id] = {"voice_id": voice_id, "name": name, "description": description, "audio_file": str(audio_path)}
        self._save_voice_clones(meta)
        return meta[voice_id]

    def list_voice_clones(self) -> list[dict]:
        return list(self._load_voice_clones().values())

    def delete_voice_clone(self, voice_id: str) -> bool:
        meta = self._load_voice_clones()
        entry = meta.pop(voice_id, None)
        if entry is None:
            return False
        self._save_voice_clones(meta)
        audio_path = Path(entry.get("audio_file", ""))
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)
        return True


tts_service = TTSService()
