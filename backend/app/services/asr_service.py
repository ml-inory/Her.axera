import ast
from datetime import datetime
import importlib.util
import os
import re
from pathlib import Path
import sys
import subprocess
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



class SenseVoiceProvider:
    name = "sensevoice"
    language_aliases = {
        "auto": "auto",
        "zh": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
        "cmn": "zh",
        "en": "en",
        "en-us": "en",
        "en-gb": "en",
        "yue": "yue",
        "ja": "ja",
        "ja-jp": "ja",
        "ko": "ko",
        "ko-kr": "ko",
    }

    def __init__(self) -> None:
        self.settings = get_settings()

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            type="local",
            models=[
                "sensevoice_ax650/sensevoice.axmodel",
                "sensevoice_ax650/streaming_sensevoice.axmodel",
                "sensevoice_ax630c/sensevoice.axmodel",
                "sensevoice_ax630c/streaming_sensevoice.axmodel",
            ],
            languages=["auto", "zh", "en", "yue", "ja", "ko"],
            audio_formats=["wav", "mp3", "flac", "pcm", "opus"],
            features=["axengine", "multilingual", "language_auto_detect", "streaming", "vad_compatible"],
            metadata={
                "source_repo": "https://huggingface.co/AXERA-TECH/SenseVoice",
                "python_entrypoint": "python/main.py",
                "python_requires": "3.12",
                "runtime": "pyaxengine==0.1.3rc2",
                "platforms": ["AX650N", "AX630C"],
                "repo_path": self.settings.sensevoice_repo_path,
                "streaming": self.settings.sensevoice_streaming,
            },
        )

    def transcribe(
        self,
        audio_content: bytes,
        filename: str | None,
        language: str | None,
    ) -> tuple[str, dict[str, str | int | bool | None]]:
        if not audio_content:
            raise AppError("asr_no_speech", "Audio payload is empty", status_code=422, stage="asr")

        repo_path = self._required_repo_path()
        script_path = self._resolve_entrypoint(repo_path)
        selected_language = self._normalize_language(language or self.settings.sensevoice_language)
        suffix = Path(filename or "audio.wav").suffix or ".wav"

        with tempfile.NamedTemporaryFile(prefix="her_sensevoice_", suffix=suffix, delete=False) as audio_file:
            audio_file.write(audio_content)
            audio_path = audio_file.name

        command = [self.settings.sensevoice_python, str(script_path), "--input", audio_path, "--language", selected_language]
        if self.settings.sensevoice_streaming:
            command.append("--streaming")

        try:
            completed = subprocess.run(
                command,
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                timeout=self.settings.sensevoice_timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AppError(
                "sensevoice_timeout",
                f"SenseVoice transcription timed out after {self.settings.sensevoice_timeout_sec}s",
                status_code=504,
                stage="asr",
                retryable=True,
            ) from exc
        except OSError as exc:
            raise AppError(
                "sensevoice_invocation_failed",
                f"Failed to execute SenseVoice command: {exc}",
                status_code=503,
                stage="asr",
                retryable=True,
            ) from exc
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        if completed.returncode != 0:
            raise AppError(
                "sensevoice_provider_error",
                f"SenseVoice exited with code {completed.returncode}: {output[-1200:]}",
                status_code=502,
                stage="asr",
                retryable=True,
            )

        text = self._parse_transcription(completed.stdout)
        if not text:
            raise AppError(
                "sensevoice_empty_result",
                f"SenseVoice completed but no transcription text was parsed: {output[-1200:]}",
                status_code=502,
                stage="asr",
                retryable=True,
            )

        model_name = "sensevoice_ax650/streaming_sensevoice.axmodel" if self.settings.sensevoice_streaming else "sensevoice_ax650/sensevoice.axmodel"
        return text, {
            "language": selected_language,
            "model": model_name,
            "repo_path": str(repo_path),
            "entrypoint": str(script_path),
            "streaming": self.settings.sensevoice_streaming,
        }

    def _required_repo_path(self) -> Path:
        if not self.settings.sensevoice_repo_path:
            raise AppError(
                "sensevoice_not_configured",
                "SENSEVOICE_REPO_PATH is required for ASR provider sensevoice",
                status_code=503,
                stage="asr",
                retryable=True,
            )
        repo_path = Path(self.settings.sensevoice_repo_path).expanduser().resolve()
        if not repo_path.exists():
            raise AppError(
                "sensevoice_path_not_found",
                f"SENSEVOICE_REPO_PATH does not exist: {repo_path}",
                status_code=503,
                stage="asr",
                retryable=True,
            )
        return repo_path

    def _resolve_entrypoint(self, repo_path: Path) -> Path:
        candidates = [repo_path / "python" / "main.py", repo_path / "main.py"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise AppError(
            "sensevoice_repo_invalid",
            f"SenseVoice main.py was not found under {repo_path} or {repo_path / 'python'}",
            status_code=503,
            stage="asr",
            retryable=True,
        )

    def _normalize_language(self, language: str | None) -> str:
        normalized = (language or "auto").strip().lower().replace("_", "-")
        if normalized not in self.language_aliases:
            raise AppError(
                "unsupported_language",
                f"SenseVoice language must be one of auto, zh, en, yue, ja, ko; got {language}",
                status_code=422,
                stage="asr",
            )
        return self.language_aliases[normalized]

    def _parse_transcription(self, stdout: str) -> str:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                value = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                value = None
            if isinstance(value, list) and value:
                return "".join(str(item) for item in value).strip()
            if isinstance(value, dict) and value.get("text"):
                return str(value["text"]).strip()

        for line in reversed(lines):
            match = re.search(r"(?:asr result|result|text|transcription)\s*[:：]\s*(.+)$", line, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"\'')
        for line in reversed(lines):
            lowered = line.lower()
            if lowered.startswith(("[info]", "load", "init", "time", "warning", "cost", "rtf", "latency", "total")):
                continue
            if set(line) <= {"-", "=", "*"}:
                continue
            return line.strip().strip('"\'')
        return ""


class FireRedASRAEDProvider:
    name = "fireredasr_aed"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._runner = None

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            type="local",
            models=["fireredasr-aed-ax650n"],
            languages=["zh", "zh-CN", "en", "en-US"],
            audio_formats=["wav", "mp3", "flac", "pcm"],
            features=["axengine", "aed", "vad_split", "max_10s_chunks"],
            metadata={
                "source_repo": "https://huggingface.co/AXERA-TECH/FireRedASR-AED",
                "mirror_repo": "https://hf-mirror.com/AXERA-TECH/FireRedASR-AED",
                "python_requires": "3.12",
                "platforms": ["AX650N"],
                "runtime": [
                    "torch==2.6.0",
                    "torchaudio==2.6.0",
                    "axengine==0.1.3rc2",
                    "silero_vad_axera",
                ],
                "repo_path": self.settings.fireredasr_repo_path,
                "model_dir": self.settings.fireredasr_model_dir,
            },
        )

    def transcribe(self, audio_content: bytes, filename: str | None) -> tuple[str, dict[str, str | int | float | None]]:
        if not audio_content:
            raise AppError("asr_no_speech", "Audio payload is empty", status_code=422, stage="asr")

        suffix = Path(filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(prefix="her_fireredasr_", suffix=suffix, delete=False) as audio_file:
            audio_file.write(audio_content)
            audio_path = audio_file.name

        try:
            result, wav_durations, transcribe_duration = self.runner.transcribe(
                [audio_path],
                beam_size=self.settings.fireredasr_beam_size,
                nbest=self.settings.fireredasr_nbest,
            )
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                "fireredasr_transcription_failed",
                f"FireRedASR-AED transcription failed: {exc}",
                status_code=502,
                stage="asr",
                retryable=True,
            ) from exc
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

        text = str(result.get("text") or "").strip()
        if not text:
            raise AppError(
                "fireredasr_empty_result",
                "FireRedASR-AED completed but returned empty text",
                status_code=502,
                stage="asr",
                retryable=True,
            )
        return text, {
            "model": "fireredasr-aed-ax650n",
            "repo_path": self.settings.fireredasr_repo_path,
            "model_dir": str(self._model_dir),
            "audio_duration_sec": float(sum(wav_durations or [])),
            "transcribe_duration_sec": float(transcribe_duration or 0),
        }

    @property
    def runner(self):
        if self._runner is None:
            self._runner = self._build_runner()
        return self._runner

    def _required_repo_path(self) -> Path:
        if not self.settings.fireredasr_repo_path:
            raise AppError(
                "fireredasr_not_configured",
                "FIREREDASR_REPO_PATH is required for ASR provider fireredasr_aed",
                status_code=503,
                stage="asr",
                retryable=True,
            )
        repo_path = Path(self.settings.fireredasr_repo_path).expanduser().resolve()
        if not repo_path.exists():
            raise AppError(
                "fireredasr_path_not_found",
                f"FIREREDASR_REPO_PATH does not exist: {repo_path}",
                status_code=503,
                stage="asr",
                retryable=True,
            )
        return repo_path

    def _resolve_model_dir(self, repo_path: Path) -> Path:
        model_dir = (
            Path(self.settings.fireredasr_model_dir).expanduser()
            if self.settings.fireredasr_model_dir
            else repo_path / "axmodel"
        )
        model_dir = model_dir.resolve()
        required_files = [
            "encoder.axmodel",
            "decoder_loop.axmodel",
            "cmvn.ark",
            "dict.txt",
            "train_bpe1000.model",
            "pe.npy",
        ]
        missing = [item for item in required_files if not (model_dir / item).exists()]
        if missing:
            raise AppError(
                "fireredasr_model_invalid",
                f"FireRedASR-AED model dir is missing files under {model_dir}: {', '.join(missing)}",
                status_code=503,
                stage="asr",
                retryable=True,
            )
        return model_dir

    def _build_runner(self):
        repo_path = self._required_repo_path()
        self._model_dir = self._resolve_model_dir(repo_path)
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        try:
            from fireredasr_axmodel import FireRedASRAxModel
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                "fireredasr_dependency_missing",
                f"Failed to import FireRedASRAxModel from {repo_path}: {exc}",
                status_code=503,
                stage="asr",
                retryable=True,
            ) from exc
        try:
            return FireRedASRAxModel(
                str(self._model_dir / "encoder.axmodel"),
                str(self._model_dir / "decoder_loop.axmodel"),
                str(self._model_dir / "cmvn.ark"),
                str(self._model_dir / "dict.txt"),
                str(self._model_dir / "train_bpe1000.model"),
                decode_max_len=self.settings.fireredasr_decode_max_len,
                audio_dur=self.settings.fireredasr_max_audio_sec,
            )
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                "fireredasr_runner_init_failed",
                f"Failed to initialize FireRedASR-AED runner: {exc}",
                status_code=503,
                stage="asr",
                retryable=True,
            ) from exc



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
            models=["ax_asr_sensevoice", "ax_asr_whisper_tiny", "ax_asr_whisper_base",
                    "ax_asr_whisper_small", "ax_asr_whisper_turbo"],
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
        self.wenet_provider = WenetONNXProvider()
        if self._should_register_wenet():
            self.providers[self.wenet_provider.name] = self.wenet_provider.info()
        self.sensevoice_provider = SenseVoiceProvider()
        if self._should_register_sensevoice():
            self.providers[self.sensevoice_provider.name] = self.sensevoice_provider.info()
        self.fireredasr_provider = FireRedASRAEDProvider()
        if self._should_register_fireredasr():
            self.providers[self.fireredasr_provider.name] = self.fireredasr_provider.info()
        self.ax_asr_provider = AXASRProvider()
        if self._should_register_ax_asr():
            self.providers[self.ax_asr_provider.name] = self.ax_asr_provider.info()
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

    def _should_register_sensevoice(self) -> bool:
        return (
            self.settings.enable_sensevoice_asr
            or self.settings.default_asr_provider == self.sensevoice_provider.name
            or bool(self.settings.sensevoice_repo_path)
        )

    def _should_register_fireredasr(self) -> bool:
        return (
            self.settings.enable_fireredasr_asr
            or self.settings.default_asr_provider == self.fireredasr_provider.name
            or bool(self.settings.fireredasr_repo_path)
        )

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

        if provider_name == self.sensevoice_provider.name:
            text, metadata = self.sensevoice_provider.transcribe(transcribe_audio, transcribe_filename, selected_language)
            processing_ms = int((perf_counter() - start) * 1000)
            duration_ms = speech_duration_ms or max(1000, min(len(audio_content) // 16, 60000))
            selected_model = model or str(metadata.get("model") or provider_info.models[0])
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

        if provider_name == self.fireredasr_provider.name:
            text, metadata = self.fireredasr_provider.transcribe(transcribe_audio, transcribe_filename)
            processing_ms = int((perf_counter() - start) * 1000)
            duration_ms = (
                speech_duration_ms
                or int(float(metadata.get("audio_duration_sec") or 0) * 1000)
                or max(1000, min(len(audio_content) // 16, 60000))
            )
            selected_model = model or str(metadata.get("model") or provider_info.models[0])
            selected_language = language or "zh"
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

        if provider_name == self.ax_asr_provider.name:
            text, metadata = self.ax_asr_provider.transcribe(transcribe_audio, transcribe_filename, selected_language)
            processing_ms = int((perf_counter() - start) * 1000)
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
