import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from time import perf_counter

from app.core.config import get_settings
from app.core.errors import AppError
from app.models.common import ProviderInfo
from app.models.speaker import SpeakerIdentifyResponse, SpeakerMatch


class SpeakerService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.providers = {
            "mock_speaker": ProviderInfo(
                name="mock_speaker",
                type="mock",
                models=["mock-speaker-general"],
                languages=[],
                audio_formats=["wav", "pcm", "mp3", "flac"],
                features=["speaker_identification"],
            ),
            "3d_speaker": ProviderInfo(
                name="3d_speaker",
                type="local",
                models=["3D-Speaker-MT.Axera"],
                languages=[],
                audio_formats=["wav", "pcm", "mp3", "flac"],
                features=["axengine", "speaker_embedding", "speaker_identification"],
                metadata={
                    "source_repo": "https://huggingface.co/AXERA-TECH/3D-Speaker-MT.Axera",
                    "repo_path": self.settings.speaker_repo_path,
                    "model_dir": self.settings.speaker_model_dir,
                    "command_env": "SPEAKER_COMMAND",
                    "enabled": self._should_enable_3d_speaker(),
                },
            ),
        }

    def list_providers(self) -> list[ProviderInfo]:
        return list(self.providers.values())

    def identify(
        self,
        *,
        trace_id: str,
        audio_content: bytes,
        filename: str | None,
        provider: str | None,
        top_k: int,
    ) -> SpeakerIdentifyResponse:
        start = perf_counter()
        provider_name = provider or self.settings.default_speaker_provider
        provider_info = self.providers.get(provider_name)
        if provider_info is None:
            raise AppError(
                "provider_not_found",
                f"Speaker provider {provider_name} is not configured",
                status_code=404,
                stage="speaker",
            )
        if not audio_content:
            raise AppError("speaker_no_audio", "Audio payload is empty", status_code=422, stage="speaker")

        if provider_name == "3d_speaker":
            return self._identify_3d_speaker(trace_id, audio_content, filename, top_k, start)

        confidence = min(0.99, max(0.5, len(audio_content) / 1_000_000))
        matches = [SpeakerMatch(speaker_id="spk_0", score=confidence, label="mock")]
        return SpeakerIdentifyResponse(
            trace_id=trace_id,
            provider=provider_name,
            model=provider_info.models[0],
            speaker_id=matches[0].speaker_id,
            confidence=matches[0].score,
            matches=matches[: max(1, top_k)],
            processing_ms=int((perf_counter() - start) * 1000),
        )

    def _should_enable_3d_speaker(self) -> bool:
        return bool(
            self.settings.enable_speaker_recognition
            or self.settings.speaker_repo_path
            or self.settings.speaker_command
        )

    def _identify_3d_speaker(
        self,
        trace_id: str,
        audio_content: bytes,
        filename: str | None,
        top_k: int,
        start: float,
    ) -> SpeakerIdentifyResponse:
        suffix = Path(filename or "audio.wav").suffix or ".wav"
        input_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="her_speaker_", suffix=suffix, delete=False) as audio_file:
                audio_file.write(audio_content)
                input_path = audio_file.name
            command = self._build_3d_speaker_command(input_path, top_k)
            completed = subprocess.run(
                command,
                cwd=self.settings.speaker_repo_path or None,
                capture_output=True,
                text=True,
                timeout=self.settings.speaker_timeout_sec,
                check=False,
            )
            output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
            if completed.returncode != 0:
                raise AppError(
                    "speaker_provider_error",
                    f"3d_speaker exited with code {completed.returncode}: {output[-1200:]}",
                    status_code=502,
                    stage="speaker",
                    retryable=True,
                )
            matches = self._parse_matches(completed.stdout)
            if not matches:
                raise AppError(
                    "speaker_empty_result",
                    f"3d_speaker completed but no speaker result was parsed: {output[-1200:]}",
                    status_code=502,
                    stage="speaker",
                    retryable=True,
                )
            return SpeakerIdentifyResponse(
                trace_id=trace_id,
                provider="3d_speaker",
                model="3D-Speaker-MT.Axera",
                speaker_id=matches[0].speaker_id,
                confidence=matches[0].score,
                matches=matches[: max(1, top_k)],
                processing_ms=int((perf_counter() - start) * 1000),
            )
        except AppError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise AppError(
                "speaker_timeout",
                f"3d_speaker identification timed out after {self.settings.speaker_timeout_sec}s",
                status_code=504,
                stage="speaker",
                retryable=True,
            ) from exc
        finally:
            if input_path:
                try:
                    Path(input_path).unlink()
                except OSError:
                    pass

    def _build_3d_speaker_command(self, input_path: str, top_k: int) -> list[str]:
        values = {
            "python": self.settings.speaker_python,
            "repo_path": self.settings.speaker_repo_path or "",
            "model_dir": self.settings.speaker_model_dir or "",
            "input_path": input_path,
            "top_k": str(top_k),
        }
        if self.settings.speaker_command:
            return shlex.split(self.settings.speaker_command.format(**values))

        if not self.settings.speaker_repo_path:
            raise AppError(
                "speaker_not_configured",
                "3d_speaker requires SPEAKER_COMMAND or SPEAKER_REPO_PATH",
                status_code=503,
                stage="speaker",
                retryable=True,
            )
        repo = Path(self.settings.speaker_repo_path).expanduser().resolve()
        candidates = [repo / "python" / "main.py", repo / "main.py", repo / "demo.py"]
        script_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if script_path is None:
            raise AppError(
                "speaker_repo_invalid",
                f"No default entrypoint was found under {repo}; set SPEAKER_COMMAND",
                status_code=503,
                stage="speaker",
                retryable=True,
            )
        command = [self.settings.speaker_python, str(script_path), "--input", input_path, "--top-k", str(top_k)]
        if self.settings.speaker_model_dir:
            command.extend(["--model-dir", self.settings.speaker_model_dir])
        return command

    def _parse_matches(self, stdout: str) -> list[SpeakerMatch]:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                if "matches" in data and isinstance(data["matches"], list):
                    return [
                        SpeakerMatch(
                            speaker_id=str(item.get("speaker_id") or item.get("id") or "spk_0"),
                            score=float(item.get("score") or item.get("confidence") or 0.0),
                            label=item.get("label"),
                        )
                        for item in data["matches"]
                        if isinstance(item, dict)
                    ]
                if data.get("speaker_id"):
                    return [
                        SpeakerMatch(
                            speaker_id=str(data["speaker_id"]),
                            score=float(data.get("score") or data.get("confidence") or 0.0),
                            label=data.get("label"),
                        )
                    ]
        return []


speaker_service = SpeakerService()
