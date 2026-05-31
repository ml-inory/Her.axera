from pathlib import Path
import os
import tempfile
from time import perf_counter
from typing import Any

from app.core.config import get_settings
from app.core.errors import AppError
from app.models.asr import VADSegment


class VADResult:
    def __init__(
        self,
        *,
        audio_content: bytes,
        segments: list[VADSegment],
        speech_duration_ms: int,
        processing_ms: int,
    ) -> None:
        self.audio_content = audio_content
        self.segments = segments
        self.speech_duration_ms = speech_duration_ms
        self.processing_ms = processing_ms


class SileroVADService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any | None = None

    def extract_speech(self, audio_content: bytes, filename: str | None) -> VADResult:
        if not audio_content:
            raise AppError("vad_no_audio", "Audio payload is empty", status_code=422, stage="vad")

        start = perf_counter()
        suffix = Path(filename or "audio.wav").suffix or ".wav"
        input_path = None
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="her_vad_input_", suffix=suffix, delete=False) as audio_file:
                audio_file.write(audio_content)
                input_path = audio_file.name
            with tempfile.NamedTemporaryFile(prefix="her_vad_speech_", suffix=".wav", delete=False) as speech_file:
                output_path = speech_file.name

            silero = self._load_silero()
            sampling_rate = self.settings.silero_vad_sampling_rate
            wav = silero["read_audio"](input_path, sampling_rate=sampling_rate)
            timestamps = silero["get_speech_timestamps"](
                wav,
                self.model,
                sampling_rate=sampling_rate,
                threshold=self.settings.silero_vad_threshold,
                min_speech_duration_ms=self.settings.silero_vad_min_speech_ms,
                min_silence_duration_ms=self.settings.silero_vad_min_silence_ms,
                speech_pad_ms=self.settings.silero_vad_speech_pad_ms,
            )
            if not timestamps:
                raise AppError("vad_no_speech", "No speech was detected in audio", status_code=422, stage="vad")

            speech_wav = silero["collect_chunks"](timestamps, wav)
            silero["save_audio"](output_path, speech_wav, sampling_rate=sampling_rate)
            speech_content = Path(output_path).read_bytes()
            segments = [
                VADSegment(
                    index=index,
                    start_ms=int(item["start"] * 1000 / sampling_rate),
                    end_ms=int(item["end"] * 1000 / sampling_rate),
                )
                for index, item in enumerate(timestamps)
            ]
            speech_duration_ms = sum(segment.end_ms - segment.start_ms for segment in segments)
            return VADResult(
                audio_content=speech_content,
                segments=segments,
                speech_duration_ms=speech_duration_ms,
                processing_ms=int((perf_counter() - start) * 1000),
            )
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                "vad_processing_failed",
                f"Silero VAD failed: {exc}",
                status_code=502,
                stage="vad",
                retryable=True,
            ) from exc
        finally:
            for path in (input_path, output_path):
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def detect_speech(self, audio_content: bytes, filename: str | None) -> VADResult:
        if not audio_content:
            raise AppError("vad_no_audio", "Audio payload is empty", status_code=422, stage="vad")

        start = perf_counter()
        suffix = Path(filename or "audio.wav").suffix or ".wav"
        input_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="her_vad_detect_", suffix=suffix, delete=False) as audio_file:
                audio_file.write(audio_content)
                input_path = audio_file.name

            silero = self._load_silero()
            sampling_rate = self.settings.silero_vad_sampling_rate
            wav = silero["read_audio"](input_path, sampling_rate=sampling_rate)
            timestamps = silero["get_speech_timestamps"](
                wav,
                self.model,
                sampling_rate=sampling_rate,
                threshold=self.settings.silero_vad_threshold,
                min_speech_duration_ms=self.settings.silero_vad_min_speech_ms,
                min_silence_duration_ms=self.settings.silero_vad_min_silence_ms,
                speech_pad_ms=self.settings.silero_vad_speech_pad_ms,
            )
            segments = [
                VADSegment(
                    index=index,
                    start_ms=int(item["start"] * 1000 / sampling_rate),
                    end_ms=int(item["end"] * 1000 / sampling_rate),
                )
                for index, item in enumerate(timestamps)
            ]
            speech_duration_ms = sum(segment.end_ms - segment.start_ms for segment in segments)
            return VADResult(
                audio_content=b"",
                segments=segments,
                speech_duration_ms=speech_duration_ms,
                processing_ms=int((perf_counter() - start) * 1000),
            )
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                "vad_processing_failed",
                f"Silero VAD failed: {exc}",
                status_code=502,
                stage="vad",
                retryable=True,
            ) from exc
        finally:
            if input_path:
                try:
                    os.remove(input_path)
                except OSError:
                    pass

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_silero()["load_silero_vad"]()
        return self._model

    def _load_silero(self) -> dict[str, Any]:
        try:
            from silero_vad import collect_chunks, get_speech_timestamps, load_silero_vad, read_audio, save_audio
        except ModuleNotFoundError as exc:
            raise AppError(
                "vad_dependency_missing",
                "silero-vad is not installed; install backend requirements with silero-vad enabled",
                status_code=503,
                stage="vad",
                retryable=True,
            ) from exc
        return {
            "collect_chunks": collect_chunks,
            "get_speech_timestamps": get_speech_timestamps,
            "load_silero_vad": load_silero_vad,
            "read_audio": read_audio,
            "save_audio": save_audio,
        }


vad_service = SileroVADService()
