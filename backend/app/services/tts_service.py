from base64 import b64encode
from io import BytesIO
import math
from pathlib import Path
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
    """Generate a sine-tone WAV with harmonics to simulate speech."""
    import io, math, struct, wave
    from hashlib import md5
    import random

    seed = int(md5(text.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    base_freq = rng.choice([180, 200, 220, 250])
    duration_ms = max(500, min(5000, len(text) * 80))
    n_samples = int(sample_rate * duration_ms / 1000)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            env = max(0.0, 1.0 - i / n_samples)
            # Fundamental + 3 harmonics with formant-like weighting
            s = 0.5 * math.sin(2 * math.pi * base_freq * t)
            s += 0.3 * math.sin(2 * math.pi * base_freq * 2 * t)
            s += 0.15 * math.sin(2 * math.pi * base_freq * 3 * t)
            s += 0.08 * math.sin(2 * math.pi * base_freq * 4 * t)
            s *= env * 0.4
            samples.append(int(max(-32768, min(32767, s * 32767))))
        w.writeframes(struct.pack(f'<{n_samples}h', *samples))
    return buf.getvalue(), duration_ms
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
            # --- ax_tts Kokoro voices (NPU local) ---
            VoiceInfo(name="af_heart", display_name="美式女声-Heart", language="en-US", gender="female", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="af_nova", display_name="美式女声-Nova", language="en-US", gender="female", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="af_bella", display_name="美式女声-Bella", language="en-US", gender="female", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="af_sarah", display_name="美式女声-Sarah", language="en-US", gender="female", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="af_nicole", display_name="美式女声-Nicole", language="en-US", gender="female", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="af_sky", display_name="美式女声-Sky", language="en-US", gender="female", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="am_adam", display_name="美式男声-Adam", language="en-US", gender="male", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="am_michael", display_name="美式男声-Michael", language="en-US", gender="male", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="bf_emma", display_name="英式女声-Emma", language="en-GB", gender="female", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="bm_george", display_name="英式男声-George", language="en-GB", gender="male", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="zf_xiaoxiao", display_name="中文女声-晓晓", language="zh-CN", gender="female", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="zf_xiaobei", display_name="中文女声-小北", language="zh-CN", gender="female", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="zm_yunxi", display_name="中文男声-云希", language="zh-CN", gender="male", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="zm_yunjian", display_name="中文男声-云剑", language="zh-CN", gender="male", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="jf_alpha", display_name="日文女声-Alpha", language="ja-JP", gender="female", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="jf_gongitsune", display_name="日文女声-狐", language="ja-JP", gender="female", styles=["neutral"], sample_rates=[24000]),
            VoiceInfo(name="jm_kumo", display_name="日文男声-雲", language="ja-JP", gender="male", styles=["neutral"], sample_rates=[24000]),
            # --- Edge TTS voices (cloud fallback) ---
            VoiceInfo(name="female_default", display_name="默认女声(Edge)", language="zh-CN", gender="female", styles=["neutral", "happy"], sample_rates=[16000, 24000]),
            VoiceInfo(name="male_default", display_name="默认男声(Edge)", language="zh-CN", gender="male", styles=["neutral"], sample_rates=[16000, 24000]),
            VoiceInfo(name="zh-CN-XiaoxiaoNeural", display_name="Edge-晓晓", language="zh-CN", gender="female", styles=["neutral"], sample_rates=[16000, 24000]),
            VoiceInfo(name="en-US-AriaNeural", display_name="Edge-Aria", language="en-US", gender="female", styles=["neutral"], sample_rates=[16000, 24000]),
            VoiceInfo(name="ja-JP-NanamiNeural", display_name="Edge-七海", language="ja-JP", gender="female", styles=["neutral"], sample_rates=[16000, 24000]),
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

    def _synthesize_blocking(self, trace_id: str, request: SpeechRequest) -> SpeechResponse:
        """Blocking wrapper for use with run_in_executor."""
        import asyncio as _asyncio
        try:
            loop = _asyncio.get_event_loop()
        except RuntimeError:
            loop = _asyncio.new_event_loop()
            _asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.synthesize(trace_id, request))

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

        if provider_name == "ax_tts":
            _lang = request.language.split("-")[0] if "-" in request.language else request.language
            if _lang in ("en", "ja") and "edge_tts" in self.providers:
                return await self._synthesize_edge_tts(trace_id, request, selected_model, selected_voice, start)
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
            # Map to ISO 639-1 for ax_tts C++ API
                language=request.language.split("-")[0] if "-" in request.language else request.language,
            audio_url=None,
            audio_base64=b64encode(audio_content).decode("ascii") if request.return_audio_base64 else None,
            audio_format="wav",
            sample_rate=request.sample_rate,
            duration_ms=duration_ms,
            processing_ms=int((perf_counter() - start) * 1000),
            cache_hit=False,
        )


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
            "af_heart": "en-US-AriaNeural", "zm_yunxi": "zh-CN-YunxiNeural",
        }
        _v = voice_aliases.get(selected_voice)
        if not _v:
            _lc = request.language.split("-")[0] if "-" in request.language else request.language
            _v = {"en": "en-US-AriaNeural", "ja": "ja-JP-NanamiNeural", "zh": "zh-CN-XiaoxiaoNeural"}.get(_lc, "zh-CN-XiaoxiaoNeural")
        voice = _v
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
            # Map to ISO 639-1 for ax_tts C++ API
                language=request.language.split("-")[0] if "-" in request.language else request.language,
            audio_url=None,
            audio_base64=audio_base64,
            audio_format="mp3",
            sample_rate=request.sample_rate,
            duration_ms=max(300, len(request.text) * 120),
            processing_ms=int((perf_counter() - start) * 1000),
            cache_hit=False,
        )

    # --- AX TTS API (ax_tts wheel) ---


    # --- AX TTS singleton ---
    _ax_tts_instance = None
    _ax_tts_lock = None

    @classmethod
    def _get_ax_tts(cls):
        import threading
        if cls._ax_tts_lock is None:
            cls._ax_tts_lock = threading.Lock()
        if cls._ax_tts_instance is None:
            with cls._ax_tts_lock:
                if cls._ax_tts_instance is None:
                    from ax_tts import AX_TTS
                    s = get_settings()
                    cls._ax_tts_instance = AX_TTS(
                        model_path=s.ax_tts_model_path,
                        espeak_data_path=s.ax_tts_espeak_data_path,
                        jieba_dict_path=s.ax_tts_jieba_dict_path,
                        max_seq_len=s.ax_tts_max_seq_len,
                        tts_type=s.ax_tts_type,
                    )
        return cls._ax_tts_instance


    # --- Voice lazy-load ---
    _VOICE_DL_LOCK = None
    _VOICE_DL_BASE = "https://hf-mirror.com/inoryQwQ/kokoro.best/resolve/main/models/voices"

    @classmethod
    def _ensure_voice(cls, voice_name: str) -> bool:
        """Check if voice .bin exists; if not, fire background download. Non-blocking."""
        import threading, logging
        from pathlib import Path as _Path
        
        settings = get_settings()
        voices_dir = _Path(settings.ax_tts_model_path).expanduser().resolve() / "voices"
        voice_file = voices_dir / f"{voice_name}.bin"
        
        if voice_file.exists():
            return True
        
        # Fire-and-forget background download
        def _dl():
            import urllib.request
            _log = logging.getLogger("app.tts")
            voices_dir.mkdir(parents=True, exist_ok=True)
            url = f"{cls._VOICE_DL_BASE}/{voice_name}.bin"
            _log.info(f"Background download: {voice_name} from {url}")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Her.axera/1.0"})
                resp = urllib.request.urlopen(req, timeout=300)
                voice_file.write_bytes(resp.read())
                _log.info(f"Voice downloaded: {voice_name}.bin ({voice_file.stat().st_size} bytes)")
            except Exception as e:
                _log.error(f"Failed to download voice {voice_name}: {e}")
        
        threading.Thread(target=_dl, daemon=True).start()
        return False

    async def _synthesize_ax_tts(
        self,
        trace_id: str,
        request: SpeechRequest,
        selected_model: str,
        selected_voice: str,
        start: float,
    ) -> SpeechResponse:
        try:
            from ax_tts import AX_TTS
        except ImportError as exc:
            raise AppError(
                "ax_tts_not_installed",
                "ax_tts wheel is not installed. Install from: "
                "https://github.com/AXERA-TECH/ax_tts_api/releases",
                status_code=503,
                stage="tts",
                retryable=True,
            ) from exc

        voice = selected_voice if selected_voice != "female_default" else self.settings.ax_tts_voice

        # Auto-correct voice based on language (avoids English voice on Chinese text)
        _lang = request.language.split("-")[0] if "-" in request.language else request.language
        _voice_lang_map = {
            "zh": ["zf_xiaoxiao", "zf_xiaobei", "zm_yunxi", "zm_yunjian"],
            "en": ["af_heart", "af_nova", "am_adam", "bf_emma", "bm_george"],
            "ja": ["jf_alpha", "jf_gongitsune", "jm_kumo"],
        }
        if _lang in _voice_lang_map and not any(voice.startswith(p[:2]) for p in _voice_lang_map[_lang]):
            # Voice language doesn't match requested language — auto-correct
            auto_voice = _voice_lang_map[_lang][0]
            import logging
            logging.getLogger("tts").warning(
                f"Voice '{voice}' (lang mismatch) — auto-corrected to '{auto_voice}' for language '{_lang}'"
            )
            voice = auto_voice

        # Lazy load: check if models exist
        from pathlib import Path as _Path
        model_dir = _Path(self.settings.ax_tts_model_path).expanduser().resolve()
        expected_file = model_dir / "kokoro_enc_axera.axmodel"
        if not expected_file.exists():
            from app.services.model_download_service import get_model_download_manager
            mgr = get_model_download_manager()
            started = mgr.start_download_all(model_type="tts")
            if started:
                raise AppError(
                    "ax_tts_model_not_ready",
                    f"TTS models are being downloaded ({len(started)} model(s) queued). "
                    "Track progress at GET /v1/models/download/status?model_type=tts",
                    status_code=503,
                    stage="tts",
                    retryable=True,
                )
            raise AppError(
                "ax_tts_model_not_found",
                f"TTS model not found at {expected_file}. "
                "Trigger download at POST /v1/models/download or run download_models.sh",
                status_code=503,
                stage="tts",
                retryable=True,
            )

        try:
            # Lazy-load voice if missing (non-blocking background download)
            if not self._ensure_voice(voice):
                raise AppError(
                    "voice_downloading",
                    f"Voice '{voice}' not found. Download started in background (~60s). Retry shortly.",
                    status_code=503,
                    stage="tts",
                    retryable=True,
                )
            tts = TTSService._get_ax_tts()
            sr, audio_np = tts.synthesize(
                request.text,
                # Map to ISO 639-1 for ax_tts C++ API
                language=request.language.split("-")[0] if "-" in request.language else request.language,
                voice=voice,
                speed=request.speed,
                fade_out=self.settings.ax_tts_fade_out,
                sample_rate=self.settings.ax_tts_sample_rate,
            )
            # tts is singleton, no close
        except AppError:
            raise  # re-raise voice_downloading and other intentional errors as-is
        except Exception as exc:
            raise AppError(
                "ax_tts_synthesis_failed",
                f"AX TTS synthesis failed: {exc}",
                status_code=502,
                stage="tts",
                retryable=True,
            ) from exc

        import numpy as np
        import io
        import wave as _wave

        pcm = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with _wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        audio_content = buf.getvalue()

        duration_ms = int(len(audio_np) / sr * 1000)
        return SpeechResponse(
            trace_id=trace_id,
            provider="ax_tts",
            model=selected_model,
            voice=voice,
            # Map to ISO 639-1 for ax_tts C++ API
                language=request.language.split("-")[0] if "-" in request.language else request.language,
            audio_url=None,
            audio_base64=b64encode(audio_content).decode("ascii") if request.return_audio_base64 else None,
            audio_format=request.audio_format,
            sample_rate=sr,
            duration_ms=duration_ms,
            processing_ms=int((perf_counter() - start) * 1000),
            cache_hit=False,
        )

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
