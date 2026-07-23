"""Wake word detection service using openwakeword (optional dependency).

Supports dynamic registration via openwakeword's from_clip API
and persistence to disk for registered custom wake words.
"""

from __future__ import annotations

from base64 import b64decode
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import struct
import tempfile
import wave
from io import BytesIO

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_oww_available = False
try:
    import openwakeword  # type: ignore[import-untyped]
    from openwakeword.model import Model as OWWModel  # type: ignore[import-untyped]

    _oww_available = True
except ImportError:
    openwakeword = None  # type: ignore[assignment]
    OWWModel = None  # type: ignore[assignment, misc]

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_WAKEWORDS_FILE = _DATA_DIR / "wakewords.json"
_WAKEWORDS_AUDIO_DIR = _DATA_DIR / "wakewords"


class WakeWordService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._custom_models: list[str] = []
        self._registry: dict[str, dict] = {}
        self._load_registry()

    # ── Availability ───────────────────────────────────────────────

    def available(self) -> bool:
        return _oww_available and self.settings.enable_wake_word

    # ── Registry Persistence ───────────────────────────────────────

    def _load_registry(self) -> None:
        if not _WAKEWORDS_FILE.exists():
            return
        try:
            self._registry = json.loads(_WAKEWORDS_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load wake word registry", exc_info=True)

    def _save_registry(self) -> None:
        _WAKEWORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _WAKEWORDS_FILE.write_text(json.dumps(self._registry, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── CRUD API ───────────────────────────────────────────────────

    def register(self, name: str, audio_base64: str, description: str = "") -> dict:
        """Register a custom wake word from a base64 WAV/PCM audio clip.

        Uses openwakeword's from_clip to create an embedding model.
        Returns status info.
        """
        if not self.available():
            raise RuntimeError("Wake word detection is not enabled. Set ENABLE_WAKE_WORD=true and install openwakeword.")

        try:
            audio_bytes = b64decode(audio_base64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 audio: {exc}") from exc

        # Save audio clip
        _WAKEWORDS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = _WAKEWORDS_AUDIO_DIR / f"{name}.wav"
        audio_path.write_bytes(audio_bytes)

        # Try openwakeword from_clip for dynamic registration
        clip_loaded = False
        if _oww_available:
            try:
                # Read audio as numpy array
                audio_np = self._read_audio_clip(audio_bytes)
                if audio_np is not None and len(audio_np) > 0:
                    # openwakeword from_clip creates an embedding model from a short audio clip
                    OWWModel.from_clip(audio_np, name, save_path=str(_WAKEWORDS_AUDIO_DIR))
                    clip_loaded = True
                    logger.info("Registered wake word '%s' via from_clip", name)
            except Exception:
                logger.warning("from_clip registration failed for '%s', falling back to file-based only", name, exc_info=True)

        # Update or create registry entry
        is_new = name not in self._registry
        self._registry[name] = {
            "name": name,
            "description": description,
            "created_at": self._registry.get(name, {}).get("created_at", datetime.now(timezone.utc).isoformat()),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sample_count": self._registry.get(name, {}).get("sample_count", 0) + 1,
            "active": True,
            "from_clip_loaded": clip_loaded,
        }
        self._save_registry()

        # Reload model to include new wake word
        self._model = None

        return {
            "status": "updated" if not is_new else "registered",
            "sample_count": self._registry[name]["sample_count"],
            "from_clip_loaded": clip_loaded,
        }

    def list_wakewords(self) -> list[dict]:
        """List all registered wake words."""
        result = []
        for name, info in self._registry.items():
            result.append({
                "name": name,
                "description": info.get("description", ""),
                "created_at": info.get("created_at", ""),
                "sample_count": info.get("sample_count", 0),
                "active": info.get("active", True),
            })
        return sorted(result, key=lambda w: w["created_at"], reverse=True)

    def delete(self, name: str) -> bool:
        """Delete a registered wake word."""
        if name not in self._registry:
            return False
        self._registry.pop(name, None)
        self._save_registry()

        # Remove audio file
        audio_path = _WAKEWORDS_AUDIO_DIR / f"{name}.wav"
        audio_path.unlink(missing_ok=True)

        # Remove from_clip model if exists
        clip_path = _WAKEWORDS_AUDIO_DIR / f"{name}.onnx"
        clip_path.unlink(missing_ok=True)

        # Reload model
        self._model = None

        logger.info("Deleted wake word '%s'", name)
        return True

    # ── Detection ──────────────────────────────────────────────────

    def detect(self, pcm_bytes: bytes, sample_rate: int = 16000) -> tuple[bool, str | None]:
        """Return (detected, wake_word_name) for the given PCM audio chunk."""
        if not self.available():
            return False, None
        model = self._get_model()
        if model is None:
            return False, None
        try:
            n_samples = len(pcm_bytes) // 2
            if n_samples == 0:
                return False, None
            audio = np.array(struct.unpack(f"<{n_samples}h", pcm_bytes[:n_samples * 2]), dtype=np.int16)
            prediction = model.predict(audio)
            for name, score in prediction.items():
                if score >= self.settings.wake_word_threshold:
                    return True, name
        except Exception:  # noqa: BLE001
            logger.debug("Wake word detection failed", exc_info=True)
        return False, None

    # ── Model Management ───────────────────────────────────────────

    def _get_model(self):
        if self._model is None and _oww_available:
            try:
                # Collect all wake word models: built-in + custom from_clip
                model_names = [self.settings.wake_word_model]

                # Add custom registered models that have onnx files
                for name, info in self._registry.items():
                    if info.get("from_clip_loaded") or info.get("active"):
                        clip_path = _WAKEWORDS_AUDIO_DIR / f"{name}.onnx"
                        if clip_path.exists():
                            model_names.append(str(clip_path))

                self._model = OWWModel(wakeword_models=model_names)
                logger.info("Wake word model loaded with %d models: %s", len(model_names), model_names)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load wake word model with custom entries", exc_info=True)
                # Fall back to just the built-in model
                try:
                    self._model = OWWModel(wakeword_models=[self.settings.wake_word_model])
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to load any wake word model", exc_info=True)
        return self._model

    def _read_audio_clip(self, audio_bytes: bytes) -> np.ndarray | None:
        """Read WAV or raw PCM bytes into a float32 numpy array at 16kHz."""
        try:
            # Try as WAV
            with wave.open(BytesIO(audio_bytes), "rb") as wf:
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sr = wf.getframerate()
                raw = wf.readframes(n_frames)

            if sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                return None

            audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
            audio = audio / np.iinfo(dtype).max
            return audio
        except Exception:
            # Try as raw 16-bit PCM at 16kHz
            try:
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                return audio
            except Exception:
                return None


wakeword_service = WakeWordService()
