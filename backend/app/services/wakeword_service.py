"""Wake word detection service using openwakeword (optional dependency)."""

from __future__ import annotations

import logging
import struct

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


class WakeWordService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None

    def available(self) -> bool:
        return _oww_available and self.settings.enable_wake_word

    def detect(self, pcm_bytes: bytes, sample_rate: int = 16000) -> bool:
        """Return True if a wake word is detected in the given PCM audio chunk."""
        if not self.available():
            return False
        model = self._get_model()
        if model is None:
            return False
        try:
            n_samples = len(pcm_bytes) // 2
            audio = np.array(struct.unpack(f"<{n_samples}h", pcm_bytes[:n_samples * 2]), dtype=np.int16)
            prediction = model.predict(audio)
            for _name, score in prediction.items():
                if score >= self.settings.wake_word_threshold:
                    return True
        except Exception:  # noqa: BLE001
            logger.debug("Wake word detection failed", exc_info=True)
        return False

    def _get_model(self):
        if self._model is None and _oww_available:
            try:
                self._model = OWWModel(wakeword_models=[self.settings.wake_word_model])
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load wake word model: %s", self.settings.wake_word_model, exc_info=True)
        return self._model


wakeword_service = WakeWordService()
