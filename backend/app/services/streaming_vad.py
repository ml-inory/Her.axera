"""Streaming Silero VAD with utterance segmentation.

Adapted from huggingface/speech-to-speech's streaming VAD approach: audio is
resampled to 16 kHz, fed to a Silero VADIterator in 512-sample windows, and
completed utterances (with pre-roll padding) are returned as raw PCM16.

The stock ``VADIterator`` (silero-vad >= 6 / silero-vad-axera) emits
``{'start': ...}`` / ``{'end': ...}`` sample-index events rather than audio
buffers, so this service keeps a small rolling history and slices the
utterance itself.  A module-level model cache makes model loading a
one-time cost shared by all connections.
"""

from __future__ import annotations

from dataclasses import dataclass
import array
import logging

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)


TARGET_SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512


def _load_vad_model():
    """Load the board-optimized VAD model, falling back to plain silero-vad."""
    global _vad_model, _vad_module
    if _vad_model is None:
        try:
            import silero_vad_axera as _vad_module
            _vad_model = _vad_module.load_silero_vad()
        except Exception:  # pragma: no cover - dev/x86 fallback (axengine absent)
            import silero_vad as _vad_module
            _vad_model = _vad_module.load_silero_vad()
        logger.info("Streaming VAD model loaded (%s)", _vad_module.__name__)
    return _vad_model, _vad_module


_vad_model = None
_vad_module = None


class _LinearResampler:
    """Streaming linear-interpolation resampler for mono PCM16 audio."""

    def __init__(self, src_rate: int, dst_rate: int = TARGET_SAMPLE_RATE) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._pos = 0.0  # fractional position into the next chunk
        self._last: float | None = None

    def feed(self, pcm: bytes) -> bytes:
        if self.src_rate == self.dst_rate:
            return pcm
        samples = array.array("h")
        samples.frombytes(pcm)
        if not samples:
            return b""
        ratio = self.dst_rate / self.src_rate
        n = len(samples)
        ext: list[float] = [self._last] if self._last is not None else []
        ext.extend(samples)
        out = array.array("h")
        pos = self._pos
        # Global index inside ext-space; ext[0] is the previous chunk's tail.
        while pos < n:
            g = pos + (1 if self._last is not None else 0)
            i = int(g)
            frac = g - i
            if i + 1 < len(ext):
                value = ext[i] * (1.0 - frac) + ext[i + 1] * frac
            else:
                value = ext[i]
            out.append(int(max(-32768, min(32767, value))))
            pos += 1.0 / ratio
        self._last = float(samples[-1])
        self._pos = pos - n
        return out.tobytes()


@dataclass
class VADFeedResult:
    utterance_pcm: bytes | None = None
    speech_active: bool = False
    utterance_duration_ms: int = 0


class StreamingSileroVAD:
    """Per-connection streaming VAD session."""

    def __init__(
        self,
        *,
        model=None,
        iterator=None,
        threshold: float | None = None,
        min_speech_ms: int | None = None,
        min_silence_ms: int | None = None,
        speech_pad_ms: int | None = None,
    ) -> None:
        settings = get_settings()
        self.sample_rate = TARGET_SAMPLE_RATE
        self.threshold = settings.silero_vad_threshold if threshold is None else threshold
        self.min_speech_ms = settings.silero_vad_min_speech_ms if min_speech_ms is None else min_speech_ms
        self.min_silence_ms = settings.silero_vad_min_silence_ms if min_silence_ms is None else min_silence_ms
        self.speech_pad_ms = settings.silero_vad_speech_pad_ms if speech_pad_ms is None else speech_pad_ms
        self._model = model
        self._iterator = iterator
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Start a fresh VAD session (call at free-talk start)."""
        self._window_buf = bytearray()
        self._history = bytearray()
        self._history_offset = 0  # sample index of history[0]
        self._consumed = 0  # total 16 kHz samples fed
        self._speech_start: int | None = None
        self._resampler: _LinearResampler | None = None
        if self._iterator is not None:
            reset = getattr(self._iterator, "reset_states", None)
            if reset:
                reset()
        self._speech_active = False

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def feed(self, pcm: bytes, sample_rate: int) -> VADFeedResult:
        """Feed a PCM16 chunk (any sample rate) and get completed utterances."""
        if self._iterator is None:
            self._iterator = self._build_iterator()
        if sample_rate != TARGET_SAMPLE_RATE:
            if self._resampler is None or self._resampler.src_rate != sample_rate:
                self._resampler = _LinearResampler(sample_rate)
            pcm = self._resampler.feed(pcm)

        self._window_buf.extend(pcm)
        result = VADFeedResult()
        while len(self._window_buf) >= WINDOW_SAMPLES * 2:
            window = bytes(self._window_buf[: WINDOW_SAMPLES * 2])
            del self._window_buf[: WINDOW_SAMPLES * 2]
            event = self._feed_window(window)
            if event is not None:
                utterance = self._handle_event(event)
                if utterance is not None:
                    result.utterance_pcm = utterance
                    result.utterance_duration_ms = len(utterance) // 2 // 16
        result.speech_active = self._speech_active
        return result

    def flush(self) -> VADFeedResult:
        """Force-close an in-progress utterance (call at free-talk end)."""
        if not self._speech_active or self._speech_start is None:
            return VADFeedResult()
        utterance = self._slice_utterance(self._consumed)
        self._finish_utterance()
        if not utterance:
            return VADFeedResult()
        return VADFeedResult(
            utterance_pcm=utterance,
            utterance_duration_ms=len(utterance) // 2 // 16,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_iterator(self):
        model, module = _load_vad_model()
        return module.VADIterator(
            self._model if self._model is not None else model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

    def _feed_window(self, window: bytes) -> dict | None:
        samples = np.frombuffer(window, dtype=np.int16).astype(np.float32) / 32768.0
        self._history.extend(window)
        self._consumed += WINDOW_SAMPLES

        event = self._iterator(np.ascontiguousarray(samples))
        if self._speech_start is None:
            # Keep only a bounded pre-roll when no utterance is in progress.
            max_keep = (self.speech_pad_ms + self.min_silence_ms + 1000) * 16 * 2
            if len(self._history) > max_keep:
                excess = len(self._history) - max_keep
                del self._history[:excess]
                self._history_offset += excess // 2
        return event

    def _handle_event(self, event: dict) -> bytes | None:
        if "start" in event:
            start_sample = max(0, int(event["start"]))
            if start_sample >= self._history_offset:
                drop = (start_sample - self._history_offset) * 2
                del self._history[:drop]
                self._history_offset = start_sample
            self._speech_start = start_sample
            self._speech_active = True
            return None
        if "end" in event:
            end_sample = max(0, int(event["end"]))
            self._speech_active = False
            utterance = self._slice_utterance(end_sample)
            self._finish_utterance()
            return utterance
        return None

    def _slice_utterance(self, end_sample: int) -> bytes | None:
        if self._speech_start is None:
            return None
        start = self._speech_start
        end = max(start, min(end_sample, self._consumed))
        duration_ms = (end - start) // 16
        if duration_ms < self.min_speech_ms:
            return None
        start_off = (start - self._history_offset) * 2
        end_off = (end - self._history_offset) * 2
        start_off = max(0, start_off)
        end_off = min(len(self._history), end_off)
        if end_off <= start_off:
            return None
        return bytes(self._history[start_off:end_off])

    def _finish_utterance(self) -> None:
        if self._speech_start is not None:
            drop = (self._consumed - self._history_offset) * 2
            del self._history[: min(len(self._history), max(0, drop))]
            self._history_offset = self._consumed
        self._speech_start = None
        self._speech_active = False
