from pathlib import Path
import wave

import numpy as np

from app.services.streaming_vad import StreamingSileroVAD, WINDOW_SAMPLES


FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "speech_zh.wav"


class FakeIterator:
    """Mimics the stock Silero VADIterator start/end event protocol."""

    def __init__(self, start_window: int = 4, end_window: int = 8) -> None:
        self.start_window = start_window
        self.end_window = end_window
        self.windows = 0
        self.triggered = False
        self.sampling_rate = 16000
        self.speech_pad_samples = 0

    def __call__(self, x) -> dict | None:
        self.windows += 1
        if not self.triggered and self.start_window <= self.windows < self.end_window:
            self.triggered = True
            return {"start": (self.windows - 1) * WINDOW_SAMPLES}
        if self.triggered and self.windows >= self.end_window:
            self.triggered = False
            return {"end": self.windows * WINDOW_SAMPLES}
        return None

    def reset_states(self) -> None:
        self.windows = 0
        self.triggered = False


def _pcm_chunk(samples: int, value: int = 1000) -> bytes:
    import array

    return array.array("h", [value] * samples).tobytes()


def _make_vad(iterator, **kwargs) -> StreamingSileroVAD:
    kwargs.setdefault("min_speech_ms", 50)
    kwargs.setdefault("min_silence_ms", 100)
    kwargs.setdefault("speech_pad_ms", 0)
    return StreamingSileroVAD(iterator=iterator, **kwargs)


class TestStreamingSileroVAD:
    def test_detects_utterance_with_correct_bounds(self) -> None:
        vad = _make_vad(FakeIterator(start_window=4, end_window=8))
        result = None
        for _ in range(12):
            feed = vad.feed(_pcm_chunk(WINDOW_SAMPLES), 16000)
            if feed.utterance_pcm:
                result = feed
        assert result is not None
        # Trigger window 4 -> start index 1536; end after window 8 -> 4096.
        expected_samples = 8 * WINDOW_SAMPLES - (4 - 1) * WINDOW_SAMPLES
        assert len(result.utterance_pcm) == expected_samples * 2
        assert result.utterance_duration_ms == expected_samples // 16
        assert vad.speech_active is False

    def test_speech_active_while_triggered(self) -> None:
        vad = _make_vad(FakeIterator(start_window=2, end_window=6))
        active_seen = False
        for _ in range(5):
            feed = vad.feed(_pcm_chunk(WINDOW_SAMPLES), 16000)
            active_seen = active_seen or feed.speech_active
        assert active_seen is True

    def test_discards_utterance_shorter_than_min_speech(self) -> None:
        vad = _make_vad(
            FakeIterator(start_window=4, end_window=8),
            min_speech_ms=200,  # 200 ms = 3200 samples > 2048-sample utterance
        )
        utterances = []
        for _ in range(12):
            feed = vad.feed(_pcm_chunk(WINDOW_SAMPLES), 16000)
            if feed.utterance_pcm:
                utterances.append(feed.utterance_pcm)
        assert utterances == []
        assert vad.speech_active is False

    def test_resamples_48k_to_16k(self) -> None:
        vad = _make_vad(FakeIterator(start_window=4, end_window=8))
        # 16k chunks of 512 samples == 48k chunks of 1536 samples.
        utterances = []
        for _ in range(12):
            feed = vad.feed(_pcm_chunk(WINDOW_SAMPLES * 3), 48000)
            if feed.utterance_pcm:
                utterances.append(feed.utterance_pcm)
        assert len(utterances) == 1
        expected_samples = 8 * WINDOW_SAMPLES - (4 - 1) * WINDOW_SAMPLES
        assert len(utterances[0]) == expected_samples * 2

    def test_flush_closes_active_utterance(self) -> None:
        vad = _make_vad(FakeIterator(start_window=2, end_window=99))
        for _ in range(5):
            vad.feed(_pcm_chunk(WINDOW_SAMPLES), 16000)
        assert vad.speech_active is True
        flushed = vad.flush()
        assert flushed.utterance_pcm
        assert vad.speech_active is False

    def test_reset_clears_session(self) -> None:
        iterator = FakeIterator(start_window=2, end_window=4)
        vad = _make_vad(iterator)
        vad.feed(_pcm_chunk(WINDOW_SAMPLES), 16000)
        vad.feed(_pcm_chunk(WINDOW_SAMPLES), 16000)
        assert vad.speech_active is True
        vad.reset()
        assert vad.speech_active is False
        assert iterator.windows == 0


class TestRealSileroIntegration:
    def _fixture_pcm(self) -> bytes:
        with wave.open(str(FIXTURE), "rb") as wav:
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1
            return wav.readframes(wav.getnframes())

    def test_segments_real_speech_at_16k(self) -> None:
        pcm = self._fixture_pcm()
        vad = StreamingSileroVAD(min_speech_ms=100, min_silence_ms=300)
        utterance = None
        for i in range(0, len(pcm), WINDOW_SAMPLES * 2):
            feed = vad.feed(pcm[i : i + WINDOW_SAMPLES * 2], 16000)
            if feed.utterance_pcm:
                utterance = feed
        assert utterance is not None
        assert utterance.utterance_duration_ms >= 100
        assert vad.speech_active is False

    def test_segments_real_speech_resampled_from_48k(self) -> None:
        pcm = self._fixture_pcm()
        samples = np.frombuffer(pcm, dtype=np.int16)
        upsampled = np.repeat(samples, 3).astype(np.int16).tobytes()  # 16k -> 48k
        vad = StreamingSileroVAD(min_speech_ms=100, min_silence_ms=300)
        utterance = None
        for i in range(0, len(upsampled), 4096 * 2):
            feed = vad.feed(upsampled[i : i + 4096 * 2], 48000)
            if feed.utterance_pcm:
                utterance = feed
        assert utterance is not None
        assert utterance.utterance_duration_ms >= 100
