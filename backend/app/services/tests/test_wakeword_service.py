from unittest.mock import MagicMock, patch

from app.services.wakeword_service import WakeWordService


def _enable_wake_word(svc: WakeWordService, threshold: float = 0.5) -> None:
    """Force-enable wake word on a frozen Settings instance."""
    object.__setattr__(svc.settings, "enable_wake_word", True)
    object.__setattr__(svc.settings, "wake_word_threshold", threshold)


def _set_oww_available(value: bool) -> None:
    import app.services.wakeword_service as mod
    mod._oww_available = value


class TestWakeWordServiceAvailability:
    def test_not_available_by_default(self) -> None:
        svc = WakeWordService()
        assert svc.available() is False

    def test_not_available_when_oww_missing(self) -> None:
        _set_oww_available(False)
        svc = WakeWordService()
        _enable_wake_word(svc)
        assert svc.available() is False

    def test_available_when_both_conditions_met(self) -> None:
        _set_oww_available(True)
        svc = WakeWordService()
        _enable_wake_word(svc)
        assert svc.available() is True


class TestWakeWordDetection:
    def test_detect_returns_false_when_not_available(self) -> None:
        svc = WakeWordService()
        detected, name = svc.detect(b"dummy")
        assert detected is False
        assert name is None

    def test_detect_returns_false_when_model_is_none(self) -> None:
        _set_oww_available(True)
        svc = WakeWordService()
        _enable_wake_word(svc)
        svc._model = None
        with patch.object(svc, "_get_model", return_value=None):
            detected, name = svc.detect(b"dummy")
        assert detected is False
        assert name is None

    def test_detect_with_model_prediction(self) -> None:
        _set_oww_available(True)
        svc = WakeWordService()
        _enable_wake_word(svc)
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.8}
        svc._model = mock_model
        pcm = b"\x00\x00" * 400
        detected, name = svc.detect(pcm, sample_rate=16000)

        assert detected is True

    def test_detect_below_threshold(self) -> None:
        _set_oww_available(True)
        svc = WakeWordService()
        _enable_wake_word(svc)
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.2}
        svc._model = mock_model
        pcm = b"\x00\x00" * 400
        detected, name = svc.detect(pcm, sample_rate=16000)

        assert detected is False

        assert name is None

    def test_detect_handle_exception(self) -> None:
        _set_oww_available(True)
        svc = WakeWordService()
        _enable_wake_word(svc)
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("audio error")
        svc._model = mock_model
        pcm = b"\x00\x00" * 400
        # Should return False silently on exception
        detected, name = svc.detect(pcm, sample_rate=16000)

        assert detected is False

        assert name is None
