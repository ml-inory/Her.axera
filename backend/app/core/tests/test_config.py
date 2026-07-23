import os
from unittest.mock import patch

from app.core.config import Settings, _get_bool, _get_int, get_settings


class TestHelpers:
    def test_get_bool_truthy(self) -> None:
        for val in ("1", "true", "True", "yes", "on", "ON"):
            with patch.dict(os.environ, {"TEST_VAR": val}):
                assert _get_bool("TEST_VAR", False) is True, f"Failed for {val}"

    def test_get_bool_falsy(self) -> None:
        for val in ("0", "false", "no", "off", ""):
            with patch.dict(os.environ, {"TEST_VAR": val}):
                assert _get_bool("TEST_VAR", True) is False, f"Failed for {val!r}"

    def test_get_bool_missing_returns_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _get_bool("MISSING", True) is True
            assert _get_bool("MISSING", False) is False

    def test_get_int_parses(self) -> None:
        with patch.dict(os.environ, {"NUM": "42"}):
            assert _get_int("NUM", 0) == 42

    def test_get_int_missing_returns_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _get_int("MISSING", 99) == 99


class TestSettings:
    # Settings default values use os.getenv() at import time.
    # Test env overrides via explicit constructor kwargs.

    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            s = Settings()
            assert s.app_name == "Her Voice Dialogue API"
            assert s.api_prefix == "/v1"
            assert s.default_asr_provider == "ax_asr"
            assert s.default_llm_provider == "deepseek"
            assert s.default_tts_provider == "edge_tts"
            assert s.llm_request_timeout == 60.0
            assert s.max_audio_size_mb == 20
            assert s.max_tts_text_length == 1000

    def test_env_override_via_constructor(self) -> None:
        s = Settings(app_name="Custom App", api_prefix="/api")
        assert s.app_name == "Custom App"
        assert s.api_prefix == "/api"

    def test_deepseek_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            s = Settings()
            assert s.deepseek_api_base == "https://api.deepseek.com"
            assert s.deepseek_model == "deepseek-v4-pro"
            assert s.deepseek_api_key is None

    def test_vad_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            s = Settings()
            assert s.silero_vad_sampling_rate == 16000
            assert s.silero_vad_threshold == 0.45
            assert s.silero_vad_min_speech_ms == 350
            assert s.silero_vad_min_silence_ms == 700
            assert s.silero_vad_speech_pad_ms == 300

    def test_vad_env_override_via_constructor(self) -> None:
        s = Settings(silero_vad_threshold=0.6, silero_vad_min_speech_ms=250)
        assert s.silero_vad_threshold == 0.6
        assert s.silero_vad_min_speech_ms == 250

    def test_get_settings_is_singleton(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError
        import pytest
        s = get_settings()
        with pytest.raises(FrozenInstanceError):
            s.app_name = "mutated"  # type: ignore[misc]

    def test_bool_features_default_off(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            s = Settings()
            assert s.enable_openai_compat is False
            assert s.enable_wake_word is False
            assert s.enable_speaker_recognition is False
            assert s.enable_ax_asr is False
            assert s.enable_ax_tts is False
