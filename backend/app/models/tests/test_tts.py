from app.models.tts import (
    SpeechRequest, SpeechResponse, VoiceInfo, VoicesResponse,
    SpeechSegmentRequest, SegmentedSpeechRequest,
)


class TestSpeechRequest:
    def test_defaults(self) -> None:
        r = SpeechRequest(text="hello")
        assert r.text == "hello"
        assert r.language == "zh-CN"
        assert r.audio_format == "wav"
        assert r.sample_rate == 24000
        assert r.speed == 1.0
        assert r.enable_cache is True
        assert r.return_audio_base64 is False

    def test_with_voice(self) -> None:
        r = SpeechRequest(text="hi", voice="alloy", speed=1.2)
        assert r.voice == "alloy"
        assert r.speed == 1.2


class TestSpeechResponse:
    def test_minimal(self) -> None:
        r = SpeechResponse(
            trace_id="t1", provider="mock", model="mock_tts",
            voice="alloy", language="zh-CN", audio_format="wav",
            sample_rate=24000, duration_ms=500, processing_ms=30,
        )
        assert r.trace_id == "t1"
        assert r.provider == "mock"
        assert r.duration_ms == 500
        assert r.cache_hit is False

    def test_with_base64(self) -> None:
        r = SpeechResponse(
            trace_id="t1", provider="mock", model="mock_tts",
            voice="alloy", language="zh-CN", audio_format="wav",
            sample_rate=24000, duration_ms=500, processing_ms=30,
            audio_base64="AAAA",
        )
        assert r.audio_base64 == "AAAA"


class TestVoiceInfo:
    def test_fields(self) -> None:
        v = VoiceInfo(
            name="alloy", display_name="Alloy",
            language="zh-CN", gender="female",
            styles=["cheerful"], sample_rates=[24000],
        )
        assert v.name == "alloy"
        assert v.display_name == "Alloy"
        assert v.styles == ["cheerful"]


class TestVoicesResponse:
    def test_fields(self) -> None:
        r = VoicesResponse(trace_id="t1", voices=[])
        assert r.trace_id == "t1"
        assert r.voices == []


class TestSegmentedSpeech:
    def test_requests(self) -> None:
        r = SegmentedSpeechRequest(
            segments=[SpeechSegmentRequest(index=0, text="hello")],
        )
        assert r.segments[0].text == "hello"
        assert r.language == "zh-CN"
