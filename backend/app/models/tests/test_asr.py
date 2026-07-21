from app.models.asr import ASRSegment, VADSegment, ASRResult, VADDetectionResponse
from app.models.common import ProviderInfo


class TestASRSegment:
    def test_minimal(self) -> None:
        s = ASRSegment(index=0, start_ms=100, end_ms=500, text="hello")
        assert s.index == 0
        assert s.start_ms == 100
        assert s.end_ms == 500
        assert s.text == "hello"
        assert s.confidence is None
        assert s.speaker is None

    def test_with_confidence_and_speaker(self) -> None:
        s = ASRSegment(index=1, start_ms=200, end_ms=600, text="hi", confidence=0.95, speaker="spk_1")
        assert s.confidence == 0.95
        assert s.speaker == "spk_1"


class TestVADSegment:
    def test_fields(self) -> None:
        v = VADSegment(index=0, start_ms=0, end_ms=1000)
        assert v.index == 0
        assert v.start_ms == 0
        assert v.end_ms == 1000


class TestASRResult:
    def test_minimal(self) -> None:
        r = ASRResult(
            trace_id="t1", provider="mock", model="mock_asr",
            language="zh-CN", text="test", confidence=0.9,
            duration_ms=1000, processing_ms=50,
        )
        assert r.trace_id == "t1"
        assert r.text == "test"
        assert r.confidence == 0.9
        assert r.segments == []
        assert r.words == []

    def test_with_vad_segments(self) -> None:
        r = ASRResult(
            trace_id="t1", provider="mock", model="mock_asr",
            language="zh-CN", text="test", confidence=0.9,
            duration_ms=1000, processing_ms=50,
            vad_segments=[VADSegment(index=0, start_ms=0, end_ms=800)],
            speech_duration_ms=800, vad_processing_ms=20,
        )
        assert len(r.vad_segments) == 1
        assert r.speech_duration_ms == 800
        assert r.vad_processing_ms == 20


class TestVADDetectionResponse:
    def test_fields(self) -> None:
        r = VADDetectionResponse(
            trace_id="t1",
            segments=[VADSegment(index=0, start_ms=0, end_ms=500)],
            speech_duration_ms=500,
            processing_ms=12,
        )
        assert r.trace_id == "t1"
        assert len(r.segments) == 1
        assert r.speech_duration_ms == 500
