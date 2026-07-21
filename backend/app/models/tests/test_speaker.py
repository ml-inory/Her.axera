from app.models.speaker import (
    SpeakerMatch, SpeakerIdentifyResponse, SpeakerProfile,
    SpeakerEnrollResponse, SpeakerDeleteResponse,
)


class TestSpeakerMatch:
    def test_fields(self) -> None:
        m = SpeakerMatch(speaker_id="spk_1", score=0.95, label="Alice")
        assert m.speaker_id == "spk_1"
        assert m.score == 0.95
        assert m.label == "Alice"

    def test_no_label(self) -> None:
        m = SpeakerMatch(speaker_id="spk_2", score=0.5)
        assert m.label is None


class TestSpeakerIdentifyResponse:
    def test_minimal(self) -> None:
        r = SpeakerIdentifyResponse(
            trace_id="t1", provider="3d_speaker", model="sm-v1",
            speaker_id="spk_1", confidence=0.9, processing_ms=200,
        )
        assert r.trace_id == "t1"
        assert r.speaker_id == "spk_1"
        assert r.matches == []

    def test_with_matches(self) -> None:
        r = SpeakerIdentifyResponse(
            trace_id="t1", provider="3d_speaker", model="sm-v1",
            speaker_id="spk_1", confidence=0.9, processing_ms=200,
            matches=[SpeakerMatch(speaker_id="spk_1", score=0.9)],
        )
        assert len(r.matches) == 1


class TestSpeakerProfile:
    def test_defaults(self) -> None:
        p = SpeakerProfile(speaker_id="spk_1", name="Alice")
        assert p.speaker_id == "spk_1"
        assert p.name == "Alice"
        assert p.description == ""
        assert p.created_at == ""
        assert p.audio_count == 0


class TestSpeakerEnrollResponse:
    def test_fields(self) -> None:
        r = SpeakerEnrollResponse(
            trace_id="t1", speaker_id="spk_1", name="Alice",
            provider="3d_speaker", processing_ms=100,
        )
        assert r.trace_id == "t1"
        assert r.speaker_id == "spk_1"


class TestSpeakerDeleteResponse:
    def test_deleted(self) -> None:
        r = SpeakerDeleteResponse(trace_id="t1", speaker_id="spk_1", deleted=True)
        assert r.deleted is True
