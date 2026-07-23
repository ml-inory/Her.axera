from app.models.common import ErrorDetail, ErrorResponse, ProviderInfo, JobCreatedResponse


class TestErrorDetail:
    def test_minimal(self) -> None:
        d = ErrorDetail(code="E1", message="fail")
        assert d.code == "E1"
        assert d.message == "fail"
        assert d.stage is None
        assert d.retryable is False

    def test_full(self) -> None:
        d = ErrorDetail(code="E2", message="oops", stage="tts", retryable=True)
        assert d.stage == "tts"
        assert d.retryable is True


class TestErrorResponse:
    def test_with_trace_id(self) -> None:
        r = ErrorResponse(trace_id="trc_x", error=ErrorDetail(code="c", message="m"))
        assert r.trace_id == "trc_x"
        assert r.error.code == "c"

    def test_without_trace_id(self) -> None:
        r = ErrorResponse(error=ErrorDetail(code="c", message="m"))
        assert r.trace_id is None


class TestProviderInfo:
    def test_minimal(self) -> None:
        p = ProviderInfo(name="mock", type="mock")
        assert p.name == "mock"
        assert p.type == "mock"
        assert p.models == []
        assert p.languages == []

    def test_full(self) -> None:
        p = ProviderInfo(
            name="ax_asr", type="local",
            models=["sv-v1"], languages=["zh-CN"],
            audio_formats=["wav"], features=["streaming"],
            metadata={"repo": "hf://..."},
        )
        assert p.models == ["sv-v1"]
        assert p.languages == ["zh-CN"]
        assert p.metadata == {"repo": "hf://..."}


class TestJobCreatedResponse:
    def test_fields(self) -> None:
        r = JobCreatedResponse(trace_id="t1", job_id="j1", status="queued", created_at="now")
        assert r.trace_id == "t1"
        assert r.job_id == "j1"
        assert r.status == "queued"
