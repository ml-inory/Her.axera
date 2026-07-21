from unittest.mock import AsyncMock

from app.core.errors import AppError, app_error_handler


def test_app_error_construction() -> None:
    err = AppError("test_code", "test message")
    assert err.code == "test_code"
    assert err.message == "test message"
    assert err.status_code == 400
    assert err.stage is None
    assert err.retryable is False


def test_app_error_full_fields() -> None:
    err = AppError("e2", "msg2", status_code=503, stage="asr", retryable=True)
    assert err.code == "e2"
    assert err.message == "msg2"
    assert err.status_code == 503
    assert err.stage == "asr"
    assert err.retryable is True


def test_app_error_is_exception() -> None:
    err = AppError("x", "y")
    assert isinstance(err, Exception)


def test_app_error_handler_returns_json_response() -> None:
    err = AppError("vr_fail", "verify failed", status_code=422, stage="vad")
    mock_request = AsyncMock()
    mock_request.state.trace_id = "trc_abc123"

    import asyncio
    response = asyncio.run(app_error_handler(mock_request, err))

    assert response.status_code == 422
    import json
    data = json.loads(response.body)
    assert data["trace_id"] == "trc_abc123"
    assert data["error"]["code"] == "vr_fail"
    assert data["error"]["message"] == "verify failed"
    assert data["error"]["stage"] == "vad"


def test_app_error_handler_without_trace_id() -> None:
    err = AppError("plain", "no trace")
    mock_request = AsyncMock()
    del mock_request.state.trace_id

    import asyncio
    response = asyncio.run(app_error_handler(mock_request, err))

    import json
    data = json.loads(response.body)
    assert data["trace_id"] is None
    assert data["error"]["code"] == "plain"
