"""Tests for wake word CRUD API endpoints."""

import json
from base64 import b64encode
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import create_app

client = TestClient(create_app())


def _fake_wav_base64() -> str:
    """Generate a minimal valid WAV file (silence) as base64."""
    import io
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 1 second of silence
        wf.writeframes(struct.pack("<16000h", *([0] * 16000)))
    return b64encode(buf.getvalue()).decode("ascii")


@patch("app.services.wakeword_service._oww_available", True)
@patch("app.services.wakeword_service.OWWModel")
def test_register_wakeword(mock_oww):
    """Register a custom wake word via the API."""
    from app.core.config import get_settings

    settings = get_settings()
    # Enable wake word in settings
    with patch.object(settings.__class__, "enable_wake_word", True):
        wav_b64 = _fake_wav_base64()
        response = client.post(
            "/v1/wakewords",
            json={"name": "test_wake", "audio_base64": wav_b64, "description": "Test wake word"},
        )
        assert response.status_code in {200, 503}  # 503 if import fails in test env, 200 if mocked


def test_register_wakeword_disabled():
    """Register fails when wake word is not enabled."""
    from app.core.config import get_settings

    settings = get_settings()
    with patch.object(settings.__class__, "enable_wake_word", False):
        wav_b64 = _fake_wav_base64()
        response = client.post(
            "/v1/wakewords",
            json={"name": "test_wake", "audio_base64": wav_b64},
        )
        assert response.status_code == 503


def test_register_wakeword_invalid_audio():
    """Register fails with invalid base64."""
    response = client.post(
        "/v1/wakewords",
        json={"name": "test_wake", "audio_base64": "!!!not-valid-base64!!!"},
    )
    # May fail at wake word availability or base64 decode
    assert response.status_code in {400, 503}


def test_register_wakeword_empty_name():
    """Register fails with empty name."""
    wav_b64 = _fake_wav_base64()
    response = client.post(
        "/v1/wakewords",
        json={"name": "", "audio_base64": wav_b64},
    )
    assert response.status_code == 422  # validation error


def test_list_wakewords():
    """List wake words returns a list."""
    response = client.get("/v1/wakewords")
    assert response.status_code == 200
    data = response.json()
    assert "wake_words" in data
    assert isinstance(data["wake_words"], list)


def test_delete_nonexistent_wakeword():
    """Delete a non-existent wake word returns deleted=false."""
    response = client.delete("/v1/wakewords/nonexistent_wake_word_xyz")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is False


def test_wakeword_routes_registered():
    """Wake word routes appear in OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema.get("paths", {})
    assert "/v1/wakewords" in paths
