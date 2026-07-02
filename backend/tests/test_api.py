from io import BytesIO
from base64 import b64encode
import asyncio
import wave

from fastapi.testclient import TestClient

from app.main import app
from app.services.dialogue_service import dialogue_service


client = TestClient(app)


def _wav_bytes(duration_ms: int = 120) -> bytes:
    sample_rate = 16000
    frame_count = int(sample_rate * duration_ms / 1000)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_static_frontend_is_served() -> None:
    response = client.get("/ui/")
    assert response.status_code == 200
    assert "Her.axera Console" in response.text


def test_cors_preflight_for_static_frontend() -> None:
    response = client.options(
        "/v1/tts/providers",
        headers={
            "Origin": "http://127.0.0.1:7860",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_openai_compatible_cascade_endpoints() -> None:
    audio = _wav_bytes()
    asr_response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("input.wav", audio, "audio/wav")},
        data={"model": "mock_asr", "response_format": "verbose_json"},
    )
    assert asr_response.status_code == 200
    asr_payload = asr_response.json()
    assert asr_payload["provider"] == "mock_asr"
    assert asr_payload["text"]

    llm_response = client.post(
        "/v1/chat/completions",
        json={"model": "mock_llm", "messages": [{"role": "user", "content": asr_payload["text"]}]},
    )
    assert llm_response.status_code == 200
    llm_payload = llm_response.json()
    assert llm_payload["provider"] == "mock_llm"
    assert llm_payload["choices"][0]["message"]["content"].startswith("收到")

    tts_response = client.post(
        "/v1/audio/speech",
        json={"model": "mock_tts", "input": llm_payload["choices"][0]["message"]["content"], "response_format": "wav"},
    )
    assert tts_response.status_code == 200
    assert tts_response.headers["content-type"].startswith("audio/wav")
    assert tts_response.headers["x-provider"] == "mock_tts"
    assert tts_response.content.startswith(b"RIFF")


def test_provider_lists_include_axera_models() -> None:
    tts_response = client.get("/v1/tts/providers")
    assert tts_response.status_code == 200
    tts_names = {provider["name"] for provider in tts_response.json()["providers"]}
    assert {"kokoro", "zipvoice"}.issubset(tts_names)

    speaker_response = client.get("/v1/speakers/providers")
    assert speaker_response.status_code == 200
    speaker_names = {provider["name"] for provider in speaker_response.json()["providers"]}
    assert {"mock_speaker", "3d_speaker"}.issubset(speaker_names)


def test_mock_speaker_identification() -> None:
    response = client.post(
        "/v1/speakers/identify",
        files={"audio": ("input.wav", _wav_bytes(), "audio/wav")},
        data={"provider": "mock_speaker"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "mock_speaker"
    assert payload["speaker_id"] == "spk_0"
    assert payload["matches"]


def test_dialogue_websocket_pipeline() -> None:
    with client.websocket_connect("/v1/dialogue/ws") as websocket:
        websocket.send_json(
            {
                "type": "audio",
                "audio_base64": b64encode(_wav_bytes()).decode("ascii"),
                "filename": "input.wav",
                "session_id": "ws-session",
                "asr_provider": "mock_asr",
                "llm_provider": "mock_llm",
                "tts_provider": "mock_tts",
                "output_audio_format": "wav",
            }
        )
        event_types = []
        tts_events = []
        while True:
            event = websocket.receive_json()
            event_types.append(event["type"])
            if event["type"] == "tts_sentence":
                tts_events.append(event)
            if event["type"] == "done":
                break
        assert event_types[0] == "accepted"
        assert "asr" in event_types
        assert "llm" in event_types
        assert "tts_sentence" in event_types
        assert "done" in event_types
        assert tts_events[0]["audio_base64"]


def test_dialogue_websocket_text_pipeline() -> None:
    with client.websocket_connect("/v1/dialogue/ws") as websocket:
        websocket.send_json(
            {
                "type": "text",
                "text": "帮我打开客厅的灯。",
                "session_id": "ws-text-session",
                "llm_provider": "mock_llm",
                "tts_provider": "mock_tts",
                "output_audio_format": "wav",
            }
        )
        event_types = []
        tts_events = []
        while True:
            event = websocket.receive_json()
            event_types.append(event["type"])
            if event["type"] == "tts_sentence":
                tts_events.append(event)
            if event["type"] == "done":
                break
        assert event_types[0] == "accepted"
        assert "user_text" in event_types
        assert "llm" in event_types
        assert "tts_sentence" in event_types
        assert "done" in event_types
        assert tts_events[0]["audio_base64"]


def test_dialogue_websocket_audio_chunk_pipeline() -> None:
    with client.websocket_connect("/v1/dialogue/ws") as websocket:
        websocket.send_json(
            {
                "type": "speech_start",
                "turn_id": "chunk-turn",
                "session_id": "chunk-session",
                "input_sample_rate": 16000,
                "asr_provider": "mock_asr",
                "llm_provider": "mock_llm",
                "tts_provider": "mock_tts",
                "output_audio_format": "wav",
            }
        )
        assert websocket.receive_json()["type"] == "accepted"
        assert websocket.receive_json()["type"] == "speech_started"
        websocket.send_json(
            {
                "type": "audio_chunk",
                "turn_id": "chunk-turn",
                "audio_base64": b64encode(b"\x00\x00" * 1600).decode("ascii"),
            }
        )
        websocket.send_json({"type": "speech_end", "turn_id": "chunk-turn"})

        event_types = []
        while True:
            event = websocket.receive_json()
            event_types.append(event["type"])
            if event["type"] == "done":
                break
        assert event_types[0] == "accepted"
        assert "asr" in event_types
        assert "llm" in event_types


def test_dialogue_websocket_interrupts_active_turn(monkeypatch) -> None:
    async def slow_text_pipeline(**kwargs):
        await asyncio.sleep(10)
        yield {"type": "user_text", "trace_id": kwargs["trace_id"], "session_id": kwargs["session_id"], "text": kwargs["text"]}

    monkeypatch.setattr(dialogue_service, "stream_text_pipeline", slow_text_pipeline)
    with client.websocket_connect("/v1/dialogue/ws") as websocket:
        websocket.send_json(
            {
                "type": "text",
                "turn_id": "old-turn",
                "text": "这是一段会被打断的回复。",
                "session_id": "interrupt-session",
                "llm_provider": "mock_llm",
                "tts_provider": "mock_tts",
            }
        )
        assert websocket.receive_json()["type"] == "accepted"
        import time; time.sleep(0.1)
        websocket.send_json({"type": "speech_start", "turn_id": "new-turn", "session_id": "interrupt-session"})
        event_types = []
        for _ in range(3):
            event_types.append(websocket.receive_json()["type"])
        assert "interrupted" in event_types
        assert "speech_started" in event_types


def test_dialogue_websocket_session_keeps_compact_turn_history() -> None:
    # Clear any residual session data.
    client.delete("/v1/llm/sessions/compact-ws-session/messages")
    for _ in range(2):
        with client.websocket_connect("/v1/dialogue/ws") as websocket:
            websocket.send_json(
                {
                    "type": "audio",
                    "audio_base64": b64encode(_wav_bytes()).decode("ascii"),
                    "filename": "input.wav",
                    "session_id": "compact-ws-session",
                    "asr_provider": "mock_asr",
                    "llm_provider": "mock_llm",
                    "tts_provider": "mock_tts",
                    "output_audio_format": "wav",
                }
            )
            while True:
                event = websocket.receive_json()
                if event["type"] == "done":
                    break

    session_response = client.get("/v1/llm/sessions/compact-ws-session/messages")
    assert session_response.status_code == 200
    messages = session_response.json()["messages"]
    assert len(messages) == 4
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
