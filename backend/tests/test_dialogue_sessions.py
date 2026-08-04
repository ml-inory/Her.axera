from base64 import b64encode
import array
import time

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _pcm_silence(samples: int) -> bytes:
    return array.array("h", [0] * samples).tobytes()


def test_one_shot_audio_accepts_raw_pcm_encoding() -> None:
    pcm = _pcm_silence(16000)  # 1 second of 16k PCM16
    with client.websocket_connect("/v1/dialogue/ws") as websocket:
        websocket.send_json(
            {
                "type": "audio",
                "audio_base64": b64encode(pcm).decode("ascii"),
                "encoding": "pcm",
                "sample_rate": 16000,
                "channels": 1,
                "session_id": "pcm-one-shot-session",
                "asr_provider": "mock_asr",
                "llm_provider": "mock_llm",
                "tts_provider": "mock_tts",
                "output_audio_format": "wav",
            }
        )
        event_types = []
        while True:
            event = websocket.receive_json()
            event_types.append(event["type"])
            if event["type"] == "done":
                break
        assert "asr" in event_types
        assert "tts_sentence" in event_types


def test_dialogue_sessions_diagnostics_clears_after_turn() -> None:
    session_id = f"diag-{int(time.time() * 1000)}"
    with client.websocket_connect("/v1/dialogue/ws") as websocket:
        websocket.send_json(
            {
                "type": "text",
                "text": "你好",
                "session_id": session_id,
                "llm_provider": "mock_llm",
                "tts_provider": "mock_tts",
                "output_audio_format": "wav",
            }
        )
        while True:
            event = websocket.receive_json()
            if event["type"] == "done":
                break

    # The pipeline wrapper marks the session idle once the generator finishes.
    for _ in range(20):
        payload = client.get("/v1/dialogue/sessions").json()
        if all(s["session_id"] != session_id for s in payload["sessions"]):
            break
        time.sleep(0.05)
    payload = client.get("/v1/dialogue/sessions").json()
    assert isinstance(payload["sessions"], list)
    assert all(s["session_id"] != session_id for s in payload["sessions"])
