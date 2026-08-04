from base64 import b64encode
from pathlib import Path
import array
import wave

from fastapi.testclient import TestClient

from app.main import app
from app.services.streaming_vad import WINDOW_SAMPLES


client = TestClient(app)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech_zh.wav"


def test_realtime_websocket_core_dialogue_flow() -> None:
    with wave.open(str(FIXTURE), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
    pcm += array.array("h", [0] * int(16000 * 1.0)).tobytes()  # trailing silence

    with client.websocket_connect("/v1/realtime") as websocket:
        event = websocket.receive_json()
        assert event["type"] == "session.created"

        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "instructions": "你是测试助手，请简短回答。",
                    "voice": "alloy",
                    "audio": {"input_sample_rate": 16000, "output_sample_rate": 24000},
                    "her": {
                        "asr_provider": "mock_asr",
                        "llm_provider": "mock_llm",
                        "tts_provider": "mock_tts",
                    },
                },
            }
        )
        event = websocket.receive_json()
        assert event["type"] == "session.updated"

        for i in range(0, len(pcm), WINDOW_SAMPLES * 2):
            chunk = pcm[i : i + WINDOW_SAMPLES * 2]
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": b64encode(chunk).decode("ascii"),
                }
            )

        types: list[str] = []
        transcript = ""
        status = None
        while True:
            event = websocket.receive_json()
            types.append(event["type"])
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                transcript = event["transcript"]
            if event["type"] == "response.done":
                status = event["response"]["status"]
                break

        assert "input_audio_buffer.speech_started" in types
        assert "input_audio_buffer.speech_stopped" in types
        assert "input_audio_buffer.committed" in types
        assert transcript.strip() != ""
        assert "response.created" in types
        assert "response.output_text.delta" in types
        assert "response.output_audio.delta" in types
        assert status == "completed"
