from base64 import b64encode
from pathlib import Path
import array
import wave

from fastapi.testclient import TestClient

from app.main import app
from app.services.streaming_vad import WINDOW_SAMPLES


client = TestClient(app)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech_zh.wav"


def test_free_talk_uses_streaming_vad_to_split_utterances() -> None:
    with wave.open(str(FIXTURE), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
    # Append ~1.2s of silence so the VAD emits the utterance end event.
    pcm += array.array("h", [0] * int(16000 * 1.2)).tobytes()

    with client.websocket_connect("/v1/dialogue/ws") as websocket:
        websocket.send_json(
            {
                "type": "free_talk_start",
                "options": {
                    "input_sample_rate": 16000,
                    "channels": 1,
                    "session_id": "free-talk-vad-session",
                    "asr_provider": "mock_asr",
                    "llm_provider": "mock_llm",
                    "tts_provider": "mock_tts",
                    "output_audio_format": "wav",
                },
            }
        )
        assert websocket.receive_json()["type"] == "free_talk_started"

        for i in range(0, len(pcm), WINDOW_SAMPLES * 2):
            chunk = pcm[i : i + WINDOW_SAMPLES * 2]
            websocket.send_json(
                {
                    "type": "audio_chunk",
                    "audio_base64": b64encode(chunk).decode("ascii"),
                }
            )

        websocket.send_json({"type": "free_talk_end"})

        event_types = []
        utterance_seen = False
        while True:
            event = websocket.receive_json()
            event_types.append(event["type"])
            if event["type"] == "utterance_detected":
                utterance_seen = True
            if event["type"] == "done":
                break
        assert utterance_seen is True
        assert "asr" in event_types
        assert "tts_sentence" in event_types
