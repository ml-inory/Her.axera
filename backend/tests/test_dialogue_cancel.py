import asyncio
import time

from fastapi.testclient import TestClient

from app.main import app
from app.services.dialogue_service import dialogue_service


client = TestClient(app)


def test_cancelled_turn_never_delivers_stale_pipeline_events(monkeypatch) -> None:
    """Barge-in must stop the old turn and suppress any late output from it."""
    async def slow_text_pipeline(**kwargs):
        yield {
            "type": "user_text",
            "trace_id": kwargs["trace_id"],
            "session_id": kwargs["session_id"],
            "text": kwargs["text"],
        }
        await asyncio.sleep(0.3)
        # These would be stale if delivered after the interrupt.
        yield {"type": "llm_delta", "trace_id": kwargs["trace_id"], "session_id": kwargs["session_id"], "text": "stale"}
        yield {"type": "done", "trace_id": kwargs["trace_id"], "session_id": kwargs["session_id"], "sentence_count": 0, "total_processing_ms": 0}

    monkeypatch.setattr(dialogue_service, "stream_text_pipeline", slow_text_pipeline)
    with client.websocket_connect("/v1/dialogue/ws") as websocket:
        websocket.send_json(
            {
                "type": "text",
                "turn_id": "old-turn",
                "text": "这是一段会被打断的回复。",
                "session_id": "cancel-scope-session",
                "llm_provider": "mock_llm",
                "tts_provider": "mock_tts",
            }
        )
        events = []
        while True:
            event = websocket.receive_json()
            events.append(event["type"])
            if event["type"] == "user_text":
                break

        websocket.send_json({"type": "speech_start", "turn_id": "new-turn", "session_id": "cancel-scope-session"})
        seen: list[str] = []
        while True:
            event = websocket.receive_json()
            seen.append(event["type"])
            if event["type"] == "speech_started":
                break

        assert "interrupted" in seen
        assert "speech_started" in seen

        # Give the stale generator time to fire if cancellation failed, then
        # drain the socket via an abort and assert no late pipeline events.
        time.sleep(0.4)
        websocket.send_json({"type": "abort", "turn_id": "probe"})
        while True:
            event = websocket.receive_json()
            seen.append(event["type"])
            if event["type"] == "accepted" and event.get("turn_id") == "probe":
                break

        # The cancelled turn must not emit any further pipeline events.
        assert "llm_delta" not in seen
        assert "done" not in seen
