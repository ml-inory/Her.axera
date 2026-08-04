from app.services.realtime_service import RealtimeConfig, RealtimeConnection
from app.services.streaming_vad import VADFeedResult


class FakeVAD:
    """Stub for StreamingSileroVAD with scripted feed responses."""

    def __init__(self, *responses: VADFeedResult) -> None:
        self.responses = list(responses)
        self.fed: list[tuple[bytes, int]] = []

    def feed(self, pcm: bytes, sample_rate: int) -> VADFeedResult:
        self.fed.append((pcm, sample_rate))
        if self.responses:
            return self.responses.pop(0)
        return VADFeedResult()


def _conn(vad: FakeVAD | None = None) -> RealtimeConnection:
    return RealtimeConnection(RealtimeConfig(session_id="test-session"), vad=vad)


class TestRealtimeConnection:
    def test_session_created_payload(self) -> None:
        conn = _conn()
        event = conn.session_created()
        assert event["type"] == "session.created"
        assert event["session"]["id"] == "test-session"
        assert event["session"]["turn_detection"]["type"] == "server_vad"

    def test_session_update_applies_config(self) -> None:
        conn = _conn()
        events = conn.handle_event(
            {
                "type": "session.update",
                "session": {
                    "instructions": "请简短回答。",
                    "voice": "nova",
                    "turn_detection": {"type": "server_vad", "interrupt_response": False},
                    "audio": {"input_sample_rate": 16000, "output_sample_rate": 24000},
                    "her": {"asr_provider": "mock_asr", "llm_provider": "mock_llm"},
                },
            }
        )
        assert events[0]["type"] == "session.updated"
        session = events[0]["session"]
        assert session["instructions"] == "请简短回答。"
        assert session["voice"] == "nova"
        assert session["turn_detection"]["interrupt_response"] is False
        assert session["her"]["llm_provider"] == "mock_llm"

    def test_feed_audio_emits_speech_and_commit_events(self) -> None:
        vad = FakeVAD(
            VADFeedResult(speech_active=True),
            VADFeedResult(speech_active=False, utterance_pcm=b"\x01\x00" * 3200, utterance_duration_ms=200),
        )
        conn = _conn(vad)
        events, commit = conn.feed_audio(b"a" * 1024, 16000)
        assert [event["type"] for event in events] == ["input_audio_buffer.speech_started"]
        assert commit is None
        events, commit = conn.feed_audio(b"b" * 1024, 16000)
        assert [event["type"] for event in events] == [
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "conversation.item.created",
        ]
        assert commit is not None
        assert commit["pcm"] == b"\x01\x00" * 3200
        assert commit["item_id"].startswith("item_")

    def test_response_create_and_cancel(self) -> None:
        conn = _conn()
        events = conn.handle_event({"type": "response.create"})
        assert events[0]["type"] == "response.created"
        assert conn.in_response is True
        events = conn.handle_event({"type": "response.cancel"})
        assert events[0]["type"] == "response.done"
        assert events[0]["response"]["status"] == "cancelled"
        assert conn.in_response is False

    def test_conversation_item_create_sets_last_transcript(self) -> None:
        conn = _conn()
        events = conn.handle_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "你好"}]},
            }
        )
        assert events[0]["type"] == "conversation.item.created"
        assert conn.last_transcript == "你好"

    def test_unsupported_event_returns_error(self) -> None:
        conn = _conn()
        events = conn.handle_event({"type": "response.audio_buffer.something"})
        assert events[0]["type"] == "error"
        assert "unsupported" in events[0]["error"]["message"].lower()
