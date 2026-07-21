from app.models.llm import ChatMessage
from app.models.openai import OpenAIChatCompletionRequest, OpenAISpeechRequest


class TestOpenAIChatCompletionRequest:
    def test_minimal(self) -> None:
        r = OpenAIChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="hi")],
        )
        assert r.model == "gpt-4o"
        assert r.temperature == 0.7
        assert r.max_tokens == 512
        assert r.stream is False

    def test_with_stream(self) -> None:
        r = OpenAIChatCompletionRequest(
            model="gpt-4o", messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        assert r.stream is True

    def test_with_session(self) -> None:
        r = OpenAIChatCompletionRequest(
            model="gpt-4o", messages=[ChatMessage(role="user", content="hi")],
            session_id="s1", user="u1",
        )
        assert r.session_id == "s1"
        assert r.user == "u1"


class TestOpenAISpeechRequest:
    def test_minimal(self) -> None:
        r = OpenAISpeechRequest(input="hello")
        assert r.model == "tts-1"
        assert r.voice == "alloy"
        assert r.response_format == "mp3"
        assert r.speed == 1.0

    def test_with_options(self) -> None:
        r = OpenAISpeechRequest(
            model="tts-1-hd", input="hi", voice="nova",
            response_format="wav", speed=0.8,
        )
        assert r.model == "tts-1-hd"
        assert r.voice == "nova"
        assert r.response_format == "wav"
        assert r.speed == 0.8
