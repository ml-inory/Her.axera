import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "synthetic_conversation_client.py"
SPEC = importlib.util.spec_from_file_location("synthetic_conversation_client", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules["synthetic_conversation_client"] = module
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class TestParser:
    def test_defaults(self) -> None:
        parser = module.build_parser()
        args = parser.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 8080
        assert args.llm_provider == "mock_llm"
        assert args.tts_provider == "mock_tts"

    def test_custom_args(self) -> None:
        parser = module.build_parser()
        args = parser.parse_args(["--host", "10.0.0.5", "--port", "9090", "--turns", "3", "--audio", "x.wav"])
        assert args.host == "10.0.0.5"
        assert args.port == 9090
        assert args.turns == 3
        assert args.audio == "x.wav"


class TestTurnReport:
    def test_to_dict_shape(self) -> None:
        report = module.TurnReport(0, "你好")
        report.first_llm_ms = 123.4
        report.tts_ms = 56.0
        report.tts_sentences = 2
        report.total_ms = 500.0
        data = report.to_dict()
        assert data["turn"] == 0
        assert data["first_llm_token_ms"] == 123.4
        assert data["tts_sentences"] == 2
        assert data["total_ms"] == 500.0


class TestBuiltinScript:
    def test_default_turns_are_defined(self) -> None:
        assert len(module.DEFAULT_TURNS) >= 3
        assert all(isinstance(text, str) and text for text in module.DEFAULT_TURNS)
