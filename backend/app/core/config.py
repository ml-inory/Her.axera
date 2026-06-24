from dataclasses import dataclass
from functools import lru_cache
import os


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Her Voice Dialogue API")
    api_prefix: str = os.getenv("API_PREFIX", "/v1")
    log_level: str = os.getenv("LOG_LEVEL", "info")
    default_asr_provider: str = os.getenv("DEFAULT_ASR_PROVIDER", "mock_asr")
    default_llm_provider: str = os.getenv("DEFAULT_LLM_PROVIDER", "mock_llm")
    default_tts_provider: str = os.getenv("DEFAULT_TTS_PROVIDER", "mock_tts")
    default_speaker_provider: str = os.getenv("DEFAULT_SPEAKER_PROVIDER", "mock_speaker")
    edge_tts_voice: str = os.getenv("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    llm_request_timeout: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
    deepseek_api_base: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    max_audio_size_mb: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "20"))
    max_tts_text_length: int = int(os.getenv("MAX_TTS_TEXT_LENGTH", "1000"))
    enable_request_auth: bool = _get_bool("ENABLE_REQUEST_AUTH", False)
    enable_kokoro_tts: bool = _get_bool("ENABLE_KOKORO_TTS", False)
    kokoro_repo_path: str | None = os.getenv("KOKORO_REPO_PATH")
    kokoro_model_dir: str | None = os.getenv("KOKORO_MODEL_DIR")
    kokoro_python: str = os.getenv("KOKORO_PYTHON", "python3")
    kokoro_command: str | None = os.getenv("KOKORO_COMMAND")
    kokoro_timeout_sec: int = _get_int("KOKORO_TIMEOUT_SEC", 120)
    enable_zipvoice_tts: bool = _get_bool("ENABLE_ZIPVOICE_TTS", False)
    zipvoice_repo_path: str | None = os.getenv("ZIPVOICE_REPO_PATH")
    zipvoice_model_dir: str | None = os.getenv("ZIPVOICE_MODEL_DIR")
    zipvoice_python: str = os.getenv("ZIPVOICE_PYTHON", "python3")
    zipvoice_command: str | None = os.getenv("ZIPVOICE_COMMAND")
    zipvoice_timeout_sec: int = _get_int("ZIPVOICE_TIMEOUT_SEC", 120)
    enable_speaker_recognition: bool = _get_bool("ENABLE_SPEAKER_RECOGNITION", False)
    speaker_repo_path: str | None = os.getenv("SPEAKER_REPO_PATH")
    speaker_model_dir: str | None = os.getenv("SPEAKER_MODEL_DIR")
    speaker_python: str = os.getenv("SPEAKER_PYTHON", "python3")
    speaker_command: str | None = os.getenv("SPEAKER_COMMAND")
    speaker_timeout_sec: int = _get_int("SPEAKER_TIMEOUT_SEC", 60)
    enable_wenet_asr: bool = _get_bool("ENABLE_WENET_ASR", False)
    enable_sensevoice_asr: bool = _get_bool("ENABLE_SENSEVOICE_ASR", False)
    enable_fireredasr_asr: bool = _get_bool("ENABLE_FIREREDASR_ASR", False)
    fireredasr_repo_path: str | None = os.getenv("FIREREDASR_REPO_PATH")
    fireredasr_model_dir: str | None = os.getenv("FIREREDASR_MODEL_DIR")
    fireredasr_beam_size: int = _get_int("FIREREDASR_BEAM_SIZE", 1)
    fireredasr_nbest: int = _get_int("FIREREDASR_NBEST", 1)
    fireredasr_decode_max_len: int = _get_int("FIREREDASR_DECODE_MAX_LEN", 128)
    fireredasr_max_audio_sec: int = _get_int("FIREREDASR_MAX_AUDIO_SEC", 10)
    sensevoice_repo_path: str | None = os.getenv("SENSEVOICE_REPO_PATH")
    sensevoice_python: str = os.getenv("SENSEVOICE_PYTHON", "python3")
    sensevoice_language: str = os.getenv("SENSEVOICE_LANGUAGE", "auto")
    sensevoice_timeout_sec: int = _get_int("SENSEVOICE_TIMEOUT_SEC", 60)
    sensevoice_streaming: bool = _get_bool("SENSEVOICE_STREAMING", False)
    wenet_repo_path: str | None = os.getenv("WENET_REPO_PATH")
    wenet_onnx_dir: str | None = os.getenv("WENET_ONNX_DIR")
    wenet_config_path: str | None = os.getenv("WENET_CONFIG_PATH")
    wenet_vocab_path: str | None = os.getenv("WENET_VOCAB_PATH")
    wenet_mode: str = os.getenv("WENET_MODE", "ctc_prefix_beam_search")
    wenet_online: bool = _get_bool("WENET_ONLINE", False)
    wenet_offline_seq_len: int = _get_int("WENET_OFFLINE_SEQ_LEN", 1024)
    wenet_decoder_len: int = _get_int("WENET_DECODER_LEN", 32)
    wenet_ort_providers: str = os.getenv("WENET_ORT_PROVIDERS", "CPUExecutionProvider")
    wenet_calib_data_path: str | None = os.getenv("WENET_CALIB_DATA_PATH")
    silero_vad_sampling_rate: int = _get_int("SILERO_VAD_SAMPLING_RATE", 16000)
    silero_vad_threshold: float = float(os.getenv("SILERO_VAD_THRESHOLD", "0.45"))
    silero_vad_min_speech_ms: int = _get_int("SILERO_VAD_MIN_SPEECH_MS", 350)
    silero_vad_min_silence_ms: int = _get_int("SILERO_VAD_MIN_SILENCE_MS", 700)
    silero_vad_speech_pad_ms: int = _get_int("SILERO_VAD_SPEECH_PAD_MS", 300)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
