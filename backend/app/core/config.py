from dataclasses import dataclass
from functools import lru_cache
import os


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Her Voice Dialogue API")
    api_prefix: str = os.getenv("API_PREFIX", "/v1")
    log_level: str = os.getenv("LOG_LEVEL", "info")
    default_asr_provider: str = os.getenv("DEFAULT_ASR_PROVIDER", "mock_asr")
    default_llm_provider: str = os.getenv("DEFAULT_LLM_PROVIDER", "mock_llm")
    default_tts_provider: str = os.getenv("DEFAULT_TTS_PROVIDER", "mock_tts")
    max_audio_size_mb: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "20"))
    max_tts_text_length: int = int(os.getenv("MAX_TTS_TEXT_LENGTH", "1000"))
    enable_request_auth: bool = _get_bool("ENABLE_REQUEST_AUTH", False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
