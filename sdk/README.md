# Her.axera SDK

Unified **ASR + LLM + TTS** voice dialogue pipeline for AX650 boards.

## Install

```bash
# 1. Install AX engine wheels first
pip install https://github.com/AXERA-TECH/ax_asr_api/releases/download/v0.1.6/ax_asr-0.1.0-cp311-cp311-linux_aarch64.whl --break-system-packages
pip install https://github.com/AXERA-TECH/ax_tts_api/releases/download/v0.1.4/ax_tts-0.1.4-cp311-cp311-linux_aarch64.whl --break-system-packages

# 2. Install SDK
pip install sdk/ --break-system-packages
```

## Quick Start

```python
from her_axera_sdk import HerAxeraSDK

sdk = HerAxeraSDK(
    asr_model_path="/opt/models/asr/models-ax650",
    tts_model_path="/opt/models/tts/models-ax650",
    llm_api_base="https://api.deepseek.com",
    llm_api_key="sk-xxx",
)

# Auto-download models + use
with sdk:
    # ASR
    text = sdk.transcribe("demo.wav")
    print(text)

    # LLM
    reply = sdk.chat("今天天气怎么样？")
    print(reply)

    # TTS
    wav = sdk.synthesize("你好世界")
    with open("output.wav", "wb") as f:
        f.write(wav)
```

## Full Dialogue Pipeline

```python
with sdk:
    for event in sdk.dialogue("audio.wav", language="zh"):
        if event["type"] == "asr_text":
            print(f"[ASR] {event['text']}")
        elif event["type"] == "llm_token":
            print(event["token"], end="", flush=True)
        elif event["type"] == "tts_audio":
            with open(f"reply.wav", "wb") as f:
                f.write(event["data"])
```

## API

### `HerAxeraSDK(...)`

| Param | Default | Description |
|---|---|---|
| `asr_model_path` | `$AX_ASR_MODEL_PATH` | ASR model root dir |
| `tts_model_path` | `$AX_TTS_MODEL_PATH` | TTS model root dir |
| `llm_api_base` | `https://api.deepseek.com` | OpenAI-compatible API |
| `llm_api_key` | `$DEEPSEEK_API_KEY` | API key |
| `llm_model` | `deepseek-chat` | Model name |
| `asr_model_type` | `sensevoice` | ASR model type |
| `tts_type` | `KOKORO` | TTS engine |
| `tts_voice` | `af_heart` | Default voice |
| `auto_download` | `True` | Auto-download models |

### Methods

| Method | Returns | Description |
|---|---|---|
| `transcribe(audio, lang)` | `str` | ASR audio→text |
| `chat(msg)` | `str` | LLM chat |
| `chat_stream(msg)` | `Iterator[Chunk]` | Streaming LLM |
| `synthesize(text)` | `bytes` | TTS text→WAV |
| `dialogue(audio)` | `Iterator[dict]` | Full pipeline |
| `download_models()` | `dict` | Download missing models |
| `check_models()` | `dict` | Check model status |
| `close()` | — | Release resources |
