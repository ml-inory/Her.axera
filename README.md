# Her.axera — 语音对话系统

基于 AX650 / AX630C 平台的端侧语音对话系统，采用 ASR + LLM + TTS 级联架构，提供 OpenAI 兼容 API 和 WebSocket 实时对话能力。

## 系统架构

```text
┌──────────┐    WebSocket / REST     ┌─────────────────────────────┐
│  Gradio  │ ◄─────────────────────► │       FastAPI Backend       │
│ Frontend │                         │                             │
└──────────┘                         │  ┌─────┐  ┌─────┐  ┌─────┐ │
                                     │  │ ASR │→ │ LLM │→ │ TTS │ │
                                     │  └─────┘  └─────┘  └─────┘ │
                                     │      ┌───────────┐         │
                                     │      │  Speaker   │         │
                                     │      └───────────┘         │
                                     └─────────────────────────────┘
```

## 特性

- **OpenAI 兼容 API**：`/v1/chat/completions`、`/v1/audio/transcriptions`、`/v1/audio/speech`
- **WebSocket 实时对话**：`/v1/dialogue/ws`，支持流式 ASR → LLM → TTS 级联
- **多 Provider 可插拔**：每个模块均支持 mock + 多种真实 Provider

| 模块 | 可用 Provider |
|------|--------------|
| ASR  | mock_asr, wenet_onnx, sensevoice, fireredasr_aed |
| LLM  | mock_llm, deepseek |
| TTS  | mock_tts, edge_tts, kokoro, zipvoice |
| Speaker | mock_speaker, 3d_speaker |

- **Gradio 前端**：文字输入、麦克风录音、Free Speak 自由对话模式
- **Silero VAD**：语音活动检测，自动切分有效语音段

## 快速开始

### 1. 启动后端

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 编辑 .env 配置 Provider
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 2. 启动前端

```bash
cd frontend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
API_BASE_URL=http://127.0.0.1:8080 python app.py
```

访问 http://127.0.0.1:7860 即可使用。

### 3. 快速测试

```bash
# ASR
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@input.wav" -F "model=mock_asr"

# LLM
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mock_llm","messages":[{"role":"user","content":"你好"}]}'

# TTS
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" --output reply.mp3 \
  -d '{"model":"tts-1","input":"你好","voice":"alloy"}'
```

## 文档

| 文档 | 说明 |
|------|------|
| [backend/README.md](backend/README.md) | 后端详细说明 |
| [frontend/README.md](frontend/README.md) | 前端详细说明 |
| [级联对话设计](docs/voice_dialogue_cascade_design.md) | 系统架构设计 |
| [OpenAI 兼容 API](docs/openai_compatible_api_design.md) | OpenAI 兼容接口规范 |
| [ASR API 设计](docs/asr_api_design.md) | ASR 接口设计 |
| [LLM API 设计](docs/llm_api_design.md) | LLM 接口设计 |
| [TTS API 设计](docs/tts_api_design.md) | TTS 接口设计 |
| [WeNet ONNX 部署](docs/wenet_onnx_asr_deploy.md) | WeNet ASR 部署指南 |
| [SenseVoice ASR](docs/sensevoice_asr_provider.md) | SenseVoice ASR 接入指南 |
| [FireRedASR-AED](docs/fireredasr_aed_asr_provider.md) | FireRedASR-AED 接入指南 |

## 支持平台

- AX650
- AX630C

## 讨论

- Github issues
- QQ 群: 139953715

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
