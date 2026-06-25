# Her Voice Backend

基于 FastAPI 的语音对话系统后端，提供 OpenAI 兼容的 RESTful API 和 WebSocket 实时对话接口，支持 ASR、LLM、TTS、Speaker 多模块 Provider 可插拔架构。

## 目录结构

```text
backend/
  app/
    api/routes/
      asr.py              # ASR 原生路由
      llm.py              # LLM 原生路由
      tts.py              # TTS 路由（含分段合成与异步任务）
      openai_compat.py    # OpenAI 兼容路由
      ws_dialogue.py      # WebSocket 实时对话
      speakers.py         # 说话人识别路由
      health.py           # 健康检查
    core/
      config.py           # 环境变量配置
      errors.py           # 错误定义
      trace.py            # 追踪工具
    models/
      openai.py           # OpenAI 兼容请求/响应模型
      speaker.py          # 说话人识别模型
      ...                 # ASR、LLM、TTS 模型
    services/
      asr_service.py      # ASR 服务（mock, wenet, sensevoice, fireredasr）
      llm_service.py      # LLM 服务（mock, deepseek）
      tts_service.py      # TTS 服务（mock, edge_tts, kokoro, zipvoice）
      speaker_service.py  # 说话人识别服务（mock, 3d_speaker）
      dialogue_service.py # 级联对话管线
    main.py               # FastAPI 应用入口
  .env.example
  requirements.txt
  pyproject.toml
```

## 本地启动

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 按需配置 Provider
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

启动后访问：

- 健康检查：`GET http://localhost:8080/health`
- OpenAPI 文档：`http://localhost:8080/docs`

## API 概览

### OpenAI 兼容接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/chat/completions` | 聊天补全（支持 DeepSeek 等） |
| POST | `/v1/audio/transcriptions` | 语音转文字 |
| POST | `/v1/audio/speech` | 文字转语音 |

### WebSocket 实时对话

| 路径 | 说明 |
|------|------|
| `/v1/dialogue/ws` | 流式语音对话，支持 ASR → LLM → TTS 级联 |

WebSocket 消息类型：
- `speech_start` / `audio_chunk` / `speech_end` — 流式录音
- `audio` / `utterance` — 完整音频
- `text` — 文字输入（跳过 ASR）
- `abort` — 中断当前轮次

事件流：`asr_started` → `asr` → `llm_started` → `llm` → `tts_sentence`(多条) → `done`

### 原生接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/asr/transcriptions` | ASR 转写 |
| POST | `/v1/llm/chat/completions` | LLM 对话 |
| POST | `/v1/tts/speech` | TTS 合成 |
| POST | `/v1/tts/speech/segments` | TTS 分段合成 |
| GET  | `/v1/asr/providers` | ASR Provider 列表 |
| GET  | `/v1/llm/providers` | LLM Provider 列表 |
| GET  | `/v1/tts/providers` | TTS Provider 列表 |
| GET  | `/v1/tts/voices` | 可用音色列表 |
| POST | `/v1/speakers/identify` | 说话人识别 |
| GET  | `/v1/speakers/providers` | Speaker Provider 列表 |

## 示例请求

### ASR（OpenAI 兼容）

```bash
curl -X POST "http://localhost:8080/v1/audio/transcriptions" \
  -F "file=@input.wav" \
  -F "model=whisper-1" \
  -F "language=zh-CN" \
  -F "response_format=verbose_json"
```

### LLM（OpenAI 兼容）

```bash
curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [
      {"role": "user", "content": "帮我打开客厅的灯。"}
    ]
  }'
```

### TTS（OpenAI 兼容）

```bash
curl -X POST "http://localhost:8080/v1/audio/speech" \
  -H "Content-Type: application/json" \
  --output reply.mp3 \
  -d '{
    "model": "tts-1",
    "input": "你好，我是你的语音助手。",
    "voice": "alloy",
    "response_format": "mp3"
  }'
```

### 说话人识别

```bash
curl -X POST "http://localhost:8080/v1/speakers/identify" \
  -F "audio=@speaker.wav" \
  -F "top_k=3"
```

## Provider 说明

### ASR Provider

| Provider | 类型 | 说明 |
|----------|------|------|
| mock_asr | Mock | 内置占位，返回模拟文本 |
| wenet_onnx | 本地 | WeNet ONNX 模型，见 [部署指南](../docs/wenet_onnx_asr_deploy.md) |
| sensevoice | 本地 | SenseVoice 多语言 ASR，见 [接入指南](../docs/sensevoice_asr_provider.md) |
| fireredasr_aed | 本地 | FireRedASR-AED（AX650N），见 [接入指南](../docs/fireredasr_aed_asr_provider.md) |

### LLM Provider

| Provider | 类型 | 说明 |
|----------|------|------|
| mock_llm | Mock | 内置占位，回显用户输入 |
| deepseek | 远程 | DeepSeek API（兼容 OpenAI 格式），需配置 `DEEPSEEK_API_KEY` |

### TTS Provider

| Provider | 类型 | 说明 |
|----------|------|------|
| mock_tts | Mock | 内置占位，生成合成 WAV |
| edge_tts | 远程 | Microsoft Edge TTS，支持多种中英文音色 |
| kokoro | 本地 | Kokoro AXEngine TTS |
| zipvoice | 本地 | ZipVoice AXEngine TTS，支持声音克隆 |

### Speaker Provider

| Provider | 类型 | 说明 |
|----------|------|------|
| mock_speaker | Mock | 内置占位 |
| 3d_speaker | 本地 | 3D-Speaker AXEngine 说话人识别 |

## 环境变量

详见 [.env.example](.env.example)，主要配置项：

- `DEFAULT_ASR_PROVIDER` / `DEFAULT_LLM_PROVIDER` / `DEFAULT_TTS_PROVIDER` — 默认 Provider
- `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` — DeepSeek LLM 配置
- `EDGE_TTS_VOICE` — Edge TTS 默认音色
- `ENABLE_*` — 各 Provider 启用开关
- `SILERO_VAD_*` — VAD 参数

## 前端 Demo

Gradio 前端位于 `../frontend`，支持文字和语音实时对话。详见 [../frontend/README.md](../frontend/README.md)。
