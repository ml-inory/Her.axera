# Her Voice Backend

这是语音对话系统的 Python 后端骨架，基于 FastAPI 实现，当前提供 ASR、LLM、TTS 三类 RESTful API，并使用 mock provider 作为可替换的服务层占位实现。

## 目录结构

```text
backend/
  app/
    api/routes/     # ASR、LLM、TTS、health 路由
    core/           # 配置、错误、trace 工具
    models/         # Pydantic 请求与响应模型
    services/       # mock 服务与后续 Provider 接入点
    main.py         # FastAPI 应用入口
  requirements.txt
  pyproject.toml
```

## 本地启动

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

启动后访问：

- 健康检查：`GET http://localhost:8080/health`
- OpenAPI 文档：`http://localhost:8080/docs`

## 示例请求

### ASR

```bash
curl -X POST "http://localhost:8080/v1/asr/transcriptions" \
  -F "audio=@input.wav" \
  -F "language=zh-CN" \
  -F "enable_timestamps=true"
```

### LLM

```bash
curl -X POST "http://localhost:8080/v1/llm/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "ses_demo",
    "messages": [
      {"role": "user", "content": "帮我打开客厅的灯。"}
    ]
  }'
```

### TTS

```bash
curl -X POST "http://localhost:8080/v1/tts/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，我是你的语音助手。",
    "voice": "female_default",
    "audio_format": "wav"
  }'
```

## 后续接入点

- 在 `app/services/asr_service.py` 中替换 `mock_asr` 为真实 ASR Provider。
- 在 `app/services/llm_service.py` 中替换 `mock_llm` 为真实 LLM Provider。
- 在 `app/services/tts_service.py` 中替换 `mock_tts` 为真实 TTS Provider。
- 如需鉴权，可在 `app/api/deps.py` 中增加 token 校验依赖。

## 前端 Demo

Gradio 前端位于 `../frontend`，启动后可通过网页进行文字和语音对话测试。详见 `../frontend/README.md`。
