# Her Voice Frontend

这是一个基于 Gradio 的基础语音对话前端，交互形式类似 chatbot，支持：

- 文字输入
- 麦克风录音输入
- 音频文件上传输入
- 聊天历史展示
- TTS 语音回复播放
- 后端健康检查

前端通过 RESTful API 调用后端：

```text
语音输入 -> /v1/asr/transcriptions -> /v1/llm/chat/completions -> /v1/tts/speech -> 语音播放
文字输入 ---------------------------> /v1/llm/chat/completions -> /v1/tts/speech -> 语音播放
```

## 启动后端

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## 启动前端

另开一个终端：

```bash
cd frontend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
API_BASE_URL=http://127.0.0.1:8080 python app.py
```

默认访问地址：

```text
http://127.0.0.1:7860
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| API_BASE_URL | `http://127.0.0.1:8080` | 后端 API 地址。 |
| REQUEST_TIMEOUT | `60` | 后端请求超时时间，单位秒。 |
| FRONTEND_HOST | `0.0.0.0` | Gradio 服务监听地址。 |
| FRONTEND_PORT | `7860` | Gradio 服务端口。 |

## 说明

当前后端仍使用 mock ASR、LLM、TTS Provider。mock TTS 在 `wav` 格式下会返回一段可播放的模拟音频，用于验证前端链路；接入真实 TTS 后，前端会直接播放真实合成结果。
