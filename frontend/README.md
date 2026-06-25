# Her Voice Frontend

基于 Gradio 的语音对话前端，通过 WebSocket 与后端实时通信，支持文字输入、麦克风录音和 Free Speak 自由对话模式。

## 功能

- 文字输入对话
- 麦克风录音输入
- **Free Speak 模式**：自动语音活动检测，检测到说话/静音自动触发对话
- ASR / LLM / TTS Provider 下拉选择
- LLM API Key 配置（用于 DeepSeek 等远程 Provider）
- 语言选择（zh-CN / en-US）
- 音色选择（alloy, echo, fable, onyx, nova, shimmer 等）
- 聊天历史展示
- TTS 语音回复自动播放
- 后端健康检查

## 对话流程

```text
┌──────────┐     WebSocket /v1/dialogue/ws     ┌─────────┐
│  Gradio  │ ◄───────────────────────────────► │ Backend │
│ Frontend │                                    └─────────┘
└──────────┘

文字输入 ──────────────────► LLM → TTS → 语音播放
语音输入 ──► ASR → LLM → TTS → 语音播放

Free Speak: 麦克风流式采集 → 自动 VAD 切分 → ASR → LLM → TTS → 语音播放
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

默认访问地址：`http://127.0.0.1:7860`

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_BASE_URL` | `http://127.0.0.1:8080` | 后端 API 地址 |
| `ASR_PROVIDER_CHOICES` | `mock_asr,wenet_onnx,sensevoice,fireredasr_aed` | ASR 下拉框可选 Provider，逗号分隔 |
| `DEFAULT_ASR_PROVIDER` | `mock_asr` | 默认 ASR Provider |
| `REQUEST_TIMEOUT` | `60` | 后端请求超时时间（秒） |
| `FRONTEND_HOST` | `0.0.0.0` | Gradio 服务监听地址 |
| `FRONTEND_PORT` | `7860` | Gradio 服务端口 |
| `FREE_SPEAK_SILENCE_MS` | `1000` | Free Speak 静音判定时长（毫秒） |
| `FREE_SPEAK_MIN_UTTERANCE_MS` | `600` | Free Speak 最短有效语音时长（毫秒） |
| `FREE_SPEAK_MAX_BUFFER_MS` | `30000` | Free Speak 最大录音缓冲时长（毫秒） |

## 说明

- 前端通过 WebSocket (`/v1/dialogue/ws`) 进行实时对话，支持流式音频传输和事件推送。
- Free Speak 模式基于 RMS 音量阈值进行端点检测，检测到有效语音段后自动触发对话流程。
- LLM / TTS Provider 和音色可在界面上实时切换，会话 ID 用于保持上下文历史。
- 使用 mock Provider 时可验证前端完整链路；接入真实 Provider 后，自动使用真实模型结果。
