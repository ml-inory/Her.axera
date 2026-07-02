# Her Voice Web Console

轻量静态 Web 控制台，通过 WebSocket 与后端实时通信，支持文字输入、麦克风流式录音、链路指标和 TTS 自动播放。

## 功能

- 文字输入对话
- 麦克风 PCM 分片流式输入
- ASR / LLM / TTS Provider 下拉选择
- LLM API Key 配置
- 音色选择
- 聊天历史展示
- TTS 语音回复自动播放
- ASR / LLM / TTS / 总耗时指标展示

## 对话流程

```text
┌────────────┐     WebSocket /v1/dialogue/ws     ┌─────────┐
│ Static SPA │ ◄───────────────────────────────► │ Backend │
└────────────┘                                    └─────────┘

文字输入 ──────────────────► LLM → TTS → 语音播放
麦克风 ──► speech_start → PCM audio_chunk → speech_end → ASR → LLM → TTS
```

## 启动后端

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## 访问前端

后端会直接托管静态控制台：

```text
http://127.0.0.1:8080/ui/
```

也可以独立启动静态文件服务：

```bash
cd frontend
python3 -m http.server 7860 --directory static --bind 0.0.0.0
```

独立运行在 `7860` 端口时，页面默认连接同主机 `8080` 端口；右侧 API 输入框可覆盖地址，并保存到浏览器 `localStorage`。

## 说明

- 前端通过 WebSocket (`/v1/dialogue/ws`) 进行实时对话，支持流式音频传输和事件推送。
- LLM / TTS Provider 和音色可在界面上实时切换，会话 ID 用于保持上下文历史。
- 使用 mock Provider 时可验证前端完整链路；接入真实 Provider 后，自动使用真实模型结果。
