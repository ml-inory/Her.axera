# Her.axera — 语音对话系统

基于 AX650 / AX630C 平台的端侧语音对话系统，采用 ASR + LLM + TTS 级联架构，提供 OpenAI 兼容 API 和 WebSocket 实时对话能力。

## 系统架构

```text
┌──────────┐    WebSocket / REST     ┌─────────────────────────────┐
│ Web UI   │ ◄─────────────────────► │       FastAPI Backend       │
│ /ui/     │                         │                             │
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

- **Web 控制台**：文字输入、麦克风流式录音、链路耗时指标、TTS 自动播放
- **Silero VAD**：语音活动检测，自动切分有效语音段

## 推荐部署：AX650 后端 + PC 前端

第一版功能体验复刻以“后端在 AX650、前端在 PC 浏览器”为主路径。AX650 负责 ASR/TTS/Speaker 等端侧能力，PC 只运行静态 Web 控制台。

### 1. 在 AX650 上准备后端

```bash
cd /mnt/her-axera   # 或你的 Her.axera 仓库路径
scripts/ax650_setup_backend.sh
```

按需下载模型，默认走 `https://hf-mirror.com`：

```bash
scripts/ax650_setup_backend.sh --models "sensevoice kokoro speaker"
```

脚本不会覆盖已有 `backend/.env`。模型下载后会生成 `backend/.env.models`，请确认后再把需要的配置合并进 `backend/.env`。

### 2. 在 AX650 上启动后端

```bash
scripts/ax650_run_backend.sh --host 0.0.0.0 --port 8080
```

健康检查：

```bash
curl http://127.0.0.1:8080/health
```

需要开机自启动时安装 systemd 服务：

```bash
scripts/ax650_install_service.sh --enable --start
```

### 3. 在 PC 上启动前端

```bash
cd /path/to/Her.axera
scripts/pc_run_frontend.sh --backend-url http://<AX650_IP>:8080
```

浏览器打开脚本输出的 URL。前端也支持手工访问：

```text
http://127.0.0.1:7860/?api=http%3A%2F%2F<AX650_IP>%3A8080
```

详细板端部署流程见 [AX 板运行文档](docs/ax_board_nfs_run.md)。

## 本地开发快速开始

### 1. 启动后端

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 编辑 .env 配置 Provider
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 2. 打开前端

```bash
open http://127.0.0.1:8080/ui/
```

访问 http://127.0.0.1:8080/ui/ 即可使用。前端也可用 `python3 -m http.server 7860 --directory frontend/static` 独立启动。

本地 Docker 主要用于开发机验证：

```bash
docker-compose up -d
```

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
| [AX 板 NFS 运行](docs/ax_board_nfs_run.md) | 通过 NFS mount 在 AX 板上验证 |

## 支持平台

- AX650
- AX630C

## 讨论

- Github issues
- QQ 群: 139953715

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
