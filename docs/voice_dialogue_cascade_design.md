# 语音对话系统级联架构设计文档

## 1. 背景与目标

本文档描述一个基于级联架构的语音对话系统设计方案。系统由 ASR、LLM、TTS 三类核心能力串联组成，通过 RESTful API 对外提供语音对话能力，并支持在 ASR 与 TTS 环节按业务需求选择不同模型。

### 1.1 建设目标

- 支持语音输入、语音输出的端到端对话体验。
- 采用 `ASR + LLM + TTS` 级联架构，模块职责清晰，便于替换与扩展。
- ASR 与 TTS 支持多模型、多供应商、多部署形态的统一接入。
- 通过 RESTful API 提供统一服务接口，便于 Web、App、嵌入式设备或第三方系统集成。
- 支持会话上下文管理、流式或非流式响应、错误恢复、日志与监控。

### 1.2 非目标

- 本设计不覆盖端侧唤醒词检测、回声消除、降噪等音频前处理算法的详细实现。
- 本设计不限定具体 LLM、ASR、TTS 模型厂商。
- 本设计不包含 UI 客户端交互细节。

## 2. 总体架构

### 2.1 架构概览

系统采用分层级联架构：客户端上传音频后，服务端依次完成语音识别、语义理解与回复生成、语音合成，最后返回文本与音频结果。

```text
Client
  |
  | RESTful API
  v
API Gateway / Dialogue API
  |
  +--> Session Manager
  +--> Model Router
  |
  v
ASR Service  --->  LLM Service  --->  TTS Service
  |                  |                 |
  v                  v                 v
ASR Models       LLM Models        TTS Models
  |
  v
Storage / Cache / Logs / Metrics
```

### 2.2 核心模块

| 模块 | 职责 |
| --- | --- |
| Client | 采集用户音频，调用 RESTful API，播放合成语音。 |
| API Gateway / Dialogue API | 对外统一入口，负责鉴权、参数校验、请求编排与结果返回。 |
| Session Manager | 管理会话 ID、历史上下文、用户画像和对话状态。 |
| Model Router | 根据请求参数、业务策略或配置选择 ASR/TTS/LLM 模型。 |
| ASR Service | 将输入音频转写为文本，屏蔽不同 ASR 模型差异。 |
| LLM Service | 基于用户文本与上下文生成回复文本。 |
| TTS Service | 将回复文本合成为语音，屏蔽不同 TTS 模型差异。 |
| Storage | 保存会话、音频文件、转写文本、合成结果与审计日志。 |
| Observability | 提供日志、指标、链路追踪、告警和质量分析。 |

## 3. 关键业务流程

### 3.1 非流式语音对话流程

适用于一次请求返回完整结果的场景，例如短语音问答、设备控制、客服 FAQ。

```text
1. Client 录制用户语音。
2. Client 调用 POST /v1/dialogue/audio。
3. Dialogue API 校验请求、鉴权并创建 trace_id。
4. Model Router 选择 ASR 模型。
5. ASR Service 将音频转写为 user_text。
6. Session Manager 读取并更新对话上下文。
7. LLM Service 生成 assistant_text。
8. Model Router 选择 TTS 模型、音色和采样率。
9. TTS Service 合成 assistant_audio。
10. Dialogue API 返回 ASR 文本、LLM 文本、TTS 音频 URL 或 Base64。
```

### 3.2 半流式语音对话流程

适用于希望降低首包延迟，但仍使用 RESTful API 的场景。

```text
1. Client 上传完整音频。
2. ASR 完成后立即返回识别文本或任务状态。
3. LLM 分段生成回复文本。
4. TTS 按句子或短文本片段合成音频。
5. Client 轮询任务接口获取阶段性结果。
```

由于 RESTful API 天然不适合低延迟双向流式音频，如果后续需要实时打断、边说边听、低延迟连续对话，可扩展 WebSocket、SSE 或 gRPC Streaming。

### 3.3 文本转语音流程

适用于已有文本回复、播报通知、测试 TTS 模型等场景。

```text
Client -> POST /v1/tts/synthesize -> TTS Service -> TTS Model -> Audio Result
```

### 3.4 语音识别流程

适用于单独测试 ASR 或其他业务复用语音转写能力。

```text
Client -> POST /v1/asr/transcribe -> ASR Service -> ASR Model -> Text Result
```

## 4. 模型接入设计

### 4.1 模型抽象

ASR、TTS 与 LLM 均通过统一 Provider 接口接入，业务层不直接依赖具体模型 SDK 或第三方 API。

```text
ASRProvider
  - transcribe(audio, options) -> ASRResult

LLMProvider
  - chat(messages, options) -> LLMResult

TTSProvider
  - synthesize(text, options) -> TTSResult
```

### 4.2 ASR 模型选择策略

ASR 支持按以下维度选择模型：

- **语言**：中文、英文、多语种、方言。
- **部署方式**：本地模型、私有化服务、第三方云服务。
- **性能目标**：低延迟、低成本、高准确率、离线可用。
- **音频格式**：`wav`、`pcm`、`mp3`、`flac`、`opus`。
- **场景类型**：近场语音、远场语音、电话音频、会议音频。

推荐配置示例：

```yaml
asr:
  default_provider: local_whisper
  providers:
    local_whisper:
      type: local
      model: whisper-small
      languages: [zh, en]
      sample_rates: [16000]
    cloud_asr:
      type: remote
      endpoint: https://asr.example.com/v1/transcribe
      timeout_ms: 15000
```

### 4.3 TTS 模型选择策略

TTS 支持按以下维度选择模型：

- **音色**：默认女声、男声、儿童声、角色音色、自定义音色。
- **语言**：中文、英文、多语种。
- **风格**：自然、客服、播报、情感化、低延迟。
- **部署方式**：本地模型、私有化服务、第三方云服务。
- **输出格式**：`wav`、`mp3`、`pcm`、`opus`。

推荐配置示例：

```yaml
tts:
  default_provider: local_tts
  providers:
    local_tts:
      type: local
      model: cosyvoice
      voices: [female_default, male_default]
      sample_rates: [16000, 24000]
    cloud_tts:
      type: remote
      endpoint: https://tts.example.com/v1/synthesize
      timeout_ms: 20000
```

### 4.4 Model Router 路由规则

模型路由建议按优先级执行：

1. 请求参数显式指定模型，例如 `asr_provider` 或 `tts_provider`。
2. 用户、租户或设备维度的固定配置。
3. 业务场景策略，例如客服、教育、陪伴、设备控制。
4. 系统默认模型。
5. 故障降级模型。

路由结果需要写入日志，便于问题排查与效果分析。

## 5. RESTful API 设计

### 5.1 通用约定

- Base URL：`/v1`
- 请求格式：`application/json` 或 `multipart/form-data`
- 音频返回方式：优先返回 `audio_url`，也可按需返回 `audio_base64`
- 编码：统一使用 UTF-8
- 请求追踪：客户端可传入 `X-Request-Id`，服务端返回 `trace_id`
- 鉴权：推荐使用 `Authorization: Bearer <token>`

### 5.2 语音对话接口

```http
POST /v1/dialogue/audio
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

请求参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| audio | file | 是 | 用户语音文件。 |
| session_id | string | 否 | 会话 ID，不传则创建新会话。 |
| user_id | string | 否 | 用户 ID。 |
| language | string | 否 | 语言，例如 `zh-CN`、`en-US`。 |
| asr_provider | string | 否 | 指定 ASR 模型供应商。 |
| tts_provider | string | 否 | 指定 TTS 模型供应商。 |
| voice | string | 否 | TTS 音色。 |
| output_audio_format | string | 否 | 输出音频格式，例如 `wav`、`mp3`。 |
| return_audio_base64 | boolean | 否 | 是否直接返回 Base64 音频。 |

响应示例：

```json
{
  "trace_id": "trc_202605210001",
  "session_id": "ses_abc123",
  "asr": {
    "provider": "local_whisper",
    "text": "今天天气怎么样？",
    "confidence": 0.94,
    "duration_ms": 820
  },
  "llm": {
    "provider": "default_llm",
    "text": "我可以帮你查询天气。请告诉我你所在的城市。",
    "duration_ms": 1250
  },
  "tts": {
    "provider": "local_tts",
    "voice": "female_default",
    "audio_url": "https://cdn.example.com/audio/resp_abc123.wav",
    "audio_format": "wav",
    "duration_ms": 960
  }
}
```

### 5.3 ASR 独立接口

```http
POST /v1/asr/transcribe
Content-Type: multipart/form-data
```

请求参数：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| audio | file | 是 | 待识别音频。 |
| language | string | 否 | 语言。 |
| provider | string | 否 | 指定 ASR Provider。 |
| enable_punctuation | boolean | 否 | 是否自动加标点。 |

响应示例：

```json
{
  "trace_id": "trc_asr_001",
  "provider": "local_whisper",
  "text": "打开客厅的灯。",
  "confidence": 0.96,
  "segments": [
    {
      "start_ms": 0,
      "end_ms": 1800,
      "text": "打开客厅的灯。"
    }
  ]
}
```

### 5.4 TTS 独立接口

```http
POST /v1/tts/synthesize
Content-Type: application/json
```

请求示例：

```json
{
  "text": "你好，我是你的语音助手。",
  "provider": "local_tts",
  "voice": "female_default",
  "audio_format": "wav",
  "sample_rate": 24000,
  "return_audio_base64": false
}
```

响应示例：

```json
{
  "trace_id": "trc_tts_001",
  "provider": "local_tts",
  "voice": "female_default",
  "audio_url": "https://cdn.example.com/audio/tts_001.wav",
  "audio_format": "wav",
  "sample_rate": 24000,
  "duration_ms": 710
}
```

### 5.5 会话接口

```http
POST /v1/sessions
GET /v1/sessions/{session_id}
DELETE /v1/sessions/{session_id}
```

会话数据建议包含：

- `session_id`：会话唯一标识。
- `user_id`：用户标识。
- `messages`：历史消息列表。
- `model_preferences`：模型偏好配置。
- `created_at`、`updated_at`：创建与更新时间。
- `metadata`：设备、地区、业务场景等扩展信息。

### 5.6 异步任务接口

长音频、慢速模型或批处理场景可采用异步接口。

```http
POST /v1/dialogue/audio-jobs
GET /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
```

任务状态：

| 状态 | 说明 |
| --- | --- |
| queued | 已进入队列。 |
| running | 正在处理。 |
| succeeded | 处理成功。 |
| failed | 处理失败。 |
| cancelled | 已取消。 |

## 6. 数据结构设计

### 6.1 Message

```json
{
  "role": "user",
  "content": "帮我打开空调",
  "content_type": "text",
  "timestamp": "2026-05-21T10:00:00+08:00",
  "metadata": {
    "asr_provider": "local_whisper",
    "audio_url": "https://cdn.example.com/audio/input.wav"
  }
}
```

### 6.2 DialogueResult

```json
{
  "trace_id": "trc_001",
  "session_id": "ses_001",
  "user_text": "帮我打开空调",
  "assistant_text": "好的，已为你打开空调。",
  "assistant_audio_url": "https://cdn.example.com/audio/output.wav",
  "latency": {
    "asr_ms": 600,
    "llm_ms": 900,
    "tts_ms": 700,
    "total_ms": 2200
  }
}
```

## 7. 错误处理与降级

### 7.1 错误码

| HTTP 状态码 | 错误码 | 说明 |
| --- | --- | --- |
| 400 | invalid_request | 请求参数错误。 |
| 401 | unauthorized | 鉴权失败。 |
| 413 | audio_too_large | 音频文件过大。 |
| 415 | unsupported_audio_format | 不支持的音频格式。 |
| 422 | asr_no_speech | 未检测到有效语音。 |
| 429 | rate_limited | 请求频率超限。 |
| 500 | internal_error | 系统内部错误。 |
| 502 | provider_error | 模型服务调用失败。 |
| 504 | provider_timeout | 模型服务调用超时。 |

错误响应示例：

```json
{
  "trace_id": "trc_error_001",
  "error": {
    "code": "provider_timeout",
    "message": "TTS provider request timed out",
    "stage": "tts",
    "retryable": true
  }
}
```

### 7.2 降级策略

- ASR 失败：切换备用 ASR Provider；若仍失败，返回可重试错误。
- LLM 失败：返回兜底话术，或切换备用模型。
- TTS 失败：返回文本结果，并标记语音合成失败。
- 存储失败：允许短期返回 Base64 音频，异步补偿上传。
- 高负载：限制高耗时模型，切换低延迟模型或进入排队模式。

## 8. 性能与容量设计

### 8.1 核心指标

| 指标 | 建议目标 |
| --- | --- |
| ASR 延迟 | 短音频 P95 小于 1500 ms。 |
| LLM 首 token 延迟 | P95 小于 2000 ms。 |
| TTS 首包延迟 | P95 小于 1200 ms。 |
| 端到端延迟 | 短语音 P95 小于 5000 ms。 |
| 可用性 | 月可用性不低于 99.5%。 |

### 8.2 优化手段

- 音频上传限制时长与大小，避免长音频阻塞在线请求。
- 对 ASR、LLM、TTS 分别设置超时、重试和熔断策略。
- LLM 输出按句切分后尽早触发 TTS。
- 对常见固定回复进行 TTS 音频缓存。
- 对本地模型使用批处理、模型常驻和资源池化。
- 使用异步任务处理长音频和低优先级请求。

## 9. 安全设计

- **鉴权**：所有接口默认启用 Token 鉴权。
- **限流**：按用户、租户、设备、IP 设置 QPS 与并发限制。
- **数据保护**：音频与文本日志脱敏，敏感字段加密存储。
- **权限隔离**：不同租户的会话、音频、模型配置相互隔离。
- **审计**：记录调用方、模型版本、请求参数、耗时与错误信息。
- **内容安全**：LLM 输入输出可接入敏感词、合规审核或安全分类器。

## 10. 日志、监控与可观测性

### 10.1 日志字段

- `trace_id`
- `session_id`
- `user_id`
- `stage`：`asr`、`llm`、`tts`、`dialogue`
- `provider`
- `model`
- `latency_ms`
- `status`
- `error_code`

### 10.2 监控指标

- 请求总量、成功率、错误率。
- 各阶段 P50、P90、P95、P99 延迟。
- ASR 空语音率、识别失败率、平均置信度。
- LLM token 数、超时率、兜底回复率。
- TTS 合成失败率、音频时长、缓存命中率。
- Provider 维度可用性和成本统计。

## 11. 部署设计

### 11.1 推荐服务拆分

```text
dialogue-api
asr-service
llm-service
tts-service
session-service
storage-service
observability-stack
```

### 11.2 部署方式

- **单机部署**：适合开发、验证、小规模私有化场景。
- **容器化部署**：适合生产环境，服务独立扩缩容。
- **混合部署**：本地模型与云端模型结合，兼顾成本、性能和可用性。
- **边缘部署**：在端侧或边缘设备运行轻量 ASR/TTS，云端运行 LLM。

### 11.3 资源建议

| 服务 | 资源关注点 |
| --- | --- |
| ASR Service | CPU/GPU/NPU、音频解码能力、并发转写能力。 |
| LLM Service | GPU/NPU 显存、上下文长度、token 吞吐。 |
| TTS Service | GPU/NPU/CPU 推理速度、音色模型加载时间。 |
| Storage | 音频对象存储容量、生命周期管理。 |
| API Gateway | 并发连接数、上传带宽、限流能力。 |

## 12. 配置示例

```yaml
server:
  host: 0.0.0.0
  port: 8080
  request_timeout_ms: 30000
  max_audio_size_mb: 20
  max_audio_duration_sec: 60

dialogue:
  default_language: zh-CN
  max_history_messages: 20
  fallback_text: 抱歉，我现在暂时无法处理这个请求，请稍后再试。

model_router:
  default_asr_provider: local_whisper
  default_llm_provider: default_llm
  default_tts_provider: local_tts
  fallback_asr_provider: cloud_asr
  fallback_tts_provider: cloud_tts

storage:
  audio_bucket: voice-dialogue-audio
  audio_retention_days: 7
  return_audio_base64_max_size_mb: 2

observability:
  enable_trace: true
  enable_metrics: true
  log_level: info
```

## 13. 测试方案

### 13.1 功能测试

- 上传不同格式音频并验证 ASR 识别结果。
- 指定不同 ASR/TTS Provider 并验证路由生效。
- 验证新会话、连续会话、删除会话流程。
- 验证 TTS 音色、采样率、音频格式参数。
- 验证异常音频、空音频、超大音频处理。

### 13.2 性能测试

- 短语音端到端延迟测试。
- ASR、LLM、TTS 单模块压测。
- 多 Provider 并发切换测试。
- 长时间稳定性测试。
- 音频上传带宽与对象存储压力测试。

### 13.3 质量测试

- ASR 字错率评估。
- LLM 回复相关性、稳定性、安全性评估。
- TTS 自然度、可懂度、音色一致性评估。
- 端到端用户体验主观评分。

## 14. 后续演进方向

- 支持 WebSocket 或 SSE，实现更低延迟的实时对话。
- 支持 VAD，减少无效音频上传和 ASR 处理成本。
- 支持语音打断，让用户在 TTS 播放过程中继续发起新请求。
- 支持多模态输入，例如图片、视频帧与语音联合理解。
- 支持 Agent 工具调用，实现设备控制、知识库问答和业务系统操作。
- 支持模型效果 A/B 测试与灰度发布。

## 15. 总结

本设计通过 `ASR + LLM + TTS` 级联架构实现语音对话能力，使用 RESTful API 作为统一通信方式，并通过 Provider 抽象与 Model Router 支持多模型灵活切换。该方案结构清晰、易于集成、便于扩展，适合从原型验证逐步演进到生产级部署。
