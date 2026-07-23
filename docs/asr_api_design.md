# ASR API 设计文档

## 1. 目标

> **📋 实现状态**：本文档描述的是**已完成实现**的设计。当前实现使用 `ax_asr`/`ax_tts`/`ax_llm` 等 NPU-native Provider。详见 [README](../README.md) 中的 Provider 表。

ASR API 用于将用户上传的音频转换为文本，为语音对话系统中的 `ASR + LLM + TTS` 级联链路提供语音识别能力。接口采用 RESTful API 设计，支持同步识别、异步识别、多模型选择、识别分段、置信度返回和错误降级。

## 2. 设计原则

- **统一入口**：屏蔽不同 ASR 模型或供应商的调用差异。
- **模型可选**：调用方可显式指定 Provider，也可使用系统默认路由策略。
- **格式兼容**：支持常见音频格式和采样率。
- **可观测**：每次请求返回 `trace_id`，便于日志追踪和问题排查。
- **可扩展**：预留热词、说话人分离、时间戳、语言识别等扩展字段。

## 3. 通用约定

### 3.1 Base URL

```text
/v1/asr
```

### 3.2 认证方式

```http
Authorization: Bearer <token>
```

### 3.3 请求追踪

客户端可通过请求头传入 `X-Request-Id`，服务端需要在响应中返回 `trace_id`。

```http
X-Request-Id: req_abc123
```

### 3.4 支持的音频格式

| 格式 | MIME Type | 说明 |
| --- | --- | --- |
| wav | audio/wav | 推荐格式，便于本地模型处理。 |
| pcm | application/octet-stream | 需额外传入采样率、声道数和位深。 |
| mp3 | audio/mpeg | 适合客户端压缩上传。 |
| flac | audio/flac | 适合高质量压缩音频。 |
| opus | audio/ogg | 适合低带宽场景。 |

## 4. 同步语音识别接口

### 4.1 接口说明

适用于短音频识别。客户端上传完整音频，服务端同步返回识别文本。

```http
POST /v1/asr/transcriptions
Content-Type: multipart/form-data
```

### 4.2 请求参数

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| audio | file | 是 | 待识别音频文件。 |
| provider | string | 否 | 指定 ASR Provider，例如 `local_whisper`、`cloud_asr`。 |
| model | string | 否 | 指定具体模型名称。 |
| language | string | 否 | 语言代码，例如 `zh-CN`、`en-US`，不传则自动识别或使用默认值。 |
| audio_format | string | 否 | 音频格式，例如 `wav`、`mp3`、`pcm`。 |
| sample_rate | integer | 否 | 采样率，PCM 音频必填。 |
| channels | integer | 否 | 声道数，PCM 音频必填。 |
| bit_depth | integer | 否 | 位深，PCM 音频必填。 |
| enable_punctuation | boolean | 否 | 是否自动添加标点，默认 `true`。 |
| enable_timestamps | boolean | 否 | 是否返回分段时间戳，默认 `false`。 |
| enable_word_timestamps | boolean | 否 | 是否返回词级时间戳，默认 `false`。 |
| enable_speaker_diarization | boolean | 否 | 是否启用说话人分离，默认 `false`。 |
| hotwords | array[string] | 否 | 热词列表，用于提升专有名词识别效果。 |
| metadata | object | 否 | 业务扩展信息。 |

### 4.3 请求示例

```bash
curl -X POST "https://api.example.com/v1/asr/transcriptions" \
  -H "Authorization: Bearer <token>" \
  -F "audio=@input.wav" \
  -F "language=zh-CN" \
  -F "provider=local_whisper" \
  -F "enable_punctuation=true" \
  -F "enable_timestamps=true"
```

### 4.4 响应字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| trace_id | string | 请求追踪 ID。 |
| provider | string | 实际使用的 ASR Provider。 |
| model | string | 实际使用的模型。 |
| language | string | 识别语言。 |
| text | string | 完整识别文本。 |
| confidence | number | 整体置信度，范围 `0-1`。 |
| duration_ms | integer | 输入音频时长。 |
| processing_ms | integer | 服务端处理耗时。 |
| segments | array | 分段识别结果。 |
| words | array | 词级识别结果，仅在启用词级时间戳时返回。 |

### 4.5 响应示例

```json
{
  "trace_id": "trc_asr_001",
  "provider": "local_whisper",
  "model": "whisper-small",
  "language": "zh-CN",
  "text": "帮我打开客厅的灯。",
  "confidence": 0.96,
  "duration_ms": 2100,
  "processing_ms": 780,
  "segments": [
    {
      "index": 0,
      "start_ms": 0,
      "end_ms": 2100,
      "text": "帮我打开客厅的灯。",
      "confidence": 0.96,
      "speaker": "spk_0"
    }
  ]
}
```

## 5. 异步语音识别接口

### 5.1 创建识别任务

适用于长音频、批处理或模型处理耗时较长的场景。

```http
POST /v1/asr/jobs
Content-Type: multipart/form-data
```

请求参数与同步接口一致，可额外传入：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| callback_url | string | 否 | 任务完成后的回调地址。 |
| priority | string | 否 | 任务优先级：`low`、`normal`、`high`。 |

响应示例：

```json
{
  "trace_id": "trc_asr_job_001",
  "job_id": "job_asr_001",
  "status": "queued",
  "created_at": "2026-05-21T10:00:00+08:00"
}
```

### 5.2 查询识别任务

```http
GET /v1/asr/jobs/{job_id}
```

响应示例：

```json
{
  "trace_id": "trc_asr_job_002",
  "job_id": "job_asr_001",
  "status": "succeeded",
  "result": {
    "provider": "cloud_asr",
    "model": "general-large",
    "language": "zh-CN",
    "text": "这是一段长音频的识别结果。",
    "confidence": 0.93,
    "duration_ms": 120000,
    "processing_ms": 18000
  }
}
```

### 5.3 取消识别任务

```http
DELETE /v1/asr/jobs/{job_id}
```

响应示例：

```json
{
  "trace_id": "trc_asr_job_003",
  "job_id": "job_asr_001",
  "status": "cancelled"
}
```

## 6. Provider 列表接口

### 6.1 查询可用 ASR Provider

```http
GET /v1/asr/providers
```

响应示例：

```json
{
  "trace_id": "trc_asr_provider_001",
  "providers": [
    {
      "name": "local_whisper",
      "type": "local",
      "models": ["whisper-small", "whisper-medium"],
      "languages": ["zh-CN", "en-US"],
      "audio_formats": ["wav", "mp3", "flac"],
      "features": ["timestamps", "punctuation"]
    },
    {
      "name": "cloud_asr",
      "type": "remote",
      "models": ["general-large"],
      "languages": ["zh-CN", "en-US"],
      "audio_formats": ["wav", "mp3", "opus"],
      "features": ["timestamps", "word_timestamps", "speaker_diarization", "hotwords"]
    }
  ]
}
```

## 7. 错误响应

### 7.1 错误格式

```json
{
  "trace_id": "trc_asr_error_001",
  "error": {
    "code": "unsupported_audio_format",
    "message": "Audio format opus is not supported by provider local_whisper",
    "stage": "asr",
    "retryable": false
  }
}
```

### 7.2 错误码

| HTTP 状态码 | 错误码 | 说明 |
| --- | --- | --- |
| 400 | invalid_request | 请求参数错误。 |
| 401 | unauthorized | 鉴权失败。 |
| 413 | audio_too_large | 音频文件过大。 |
| 415 | unsupported_audio_format | 不支持的音频格式。 |
| 422 | asr_no_speech | 未检测到有效语音。 |
| 422 | asr_low_confidence | 识别置信度过低。 |
| 429 | rate_limited | 请求频率或并发超限。 |
| 502 | provider_error | ASR Provider 调用失败。 |
| 504 | provider_timeout | ASR Provider 调用超时。 |

## 8. 限制与建议

- 同步接口建议限制音频时长不超过 60 秒。
- 长音频建议使用异步任务接口。
- PCM 音频必须传入 `sample_rate`、`channels` 和 `bit_depth`。
- 开启词级时间戳和说话人分离会增加处理耗时。
- 热词数量和长度应设置上限，避免影响模型稳定性。

