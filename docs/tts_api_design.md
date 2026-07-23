# TTS API 设计文档

## 1. 目标

> **📋 实现状态**：本文档描述的是**已完成实现**的设计。当前实现使用 `ax_asr`/`ax_tts`/`ax_llm` 等 NPU-native Provider。详见 [README](../README.md) 中的 Provider 表。

TTS API 用于将文本合成为语音，为语音对话系统中的 `ASR + LLM + TTS` 级联链路提供语音输出能力。接口采用 RESTful API 设计，支持同步合成、异步合成、多模型选择、多音色、音频格式控制、语速语调控制和结果缓存。

## 2. 设计原则

- **统一入口**：屏蔽不同 TTS 模型、音色库和供应商的接入差异。
- **模型可选**：支持通过参数指定 Provider、模型和音色，也支持系统默认路由。
- **音频可控**：支持输出格式、采样率、语速、音量、语调等参数。
- **低延迟优先**：语音对话场景支持短文本快速合成和分句合成。
- **结果可复用**：支持音频 URL、Base64 返回和缓存命中信息。

## 3. 通用约定

### 3.1 Base URL

```text
/v1/tts
```

### 3.2 认证方式

```http
Authorization: Bearer <token>
```

### 3.3 请求追踪

客户端可通过 `X-Request-Id` 传入请求 ID，服务端响应中返回 `trace_id`。

```http
X-Request-Id: req_abc123
```

### 3.4 支持的输出格式

| 格式 | MIME Type | 说明 |
| --- | --- | --- |
| wav | audio/wav | 推荐用于高质量和本地播放。 |
| mp3 | audio/mpeg | 推荐用于 Web 和移动端。 |
| pcm | application/octet-stream | 适合嵌入式设备或实时播放链路。 |
| opus | audio/ogg | 适合低带宽场景。 |

## 4. 同步语音合成接口

### 4.1 接口说明

适用于短文本语音合成。客户端提交文本，服务端同步返回音频 URL 或 Base64 音频。

```http
POST /v1/tts/speech
Content-Type: application/json
```

### 4.2 请求字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| text | string | 是 | 待合成文本。 |
| provider | string | 否 | 指定 TTS Provider，例如 `local_tts`、`cloud_tts`。 |
| model | string | 否 | 指定具体模型名称。 |
| voice | string | 否 | 音色名称，例如 `female_default`、`male_default`。 |
| language | string | 否 | 语言代码，例如 `zh-CN`、`en-US`。 |
| audio_format | string | 否 | 输出音频格式，默认 `wav`。 |
| sample_rate | integer | 否 | 输出采样率，例如 `16000`、`24000`、`48000`。 |
| speed | number | 否 | 语速，建议范围 `0.5-2.0`，默认 `1.0`。 |
| pitch | number | 否 | 语调，建议范围 `0.5-2.0`，默认 `1.0`。 |
| volume | number | 否 | 音量，建议范围 `0.0-2.0`，默认 `1.0`。 |
| emotion | string | 否 | 情感风格，例如 `neutral`、`happy`、`sad`。 |
| return_audio_base64 | boolean | 否 | 是否直接返回 Base64 音频，默认 `false`。 |
| enable_cache | boolean | 否 | 是否启用文本音频缓存，默认 `true`。 |
| metadata | object | 否 | 业务扩展信息。 |

### 4.3 请求示例

```json
{
  "text": "你好，我是你的语音助手。",
  "provider": "local_tts",
  "model": "cosyvoice",
  "voice": "female_default",
  "language": "zh-CN",
  "audio_format": "wav",
  "sample_rate": 24000,
  "speed": 1.0,
  "pitch": 1.0,
  "volume": 1.0,
  "return_audio_base64": false,
  "enable_cache": true
}
```

### 4.4 响应字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| trace_id | string | 请求追踪 ID。 |
| provider | string | 实际使用的 TTS Provider。 |
| model | string | 实际使用的模型。 |
| voice | string | 实际使用的音色。 |
| language | string | 合成语言。 |
| audio_url | string | 合成音频 URL。 |
| audio_base64 | string | Base64 音频，仅在请求开启时返回。 |
| audio_format | string | 音频格式。 |
| sample_rate | integer | 采样率。 |
| duration_ms | integer | 合成音频时长。 |
| processing_ms | integer | 服务端处理耗时。 |
| cache_hit | boolean | 是否命中缓存。 |

### 4.5 响应示例

```json
{
  "trace_id": "trc_tts_001",
  "provider": "local_tts",
  "model": "cosyvoice",
  "voice": "female_default",
  "language": "zh-CN",
  "audio_url": "https://cdn.example.com/audio/tts_001.wav",
  "audio_format": "wav",
  "sample_rate": 24000,
  "duration_ms": 1800,
  "processing_ms": 620,
  "cache_hit": false
}
```

## 5. 分句合成接口

### 5.1 接口说明

适用于 LLM 分段输出后的低延迟播放。客户端传入多个文本片段，服务端按顺序合成并返回多个音频片段。

```http
POST /v1/tts/speech/segments
Content-Type: application/json
```

### 5.2 请求示例

```json
{
  "segments": [
    {
      "index": 0,
      "text": "好的，"
    },
    {
      "index": 1,
      "text": "已为你打开客厅的灯。"
    }
  ],
  "provider": "local_tts",
  "voice": "female_default",
  "audio_format": "mp3",
  "sample_rate": 24000
}
```

### 5.3 响应示例

```json
{
  "trace_id": "trc_tts_segments_001",
  "provider": "local_tts",
  "voice": "female_default",
  "segments": [
    {
      "index": 0,
      "text": "好的，",
      "audio_url": "https://cdn.example.com/audio/seg_0.mp3",
      "duration_ms": 420,
      "processing_ms": 180
    },
    {
      "index": 1,
      "text": "已为你打开客厅的灯。",
      "audio_url": "https://cdn.example.com/audio/seg_1.mp3",
      "duration_ms": 1280,
      "processing_ms": 360
    }
  ],
  "total_duration_ms": 1700,
  "processing_ms": 540
}
```

## 6. 异步语音合成接口

### 6.1 创建合成任务

适用于长文本、批量播报、低优先级任务或耗时较长的高质量合成。

```http
POST /v1/tts/jobs
Content-Type: application/json
```

请求字段与同步合成接口一致，可额外传入：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| callback_url | string | 否 | 任务完成后的回调地址。 |
| priority | string | 否 | 任务优先级：`low`、`normal`、`high`。 |

响应示例：

```json
{
  "trace_id": "trc_tts_job_001",
  "job_id": "job_tts_001",
  "status": "queued",
  "created_at": "2026-05-21T10:00:00+08:00"
}
```

### 6.2 查询合成任务

```http
GET /v1/tts/jobs/{job_id}
```

响应示例：

```json
{
  "trace_id": "trc_tts_job_002",
  "job_id": "job_tts_001",
  "status": "succeeded",
  "result": {
    "provider": "cloud_tts",
    "model": "general-voice-large",
    "voice": "female_news",
    "audio_url": "https://cdn.example.com/audio/job_tts_001.mp3",
    "audio_format": "mp3",
    "sample_rate": 24000,
    "duration_ms": 30000,
    "processing_ms": 6500
  }
}
```

### 6.3 取消合成任务

```http
DELETE /v1/tts/jobs/{job_id}
```

响应示例：

```json
{
  "trace_id": "trc_tts_job_003",
  "job_id": "job_tts_001",
  "status": "cancelled"
}
```

## 7. Provider 和音色接口

### 7.1 查询可用 TTS Provider

```http
GET /v1/tts/providers
```

响应示例：

```json
{
  "trace_id": "trc_tts_provider_001",
  "providers": [
    {
      "name": "local_tts",
      "type": "local",
      "models": ["cosyvoice", "piper"],
      "languages": ["zh-CN", "en-US"],
      "audio_formats": ["wav", "pcm"],
      "features": ["speed", "pitch", "volume", "emotion"]
    },
    {
      "name": "cloud_tts",
      "type": "remote",
      "models": ["general-voice-large"],
      "languages": ["zh-CN", "en-US"],
      "audio_formats": ["wav", "mp3", "opus"],
      "features": ["speed", "pitch", "volume", "emotion", "voice_clone"]
    }
  ]
}
```

### 7.2 查询可用音色

```http
GET /v1/tts/voices?provider=local_tts&language=zh-CN
```

响应示例：

```json
{
  "trace_id": "trc_tts_voice_001",
  "voices": [
    {
      "name": "female_default",
      "display_name": "默认女声",
      "language": "zh-CN",
      "gender": "female",
      "styles": ["neutral", "happy"],
      "sample_rates": [16000, 24000]
    },
    {
      "name": "male_default",
      "display_name": "默认男声",
      "language": "zh-CN",
      "gender": "male",
      "styles": ["neutral"],
      "sample_rates": [16000, 24000]
    }
  ]
}
```

## 8. 缓存接口

### 8.1 查询缓存音频

```http
GET /v1/tts/cache?text_hash={text_hash}&voice=female_default&audio_format=wav
```

响应示例：

```json
{
  "trace_id": "trc_tts_cache_001",
  "cache_hit": true,
  "audio_url": "https://cdn.example.com/audio/cache_001.wav",
  "duration_ms": 1800
}
```

### 8.2 删除缓存音频

```http
DELETE /v1/tts/cache/{cache_id}
```

响应示例：

```json
{
  "trace_id": "trc_tts_cache_002",
  "cache_id": "cache_001",
  "deleted": true
}
```

## 9. 错误响应

### 9.1 错误格式

```json
{
  "trace_id": "trc_tts_error_001",
  "error": {
    "code": "voice_not_found",
    "message": "Voice female_news is not available for provider local_tts",
    "stage": "tts",
    "retryable": false
  }
}
```

### 9.2 错误码

| HTTP 状态码 | 错误码 | 说明 |
| --- | --- | --- |
| 400 | invalid_request | 请求参数错误。 |
| 401 | unauthorized | 鉴权失败。 |
| 404 | voice_not_found | 指定音色不存在。 |
| 413 | text_too_long | 文本长度超过限制。 |
| 415 | unsupported_audio_format | 不支持的输出音频格式。 |
| 422 | unsupported_language | 不支持的语言。 |
| 429 | rate_limited | 请求频率或并发超限。 |
| 502 | provider_error | TTS Provider 调用失败。 |
| 504 | provider_timeout | TTS Provider 调用超时。 |

## 10. 限制与建议

- 同步接口建议限制文本长度，例如不超过 1000 字。
- 语音对话场景建议按句子合成，降低首段播放延迟。
- `return_audio_base64=true` 只适合短音频，长音频建议返回 `audio_url`。
- 同一文本、音色、格式、采样率组合可开启缓存。
- 不同 Provider 对 `speed`、`pitch`、`emotion` 的支持程度可能不同，调用前建议查询 Provider 能力。
- PCM 输出需要调用方明确处理采样率、声道数和位深。
