# OpenAI-Compatible API 设计文档

本文档定义后端对外暴露的 OpenAI-Compatible API。后端不再对外注册旧版 `/v1/asr/*`、`/v1/llm/*`、`/v1/tts/*` 路由，后续统一使用本文档中的接口。

## 1. 设计目标

- 对齐 OpenAI 常见接口路径和请求/响应结构，降低客户端集成成本。
- ASR、LLM、TTS 统一放在 `/v1` 下。
- 兼容标准字段，同时允许通过扩展字段选择本系统 Provider。
- 保留多模型路由能力，例如 `mock_asr`、`ax_asr`、`ax_asr`、`deepseek`、`edge_tts`。

## 2. 接口总览

| 能力 | OpenAI-Compatible 路径 | 说明 |
| --- | --- | --- |
| LLM Chat | `POST /v1/chat/completions` | 对齐 Chat Completions。 |
| ASR | `POST /v1/audio/transcriptions` | 对齐 Audio Transcriptions。 |
| TTS | `POST /v1/audio/speech` | 对齐 Audio Speech。 |

## 3. 通用约定

### 3.1 鉴权

```http
Authorization: Bearer <token>
```

当前骨架未强制鉴权，后续可在依赖层统一开启。

### 3.2 Trace

客户端可传：

```http
X-Request-Id: req_xxx
```

服务端会在响应中返回 `trace_id`，或在音频响应头返回 `X-Trace-Id`。

### 3.3 Provider 扩展字段

OpenAI 标准字段通常使用 `model`。本系统支持额外传入 `provider` 来明确选择底层 Provider。

示例：

```json
{
  "model": "deepseek-v4-pro",
  "provider": "deepseek",
  "messages": [
    {"role": "user", "content": "你好"}
  ]
}
```

若不传 `provider`，系统会尝试从 `model` 推断 Provider；无法推断时使用默认 Provider。

## 4. LLM Chat Completions

### 4.1 请求

```http
POST /v1/chat/completions
Content-Type: application/json
```

请求字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| model | string | 是 | 模型名，也可传 Provider 名，例如 `mock_llm`、`deepseek`。 |
| messages | array | 是 | OpenAI 格式消息列表。 |
| temperature | number | 否 | 默认 `0.7`。 |
| top_p | number | 否 | 默认 `0.9`。 |
| max_tokens | integer | 否 | 默认 `512`。 |
| stop | string/array | 否 | 停止词。 |
| stream | boolean | 否 | 当前暂不支持，传 `true` 会返回错误。 |
| response_format | object | 否 | 结构化输出配置。 |
| provider | string | 否 | 扩展字段，指定后端 Provider。 |
| api_key | string | 否 | 扩展字段，用于透传第三方模型 API Key。 |
| session_id | string | 否 | 扩展字段，用于服务端会话记忆。 |

请求示例：

```json
{
  "model": "mock_llm",
  "messages": [
    {"role": "user", "content": "帮我打开客厅的灯。"}
  ],
  "temperature": 0.3
}
```

### 4.2 响应

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "mock-chat-general",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "收到：帮我打开客厅的灯。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  },
  "trace_id": "trc_xxx",
  "provider": "mock_llm"
}
```

## 5. Audio Transcriptions

### 5.1 请求

```http
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

请求字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| file | file | 是 | 待识别音频文件，对齐 OpenAI 字段名。 |
| model | string | 否 | 默认 `whisper-1`；也可传 `ax_asr`、`ax_asr`、`mock_asr` 等。 |
| language | string | 否 | 语言，例如 `zh`、`zh-CN`、`en`、`auto`。 |
| prompt | string | 否 | 保留兼容字段。 |
| response_format | string | 否 | `json`、`text`、`verbose_json`；`srt/vtt` 暂不支持。 |
| temperature | number | 否 | 保留兼容字段。 |
| provider | string | 否 | 扩展字段，指定 ASR Provider。 |
| enable_vad | boolean | 否 | 扩展字段，是否启用 VAD。 |

请求示例：

```bash
curl -X POST "http://127.0.0.1:8080/v1/audio/transcriptions" \
  -F "file=@input.wav" \
  -F "model=ax_asr" \
  -F "language=zh" \
  -F "response_format=verbose_json"
```

### 5.2 `json` 响应

```json
{
  "text": "开放时间早上9点至下午5点。"
}
```

### 5.3 `verbose_json` 响应

```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 5.616,
  "text": "开放时间早上9点至下午5点。",
  "segments": [],
  "trace_id": "trc_xxx",
  "provider": "ax_asr",
  "model": "ax_asr_sensevoice"
}
```

## 6. Audio Speech

### 6.1 请求

```http
POST /v1/audio/speech
Content-Type: application/json
```

请求字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| model | string | 否 | 默认 `tts-1`；也可传 `edge_tts`、`mock_tts`。 |
| input | string | 是 | 待合成文本，对齐 OpenAI 字段名。 |
| voice | string | 否 | 默认 `alloy`。OpenAI 标准音色会映射到本系统音色。 |
| response_format | string | 否 | `mp3`、`opus`、`aac`、`flac`、`wav`、`pcm`。 |
| speed | number | 否 | 默认 `1.0`。 |
| provider | string | 否 | 扩展字段，指定 TTS Provider。 |
| language | string | 否 | 扩展字段，默认 `zh-CN`。 |
| sample_rate | integer | 否 | 扩展字段，默认 `24000`。 |

请求示例：

```json
{
  "model": "mock_tts",
  "input": "你好，我是你的语音助手。",
  "voice": "alloy",
  "response_format": "wav"
}
```

### 6.2 响应

响应体直接返回音频字节流，而不是 JSON。

响应头：

```http
Content-Type: audio/wav
X-Trace-Id: trc_xxx
X-Provider: mock_tts
X-Model: mock-tts-general
```

## 7. 兼容性说明

- `stream=true` 暂未实现，会返回 `stream_not_supported`。
- ASR 的 `srt` 和 `vtt` 暂未实现，会返回 `response_format_not_supported`。
- `provider`、`api_key`、`session_id`、`enable_vad`、`language`、`sample_rate` 是本系统扩展字段。
- OpenAI 标准音色会映射到当前 TTS Provider 支持的音色。
- 旧版 `/v1/asr/*`、`/v1/llm/*`、`/v1/tts/*` 路由不再注册。

## 8. 参考资料

- OpenAI Chat Completions API Reference: <https://platform.openai.com/docs/api-reference/chat/completions/create>
- OpenAI Audio API Reference: <https://platform.openai.com/docs/api-reference/audio>
- OpenAI Speech to text Guide: <https://platform.openai.com/docs/guides/speech-to-text>
