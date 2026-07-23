# LLM API 设计文档

## 1. 目标

> **📋 实现状态**：本文档描述的是**已完成实现**的设计。当前实现使用 `ax_asr`/`ax_tts`/`ax_llm` 等 NPU-native Provider。详见 [README](../README.md) 中的 Provider 表。

LLM API 用于在语音对话系统中完成语义理解、上下文推理和回复生成，是 `ASR + LLM + TTS` 级联链路中的核心决策模块。接口采用 RESTful API 设计，支持同步对话、异步任务、多模型路由、会话上下文、工具调用预留、结构化输出和安全控制。

## 2. 设计原则

- **模型解耦**：业务接口不绑定具体大模型供应商或推理框架。
- **上下文可控**：支持由客户端传入消息，也支持通过 `session_id` 使用服务端上下文。
- **可配置生成**：支持温度、最大 token、停止词、系统提示词等生成参数。
- **安全可审计**：保留 trace、模型版本、输入输出和安全拦截结果。
- **易于级联**：输入可直接消费 ASR 文本，输出可直接传递给 TTS。

## 3. 通用约定

### 3.1 Base URL

```text
/v1/llm
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

### 3.4 Message 格式

```json
{
  "role": "user",
  "content": "帮我打开客厅的灯。",
  "content_type": "text",
  "metadata": {
    "source": "asr",
    "asr_provider": "local_whisper"
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| role | string | 是 | 消息角色：`system`、`user`、`assistant`、`tool`。 |
| content | string | 是 | 消息内容。 |
| content_type | string | 否 | 内容类型，默认 `text`。 |
| metadata | object | 否 | 扩展信息。 |

## 4. 同步对话接口

### 4.1 接口说明

适用于短文本输入和实时语音对话中的文本回复生成。服务端同步返回完整回复文本。

```http
POST /v1/llm/chat/completions
Content-Type: application/json
```

### 4.2 请求字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| messages | array | 是 | 对话消息列表。 |
| session_id | string | 否 | 会话 ID。传入后服务端可自动加载上下文。 |
| user_id | string | 否 | 用户 ID。 |
| provider | string | 否 | 指定 LLM Provider，例如 `local_llm`、`cloud_llm`。 |
| model | string | 否 | 指定模型名称。 |
| system_prompt | string | 否 | 系统提示词。优先级可高于默认提示词。 |
| temperature | number | 否 | 采样温度，默认 `0.7`。 |
| top_p | number | 否 | nucleus sampling 参数，默认 `0.9`。 |
| max_tokens | integer | 否 | 最大生成 token 数。 |
| stop | array[string] | 否 | 停止词列表。 |
| response_format | object | 否 | 输出格式约束，例如 JSON。 |
| tools | array | 否 | 可调用工具定义，预留 Agent 场景。 |
| tool_choice | string/object | 否 | 工具选择策略。 |
| safety | object | 否 | 内容安全配置。 |
| metadata | object | 否 | 业务扩展信息。 |

### 4.3 请求示例

```json
{
  "session_id": "ses_abc123",
  "user_id": "user_001",
  "provider": "local_llm",
  "model": "qwen2.5-7b-instruct",
  "messages": [
    {
      "role": "system",
      "content": "你是一个简洁、友好的语音助手。"
    },
    {
      "role": "user",
      "content": "帮我打开客厅的灯。",
      "metadata": {
        "source": "asr"
      }
    }
  ],
  "temperature": 0.3,
  "max_tokens": 512
}
```

### 4.4 响应字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| trace_id | string | 请求追踪 ID。 |
| session_id | string | 会话 ID。 |
| provider | string | 实际使用的 LLM Provider。 |
| model | string | 实际使用的模型。 |
| message | object | 回复消息。 |
| finish_reason | string | 结束原因：`stop`、`length`、`tool_calls`、`content_filter`。 |
| usage | object | token 用量。 |
| safety | object | 安全审核结果。 |
| processing_ms | integer | 服务端处理耗时。 |

### 4.5 响应示例

```json
{
  "trace_id": "trc_llm_001",
  "session_id": "ses_abc123",
  "provider": "local_llm",
  "model": "qwen2.5-7b-instruct",
  "message": {
    "role": "assistant",
    "content": "好的，已为你打开客厅的灯。",
    "content_type": "text"
  },
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 128,
    "completion_tokens": 18,
    "total_tokens": 146
  },
  "safety": {
    "blocked": false,
    "categories": []
  },
  "processing_ms": 920
}
```

## 5. 流式兼容设计

RESTful API 本身不适合真正的双向流式通信，但可通过以下两种方式兼容逐步输出需求。

### 5.1 SSE 流式接口

如果服务端允许 Server-Sent Events，可提供如下接口。

```http
POST /v1/llm/chat/stream
Content-Type: application/json
Accept: text/event-stream
```

事件示例：

```text
event: message.delta
data: {"trace_id":"trc_llm_stream_001","delta":"好的"}

event: message.delta
data: {"trace_id":"trc_llm_stream_001","delta":"，已为你打开客厅的灯。"}

event: message.completed
data: {"trace_id":"trc_llm_stream_001","finish_reason":"stop"}
```

### 5.2 分段任务接口

若严格限制为普通 RESTful API，可使用异步任务接口轮询生成状态。

```http
POST /v1/llm/chat/jobs
GET /v1/llm/chat/jobs/{job_id}
DELETE /v1/llm/chat/jobs/{job_id}
```

## 6. 异步对话接口

### 6.1 创建任务

```http
POST /v1/llm/chat/jobs
Content-Type: application/json
```

请求字段与同步对话接口一致，可额外传入：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| callback_url | string | 否 | 任务完成后的回调地址。 |
| priority | string | 否 | 任务优先级：`low`、`normal`、`high`。 |

响应示例：

```json
{
  "trace_id": "trc_llm_job_001",
  "job_id": "job_llm_001",
  "status": "queued",
  "created_at": "2026-05-21T10:00:00+08:00"
}
```

### 6.2 查询任务

```http
GET /v1/llm/chat/jobs/{job_id}
```

响应示例：

```json
{
  "trace_id": "trc_llm_job_002",
  "job_id": "job_llm_001",
  "status": "succeeded",
  "result": {
    "provider": "cloud_llm",
    "model": "general-chat-large",
    "message": {
      "role": "assistant",
      "content": "我已经根据你的要求生成了完整回复。"
    },
    "finish_reason": "stop",
    "usage": {
      "prompt_tokens": 256,
      "completion_tokens": 64,
      "total_tokens": 320
    }
  }
}
```

### 6.3 取消任务

```http
DELETE /v1/llm/chat/jobs/{job_id}
```

响应示例：

```json
{
  "trace_id": "trc_llm_job_003",
  "job_id": "job_llm_001",
  "status": "cancelled"
}
```

## 7. 会话上下文接口

### 7.1 查询会话消息

```http
GET /v1/llm/sessions/{session_id}/messages
```

响应示例：

```json
{
  "trace_id": "trc_llm_session_001",
  "session_id": "ses_abc123",
  "messages": [
    {
      "role": "user",
      "content": "帮我打开客厅的灯。",
      "created_at": "2026-05-21T10:00:00+08:00"
    },
    {
      "role": "assistant",
      "content": "好的，已为你打开客厅的灯。",
      "created_at": "2026-05-21T10:00:01+08:00"
    }
  ]
}
```

### 7.2 清空会话消息

```http
DELETE /v1/llm/sessions/{session_id}/messages
```

响应示例：

```json
{
  "trace_id": "trc_llm_session_002",
  "session_id": "ses_abc123",
  "deleted": true
}
```

## 8. Provider 列表接口

### 8.1 查询可用 LLM Provider

```http
GET /v1/llm/providers
```

响应示例：

```json
{
  "trace_id": "trc_llm_provider_001",
  "providers": [
    {
      "name": "local_llm",
      "type": "local",
      "models": ["qwen2.5-7b-instruct", "llama-3.1-8b-instruct"],
      "features": ["chat", "json_output", "tool_calls"],
      "max_context_tokens": 32768
    },
    {
      "name": "cloud_llm",
      "type": "remote",
      "models": ["general-chat-large"],
      "features": ["chat", "stream", "json_output", "tool_calls", "safety_filter"],
      "max_context_tokens": 128000
    }
  ]
}
```

## 9. 结构化输出

当业务需要稳定 JSON 输出时，可传入 `response_format`。

请求示例：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "从这句话中提取意图：打开客厅的灯。"
    }
  ],
  "response_format": {
    "type": "json_schema",
    "schema": {
      "type": "object",
      "properties": {
        "intent": { "type": "string" },
        "location": { "type": "string" },
        "device": { "type": "string" }
      },
      "required": ["intent", "device"]
    }
  }
}
```

响应示例：

```json
{
  "trace_id": "trc_llm_json_001",
  "message": {
    "role": "assistant",
    "content": "{\"intent\":\"turn_on\",\"location\":\"客厅\",\"device\":\"灯\"}",
    "content_type": "json"
  },
  "finish_reason": "stop"
}
```

## 10. 错误响应

### 10.1 错误格式

```json
{
  "trace_id": "trc_llm_error_001",
  "error": {
    "code": "context_length_exceeded",
    "message": "Input messages exceed model context limit",
    "stage": "llm",
    "retryable": false
  }
}
```

### 10.2 错误码

| HTTP 状态码 | 错误码 | 说明 |
| --- | --- | --- |
| 400 | invalid_request | 请求参数错误。 |
| 401 | unauthorized | 鉴权失败。 |
| 404 | session_not_found | 会话不存在。 |
| 413 | context_length_exceeded | 上下文超出模型限制。 |
| 422 | content_filtered | 输入或输出被安全策略拦截。 |
| 429 | rate_limited | 请求频率或并发超限。 |
| 502 | provider_error | LLM Provider 调用失败。 |
| 504 | provider_timeout | LLM Provider 调用超时。 |

## 11. 限制与建议

- 语音对话场景建议控制回复长度，降低 TTS 合成延迟。
- `session_id` 与 `messages` 同时传入时，需明确合并策略，推荐服务端上下文在前、请求消息在后。
- 工具调用需要设置超时、权限和审计日志。
- 结构化输出应配合 JSON 校验和失败重试。
- 内容安全策略建议覆盖输入和输出两个方向。
