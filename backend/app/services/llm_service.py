from collections.abc import AsyncIterator
from datetime import datetime
import asyncio
import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import requests

from app.core.config import get_settings
from app.core.errors import AppError
from app.models.common import JobCreatedResponse, ProviderInfo
from app.models.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeleteSessionMessagesResponse,
    LLMJobResponse,
    SafetyResult,
    SessionMessagesResponse,
    TokenUsage,
)

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        deepseek_models = list(
            dict.fromkeys(
                [
                    self.settings.deepseek_model,
                    "deepseek-v4-pro",
                    "deepseek-v4-flash",
                    "deepseek-chat",
                    "deepseek-reasoner",
                ]
            )
        )
        self.providers = {
            "mock_llm": ProviderInfo(
                name="mock_llm",
                type="mock",
                models=["mock-llm"],
                languages=["zh-CN", "en-US"],
                features=["chat", "streaming", "deterministic"],
            ),
            "deepseek": ProviderInfo(
                name="deepseek",
                type="remote",
                models=deepseek_models,
                languages=["zh-CN", "en-US"],
                features=["chat", "json_output"],
                metadata={"api_base": self.settings.deepseek_api_base, "openai_compatible": True},
            ),
        }
        if self._should_register_openai_compat():
            self.providers["openai_compat"] = ProviderInfo(
                name="openai_compat",
                type="remote",
                models=[self.settings.openai_compat_model],
                languages=["zh-CN", "en-US"],
                features=["chat", "json_output"],
                metadata={"api_base": self.settings.openai_compat_api_base, "openai_compatible": True},
            )
        self.sessions: dict[str, list[ChatMessage]] = {}
        self._session_meta: dict[str, dict] = {}  # {session_id: {title, created_at, last_active, user_id}}
        self.jobs: dict[str, LLMJobResponse] = {}
        self._load_sessions()

    def _should_register_openai_compat(self) -> bool:
        return bool(self.settings.enable_openai_compat or self.settings.openai_compat_api_base)

    # ── Session Persistence ────────────────────────────────────────

    def _session_path(self) -> Path:
        return _DATA_DIR / self.settings.session_persistence_path

    def _load_sessions(self) -> None:
        if not self.settings.enable_session_persistence:
            return
        path = self._session_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for sid, messages in data.items():
                self.sessions[sid] = [ChatMessage(**m) for m in messages]
            logger.info("Loaded %d sessions from %s", len(self.sessions), path)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load sessions from %s", path, exc_info=True)
        self._load_session_meta()

    def _save_sessions(self) -> None:
        if not self.settings.enable_session_persistence:
            return
        path = self._session_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {sid: [m.model_dump() for m in msgs] for sid, msgs in self.sessions.items()}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _maybe_trim(self, session_id: str) -> None:
        msgs = self.sessions.get(session_id)
        if not msgs or len(msgs) <= self.settings.session_max_messages:
            return
        self.sessions[session_id] = msgs[-self.settings.session_max_messages :]

    # ── Session Metadata ───────────────────────────────────────────

    def _ensure_meta(self, session_id: str, user_id: str | None = None) -> None:
        if session_id not in self._session_meta:
            now = datetime.now().astimezone().isoformat()
            self._session_meta[session_id] = {
                "title": "",
                "created_at": now,
                "last_active": now,
                "user_id": user_id,
            }

    def _touch_session(self, session_id: str) -> None:
        if session_id in self._session_meta:
            self._session_meta[session_id]["last_active"] = datetime.now().astimezone().isoformat()

    def _auto_title(self, session_id: str) -> None:
        """Generate a title from the first user message."""
        if session_id not in self._session_meta:
            return
        meta = self._session_meta[session_id]
        if meta.get("title"):
            return
        msgs = self.sessions.get(session_id, [])
        for m in msgs:
            if m.role == "user":
                text = m.text_content.strip()
                meta["title"] = text[:30] + ("..." if len(text) > 30 else "")
                break
        self._save_session_meta()

    def _save_session_meta(self) -> None:
        if not self.settings.enable_session_persistence:
            return
        path = _DATA_DIR / "session_meta.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._session_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("Failed to save session meta", exc_info=True)

    def _load_session_meta(self) -> None:
        if not self.settings.enable_session_persistence:
            return
        path = _DATA_DIR / "session_meta.json"
        if not path.exists():
            return
        try:
            self._session_meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load session meta", exc_info=True)

    def list_sessions(self, user_id: str | None = None) -> list[dict[str, object]]:
        """List all sessions, optionally filtered by user."""
        result = []
        for sid, meta in self._session_meta.items():
            if user_id and meta.get("user_id") != user_id:
                continue
            msgs = self.sessions.get(sid, [])
            result.append({
                "session_id": sid,
                "title": meta.get("title", ""),
                "message_count": len(msgs),
                "created_at": meta.get("created_at", ""),
                "last_active": meta.get("last_active", ""),
                "user_id": meta.get("user_id"),
            })
        result.sort(key=lambda s: s["last_active"], reverse=True)
        return result

    def get_session(self, session_id: str) -> dict[str, object] | None:
        meta = self._session_meta.get(session_id)
        if not meta:
            return None
        msgs = self.sessions.get(session_id, [])
        return {
            "session_id": session_id,
            "title": meta.get("title", ""),
            "message_count": len(msgs),
            "created_at": meta.get("created_at", ""),
            "last_active": meta.get("last_active", ""),
            "user_id": meta.get("user_id"),
            "messages": [m.model_dump() for m in msgs],
        }

    def delete_session(self, session_id: str) -> bool:
        self.sessions.pop(session_id, None)
        self._session_meta.pop(session_id, None)
        self._save_sessions()
        self._save_session_meta()
        return True

    # ── Provider helpers ───────────────────────────────────────────

    def _resolve_api(self, provider_name: str, request: ChatCompletionRequest) -> tuple[str, str, str]:
        """Return (api_base, api_key, model) for an OpenAI-compatible provider."""
        if provider_name == "deepseek":
            api_base = self.settings.deepseek_api_base
            api_key = (request.api_key or self.settings.deepseek_api_key or "").strip()
            model = request.model or self.settings.deepseek_model
        elif provider_name == "openai_compat":
            api_base = self.settings.openai_compat_api_base
            api_key = (request.api_key or self.settings.openai_compat_api_key or "").strip()
            model = request.model or self.settings.openai_compat_model
        else:
            raise AppError("provider_not_found", f"Unknown API provider: {provider_name}", status_code=404, stage="llm")
        if not api_key:
            raise AppError("missing_api_key", f"{provider_name} API KEY is required", status_code=400, stage="llm")
        return api_base, api_key, model

    def _build_payload(self, request: ChatCompletionRequest, model: str, *, stream: bool = False) -> dict:
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content if isinstance(msg.content, (str, list)) else str(msg.content)} for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        if stream:
            payload["stream"] = True
        if request.stop:
            payload["stop"] = request.stop
        if request.response_format:
            payload["response_format"] = request.response_format
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        return payload

    # ── Non-streaming chat ─────────────────────────────────────────

    def list_providers(self) -> list[ProviderInfo]:
        return list(self.providers.values())

    def chat(self, trace_id: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        start = perf_counter()
        provider_name = request.provider or self.settings.default_llm_provider
        provider_info = self.providers.get(provider_name)
        if provider_info is None:
            raise AppError("provider_not_found", f"LLM provider {provider_name} is not configured", status_code=404, stage="llm")
        if not request.messages:
            raise AppError("invalid_request", "messages must not be empty", status_code=400, stage="llm")

        if provider_name == "mock_llm":
            return self._chat_mock(trace_id, request, provider_info.models[0], start)
        selected_model = request.model or provider_info.models[0]
        if provider_name not in ("deepseek", "openai_compat"):
            raise AppError("provider_not_found", f"LLM provider {provider_name} is not configured", status_code=404, stage="llm")
        return self._chat_openai_api(trace_id, request, provider_name, selected_model, start)

    def _chat_mock(
        self,
        trace_id: str,
        request: ChatCompletionRequest,
        selected_model: str,
        start: float,
    ) -> ChatCompletionResponse:
        user_text = next((message.text_content for message in reversed(request.messages) if message.role == "user"), "")
        content = f"收到：{user_text}。我会用简短自然的方式回应。"
        message = ChatMessage(role="assistant", content=content)
        session_id = request.session_id
        if session_id:
            self.sessions.setdefault(session_id, []).extend(request.messages)
            self.sessions[session_id].append(message)
            self._maybe_trim(session_id)
            self._ensure_meta(session_id, request.user_id)
            self._touch_session(session_id)
            self._auto_title(session_id)
            self._save_sessions()
        prompt_tokens = sum(len(message.text_content) for message in request.messages)
        completion_tokens = len(content)
        return ChatCompletionResponse(
            trace_id=trace_id,
            session_id=session_id,
            provider="mock_llm",
            model=selected_model,
            message=message,
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            safety=SafetyResult(blocked=False, categories=[]),
            processing_ms=int((perf_counter() - start) * 1000),
        )

    def _chat_openai_api(self, trace_id: str, request: ChatCompletionRequest, provider_name: str, selected_model: str, start: float) -> ChatCompletionResponse:
        api_base, api_key, model = self._resolve_api(provider_name, request)
        payload = self._build_payload(request, model or selected_model)
        try:
            response = requests.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload, timeout=self.settings.llm_request_timeout,
            )
        except requests.RequestException as exc:
            raise AppError("provider_unavailable", f"{provider_name} request failed: {exc}", status_code=502, stage="llm", retryable=True) from exc
        if response.status_code >= 400:
            raise AppError("provider_error", f"{provider_name} returned {response.status_code}: {response.text[:500]}", status_code=502, stage="llm", retryable=response.status_code >= 500)
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise AppError("provider_error", f"{provider_name} returned no choices", status_code=502, stage="llm")
        choice = choices[0]
        response_message = choice.get("message") or {}
        content = response_message.get("content") or ""
        usage_data = data.get("usage") or {}
        finish_reason = choice.get("finish_reason") or "stop"
        if finish_reason not in {"stop", "length", "tool_calls", "content_filter"}:
            finish_reason = "stop"
        message = ChatMessage(role="assistant", content=content)
        session_id = request.session_id
        if session_id:
            self.sessions.setdefault(session_id, []).extend(request.messages)
            self.sessions[session_id].append(message)
            self._maybe_trim(session_id)
            self._ensure_meta(session_id, request.user_id)
            self._touch_session(session_id)
            self._auto_title(session_id)
            self._save_sessions()
        prompt_tokens = int(usage_data.get("prompt_tokens") or 0)
        completion_tokens = int(usage_data.get("completion_tokens") or 0)
        return ChatCompletionResponse(
            trace_id=trace_id, session_id=session_id, provider=provider_name,
            model=data.get("model") or selected_model, message=message, finish_reason=finish_reason,
            usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=int(usage_data.get("total_tokens") or prompt_tokens + completion_tokens)),
            safety=SafetyResult(blocked=False, categories=[]),
            processing_ms=int((perf_counter() - start) * 1000),
        )

    # ── Streaming chat ─────────────────────────────────────────────

    async def chat_stream(self, trace_id: str, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Stream LLM response token by token. Yields content delta strings."""
        provider_name = request.provider or self.settings.default_llm_provider
        provider_info = self.providers.get(provider_name)
        if provider_info is None:
            raise AppError("provider_not_found", f"LLM provider {provider_name} is not configured", status_code=404, stage="llm")
        if not request.messages:
            raise AppError("invalid_request", "messages must not be empty", status_code=400, stage="llm")
        if provider_name == "mock_llm":
            response = self._chat_mock(trace_id, request, provider_info.models[0], perf_counter())
            for token in response.message.content.split("，"):
                yield token if token.endswith("。") else f"{token}，"
            return
        if provider_name not in ("deepseek", "openai_compat"):
            raise AppError("provider_not_found", f"LLM provider {provider_name} is not configured", status_code=404, stage="llm")
        async for token in self._stream_openai_api(request, provider_name):
            yield token

    async def _stream_openai_api(self, request: ChatCompletionRequest, provider_name: str) -> AsyncIterator[str]:
        api_base, api_key, model = self._resolve_api(provider_name, request)
        payload = self._build_payload(request, model, stream=True)
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    f"{api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload, timeout=self.settings.llm_request_timeout, stream=True,
                ),
            )
        except requests.RequestException as exc:
            raise AppError("provider_unavailable", f"{provider_name} stream request failed: {exc}", status_code=502, stage="llm", retryable=True) from exc
        if response.status_code >= 400:
            raise AppError("provider_error", f"{provider_name} returned {response.status_code}: {response.text[:500]}", status_code=502, stage="llm", retryable=response.status_code >= 500)
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield content

    # ── Jobs ───────────────────────────────────────────────────────

    def create_job(self, trace_id: str) -> JobCreatedResponse:
        job_id = f"job_llm_{uuid4().hex}"
        response = LLMJobResponse(trace_id=trace_id, job_id=job_id, status="queued")
        self.jobs[job_id] = response
        return JobCreatedResponse(trace_id=trace_id, job_id=job_id, status="queued", created_at=datetime.now().astimezone().isoformat())

    def get_job(self, trace_id: str, job_id: str) -> LLMJobResponse:
        job = self.jobs.get(job_id)
        if job is None:
            raise AppError("job_not_found", f"LLM job {job_id} not found", status_code=404, stage="llm")
        return job.model_copy(update={"trace_id": trace_id})

    def cancel_job(self, trace_id: str, job_id: str) -> LLMJobResponse:
        job = self.get_job(trace_id, job_id)
        cancelled = job.model_copy(update={"status": "cancelled", "trace_id": trace_id})
        self.jobs[job_id] = cancelled
        return cancelled

    # ── Session management ─────────────────────────────────────────

    def get_session_messages(self, trace_id: str, session_id: str) -> SessionMessagesResponse:
        return SessionMessagesResponse(trace_id=trace_id, session_id=session_id, messages=self.sessions.get(session_id, []))

    def delete_session_messages(self, trace_id: str, session_id: str) -> DeleteSessionMessagesResponse:
        self.sessions.pop(session_id, None)
        self._save_sessions()
        return DeleteSessionMessagesResponse(trace_id=trace_id, session_id=session_id, deleted=True)


llm_service = LLMService()
