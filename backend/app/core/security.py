"""Authentication and rate-limiting middleware for Her.axera backend."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Callable

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ── Token Auth Middleware ──────────────────────────────────────────

class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Simple Bearer token authentication middleware.

    Enabled via ENABLE_REQUEST_AUTH=true and configured via API_TOKEN.
    Skips health check and OpenAPI schema endpoints.
    """

    SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/ui/"}

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.settings.enable_request_auth:
            return await call_next(request)

        # Skip auth for public paths
        path = request.url.path
        if any(path == p or path.startswith(p) for p in self.SKIP_PATHS):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

        token = auth_header[7:]
        expected_token = (self.settings.api_token or "").strip()
        if not expected_token:
            logger.warning("ENABLE_REQUEST_AUTH is true but API_TOKEN is not set — denying all requests")
            raise HTTPException(status_code=401, detail="Server authentication is not configured")

        if token != expected_token:
            raise HTTPException(status_code=403, detail="Invalid API token")

        return await call_next(request)


# ── Rate Limiter Middleware ────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple sliding-window in-memory rate limiter.

    Configured via RATE_LIMIT_REQUESTS and RATE_LIMIT_WINDOW_SEC.
    Tracks per-IP request counts. Returns 429 when exceeded.
    """

    SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.settings = get_settings()
        self._windows: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.settings.enable_rate_limit:
            return await call_next(request)

        path = request.url.path
        if any(path == p or path.startswith(p) for p in self.SKIP_PATHS):
            return await call_next(request)

        # Identify client
        client_ip = request.client.host if request.client else "unknown"
        max_requests = self.settings.rate_limit_requests
        window_sec = self.settings.rate_limit_window_sec

        now = time.monotonic()
        window = self._windows[client_ip]

        # Remove expired entries
        cutoff = now - window_sec
        while window and window[0] < cutoff:
            window.pop(0)

        if len(window) >= max_requests:
            retry_after = int(window[0] + window_sec - now) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)
        return await call_next(request)
