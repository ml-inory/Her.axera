from typing import Annotated

from fastapi import Header, Request

from app.core.tracing import new_trace_id


async def bind_trace_id(
    request: Request,
    x_request_id: Annotated[str | None, Header(alias="X-Request-Id")] = None,
) -> str:
    trace_id = x_request_id or new_trace_id()
    request.state.trace_id = trace_id
    return trace_id
