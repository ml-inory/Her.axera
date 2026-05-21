from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        stage: str | None = None,
        retryable: bool = False,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.stage = stage
        self.retryable = retryable


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "trace_id": trace_id,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "stage": exc.stage,
                "retryable": exc.retryable,
            },
        },
    )
