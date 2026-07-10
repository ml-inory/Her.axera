from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import asr, health, llm, models, openai_compat, sessions, speakers, tts, users, ws_dialogue
from app.core.config import get_settings
from app.core.errors import AppError, app_error_handler


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="RESTful backend skeleton for cascaded ASR + LLM + TTS voice dialogue.",
    )
    app.add_exception_handler(AppError, app_error_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(openai_compat.router, prefix=settings.api_prefix)
    app.include_router(asr.router, prefix=settings.api_prefix)
    app.include_router(llm.router, prefix=settings.api_prefix)
    app.include_router(tts.router, prefix=settings.api_prefix)
    app.include_router(speakers.router, prefix=settings.api_prefix)
    app.include_router(users.router, prefix=settings.api_prefix)
    app.include_router(ws_dialogue.router, prefix=settings.api_prefix)
    app.include_router(sessions.router, prefix=settings.api_prefix)
    app.include_router(models.router, prefix=settings.api_prefix)

    app_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    frontend_dir = next(
        (
            candidate
            for candidate in (
                app_root / "frontend" / "static",
                repo_root / "frontend" / "static",
            )
            if candidate.exists()
        ),
        None,
    )
    if frontend_dir is not None:
        app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def frontend_redirect() -> RedirectResponse:
            return RedirectResponse(url="/ui/")

    return app


app = create_app()
