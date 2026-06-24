from fastapi import FastAPI

from app.api.routes import asr, health, llm, openai_compat, speakers, tts, ws_dialogue
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
    app.include_router(health.router)
    app.include_router(openai_compat.router, prefix=settings.api_prefix)
    app.include_router(asr.router, prefix=settings.api_prefix)
    app.include_router(llm.router, prefix=settings.api_prefix)
    app.include_router(tts.router, prefix=settings.api_prefix)
    app.include_router(speakers.router, prefix=settings.api_prefix)
    app.include_router(ws_dialogue.router, prefix=settings.api_prefix)
    return app


app = create_app()
