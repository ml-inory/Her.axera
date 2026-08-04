from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend/.. or backend/.env
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
elif (Path(__file__).resolve().parent / ".env").exists():
    load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import asr, health, llm, models, openai_compat, realtime, sessions, speakers, system, system_prompt, tts, users, wakewords, ws_dialogue
from app.core.config import get_settings
from app.core.security import RateLimitMiddleware, TokenAuthMiddleware
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
    # Security middleware (applied in order)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TokenAuthMiddleware)
    # --- Model pre-download check at startup ---
    from app.services.model_download_service import get_model_download_manager

    @app.on_event("startup")
    async def startup_model_check():
        """Pre-download models if missing (non-blocking)."""
        import logging
        _log = logging.getLogger("app.startup")
        mgr = get_model_download_manager()
        for key, state in mgr.get_all_states().items():
            if state.status.value == "not_started":
                _log.info(f"Model pre-download triggered: {key}")
                mgr.start_download(key)



    app.include_router(health.router)
    app.include_router(system.router)
    app.include_router(system_prompt.router, prefix=settings.api_prefix)
    app.include_router(openai_compat.router, prefix=settings.api_prefix)
    app.include_router(asr.router, prefix=settings.api_prefix)
    app.include_router(llm.router, prefix=settings.api_prefix)
    app.include_router(tts.router, prefix=settings.api_prefix)
    app.include_router(speakers.router, prefix=settings.api_prefix)
    app.include_router(users.router, prefix=settings.api_prefix)
    app.include_router(wakewords.router, prefix=settings.api_prefix)
    app.include_router(ws_dialogue.router, prefix=settings.api_prefix)
    app.include_router(realtime.router, prefix=settings.api_prefix)
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
