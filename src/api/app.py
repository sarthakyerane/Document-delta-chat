"""
delta-chat · src/api/app.py
FastAPI application factory with lifespan management.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.observability.logging import configure_logging, get_logger
from src.observability.tracing import setup_otel

log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup + shutdown."""
    # Startup
    configure_logging()
    setup_otel()

    from src.db.session import init_db
    try:
        init_db()
        log.info("startup.db.ok")
    except Exception as e:
        log.warning("startup.db.failed", error=str(e))

    log.info("startup.complete",
             service=settings.otel_service_name,
             version=settings.otel_service_version)

    yield

    # Shutdown
    log.info("shutdown.complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Document Delta & Grounded Chat",
        description=(
            "Format-agnostic pipeline: ingest two document revisions, "
            "compute a structured delta, and chat with grounded citations."
        ),
        version=settings.otel_service_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from src.api.routes import health, ingest, delta, chat, traces
    app.include_router(health.router, tags=["Health"])
    app.include_router(ingest.router, tags=["Pipeline"])
    app.include_router(delta.router, tags=["Delta"])
    app.include_router(chat.router, tags=["Chat"])
    app.include_router(traces.router, tags=["Observability"])

    return app
