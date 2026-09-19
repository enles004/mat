import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import RuntimeState, startup_load, startup_normalization
from src.api.errors import install_error_handlers
from src.api.middlewares.observability import ObservabilityMiddleware
from src.api.middlewares.request_context import RequestContextMiddleware
from src.api.v1.endpoints import health, predict
from src.core.logging import setup_logging, stop_audit_listener
from src.core.settings import Settings
from src.domain.contracts import ModelLoader


def create_app(settings: Settings, loader: ModelLoader) -> FastAPI:
    """Build the API app: logging channels, startup loading, and routes."""
    setup_logging(settings.service_name, settings.environment, getattr(logging, settings.log_level))
    state = RuntimeState()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # The model and the normalization catalog load here (and only here),
        # never at import time. A catalog configuration failure (drift,
        # unusable lock) propagates and keeps the service from becoming ready.
        startup_load(state, settings, loader)
        startup_normalization(state, settings)
        yield
        stop_audit_listener()

    app = FastAPI(lifespan=lifespan)
    install_error_handlers(app)
    # RequestContext first (inner), Observability second (outer): the 413
    # rewrite is produced by the inner middleware, so the outer capture sees
    # it, and the capture drains the body before the limit wrapper can.
    app.add_middleware(RequestContextMiddleware, max_request_bytes=settings.max_request_bytes)
    app.add_middleware(ObservabilityMiddleware, max_capture_bytes=settings.max_request_bytes)
    # Added last so it is the outermost middleware: preflight OPTIONS must be
    # answered (and headers attached) before any other layer sees the request.
    cors_origins = [
        origin.strip() for origin in settings.cors_allow_origins.split(",") if origin.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    app.include_router(health.router)
    app.include_router(predict.router)
    app.state.runtime = state
    return app
