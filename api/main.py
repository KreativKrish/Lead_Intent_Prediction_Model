"""FastAPI application factory."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..src.utils.config import get_config
from ..src.utils.logger import get_logger
from .dependencies import lifespan
from .routers import health, predict

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI app.
    """
    config = get_config()

    app = FastAPI(
        title=config.get("api.title", "Lead Intent Prediction API"),
        version=config.get("api.version", "0.1.0"),
        description=config.get("api.description", "API for lead intent predictions"),
        lifespan=lifespan,
    )

    # Add CORS middleware
    origins = config.get("api.cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router)
    app.include_router(predict.router)

    logger.info("FastAPI application created")

    return app


app = create_app()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Lead Intent Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


def main():
    """Run API server."""
    config = get_config()
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    workers = config.get("api.workers", 4)

    logger.info(f"Starting API server on {host}:{port}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=os.getenv("ENV") == "development",
    )


if __name__ == "__main__":
    main()
