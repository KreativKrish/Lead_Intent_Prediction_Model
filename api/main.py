"""FastAPI application factory."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.utils.config import get_config
from src.utils.logger import get_logger
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

    # Serve frontend static files at /static (CSS, JS)
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if os.path.isdir(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

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
        "ui": "/ui",
    }


@app.get("/ui", include_in_schema=False)
async def serve_ui():
    """Serve the prediction dashboard frontend."""
    frontend_index = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(frontend_index)


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
