"""Dependency injection for FastAPI."""

import os
import uuid
from contextlib import asynccontextmanager

from ..src.models import ModelRegistry
from ..src.utils.logger import get_logger

logger = get_logger(__name__)

# Global model cache
_model_cache = None
_model_registry = None


def init_model() -> None:
    """Initialize and load model on startup."""
    global _model_cache, _model_registry

    try:
        _model_registry = ModelRegistry()
        stage = os.getenv("MODEL_STAGE", "Production")
        _model_cache = _model_registry.load_champion(stage=stage)
        logger.info(f"Model loaded successfully from {stage} stage")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def get_model():
    """Get loaded model from cache."""
    if _model_cache is None:
        raise RuntimeError("Model not initialized. Call init_model() first.")
    return _model_cache


def get_model_registry() -> ModelRegistry:
    """Get model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def get_request_id() -> str:
    """Generate unique request ID."""
    return str(uuid.uuid4())


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan context manager."""
    # Startup
    logger.info("Starting up API service")
    init_model()
    yield
    # Shutdown
    logger.info("Shutting down API service")
