"""Health check endpoints."""

from fastapi import APIRouter

from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=True,
        message="Service is running",
    )


@router.get("/ready", response_model=HealthResponse)
async def ready() -> HealthResponse:
    """Readiness check endpoint."""
    return HealthResponse(
        status="ready",
        version="0.1.0",
        model_loaded=True,
        message="Service is ready to handle requests",
    )
