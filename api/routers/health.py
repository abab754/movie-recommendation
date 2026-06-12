"""Health check endpoint."""

from fastapi import APIRouter

from api.services.svd_service import get_model_version

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_version": get_model_version(),
    }
