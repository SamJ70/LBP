from fastapi import APIRouter
from app.services.model_registry import list_models

router = APIRouter()

@router.get("/")
def get_available_models():
    """List all registered AI models."""
    return list_models()