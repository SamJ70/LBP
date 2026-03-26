from typing import Dict, Type
from app.ml_models.base_model import BaseMLModel
from app.ml_models.physics_model import PhysicsBasedModel
from app.ml_models.sklearn_baseline import SklearnBaselineModel
from app.ml_models.groq_model import GroqModel
from app.ml_models.calculated_baseline import CalculatedBaselineModel

# First entry is the default model used if client doesn't specify one
MODEL_REGISTRY: Dict[str, Type[BaseMLModel]] = {
    "sklearn_baseline": SklearnBaselineModel,
    "calculated_baseline": CalculatedBaselineModel,
}

_instances: Dict[str, BaseMLModel] = {}


def get_model(model_id: str) -> BaseMLModel:
    if model_id not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{model_id}'. Available: {available}")
    if model_id not in _instances:
        _instances[model_id] = MODEL_REGISTRY[model_id]()
    return _instances[model_id]


def list_models():
    result = []
    for model_id, _ in MODEL_REGISTRY.items():
        try:
            inst = get_model(model_id)
            info = inst.get_info()
            info["available"] = inst.is_available()
        except Exception as e:
            info = {
                "id": model_id, "name": model_id,
                "description": f"Error: {e}", "type": "unknown",
                "available": False, "supported_processes": [],
            }
        result.append(info)
    return result