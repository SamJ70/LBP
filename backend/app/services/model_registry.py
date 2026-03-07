"""
MODEL REGISTRY
==============
Central registry for all prediction engines.

DEFAULT: physics_based — runs immediately, no setup, uses engineering equations.

To add a new AI model:
1. Create a class in ml_models/ that extends BaseMLModel
2. Implement predict() and get_info() methods
3. Register it below in MODEL_REGISTRY

The frontend automatically shows all registered models.
Physics model is always listed first (default).
"""

from typing import Dict, Type
from app.ml_models.base_model import BaseMLModel
from app.ml_models.physics_model import PhysicsBasedModel
from app.ml_models.sklearn_baseline import SklearnBaselineModel
from app.ml_models.huggingface_model import HuggingFaceModel

# ============================================================
# ORDER MATTERS — first entry is the default
# ============================================================
MODEL_REGISTRY: Dict[str, Type[BaseMLModel]] = {
    "physics_based":    PhysicsBasedModel,    # DEFAULT — always available, no setup
    "sklearn_baseline": SklearnBaselineModel, # ML model — auto-trains on first run
    "huggingface_llm":  HuggingFaceModel,     # Cloud API — needs HUGGINGFACE_API_KEY in .env

    # ---- ADD YOUR TRAINED MODELS HERE ----
    # "my_pytorch_model": MyPyTorchModel,
    # "my_xgboost":       MyXGBoostModel,
    # "my_custom_cnn":    MyCustomCNNModel,
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
    for model_id, model_class in MODEL_REGISTRY.items():
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