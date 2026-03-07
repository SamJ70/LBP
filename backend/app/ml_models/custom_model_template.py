"""
CUSTOM TRAINED MODEL TEMPLATE
==============================
Use this template when you train your own model.

Steps:
1. Copy this file: cp custom_model_template.py my_trained_model.py
2. Fill in the predict() method to load and use your model
3. Register it in: app/services/model_registry.py

Supports: PyTorch, TensorFlow, XGBoost, sklearn, ONNX — any framework.
"""

import os
import numpy as np
from typing import Dict, Any
from app.ml_models.base_model import BaseMLModel


class CustomTrainedModel(BaseMLModel):
    """
    Template for your own trained model.
    Replace the predict() method with your actual model loading + inference.
    """

    MODEL_FILE = "./ml_models/saved_models/my_model.pkl"  # Change this path

    def __init__(self):
        self._model = None

    def _load_model(self):
        """Load your trained model. Called lazily on first prediction."""
        if self._model is not None:
            return

        # ---- OPTION A: scikit-learn / XGBoost ----
        # import pickle
        # with open(self.MODEL_FILE, "rb") as f:
        #     self._model = pickle.load(f)

        # ---- OPTION B: PyTorch ----
        # import torch
        # self._model = torch.load(self.MODEL_FILE)
        # self._model.eval()

        # ---- OPTION C: TensorFlow/Keras ----
        # import tensorflow as tf
        # self._model = tf.keras.models.load_model(self.MODEL_FILE)

        # ---- OPTION D: ONNX ----
        # import onnxruntime as ort
        # self._model = ort.InferenceSession(self.MODEL_FILE)

        # ---- OPTION E: HuggingFace local fine-tuned ----
        # from transformers import pipeline
        # self._model = pipeline("text2text-generation", model=self.MODEL_FILE)

        raise NotImplementedError("Load your model in _load_model()")

    def _preprocess(self, inputs: Dict[str, Any]) -> np.ndarray:
        """
        Convert inputs dict to model input format.
        Match the preprocessing used during training!
        """
        # Example: encode categoricals and return numpy array
        label_map = {
            "process_type":   {"turning": 0, "milling": 1, "drilling": 2, "grinding": 3},
            "material":       {"aluminum": 0, "steel_mild": 1, "steel_stainless": 2, "cast_iron": 3, "titanium": 4, "copper": 5},
            "tool_material":  {"hss": 0, "carbide": 1, "ceramic": 2, "cbn": 3, "diamond": 4},
        }
        return np.array([[
            label_map["process_type"].get(inputs["process_type"], 0),
            label_map["material"].get(inputs["material"], 0),
            label_map["tool_material"].get(inputs["tool_material"], 0),
            float(inputs["spindle_speed"]),
            float(inputs["feed_rate"]),
            float(inputs["depth_of_cut"]),
            float(inputs.get("width_of_cut") or 5.0),
            float(inputs.get("tool_diameter") or 20.0),
            int(inputs.get("coolant_used", False)),
        ]])

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._load_model()
        X = self._preprocess(inputs)

        # === REPLACE THIS with your model's inference call ===
        # For sklearn:  preds = self._model.predict(X)[0]
        # For PyTorch:  preds = self._model(torch.tensor(X)).detach().numpy()[0]
        # For Keras:    preds = self._model.predict(X)[0]
        preds = [0, 0, 0, 0]  # [energy, Ra, wear_rate, mrr]

        return {
            "energy_consumption": round(max(0, float(preds[0])), 2),
            "surface_roughness":  round(max(0, float(preds[1])), 3),
            "tool_wear_rate":     round(max(0, float(preds[2])), 6),
            "mrr":                round(max(0, float(preds[3])), 2),
            "confidence_score":   0.90,  # Set based on your model's validation accuracy
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            "id": "custom_trained",
            "name": "My Custom Trained Model",
            "description": "Replace this description with your model's details.",
            "type": "custom_nn",  # or "xgboost", "pytorch", etc.
            "accuracy_metrics": {"R2": 0.95, "RMSE": "TBD"},
            "supported_processes": ["turning", "milling", "drilling", "grinding"],
        }

    def is_available(self) -> bool:
        return os.path.exists(self.MODEL_FILE)