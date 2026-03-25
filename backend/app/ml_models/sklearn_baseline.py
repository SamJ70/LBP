from __future__ import annotations

import os
import pickle
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from app.ml_models.base_model import BaseMLModel


LABEL_MAPS = {
    "process_type": {"turning": 0, "milling": 1, "drilling": 2, "grinding": 3},
    "material": {
        "aluminum": 0,
        "steel_mild": 1,
        "steel_stainless": 2,
        "cast_iron": 3,
        "titanium": 4,
        "copper": 5,
    },
    "tool_material": {"hss": 0, "carbide": 1, "ceramic": 2, "cbn": 3, "diamond": 4},
}

FEATURE_NAMES: List[str] = [
    "process_type",
    "material",
    "tool_material",
    "spindle_speed",
    "feed_rate",
    "depth_of_cut",
    "width_of_cut",
    "tool_diameter",
    "coolant_used",
    "physics_energy_consumption",
    "physics_surface_roughness",
    "physics_tool_wear_rate",
    "physics_mrr",
    "physics_tool_life_min",
    "physics_vc_mpm",
]

TARGET_NAMES: List[str] = [
    "delta_energy",
    "delta_ra",
    "delta_wear",
    "delta_mrr",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "sklearn_baseline.pkl")

# Typical training-domain bounds used only for a simple confidence heuristic.
TRAINING_BOUNDS = {
    "spindle_speed": (200.0, 3000.0),
    "feed_rate": (0.05, 0.5),
    "depth_of_cut": (0.2, 5.0),
    "width_of_cut": (1.0, 20.0),
    "tool_diameter": (6.0, 50.0),
    "physics_energy_consumption": (10.0, 4000.0),
    "physics_surface_roughness": (0.05, 15.0),
    "physics_tool_wear_rate": (0.0, 1.0),
    "physics_mrr": (0.1, 50000.0),
    "physics_tool_life_min": (0.0, 9999.0),
    "physics_vc_mpm": (1.0, 4000.0),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _encode_categorical(name: str, value: str) -> int:
    return LABEL_MAPS[name].get(str(value), 0)


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _build_feature_row(inputs: Dict[str, Any], physics_pred: Dict[str, Any]) -> List[float]:
    return [
        _encode_categorical("process_type", inputs.get("process_type", "turning")),
        _encode_categorical("material", inputs.get("material", "steel_mild")),
        _encode_categorical("tool_material", inputs.get("tool_material", "carbide")),
        _coerce_float(inputs.get("spindle_speed"), 800.0),
        _coerce_float(inputs.get("feed_rate"), 0.15),
        _coerce_float(inputs.get("depth_of_cut"), 1.5),
        _coerce_float(inputs.get("width_of_cut"), 5.0),
        _coerce_float(inputs.get("tool_diameter"), 20.0),
        float(int(bool(inputs.get("coolant_used", False)))),
        _coerce_float(physics_pred.get("energy_consumption"), 0.0),
        _coerce_float(physics_pred.get("surface_roughness"), 0.0),
        _coerce_float(physics_pred.get("tool_wear_rate"), 0.0),
        _coerce_float(physics_pred.get("mrr"), 0.0),
        _coerce_float(physics_pred.get("tool_life_min"), 0.0),
        _coerce_float(physics_pred.get("vc_mpm"), 0.0),
    ]


def _generate_training_data(n: int = 3000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic residual-learning data.

    X = [input params + physics outputs]
    Y = [delta_energy, delta_ra, delta_wear, delta_mrr]

    We simulate "real-world" values by applying structured, input-dependent noise
    on top of the physics output. That gives the ML model a meaningful residual
    target to learn.
    """
    from app.ml_models.physics_model import PhysicsBasedModel

    rng = random.Random(seed)
    physics = PhysicsBasedModel()

    processes = ["turning", "milling", "drilling", "grinding"]
    materials = ["aluminum", "steel_mild", "steel_stainless", "cast_iron", "titanium", "copper"]
    tools = ["hss", "carbide", "ceramic", "cbn", "diamond"]

    X: List[List[float]] = []
    Y: List[List[float]] = []

    for _ in range(n):
        process = rng.choice(processes)
        material = rng.choice(materials)
        tool = rng.choice(tools)

        inp = {
            "process_type": process,
            "material": material,
            "tool_material": tool,
            "spindle_speed": rng.uniform(200, 3000),
            "feed_rate": rng.uniform(0.05, 0.5),
            "depth_of_cut": rng.uniform(0.2, 5.0),
            "width_of_cut": rng.uniform(1.0, 20.0),
            "tool_diameter": rng.uniform(6.0, 50.0),
            "coolant_used": rng.choice([True, False]),
        }

        physics_pred = physics.predict(inp)

        # Structured noise model:
        # - harder materials tend to show larger deviations
        # - missing coolant tends to worsen Ra and wear
        # - higher speed tends to increase deviation slightly
        hard_material = material in {"steel_stainless", "titanium", "cast_iron"}
        soft_material = material in {"aluminum", "copper"}
        coolant = bool(inp["coolant_used"])
        speed = float(inp["spindle_speed"])
        feed = float(inp["feed_rate"])
        doc = float(inp["depth_of_cut"])

        base_scale = 0.04
        if hard_material:
            base_scale += 0.04
        if not coolant:
            base_scale += 0.03
        if speed > 2000:
            base_scale += 0.02
        if feed > 0.30:
            base_scale += 0.01
        if doc > 3.0:
            base_scale += 0.01
        if soft_material:
            base_scale -= 0.01

        # "Real" outputs are physics outputs perturbed by structured residuals.
        # These residuals are what the model learns.
        energy_bias = 0.0
        ra_bias = 0.0
        wear_bias = 0.0
        mrr_bias = 0.0

        if hard_material:
            energy_bias += 0.06
            ra_bias += 0.12
            wear_bias += 0.10
            mrr_bias -= 0.03
        if not coolant:
            energy_bias += 0.03
            ra_bias += 0.08
            wear_bias += 0.12
        else:
            energy_bias -= 0.02
            ra_bias -= 0.04
            wear_bias -= 0.03
        if speed > 1800:
            energy_bias += 0.03
            ra_bias += 0.05
            wear_bias += 0.04
        if feed > 0.25:
            ra_bias += 0.04
            mrr_bias += 0.05
        if doc > 2.5:
            energy_bias += 0.02
            wear_bias += 0.03
            mrr_bias += 0.04

        def noisy_multiplier(bias: float) -> float:
            return 1.0 + bias + rng.gauss(0.0, base_scale)

        physics_energy = float(physics_pred["energy_consumption"])
        physics_ra = float(physics_pred["surface_roughness"])
        physics_wear = float(physics_pred["tool_wear_rate"])
        physics_mrr = float(physics_pred["mrr"])

        real_energy = max(10.0, physics_energy * noisy_multiplier(energy_bias))
        real_ra = max(0.01, physics_ra * noisy_multiplier(ra_bias))
        real_wear = max(0.0, physics_wear * noisy_multiplier(wear_bias))
        real_mrr = max(0.01, physics_mrr * noisy_multiplier(mrr_bias))

        delta_energy = real_energy - physics_energy
        delta_ra = real_ra - physics_ra
        delta_wear = real_wear - physics_wear
        delta_mrr = real_mrr - physics_mrr

        X.append(_build_feature_row(inp, physics_pred))
        Y.append([delta_energy, delta_ra, delta_wear, delta_mrr])

    return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)


def _train_and_save():
    """
    Train a multi-output regression model on residuals and save it to disk.

    The saved artifact is a dict, not just the pipeline, so future versions can
    inspect metadata without breaking backwards compatibility.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("Training sklearn residual model on synthetic data...")
    X, Y = _generate_training_data(n=3000, seed=42)

    model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
        )
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    pipe.fit(X, Y)

    artifact = {
        "model": pipe,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "version": 2,
        "trained_on": "synthetic_residual_data",
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Model saved to {MODEL_PATH}")
    return artifact


class SklearnBaselineModel(BaseMLModel):
    """
    Residual-correction model.

    Predict() returns final corrected outputs:
        physics_output + predicted_delta

    This keeps the API stable while making the ML model actually useful.
    """

    def __init__(self):
        self._artifact = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                loaded = pickle.load(f)

            # Backward compatibility: support both old plain pipeline pickle
            # and the new artifact dict format.
            if isinstance(loaded, dict) and "model" in loaded:
                self._artifact = loaded
                self._model = loaded["model"]
            else:
                self._artifact = {
                    "model": loaded,
                    "feature_names": FEATURE_NAMES,
                    "target_names": TARGET_NAMES,
                    "version": 1,
                    "trained_on": "legacy_plain_pipeline",
                }
                self._model = loaded
        else:
            self._artifact = _train_and_save()
            self._model = self._artifact["model"]

    def _physics_predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from app.ml_models.physics_model import PhysicsBasedModel

        physics = PhysicsBasedModel()
        return physics.predict(inputs)

    def _encode(self, inputs: Dict[str, Any]) -> np.ndarray:
        physics_pred = self._physics_predict(inputs)
        return np.asarray([_build_feature_row(inputs, physics_pred)], dtype=float)

    def _estimate_confidence(
        self,
        inputs: Dict[str, Any],
        physics_pred: Dict[str, Any],
        deltas: np.ndarray,
    ) -> float:
        """
        Heuristic confidence score:
        - higher when inputs lie within the synthetic training domain
        - lower when residual corrections are large
        This is not a probabilistic confidence; it is a simple usability metric.
        """
        values = {
            "spindle_speed": _coerce_float(inputs.get("spindle_speed"), 800.0),
            "feed_rate": _coerce_float(inputs.get("feed_rate"), 0.15),
            "depth_of_cut": _coerce_float(inputs.get("depth_of_cut"), 1.5),
            "width_of_cut": _coerce_float(inputs.get("width_of_cut"), 5.0),
            "tool_diameter": _coerce_float(inputs.get("tool_diameter"), 20.0),
            "physics_energy_consumption": _coerce_float(physics_pred.get("energy_consumption"), 0.0),
            "physics_surface_roughness": _coerce_float(physics_pred.get("surface_roughness"), 0.0),
            "physics_tool_wear_rate": _coerce_float(physics_pred.get("tool_wear_rate"), 0.0),
            "physics_mrr": _coerce_float(physics_pred.get("mrr"), 0.0),
            "physics_tool_life_min": _coerce_float(physics_pred.get("tool_life_min"), 0.0),
            "physics_vc_mpm": _coerce_float(physics_pred.get("vc_mpm"), 0.0),
        }

        domain_score = 1.0
        for key, (lo, hi) in TRAINING_BOUNDS.items():
            v = values[key]
            if hi <= lo:
                continue
            if v < lo:
                distance = (lo - v) / (hi - lo)
                domain_score -= min(0.15, 0.15 * distance)
            elif v > hi:
                distance = (v - hi) / (hi - lo)
                domain_score -= min(0.15, 0.15 * distance)

        residual_ratio = 0.0
        if deltas.size > 0:
            # Normalize by the magnitude of physics predictions to avoid over-penalizing
            # large absolute units like energy and MRR.
            denom = np.array(
                [
                    max(abs(_coerce_float(physics_pred.get("energy_consumption"), 1.0)), 1.0),
                    max(abs(_coerce_float(physics_pred.get("surface_roughness"), 1.0)), 1.0),
                    max(abs(_coerce_float(physics_pred.get("tool_wear_rate"), 1.0)), 1.0),
                    max(abs(_coerce_float(physics_pred.get("mrr"), 1.0)), 1.0),
                ],
                dtype=float,
            )
            residual_ratio = float(np.mean(np.abs(deltas) / denom))

        confidence = 0.95 * domain_score - 0.40 * residual_ratio
        return float(_clamp(confidence, 0.50, 0.97))

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return final corrected predictions:
            final = physics + predicted_delta
        """
        self._load()

        physics_pred = self._physics_predict(inputs)
        X = self._encode(inputs)
        deltas = np.asarray(self._model.predict(X)[0], dtype=float)

        final_energy = max(10.0, float(physics_pred["energy_consumption"]) + float(deltas[0]))
        final_ra = max(0.01, float(physics_pred["surface_roughness"]) + float(deltas[1]))
        final_wear = max(0.0, float(physics_pred["tool_wear_rate"]) + float(deltas[2]))
        final_mrr = max(0.01, float(physics_pred["mrr"]) + float(deltas[3]))

        confidence = self._estimate_confidence(inputs, physics_pred, deltas)

        return {
            "energy_consumption": round(final_energy, 2),
            "surface_roughness": round(final_ra, 3),
            "tool_wear_rate": round(final_wear, 6),
            "mrr": round(final_mrr, 2),
        }

    def get_info(self) -> Dict[str, Any]:
        loaded = self._artifact is not None
        version = self._artifact.get("version", 2) if loaded else 2

        return {
            "id": "sklearn_baseline",
            "name": "Residual Gradient Boosting (Sklearn)",
            "description": (
                "A residual-correction model trained on synthetic machining data. "
                "It learns delta corrections on top of the physics baseline, then returns "
                "final corrected outputs. This keeps the interface stable while making the "
                "ML layer meaningful."
            ),
            "type": "sklearn_residual",
            "available": True,
            "accuracy_metrics": {
                "version": version,
                "trained_on": "synthetic_residual_data",
                "note": "Residual model: predicts corrections over physics outputs",
            },
            "supported_processes": ["turning", "milling", "drilling", "grinding"],
        }

    def is_available(self) -> bool:
        return True