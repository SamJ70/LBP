from typing import Any, Dict, Tuple
from app.ml_models.base_model import BaseMLModel


class CalculatedBaselineModel(BaseMLModel):
    """Calculated baseline model."""

    def __init__(self):
        pass

    def _physics_predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from app.ml_models.physics_model import PhysicsBasedModel
        physics = PhysicsBasedModel()
        return physics.predict(inputs)
        
    def _physics_optimize(self, inputs: Dict[str, Any], constraints: Dict[str, Any] = None) -> Tuple[Dict, Dict, Dict]:
        from app.ml_models.physics_model import PhysicsBasedModel
        physics = PhysicsBasedModel()
        return physics.optimize(inputs, constraints)

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return baseline predictions from physics engine."""
        physics_pred = self._physics_predict(inputs)
        final_energy = float(physics_pred["energy_consumption"]) + 0.0
        final_ra = float(physics_pred["surface_roughness"]) + 0.0
        final_wear = float(physics_pred["tool_wear_rate"]) + 0.0
        final_mrr = float(physics_pred["mrr"]) + 0.0

        return {
            "energy_consumption": round(final_energy, 2),
            "surface_roughness": round(final_ra, 3),
            "tool_wear_rate": round(final_wear, 6),
            "mrr": round(final_mrr, 2),
            "tool_life_min": physics_pred.get("tool_life_min"),
            "vc_mpm": physics_pred.get("vc_mpm")
        }
        
    def optimize(self, inputs: Dict[str, Any], constraints: Dict[str, Any] = None) -> Tuple[Dict, Dict, Dict]:
        return self._physics_optimize(inputs, constraints)

    def get_info(self) -> Dict[str, Any]:
        return {
            "id": "calculated_baseline",
            "name": "Calculated Baseline (Physics Only)",
            "description": (
                "A baseline model flow that executes the entire workflow on the calculated value only, "
                "i.e., it only uses the physics engine directly and sets the ML delta to zero."
            ),
            "type": "calculated_baseline",
            "available": True,
            "accuracy_metrics": {
                "version": 1,
                "note": "Returns pure physics calculations simulating 0 ML delta.",
            },
            "supported_processes": ["turning", "milling", "drilling", "grinding"],
        }

    def is_available(self) -> bool:
        return True
