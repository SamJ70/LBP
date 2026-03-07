"""
BASE MODEL
==========
All AI models must extend this class.
Implement predict() and get_info() at minimum.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseMLModel(ABC):

    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict machining outputs for given inputs.
        
        Args:
            inputs: dict with keys:
                - process_type: str
                - material: str
                - tool_material: str
                - spindle_speed: float (RPM)
                - feed_rate: float (mm/rev)
                - depth_of_cut: float (mm)
                - width_of_cut: float (mm, optional)
                - tool_diameter: float (mm, optional)
                - coolant_used: bool

        Returns:
            dict with keys:
                - energy_consumption: float (Watts)
                - surface_roughness: float (Ra, micrometers)
                - tool_wear_rate: float (mm/min)
                - mrr: float (mm³/min)
                - confidence_score: float (0-1)
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        pass

    def is_available(self) -> bool:
        """Check if model is ready to use (e.g., API key present, file exists)."""
        return True

    def optimize(self, inputs: Dict[str, Any], constraints: Dict[str, Any] = None) -> Tuple[Dict, Dict]:
        """
        Find optimized parameters.
        Default implementation: grid search over parameter space.
        Override for smarter optimization (e.g., genetic algorithm, Bayesian).
        
        Returns:
            (optimized_params, optimized_predictions)
        """
        import itertools

        constraints = constraints or {}
        min_ra = constraints.get("max_surface_roughness", 3.0)  # max Ra allowed
        
        best_params = None
        best_energy = float('inf')
        best_preds = None

        # Search space: ±20% around input values
        speed = inputs["spindle_speed"]
        feed = inputs["feed_rate"]
        doc = inputs["depth_of_cut"]

        candidates = list(itertools.product(
            [speed * 0.8, speed * 0.9, speed, speed * 1.1, speed * 1.2],
            [feed * 0.8, feed * 0.9, feed, feed * 1.1],
            [doc * 0.7, doc * 0.8, doc * 0.9, doc],
        ))

        for s, f, d in candidates:
            test_input = {**inputs, "spindle_speed": s, "feed_rate": f, "depth_of_cut": d}
            preds = self.predict(test_input)
            if preds["surface_roughness"] <= min_ra:
                if preds["energy_consumption"] < best_energy:
                    best_energy = preds["energy_consumption"]
                    best_params = test_input
                    best_preds = preds

        if best_params is None:
            best_params = inputs
            best_preds = self.predict(inputs)

        return best_params, best_preds