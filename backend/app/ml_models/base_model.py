from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import itertools


class BaseMLModel(ABC):

    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict machining outputs.
        Returns dict with:
          energy_consumption (W), surface_roughness (Ra μm),
          tool_wear_rate (mm/min), mrr (mm³/min), confidence_score (0-1)
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        pass

    def is_available(self) -> bool:
        return True

    def optimize(self, inputs: Dict[str, Any], constraints: Dict[str, Any] = None) -> Tuple[Dict, Dict]:
        """
        Default constrained optimizer for all models.
        Uses self.predict() so it works correctly with any model's predictions.
        """
        constraints = constraints or {}

        current   = self.predict(inputs)
        Ra_cur    = current["surface_roughness"]
        MRR_cur   = current["mrr"]
        E_cur     = current["energy_consumption"]

        # Adaptive constraints — never hard-coded
        Ra_max  = constraints.get("max_surface_roughness", Ra_cur * 1.10)
        MRR_min = MRR_cur * 0.60   # keep ≥60% productivity — realistic for production

        speed = float(inputs["spindle_speed"])
        feed  = float(inputs["feed_rate"])
        doc   = float(inputs["depth_of_cut"])

        # Physics insight: P ∝ Fc·Vc ∝ Kc·ap·f^(1-mc)·Vc
        # → Reducing speed and DoC always reduces energy
        # → Speed range capped at 100% (higher = higher power, never optimal for energy)
        # → Feed reduced to improve Ra, but floor at 40% to prevent killing productivity
        speed_scales = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        feed_scales  = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        doc_scales   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

        best_params = None
        best_energy = float('inf')
        best_preds  = None

        # Pass 1: strict Ra + MRR constraints
        for ss, fs, ds in itertools.product(speed_scales, feed_scales, doc_scales):
            test = {
                **inputs,
                "spindle_speed": round(speed * ss, 1),
                "feed_rate":     round(feed * fs,  5),
                "depth_of_cut":  round(doc * ds,   5),
            }
            p = self.predict(test)
            if p["surface_roughness"] > Ra_max:
                continue
            if p["mrr"] < MRR_min:
                continue
            if p["energy_consumption"] < best_energy:
                best_energy = p["energy_consumption"]
                best_params = test
                best_preds  = p

        # Pass 2: relax MRR to 0% floor, widen Ra to 1.5× (rare fallback)
        if best_params is None:
            for ss, fs, ds in itertools.product(speed_scales, feed_scales, doc_scales):
                test = {
                    **inputs,
                    "spindle_speed": round(speed * ss, 1),
                    "feed_rate":     round(feed * fs,  5),
                    "depth_of_cut":  round(doc * ds,   5),
                }
                p = self.predict(test)
                if p["surface_roughness"] <= Ra_cur * 1.50:
                    if p["energy_consumption"] < best_energy:
                        best_energy = p["energy_consumption"]
                        best_params = test
                        best_preds  = p

        if best_params is None:
            best_params = inputs
            best_preds  = current

        return best_params, best_preds