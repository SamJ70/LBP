from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import itertools


class BaseMLModel(ABC):

    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Predict machining outputs."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        pass

    def is_available(self) -> bool:
        return True

    def optimize(self, inputs: Dict[str, Any], constraints: Dict[str, Any] = None) -> Tuple[Dict, Dict, Dict]:
        """Constrained optimizer."""
        constraints = constraints or {}

        current   = self.predict(inputs)
        Ra_cur    = current["surface_roughness"]
        MRR_cur   = current["mrr"]
        E_cur     = current["energy_consumption"]

        Ra_max  = constraints.get("max_surface_roughness_factor", 1.10) * Ra_cur
        MRR_min = constraints.get("min_mrr_factor", 0.60) * MRR_cur
        
        applied_constraints = {
            "max_surface_roughness_factor": constraints.get("max_surface_roughness_factor", 1.10),
            "min_mrr_factor": constraints.get("min_mrr_factor", 0.60),
            "min_tool_life": 0.0
        }

        speed = float(inputs["spindle_speed"])
        feed  = float(inputs["feed_rate"])
        doc   = float(inputs["depth_of_cut"])

        speed_scales = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        feed_scales  = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        doc_scales   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

        best_params = None
        best_energy = float('inf')
        best_preds  = None

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

        return best_params, best_preds, applied_constraints