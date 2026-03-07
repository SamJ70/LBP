from fastapi import APIRouter, HTTPException
from app.models.schemas import MachiningInput, OptimizationResult, PredictionOutput
from app.services.model_registry import get_model
from app.ml_models.huggingface_model import HuggingFaceModel
from typing import Dict, Any

router = APIRouter()


def _input_to_dict(inp: MachiningInput) -> Dict[str, Any]:
    return {
        "process_type":  inp.process_type.value,
        "material":      inp.material.value,
        "tool_material": inp.tool_material.value,
        "spindle_speed": inp.spindle_speed,
        "feed_rate":     inp.feed_rate,
        "depth_of_cut":  inp.depth_of_cut,
        "width_of_cut":  inp.width_of_cut,
        "tool_diameter": inp.tool_diameter,
        "coolant_used":  inp.coolant_used,
    }


@router.post("/", response_model=PredictionOutput)
def predict(inp: MachiningInput):
    """Predict energy, surface roughness, tool wear for given parameters."""
    try:
        model = get_model(inp.model_id)
        result = model.predict(_input_to_dict(inp))
        return PredictionOutput(**{k: result[k] for k in PredictionOutput.model_fields})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/optimize", response_model=OptimizationResult)
def optimize(inp: MachiningInput):
    """
    Find energy-optimal parameters while maintaining surface quality.
    Physics model: uses engineering equation constraints (Ra, tool life).
    ML models: uses guided grid search.
    """
    try:
        model = get_model(inp.model_id)
        input_dict = _input_to_dict(inp)

        original_preds = model.predict(input_dict)
        opt_params, opt_preds = model.optimize(input_dict)

        energy_orig = original_preds["energy_consumption"]
        energy_opt  = opt_preds["energy_consumption"]
        savings_pct = ((energy_orig - energy_opt) / energy_orig * 100) if energy_orig > 0 else 0

        quality_ok = opt_preds["surface_roughness"] <= original_preds["surface_roughness"] * 1.1

        notes = []
        if abs(opt_params.get("spindle_speed", 0) - input_dict["spindle_speed"]) > 1:
            notes.append(f"Spindle speed: {input_dict['spindle_speed']:.0f} → {opt_params['spindle_speed']:.0f} RPM")
        if abs(opt_params.get("feed_rate", 0) - input_dict["feed_rate"]) > 0.001:
            notes.append(f"Feed rate: {input_dict['feed_rate']:.3f} → {opt_params['feed_rate']:.4f} mm/rev")
        if abs(opt_params.get("depth_of_cut", 0) - input_dict["depth_of_cut"]) > 0.001:
            notes.append(f"Depth of cut: {input_dict['depth_of_cut']:.2f} → {opt_params['depth_of_cut']:.3f} mm")
        if not notes:
            notes.append("Current parameters are already near the energy-optimal point for given constraints.")

        # Determine optimization method label
        model_info = model.get_info()
        if model_info.get("type") == "physics":
            opt_method = "Constrained optimization: minimize Pc subject to Ra ≤ Ra_max and T ≥ T_min (Taylor / Merchant equations)"
        else:
            opt_method = f"Guided grid search via {model_info.get('name', inp.model_id)}"

        # Expert advice
        hf = HuggingFaceModel()
        ai_advice = hf.get_advice(opt_params, opt_preds) if inp.model_id == "huggingface_llm" \
                    else hf._engineering_advice(opt_params, opt_preds)

        return OptimizationResult(
            original_params=inp,
            original_predictions=PredictionOutput(**{k: original_preds[k] for k in PredictionOutput.model_fields}),
            optimized_params=opt_params,
            optimized_predictions=PredictionOutput(**{k: opt_preds[k] for k in PredictionOutput.model_fields}),
            energy_savings_percent=round(savings_pct, 2),
            quality_maintained=quality_ok,
            optimization_notes=notes,
            model_used=inp.model_id,
            ai_advice=ai_advice,
            optimization_method=opt_method,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")