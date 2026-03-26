from fastapi import APIRouter, HTTPException
from app.models.schemas import MachiningInput, OptimizationResult, PredictionOutput
from app.services.model_registry import get_model
from app.ml_models.groq_model import GroqModel
from typing import Dict, Any

router = APIRouter()


def _to_dict(inp: MachiningInput) -> Dict[str, Any]:
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


def _safe_pred(preds: dict) -> PredictionOutput:
    """Strip internal keys."""
    fields = PredictionOutput.model_fields.keys()
    return PredictionOutput(**{k: preds[k] for k in fields if k in preds})



@router.post("/optimize", response_model=OptimizationResult)
def optimize(inp: MachiningInput):
    try:
        model      = get_model(inp.model_id)
        input_dict = _to_dict(inp)

        constraints = {}
        if inp.max_surface_roughness_factor is not None:
            constraints["max_surface_roughness_factor"] = inp.max_surface_roughness_factor
        if inp.min_mrr_factor is not None:
            constraints["min_mrr_factor"] = inp.min_mrr_factor
        if inp.min_tool_life is not None:
            constraints["min_tool_life"] = inp.min_tool_life

        original_preds_raw        = model.predict(input_dict)
        opt_params, opt_preds_raw, applied_constraints = model.optimize(input_dict, constraints)

        original_preds = _safe_pred(original_preds_raw)
        opt_preds      = _safe_pred(opt_preds_raw)

        E_orig = original_preds.energy_consumption
        E_opt  = opt_preds.energy_consumption
        savings_pct = ((E_orig - E_opt) / E_orig * 100.0) if E_orig > 0 else 0.0

        quality_ok = opt_preds.surface_roughness <= original_preds.surface_roughness * 1.05

        notes = []
        dN  = float(opt_params.get("spindle_speed", inp.spindle_speed))
        df  = float(opt_params.get("feed_rate",     inp.feed_rate))
        dap = float(opt_params.get("depth_of_cut",  inp.depth_of_cut))

        if abs(dN - inp.spindle_speed) > 1:
            pct = (dN - inp.spindle_speed) / inp.spindle_speed * 100
            notes.append(f"Spindle: {inp.spindle_speed:.0f} → {dN:.0f} RPM  ({pct:+.1f}%)")
        if abs(df - inp.feed_rate) > 0.0005:
            pct = (df - inp.feed_rate) / inp.feed_rate * 100
            notes.append(f"Feed: {inp.feed_rate:.4f} → {df:.4f} mm/rev  ({pct:+.1f}%)")
        if abs(dap - inp.depth_of_cut) > 0.0005:
            pct = (dap - inp.depth_of_cut) / inp.depth_of_cut * 100
            notes.append(f"Depth of cut: {inp.depth_of_cut:.3f} → {dap:.3f} mm  ({pct:+.1f}%)")
        if not notes:
            notes.append("Current parameters are already at the energy-optimal point for these constraints.")

        model_info = model.get_info()
        if model_info.get("type") == "physics":
            opt_method = (
                "Constrained grid search — objective: minimize P = Kc·ap·f^(1-mc)·Vc / (60·η)  "
                "subject to: Ra ≤ Ra_current × 1.1  |  T ≥ T_min (adaptive, Taylor: T=(C/Vc)^(1/n))  "
                "|  MRR ≥ MRR_current × 0.30"
            )
        elif model_info.get("type") == "llm":
            opt_method = (
                "Physics-backed grid search (same as physics engine) — "
                "LLM used for expert recommendations only, not numeric optimization"
            )
        else:
            opt_method = (
                f"Constrained grid search via {model_info.get('name', inp.model_id)} — "
                f"minimize energy subject to Ra ≤ Ra_current×1.1, MRR ≥ MRR_current×0.30"
            )

        hf = GroqModel()
        if inp.model_id == "groq_llm":
            ai_advice = hf.get_advice(opt_params, opt_preds_raw)
        else:
            ai_advice = hf._engineering_advice(opt_params, opt_preds_raw)

        return OptimizationResult(
            original_params=inp,
            original_predictions=original_preds,
            optimized_params=opt_params,
            optimized_predictions=opt_preds,
            energy_savings_percent=round(savings_pct, 2),
            quality_maintained=quality_ok,
            optimization_notes=notes,
            model_used=inp.model_id,
            ai_advice=ai_advice,
            optimization_method=opt_method,
            applied_constraints=applied_constraints,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")