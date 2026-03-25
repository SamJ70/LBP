import os
import math
from openai import OpenAI
from typing import Dict, Any, Tuple
from app.ml_models.base_model import BaseMLModel
from app.ml_models.physics_model import PhysicsBasedModel, MATERIALS
from dotenv import load_dotenv

GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
load_dotenv()

# Human-readable labels for LLM prompt (not internal enum strings)
_MAT_LABEL = {
    "aluminum":        "Aluminum",
    "steel_mild":      "Mild Steel",
    "steel_stainless": "Stainless Steel (304/316)",
    "cast_iron":       "Cast Iron",
    "titanium":        "Titanium",
    "copper":          "Copper",
}
_TOOL_LABEL = {
    "hss":     "HSS",
    "carbide": "Carbide",
    "ceramic": "Ceramic",
    "cbn":     "CBN",
    "diamond": "Diamond",
}
_PROC_LABEL = {
    "turning":  "CNC Turning",
    "milling":  "CNC Milling",
    "drilling": "Drilling",
    "grinding": "Grinding",
}


class GroqModel(BaseMLModel):

    def __init__(self):
        self._physics  = PhysicsBasedModel()
        self._api_key  = os.getenv("GROQ_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()

    # ─────────────────────────────────────────────────────────────────────
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Numeric predictions always from physics (reliable, deterministic)."""
        preds = self._physics.predict(inputs)
        return preds

    def optimize(self, inputs: Dict[str, Any], constraints: Dict[str, Any] = None) -> Tuple[Dict, Dict]:
        """
        Use physics optimizer directly — HF model gives identical numeric results
        since predict() is physics-backed. No point calling base_model grid search
        separately; physics optimizer is more accurate (includes Taylor constraints).
        """
        return self._physics.optimize(inputs, constraints)

    # ─────────────────────────────────────────────────────────────────────
    def _call_llm(self, prompt: str) -> str:
        api_key = self._api_key
        if not api_key:
            return ""

        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            response = client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a CNC machining expert. Provide precise, numeric, "
                            "engineering-based recommendations. Avoid vague advice."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=300,
                top_p=0.9,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"[Groq API error: {str(e)}]"

    def _validate_llm_response(self, text: str) -> bool:
        """
        Reject LLM output that is clearly hallucinated or irrelevant.
        Checks: non-empty, contains at least one digit (real advice has numbers),
                not an error message, not excessively short.
        """
        if not text or text.startswith("["):
            return False
        if len(text) < 40:
            return False
        has_number = any(c.isdigit() for c in text)
        return has_number

    def get_advice(self, inputs: Dict[str, Any], predictions: Dict[str, Any]) -> str:
        """
        Generate expert recommendations.
        Uses LLM if API key is present and response passes validation.
        Falls back to deterministic rule-based advice otherwise.
        """
        if self._api_key:
            llm_text = self._call_llm(self._build_prompt(inputs, predictions))
            if self._validate_llm_response(llm_text):
                return llm_text

        return self._engineering_advice(inputs, predictions)

    def _build_prompt(self, inputs: Dict[str, Any], predictions: Dict[str, Any]) -> str:
        """
        High-quality engineering prompt for LLM.
        Focus: grounded, numeric, constraint-aware recommendations.
        """

        proc  = _PROC_LABEL.get(inputs.get("process_type","turning"), inputs.get("process_type",""))
        mat   = _MAT_LABEL.get(inputs.get("material","steel_mild"), inputs.get("material",""))
        tool  = _TOOL_LABEL.get(inputs.get("tool_material","carbide"), inputs.get("tool_material",""))

        N     = inputs.get("spindle_speed", 800)
        f     = inputs.get("feed_rate", 0.15)
        ap    = inputs.get("depth_of_cut", 1.5)
        cool  = "yes" if inputs.get("coolant_used") else "no"

        E     = predictions.get("energy_consumption", 0)
        Ra    = predictions.get("surface_roughness", 0)
        mrr   = predictions.get("mrr", 0)
        T_min = predictions.get("tool_life_min", 0)

        return (
            f"[INST]\n"
            f"You are a senior CNC machining process optimization engineer.\n"
            f"Your task is to improve machining efficiency using REAL engineering principles.\n\n"

            f"⚠️ STRICT RULES:\n"
            f"- Do NOT hallucinate values\n"
            f"- Use machining physics (Vc, feed, chip load, Taylor tool life, etc.)\n"
            f"- Every recommendation MUST include numeric changes\n"
            f"- Keep surface finish within acceptable range (Ra should not worsen significantly)\n"
            f"- Maintain reasonable productivity (MRR should not drop drastically)\n"
            f"- Avoid generic advice — be specific and actionable\n\n"

            f"--- CURRENT MACHINING SETUP ---\n"
            f"Process: {proc}\n"
            f"Material: {mat}\n"
            f"Tool: {tool}\n"
            f"Spindle Speed (N): {N} RPM\n"
            f"Feed Rate (f): {f} mm/rev\n"
            f"Depth of Cut (ap): {ap} mm\n"
            f"Coolant: {cool}\n\n"

            f"--- PERFORMANCE METRICS ---\n"
            f"Energy Consumption: {E:.0f} W\n"
            f"Surface Roughness (Ra): {Ra:.2f} μm\n"
            f"Material Removal Rate (MRR): {mrr:.0f} mm³/min\n"
            f"Tool Life: {T_min:.0f} min\n\n"

            f"--- TASK ---\n"
            f"Provide EXACTLY 3 numbered recommendations to reduce energy consumption "
            f"while maintaining machining quality and productivity.\n\n"

            f"--- OUTPUT FORMAT (STRICT) ---\n"
            f"1. [Change] → [New Value] → [Reason with formula/principle]\n"
            f"2. ...\n"
            f"3. ...\n\n"

            f"Example:\n"
            f"1. Reduce spindle speed from 1200 RPM to 900 RPM → reduces cutting speed (Vc ∝ N), "
            f"leading to ~25% lower power consumption.\n\n"

            "NOTE:\n"
            "If current setup is already near optimal, explicitly say so and suggest monitoring strategies."

            f"Now generate the recommendations.\n"
            f"[/INST]"
        )

    def _engineering_advice(self, inputs: Dict[str, Any], predictions: Dict[str, Any]) -> str:
        """
        Deterministic rule-based engineering advice.
        All tips are physically correct and grounded in machining handbook data.

        FIX: removed wrong tip 'reduce DoC + increase speed' (that increases power).
        FIX: feed suggestion now uses correct Ra = 32*f²/r formula.
        FIX: grinding advice uses correct units (table speed, not mm/rev).
        FIX: all advice cross-checked against P = Kc·ap·f^(1-mc)·Vc / (60·η).
        """
        tips = []
        Ra    = float(predictions.get("surface_roughness", 0))
        power = float(predictions.get("energy_consumption", 0))
        T_min = float(predictions.get("tool_life_min", 9999))
        mrr   = float(predictions.get("mrr", 0))
        f     = float(inputs.get("feed_rate", 0.15))
        ap    = float(inputs.get("depth_of_cut", 1.5))
        N     = float(inputs.get("spindle_speed", 800))
        mat   = inputs.get("material", "steel_mild")
        tool  = inputs.get("tool_material", "carbide")
        proc  = inputs.get("process_type", "turning")
        cool  = bool(inputs.get("coolant_used", False))
        D     = float(inputs.get("tool_diameter") or 20.0)

        # ── 1. Surface roughness guidance ─────────────────────────────────
        # Ra = 32·f²/r  →  to hit Ra_target: f_new = f × sqrt(Ra_target/Ra_current)
        r_nose = 0.4
        if Ra > 3.2 and proc in ("turning", "milling", "drilling"):
            f_new = round(f * math.sqrt(3.2 / Ra), 3)
            tips.append(
                f"Ra {Ra:.2f} μm exceeds N7 grade (3.2 μm). "
                f"Reduce feed rate from {f:.3f} to {f_new:.3f} mm/rev "
                f"to achieve Ra ≤ 3.2 μm (formula: f_new = f·√(Ra_target/Ra_current))."
            )
        elif Ra > 1.6 and proc in ("turning", "milling"):
            f_new = round(f * math.sqrt(1.6 / Ra), 3)
            tips.append(
                f"Ra {Ra:.2f} μm is N8 grade. "
                f"For N7 finish (Ra ≤ 1.6 μm) reduce feed to {f_new:.3f} mm/rev."
            )

        # ── 2. Energy reduction ───────────────────────────────────────────
        # P = Kc·ap·f^(1-mc)·Vc / (60·η)  →  P ∝ ap × Vc
        # Most effective lever: reduce spindle speed (direct Vc effect)
        # Secondary: reduce ap (reduces Fc linearly)
        # Wrong: "increase speed" never reduces energy
        Kc1, mc, _, _, _ = MATERIALS.get(mat, MATERIALS["steel_mild"])
        if power > 1500:
            N_new  = round(N * 0.75)
            ap_new = round(ap * 0.80, 2)
            tips.append(
                f"High power ({power:.0f} W): reduce spindle speed to {N_new} RPM (−25%) "
                f"and depth of cut to {ap_new} mm (−20%). "
                f"Combined effect: ~45% power reduction (P ∝ Vc·ap)."
            )
        elif power > 800:
            N_new = round(N * 0.80)
            tips.append(
                f"Power {power:.0f} W: reduce spindle speed to {N_new} RPM (−20%) "
                f"for ~20% energy saving. Vc = π·D·N/1000; power scales linearly with Vc."
            )
        elif power > 400:
            ap_new = round(ap * 0.85, 2)
            tips.append(
                f"Power {power:.0f} W: reduce depth of cut to {ap_new} mm (−15%). "
                f"Cutting force scales linearly with ap (Fc = Kc1·ap·f^(1-mc)), "
                f"giving ~15% power reduction with minimal MRR impact if feed is increased proportionally."
            )

        # ── 3. Coolant recommendation ─────────────────────────────────────
        if not cool:
            if mat in ("titanium", "steel_stainless"):
                mat_label = _MAT_LABEL.get(mat, mat)
                tips.append(
                    f"Enable flood coolant for {mat_label}: "
                    f"reduces cutting zone temperature ~30–40°C, "
                    f"lowers tool wear rate, improves Ra by ~10%, "
                    f"and reduces machine power by ~12% (less friction heat)."
                )
            elif power > 500:
                tips.append(
                    f"Enable coolant: reduces machine power ~12% by lowering "
                    f"friction coefficient, and extends tool life."
                )

        # ── 4. Tool material upgrade ──────────────────────────────────────
        if tool == "hss" and mat in ("titanium", "steel_stainless", "cast_iron"):
            mat_label = _MAT_LABEL.get(mat, mat)
            tips.append(
                f"Upgrade from HSS to uncoated carbide for {mat_label}: "
                f"allows Vc up to 3–5× higher at same tool life (Taylor C increases ~8×), "
                f"or same Vc with dramatically longer tool life. "
                f"Specific cutting force Kc1 unchanged but productivity improves significantly."
            )
        elif tool == "carbide" and mat == "cast_iron":
            tips.append(
                "For cast iron, consider CBN inserts: "
                "allows Vc 400–600 m/min vs 150–200 m/min for carbide, "
                "reducing cycle time by ~3× at same power level."
            )

        # ── 5. Tool life warning ──────────────────────────────────────────
        if T_min < 5.0 and T_min > 0:
            tool_label = _TOOL_LABEL.get(tool, tool)
            tips.append(
                f"Tool life estimated at {T_min:.1f} min — critically short. "
                f"Reduce cutting speed by 30–40% immediately "
                f"(Taylor equation: T = (C/Vc)^(1/n); halving Vc can increase T by 5–50×). "
                f"Consider upgrading to a harder tool grade."
            )
        elif T_min < 15.0 and T_min > 0:
            tips.append(
                f"Tool life ~{T_min:.0f} min is below recommended 15–30 min threshold. "
                f"Reduce spindle speed by 15–20% to extend tool life."
            )

        # ── 6. Process-specific ───────────────────────────────────────────
        if proc == "grinding":
            # Grinding uses table feed speed (m/min), NOT mm/rev
            vw = max(f * 10.0, 1.0)   # table speed estimate
            if vw > 15:
                tips.append(
                    f"Grinding table speed ~{vw:.0f} m/min is high. "
                    f"Reduce to 8–12 m/min to lower specific grinding energy "
                    f"and prevent thermal damage (grinding burn). "
                    f"Use Malkin's model: Ra ∝ (vw/vs)^0.5."
                )

        if proc == "milling":
            ae = float(inputs.get("width_of_cut") or ap)
            if ae / D > 0.7:
                tips.append(
                    f"Radial engagement {ae/D*100:.0f}% of cutter diameter — high. "
                    f"Reduce to 50–60% by using smaller step-over. "
                    f"This lowers average chip load and cutting force by ~20–30%."
                )

        # ── Fallback if no tips generated ────────────────────────────────
        if not tips:
            tips.append(
                f"Current parameters are in an efficient operating range. "
                f"Power: {power:.0f} W, Ra: {Ra:.2f} μm, MRR: {mrr:.0f} mm³/min, "
                f"Tool life: {T_min:.0f} min. "
                f"Monitor tool wear every {max(5, int(T_min*0.5))} min and verify "
                f"actual cutting forces with a dynamometer for precise Kc1 calibration."
            )

        return " | ".join(tips)

    # ─────────────────────────────────────────────────────────────────────
    def get_info(self) -> Dict[str, Any]:
        has_key = bool(self._api_key)
        return {
            "id":   "groq_llm",
            "name": f"Groq LLM ({GROQ_LLM_MODEL})",
            "description": (
                "Physics predictions + LLaMA expert advice via Groq cloud API. "
                "Zero GPU/RAM on your machine — model runs on Groq servers. "
                f"API key: {'✓ configured' if has_key else '✗ not set — add GROQ_API_KEY or OPENAI_API_KEY to .env'}. "
                "Falls back to rule-based engineering advice if API unavailable."
            ),
            "type": "llm",
            "accuracy_metrics": {
                "Numbers": "Physics-backed (same as physics engine)",
                "Advice":  "LLM (LLaMA) or deterministic rule-based",
            },
            "supported_processes": ["turning", "milling", "drilling", "grinding"],
        }

    def is_available(self) -> bool:
        return True