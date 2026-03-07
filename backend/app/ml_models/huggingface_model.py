"""
HUGGINGFACE API MODEL
======================
IMPORTANT: This model calls the HuggingFace Inference API.
The AI model RUNS ON HUGGINGFACE'S CLOUD SERVERS — NOT on your laptop.
Zero GPU, zero RAM, zero download. Just an HTTP request.

Setup (one-time, free):
  1. Create account at huggingface.co (free)
  2. Go to huggingface.co/settings/tokens → New token → Read
  3. Copy token (starts with hf_...)
  4. In backend/.env set: HUGGINGFACE_API_KEY=hf_your_token_here
  5. Restart backend

When you train your own model on HuggingFace:
  - Push your model to HuggingFace Hub: model.push_to_hub("your-username/machining-v1")
  - Change HF_LLM_MODEL below to "your-username/machining-v1"
  - No other code changes needed

If API key is not set or request fails → falls back to rule-based advice automatically.
Predictions (numbers) always use physics equations as backbone.
"""

import os
import json
import requests
from typing import Dict, Any
from app.ml_models.base_model import BaseMLModel
from app.ml_models.physics_model import PhysicsBasedModel


# ===================================================================
# CHANGE THIS when you have a fine-tuned model on HuggingFace Hub:
# ===================================================================
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
# HF_LLM_MODEL = "your-username/machining-optimizer-v1"   # ← your model


class HuggingFaceModel(BaseMLModel):
    """
    Numeric predictions: backed by physics equations (reliable).
    Expert text advice: Mistral-7B via HuggingFace API (cloud, free tier).
    """

    def __init__(self):
        self._physics = PhysicsBasedModel()
        self._api_key = os.getenv("HUGGINGFACE_API_KEY", "").strip()

    def _api_url(self) -> str:
        return f"https://api-inference.huggingface.co/models/{HF_LLM_MODEL}"

    def _call_llm(self, prompt: str) -> str:
        """
        Call HuggingFace Inference API.
        Request runs on HF servers — nothing downloads to your machine.
        Free tier: ~1000 requests/day for Mistral-7B.
        """
        if not self._api_key:
            return ""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 250,
                "temperature": 0.3,
                "return_full_text": False,
                "stop": ["\n\n", "###"],
            }
        }
        try:
            resp = requests.post(self._api_url(), headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "").strip()
                    return text
            elif resp.status_code == 503:
                # Model loading — first call takes ~20s
                return "[Model loading on HF servers, try again in 30s]"
            elif resp.status_code == 401:
                return "[Invalid HuggingFace API key — check your .env file]"
            return ""
        except requests.exceptions.Timeout:
            return "[HuggingFace API timeout — try again]"
        except Exception as e:
            return f"[HF API error: {e}]"

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Use physics model for all numeric predictions."""
        preds = self._physics.predict(inputs)
        preds["confidence_score"] = 0.76
        return preds

    def get_advice(self, inputs: Dict[str, Any], predictions: Dict[str, Any]) -> str:
        """
        Get expert advice from LLM if API key available,
        otherwise fall back to rule-based engineering advice.
        """
        if not self._api_key:
            return self._engineering_advice(inputs, predictions)

        prompt = f"""[INST]You are a CNC machining expert. Analyze these parameters and give 3 specific actionable recommendations to reduce energy while maintaining quality.

Process: {inputs['process_type']} | Material: {inputs['material']} | Tool: {inputs['tool_material']}
Speed: {inputs['spindle_speed']} RPM | Feed: {inputs['feed_rate']} mm/rev | DoC: {inputs['depth_of_cut']} mm | Coolant: {inputs.get('coolant_used')}
Results: Energy={predictions['energy_consumption']:.0f}W | Ra={predictions['surface_roughness']:.2f}μm | MRR={predictions['mrr']:.0f}mm³/min | Tool life est.={1/(predictions['tool_wear_rate']+1e-9):.0f}min

Give 3 numbered recommendations, each one sentence. Be specific with numbers.[/INST]"""

        response = self._call_llm(prompt)
        if response and not response.startswith("["):
            return response
        return self._engineering_advice(inputs, predictions)

    def _engineering_advice(self, inputs: Dict[str, Any], predictions: Dict[str, Any]) -> str:
        """
        Rule-based engineering advice when LLM is unavailable.
        Based on machining handbook guidelines.
        """
        tips = []
        Ra    = predictions["surface_roughness"]
        power = predictions["energy_consumption"]
        f     = inputs["feed_rate"]
        N     = inputs["spindle_speed"]
        mat   = inputs["material"]
        tool  = inputs["tool_material"]
        proc  = inputs["process_type"]
        cool  = inputs.get("coolant_used", False)

        # Surface quality guidance
        if Ra > 3.2:
            new_f = round(f * 0.75, 3)
            tips.append(f"Ra={Ra:.2f}μm exceeds N7 grade. Reduce feed to ~{new_f} mm/rev to bring Ra below 3.2μm.")
        elif Ra > 1.6:
            new_f = round(f * 0.80, 3)
            tips.append(f"Ra={Ra:.2f}μm is N8 grade. For N7 finish reduce feed to ~{new_f} mm/rev.")

        # Energy reduction
        if power > 800:
            tips.append(f"High power ({power:.0f}W): reduce depth of cut by 20% and compensate with +15% spindle speed to lower specific cutting energy.")
        elif power > 400:
            tips.append(f"Power ({power:.0f}W) is moderate. Reducing ap by 10-15% with constant MRR will improve energy efficiency.")

        # Coolant
        if not cool and mat in ["titanium", "steel_stainless"]:
            tips.append(f"For {mat}: coolant is strongly recommended — reduces cutting zone temp by ~30%, improves tool life and Ra.")
        elif not cool and power > 300:
            tips.append("Enabling coolant reduces total machine energy by ~10% and extends tool life.")

        # Tool upgrade
        if tool == "hss" and mat in ["titanium", "steel_stainless", "cast_iron"]:
            tips.append(f"Upgrade HSS to carbide for {mat}: allows +50% Vc, reduces specific cutting force, improves energy efficiency.")

        # Grinding specific
        if proc == "grinding" and f > 0.02:
            tips.append(f"Grinding feed {f} is high — reduce to 0.005–0.01 mm/rev to reduce surface burning and specific energy.")

        if not tips:
            tips.append("Parameters are within efficient operating range. Monitor tool wear periodically and recalibrate Kc values with actual force measurements for better accuracy.")

        return " | ".join(tips)

    def get_info(self) -> Dict[str, Any]:
        has_key = bool(self._api_key)
        return {
            "id":   "huggingface_llm",
            "name": f"HuggingFace API ({HF_LLM_MODEL.split('/')[-1]})",
            "description": (
                "Physics predictions + LLM expert advice via HuggingFace cloud API. "
                f"Model runs on HF servers — zero load on your machine. "
                f"API key: {'✓ configured' if has_key else '✗ not set (add HUGGINGFACE_API_KEY to .env)'}. "
                "Falls back to rule-based advice if API unavailable."
            ),
            "type": "llm",
            "accuracy_metrics": {
                "Numbers": "Physics-backed",
                "Advice":  "LLM (cloud) or rule-based",
            },
            "supported_processes": ["turning", "milling", "drilling", "grinding"],
        }

    def is_available(self) -> bool:
        return True  # always available (falls back gracefully)