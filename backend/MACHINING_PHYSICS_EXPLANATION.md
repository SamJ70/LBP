# Machining Physics Modeling & Implementation in LBP Backend

This document details the physics-based mathematical models and their backend implementation for managing CNC machining processes (turning, milling, drilling, grinding). It serves as a definitive guide to "what" we have modeled and "how" the prediction and optimization pipelines have been constructed in the LBP application.

---

## 1. Core Physics Formulations

Our implementation relies entirely on foundational engineering physics, ensuring high predictability across different processes. All parameters are grounded in established machining literature (Boothroyd & Knight, Kienzle form, ISO 3685, Sandvik handbooks).

### 1.1 Specific Cutting Force (Kienzle)
We use the Kienzle specific cutting force framework to compute the force opposing the tool. The equation accounts for the non-linear relationship between chip thickness (represented by feed) and force.

$$ F_c = K_{c1} \times a_p \times f^{(1-m_c)} $$

**Where:**
*   $F_c$: Cutting Force in Newtons.
*   $K_{c1}$: Specific cutting pressure at $1 \text{ mm}^2$ chip section (material-dependent constant, e.g., $1800 \text{ N/mm}^2$ for mild steel).
*   $m_c$: Kienzle chip exponential factor (typical values between $0.15 - 0.28$ based on the material).
*   $a_p$: Depth of cut (chip width, mm).
*   $f$: Feed rate (chip thickness parameter, mm/rev).

### 1.2 Tool Life (Taylor's Equation)
The model leverages standard Taylor tool life formulations calibrated using Sandvik Coromant handbook points, which predict how tool life rapidly decays with faster cutting speeds.

$$ V_c \times T^n = C $$
$$ T = \left( \frac{C_{eff}}{V_c} \right)^{1/n_{eff}} $$

**Where:**
*   $V_c$: Cutting speed ($\text{m/min}$).
*   $T$: Tool Life ($\text{minutes}$).
*   $C$: Taylor constant (velocity for 1-minute life), scaled relatively between baseline materials (mild steel) and tool inserts (carbide, CBN).
*   $n$: Taylor exponent expressing the sensitivity to velocity (typically $0.25$ for carbide, adjusted via a material multiplier).
*   $C_{eff}, n_{eff}$: The effectives obtained by taking the exact material reference and augmenting it by tool type modifiers (e.g., $C \times 6$ for CBN, $n \times 1.40$).

### 1.3 Surface Roughness (Boothroyd)
The theoretical surface roughness ($R_a$) largely depends on the feed speed dragging across the surface, offset by tool corner geometries (nose radius).

$$ R_a = \frac{32 \cdot f^2}{r_{\epsilon}} \times k_{Ra} $$

**Where:**
*   $R_a$: Theoretical surface roughness ($\mu\text{m}$).
*   $r_{\epsilon}$: Tool nose radius ($\text{mm}$).
*   $k_{Ra}$: Empirical correction factor compensating for side-flow and built-up edge common to different materials (e.g., $1.30$ for Titanium).

### 1.4 Energy & Machine Power Requirements
Knowing the cutting forces generated, we can compute total operational energy draw.

$$ P_{cut} (\text{Watts}) = \frac{F_c \times V_c}{60} $$
$$ P_{machine} = \frac{P_{cut}}{\eta} $$

**Where:**
*   $\eta$: Machine drive efficiency (e.g., $75\%$ for turning, $65\%$ for drilling) scaled for each operation type under `EFF`.

---

## 2. Process Implementations

The implementation lives in the backend at `/app/ml_models/physics_model.py`. The fundamental models are extended to suit specific kinematics.

### 2.1 Turning
Translates straightforwardly using standard Kienzle formulas and Boothroyd's $R_a$. Material removal rate ($MRR$) tracks purely linear motion.

### 2.2 Milling
Milling incorporates tool-engagement calculations. 
*   **Force:** Engagement angle scales down the basic Kienzle formulation: $F_c = Kienzle \times \min(a_e/D, 1.0)$.
*   **Roughness:** Equivalent $r_{\epsilon}$ acts broader than turning models ($0.8 \text{ mm}$ assumption).
*   $MRR = a_e \times a_p \times f_z \times RPM$.

### 2.3 Drilling
Hole-making drastically changes the specific point forces.
*   **Force:** The drill point applies Kienzle along lips scaled by half the tool diameter: $F_c = 0.8 \cdot K_{c1} \cdot (D/2) \cdot f^{(1-m_c)}$.
*   **Roughness:** Hole cutting inherently scales $1.5\times$ rougher due to internal chip evacuation against the bore.

### 2.4 Grinding
Instead of regular turning principles, Grinding models specific grinding energy using Malkin's principles:
*   $R_a \propto (v_w/V_c)^{0.5} \cdot a_e^{0.4}$.
*   $Energy (u_s) \propto K_{c1} / 1000$.

---

## 3. How It Is Implemented (Backend Infrastructure)

The architecture is designed across three key files:

### **1. `app/models/schemas.py` & `app/api/processes.py`**
*   **Data Models:** Exposes strongly typed inputs (`MachiningInput`) requiring materials, tools, feed rates, spindle speeds, depth, and tool diameters.
*   **Constants Maps:** Restricts operations to `ProcessType`, `MaterialType`, and `ToolMaterial` preventing physics evaluations on impossible scenarios, accompanied by standard limits (e.g., Turning speed $100 - 4000 \text{ RPM}$).

### **2. `app/ml_models/physics_model.py`**
*   **Prediction Pipeline:** The `predict()` function calculates $V_c$, maps tool/material multipliers, calculates derived arrays ($P, MRR, R_a, F_c$, and Tool Wear), applies Coolant penalties ($\approx 12\%$ power save, $\approx 10\%$ $R_a$ finish improvement), and normalizes values to robust API contracts.
*   **Optimization Engine:** A deterministic grid-search optimizer operating within $V_c$, $f$, and $a_p$ increments.
    *   **Objective**: Minimize constraint-bound machine energy.
    *   **Constraints**:
        1.  Maintain product surface roughness: $R_a \leq R_{a, cur} \times 1.10$.
        2.  Enforce throughput floors: $MRR \geq MRR_{cur} \times 0.60$.
        3.  Keep tool life functional: Target floor scales fluidly off current performance ($T \geq 1\text{-}15\text{ mins}$).
    *   **Heuristics**: If the base constraint fails completely, it systematically relaxes to $1.5\times$ $R_a$ deviations to preserve an optimum.

### **3. `app/api/predict.py`**
*   Houses the Restful API integration logic via `POST /predict` and `POST /optimize`.
*   Connects physical model predictions to Higher-level contextual systems. Upon returning optimizations, it formulates percent changes for parameters (e.g., $+15.2\%$ RPM adjustments) and dynamically interfaces an LLM (Hugging Face / General models) inside `.get_advice()` to render the numeric optimization as human-actionable insights.
