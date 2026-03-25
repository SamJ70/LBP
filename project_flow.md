# LBP Optimization Flow

This document outlines the step-by-step logic detailing how manufacturing process optimization behaves from user input to backend calculation and back to the front-end display.

---

## 1. Step 1: User Input (Frontend `Optimizer.jsx`)

The optimization journey starts in the React frontend on the **Optimizer** page.
1.  **Selection of Context:** The user selects the fundamental context:
    *   **Process Setup:** Turning, Milling, Drilling, or Grinding.
    *   **Materials:** Workpiece (e.g., Mild Steel) and Cutting Tool (e.g., Carbide).
    *   **Coolant Configuration:** On/Off (which mathematically impacts friction and temperature abstractions).
2.  **Definition of Baseline Parameters:** The user inputs the current "baseline" operating conditions:
    *   **Spindle Speed (RPM):** Controls rotation speed.
    *   **Feed Rate (mm/rev):** Controls how fast the tool translates into the material.
    *   **Depth of Cut & Tool Diameter (mm).**
3.  **Engine Selection:** The user selects the computing engine to perform the job (e.g., **Physics / Engineering Equations**).
4.  **Submission:** The user clicks "**Run Optimization**".

The UI bundles this data into a JSON payload and transmits it via `POST /api/predict/optimize` using axios (`frontend/src/utils/api.js`).

---

## 2. Step 2: Backend Orchestration (Backend `api/predict.py`)

Upon receiving the payload, the backend API structures the run:
1.  **Validation:** FastAPI and Pydantic (`models/schemas.py`) strictly validate the `MachiningInput` to ensure no impossible strings or negative numbers proceed.
2.  **Model Instantiation:** The registry fetches the requested engine (usually `PhysicsBasedModel`).
3.  **Execution Call:** The route invokes the `.optimize(input_dict)` subroutine on the physics model.

---

## 3. Step 3: Physics-Based Optimization Engine (`physics_model.py`)

The core intelligence resides in the physics model’s `optimize` method, which acts as a deterministic, constraint-bound solver focusing on **energy minimization**.

1.  **Baseline Evaluation:** 
    *   The model evaluates the original user parameters to establish identical starting lines for Material Removal Rate ($MRR_{cur}$) and Surface Roughness ($R_{a, cur}$).
2.  **Constraint Formulation:**
    *   **Quality Constraint:** New parameters must not worsen surface roughness by more than 10% ($R_{a, max} \leq R_{a, cur} \times 1.10$).
    *   **Productivity Constraint:** New parameters must maintain at least 60% of the original throughput ($MRR \geq MRR_{cur} \times 0.60$).
    *   **Tool Life Constraint:** Uses Taylor tool life equations to determine an adaptive floor. If the current tool easily lives 30 minutes, the optimization won't degrade it below 15 minutes.
3.  **Grid Search:**
    *   The algorithm defines scaling grids. For example, it searches Spindle Speeds between $50\\%$ to $130\\%$ of original, Feed Rates between $40\\%$ to $100\\%$, and Depths of Cut between $50\\%$ to $100\\%$.
    *   The grid is mathematically biased towards reducing cutting speed ($V_c$) and depth of cut ($a_p$), as they disproportionately drive high energy outputs according to the Kienzle specific cutting force.
4.  **Selection:**
    *   It simulates every combination.
    *   It discards any state violating the $R_a$, $MRR$, or Tool Life constraints.
    *   Of the surviving valid parameter sets, it selects the singular combination that produces the **absolute lowest Wattage** ($P_{machine}$).
    *   *(Fallback)*: If no solution exists under tight constraints, it automatically drops the $MRR$ requirement and relaxes the $R_a$ constraint to $1.5\times$ prior to trying again, ensuring the system never "fails" to offer a valid engineering suggestion.

---

## 4. Step 4: Post-Processing & Insights (`predict.py`)

Once the optimal setup is found:
1.  **Delta Calculation:** The router computes savings: `(Original Energy - Optimized Energy) / Original Energy * 100`.
2.  **Recommendation Text Generation:** It strings together human-readable notes detailing exactly what was changed (e.g., `"Spindle: 800 → 650 RPM (-18.8%)"`).
3.  **Heuristic Advice Engine:** 
    *   Even if using standard physics models, the endpoint invokes the `HuggingFaceModel` (`get_advice`) purely as an expert observer.
    *   The ML model reviews the optimized numbers and appends "Expert Recommendations" (e.g., advising on tool-wear patterns or specific chip-formation risks based on the new conditions).
4.  **Response Construction:** Everything is packed into an `OptimizationResult` JSON schema and yielded back to the client.

---

## 5. Step 5: Frontend Display (`ResultsPanel.jsx`)

The newly optimized variables return to the React UI where they are rapidly charted.
1.  **Top Banner:** Highlights the total **percentage points of energy saved**, along with a flag confirming if Surface Quality was perfectly maintained or traded-off.
2.  **Comparison Grid:** Directly maps `Original` vs `Optimized` cards for Energy and Surface Roughness showing exact drop percentages.
3.  **D3 Charts (Recharts):** Renders a Before/After comparison bar chart visualizing the energy reduction alongside normalized Roughness and MRR.
4.  **Actionable List:** Prints the distinct arrows instructing the machinist exactly which dial to turn on the CNC controller to achieve these savings.
5.  **AI Expert Box:** Renders the attached heuristics or context generated in Step 4.
