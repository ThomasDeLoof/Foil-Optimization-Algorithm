# Hydrofoil Geometry Optimizer

<img width="878" height="570" alt="Foil_Cp" src="https://github.com/user-attachments/assets/98c37e5a-8283-4a17-9957-002c9ed82771" />

## Description

This project focuses on the generation and optimization of hydrofoil profiles. The core objective is to find the mathematical optimum for a complete foil assembly (main wing, fuselage, mast, and stabilizer) using multidisciplinary design and numerical optimization.

---
## First parametric Approach
The first script relies on numerical optimization (IPOPT), and the hydrodynamic calculations were made with the AeroSandBox library (used within a "water atomsphere"). To provide a first intuitive model, I implemented the following:
* **NACA Integration:** Standard NACA profiles as a stable geometric anchor.
* **Custom Reflex Control:** Local trailing-edge deformation for pitch moment control ($M_{total} \approx 0$).
* **Integrated Drag Analysis:** Included mast and fuselage drag to refine system-wide pitching moments.
* **Static Margin Management:** Iterative convergence achieved an acceptable theoretical static margin of ~30% for the wing case, which acted as a natural stability trade-off for longitudinal trim given the chosen fuselage length.

### Limitations of this approach
The initial script focused on a heavily constrained parametric sweep. While it successfully generated some stable geometries, real-world scaling and deeper physical testing revealed major mathematical and structural flaws in this approach:
* **Over-Constraining:**
The solver was heavily boxed into tightly restricted parameter margins. The optimizer constantly fought against rigid geometric boundaries, acting as "cliffs" in the mathematical landscape. Because the boundaries were too tight, the local gradient-based solver (`IPOPT`) frequently got stuck in local minima or completely failed to converge, requiring tedious, manual "initial guess" fine-tuning.
* **Non viable outputs:**
When certain geometric bounds were relaxed to help convergence, the V1 framework created "laboratory monsters." For example too high aspect ratios, stabilizers with high positive lift, etc.
These results made me think about what was missed by the solver : mainly it didn't take into account any **structural strenght constraints** and minimum carbon thickness and other **manufacturing constraints**.
* **Ignored Hydrodynamic Reality:**
V1 evaluated global aerodynamic coefficients but completely missed localized pressure gradients and was very light on stalling constraints. For NACA profiles, I don't think that it is a big issue, but it will surely be when considering free morphing profiles in V2. I learned that even at amateur freeride cruising speeds (20 knots), minor surface imperfections or aggressive cambers create sharp local velocity peaks that can generate **extrados cavitation**, causing immediate flow ventilation, extreme sibilance, and dynamic stalling long before reaching 45 knots.

---

## Multidisciplinary Design approach 
To solve the previous issues, a complete architectural overhaul is underway for Version 2. The design philosophy shifts from a purely aerodynamic parametric tool to a true **Multidisciplinary Design Optimization (MDO)** framework built on the following pillars:
* **Quasi-Free Kulfan (CST) Parametrization:** Replacing rigid NACA profiles with Kulfan curves for root and tip sections to unlock highly innovative, high-performance shapes (e.g., precise rooftop pressure distributions and optimized reflexed cambers).
* **Low-Cost "Fast Fail" Geometric Filters:** A pre-computation script will instantly filter out self-intersecting profiles, negative thicknesses, or un-manufacturable trailing edges ($<1\text{ mm}$) before they ever reach the fluid solver, saving days of computational overhead.
* **True Structural Constraints (Bending Moment & Inertia):** V2 integrates a classical beam-theory mechanics model. The solver will compute the section modulus and area moment of inertia ($I_{xx}$) along the span. If the maximum material stress ($\sigma_{max}$) exceeds the structural limits of standard high-modulus carbon/epoxy composite ($400\text{--}500\text{ MPa}$), the individual is penalized. This will naturally force the optimizer to lower the Aspect Ratio to realistic values ($8\text{--}12$) and maintain structural thickness where it matters.
* **Local Fluid Safeguards (Non-Stalling & Cavitation Prevention):** Implementation of local Reynolds number monitoring and a critical cavitation limit boundary:
  $$Cp_{min} \ge -\sigma_{cavitation}$$
  This forces the algorithm to eliminate aggressive local pressure peaks, yielding modern, smooth-ventilation profiles with set-back master couples.
* **Global Evolutionary Solvers:** Transitioning from local gradient methods to global genetic algorithms (such as Differential Evolution or `NSGA-II`). This smoothens the mathematical landscape, accepting temporarily sub-optimal intermediate steps to ensure robust, dependable convergence toward realistic, highly stable hydrofoils.

---

## V1 Example Output (Freeride Baseline)

Below is an example of a validated design generated by the V1 script for a *Wingfoil freeride* configuration prior to structural correction:

| **WING GEOMETRY** | - |
| :--- | :--- |
| Surface Area | 1180.0 cm² |
| Wingspan | 90.0 cm |
| Aspect Ratio | 7.22 |
| Root profile | Naca1415 |
| Root Chord | 176.6 mm |
| Tip Chord | 10.0 mm |
| Twist | -1.30° |
| Wing Loading | 6650.85 N/m² |
| **STABILIZER GEOMETRY** | - |
| Surface Area | 160.0 cm² |
| Wingspan | 32.0 cm |
| Aspect Ratio | 6.43 |
| Root Chord | 73.4 mm |
| Twist | -2.74° |
| **PERFORMANCE** | - |
| Glide Ratio (L/D) | 11.84 |
| Drag | 66.30 N |
| Angle of Attack | 2.55° |
| CL Cruising | 0.147 |
| CD Cruising | 0.0125 |
| **STABILITY & BALANCE** | - |
| Static Margin | 63.36%|
| CG Position | 54.1% |
| Stability Force | -10.00 N |
| Residual Moment | 1.0030 N·m |
| Tail Volume | 0.7427 |

<img width="817" height="639" alt="Foil_Downstream" src="https://github.com/user-attachments/assets/6243c017-91a7-4880-a509-580f1d4e0084" />

<img width="828" height="852" alt="Capture d’écran 2026-05-05 à 08 23 06" src="https://github.com/user-attachments/assets/781b081f-e242-48f6-9e1c-2e7f3f9fbcc5" />

---

## Neglected physical aspects of this project
I quickly realized that it would be too complex to consider certain physical aspects in this optimization. Therefore, I intentionally ignored :
* yaw and roll stability,
* the variation of immersion depth and the air ventilation (free surface effect) that frequently occurs in agitated waters
* neglected elasticity (mast flexion and wing twisting), as well as the interference drag caused by the interaction of different foil parts
* steady-state regime (which does not account for unsteady cases like pumping).

I am well aware that this assumptions can make my code completly non realistic for certain cases.

## Final Thoughts
I am well aware that this code is not guaranteed to output a ready-to-build hydrofoil. However, building this from scratch was an attempt to tackle a massive engineering problem, one that entire companies spend years working on, and it forced me to dive deep into hydrodynamics and parametric optimization, I have learned an lot in the process.

---

## Acknowledgements

This work was carried out within the ISAE Supaéro Foil club. I would like to acknowledge fellow club member Gaspard Bougnoux for his initial work, which provided a valuable baseline for comparison and pushed me to explore alternative geometric solutions to address specific physical performance constraints.

## License

Distributed under the MIT License. See `LICENSE` for more information.
