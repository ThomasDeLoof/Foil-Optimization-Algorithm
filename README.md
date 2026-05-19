# Hydrofoil Geometry Optimizer

<img width="878" height="570" alt="Foil_Cp" src="https://github.com/user-attachments/assets/98c37e5a-8283-4a17-9957-002c9ed82771" />

## What this does

The project looks for the geometry of a complete foil — main wing, fuselage, mast, stabilizer — that minimizes cruise drag for a given pilot weight and speed, under a stack of constraints: lift equals weight at cruise and takeoff, pitching moment balanced, structural stress below carbon limits, no cavitation, pilotability (pitch natural frequency `ω_n`) in a target range for the discipline. Hydrodynamics are handled by AeroSandBox with seawater as the working fluid. Everything lives in `src/`.

Three approaches have been tried in sequence. The current one is **V3**.

---

## V1 — parametric NACA with IPOPT

The first script (`optFixedProfileV1.py`) ran a structured sweep over NACA camber, thickness and sweep, with IPOPT handling the trim. It produced coherent wingfoil-class geometries but had a few persistent issues:

* The solver was boxed into tight parameter bounds. Because IPOPT is local and gradient-based, it hit these bounds constantly and either got stuck or refused to converge without manual tuning of the initial guess.
* When I loosened the bounds to help convergence, the optimizer drifted into "laboratory" designs — aspect ratios above 15, stabilizers producing positive lift, walls thinner than 1 mm — because nothing in the cost function captured structural strength or manufacturing limits.
* The aerodynamic check was global (one CL, one CD per evaluation). At freeride cruising speed (~10 m/s) the surface pressure peaks needed for cavitation analysis weren't computed, so nothing prevented the solver from picking airfoils that would ventilate in real water.

V1 was the right starting point to learn the problem, but its workflow scaled poorly.

---

## V2 — Kulfan section + structural model

The follow-up (`optMultidisciplinary.py`) moves to a Kulfan / CST parametrization of the airfoil itself rather than locking it to NACA. A handful of early-stage geometric filters reject self-intersecting profiles or trailing edges below 1 mm before the fluid solver sees them. A bending-moment + section-modulus check at the root forces realistic carbon thicknesses, and cavitation is treated as a hard constraint via `Cp_min ≥ −σ_cav`. The global solver is Differential Evolution, which copes much better than IPOPT with the messier landscape introduced by free airfoils.

V2 works, but giving the airfoil that much freedom makes the search costly and noisy. It became clear that the section itself is not where the biggest wins are at this design fidelity — the planform and the trim are.

---

## Current state

`optFixedProfileV2` keeps the physics from the previous script and goes the other way on the airfoil question. The section is **fixed per scenario** (a NACA or others chosen for each discipline), and the optimizer is free to choose the **planform** — span, root chord, tip chord — along with the trim variables: CG position, root incidence, washout, stab incidence, alpha at cruise, alpha at takeoff, fuselage length. Ten decision variables in total.

### How the geometry is built

The wing chord follows a pure ellipse, `c(r) = c_tip + (c_root − c_tip)·√(1 − r²)`, with the quarter-chord swept via a power law (`r^1.5`) plus a smoothstep "tip kick" applied only over the outer 15%. The kick amount is computed from the current chord shrinkage, so the saumon stays clearly behind the previous section whatever the dimensions the optimizer settles on. Without it, the tip section visually "tucks" forward of the penultimate section, which doesn't look like any industry foil. The stab uses the same construction with its own (smaller) kick.

Each scenario is anchored to a real industry foil whose specifications are public. The warm-start dimensions and the companion stab were chosen so that the optimizer starts from a known good design and the area target range overlaps the manufacturer's value. The tip chord is back-solved numerically so that the pure-elliptic chord law reproduces the manufacturer's stated area within 0.5 %.

| Scenario | Reference wing | Span | Area | AR | Profile | Companion stab |
|---|---|---|---|---|---|---|
| **wingfoil** | [AXIS BSC 890](https://www.mackiteboarding.com/axis-bsc-carbon-front-wing-890/) | 890 mm | 1290 cm² | 6.43 | SD7062 (also tested NACA 4412) | [AXIS Skinny 365/55](https://www.mackiteboarding.com/news/axis-skinny-vs-progressive-stabilizers/) — 365 mm / 168 cm² / AR 8.06 |
| **windsurf** | [Starboard Race 800](https://starboardfoils.com/pages/2022-race) | 800 mm | 800 cm² | 8.0 | NACA 1410 (thin, low camber — matches the Tom Speer "thin and relatively symmetrical" race profile) | Starboard Tail 255 — 400 mm / 255 cm² / AR 6.3 |
| **downwind** | [Armstrong HA 1080](https://foiloutlet.com/product/armstrong-ha1080-foil-kit/) | 1020 mm | 1080 cm² | 9.6 | Eppler 387 (thin laminar) | [Armstrong HA 195](https://foiloutlet.com/product/armstrong-ha195-high-aspect-tail-wing/) — 385 mm / 195 cm² / AR 7.6 |
| **pumping** | [Armstrong HA 1525](https://foiloutlet.com/product/armstrong-ha1525-foil-front-wing/) | 1200 mm | 1525 cm² | 9.5 | Eppler 423 (high camber for low-speed CL_max) | Armstrong HA 195 (same kit as downwind) |

A quick word on the design rationale behind each pair. The **wingfoil** BSC 890 is a moderate-aspect freeride wing, sized for forgiving takeoff at 5-6 m/s; the camber of its profile lands the cruise CL around 0.14 at 9.5 m/s, which is low but normal for the discipline. The **windsurf** Race 800 trades wing area for raw glide — half the area of the wingfoil at 60 % more speed — and uses a thin, near-symmetric section because the cruise CL drops below 0.15. The **downwind** HA 1080 is the "long, thin glider" of the family, with the highest aspect ratio of the four; its companion HA 195 stab is itself high-aspect (AR 7.6). The **pumping** HA 1525 is the biggest of the lot, sized to take off below 4 m/s, with a strongly-cambered Eppler 423 to keep CL_max usable in pumping cycles.

The airfoil loader has a small fallback because AeroSandBox's NACA generator is incomplete — most 6-series profiles (`naca63412`, `naca64412`, …) silently return empty coordinates, while the modified 6A variants like `naca64a410` work fine, as do Eppler 387/423/836-838 and the SD7037/7062 family. The loader checks the coordinate count and falls back to a sensible NACA 4-digit if the requested profile isn't usable.

### Aerodynamics

The DE evaluation uses AeroBuildup — fast (≈ 0.5 s per call), good enough to run a 250-individual population over 60 generations in roughly ten minutes. AeroBuildup is purely additive and misses the wing → stab downwash interaction. We accept that during the search loop because the relative ranking of candidate designs is preserved between AeroBuildup and a fuller solver — so the cheap one is good for picking the shape. Once the DE has converged, the best design is automatically passed to a **3D trim refinement** using LiftingLine (downwash and induced drag computed properly): planform fixed, six trim angles re-optimised with Nelder-Mead (bounded), about 80 iterations, two more minutes. Switching from AeroBuildup to LiftingLine on the same geometry typically lowers the estimated drag by 30-40 % and raises L/D from around 17 to around 24 — that's the gap between "additive 2D" and "real 3D circulation" rather than an optimization failure.

The old local L-BFGS-B polish on the DE result was dropped: gradient-based methods don't behave well on AeroBuildup's noisy stall transitions, and "polishing in 2D the wrong physics" wasted compute. The 3D refinement replaces it with something that's slower per evaluation but actually moves toward the real optimum.

The objective is multi-point: `D_cruise + 0.3 · D_takeoff`. The takeoff weight (0.3) discourages designs that need huge induced drag to lift off, without overpowering the cruise term.

### Stability and pilotability

This is the part that took the longest to figure out. The classical aircraft static margin (SM normalised by mean chord) doesn't really work for a wingfoil : the pilot's CG sits up at the board level, not in the wing chord plane, so SM/c̄ values look absurd (50-100 %) while the foil is actually well-behaved. I dropped SM/c̄ as the design constraint and replaced it with a **pilotability** target expressed as a pitch natural frequency `ω_n` (Hz). What the pilot actually feels isn't a static margin in chord units — it's the speed at which the foil responds to a pitch perturbation, i.e. the short-period mode. We compute it from `ω_n² = -Cm_α · q · S · c̄ / I_yy`, with `I_yy = m_total · r_gyr²` (gyration radius ≈ 30 cm for a rider standing on the board, set in `parameters.yaml`).

Getting an accurate `Cm_α` turned out to be the harder problem. I tried two approaches:

1. **Analytical formula + empirical de_da calibration** (`calibrate_de_da.py`, kept for reference). The textbook `Cm_α = -SM · CL_α_total` with `SM` from a vortex-horsepower formula needs an `ε = de_da` term for the downwash the stab sees. The textbook `4/(AR+2) ≈ 0.5` is wildly off for hydrofoils — the tail arm is so long that the downwash has mostly dissipated by the time it reaches the stab. So at startup, the script ran 4 VLM calls at the bounds of `fuselage_length` to invert the formula and fit `de_da(fl) = slope·fl + intercept` by linear regression. 
2. **Direct finite-difference on AeroBuildup** (current). The trim solver already calls AeroBuildup at `α = 0°` and `α = 3°` to bracket the cruise lift target. Those two calls return `Cm` too, so `dCm/dα = (Cm_hi − Cm_lo) / Δα` is free — no extra solver call, no startup VLM. AeroBuildup includes the actual wing→stab downwash (via its lifting-line strip integration), so the resulting `Cm_α` is within ~3 % of a full VLM verification, vs the 15-25 % residual error the analytical formula carried even with calibrated `de_da`.

The second approach was chosen, despite the calculation additional costs, because the precision on stability was crucial to the optimisation.

### Structure

The root cross-section is modeled as a hollow elliptic carbon shell, around 1.5 mm thick by default (`wing.skin_thickness` in `parameters.yaml`), with a polystyrene core whose contribution is neglected. Calculating and using as a constraint the structural strenght of the foil is critical to ensure the optimization doesn't create absurd shapes (like AR=20 for example).

The constraint is **fatigue**, not ultimate. Carbon-epoxy cross-ply breaks around 300 MPa, but nobody designs foils to ultimate, the structure fatigues long before it fractures. The allowable working stress is therefore set at `fatigue_allowable_ratio · σ_ult` (default 40 %, so ~120 MPa). On top of that, the loading we compute multiplies the static cruise load by a `load_peak_factor` (default 2.5×) to estimate the dynamic peak the structure sees in real conditions (wave impacts, tight turns, hard pumping). So the check is: peak von Mises stress under 2.5g loading < 120 MPa.

With this new fatigue-based criterion, AR 12 sits at the limit, AR 15 and above are infeasible : that maps closely to why real freeride wings cluster around AR 6-10 and why race wings rarely go past AR 12 even though carbon could "in theory" allow much more.

### Scripts

* `src/optFixedProfileV2.py` — main optimizer: Differential Evolution on 10 variables (planform + trim), followed automatically by 3D trim refinement (LiftingLine + Nelder-Mead with bounds). Produces the technical sheet, XFLR5 XML, and a `.dat` per wing section so XFLR5 picks the geometry up automatically.
* `src/optFixedProfileRefine3d.py` — same refinement step, but standalone: re-loads the latest `x_best.npy` and runs LiftingLine post-processing on it. Useful to re-process a saved design or compare 2D vs 3D side by side. Called automatically by V2 at the end of every run, so usually you don't need to invoke it yourself.
* `src/export_STL.py` — re-instantiates a saved `x_best.npy` at high resolution (150 spanwise × 200 chordwise sections, airfoils repaneled to 200 pts/side) and writes 4 watertight STL files (wing, stab, mast, fuselage) ready for CFD meshing or 3D-printing the moulds.
* `src/calibrate_de_da.py` — *archived*, no longer used. See the Stability section above.

### Scenarios

Four scenarios share the same code path: `wingfoil`, `windsurf`, `downwind`, `pumping`. Each one sets its own velocities, mass of the rig, area target range, allowable CG range, achievable SM range, wing airfoil, and stab dimensions. Switching scenario is a one-line edit (`case:` in `parameters.yaml`). The wingfoil case is the most extensively tuned and the one to start from.

Three pilotability levels — **débutant**, **intermédiaire**, **avancé** — and each *scenario* has its own $\omega_n$ range per level, because advanced on a $13 m/s$ windsurf race foil isn't the same physical feeling as advanced on a $6 m/s$ pumping wing. The active level is picked once globally in `parameters.yaml` (`pilotability: "intermédiaire"`), and each scenario's `pilotability_freq` table maps it to a target range in Hz:

| Scenario | débutant | intermédiaire | avancé |
|---|---|---|---|
| wingfoil | 1.5–2.2 Hz | 2.2–3.0 Hz | 3.0–4.0 Hz |
| windsurf | 2.0–2.8 Hz | 2.8–3.8 Hz | 3.8–5.5 Hz |
| downwind | 1.3–1.8 Hz | 1.8–2.5 Hz | 2.5–3.5 Hz |
| pumping | 0.8–1.3 Hz | 1.3–1.9 Hz | 1.9–2.8 Hz |

## Outputs

Run V3 yourself to get the equivalent table for any of the four scenarios — it lands in `outputs/<scenario>_v3param_<timestamp>/fiche_technique.md`.

---

## What I knowingly ignored

A few things I left out, in full awareness that they would matter for a really finished design:

* Yaw and roll stability — only longitudinal (pitch) trim is enforced.
* Variation of immersion depth and air ventilation (free-surface effects) in choppy water.
* Elastic deformation of the mast and twist of the wing under load, plus interference drag where the wing, fuselage and mast meet.
* Unsteady regimes (pumping cycle in particular). Everything is computed as if the foil were in steady cruise.

Any of these can make the output unrealistic for a specific use case. Pumping especially is poorly captured because the whole physics is unsteady, even though the geometry it produces is still useful as a starting point.

---

## Final thoughts

This isn't going to spit out a ready-to-mould foil. It's a parametric exploration tool with enough structural and hydrodynamic discipline that the outputs land in the right neighbourhood. Building it from scratch was an attempt to take seriously a problem that whole companies spend years on — it forced me into the hydrodynamics, the structural side, the numerical optimization, the way real CG and stability behave on a hydrofoil, and a lot of fiddly engineering judgement that no textbook lays out clearly. The code reflects that learning curve.

---

## Acknowledgements

This work was carried out within the ISAE Supaéro Foil club. Thanks to fellow club member Gaspard Bougnoux for his initial work, which gave me something to push against and forced me to keep looking for alternative solutions when the obvious ones didn't work.

## License

Distributed under the MIT License. See `LICENSE` for details.
