# Hydrofoil Geometry Optimizer

<img width="878" height="570" alt="Foil_Cp" src="https://github.com/user-attachments/assets/98c37e5a-8283-4a17-9957-002c9ed82771" />

## What this does

The project looks for the geometry of a complete foil — main wing, fuselage, mast, stabilizer — that minimizes cruise drag for a given pilot weight and speed, under a stack of constraints: lift equals weight at cruise and takeoff, pitching moment balanced, structural stress below carbon limits, no cavitation, static margin in a sensible range. Hydrodynamics are handled by AeroSandBox with seawater as the working fluid. Everything lives in `src/`.

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

The DE evaluation uses AeroBuildup — fast (≈ 0.5 s per call), good enough to run a 250-individual population over 60 generations in roughly ten minutes. AeroBuildup is purely additive and misses the wing → stab downwash interaction. We accept that during the search loop. The final design is checked with LiftingLine in a separate script (`refine_3d.py`), which captures induced drag and downwash properly. On the same geometry, switching from AeroBuildup to LiftingLine typically lowers the estimated drag by 30-40% and raises L/D from around 17 to around 24. What matters is that the relative ranking of candidate designs is preserved between the two — so the cheap solver picks the right shape and the expensive one gives the right numbers.

The objective is multi-point: `D_cruise + 0.3 · D_takeoff`. The takeoff weight (0.3) discourages designs that need huge induced drag to lift off, without overpowering the cruise term.

### Static margin

This is the part that took the longest to figure out. The classical aircraft static margin doesn't really apply to a wingfoil. The dominant mass of the system — pilot + board + rig, roughly 84 kg out of 85 — sits on the board above the mast, which is well behind the wing in chord units. Compute SM about that *physical* CG and you get something around −80 %: the foil is statically unstable in the classical sense, exactly like a modern fighter, and the pilot stabilises it dynamically with their stance.

The SM that's actually optimized in V3 is a proxy. It assumes the CG sits at some fraction of the wing's mean chord (controlled by `cg_ratio`) and captures the geometric relationship between the wing AC, the stab contribution and that assumed CG position. The range we constrain it to in `scenarios.yaml` (40-75 % for wingfoil, 50-80 % for windsurf, etc.) reflects what's achievable under that proxy. Calling it 60 % doesn't mean the real-life foil is 60 % stable — it means the proxy puts it there.

The downwash factor `de_da` used in the analytical formula is calibrated empirically against VLM by `calibrate_de_da.py`. The textbook `4 / (AR + 2) ≈ 0.5` value is borrowed from aircraft and is off by a factor of five for a hydrofoil, because `l_t / c̄ ≈ 5` means the downwash has mostly dissipated by the time it reaches the stab. The calibrated linear fit `de_da(fl) ≈ 0.54·fl − 0.43` brings the analytical SM to within 2 percentage points of the VLM measurement across the whole fuselage-length range. Re-run the calibration script if you change the wing or stab dimensions.

### Structure

The root cross-section is modeled as a hollow elliptic carbon shell, 1.5 mm thick by default (`wing.skin_thickness` in `parameters.yaml`), with a polystyrene core whose contribution is neglected. Bending uses the difference of two elliptic second moments (outer minus inner ellipse); torsion uses Bredt's formula for thin-walled closed sections, which is the right one for a shell — much more punishing than the solid-ellipse `J` it replaced. Von Mises at the root is then checked against 300 MPa, a conservative figure for high-modulus carbon/epoxy.

### Scripts

* `src/optFixedProfileV2.py` — main optimizer (Differential Evolution + L-BFGS-B polish), reports, XFLR5 XML export, plus a `.dat` for every airfoil used so XFLR5 picks them up automatically when you open the plane file
* `src/calibrate_de_da.py` — one-shot empirical calibration of `de_da` against VLM, prints the slope and intercept to paste into V3
* `src/optFixedProfileRefine3d.py` — runs LiftingLine on the DE solution and does a small Nelder-Mead refinement of the trim angles while keeping the planform fixed

The intended workflow:

```bash
python src/optFixedProfileV2.py                  # ~10 min — full DE on planform + trim
python src/calibrate_de_da.py     # ~5 s — re-run if wing/stab dimensions change
python src/optFixedProfileRefine3d.py           # ~2 min — 3D refinement of trim
```

### Scenarios

Four scenarios share the same code path: `wingfoil`, `windsurf`, `downwind`, `pumping`. Each one sets its own velocities, mass of the rig, area target range, allowable CG range, achievable SM range, wing airfoil, and stab dimensions. Switching scenario is a one-line edit (`case:` in `parameters.yaml`). The wingfoil case is the most extensively tuned and the one to start from.

---

## V1 example output (kept for reference)

For comparison, here is a wingfoil-freeride design generated by V1 before the structural correction was added:

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
| Static Margin | 63.36% |
| CG Position | 54.1% |
| Stability Force | -10.00 N |
| Residual Moment | 1.0030 N·m |
| Tail Volume | 0.7427 |

<img width="817" height="639" alt="Foil_Downstream" src="https://github.com/user-attachments/assets/6243c017-91a7-4880-a509-580f1d4e0084" />

<img width="828" height="852" alt="Capture d’écran 2026-05-05 à 08 23 06" src="https://github.com/user-attachments/assets/781b081f-e242-48f6-9e1c-2e7f3f9fbcc5" />

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
