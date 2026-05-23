# Technical Sheet — PUMPING  |  3D REFINEMENT (LiftingLine)

*Generated on 2026-05-24 00:05:27*

This foil was refined in 3D from the V2 AeroBuildup solution.
The L, D, CL, CD, L/D values below come from **LiftingLine**
(exact downwash and induced drag) — therefore more representative than V2.

---

## 0. Configuration

| Element | Airfoil | Span | Chord R/T |
|:---|:---|:---|:---|
| Wing | naca4412 | 115 cm | 156 / 39 mm |
| Stab | naca0012 | 38 cm | 60 / 24 mm |

---

## 1. Trim (refined in 3D)

| Variable | Value |
|:---|:---|
| Fuselage length | 47.4 cm |
| CG ratio | 60.3% c̄ |
| Wing incidence angle | 0.02° |
| Twist | -3.59° |
| Stab incidence angle | -3.63° |
| α takeoff | 5.99° |
| α cruise | 1.31° |

---

## 2. 3D Performance (LiftingLine)

| Parameter | Cruise (LL) | Takeoff (AB) |
|:---|:---|:---|
| V (m/s) | 6.0 | 3.5 |
| L (N) | 794.6 | 794.6 |
| D aero (N) | 32.84 | 41.47 |
| CL | 0.289 | 0.850 |
| D total (+ mast) | 39.60 | — |
| **L/D ratio** | **20.07** | — |
| L vs weight gap | -0.0% | +0.0% |

---

## 2b. Takeoff validity check (LiftingLine vs AeroBuildup)

AB and LL evaluated at the same α_to as the optimizer chose; `α_to required` is the LL trim solving L=WEIGHT at v_takeoff.

| Quantity | AB (used by optimizer) | LL (real 3D circulation) |
|:---|:---|:---|
| α_to | 5.99° | 8.21° (needed for L=W) |
| L at α_to (N) | 794.6 | 643.5 |
| CL at α_to | 0.850 | 0.688 |
| CL at L=W trim | — | 0.846 |
| Stall margin (CL_to_real / CL_max = 1.05) | — | 81% |
| Status | — | ✓ within 0.89 target |
| LL takeoff used during refine | no (AB) | — |

---

## 3. Handling, Stability & Structure

| Parameter | Value | Target |
|:---|:---|:---|
| **ω_n** (pitch frequency) | **1.76 Hz** | freeride [1.3–1.9] Hz |
| Cm_α (pitch stiffness) | -2.52 rad⁻¹ | < 0 = stable |
| SM/l_t (scale-invariant) | 14.4% | typical aviation 10-25% |
| NP-CG gap (absolute) | 65.4 mm | — |
| SM/c̄ (legacy, chord-normalized) | 50.6% | — |
| Residual moment | -3.419 N·m | < 25.0 N·m (pilot trim) |
| Tail volume V_h | 0.471 | — |
| Von Mises root (peak ×2.5g) | 63.9 MPa | < 160 MPa (fatigue) |
| Von Mises root (static 1g) | 25.6 MPa | < 400 MPa (rupture) |
