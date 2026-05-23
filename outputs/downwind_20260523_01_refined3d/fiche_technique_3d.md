# Technical Sheet — DOWNWIND  |  3D REFINEMENT (LiftingLine)

*Generated on 2026-05-23 23:52:44*

This foil was refined in 3D from the V2 AeroBuildup solution.
The L, D, CL, CD, L/D values below come from **LiftingLine**
(exact downwash and induced drag) — therefore more representative than V2.

---

## 0. Configuration

| Element | Airfoil | Span | Chord R/T |
|:---|:---|:---|:---|
| Wing | e387 | 96 cm | 124 / 31 mm |
| Stab | naca0012 | 38 cm | 60 / 24 mm |

---

## 1. Trim (refined in 3D)

| Variable | Value |
|:---|:---|
| Fuselage length | 63.2 cm |
| CG ratio | 59.3% c̄ |
| Wing incidence angle | -0.94° |
| Twist | -4.39° |
| Stab incidence angle | -1.25° |
| α takeoff | 3.38° |
| α cruise | 1.25° |

---

## 2. 3D Performance (LiftingLine)

| Parameter | Cruise (LL) | Takeoff (AB) |
|:---|:---|:---|
| V (m/s) | 10.0 | 6.0 |
| L (N) | 794.6 | 794.6 |
| D aero (N) | 49.79 | 37.22 |
| CL | 0.157 | 0.435 |
| D total (+ mast) | 68.57 | — |
| **L/D ratio** | **11.59** | — |
| L vs weight gap | +0.0% | +0.0% |

---

## 2b. Takeoff validity check (LiftingLine vs AeroBuildup)

AB and LL evaluated at the same α_to as the optimizer chose; `α_to required` is the LL trim solving L=WEIGHT at v_takeoff.

| Quantity | AB (used by optimizer) | LL (real 3D circulation) |
|:---|:---|:---|
| α_to | 3.38° | 4.39° (needed for L=W) |
| L at α_to (N) | 794.6 | 644.2 |
| CL at α_to | 0.435 | 0.353 |
| CL at L=W trim | — | 0.449 |
| Stall margin (CL_to_real / CL_max = 1.05) | — | 43% |
| Status | — | ✓ within 0.58 target |
| LL takeoff used during refine | no (AB) | — |

---

## 3. Handling, Stability & Structure

| Parameter | Value | Target |
|:---|:---|:---|
| **ω_n** (pitch frequency) | **2.85 Hz** | freeride [2.0–3.2] Hz |
| Cm_α (pitch stiffness) | -4.47 rad⁻¹ | < 0 = stable |
| SM/l_t (scale-invariant) | 14.5% | typical aviation 10-25% |
| NP-CG gap (absolute) | 89.2 mm | — |
| SM/c̄ (legacy, chord-normalized) | 86.7% | — |
| Residual moment | 1.641 N·m | < 25.0 N·m (pilot trim) |
| Tail volume V_h | 1.113 | — |
| Von Mises root (peak ×2.5g) | 126.4 MPa | < 160 MPa (fatigue) |
| Von Mises root (static 1g) | 50.6 MPa | < 400 MPa (rupture) |
