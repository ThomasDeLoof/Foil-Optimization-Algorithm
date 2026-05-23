# Technical Sheet — WINDSURF  |  3D REFINEMENT (LiftingLine)

*Generated on 2026-05-23 18:58:31*

This foil was refined in 3D from the V2 AeroBuildup solution.
The L, D, CL, CD, L/D values below come from **LiftingLine**
(exact downwash and induced drag) — therefore more representative than V2.

---

## 0. Configuration

| Element | Airfoil | Span | Chord R/T |
|:---|:---|:---|:---|
| Wing | naca2412 | 78 cm | 141 / 35 mm |
| Stab | naca0012 | 40 cm | 76 / 30 mm |

---

## 1. Trim (refined in 3D)

| Variable | Value |
|:---|:---|
| Fuselage length | 71.3 cm |
| CG ratio | 34.8% c̄ |
| Wing incidence angle | 0.19° |
| Twist | -1.79° |
| Stab incidence angle | -0.29° |
| α takeoff | 2.44° |
| α cruise | -0.00° |

---

## 2. 3D Performance (LiftingLine)

| Parameter | Cruise (LL) | Takeoff (AB) |
|:---|:---|:---|
| V (m/s) | 13.0 | 7.0 |
| L (N) | 882.9 | 882.9 |
| D aero (N) | 58.82 | 48.35 |
| CL | 0.112 | 0.385 |
| D total (+ mast) | 90.54 | — |
| **L/D ratio** | **9.75** | — |
| L vs weight gap | -0.0% | +0.0% |

---

## 2b. Takeoff validity check (LiftingLine vs AeroBuildup)

AB and LL evaluated at the same α_to as the optimizer chose; `α_to required` is the LL trim solving L=WEIGHT at v_takeoff.

| Quantity | AB (used by optimizer) | LL (real 3D circulation) |
|:---|:---|:---|
| α_to | 2.44° | 3.58° (needed for L=W) |
| L at α_to (N) | 882.9 | 695.5 |
| CL at α_to | 0.385 | 0.303 |
| CL at L=W trim | — | 0.405 |
| Stall margin (CL_to_real / CL_max = 1.05) | — | 39% |
| Status | — | ✓ within 0.73 target |
| LL takeoff used during refine | no (AB) | — |

---

## 3. Handling, Stability & Structure

| Parameter | Value | Target |
|:---|:---|:---|
| **ω_n** (pitch frequency) | **3.42 Hz** | freeride [2.8–3.8] Hz |
| Cm_α (pitch stiffness) | -4.04 rad⁻¹ | < 0 = stable |
| SM/l_t (scale-invariant) | 15.2% | typical aviation 10-25% |
| NP-CG gap (absolute) | 105.8 mm | — |
| SM/c̄ (legacy, chord-normalized) | 90.6% | — |
| Residual moment | 4.619 N·m | < 25.0 N·m (pilot trim) |
| Tail volume V_h | 0.986 | — |
| Von Mises root (peak ×2.5g) | 60.1 MPa | < 160 MPa (fatigue) |
| Von Mises root (static 1g) | 24.0 MPa | < 400 MPa (rupture) |
