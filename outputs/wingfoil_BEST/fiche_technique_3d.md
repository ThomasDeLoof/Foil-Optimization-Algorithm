# Technical Sheet — WINGFOIL  |  3D REFINEMENT (LiftingLine)

*Generated on 2026-05-23 18:44:27*

This foil was refined in 3D from the V2 AeroBuildup solution.
The L, D, CL, CD, L/D values below come from **LiftingLine**
(exact downwash and induced drag) — therefore more representative than V2.

---

## 0. Configuration

| Element | Airfoil | Span | Chord R/T |
|:---|:---|:---|:---|
| Wing | naca2412 | 86 cm | 148 / 37 mm |
| Stab | naca0012 | 36 cm | 55 / 22 mm |

---

## 1. Trim (refined in 3D)

| Variable | Value |
|:---|:---|
| Fuselage length | 65.2 cm |
| CG ratio | 65.0% c̄ |
| Wing incidence angle | -0.13° |
| Twist | 0.34° |
| Stab incidence angle | -0.78° |
| α takeoff | 3.17° |
| α cruise | 0.28° |

---

## 2. 3D Performance (LiftingLine)

| Parameter | Cruise (LL) | Takeoff (AB) |
|:---|:---|:---|
| V (m/s) | 9.5 | 5.5 |
| L (N) | 824.0 | 824.2 |
| D aero (N) | 39.52 | 43.30 |
| CL | 0.169 | 0.504 |
| D total (+ mast) | 56.46 | — |
| **L/D ratio** | **14.59** | — |
| L vs weight gap | +0.0% | +0.0% |

---

## 2b. Takeoff validity check (LiftingLine vs AeroBuildup)

AB and LL evaluated at the same α_to as the optimizer chose; `α_to required` is the LL trim solving L=WEIGHT at v_takeoff.

| Quantity | AB (used by optimizer) | LL (real 3D circulation) |
|:---|:---|:---|
| α_to | 3.17° | 4.71° (needed for L=W) |
| L at α_to (N) | 824.2 | 678.9 |
| CL at α_to | 0.504 | 0.415 |
| CL at L=W trim | — | 0.535 |
| Stall margin (CL_to_real / CL_max = 1.05) | — | 51% |
| Status | — | ✓ within 0.58 target |
| LL takeoff used during refine | no (AB) | — |

---

## 3. Handling, Stability & Structure

| Parameter | Value | Target |
|:---|:---|:---|
| **ω_n** (pitch frequency) | **2.10 Hz** | freeride [2.2–3.0] Hz |
| Cm_α (pitch stiffness) | -2.20 rad⁻¹ | < 0 = stable |
| SM/l_t (scale-invariant) | 9.4% | typical aviation 10-25% |
| NP-CG gap (absolute) | 59.4 mm | — |
| SM/c̄ (legacy, chord-normalized) | 48.3% | — |
| Residual moment | -5.037 N·m | < 25.0 N·m (pilot trim) |
| Tail volume V_h | 0.646 | — |
| Von Mises root (peak ×2.5g) | 55.1 MPa | < 160 MPa (fatigue) |
| Von Mises root (static 1g) | 22.0 MPa | < 400 MPa (rupture) |
