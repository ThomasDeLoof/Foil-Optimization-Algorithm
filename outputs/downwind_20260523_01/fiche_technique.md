# Technical Sheet — DOWNWIND freeride

*Generated on 2026-05-23 23:50:03*

---

## 1. Optimized variables (10)

| Variable | Value | Bounds |
|:---|:---|:---|
| Fuselage length | 63.2 cm | [60–90] cm |
| CG ratio | 57.8% c̄ | [5–65]% |
| Wing incidence | -0.94° | [-3.0–5.0]° |
| Twist | -4.39° | [-5.0–2.0]° |
| Stab incidence | -1.25° | [-6.0–0.0]° |
| α takeoff | 3.38° | [-2.0–12.0]° |
| Wing span | 96.3 cm | [95–125] cm |
| Wing root chord | 124 mm | [110–220] mm |
| Stab span | 34.3 cm | [25–50] cm |
| Stab root chord | 62 mm | [40–120] mm |
| α cruise (derived L=W) | 0.46° | — |

---

## 2. Flight conditions

| Parameter | Value |
|:---|:---|
| Total weight | 794.6 N (81 kg) |
| V takeoff | 6.0 m/s |
| V cruise | 10.0 m/s |
| Re root | 1.26e+06 |
| Re tip | 3.16e+05 |

---

## 3. Performance (2 flight points)

| Parameter | Cruise | Takeoff |
|:---|:---|:---|
| Speed (m/s) | 10.0 | 6.0 |
| α (°) | 0.46 | 3.38 |
| L (N) | 794.9 | 794.6 |
| D (N) | 71.35 | — |
| CL | 0.157 | 0.435 |
| CD | 0.0141 | — |
| D total (+ mast) | 90.13 | — |
| L/D ratio | 8.82 | — |

---

## 4. Geometry from opti

| Parameter | Wing | Stab |
|:---|:---|:---|
| Airfoil  | e387 | naca0012 |
| Area (cm²) | 994 | 180 |
| Aspect ratio | 9.39 | 6.56 |
| Mean chord (mm) | 103 | 42 |
| Chord R/T (mm) | 124/31 | 60 / 24 |

---

## 5. Pilotability, Stability & Structure

| Parameter | Value | Target |
|:---|:---|:---|
| **ω_n** (pitch frequency, Hz) | 3.28 Hz | freeride [2.0–3.2] Hz |
| Cm_α (pitch stiffness, rad⁻¹) | -5.94 | (<0 = stable) |
| SM/l_t (scale-invariant) | 18.3% | typical aviation 10-25% |
| NP-CG gap (absolute) | 112.7 mm | — |
| SM/c̄ | 109.5% | — (chord-normalized) |
| CG | 57.8% c̄ (5.9 cm) | [5–65]% |
| Residual moment | -18.977 N·m | < 25.0 N·m (pilot trim authority) |
| Stab force | -55.3 N | (<0 = stable) |
| Stab control authority dF/dα | 91.9 N/° | ≥ 90 (target) |
| Tail volume (info) | 1.113 | — |
| Von Mises root (peak ×2.5g) | 126.4 MPa | < 160 MPa (fatigue) |
| Von Mises root (static 1g) | 50.6 MPa | < 400 MPa (rupture) |
| σ_v cavitation | 2.12 | — |

## ⚠️ Warnings

- ⚠️ ω_n 3.28 Hz outside freeride target [2.0–3.2] Hz
