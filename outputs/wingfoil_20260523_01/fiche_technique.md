# Technical Sheet — WINGFOIL freeride

*Generated on 2026-05-23 23:23:06*

---

## 1. Optimized variables (10)

| Variable | Value | Bounds |
|:---|:---|:---|
| Fuselage length | 65.2 cm | [65–95] cm |
| CG ratio | 65.0% c̄ | [5–65]% |
| Wing incidence | -0.12° | [-3.0–5.0]° |
| Twist | 0.40° | [-5.0–2.0]° |
| Stab incidence | -0.84° | [-6.0–0.0]° |
| α takeoff | 3.15° | [-2.0–12.0]° |
| Wing span | 86.0 cm | [85–120] cm |
| Wing root chord | 148 mm | [130–250] mm |
| Stab span | 30.1 cm | [25–50] cm |
| Stab root chord | 50 mm | [40–120] mm |
| α cruise (derived L=W) | -0.58° | — |

---

## 2. Flight conditions

| Parameter | Value |
|:---|:---|
| Total weight | 824.0 N (84 kg) |
| V takeoff | 5.5 m/s |
| V cruise | 9.5 m/s |
| Re root | 1.43e+06 |
| Re tip | 3.59e+05 |

---

## 3. Performance (2 flight points)

| Parameter | Cruise | Takeoff |
|:---|:---|:---|
| Speed (m/s) | 9.5 | 5.5 |
| α (°) | -0.58 | 3.15 |
| L (N) | 825.1 | 824.2 |
| D (N) | 61.20 | — |
| CL | 0.169 | 0.504 |
| CD | 0.0125 | — |
| D total (+ mast) | 78.15 | — |
| L/D ratio | 10.56 | — |

---

## 4. Geometry from opti

| Parameter | Wing | Stab |
|:---|:---|:---|
| Airfoil  | naca2412 | naca0012 |
| Area (cm²) | 1060 | 129 |
| Aspect ratio | 7.01 | 7.03 |
| Mean chord (mm) | 123 | 38 |
| Chord R/T (mm) | 148/37 | 55 / 22 |

---

## 5. Pilotability, Stability & Structure

| Parameter | Value | Target |
|:---|:---|:---|
| **ω_n** (pitch frequency, Hz) | 2.54 Hz | freeride [2.2–3.0] Hz |
| Cm_α (pitch stiffness, rad⁻¹) | -3.22 | (<0 = stable) |
| SM/l_t (scale-invariant) | 12.4% | typical aviation 10-25% |
| NP-CG gap (absolute) | 78.3 mm | — |
| SM/c̄ | 63.7% | — (chord-normalized) |
| CG | 65.0% c̄ (8.0 cm) | [5–65]% |
| Residual moment | -6.005 N·m | < 25.0 N·m (pilot trim authority) |
| Stab force | -68.2 N | (<0 = stable) |
| Stab control authority dF/dα | 59.5 N/° | ≥ 55 (target) |
| Tail volume (info) | 0.646 | — |
| Von Mises root (peak ×2.5g) | 55.1 MPa | < 160 MPa (fatigue) |
| Von Mises root (static 1g) | 22.0 MPa | < 400 MPa (rupture) |
| σ_v cavitation | 2.35 | — |
