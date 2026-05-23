# Technical Sheet — PUMPING freeride

*Generated on 2026-05-24 00:03:00*

---

## 1. Optimized variables (10)

| Variable | Value | Bounds |
|:---|:---|:---|
| Fuselage length | 47.4 cm | [45–75] cm |
| CG ratio | 65.0% c̄ | [5–65]% |
| Wing incidence | 0.02° | [-3.0–5.0]° |
| Twist | -3.59° | [-5.0–2.0]° |
| Stab incidence | -3.63° | [-6.0–0.0]° |
| α takeoff | 5.99° | [-2.0–12.0]° |
| Wing span | 115.4 cm | [105–135] cm |
| Wing root chord | 156 mm | [140–250] mm |
| Stab span | 35.7 cm | [25–50] cm |
| Stab root chord | 63 mm | [40–120] mm |
| α cruise (derived L=W) | 0.09° | — |

---

## 2. Flight conditions

| Parameter | Value |
|:---|:---|
| Total weight | 794.6 N (81 kg) |
| V takeoff | 3.5 m/s |
| V cruise | 6.0 m/s |
| Re root | 9.52e+05 |
| Re tip | 2.38e+05 |

---

## 3. Performance (2 flight points)

| Parameter | Cruise | Takeoff |
|:---|:---|:---|
| Speed (m/s) | 6.0 | 3.5 |
| α (°) | 0.09 | 5.99 |
| L (N) | 794.7 | 794.7 |
| D (N) | 41.30 | — |
| CL | 0.289 | 0.850 |
| CD | 0.0150 | — |
| D total (+ mast) | 48.06 | — |
| L/D ratio | 16.54 | — |

---

## 4. Geometry from opti

| Parameter | Wing | Stab |
|:---|:---|:---|
| Airfoil  | naca4412 | naca0012 |
| Area (cm²) | 1496 | 192 |
| Aspect ratio | 8.96 | 6.66 |
| Mean chord (mm) | 129 | 42 |
| Chord R/T (mm) | 156/39 | 60 / 24 |

---

## 5. Pilotability, Stability & Structure

| Parameter | Value | Target |
|:---|:---|:---|
| **ω_n** (pitch frequency, Hz) | 1.89 Hz | freeride [1.3–1.9] Hz |
| Cm_α (pitch stiffness, rad⁻¹) | -2.90 | (<0 = stable) |
| SM/l_t (scale-invariant) | 15.9% | typical aviation 10-25% |
| NP-CG gap (absolute) | 71.9 mm | — |
| SM/c̄ | 55.6% | — (chord-normalized) |
| CG | 65.0% c̄ (8.4 cm) | [5–65]% |
| Residual moment | -6.833 N·m | < 25.0 N·m (pilot trim authority) |
| Stab force | -110.0 N | (<0 = stable) |
| Stab control authority dF/dα | 35.2 N/° | ≥ 35 (target) |
| Tail volume (info) | 0.471 | — |
| Von Mises root (peak ×2.5g) | 63.9 MPa | < 160 MPa (fatigue) |
| Von Mises root (static 1g) | 25.6 MPa | < 400 MPa (rupture) |
| σ_v cavitation | 5.88 | — |
