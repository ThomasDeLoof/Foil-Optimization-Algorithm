# Technical Sheet — WINDSURF freeride

*Generated on 2026-05-23 23:36:40*

---

## 1. Optimized variables (10)

| Variable | Value | Bounds |
|:---|:---|:---|
| Fuselage length | 71.3 cm | [70–100] cm |
| CG ratio | 35.3% c̄ | [30–45]% |
| Wing incidence | 0.17° | [-3.0–5.0]° |
| Twist | -1.76° | [-5.0–2.0]° |
| Stab incidence | -0.30° | [-6.0–0.0]° |
| α takeoff | 4.07° | [-2.0–12.0]° |
| Wing span | 78.2 cm | [78–110] cm |
| Wing root chord | 141 mm | [120–200] mm |
| Stab span | 27.2 cm | [25–50] cm |
| Stab root chord | 64 mm | [40–120] mm |
| α cruise (derived L=W) | -0.72° | — |

---

## 2. Flight conditions

| Parameter | Value |
|:---|:---|
| Total weight | 882.9 N (90 kg) |
| V takeoff | 7.0 m/s |
| V cruise | 13.0 m/s |
| Re root | 1.87e+06 |
| Re tip | 4.66e+05 |

---

## 3. Performance (2 flight points)

| Parameter | Cruise | Takeoff |
|:---|:---|:---|
| Speed (m/s) | 13.0 | 7.0 |
| α (°) | -0.72 | 4.07 |
| L (N) | 883.9 | 1236.7 |
| D (N) | 99.53 | — |
| CL | 0.112 | 0.540 |
| CD | 0.0126 | — |
| D total (+ mast) | 131.26 | — |
| L/D ratio | 6.73 | — |

---

## 4. Geometry from opti

| Parameter | Wing | Stab |
|:---|:---|:---|
| Airfoil  | naca2412 | naca0012 |
| Area (cm²) | 917 | 148 |
| Aspect ratio | 6.72 | 5.00 |
| Mean chord (mm) | 117 | 53 |
| Chord R/T (mm) | 141/35 | 76 / 30 |

---

## 5. Pilotability, Stability & Structure

| Parameter | Value | Target |
|:---|:---|:---|
| **ω_n** (pitch frequency, Hz) | 4.02 Hz | freeride [2.8–3.8] Hz |
| Cm_α (pitch stiffness, rad⁻¹) | -5.61 | (<0 = stable) |
| SM/l_t (scale-invariant) | 18.6% | typical aviation 10-25% |
| NP-CG gap (absolute) | 129.2 mm | — |
| SM/c̄ | 110.6% | — (chord-normalized) |
| CG | 35.3% c̄ (4.1 cm) | [30–45]% |
| Residual moment | -10.807 N·m | < 25.0 N·m (pilot trim authority) |
| Stab force | -90.6 N | (<0 = stable) |
| Stab control authority dF/dα | 127.7 N/° | ≥ 130 (target) |
| Tail volume (info) | 0.986 | — |
| Von Mises root (peak ×2.5g) | 60.1 MPa | < 160 MPa (fatigue) |
| Von Mises root (static 1g) | 24.0 MPa | < 400 MPa (rupture) |
| σ_v cavitation | 1.25 | — |

## ⚠️ Warnings

- ⚠️ ω_n 4.02 Hz outside freeride target [2.8–3.8] Hz
