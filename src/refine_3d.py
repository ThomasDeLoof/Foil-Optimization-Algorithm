# =================================================================================
# refine_3d.py — Vérification et raffinement 3D du meilleur foil V3
# -------------------------------------------------------------------------------
# V3 (AeroBuildup, ~0.5s/eval) trouve la meilleure planform via DE — rapide
# mais ignore les interactions 3D aile↔stab (downwash, induced drag exact)

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np
import aerosandbox as asb
from scipy.optimize import minimize

import V3                                # build_airplane, constants, etc.


# ─────────────────────────────────────────────────────────────────────────────
# 1. Évaluateur 3D (LiftingLine)
# ─────────────────────────────────────────────────────────────────────────────
def aero_3d(airplane: asb.Airplane, alpha: float) -> dict:
    """Évalue L/D/Cm avec LiftingLine (capture downwash & induced drag 3D)."""
    plane_ll = asb.Airplane(
        wings=airplane.wings, xyz_ref=airplane.xyz_ref,
        s_ref=airplane.s_ref, c_ref=airplane.c_ref, b_ref=airplane.b_ref,
    )
    op = asb.OperatingPoint(velocity=V3.cfg["v_cruise"], alpha=alpha,
                            atmosphere=V3.atmosphere)
    a = asb.LiftingLine(plane_ll, op).run()
    return {k: float(a[k]) for k in ("L", "D", "Cm", "CL", "CD")}


def compare_2d_3d(x: np.ndarray) -> None:
    """Affiche D_2D (AeroBuildup) vs D_3D (LiftingLine) sur le même X."""
    p = V3.decode(np.clip(x, V3.LB, V3.UB))
    airplane, *_ = V3.build_airplane(p)

    op = asb.OperatingPoint(velocity=V3.cfg["v_cruise"], alpha=p["alpha_cruise"],
                            atmosphere=V3.atmosphere)
    ab = asb.AeroBuildup(airplane, op).run()
    ll = aero_3d(airplane, p["alpha_cruise"])

    L_2D, D_2D = float(ab["L"]), float(ab["D"])
    L_3D, D_3D = ll["L"], ll["D"]
    print(f"\n  {'Métrique':<14} {'AeroBuildup (2D)':>20} {'LiftingLine (3D)':>20} {'Écart':>8}")
    print(f"  {'-'*68}")
    print(f"  {'L (N)':<14} {L_2D:>20.1f} {L_3D:>20.1f} {(L_3D-L_2D)/L_2D*100:>+7.1f}%")
    print(f"  {'D_aéro (N)':<14} {D_2D:>20.2f} {D_3D:>20.2f} {(D_3D-D_2D)/D_2D*100:>+7.1f}%")
    print(f"  {'L/D':<14} {L_2D/D_2D:>20.2f} {L_3D/D_3D:>20.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Refinement 3D : trim angles (planform fixée)
# ─────────────────────────────────────────────────────────────────────────────
# On garde la planform de la solution DE et on raffine uniquement les 6 angles
# de trim + CG avec LiftingLine comme évaluateur.

TRIM_INDICES = [1, 2, 3, 4, 5, 6]    # cg_ratio, calage, twist, s_twist, α_to, α_cruise


def _objective_3d(x_trim: np.ndarray, x_template: np.ndarray) -> float:
    """Coût = D_3D + pénalités lift/moment (mêmes que V3 mais avec LiftingLine)."""
    x = x_template.copy()
    x[TRIM_INDICES] = x_trim
    x = np.clip(x, V3.LB, V3.UB)
    p = V3.decode(x)

    try:
        airplane, wing, stab, mc, _, _ = V3.build_airplane(p)
        ll = aero_3d(airplane, p["alpha_cruise"])
        L, D, Cm, CL = ll["L"], ll["D"], ll["Cm"], ll["CL"]
    except Exception:
        return 1e6

    # Décollage à AeroBuildup (rapide, peu d'effets 3D dominants à V_takeoff)
    try:
        op_to = asb.OperatingPoint(velocity=V3.cfg["v_takeoff"],
                                   alpha=p["alpha_to"], atmosphere=V3.atmosphere)
        ab_to = asb.AeroBuildup(airplane, op_to).run()
        L_to, D_to, CL_to = float(ab_to["L"]), float(ab_to["D"]), float(ab_to["CL"])
    except Exception:
        return 1e6

    D_total = D + V3.D_MAST
    pen = 0.0
    pen += V3.K1 * V3.soft_penalty(L,    V3.WEIGHT, np.inf, ref=V3.WEIGHT)
    pen += V3.K1 * V3.soft_penalty(L_to, V3.WEIGHT, np.inf, ref=V3.WEIGHT)
    pen += V3.K1 * V3.soft_penalty(CL_to, -np.inf, V3.CL_MAX_TO, ref=V3.CL_MAX_TO)

    X_cg = p["cg_ratio"] * mc
    M_total = (Cm * V3.q_cruise * wing.area() * mc
               + V3.M_MAST + V3.rig_mass * 9.81 * (-(X_cg - V3.x_mast)))
    M_ref = V3.WEIGHT * mc
    pen += V3.K1 * V3.soft_penalty(M_total, -0.05*M_ref, 0.05*M_ref, ref=M_ref)

    pen += V3.K2 * V3.soft_penalty(p["alpha_cruise"], -1.0, np.inf, ref=2.0)

    return D_total + 0.3 * D_to + pen


def refine_trim_3d(x_start: np.ndarray, maxiter: int = 80) -> tuple:
    """Nelder-Mead sur les 6 angles de trim, planform fixée. ~1-3 min."""
    x_start = np.clip(x_start.copy(), V3.LB, V3.UB)
    x_trim0 = x_start[TRIM_INDICES]

    print(f"\n  Refinement 3D (Nelder-Mead, {maxiter} iter max)...", flush=True)
    res = minimize(
        _objective_3d, x_trim0, args=(x_start,),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-2, "disp": False},
    )
    x_final = x_start.copy()
    x_final[TRIM_INDICES] = np.clip(res.x, V3.LB[TRIM_INDICES], V3.UB[TRIM_INDICES])
    return x_final, res.fun


# ─────────────────────────────────────────────────────────────────────────────
# 3. Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Warm-start : X_REF par défaut, ou .npy fourni en argument
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        x = np.load(sys.argv[1])
        print(f"  Load X depuis : {sys.argv[1]}")
    else:
        x = V3.X_REF.copy()
        print(f"  Warm-start : X_REF (centre des bornes)")

    print(f"\n{'='*70}")
    print(f"  REFINE_3D — scénario {V3.CASE.upper()}, profil {V3.WING_AIRFOIL_NAME}")
    print(f"{'='*70}")

    # 1) Comparaison initiale
    print("\n[1/3] Comparaison 2D vs 3D sur la solution warm-start :")
    compare_2d_3d(x)

    # 2) Refinement
    print("\n[2/3] Refinement 3D des angles de trim...")
    x_refined, J_refined = refine_trim_3d(x, maxiter=80)
    p0, pr = V3.decode(x), V3.decode(x_refined)
    print(f"  Coût : {J_refined:.2f}  (planform inchangée)")
    print(f"  Évolutions :")
    for k in ("cg_ratio", "wing_setting_angle", "twist", "s_twist", "alpha_to", "alpha_cruise"):
        print(f"    {k:<20} : {p0[k]:>7.3f}  →  {pr[k]:>7.3f}")

    # 3) Comparaison finale
    print("\n[3/3] Aéro 3D du foil raffiné :")
    compare_2d_3d(x_refined)
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
