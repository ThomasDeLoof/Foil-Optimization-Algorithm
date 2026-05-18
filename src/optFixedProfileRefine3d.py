# =================================================================================
# optFixedProfileRefine3d.py — Vérification et raffinement 3D du meilleur foil V2
# -------------------------------------------------------------------------------
# optFixedProfileV2 (AeroBuildup, ~0.5s/eval) trouve la meilleure planform via
# DE — rapide mais ignore les interactions 3D aile↔stab (downwash, induced drag
# exact). Ce script applique LiftingLine sur la SOLUTION DE
#
# Auto-load : par défaut on prend le x_best.npy le plus récent dans outputs/.
# Sinon : python refine_3d.py <chemin/x_best.npy>
# -------------------------------------------------------------------------------

import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np
import aerosandbox as asb
from scipy.optimize import minimize

import optFixedProfileV2 as V2          # build_airplane, constants, etc.


# ─────────────────────────────────────────────────────────────────────────────
# 1. Auto-load du dernier x_best.npy
# ─────────────────────────────────────────────────────────────────────────────
def _find_latest_x_best():
    """Cherche le x_best.npy le plus récent dans outputs/."""
    pattern = str(ROOT.parent / "outputs" / "*" / "x_best.npy")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Évaluateur 3D (LiftingLine)
# ─────────────────────────────────────────────────────────────────────────────
def aero_3d(airplane: asb.Airplane, alpha: float) -> dict:
    """Évalue L/D/Cm avec LiftingLine (capture downwash & induced drag 3D)."""
    plane_ll = asb.Airplane(
        wings=airplane.wings, xyz_ref=airplane.xyz_ref,
        s_ref=airplane.s_ref, c_ref=airplane.c_ref, b_ref=airplane.b_ref,
    )
    op = asb.OperatingPoint(velocity=V2.cfg["v_cruise"], alpha=alpha,
                            atmosphere=V2.atmosphere)
    a = asb.LiftingLine(plane_ll, op).run()
    return {k: float(a[k]) for k in ("L", "D", "Cm", "CL", "CD")}


def compare_2d_3d(x: np.ndarray, label: str = "") -> dict:
    """Affiche D_2D (AeroBuildup) vs D_3D (LiftingLine) sur le même X.
    Renvoie les valeurs 3D pour usage ultérieur."""
    p = V2.decode(np.clip(x, V2.LB, V2.UB))
    airplane, *_ = V2.build_airplane(p)

    op = asb.OperatingPoint(velocity=V2.cfg["v_cruise"], alpha=p["alpha_cruise"],
                            atmosphere=V2.atmosphere)
    ab = asb.AeroBuildup(airplane, op).run()
    ll = aero_3d(airplane, p["alpha_cruise"])

    L_2D, D_2D = float(ab["L"]), float(ab["D"])
    L_3D, D_3D = ll["L"], ll["D"]
    weight = V2.WEIGHT
    print(f"\n  {label}" if label else "")
    print(f"  {'Métrique':<20} {'AeroBuildup (2D)':>18} {'LiftingLine (3D)':>18}")
    print(f"  {'-'*58}")
    print(f"  {'L (N)':<20} {L_2D:>18.1f} {L_3D:>18.1f}")
    print(f"    écart au poids       {(L_2D-weight)/weight*100:>+15.1f}%  {(L_3D-weight)/weight*100:>+15.1f}%")
    print(f"  {'D_aéro (N)':<20} {D_2D:>18.2f} {D_3D:>18.2f}")
    print(f"  {'L/D':<20} {L_2D/D_2D:>18.2f} {L_3D/D_3D:>18.2f}")
    return {"L": L_3D, "D": D_3D, "Cm": ll["Cm"], "CL": ll["CL"]}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Refinement 3D : trim angles (planform fixée, bornes appliquées)
# ─────────────────────────────────────────────────────────────────────────────
TRIM_INDICES = [1, 2, 3, 4, 5, 6]    # cg_ratio, calage, twist, s_twist, α_to, α_cruise


def _objective_3d(x_trim: np.ndarray, x_template: np.ndarray) -> float:
    """Coût = D_3D + pénalités. Identique à V2.objective mais avec LiftingLine."""
    x = x_template.copy()
    x[TRIM_INDICES] = x_trim
    x = np.clip(x, V2.LB, V2.UB)
    p = V2.decode(x)

    try:
        airplane, wing, stab, mc, _, _ = V2.build_airplane(p)
        ll = aero_3d(airplane, p["alpha_cruise"])
        L, D, Cm = ll["L"], ll["D"], ll["Cm"]
    except Exception:
        return 1e6

    try:
        op_to = asb.OperatingPoint(velocity=V2.cfg["v_takeoff"],
                                   alpha=p["alpha_to"], atmosphere=V2.atmosphere)
        ab_to = asb.AeroBuildup(airplane, op_to).run()
        L_to, D_to, CL_to = float(ab_to["L"]), float(ab_to["D"]), float(ab_to["CL"])
    except Exception:
        return 1e6

    D_total = D + V2.D_MAST
    pen = 0.0
    pen += V2.K1 * V2.soft_penalty(L,    V2.WEIGHT, np.inf,        ref=V2.WEIGHT)
    pen += V2.K1 * V2.soft_penalty(L_to, V2.WEIGHT, np.inf,        ref=V2.WEIGHT)
    pen += V2.K1 * V2.soft_penalty(CL_to, -np.inf, V2.CL_MAX_TO,   ref=V2.CL_MAX_TO)

    X_cg = p["cg_ratio"] * mc
    M_total = (Cm * V2.q_cruise * wing.area() * mc
               + V2.M_MAST + V2.rig_mass * 9.81 * (-(X_cg - V2.x_mast)))
    M_ref = V2.WEIGHT * mc
    pen += V2.K1 * V2.soft_penalty(M_total, -0.05 * M_ref, 0.05 * M_ref, ref=M_ref)
    pen += V2.K2 * V2.soft_penalty(p["alpha_cruise"], -1.0, np.inf, ref=2.0)

    return D_total + 0.3 * D_to + pen


def refine_trim_3d(x_start: np.ndarray, maxiter: int = 80) -> tuple:
    """
    Nelder-Mead sur les 6 angles de trim, planform fixée.
    Les bornes de l'opti DE sont propagées au raffinage (scipy 1.7+ supporte
    bounds= sur Nelder-Mead via reflection/contraction adaptées).
    """
    x_start = np.clip(x_start.copy(), V2.LB, V2.UB)
    x_trim0 = x_start[TRIM_INDICES]
    trim_bounds = list(zip(V2.LB[TRIM_INDICES], V2.UB[TRIM_INDICES]))

    print(f"\n  Refinement 3D (Nelder-Mead bornes={[(f'{lo:.2g}',f'{hi:.2g}') for lo,hi in trim_bounds]}, {maxiter} iter max)...", flush=True)
    res = minimize(
        _objective_3d, x_trim0, args=(x_start,),
        method="Nelder-Mead",
        bounds=trim_bounds,
        options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-2, "disp": False},
    )
    x_final = x_start.copy()
    x_final[TRIM_INDICES] = np.clip(res.x, V2.LB[TRIM_INDICES], V2.UB[TRIM_INDICES])
    return x_final, res.fun, res.nit


# ─────────────────────────────────────────────────────────────────────────────
# 4. Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Chargement du X : argv > dernier outputs/*/x_best.npy > X_REF
    src = None
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        src = sys.argv[1]
    else:
        src = _find_latest_x_best()

    if src is not None:
        x = np.load(src)
        if len(x) != V2.N_VAR:
            print(f"  ⚠ x_best.npy a {len(x)} variables, attendu {V2.N_VAR}. "
                  f"Probablement issue d'une ancienne version du code — fallback sur X_REF.")
            x = V2.X_REF.copy()
        else:
            print(f"  ✓ Auto-load X depuis : {src}")
    else:
        x = V2.X_REF.copy()
        print(f"  ⚠ Aucun x_best.npy trouvé dans outputs/ — fallback sur X_REF.")
        print(f"    (Lance d'abord `python src/optFixedProfileV2.py` pour générer un X optimal.)")

    print(f"\n{'='*70}")
    print(f"  REFINE_3D — scénario {V2.CASE.upper()}, profil {V2.WING_AIRFOIL_NAME}")
    print(f"{'='*70}")

    # 1) Comparaison initiale (sur la solution warm-start)
    print("\n[1/3] Comparaison 2D vs 3D sur la solution chargée :")
    compare_2d_3d(x, label="(point de départ)")

    # 2) Refinement
    print("\n[2/3] Refinement 3D des angles de trim...")
    x_refined, J_refined, n_iter = refine_trim_3d(x, maxiter=80)
    p0, pr = V2.decode(x), V2.decode(x_refined)
    print(f"  Coût final : {J_refined:.2f}   ({n_iter} itérations Nelder-Mead, planform inchangée)")
    print(f"  Évolutions des angles de trim :")
    for k in ("cg_ratio", "wing_setting_angle", "twist", "s_twist", "alpha_to", "alpha_cruise"):
        delta = pr[k] - p0[k]
        flag = " *" if abs(delta) > 0.05 else ""
        print(f"    {k:<22} : {p0[k]:>7.3f}  →  {pr[k]:>7.3f}   (Δ={delta:+.3f}){flag}")

    # 3) Comparaison finale
    print("\n[3/3] Aéro 3D du foil raffiné :")
    compare_2d_3d(x_refined, label="(après refinement)")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
