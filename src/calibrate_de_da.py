# =================================================================================
# Calibration empirique de de_da (downwash effectif aile→stab)
# -------------------------------------------------------------------------------
# Pour un hydrofoil avec l_t/c̄ ≈ 5, la formule classique 4/(AR+2) ≈ 0.5
# surestime fortement le downwash. On l'inverse numériquement à partir de
# SM_VLM (4 appels VLM aux bornes de fuselage_length).
# -------------------------------------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np
import aerosandbox as asb


# Pure math — dupliqué ici pour rendre ce module indépendant de V2.
def _cl_alpha_helmbold(AR: float) -> float:
    a0 = 2.0 * np.pi
    return a0 / (np.sqrt(1.0 + (a0 / (np.pi * AR)) ** 2) + a0 / (np.pi * AR))


def _measure_de_da_at(ctx: dict, fl: float) -> tuple:
    """
    Mesure de_da empirique à un fuselage_length donné via VLM.

    `ctx` doit contenir :
        - "build_airplane"          : callable(p_dict) -> (airplane, wing, stab, mc, _, _)
        - "init_p_neutral"          : dict des params planform/trim figés pour la mesure
                                      (cg_ratio, wing_*, stab_*, s_twist, calage, alpha_to)
        - "atmosphere"              : asb.Atmosphere
        - "v_cruise"                : float (m/s)
        - "x_fuselage_start"        : float (m)
        - "stab_fuse_offset"        : float (m)
    """
    p = {**ctx["init_p_neutral"], "fuselage_length": fl}
    airplane, wing, stab, mc, _, _ = ctx["build_airplane"](p)

    plane_vlm = asb.Airplane(
        wings=airplane.wings, xyz_ref=airplane.xyz_ref,
        s_ref=airplane.s_ref, c_ref=airplane.c_ref, b_ref=airplane.b_ref,
    )
    op_p = asb.OperatingPoint(velocity=ctx["v_cruise"], alpha=3.25, atmosphere=ctx["atmosphere"])
    op_m = asb.OperatingPoint(velocity=ctx["v_cruise"], alpha=2.75, atmosphere=ctx["atmosphere"])
    a_p = asb.VortexLatticeMethod(plane_vlm, op_p).run()
    a_m = asb.VortexLatticeMethod(plane_vlm, op_m).run()
    sm_vlm = -(float(a_p["Cm"]) - float(a_m["Cm"])) / (float(a_p["CL"]) - float(a_m["CL"]))

    # Inversion : SM_VLM = X_ac_w/c + V_H × (CL_a_s/CL_a_w) × (1 − de_da) − X_cg/c
    AR_w = float(wing.aspect_ratio())
    AR_s = float(stab.aspect_ratio())
    CL_a_w = _cl_alpha_helmbold(AR_w)
    CL_a_s = _cl_alpha_helmbold(AR_s)
    X_ac_w   = 0.25 * mc
    c_s_mean = 0.5 * (p["stab_root_chord"] + p["stab_tip_chord"])
    X_ac_s   = (ctx["x_fuselage_start"] + fl - ctx["stab_fuse_offset"]) + 0.25 * c_s_mean
    V_H      = (stab.area() * (X_ac_s - X_ac_w)) / (wing.area() * mc)
    X_cg     = p["cg_ratio"] * mc

    one_minus = (sm_vlm + X_cg / mc - X_ac_w / mc) / (V_H * CL_a_s / CL_a_w)
    return 1.0 - one_minus, sm_vlm, V_H


def calibrate(ctx: dict, verbose: bool = True) -> tuple:
    """
    Recalibre (DE_DA_SLOPE, DE_DA_INTERCEPT) via 4 appels VLM aux bornes de
    fuselage_length + régression linéaire. Self-contained — utilise UNIQUEMENT
    ctx (aucune dépendance globale). Retourne (slope, intercept).

    `ctx` doit contenir tout ce qu'attend _measure_de_da_at, plus :
        - "fuselage_length_bounds"  : (fl_lo, fl_hi)
    """
    fl_lo, fl_hi = ctx["fuselage_length_bounds"]
    if verbose:
        print(f"  Calibration de_da via VLM (4 appels, ~5s)...", flush=True)

    d_lo, _, _ = _measure_de_da_at(ctx, fl_lo)
    d_hi, _, _ = _measure_de_da_at(ctx, fl_hi)
    slope     = (d_hi - d_lo) / (fl_hi - fl_lo)
    intercept = d_lo - slope * fl_lo

    if verbose:
        print(f"    de_da({fl_lo*100:.0f}cm)={d_lo:+.3f}   "
              f"de_da({fl_hi*100:.0f}cm)={d_hi:+.3f}")
        print(f"    SLOPE     = {slope:+.3f}")
        print(f"    INTERCEPT = {intercept:+.3f}")

    return slope, intercept