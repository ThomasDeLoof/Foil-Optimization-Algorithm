# =================================================================================
# Calibration empirique de de_da (downwash effectif aile→stab)
# -------------------------------------------------------------------------------
# Pour un hydrofoil avec l_t/c̄ ≈ 5, la formule classique de_da = 4/(AR+2) ≈ 0.5
# surestime fortement le downwash (déjà dissipé au stab). On le mesure par VLM
# et on ajuste une régression linéaire de_da(fuselage_length) = a·fl + b.
#
# on inverse la formule de SM analytique d'Abzug à partir de SM_VLM mesurée.
# -------------------------------------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np
import aerosandbox as asb

import V3                                # réutilise build_airplane, constants, etc.


def _de_da_at(fl: float, p_neutral: dict) -> tuple:
    """Mesure de_da empirique à une longueur de fuselage donnée."""
    p = {**p_neutral, "fuselage_length": fl}
    airplane, wing, stab, mc, _, _ = V3.build_airplane(p)

    plane_vlm = asb.Airplane(
        wings=airplane.wings, xyz_ref=airplane.xyz_ref,
        s_ref=airplane.s_ref, c_ref=airplane.c_ref, b_ref=airplane.b_ref,
    )
    op_p = asb.OperatingPoint(velocity=V3.cfg["v_cruise"], alpha=3.25, atmosphere=V3.atmosphere)
    op_m = asb.OperatingPoint(velocity=V3.cfg["v_cruise"], alpha=2.75, atmosphere=V3.atmosphere)
    a_p = asb.VortexLatticeMethod(plane_vlm, op_p).run()
    a_m = asb.VortexLatticeMethod(plane_vlm, op_m).run()
    dCL = float(a_p["CL"]) - float(a_m["CL"])
    sm_vlm = -(float(a_p["Cm"]) - float(a_m["Cm"])) / dCL

    # Inversion de la formule analytique → de_da
    AR_w, AR_s = float(wing.aspect_ratio()), float(stab.aspect_ratio())
    CL_a_w = V3._cl_alpha_helmbold(AR_w)
    CL_a_s = V3._cl_alpha_helmbold(AR_s)
    X_ac_w = 0.25 * mc
    c_s_mean = 0.5 * (V3.STAB_ROOT_CHORD + V3.STAB_TIP_CHORD)
    X_ac_s = (V3.x_fuselage_start + fl - V3.STAB_FUSE_OFFSET) + 0.25 * c_s_mean
    V_H = (stab.area() * (X_ac_s - X_ac_w)) / (wing.area() * mc)
    X_cg = p["cg_ratio"] * mc

    one_minus = (sm_vlm + X_cg / mc - X_ac_w / mc) / (V_H * CL_a_s / CL_a_w)
    return 1.0 - one_minus, sm_vlm, V_H


def main() -> None:
    fl_lo, fl_hi = V3.phy["fuselage"]["length_bounds"]

    # Config "neutre" — cg, calage, twist, α n'affectent pas de_da (vérifié).
    p_neutral = {"cg_ratio": 0.38, "wing_setting_angle": 0.0, "twist": -1.0,
                 "s_twist": -2.0, "alpha_to": 7.0, "alpha_cruise": 3.0}

    print(f"\n{'='*65}")
    print(f"  Calibration de_da — scénario : {V3.CASE.upper()}")
    print(f"  Aile : {V3.WING_AIRFOIL_NAME}  b={V3.WING_SPAN*100:.0f} cm  "
          f"c_R/T={V3.WING_ROOT_CHORD*1000:.0f}/{V3.WING_TIP_CHORD*1000:.0f} mm")
    print(f"  Stab : {V3.STAB_AIRFOIL_NAME}  b={V3.STAB_SPAN*100:.0f} cm  "
          f"c_R/T={V3.STAB_ROOT_CHORD*1000:.0f}/{V3.STAB_TIP_CHORD*1000:.0f} mm")
    print(f"  Bornes fuselage : [{fl_lo*100:.0f}, {fl_hi*100:.0f}] cm")
    print(f"{'='*65}")

    print(f"\n  Mesures VLM (4 appels)...", flush=True)
    d_lo, sm_lo, V_H_lo = _de_da_at(fl_lo, p_neutral)
    d_hi, sm_hi, V_H_hi = _de_da_at(fl_hi, p_neutral)

    slope     = (d_hi - d_lo) / (fl_hi - fl_lo)
    intercept = d_lo - slope * fl_lo

    print(f"\n  {'fl (cm)':>8} {'V_H':>6} {'SM_VLM':>8} {'de_da':>8}")
    print(f"  {'-'*36}")
    print(f"  {fl_lo*100:>8.0f} {V_H_lo:>6.2f} {sm_lo*100:>7.1f}% {d_lo:>+8.3f}")
    print(f"  {fl_hi*100:>8.0f} {V_H_hi:>6.2f} {sm_hi*100:>7.1f}% {d_hi:>+8.3f}")

    # Comparaison avec la formule classique
    AR_w_nom = V3.WING_SPAN ** 2 / (0.5 * (V3.WING_ROOT_CHORD + V3.WING_TIP_CHORD) * V3.WING_SPAN)
    de_da_classic = 4.0 / (AR_w_nom + 2.0)
    print(f"\n  Formule classique 4/(AR+2) = {de_da_classic:.3f}  (sur-estimée pour hydrofoils)")

    print(f"\n  ▶ Résultat Régression linéaire :")
    print(f"      DE_DA_SLOPE     = {slope:+.3f}")
    print(f"      DE_DA_INTERCEPT = {intercept:+.3f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
