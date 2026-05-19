# =================================================================================
# Calibration empirique de de_da (downwash effectif aile→stab)
# -------------------------------------------------------------------------------
# Pour un hydrofoil avec l_t/c̄ ≈ 5, la formule classique 4/(AR+2) ≈ 0.5
# surestime fortement le downwash (déjà dissipé au stab). On l'inverse
# numériquement à partir de SM_VLM (4 appels VLM aux bornes de fuselage_length).
# -------------------------------------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import aerosandbox as asb

import optFixedProfileV2 as V2         

def _measure_de_da_at(fl: float, p_neutral: dict) -> tuple:
    """
    Mesure de_da empirique à un fuselage_length donné via VLM.
    Retourne (de_da, SM_VLM, V_H).
    """
    p = {**p_neutral, "fuselage_length": fl}
    airplane, wing, stab, mc, _, _ = V2.build_airplane(p)

    plane_vlm = asb.Airplane(
        wings=airplane.wings, xyz_ref=airplane.xyz_ref,
        s_ref=airplane.s_ref, c_ref=airplane.c_ref, b_ref=airplane.b_ref,
    )
    op_p = asb.OperatingPoint(velocity=V2.cfg["v_cruise"], alpha=3.25, atmosphere=V2.atmosphere)
    op_m = asb.OperatingPoint(velocity=V2.cfg["v_cruise"], alpha=2.75, atmosphere=V2.atmosphere)
    a_p = asb.VortexLatticeMethod(plane_vlm, op_p).run()
    a_m = asb.VortexLatticeMethod(plane_vlm, op_m).run()
    sm_vlm = -(float(a_p["Cm"]) - float(a_m["Cm"])) / (float(a_p["CL"]) - float(a_m["CL"]))

    # Inversion : SM = X_ac_w/c + V_H × (CL_a_s/CL_a_w) × (1 − de_da) − X_cg/c
    AR_w = float(wing.aspect_ratio())
    AR_s = float(stab.aspect_ratio())
    CL_a_w = V2._cl_alpha_helmbold(AR_w)
    CL_a_s = V2._cl_alpha_helmbold(AR_s)
    X_ac_w   = 0.25 * mc
    c_s_mean = 0.5 * (V2.STAB_ROOT_CHORD + V2.STAB_TIP_CHORD)
    X_ac_s   = (V2.x_fuselage_start + fl - V2.STAB_FUSE_OFFSET) + 0.25 * c_s_mean
    V_H      = (stab.area() * (X_ac_s - X_ac_w)) / (wing.area() * mc)
    X_cg     = p["cg_ratio"] * mc
    one_minus = (sm_vlm + X_cg / mc - X_ac_w / mc) / (V_H * CL_a_s / CL_a_w)
    return 1.0 - one_minus, sm_vlm, V_H


def calibrate(verbose: bool = True) -> tuple:
    """
    Recalibre V2.DE_DA_SLOPE et V2.DE_DA_INTERCEPT via 4 appels VLM aux bornes
    de fuselage_length, puis régression linéaire.
    """
    fl_lo, fl_hi = V2.phy["fuselage"]["length_bounds"]
    p_neutral = {
        "cg_ratio":           0.40,
        "wing_setting_angle": 0.0,
        "twist":              -1.0,
        "s_twist":            -2.0,
        "alpha_to":            7.0,
        "alpha_cruise":        3.0,
        "wing_span":          V2.WING_SPAN,
        "wing_root_chord":    V2.WING_ROOT_CHORD,
        "wing_tip_chord":     V2.WING_TIP_CHORD,
    }
    if verbose:
        print(f"  Calibration de_da via VLM (4 appels, ~3s)...", flush=True)

    d_lo, _, _ = _measure_de_da_at(fl_lo, p_neutral)
    d_hi, _, _ = _measure_de_da_at(fl_hi, p_neutral)
    slope     = (d_hi - d_lo) / (fl_hi - fl_lo)
    intercept = d_lo - slope * fl_lo

    if verbose:
        old_s, old_i = V2.DE_DA_SLOPE, V2.DE_DA_INTERCEPT
        print(f"    de_da({fl_lo*100:.0f}cm)={d_lo:+.3f}   "
              f"de_da({fl_hi*100:.0f}cm)={d_hi:+.3f}")
        print(f"    SLOPE     = {slope:+.3f}   (ancien : {old_s:+.3f},  Δ {slope-old_s:+.3f})")
        print(f"    INTERCEPT = {intercept:+.3f}   (ancien : {old_i:+.3f},  Δ {intercept-old_i:+.3f})")

    V2.DE_DA_SLOPE     = slope
    V2.DE_DA_INTERCEPT = intercept
    return slope, intercept


def main() -> None:
    """Standalone CLI : calibre et affiche le résultat avec contexte."""
    fl_lo, fl_hi = V2.phy["fuselage"]["length_bounds"]
    AR_w_nom = V2.WING_SPAN ** 2 / (0.5 * (V2.WING_ROOT_CHORD + V2.WING_TIP_CHORD) * V2.WING_SPAN)

    print(f"\n{'='*65}")
    print(f"  Calibration de_da — scénario : {V2.CASE.upper()}")
    print(f"  Aile : {V2.WING_AIRFOIL_NAME}  b={V2.WING_SPAN*100:.0f} cm  "
          f"c_R/T={V2.WING_ROOT_CHORD*1000:.0f}/{V2.WING_TIP_CHORD*1000:.0f} mm")
    print(f"  Stab : {V2.STAB_AIRFOIL_NAME}  b={V2.STAB_SPAN*100:.0f} cm  "
          f"c_R/T={V2.STAB_ROOT_CHORD*1000:.0f}/{V2.STAB_TIP_CHORD*1000:.0f} mm")
    print(f"  Bornes fuselage : [{fl_lo*100:.0f}, {fl_hi*100:.0f}] cm")
    print(f"  Hardcodé actuellement : SLOPE={V2.DE_DA_SLOPE:+.3f}  "
          f"INTERCEPT={V2.DE_DA_INTERCEPT:+.3f}")
    print(f"  Formule classique 4/(AR+2) = {4.0/(AR_w_nom+2.0):.3f}  "
          f"(surestimée pour hydrofoils)")
    print(f"{'='*65}")

    slope, intercept = calibrate(verbose=True)

    print(f"\n  ▶ Résultats régression linéaire")
    print(f"\n        DE_DA_SLOPE     = {slope:+.3f}")
    print(f"        DE_DA_INTERCEPT = {intercept:+.3f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
