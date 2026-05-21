# =================================================================================
# optFixedProfileRefine3d.py — 3D verification and refinement of the best V2 foil
# -------------------------------------------------------------------------------
# optFixedProfileV2 (AeroBuildup, ~0.5s/eval) finds the best planform via DE —
# fast but ignores 3D wing↔stab interactions (downwash, exact induced drag).
# This script applies LiftingLine on the DE SOLUTION.
#
# Auto-load: by default we take the most recent x_best.npy from outputs/.
# Otherwise: python refine_3d.py <path/x_best.npy>
# -------------------------------------------------------------------------------

import copy
import datetime as dt
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
# 1. Auto-load of the latest x_best.npy
# ─────────────────────────────────────────────────────────────────────────────
def _find_latest_x_best():
    """Searches for the most recent x_best.npy in outputs/."""
    pattern = str(ROOT.parent / "outputs" / "*" / "x_best.npy")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 3D evaluator (LiftingLine)
# ─────────────────────────────────────────────────────────────────────────────
def aero_3d(airplane: asb.Airplane, alpha: float) -> dict:
    """Evaluates L/D/Cm with LiftingLine (captures 3D downwash & induced drag)."""
    plane_ll = asb.Airplane(
        wings=airplane.wings, xyz_ref=airplane.xyz_ref,
        s_ref=airplane.s_ref, c_ref=airplane.c_ref, b_ref=airplane.b_ref,
    )
    op = asb.OperatingPoint(velocity=V2.cfg["v_cruise"], alpha=alpha,
                            atmosphere=V2.atmosphere)
    a = asb.LiftingLine(plane_ll, op).run()
    return {k: float(a[k]) for k in ("L", "D", "Cm", "CL", "CD")}


def compare_2d_3d(x: np.ndarray, label: str = "") -> dict:
    """Displays D_2D (AeroBuildup) vs D_3D (LiftingLine) on the same X.
    α_cruise is derived by trim L=WEIGHT (AeroBuildup), same value for both."""
    p = V2.decode(np.clip(x, V2.LB, V2.UB))
    airplane, *_ = V2.build_airplane(p)

    alpha_trim, ab, _ = V2.trim_alpha_for_lift(airplane, target_L=V2.WEIGHT)
    ll = aero_3d(airplane, alpha_trim)

    L_2D, D_2D = float(ab["L"]), float(ab["D"])
    D_2D += V2.D_MAST
    L_3D, D_3D = ll["L"], ll["D"]
    D_3D += V2.D_MAST
    weight = V2.WEIGHT
    print(f"\n  {label}" if label else "")
    print(f"  {'Metric':<20} {'AeroBuildup (2D)':>18} {'LiftingLine (3D)':>18}")
    print(f"  {'-'*58}")
    print(f"  {'L (N)':<20} {L_2D:>18.1f} {L_3D:>18.1f}")
    print(f"  {'D_aero (N)':<20} {D_2D:>18.2f} {D_3D:>18.2f}")
    print(f"  {'L/D_total':<20} {L_2D/D_2D:>18.2f} {L_3D/D_3D:>18.2f}")
    return {"L": L_3D, "D": D_3D, "Cm": ll["Cm"], "CL": ll["CL"]}


# ─────────────────────────────────────────────────────────────────────────────
# 3. 3D refinement: trim angles (fixed planform, bounds applied)
# ─────────────────────────────────────────────────────────────────────────────
# Indices to refine: everything except α_cruise (derived) and planform (frozen post-DE).
# x = [fl, cg, incidence angle, twist, s_twist, α_to, w_span, w_root, w_tip, s_span, s_root, s_tip]
# We refine cg + 3 angles + α_to.
TRIM_INDICES = [1, 2, 3, 4, 5]   # cg_ratio, incidence angle, twist, s_twist, α_to


def _trim_alpha_3d(airplane, target_L: float,
                   alpha_lo: float = 0.0, alpha_hi: float = 3.0) -> tuple:
    """LiftingLine equivalent of V2.trim_alpha_for_lift (3 LL calls).
    Returns (alpha_trim, aero_at_trim, bracket) — the bracket provides dCm/dα
    for pitch_dynamics_from_aero (zero cost)."""
    aero_lo = aero_3d(airplane, alpha_lo)
    aero_hi = aero_3d(airplane, alpha_hi)
    bracket = {"aero_lo": aero_lo, "aero_hi": aero_hi,
               "alpha_lo": alpha_lo, "alpha_hi": alpha_hi}
    L_lo, L_hi = aero_lo["L"], aero_hi["L"]
    if abs(L_hi - L_lo) < 1.0:
        return float(alpha_lo), aero_lo, bracket
    alpha_trim = alpha_lo + (target_L - L_lo) / (L_hi - L_lo) * (alpha_hi - alpha_lo)
    alpha_trim = max(-3.0, min(12.0, alpha_trim))
    aero_trim = aero_3d(airplane, alpha_trim)
    return float(alpha_trim), aero_trim, bracket


def _objective_3d(x_trim: np.ndarray, x_template: np.ndarray) -> float:
    """LiftingLine cost + penalties. α_cruise derived by L=WEIGHT trim in LL."""
    x = x_template.copy()
    x[TRIM_INDICES] = x_trim
    x = np.clip(x, V2.LB, V2.UB)
    p = V2.decode(x)

    try:
        airplane, wing, stab, mc, _, _ = V2.build_airplane(p)
        # α_cruise solved in LiftingLine
        alpha_trim, ll, _ = _trim_alpha_3d(airplane, target_L=V2.WEIGHT)
        p["alpha_cruise"] = alpha_trim
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
    # L_cruise is already ≈ WEIGHT by construction (trim), small residual penalty
    pen += V2.K1 * V2.soft_penalty(L,    V2.WEIGHT, V2.WEIGHT,      ref=V2.WEIGHT)
    pen += V2.K1 * V2.soft_penalty(L_to, V2.WEIGHT, np.inf,         ref=V2.WEIGHT)
    pen += V2.K1 * V2.soft_penalty(CL_to, -np.inf, V2.CL_MAX_TO,    ref=V2.CL_MAX_TO)

    X_cg = p["cg_ratio"] * mc
    M_total = (Cm * V2.q_cruise * wing.area() * mc
               + V2.M_MAST + V2.rig_mass * 9.81 * (-(X_cg - V2.x_mast)))
    M_ref = V2.WEIGHT * mc
    pen += V2.K1 * V2.soft_penalty(M_total, -0.05 * M_ref, 0.05 * M_ref, ref=M_ref)
    pen += V2.K2 * V2.soft_penalty(p["alpha_cruise"], -1.0, np.inf, ref=2.0)

    return D_total + 0.3 * D_to + pen


def refine_trim_3d(x_start: np.ndarray, maxiter: int = 80) -> tuple:
    """
    Nelder-Mead on the 6 trim angles, planform fixed.
    The DE opt bounds are propagated to the refinement (scipy 1.7+ supports
    bounds= on Nelder-Mead via adapted reflection/contraction).
    """
    x_start = np.clip(x_start.copy(), V2.LB, V2.UB)
    x_trim0 = x_start[TRIM_INDICES]
    trim_bounds = list(zip(V2.LB[TRIM_INDICES], V2.UB[TRIM_INDICES]))

    print(f"\n  3D Refinement (Nelder-Mead bounds={[(f'{lo:.2g}',f'{hi:.2g}') for lo,hi in trim_bounds]}, {maxiter} iter max)...", flush=True)
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
# 4. Full export of the refined foil (3D markdown + XML + airfoils/ + x_best)
# ─────────────────────────────────────────────────────────────────────────────
def _export_profile_dat(af_obj, airfoils_dir, filename, name_internal):
    """Normalized Selig format, repaneled at 50 pts/side (cf. V2)."""
    af_obj = af_obj.repanel(n_points_per_side=50)
    coords = af_obj.coordinates.copy()
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    coords[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min)
    coords[:, 1] = coords[:, 1] / (x_max - x_min)
    idx_le = int(np.argmin(coords[:, 0]))
    upper, lower = coords[:idx_le + 1], coords[idx_le:]
    if upper[0, 0] < upper[-1, 0]: upper = upper[::-1]
    if lower[0, 0] > lower[-1, 0]: lower = lower[::-1]
    final = np.concatenate([upper, lower[1:]])
    with open(os.path.join(airfoils_dir, filename), "w") as f:
        f.write(f"{name_internal}\n")
        for x, y in final:
            f.write(f" {x:.6f} {y:.6f}\n")


def export_refined(x: np.ndarray) -> str:
    """
    Generates an outputs/<scenario>_refined3d_<timestamp>/ folder with:
      - fiche_technique_3d.md (LiftingLine metrics)
      - <case>_<airfoil>_<timestamp>_plane.xml (XFLR5)
      - airfoils/wing_sec_*.dat, stab_sec_*.dat, mast_sec_*.dat
      - x_refined.npy (10D vector of the refined foil, re-loadable)
    """
    x = np.clip(x, V2.LB, V2.UB)
    p = V2.decode(x)
    airplane, wing, stab, mc, mast_obj, fuselage_obj = V2.build_airplane(p)

    # --- Aero: LL for cruise (3D, α derived), AB for takeoff ---
    alpha_trim, ll_c, bracket = _trim_alpha_3d(airplane, target_L=V2.WEIGHT)
    p["alpha_cruise"] = alpha_trim
    L, D, Cm, CL = ll_c["L"], ll_c["D"], ll_c["Cm"], ll_c["CL"]
    D_total = D + V2.D_MAST

    op_to = asb.OperatingPoint(velocity=V2.cfg["v_takeoff"], alpha=p["alpha_to"],
                               atmosphere=V2.atmosphere)
    ab_to = asb.AeroBuildup(airplane, op_to).run()
    L_to, D_to, CL_to = float(ab_to["L"]), float(ab_to["D"]), float(ab_to["CL"])

    # --- Balance / structure metrics ---
    X_cg    = p["cg_ratio"] * mc
    M_wing  = Cm * V2.q_cruise * wing.area() * mc
    M_rig   = V2.rig_mass * 9.81 * (-(X_cg - V2.x_mast))
    M_total = M_wing + V2.M_MAST + M_rig
    v_h     = (stab.area() * p["fuselage_length"]) / (wing.area() * mc)
    sigma_vm_static = V2.von_mises_root(p["wing_root_chord"], p["wing_span"], load_factor=1.0)
    sigma_vm        = V2.von_mises_root(p["wing_root_chord"], p["wing_span"],
                                        load_factor=V2.LOAD_PEAK_FACTOR)

    # REAL pitch dynamics from the 3D LiftingLine bracket (zero cost)
    try:
        c_stab_mean = 0.5 * (p["stab_root_chord"] + p["stab_tip_chord"])
        l_t = (V2.x_fuselage_start + p["fuselage_length"] - V2.STAB_FUSE_OFFSET
               + 0.25 * c_stab_mean) - 0.25 * mc
        pd = V2.pitch_dynamics_from_aero(bracket, mc, l_t)
        omega_n = V2.pitch_frequency_hz(pd["Cm_alpha"], V2.q_cruise, wing.area(), mc)
    except Exception:
        pd = {"SM_chord": float("nan"), "SM_abs": float("nan"),
              "SM_lt": float("nan"), "Cm_alpha": float("nan")}
        omega_n = float("nan")

    # --- Output folder ---
    out_dir = V2.next_output_dir(suffix="refined3d")
    run_tag = os.path.basename(out_dir)
    airfoils_dir = os.path.join(out_dir, "airfoils")
    os.makedirs(airfoils_dir, exist_ok=True)

    # --- Rename + export each section (deepcopy required because xsecs
    #     share the same airfoil by reference in memory) ---
    def _rename_and_export(xsecs, prefix):
        for i, xs in enumerate(xsecs):
            n = f"{prefix}_sec_{i}"
            af_copy = copy.deepcopy(xs.airfoil)
            af_copy.name = n
            xs.airfoil = af_copy
            _export_profile_dat(af_copy, airfoils_dir, f"{n}.dat", n)

    _rename_and_export(airplane.wings[0].xsecs, "wing")
    _rename_and_export(airplane.wings[1].xsecs, "stab")
    _rename_and_export(mast_obj.xsecs,          "mast")

    # --- XFLR5 XML ---
    xml_path = os.path.join(out_dir, f"{run_tag}_plane.xml")
    try:
        asb.Airplane(
            wings=[
                asb.Wing(symmetric=True, name="mainwing", xsecs=airplane.wings[0].xsecs),
                asb.Wing(symmetric=True, name="elevator", xsecs=airplane.wings[1].xsecs),
                mast_obj,
            ],
            fuselages=[fuselage_obj],
            xyz_ref=airplane.xyz_ref,
        ).export_XFLR5_xml(xml_path)
    except Exception as e:
        print(f"  ~ XFLR5 XML not exported: {e}")

    # --- Markdown sheet (3D / LiftingLine oriented) ---
    now_disp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sm_chord_str = f"{pd['SM_chord']*100:.1f}%" if np.isfinite(pd['SM_chord']) else "n/a"
    omega_str = f"{omega_n:.2f} Hz" if np.isfinite(omega_n) else "UNSTABLE"
    f_lo, f_hi = V2.get_pilot_freq_range()
    lines = [
        f"# Technical Sheet — {V2.CASE.upper()}  |  3D REFINEMENT (LiftingLine)",
        "", f"*Generated on {now_disp}*",
        "", "This foil was refined in 3D from the V2 AeroBuildup solution.",
        "The L, D, CL, CD, L/D values below come from **LiftingLine**",
        "(exact downwash and induced drag) — therefore more representative than V2.",
        "", "---", "",
        "## 0. Configuration", "",
        f"| Element | Airfoil | Span | Chord R/T |",
        "|:---|:---|:---|:---|",
        f"| Wing | {V2.WING_AIRFOIL_NAME} | {p['wing_span']*100:.0f} cm | "
        f"{p['wing_root_chord']*1000:.0f} / {p['wing_tip_chord']*1000:.0f} mm |",
        f"| Stab | {V2.STAB_AIRFOIL_NAME} | {V2.STAB_SPAN*100:.0f} cm | "
        f"{V2.STAB_ROOT_CHORD*1000:.0f} / {V2.STAB_TIP_CHORD*1000:.0f} mm |",
        "", "---", "",
        "## 1. Trim (refined in 3D)", "",
        "| Variable | Value |", "|:---|:---|",
        f"| Fuselage length | {p['fuselage_length']*100:.1f} cm |",
        f"| CG ratio | {p['cg_ratio']*100:.1f}% c̄ |",
        f"| Wing incidence angle | {p['wing_setting_angle']:.2f}° |",
        f"| Twist | {p['twist']:.2f}° |",
        f"| Stab incidence angle | {p['s_twist']:.2f}° |",
        f"| α takeoff | {p['alpha_to']:.2f}° |",
        f"| α cruise | {p['alpha_cruise']:.2f}° |",
        "", "---", "",
        "## 2. 3D Performance (LiftingLine)", "",
        "| Parameter | Cruise (LL) | Takeoff (AB) |",
        "|:---|:---|:---|",
        f"| V (m/s) | {V2.cfg['v_cruise']} | {V2.cfg['v_takeoff']} |",
        f"| L (N) | {L:.1f} | {L_to:.1f} |",
        f"| D aero (N) | {D:.2f} | {D_to:.2f} |",
        f"| CL | {CL:.3f} | {CL_to:.3f} |",
        f"| D total (+ mast) | {D_total:.2f} | — |",
        f"| **L/D ratio** | **{L/D_total:.2f}** | — |",
        f"| L vs weight gap | {(L-V2.WEIGHT)/V2.WEIGHT*100:+.1f}% | {(L_to-V2.WEIGHT)/V2.WEIGHT*100:+.1f}% |",
        "", "---", "",
        "## 3. Handling, Stability & Structure", "",
        "| Parameter | Value | Target |", "|:---|:---|:---|",
        f"| **ω_n** (pitch frequency) | **{omega_str}** | freeride [{f_lo:.1f}–{f_hi:.1f}] Hz |",
        f"| Cm_α (pitch stiffness) | {pd['Cm_alpha']:.2f} rad⁻¹ | < 0 = stable |",
        f"| SM/l_t (scale-invariant) | {pd['SM_lt']*100:.1f}% | typical aviation 10-25% |",
        f"| NP-CG gap (absolute) | {pd['SM_abs']*1000:.1f} mm | — |",
        f"| SM/c̄ (legacy, chord-normalized) | {sm_chord_str} | — |",
        f"| Residual moment | {M_total:.3f} N·m | < {V2.M_TOL_TRIM:.1f} N·m (pilot trim) |",
        f"| Tail volume V_h | {v_h:.3f} | [{V2.cfg['vh_range'][0]:.2f}–{V2.cfg['vh_range'][1]:.2f}] |",
        f"| Von Mises root (peak ×{V2.LOAD_PEAK_FACTOR:.1f}g) | {sigma_vm/1e6:.1f} MPa | < {V2.SIGMA_ADMISSIBLE/1e6:.0f} MPa (fatigue) |",
        f"| Von Mises root (static 1g) | {sigma_vm_static/1e6:.1f} MPa | < {V2.SIGMA_ULTIMATE/1e6:.0f} MPa (rupture) |",
        "",
    ]
    with open(os.path.join(out_dir, "fiche_technique_3d.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --- Save the refined X for re-load ---
    np.save(os.path.join(out_dir, "x_refined.npy"), x)

    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # X loading: argv > latest outputs/*/x_best.npy > X_REF
    src = None
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        src = sys.argv[1]
    else:
        src = _find_latest_x_best()

    if src is not None:
        x = np.load(src)
        if len(x) != V2.N_VAR:
            print(f"  ⚠ x_best.npy has {len(x)} variables, expected {V2.N_VAR}. "
                  f"Probably from an older version of the code — falling back on X_REF.")
            x = V2.X_REF.copy()
        else:
            print(f"  ✓ Auto-loading X from: {src}")
    else:
        x = V2.X_REF.copy()
        print(f"  ⚠ No x_best.npy found in outputs/ — falling back on X_REF.")
        print(f"    (Run `python src/optFixedProfileV2.py` first to generate an optimal X.)")

    print(f"\n{'='*70}")
    print(f"  REFINE_3D — scenario {V2.CASE.upper()}, airfoil {V2.WING_AIRFOIL_NAME}")
    print(f"{'='*70}")

    # 1) Initial comparison (on the warm-start solution)
    print("\n[1/3] 2D vs 3D comparison on the loaded solution:")
    compare_2d_3d(x, label="(starting point)")

    # 2) Refinement
    print("\n[2/3] 3D refinement of trim angles...")
    x_refined, J_refined, n_iter = refine_trim_3d(x, maxiter=80)
    p0, pr = V2.decode(x), V2.decode(x_refined)
    print(f"  Final cost: {J_refined:.2f}   ({n_iter} Nelder-Mead iterations, planform unchanged)")
    print(f"  Trim angle evolutions:")
    for k in ("cg_ratio", "wing_setting_angle", "twist", "s_twist", "alpha_to", "alpha_cruise"):
        delta = pr[k] - p0[k]
        flag = " *" if abs(delta) > 0.05 else ""
        print(f"    {k:<22} : {p0[k]:>7.3f}  →  {pr[k]:>7.3f}   (Δ={delta:+.3f}){flag}")

    # 3) Final comparison
    print("\n[3/3] 3D aero of the refined foil:")
    compare_2d_3d(x_refined, label="(after refinement)")

    # 4) Full export of the refined foil
    print("\n[4/4] Exporting the refined foil...")
    out_dir = export_refined(x_refined)
    print(f"  ✓ Technical sheet : {out_dir}/fiche_technique_3d.md")
    print(f"  ✓ XFLR5 XML       : {out_dir}/*.xml")
    print(f"  ✓ .dat airfoils   : {out_dir}/airfoils/")
    print(f"  ✓ Refined X       : {out_dir}/x_refined.npy")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
