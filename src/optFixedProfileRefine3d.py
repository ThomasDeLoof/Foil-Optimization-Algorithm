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
    D_2D += V2.D_MAST
    L_3D, D_3D = ll["L"], ll["D"]
    D_3D += V2.D_MAST
    weight = V2.WEIGHT
    print(f"\n  {label}" if label else "")
    print(f"  {'Métrique':<20} {'AeroBuildup (2D)':>18} {'LiftingLine (3D)':>18}")
    print(f"  {'-'*58}")
    print(f"  {'L (N)':<20} {L_2D:>18.1f} {L_3D:>18.1f}")
    print(f"  {'D_aéro (N)':<20} {D_2D:>18.2f} {D_3D:>18.2f}")
    print(f"  {'L/D_total':<20} {L_2D/D_2D:>18.2f} {L_3D/D_3D:>18.2f}")
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
# 4. Export complet du foil raffiné (markdown 3D + XML + airfoils/ + x_best)
# ─────────────────────────────────────────────────────────────────────────────
def _export_profile_dat(af_obj, airfoils_dir, filename, name_internal):
    """Format Selig normalisé, repané à 50 pts/side (cf. V2)."""
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
    Génère un dossier outputs/<scenario>_refined3d_<timestamp>/ avec :
      - fiche_technique_3d.md (métriques LiftingLine)
      - <case>_<airfoil>_<timestamp>_plane.xml (XFLR5)
      - airfoils/wing_sec_*.dat, stab_sec_*.dat, mast_sec_*.dat
      - x_refined.npy (vecteur 10D du foil raffiné, re-loadable)
    """
    x = np.clip(x, V2.LB, V2.UB)
    p = V2.decode(x)
    airplane, wing, stab, mc, mast_obj, fuselage_obj = V2.build_airplane(p)

    # --- Aéro : LL pour croisière (3D), AB pour décollage ---
    ll_c = aero_3d(airplane, p["alpha_cruise"])
    L, D, Cm, CL = ll_c["L"], ll_c["D"], ll_c["Cm"], ll_c["CL"]
    D_total = D + V2.D_MAST

    op_to = asb.OperatingPoint(velocity=V2.cfg["v_takeoff"], alpha=p["alpha_to"],
                               atmosphere=V2.atmosphere)
    ab_to = asb.AeroBuildup(airplane, op_to).run()
    L_to, D_to, CL_to = float(ab_to["L"]), float(ab_to["D"]), float(ab_to["CL"])

    # --- Métriques équilibre / structure ---
    X_cg    = p["cg_ratio"] * mc
    M_wing  = Cm * V2.q_cruise * wing.area() * mc
    M_rig   = V2.rig_mass * 9.81 * (-(X_cg - V2.x_mast))
    M_total = M_wing + V2.M_MAST + M_rig
    v_h     = (stab.area() * p["fuselage_length"]) / (wing.area() * mc)
    sigma_vm = V2.von_mises_root(p["wing_root_chord"], p["wing_span"])

    # Pitch dynamics (analytique) — toutes les métriques de pilotabilité
    try:
        pd = V2.pitch_dynamics(wing, stab, mc, p["fuselage_length"], p["cg_ratio"])
        omega_n = V2.pitch_frequency_hz(pd["Cm_alpha"], V2.q_cruise, wing.area(), mc)
    except Exception:
        pd = {"SM_chord": float("nan"), "SM_abs": float("nan"),
              "SM_lt": float("nan"), "Cm_alpha": float("nan")}
        omega_n = float("nan")

    # SM/c̄ via VLM (vérification — downwash inclus)
    try:
        plane_vlm = asb.Airplane(wings=airplane.wings, xyz_ref=airplane.xyz_ref,
                                 s_ref=airplane.s_ref, c_ref=airplane.c_ref, b_ref=airplane.b_ref)
        op_p = asb.OperatingPoint(velocity=V2.cfg["v_cruise"], alpha=3.25, atmosphere=V2.atmosphere)
        op_m = asb.OperatingPoint(velocity=V2.cfg["v_cruise"], alpha=2.75, atmosphere=V2.atmosphere)
        ap, am = asb.VortexLatticeMethod(plane_vlm, op_p).run(), asb.VortexLatticeMethod(plane_vlm, op_m).run()
        sm_vlm = -(float(ap["Cm"]) - float(am["Cm"])) / (float(ap["CL"]) - float(am["CL"]))
    except Exception:
        sm_vlm = float("nan")

    # --- Dossier de sortie ---
    now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"{V2.CASE}_refined3d_{now_str}")
    airfoils_dir = os.path.join(out_dir, "airfoils")
    os.makedirs(airfoils_dir, exist_ok=True)

    # --- Renommer + exporter chaque section (deepcopy obligatoire car les
    #     xsecs partagent le même airfoil par référence en mémoire) ---
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

    # --- XML XFLR5 ---
    xml_path = os.path.join(out_dir, f"{V2.CASE}_refined3d_{now_str}_plane.xml")
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
        print(f"  ~ XML XFLR5 non exporté : {e}")

    # --- Fiche markdown (orientée 3D / LiftingLine) ---
    now_disp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sm_real_str = f"{sm_vlm*100:.1f}%" if np.isfinite(sm_vlm) else "n/a"
    sm_chord_str = f"{pd['SM_chord']*100:.1f}%" if np.isfinite(pd['SM_chord']) else "n/a"
    omega_str = f"{omega_n:.2f} Hz" if np.isfinite(omega_n) else "UNSTABLE"
    pilot_lvl = V2.PILOT_LEVEL
    f_lo, f_hi = V2.get_pilot_freq_range()
    lines = [
        f"# Fiche Technique — {V2.CASE.upper()}  |  REFINEMENT 3D (LiftingLine)",
        "", f"*Générée le {now_disp}*",
        "", "Ce foil a été raffiné en 3D depuis la solution AeroBuildup de V2.",
        "Les valeurs L, D, CL, CD, L/D ci-dessous viennent de **LiftingLine**",
        "(downwash et induced drag exacts) — donc plus représentatives que V2.",
        "", "---", "",
        "## 0. Configuration", "",
        f"| Élément | Profil | Span | Corde R/T |",
        "|:---|:---|:---|:---|",
        f"| Aile | {V2.WING_AIRFOIL_NAME} | {p['wing_span']*100:.0f} cm | "
        f"{p['wing_root_chord']*1000:.0f} / {p['wing_tip_chord']*1000:.0f} mm |",
        f"| Stab | {V2.STAB_AIRFOIL_NAME} | {V2.STAB_SPAN*100:.0f} cm | "
        f"{V2.STAB_ROOT_CHORD*1000:.0f} / {V2.STAB_TIP_CHORD*1000:.0f} mm |",
        "", "---", "",
        "## 1. Trim (raffiné en 3D)", "",
        "| Variable | Valeur |", "|:---|:---|",
        f"| Fuselage length | {p['fuselage_length']*100:.1f} cm |",
        f"| CG ratio | {p['cg_ratio']*100:.1f}% c̄ |",
        f"| Calage aile | {p['wing_setting_angle']:.2f}° |",
        f"| Twist | {p['twist']:.2f}° |",
        f"| Calage stab | {p['s_twist']:.2f}° |",
        f"| α décollage | {p['alpha_to']:.2f}° |",
        f"| α croisière | {p['alpha_cruise']:.2f}° |",
        "", "---", "",
        "## 2. Performances 3D (LiftingLine)", "",
        "| Paramètre | Croisière (LL) | Décollage (AB) |",
        "|:---|:---|:---|",
        f"| V (m/s) | {V2.cfg['v_cruise']} | {V2.cfg['v_takeoff']} |",
        f"| L (N) | {L:.1f} | {L_to:.1f} |",
        f"| D aéro (N) | {D:.2f} | {D_to:.2f} |",
        f"| CL | {CL:.3f} | {CL_to:.3f} |",
        f"| D total (+ mât) | {D_total:.2f} | — |",
        f"| **Finesse L/D** | **{L/D_total:.2f}** | — |",
        f"| Écart L vs poids | {(L-V2.WEIGHT)/V2.WEIGHT*100:+.1f}% | {(L_to-V2.WEIGHT)/V2.WEIGHT*100:+.1f}% |",
        "", "---", "",
        "## 3. Pilotabilité, Stabilité & Structure", "",
        "| Paramètre | Valeur | Cible |", "|:---|:---|:---|",
        f"| **ω_n** (fréquence pitch) | **{omega_str}** | '{pilot_lvl}' [{f_lo:.1f}–{f_hi:.1f}] Hz |",
        f"| Cm_α (raideur tangage) | {pd['Cm_alpha']:.2f} rad⁻¹ | < 0 = stable |",
        f"| SM/l_t (scale-invariant) | {pd['SM_lt']*100:.1f}% | typique aviation 10-25% |",
        f"| Gap NP-CG (absolu) | {pd['SM_abs']*1000:.1f} mm | — |",
        f"| SM/c̄ (legacy, analytique) | {sm_chord_str} | — (chord-normalisé) |",
        f"| SM/c̄ via VLM (verif) | {sm_real_str} | écart ≤ 10 pts attendu |",
        f"| Moment résiduel | {M_total:.3f} N·m | < {0.05*V2.WEIGHT*mc:.2f} N·m |",
        f"| Volume de queue V_h | {v_h:.3f} | [{V2.cfg['vh_range'][0]:.2f}–{V2.cfg['vh_range'][1]:.2f}] |",
        f"| Von Mises root | {sigma_vm/1e6:.1f} MPa | < {V2.SIGMA_CARBONE/1e6:.0f} MPa |",
        "",
    ]
    with open(os.path.join(out_dir, "fiche_technique_3d.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --- Sauvegarde du X raffiné pour re-load ---
    np.save(os.path.join(out_dir, "x_refined.npy"), x)

    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# 5. Point d'entrée
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

    # 4) Export complet du foil raffiné
    print("\n[4/4] Export du foil raffiné...")
    out_dir = export_refined(x_refined)
    print(f"  ✓ Fiche technique : {out_dir}/fiche_technique_3d.md")
    print(f"  ✓ XML XFLR5       : {out_dir}/*.xml")
    print(f"  ✓ Profils .dat    : {out_dir}/airfoils/")
    print(f"  ✓ X raffiné       : {out_dir}/x_refined.npy")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
