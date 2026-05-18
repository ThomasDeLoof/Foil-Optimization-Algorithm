# =================================================================================
# Hydrofoil Optimization V3 — Paramétrique Macroscopique
# -------------------------------------------------------------------------------

import os
import sys
import datetime as dt
import warnings
import numpy as np
import yaml
from pathlib import Path
from scipy.optimize import differential_evolution, minimize, Bounds
from scipy.stats import qmc

import aerosandbox as asb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from config.water_atmosphere import Water as Atmosphere

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

with open(ROOT / "config/parameters.yaml") as f:
    phy = yaml.safe_load(f)
with open(ROOT / "config/scenarios.yaml") as f:
    SCENARIOS = yaml.safe_load(f)

CASE = phy["case"]
if CASE not in SCENARIOS:
    raise ValueError(f"Cas '{CASE}' introuvable. Options : {list(SCENARIOS.keys())}")
cfg = SCENARIOS[CASE]

atmosphere = Atmosphere()

# ── Masses ────────────────────────────────────────────────────────────────────
mass     = phy["pilot"]["mass_kg"] + phy["board"]["mass_kg"]
rig_mass = cfg["rig_mass_kg"]
WEIGHT   = (mass + rig_mass) * 9.81

# ── Géométries fixes héritées du YAML (mât, fuselage, sweep aile, etc.) ──────
sweep_deg          = phy["wing"]["sweep_deg"]
wing_anhedral_deg  = phy["wing"]["anhedral_deg"]
N_WING             = phy["wing"]["n_sections"]
CL_MAX_TO          = phy["wing"]["cl_max_takeoff"]
WING_SKIN_THICK    = phy["wing"]["skin_thickness"]
fuselage_diameter  = phy["fuselage"]["diameter"]
x_fuselage_start   = phy["fuselage"]["x_start"]
N_FUSE             = phy["fuselage"]["n_sections"]
mast_length        = phy["mast"]["length"]
profondeur_imm     = phy["mast"]["immersion_depth"]
mast_chord_top     = phy["mast"]["chord_top"]
mast_chord_bot     = phy["mast"]["chord_bot"]
mast_profile       = phy["mast"]["profile"]
x_mast             = phy["mast"]["x_position"]
cd_mast            = phy["mast"]["cd"]
N_MAST             = phy["mast"]["n_sections"]
N_STAB             = phy["stab"]["n_sections"]
STAB_SWEEP_DEG     = phy["stab"].get("sweep_deg", 8.0)
STAB_DIHEDRAL_DEG  = phy["stab"].get("dihedral_deg", 3.0)
STAB_FUSE_OFFSET   = phy["stab"].get("fuselage_offset", 0.10)

# Sanity check géométrique : le BA stab doit rester à l'intérieur du fuselage
_fuse_len_min = phy["fuselage"]["length_bounds"][0]
if STAB_FUSE_OFFSET > _fuse_len_min:
    raise ValueError(
        f"stab.fuselage_offset ({STAB_FUSE_OFFSET*100:.1f} cm) > "
        f"fuselage.length_bounds[0] ({_fuse_len_min*100:.1f} cm) "
        "— le BA stab serait avant le nez fuselage."
    )

# ── Mât (constants) ───────────────────────────────────────────────────────────
chord_wl = mast_chord_bot * (1 - profondeur_imm / mast_length) \
         + mast_chord_top * (profondeur_imm / mast_length)
q_cruise = 0.5 * atmosphere.density() * cfg["v_cruise"] ** 2
D_MAST   = q_cruise * ((mast_chord_bot + chord_wl) / 2) * profondeur_imm * cd_mast
M_MAST   = -D_MAST * (profondeur_imm / 2)

# ── Limites physiques ─────────────────────────────────
SIGMA_CARBONE = 300e6
P_ATM         = atmosphere.pressure()
P_VAPOR       = atmosphere.vapor_pressure()
SIGMA_CAV     = (P_ATM + atmosphere.density() * 9.81 * profondeur_imm - P_VAPOR) \
              / (0.5 * atmosphere.density() * cfg["v_cruise"] ** 2)

# ─────────────────────────────────────────────────────────────────────────────
# 2. GÉOMÉTRIE — profils et dimensions stab
# ─────────────────────────────────────────────────────────────────────────────

WING_AIRFOIL_NAME = cfg["wing_airfoil"]
STAB_AIRFOIL_NAME = phy["stab"]["airfoil"]

# Warm-start de la planform aile : on prend la référence du scénario si fournie
# (scenarios.yaml), sinon les valeurs par défaut de parameters.yaml.
WING_SPAN         = cfg.get("wing_span_init",       phy["wing"]["span_init"])
WING_ROOT_CHORD   = cfg.get("wing_root_chord_init", phy["wing"]["root_chord_init"])
WING_TIP_CHORD    = cfg.get("wing_tip_chord_init",  phy["wing"]["tip_chord_init"])

# Stab figé par scénario
STAB_SPAN         = cfg["stab_span"]
STAB_ROOT_CHORD   = cfg["stab_root_chord"]
STAB_TIP_CHORD    = cfg["stab_tip_chord"]

# Précalcul des airfoils, avec import robuste (ASB n'a PAS tous les NACA 6-series).
def _load_airfoil(name: str, fallback: str = "naca2410"):
    """Charge un profil ASB, retombe sur `fallback` si la librairie n'a pas les coords."""
    try:
        af = asb.Airfoil(name)
        if af.coordinates is not None and len(af.coordinates) >= 30:
            return af
    except Exception:
        pass
    print(f"  ⚠ Profil '{name}' indisponible dans ASB → fallback '{fallback}'")
    return asb.Airfoil(fallback)


WING_AIRFOIL = _load_airfoil(WING_AIRFOIL_NAME, fallback="naca2410")
STAB_AIRFOIL = _load_airfoil(STAB_AIRFOIL_NAME, fallback="naca0012")
# Si fallback a eu lieu, le nom officiel utilisé en aval reste le nom demandé,
# mais le .dat exporté correspondra au fallback (visible dans XFLR5).
WING_AIRFOIL_NAME = WING_AIRFOIL.name
STAB_AIRFOIL_NAME = STAB_AIRFOIL.name

try:
    WING_THICKNESS_REL = float(WING_AIRFOIL.max_thickness())
except Exception:
    WING_THICKNESS_REL = 0.12

# ─────────────────────────────────────────────────────────────────────────────
# 3. VECTEUR D'OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
# x = [fuselage_length, cg_ratio, wing_setting_angle, twist,
#      s_twist, alpha_to, alpha_cruise,
#      wing_span, wing_root_chord, wing_tip_chord]

BOUNDS = [
    tuple(phy["fuselage"]["length_bounds"]),                           # 0 fuselage_length (m)
    tuple(cfg["cg_range"]),                                            # 1 cg_ratio (-)
    tuple(phy["wing"].get("calage_bounds",  [-2.0, 5.0])),             # 2 wing_setting_angle (°)
    tuple(phy["wing"].get("washout_bounds", [-5.0, 0.5])),             # 3 twist (°)
    tuple(phy["stab"]["twist_bounds"]),                                # 4 s_twist (°)
    tuple(phy["alpha"]["bounds"]),                                     # 5 alpha_to (°)
    tuple(phy["alpha"]["bounds"]),                                     # 6 alpha_cruise (°)
    tuple(phy["wing"]["span_bounds"]),                                 # 7 wing_span (m)
    tuple(phy["wing"]["root_chord_bounds"]),                           # 8 wing_root_chord (m)
    tuple(phy["wing"]["tip_chord_bounds"]),                            # 9 wing_tip_chord (m)
]
N_VAR = len(BOUNDS)
LB    = np.array([b[0] for b in BOUNDS])
UB    = np.array([b[1] for b in BOUNDS])


def decode(x: np.ndarray) -> dict:
    """Découpe le vecteur DE → dictionnaire de paramètres macroscopiques."""
    return {
        "fuselage_length":    float(x[0]),
        "cg_ratio":           float(x[1]),
        "wing_setting_angle": float(x[2]),
        "twist":              float(x[3]),
        "s_twist":            float(x[4]),
        "alpha_to":           float(x[5]),
        "alpha_cruise":       float(x[6]),
        "wing_span":          float(x[7]),
        "wing_root_chord":    float(x[8]),
        "wing_tip_chord":     float(x[9]),
    }


# Warm-start : milieu des bornes, mais on initialise la planform sur la
# géométrie de référence du scénario (AXIS BSC 890 pour wingfoil).
X_REF = np.array([(b[0] + b[1]) / 2 for b in BOUNDS])
X_REF[7] = WING_SPAN
X_REF[8] = WING_ROOT_CHORD
X_REF[9] = WING_TIP_CHORD
X_REF = np.clip(X_REF, LB, UB)

# ─────────────────────────────────────────────────────────────────────────────
# 4. CONSTRUCTION DE L'AVION
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_tip_kick(c_root: float, c_tip: float, span: float,
                       sweep_deg_local: float, N: int,
                       sweep_power: float = 1.5,
                       kick_start: float = 0.85,
                       target_push: float = 0.020) -> float:
    """
    Calcule le 'tip kick' (recul additionnel au 1/4 corde près du saumon) qui
    garantit un push-back ≥ `target_push` (m) entre la section pénultième et
    la section de tip. Si la sweep naturelle suffit déjà, retourne 0.
    """
    if N < 2:
        return 0.0
    r_pen = (N - 2) / (N - 1)
    delta_c = (c_root - c_tip) * np.sqrt(max(1 - r_pen ** 2, 0.0))
    K       = (span / 2.0) * np.tan(np.radians(sweep_deg_local))
    push_natural = K * (1.0 - r_pen ** sweep_power) - 0.75 * delta_c
    if r_pen <= kick_start:
        return max(0.0, target_push - push_natural)
    s  = (r_pen - kick_start) / (1.0 - kick_start)
    ss = s * s * (3.0 - 2.0 * s)
    return max(0.0, (target_push - push_natural) / max(1.0 - ss, 0.01))


def build_airplane(p: dict) -> tuple:
    """
    Assemble l'aile, le fuselage, le mât et le stabilisateur AeroSandBox.

    Aile principale : géométrie figée (scénario) + loi de corde existante.
        twist_section = wing_setting_angle + twist * r
    Stab : géométrie figée + twist uniforme = s_twist.
    """
    # ── Aile principale ──────────────────────────────────────────────────────
    # Planform pure-elliptique + sweep non-linéaire + tip kick ADAPTATIF.
    # Le kick (recul additionnel du 1/4 corde au saumon, smoothstep C¹) est
    # calculé pour garantir un push-back monotone du TE quelle que soit la
    # géométrie courante (span/cordes/sweep libres).
    span_w  = p["wing_span"]
    root_w  = p["wing_root_chord"]
    tip_w   = min(p["wing_tip_chord"], root_w * 0.95)  # sécurité tip < root
    sweep_power_w    = 1.5
    tip_kick_start_w = 0.85
    tip_kick_amount_w = _adaptive_tip_kick(root_w, tip_w, span_w, sweep_deg,
                                           N_WING, sweep_power_w, tip_kick_start_w)

    wing_xsecs = []
    for i in range(N_WING):
        r = i / (N_WING - 1)

        c_dist = tip_w + (root_w - tip_w) * np.sqrt(max(1 - r ** 2, 0))

        kick_w = 0.0
        if r > tip_kick_start_w:
            s = (r - tip_kick_start_w) / (1.0 - tip_kick_start_w)
            kick_w = tip_kick_amount_w * s * s * (3.0 - 2.0 * s)
        x_qc = (r ** sweep_power_w) * span_w / 2 * np.tan(np.radians(sweep_deg)) + kick_w
        x_le = x_qc + 0.25 * (root_w - c_dist)

        z_pos = -((r ** 2) * span_w / 2) * np.tan(np.radians(wing_anhedral_deg)) \
                - 0.020 * r ** 5

        section_twist = p["wing_setting_angle"] + p["twist"] * r

        wing_xsecs.append(asb.WingXSec(
            xyz_le=[x_le, r * span_w / 2, z_pos],
            chord=c_dist,
            twist=section_twist,
            airfoil=WING_AIRFOIL,
        ))

    wing       = asb.Wing(symmetric=True, name="MainWing", xsecs=wing_xsecs)
    mean_chord = wing.mean_geometric_chord()
    X_cg       = p["cg_ratio"] * mean_chord

    # ── Fuselage ──────────────────────────────────────────────────
    fuse_xsecs = []
    for i in range(N_FUSE):
        xi_rel = i / (N_FUSE - 1)
        xi     = x_fuselage_start + xi_rel * p["fuselage_length"]
        width  = (0.001 if i == 0
                  else fuselage_diameter * (i / 3) if i < 3
                  else fuselage_diameter * (1 - 0.5 * xi_rel ** 3))
        fuse_xsecs.append(asb.FuselageXSec(xyz_c=[xi, 0, 0], radius=width))
    fuselage_obj = asb.Fuselage(name="Fuselage", xsecs=fuse_xsecs)

    # ── Mât ───────────────────────────────────────────────────────
    mast_xsecs = []
    for i in range(N_MAST):
        r = i / (N_MAST - 1)
        mast_xsecs.append(asb.WingXSec(
            xyz_le=[x_mast, 0, -r * profondeur_imm],
            chord=mast_chord_bot * (1 - r) + chord_wl * r,
            twist=0,
            airfoil=asb.Airfoil(mast_profile),
        ))
    mast_obj = asb.Wing(name="Mast", symmetric=False, xsecs=mast_xsecs)

    # ── Stabilisateur ────────────────────────────────────────────────────────
    # Stab figé (companion typique du foil). Tip kick adaptatif via helper.
    # kick_start abaissé à 0.65 car N_STAB plus faible (sinon wobble visible).
    x_stab_root = x_fuselage_start + p["fuselage_length"] - STAB_FUSE_OFFSET
    sweep_power_s    = 1.5
    tip_kick_start_s = 0.65
    tip_kick_amount_s = _adaptive_tip_kick(STAB_ROOT_CHORD, STAB_TIP_CHORD,
                                           STAB_SPAN, STAB_SWEEP_DEG, N_STAB,
                                           sweep_power_s, tip_kick_start_s)
    stab_xsecs = []
    for i in range(N_STAB):
        r = i / (N_STAB - 1)

        c_s = STAB_TIP_CHORD + (STAB_ROOT_CHORD - STAB_TIP_CHORD) \
              * np.sqrt(max(1 - r ** 2, 0))

        kick_s = 0.0
        if r > tip_kick_start_s:
            s = (r - tip_kick_start_s) / (1.0 - tip_kick_start_s)
            kick_s = tip_kick_amount_s * s * s * (3.0 - 2.0 * s)
        x_qc_s = (r ** sweep_power_s) * STAB_SPAN / 2 * np.tan(np.radians(STAB_SWEEP_DEG)) + kick_s
        x_le_s = x_qc_s + 0.25 * (STAB_ROOT_CHORD - c_s)
        z_s    = (r ** 1.5 * STAB_SPAN / 2) * np.tan(np.radians(STAB_DIHEDRAL_DEG))

        stab_xsecs.append(asb.WingXSec(
            xyz_le=[x_stab_root + x_le_s, r * STAB_SPAN / 2, z_s],
            chord=c_s,
            twist=p["s_twist"],
            airfoil=STAB_AIRFOIL,
        ))
    stab = asb.Wing(symmetric=True, name="Stab", xsecs=stab_xsecs)

    airplane = asb.Airplane(
        wings=[wing, stab],
        fuselages=[fuselage_obj],
        xyz_ref=np.array([X_cg, 0.0, 0.0]),
        s_ref=wing.area(), c_ref=mean_chord, b_ref=wing.span(),
    )
    return airplane, wing, stab, mean_chord, mast_obj, fuselage_obj


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONTRAINTE STRUCTURELLE 
# ─────────────────────────────────────────────────────────────────────────────

def von_mises_root(chord_root: float, span: float) -> float:
    """
    Approximation Von Mises à l'emplanture — section coque carbone + âme polystyrène.
    Maintenant fonction de la planform (planform libérée dans l'optimisation).
    """
    span_semi = span / 2.0
    t_max     = WING_THICKNESS_REL * chord_root
    a, b      = chord_root / 2.0, t_max / 2.0

    t_skin     = min(WING_SKIN_THICK, 0.9 * b)
    a_in, b_in = max(a - t_skin, 0.0), max(b - t_skin, 0.0)
    I_xx  = (np.pi / 4.0) * (a * b ** 3 - a_in * b_in ** 3)
    A_enc = np.pi * a * b

    M_flex = (WEIGHT / 2.0) * (span_semi / 4.0)
    M_tors = (WEIGHT / 2.0) * (0.05 * chord_root) * (span_semi / 2.0)

    sigma_flex = M_flex * b / (I_xx + 1e-12)
    tau        = M_tors / (2.0 * A_enc * t_skin + 1e-12)
    return float(np.sqrt(sigma_flex ** 2 + 3 * tau ** 2))

# ─────────────────────────────────────────────────────────────────────────────
# 6. MARGE STATIQUE ANALYTIQUE
# ─────────────────────────────────────────────────────────────────────────────
# Approximation du calcul de marge statique pour réduire le coût

# de_da(fl) = SLOPE × fuselage_length + INTERCEPT
# Calibré par src/calibrate_de_da.py (re-exécuter si géométrie ou scénario change).
DE_DA_SLOPE     =  0.537
DE_DA_INTERCEPT = -0.430


def _cl_alpha_helmbold(AR: float) -> float:
    """
    Pente de portance 3D (rad⁻¹) — correction Helmbold.
    """
    a0 = 2.0 * np.pi
    return a0 / (np.sqrt(1.0 + (a0 / (np.pi * AR)) ** 2) + a0 / (np.pi * AR))


def neutral_point_ratio(wing: asb.Wing, stab: asb.Wing,
                        mean_chord: float, fuselage_length: float) -> float:
    """
    Position du point neutre rapportée à la corde moyenne (X_np/c)
    selon la formule analytique d'Abzug (volume de queue corrigé d'AR).
    """
    AR_w = float(wing.aspect_ratio())
    AR_s = float(stab.aspect_ratio())

    CL_a_w = _cl_alpha_helmbold(AR_w)
    CL_a_s = _cl_alpha_helmbold(AR_s)
    # de_da CALIBRÉ empiriquement via VLM.
    de_da = DE_DA_SLOPE * fuselage_length + DE_DA_INTERCEPT

    # Centres aérodynamiques (frame avion, x=0 au BA emplanture aile)
    X_ac_w      = 0.25 * mean_chord
    c_stab_mean = 0.5 * (STAB_ROOT_CHORD + STAB_TIP_CHORD)
    x_stab_root = x_fuselage_start + fuselage_length - STAB_FUSE_OFFSET
    X_ac_s      = x_stab_root + 0.25 * c_stab_mean

    l_t = X_ac_s - X_ac_w
    V_H = (stab.area() * l_t) / (wing.area() * mean_chord + 1e-12)

    return (X_ac_w / mean_chord) + V_H * (CL_a_s / CL_a_w) * (1.0 - de_da)


# ─────────────────────────────────────────────────────────────────────────────
# 7. FONCTION OBJECTIF 
# ─────────────────────────────────────────────────────────────────────────────

K1 = 2000.0   # contraintes critiques (portance, équilibre, structure)
K2 = 1000.0   # contraintes de pilotage (SM, surface)
K3 = 500.0    # contraintes de confort  (Vh, force stab, cavitation)

_run_counter = {"n": 0}


def soft_penalty(val: float, lo: float, hi: float, ref: float) -> float:
    """Pénalité quadratique normalisée continue"""
    ref_safe = abs(ref) + 1e-9
    if val < lo:
        return ((lo - val) / ref_safe) ** 2
    if val > hi:
        return ((val - hi) / ref_safe) ** 2
    return 0.0


def objective(x: np.ndarray) -> float:
    """
    Minimise la traînée totale (D_aero + D_mât) sous contraintes douce
    """
    x = np.clip(x, LB, UB)  # robuste aux solvers polish hors bornes
    p = decode(x)

    try:
        airplane, wing, stab, mean_chord, _, _ = build_airplane(p)
    except Exception:
        return 1e6  # exception structurelle AeroSandBox

    # ── Évaluation aérodynamique — point de croisière ───────────────────────
    try:
        op_c   = asb.OperatingPoint(velocity=cfg["v_cruise"],
                                    alpha=p["alpha_cruise"],
                                    atmosphere=atmosphere)
        aero_c = asb.AeroBuildup(airplane, op_c).run()
        L  = float(aero_c["L"])
        D  = float(aero_c["D"])
        Cm = float(aero_c["Cm"])
        CL = float(aero_c["CL"])
    except Exception:
        return 1e6  # exception matrice AeroBuildup

    # ── Évaluation aérodynamique — point de décollage ───────────────────────
    try:
        op_to   = asb.OperatingPoint(velocity=cfg["v_takeoff"],
                                     alpha=p["alpha_to"],
                                     atmosphere=atmosphere)
        aero_to = asb.AeroBuildup(airplane, op_to).run()
        L_to    = float(aero_to["L"])
        D_to    = float(aero_to["D"])
        CL_to   = float(aero_to["CL"])
    except Exception:
        return 1e6  # exception matrice AeroBuildup

    D_total = D + D_MAST
    S_wing  = wing.area()
    S_stab  = stab.area()
    penalty = 0.0

    # Portance croisière ≥ poids
    penalty += K1 * soft_penalty(L, WEIGHT, np.inf, ref=WEIGHT)

    # Portance décollage : L_to ≥ poids + CL_to ≤ CL_max_to
    penalty += K1 * soft_penalty(L_to, WEIGHT, np.inf, ref=WEIGHT)
    penalty += K1 * soft_penalty(CL_to, -np.inf, CL_MAX_TO, ref=CL_MAX_TO)

    # Régime d'opération sain : α_cruise ≥ -1° sur profil cambré
    # (en α<-1° on tombe en zone non-linéaire négative pour NACA modérément cambré,
    #  Cm(α) devient erratique. Mais α ∈ [-1, 0]° reste en régime linéaire correct.
    #  Tolérance plus large que 0° car certains scénarios (wingfoil/windsurf) opèrent
    #  intrinsèquement à CL très faible vu leurs combinaisons foil+vitesse.)
    penalty += K2 * soft_penalty(p["alpha_cruise"], -1.0, np.inf, ref=2.0)

    # Équilibre de tangage en croisière (tolérance ±5 % de M_ref)
    X_cg    = p["cg_ratio"] * mean_chord
    M_wing  = Cm * q_cruise * S_wing * mean_chord
    M_rig = rig_mass * 9.81 * (-(X_cg - x_mast))
    M_total = M_wing + M_MAST + M_rig
    M_ref   = WEIGHT * mean_chord
    tol_M   = 0.05 * M_ref
    penalty += K1 * soft_penalty(M_total, -tol_M, +tol_M, ref=M_ref)

    # Marge statique analytique (Abzug + Helmbold)
    try:
        X_np_ratio = neutral_point_ratio(wing, stab, mean_chord, p["fuselage_length"])
        SM         = X_np_ratio - p["cg_ratio"]
        sm_lo, sm_hi = cfg["sm_range"]
        penalty += K2 * soft_penalty(SM, sm_lo, sm_hi, ref=sm_lo)
    except Exception:
        penalty += K2  

    # ── Pénalités auxiliaires ────────────────────────────────

    # Contrainte structurelle : von Mises emplanture (coque carbone)
    sigma_vm = von_mises_root(p["wing_root_chord"], p["wing_span"])
    penalty += K1 * soft_penalty(sigma_vm, 0.0, SIGMA_CARBONE, ref=SIGMA_CARBONE)

    # Volume de queue géométrique
    v_h = (S_stab * p["fuselage_length"]) / (S_wing * mean_chord + 1e-12)
    vh_lo, vh_hi = cfg["vh_range"]
    penalty += K3 * soft_penalty(v_h, vh_lo, vh_hi, ref=vh_lo)

    # Cavitation : Cp_min ≥ −σ_v
    Cp_min = -(1.2 * abs(CL) + 3.0 * WING_THICKNESS_REL)
    penalty += K3 * soft_penalty(Cp_min, -SIGMA_CAV, np.inf, ref=SIGMA_CAV)

    # Cible d'aire douce : reste dans la fourchette du scénario (S_wing libéré)
    sw_lo, sw_hi = cfg["area_target_range"]
    penalty += K3 * soft_penalty(S_wing, sw_lo, sw_hi, ref=0.5*(sw_lo+sw_hi))

    # Objectif multi-point : D_cruise + W_TO × D_takeoff
    # (le décollage compte mais moins que la croisière, qui dure plus longtemps)
    W_TAKEOFF = 0.3
    return D_total + W_TAKEOFF * D_to + penalty


# ─────────────────────────────────────────────────────────────────────────────
# 8. OPTIMISATION — DIFFERENTIAL EVOLUTION + AFFINAGE
# ─────────────────────────────────────────────────────────────────────────────

DE_PARAMS = {
    "strategy":      "best1bin",
    "maxiter":       100,
    "popsize":       25,           # 25 × 7 = 175 individus — suffisant pour 7 var
    "tol":           1e-4,
    "atol":          1e-2,
    "mutation":      (0.5, 1.0),
    "recombination": 0.85,
    "workers":       -1,
    "polish":        False,
    "updating":      "deferred",
}


def _de_callback(_xk: np.ndarray, convergence: float) -> bool:
    """
    Affichage léger toutes les 5 générations.
    Attention ne ré-évalue PAS objective(xk) (ça doublerait le coût des gens)
    """
    _run_counter["n"] += 1
    gen = _run_counter["n"]
    if gen % 5 != 0:
        return False
    pct = min(gen / DE_PARAMS["maxiter"] * 100.0, 100.0)
    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
    print(f"    G{gen:3d} [{bar}] {pct:4.0f}%  conv={convergence:.2e}")
    return False


def _heuristic_starts() -> list:
    """
    Quelques individus heuristiques pour amorcer la population DE.
    Couvrent typiquement les zones où un foil bien dimensionné tombe :
        - α_cruise modéré (2-3°), α_to plus chargé (6-8°)
        - calage 0°, washout léger (~-1°), s_twist négatif (-2°)
        - CG vers l'avant du milieu (équilibre stable)
    """
    seeds = []
    fl_mid = 0.5 * (BOUNDS[0][0] + BOUNDS[0][1])
    # Planform : warm-start sur la référence scénario + une variante plus petite
    planforms = [
        (WING_SPAN, WING_ROOT_CHORD, WING_TIP_CHORD),                    # référence
        (0.85 * WING_SPAN, 0.85 * WING_ROOT_CHORD, 0.85 * WING_TIP_CHORD),  # 15% plus petit
    ]
    for cg in (0.35 * (BOUNDS[1][0] + BOUNDS[1][1]),
               0.60 * (BOUNDS[1][0] + BOUNDS[1][1])):
        for ac in (2.0, 4.0):
            for sp, rc, tc in planforms:
                seeds.append(np.array([fl_mid, cg, 0.0, -1.0, -2.0, 7.0, ac, sp, rc, tc]))
    return [np.clip(s, LB, UB) for s in seeds]


def run_multistart(n_starts: int = 1) -> np.ndarray:
    print(f"\n{'='*65}")
    print(f"  MULTI-START DE — {n_starts} runs | {CASE.upper()}")
    print(f"  Profil aile : {WING_AIRFOIL_NAME}    Profil stab : {STAB_AIRFOIL_NAME}")
    print(f"  Init aile (warm-start) : b={WING_SPAN*100:.0f} cm  "
          f"c_R/T={WING_ROOT_CHORD*1000:.0f}/{WING_TIP_CHORD*1000:.0f} mm — librement optimisé")
    print(f"  Stab figé : b={STAB_SPAN*100:.0f} cm  "
          f"c_R/T={STAB_ROOT_CHORD*1000:.0f}/{STAB_TIP_CHORD*1000:.0f} mm")
    print(f"  N_VAR={N_VAR}  pop={DE_PARAMS['popsize']*N_VAR}  gen={DE_PARAMS['maxiter']}")
    print(f"{'='*65}")

    val_ref = objective(X_REF)
    print(f"  Référence (centre des bornes) : obj={val_ref:.1f} N\n")

    heuristics = _heuristic_starts()
    best_x     = X_REF.copy()
    best_val   = val_ref

    for run_idx in range(n_starts):
        seed = 42 + run_idx * 137
        _run_counter["n"] = 0

        # Sampler recréé à chaque run pour vraies populations indépendantes
        sampler     = qmc.Sobol(d=N_VAR, scramble=True, seed=seed)
        pop_size    = DE_PARAMS["popsize"] * N_VAR
        init_pop    = qmc.scale(sampler.random(pop_size), LB, UB)

        # Warm-start : X_REF + jusqu'à 4 individus heuristiques en tête
        init_pop[0] = X_REF
        for k, s in enumerate(heuristics, start=1):
            if k < pop_size:
                init_pop[k] = s

        print(f"  ─── Run {run_idx + 1}/{n_starts} ───")
        result = differential_evolution(
            objective, BOUNDS,
            init=init_pop, seed=seed,
            callback=_de_callback,
            strategy=DE_PARAMS["strategy"],
            maxiter=DE_PARAMS["maxiter"],
            tol=DE_PARAMS["tol"],
            atol=DE_PARAMS["atol"],
            popsize=DE_PARAMS["popsize"],
            mutation=DE_PARAMS["mutation"],
            recombination=DE_PARAMS["recombination"],
            workers=DE_PARAMS["workers"],
            polish=False,
            updating=DE_PARAMS["updating"],
            disp=False,
        )

        status = "✓" if result.success else "~"
        print(f"  {status} Run {run_idx+1} : obj={result.fun:.1f} N", end="")
        if result.fun < best_val:
            best_val = result.fun
            best_x   = result.x.copy()
            print("  ★ nouveau meilleur")
        else:
            print()

    # ── Affinage local L-BFGS-B ──────────────────────────────────────────────
    print(f"\n  Affinage L-BFGS-B depuis le meilleur ({best_val:.1f} N)...")
    refined = minimize(
        objective, best_x,
        method="L-BFGS-B",
        bounds=Bounds(LB, UB),
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7, "disp": False},
    )
    x_final = np.clip(refined.x, LB, UB)
    if refined.fun < best_val:
        print(f"  ✓ Affinage : {best_val:.1f} → {refined.fun:.1f} N")
        best_x = x_final
    else:
        print("  ~ Affinage non améliorant")
        best_x = np.clip(best_x, LB, UB)

    return best_x


# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPORT & REPORT
# ─────────────────────────────────────────────────────────────────────────────

def full_report(x: np.ndarray) -> None:
    """Bilan console + fiche markdown + XML XFLR5."""
    x = np.clip(x, LB, UB)
    p = decode(x)

    try:
        airplane, wing, stab, mean_chord, mast_obj, fuselage_obj = build_airplane(p)
    except Exception as e:
        print(f"Erreur construction avion : {e}")
        return

    rho = atmosphere.density()
    mu  = atmosphere.dynamic_viscosity()

    # Croisière
    op_c   = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=p["alpha_cruise"], atmosphere=atmosphere)
    aero_c = asb.AeroBuildup(airplane, op_c).run()
    L, D   = float(aero_c["L"]), float(aero_c["D"])
    Cm, CL = float(aero_c["Cm"]), float(aero_c["CL"])
    D_total = D + D_MAST

    # Décollage
    op_to   = asb.OperatingPoint(velocity=cfg["v_takeoff"], alpha=p["alpha_to"], atmosphere=atmosphere)
    aero_to = asb.AeroBuildup(airplane, op_to).run()
    L_to, D_to, CL_to = float(aero_to["L"]), float(aero_to["D"]), float(aero_to["CL"])

    # Stabilité — comparaison approx. analytique (utilisée en opti) vs réelle (AeroBuildup)
    X_np_ratio = neutral_point_ratio(wing, stab, mean_chord, p["fuselage_length"])
    SM_approx  = X_np_ratio - p["cg_ratio"]

    # SM réelle : SM = -dCm/dCL au point LINÉAIRE (α=3°, δα=0.25°).
    # IMPORTANT : on utilise VortexLatticeMethod (capture le downwash aile→stab),
    # pas AeroBuildup (modèle additif qui ignore les interactions ; vérifié :
    # AeroBuildup surestime la SM d'un facteur ~3 par rapport à VLM).
    # VLM ne gère pas les fuselages → on construit un avion réduit aux ailes.
    try:
        plane_vlm = asb.Airplane(
            wings=airplane.wings, xyz_ref=airplane.xyz_ref,
            s_ref=airplane.s_ref, c_ref=airplane.c_ref, b_ref=airplane.b_ref,
        )
        alpha_lin, d_alpha = 3.0, 0.25
        op_p   = asb.OperatingPoint(velocity=cfg["v_cruise"],
                                    alpha=alpha_lin + d_alpha, atmosphere=atmosphere)
        op_m   = asb.OperatingPoint(velocity=cfg["v_cruise"],
                                    alpha=alpha_lin - d_alpha, atmosphere=atmosphere)
        aero_p = asb.VortexLatticeMethod(plane_vlm, op_p).run()
        aero_m = asb.VortexLatticeMethod(plane_vlm, op_m).run()
        dCL    = float(aero_p["CL"]) - float(aero_m["CL"])
        if abs(dCL) < 1e-4:
            SM_real = float("nan")
        else:
            SM_real = -(float(aero_p["Cm"]) - float(aero_m["Cm"])) / dCL
    except Exception as e:
        print(f"  ~ VLM SM_real échec : {type(e).__name__}: {str(e)[:60]}")
        SM_real = float("nan")

    # Équilibre
    X_cg    = p["cg_ratio"] * mean_chord
    M_wing  = Cm * q_cruise * wing.area() * mean_chord
    M_rig = rig_mass * 9.81 * (-(X_cg - x_mast))
    M_total = M_wing + M_MAST + M_rig

    try:
        F_stab = float(aero_c["wing_aero_components"][1].L)
    except Exception:
        F_stab = float("nan")
    v_h = (stab.area() * p["fuselage_length"]) / (wing.area() * mean_chord)

    AR_w, AR_s = float(wing.aspect_ratio()), float(stab.aspect_ratio())

    # Von Mises emplanture (recalculé pour la planform courante)
    sigma_vm = von_mises_root(p["wing_root_chord"], p["wing_span"])

    print(f"\n{'='*65}")
    print(f"  RÉSULTAT FINAL — {CASE.upper()}  (V3 Paramétrique)")
    print(f"{'='*65}")
    print(f"  Aile (optimisée) : {WING_AIRFOIL_NAME}  b={p['wing_span']*100:.0f} cm  "
          f"c_R/T={p['wing_root_chord']*1000:.0f}/{p['wing_tip_chord']*1000:.0f} mm")
    print(f"  Stab (figé)      : {STAB_AIRFOIL_NAME}  b={STAB_SPAN*100:.0f} cm  "
          f"c_R/T={STAB_ROOT_CHORD*1000:.0f}/{STAB_TIP_CHORD*1000:.0f} mm")
    print(f"  {'─'*55}")
    print(f"  Traînée croisière: {D_total:6.2f} N    Finesse L/D : {L/D_total:5.2f}")
    print(f"  Traînée décollage: {D_to:6.2f} N    (multi-point objective w·D_to)")
    print(f"  α cruise / α to  : {p['alpha_cruise']:5.2f}° / {p['alpha_to']:5.2f}°")
    print(f"  Calage aile      : {p['wing_setting_angle']:5.2f}°    Twist : {p['twist']:5.2f}°")
    print(f"  Calage stab      : {p['s_twist']:5.2f}°")
    print(f"  Fuselage         : {p['fuselage_length']*100:5.1f} cm   CG : {p['cg_ratio']*100:5.1f} % c̄")
    print(f"  {'─'*55}")
    print(f"  Surface aile     : {wing.area()*1e4:5.0f} cm²   AR : {AR_w:.2f}")
    print(f"  Surface stab     : {stab.area()*1e4:5.0f} cm²   AR : {AR_s:.2f}")
    sm_real_str = f"{SM_real*100:5.1f} %" if np.isfinite(SM_real) else "   n/a"
    print(f"  Marge statique   : approx (opti) {SM_approx*100:5.1f} %   |   réelle (VLM) {sm_real_str}"
          f"    (cible [{cfg['sm_range'][0]*100:.0f}–{cfg['sm_range'][1]*100:.0f}]%)")
    print(f"  Volume de queue  : {v_h:5.3f}        (cible [{cfg['vh_range'][0]:.2f}–{cfg['vh_range'][1]:.2f}])")
    print(f"  L décollage      : {L_to:6.1f} / {WEIGHT:.1f} N   CL_to : {CL_to:.3f} / {CL_MAX_TO}")
    print(f"  Moment résiduel  : {M_total:.3f} N·m   Force stab : {F_stab:.1f} N")
    print(f"  Von Mises root   : {sigma_vm/1e6:.1f} MPa / {SIGMA_CARBONE/1e6:.0f} MPa")
    print(f"  σ_v cavitation   : {SIGMA_CAV:.2f}")

    # ── Récap des contraintes ───────────────────────────────────────────────
    Cp_min   = -(1.2 * abs(CL) + 3.0 * WING_THICKNESS_REL)
    sm_lo, sm_hi = cfg["sm_range"]
    M_tol    = 0.05 * WEIGHT * mean_chord
    checks = [
        ("L_cruise ≥ poids",        L     >= 0.99 * WEIGHT),
        ("L_takeoff ≥ poids",       L_to  >= 0.99 * WEIGHT),
        ("CL_takeoff ≤ CL_max",     CL_to <= CL_MAX_TO + 1e-3),
        ("|M_total| ≤ 5% M_ref",    abs(M_total) <= M_tol),
        ("SM (opti) dans cible",    sm_lo <= SM_approx <= sm_hi),
        ("Cavitation OK",           Cp_min >= -SIGMA_CAV),
        ("σ_VM ≤ σ_carbone",        sigma_vm <= SIGMA_CARBONE),
        ("V_h dans cible",          cfg["vh_range"][0] <= v_h <= cfg["vh_range"][1]),
        ("α_cruise ≥ -1°",          p["alpha_cruise"] >= -1.0),
    ]
    print(f"  {'─'*55}")
    n_ok = sum(1 for _, ok in checks if ok)
    print(f"  Validation : {n_ok}/{len(checks)} contraintes satisfaites")
    for name, ok in checks:
        mark = "✓" if ok else "⚠"
        print(f"    {mark}  {name}")
    print(f"{'='*65}\n")

    # ── Export ───────────────────────────────────────────────────────────────
    now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"{CASE}_v3param_{now_str}")
    os.makedirs(out_dir, exist_ok=True)

    _export_md(out_dir, p, wing, stab, mean_chord, D_total, L, D,
               SM_approx, SM_real, F_stab, v_h, M_total, L_to, CL_to, rho, mu, X_cg, AR_w, AR_s,
               sigma_vm)

    # ── Export XFLR5 + profils .dat — méthode V1 (la seule qui marche en pratique).
    # On RENOMME chaque section avec un nom unique (wing_sec_0, ...) et on écrit
    # le .dat correspondant dans airfoils/. Au load du XML, XFLR5 retrouve
    # chaque section par son nom dans le sous-dossier.
    airfoils_dir = os.path.join(out_dir, "airfoils")
    os.makedirs(airfoils_dir, exist_ok=True)

    def _export_profile_dat(af_obj, filename, name_internal):
        """Format Selig normalisé, repané à 50 pts/side (cf. V1)."""
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

    # IMPORTANT : toutes les xsecs d'une aile partagent le même objet Airfoil
    # en mémoire (WING_AIRFOIL est créé une fois et passé par référence). Si on
    # renomme `xs.airfoil.name`, on écrase aussi les sections précédentes.
    # → on COPIE l'airfoil par section avant de renommer.
    import copy
    def _rename_and_export(xsecs, prefix):
        for i, xs in enumerate(xsecs):
            n = f"{prefix}_sec_{i}"
            af_copy = copy.deepcopy(xs.airfoil)
            af_copy.name = n
            xs.airfoil = af_copy
            _export_profile_dat(af_copy, f"{n}.dat", n)

    _rename_and_export(airplane.wings[0].xsecs, "wing")
    _rename_and_export(airplane.wings[1].xsecs, "stab")
    _rename_and_export(mast_obj.xsecs,          "mast")

    # XML XFLR5 — les noms d'airfoils des xsecs viennent d'être renommés
    xml_path = os.path.join(out_dir, f"{CASE}_v3param_{now_str}_plane.xml")
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
        print(f"  ✓ XML XFLR5       : {xml_path}")
    except Exception as e:
        print(f"  ~ XML XFLR5 non exporté : {e}")
    print(f"  ✓ Profils .dat    : {airfoils_dir}/")
    print(f"  ✓ Fiche technique : {out_dir}/fiche_technique.md")

    # Sauvegarde du X optimal pour refine_3d.py (load auto)
    np.save(os.path.join(out_dir, "x_best.npy"), x)
    print(f"  ✓ X optimal       : {out_dir}/x_best.npy")

    # ── Rapport des bornes saturées ─────────────────────────────────────────
    VAR_NAMES = ["fuselage_length", "cg_ratio", "wing_setting_angle", "twist",
                 "s_twist", "alpha_to", "alpha_cruise",
                 "wing_span", "wing_root_chord", "wing_tip_chord"]
    TOL = 0.02  # 2% de la largeur de bornes
    saturated = []
    for i, (lo, hi) in enumerate(BOUNDS):
        width = hi - lo
        if x[i] - lo < TOL * width:
            saturated.append(f"{VAR_NAMES[i]} ↓ {lo:.3g}")
        elif hi - x[i] < TOL * width:
            saturated.append(f"{VAR_NAMES[i]} ↑ {hi:.3g}")
    if saturated:
        print(f"\n  ⚠ Bornes saturées ({len(saturated)}) : {', '.join(saturated)}")
        print(f"    → Considérer relâcher ces bornes pour explorer plus loin.\n")


def _export_md(out_dir, p, wing, stab, mean_chord, D_total, L, D,
               SM_approx, SM_real, F_stab, v_h, M_total, L_to, CL_to, rho, mu, X_cg, AR_w, AR_s,
               sigma_vm):
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Re_root = rho * cfg["v_cruise"] * p["wing_root_chord"] / mu
    Re_tip  = rho * cfg["v_cruise"] * p["wing_tip_chord"]  / mu
    CL_c    = L / (0.5 * rho * cfg["v_cruise"] ** 2 * wing.area())
    CD_c    = D / (0.5 * rho * cfg["v_cruise"] ** 2 * wing.area())

    lines = [
        f"# Fiche Technique — {CASE.upper()}  |  V3 Paramétrique Macroscopique",
        "", f"*Générée le {now_str}*", "", "---", "",
        "## 0. Configuration — profils & stab figés", "",
        "| Élément | Profil | Span | Corde R/T |",
        "|:---|:---|:---|:---|",
        f"| Aile  | {WING_AIRFOIL_NAME} | {p['wing_span']*100:.0f} cm | {p['wing_root_chord']*1000:.0f} / {p['wing_tip_chord']*1000:.0f} mm |",
        f"| Stab  | {STAB_AIRFOIL_NAME} | {STAB_SPAN*100:.0f} cm | {STAB_ROOT_CHORD*1000:.0f} / {STAB_TIP_CHORD*1000:.0f} mm |",
        "", "---", "",
        "## 1. Variables optimisées (7)", "",
        "| Variable | Valeur | Bornes |", "|:---|:---|:---|",
        f"| Fuselage length | {p['fuselage_length']*100:.1f} cm | [{BOUNDS[0][0]*100:.0f}–{BOUNDS[0][1]*100:.0f}] cm |",
        f"| CG ratio | {p['cg_ratio']*100:.1f}% c̄ | [{BOUNDS[1][0]*100:.0f}–{BOUNDS[1][1]*100:.0f}]% |",
        f"| Calage aile | {p['wing_setting_angle']:.2f}° | [{BOUNDS[2][0]}–{BOUNDS[2][1]}]° |",
        f"| Twist | {p['twist']:.2f}° | [{BOUNDS[3][0]}–{BOUNDS[3][1]}]° |",
        f"| Calage stab | {p['s_twist']:.2f}° | [{BOUNDS[4][0]}–{BOUNDS[4][1]}]° |",
        f"| α décollage | {p['alpha_to']:.2f}° | [{BOUNDS[5][0]}–{BOUNDS[5][1]}]° |",
        f"| α croisière | {p['alpha_cruise']:.2f}° | [{BOUNDS[6][0]}–{BOUNDS[6][1]}]° |",
        "", "---", "",
        "## 2. Conditions de vol", "",
        "| Paramètre | Valeur |", "|:---|:---|",
        f"| Poids total | {WEIGHT:.1f} N ({WEIGHT/9.81:.0f} kg) |",
        f"| V décollage | {cfg['v_takeoff']} m/s |",
        f"| V croisière | {cfg['v_cruise']} m/s |",
        f"| Re emplanture | {Re_root:.2e} |",
        f"| Re saumon | {Re_tip:.2e} |",
        "", "---", "",
        "## 3. Performances (2 points de vol)", "",
        "| Paramètre | Croisière | Décollage |", "|:---|:---|:---|",
        f"| Vitesse (m/s) | {cfg['v_cruise']} | {cfg['v_takeoff']} |",
        f"| α (°) | {p['alpha_cruise']:.2f} | {p['alpha_to']:.2f} |",
        f"| L (N) | {L:.1f} | {L_to:.1f} |",
        f"| D (N) | {D:.2f} | — |",
        f"| CL | {CL_c:.3f} | {CL_to:.3f} |",
        f"| CD | {CD_c:.4f} | — |",
        f"| D total (+ mât) | {D_total:.2f} | — |",
        f"| Finesse L/D | {L/D_total:.2f} | — |",
        "", "---", "",
        "## 4. Géométrie issue de l'opti", "",
        "| Paramètre | Aile | Stab |", "|:---|:---|:---|",
        f"| Surface (cm²) | {wing.area()*1e4:.0f} | {stab.area()*1e4:.0f} |",
        f"| Allongement | {AR_w:.2f} | {AR_s:.2f} |",
        f"| Corde moyenne (mm) | {mean_chord*1000:.0f} | {(STAB_ROOT_CHORD+STAB_TIP_CHORD)*500:.0f} |",
        "", "---", "",
        "## 5. Stabilité & Structure", "",
        "| Paramètre | Valeur | Cible |", "|:---|:---|:---|",
        f"| SM — approx (opti, Abzug) | {SM_approx*100:.1f}% | [{cfg['sm_range'][0]*100:.0f}–{cfg['sm_range'][1]*100:.0f}]% |",
        f"| SM — réelle (VLM, -dCm/dCL avec downwash) | {(f'{SM_real*100:.1f}%') if np.isfinite(SM_real) else 'n/a'} | (vérification) |",
        f"| CG | {p['cg_ratio']*100:.1f}% c̄ ({X_cg*100:.1f} cm) | [{cfg['cg_range'][0]*100:.0f}–{cfg['cg_range'][1]*100:.0f}]% |",
        f"| Moment résiduel | {M_total:.3f} N·m | < {0.05*WEIGHT*mean_chord:.2f} N·m |",
        f"| Force stab (info) | {F_stab:.1f} N | — |",
        f"| Volume de queue | {v_h:.3f} | [{cfg['vh_range'][0]:.2f}–{cfg['vh_range'][1]:.2f}] |",
        f"| Von Mises root | {sigma_vm/1e6:.1f} MPa | < {SIGMA_CARBONE/1e6:.0f} MPa |",
        f"| σ_v cavitation | {SIGMA_CAV:.2f} | — |",
        "",
    ]

    warn = []
    if not (cfg["sm_range"][0] <= SM_approx <= cfg["sm_range"][1]):
        warn.append(f"⚠️ SM approx {SM_approx*100:.1f}% hors contrainte [{cfg['sm_range'][0]*100:.0f}–{cfg['sm_range'][1]*100:.0f}]%")
    if np.isfinite(SM_real) and abs(SM_real - SM_approx) > 0.05:
        warn.append(f"⚠️ Écart SM approx/réelle > 5 points ({SM_approx*100:.1f}% vs {SM_real*100:.1f}%) — approximation Abzug à revisiter")
    if L_to < WEIGHT * 0.98:
        warn.append(f"⚠️ Portance décollage insuffisante : {L_to:.1f} / {WEIGHT:.1f} N")
    if CL_to > CL_MAX_TO:
        warn.append(f"⚠️ CL_to {CL_to:.2f} > CL_max {CL_MAX_TO}")
    if abs(M_total) > 0.05 * WEIGHT * mean_chord:
        warn.append(f"⚠️ Moment résiduel {M_total:.2f} N·m hors tolérance 5%")
    if sigma_vm > SIGMA_CARBONE:
        warn.append(f"⚠️ Von Mises {sigma_vm/1e6:.0f} MPa > admissible")
    if warn:
        lines += ["## ⚠️ Avertissements", ""] + [f"- {w}" for w in warn] + [""]

    with open(os.path.join(out_dir, "fiche_technique.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 10. POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_starts = phy["search_space"]["N_starts"]
    x_best   = run_multistart(n_starts=N_starts)
    full_report(x_best)