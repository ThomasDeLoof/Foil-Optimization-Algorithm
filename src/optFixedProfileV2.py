# =================================================================================
# Hydrofoil Optimization V3 — Paramétrique Macroscopique
# -------------------------------------------------------------------------------

import os
import re
import sys
import datetime as dt
import warnings
import numpy as np
import yaml
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.stats import qmc

import aerosandbox as asb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from config.water_atmosphere import Water as Atmosphere


# ── Couleurs ANSI pour la console (auto-désactivées si non-TTY) ─────────────
class C:
    _ON = sys.stdout.isatty()
    @classmethod
    def _w(cls, code, t): return f"\033[{code}m{t}\033[0m" if cls._ON else t
    @classmethod
    def head(cls, t): return cls._w("1;36", t)    # cyan bold (titres)
    @classmethod
    def sec(cls, t):  return cls._w("36",   t)    # cyan (sections)
    @classmethod
    def ok(cls, t):   return cls._w("32",   t)    # vert
    @classmethod
    def warn(cls, t): return cls._w("33",   t)    # jaune
    @classmethod
    def dim(cls, t):  return cls._w("2",    t)    # gris/dim
    @classmethod
    def bold(cls, t): return cls._w("1",    t)    # gras

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
_fuse_len_min = cfg["fuselage_length_bounds"][0]
if STAB_FUSE_OFFSET > _fuse_len_min:
    raise ValueError(
        f"stab.fuselage_offset ({STAB_FUSE_OFFSET*100:.1f} cm) > "
        f"fuselage_length_bounds[0] ({_fuse_len_min*100:.1f} cm) — "
        f"le BA stab serait avant le nez fuselage."
    )

# ── Mât (constants) ───────────────────────────────────────────────────────────
chord_wl = mast_chord_bot * (1 - profondeur_imm / mast_length) \
         + mast_chord_top * (profondeur_imm / mast_length)
q_cruise = 0.5 * atmosphere.density() * cfg["v_cruise"] ** 2
D_MAST   = q_cruise * ((mast_chord_bot + chord_wl) / 2) * profondeur_imm * cd_mast
M_MAST   = -D_MAST * (profondeur_imm / 2)

# ── Limites mécaniques — dimensionnement fatigue ──
# σ_ult = allowable "design" stratifié carbone-époxy (~400 MPa, knockdowns implicites).
# σ_admissible = σ_ult × fatigue_ratio (cyclique >10⁶ cycles) ≈ 160 MPa.
# La contrainte de design compare σ_pic (charge dynamique × LOAD_PEAK_FACTOR) à σ_admissible.
SIGMA_ULTIMATE     = phy["wing"]["ultimate_stress_mpa"] * 1e6
FATIGUE_RATIO      = phy["wing"]["fatigue_allowable_ratio"]
LOAD_PEAK_FACTOR   = phy["wing"]["load_peak_factor"]
SIGMA_ADMISSIBLE   = FATIGUE_RATIO * SIGMA_ULTIMATE          # ex: 0.40 × 400 = 160 MPa
P_ATM         = atmosphere.pressure()
P_VAPOR       = atmosphere.vapor_pressure()
SIGMA_CAV     = (P_ATM + atmosphere.density() * 9.81 * profondeur_imm - P_VAPOR) \
              / (0.5 * atmosphere.density() * cfg["v_cruise"] ** 2)

# ─────────────────────────────────────────────────────────────────────────────
# 2. GÉOMÉTRIE — profils et dimensions 
# ─────────────────────────────────────────────────────────────────────────────

WING_AIRFOIL_NAME = cfg["wing_airfoil"]
STAB_AIRFOIL_NAME = phy["stab"]["airfoil"]

# Taux de tip — tip = TIP_RATIO × root (corde au saumon dérivée, pas variable)
TIP_RATIO_W = phy["wing"]["tip_chord_ratio"]
TIP_RATIO_S = phy["stab"]["tip_chord_ratio"]

# Warm-start de la planform aile : on prend la référence du scénario si fournie
# (scenarios.yaml), sinon les valeurs par défaut de parameters.yaml.
WING_SPAN         = cfg.get("wing_span_init",       phy["wing"]["span_init"])
WING_ROOT_CHORD   = cfg.get("wing_root_chord_init", phy["wing"]["root_chord_init"])
WING_TIP_CHORD    = WING_ROOT_CHORD * TIP_RATIO_W

# Stab warm-start
STAB_SPAN         = cfg["stab_span"]
STAB_ROOT_CHORD   = cfg["stab_root_chord"]
STAB_TIP_CHORD    = STAB_ROOT_CHORD * TIP_RATIO_S

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
# NB : α_cruise N'EST PAS variable d'opti — il est dérivé en solve interne
# (L = WEIGHT) à chaque évaluation, ce qui garantit l'équilibre vertical.
# Cordes au saumon dérivées : tip = TIP_RATIO × root (pas de variable d'opti
# dédiée, voir TIP_RATIO_W / TIP_RATIO_S).
# x = [fuselage_length, cg_ratio, wing_setting_angle, twist, s_twist, alpha_to,
#      wing_span, wing_root_chord, stab_span, stab_root_chord]

# Wing bornes : scenario override (calage discipline-spécifique) avec fallback
# sur les defaults globaux de parameters.yaml.
_wing_span_bounds = tuple(cfg.get("wing_span_bounds",
                                  phy["wing"].get("span_bounds", [0.75, 1.30])))
_wing_root_bounds = tuple(cfg.get("wing_root_chord_bounds",
                                  phy["wing"].get("root_chord_bounds", [0.10, 0.30])))

BOUNDS = [
    tuple(cfg["fuselage_length_bounds"]),                     #  0 fuselage_length (m)
    tuple(cfg["cg_range"]),                                   #  1 cg_ratio (-)
    tuple(phy["wing"].get("calage_bounds", [-2.0, 5.0])),     #  2 wing_setting_angle (°)
    tuple(phy["wing"].get("twist_bounds",  [-5.0, 0.5])),     #  3 twist (°)
    tuple(phy["stab"]["twist_bounds"]),                       #  4 s_twist (°)
    tuple(phy["alpha"]["bounds"]),                            #  5 alpha_to (°)
    _wing_span_bounds,                                        #  6 wing_span (m)
    _wing_root_bounds,                                        #  7 wing_root_chord (m)
    tuple(phy["stab"]["span_bounds"]),                        #  8 stab_span (m)
    tuple(phy["stab"]["root_chord_bounds"]),                  #  9 stab_root_chord (m)
]
N_VAR = len(BOUNDS)
LB    = np.array([b[0] for b in BOUNDS])
UB    = np.array([b[1] for b in BOUNDS])

STAB_AR_RANGE = tuple(phy["stab"].get("aspect_ratio_range", [4.0, 14.0]))


def decode(x: np.ndarray) -> dict:
    """Découpe le vecteur DE → dictionnaire de paramètres macroscopiques.
    Les cordes au saumon sont dérivées (root × TIP_RATIO) — non variables d'opti."""
    root_w = float(x[7])
    root_s = float(x[9])
    return {
        "fuselage_length":    float(x[0]),
        "cg_ratio":           float(x[1]),
        "wing_setting_angle": float(x[2]),
        "twist":              float(x[3]),
        "s_twist":            float(x[4]),
        "alpha_to":           float(x[5]),
        "wing_span":          float(x[6]),
        "wing_root_chord":    root_w,
        "wing_tip_chord":     root_w * TIP_RATIO_W,
        "stab_span":          float(x[8]),
        "stab_root_chord":    root_s,
        "stab_tip_chord":     root_s * TIP_RATIO_S,
    }


# Warm-start : milieu des bornes, planform aile + stab initialisés sur la
# géométrie de référence du scénario.
X_REF = np.array([(b[0] + b[1]) / 2 for b in BOUNDS])
X_REF[6]  = WING_SPAN
X_REF[7]  = WING_ROOT_CHORD
X_REF[8]  = STAB_SPAN
X_REF[9]  = STAB_ROOT_CHORD
X_REF = np.clip(X_REF, LB, UB)

# ─────────────────────────────────────────────────────────────────────────────
# 4. CONSTRUCTION DE L'AVION
# ─────────────────────────────────────────────────────────────────────────────

def build_airplane(p: dict) -> tuple:
    """
    Assemble l'aile, le fuselage, le mât et le stabilisateur AeroSandBox.

    Géométrie ancrée sur le BORD DE FUITE :
        x_te(r) = x_te_root + r^sweep_power × (span/2 × tan(sweep_deg))
        x_le(r) = x_te(r) − c(r)
        c(r)   = corde elliptique
    Le TE est donc monotone par construction, peu importe la loi de corde
    (l'elliptique a dc/dr → −∞ au saumon, ce qui faisait remonter le TE en
    fin de span avec l'ancien parametrage QC-ancré).
    """
    SWEEP_POWER_W = 1.5   # courbure du TE aile (1=linéaire, >1=courbé)
    SWEEP_POWER_S = 1.5   # idem stab

    # ── Aile principale ──────────────────────────────────────────────────────
    span_w  = p["wing_span"]
    root_w  = p["wing_root_chord"]
    tip_w   = min(p["wing_tip_chord"], root_w * 0.95)  # sécurité tip < root
    K_sweep_w = (span_w / 2) * np.tan(np.radians(sweep_deg))

    wing_xsecs = []
    for i in range(N_WING):
        r = i / (N_WING - 1)

        c_dist = tip_w + (root_w - tip_w) * np.sqrt(max(1 - r ** 2, 0))
        x_te   = root_w + (r ** SWEEP_POWER_W) * K_sweep_w
        x_le   = x_te - c_dist

        z_pos  = -((r ** 2) * span_w / 2) * np.tan(np.radians(wing_anhedral_deg))

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
    stab_span_p = p["stab_span"]
    stab_root_p = p["stab_root_chord"]
    stab_tip_p  = min(p["stab_tip_chord"], stab_root_p * 0.95)  # safety tip<root
    x_stab_root = x_fuselage_start + p["fuselage_length"] - STAB_FUSE_OFFSET
    K_sweep_s = (stab_span_p / 2) * np.tan(np.radians(STAB_SWEEP_DEG))

    stab_xsecs = []
    for i in range(N_STAB):
        r = i / (N_STAB - 1)

        c_s    = stab_tip_p + (stab_root_p - stab_tip_p) * np.sqrt(max(1 - r ** 2, 0))
        x_te_s = stab_root_p + (r ** SWEEP_POWER_S) * K_sweep_s
        x_le_s = x_te_s - c_s
        z_s    = (r ** 1.5 * stab_span_p / 2) * np.tan(np.radians(STAB_DIHEDRAL_DEG))

        stab_xsecs.append(asb.WingXSec(
            xyz_le=[x_stab_root + x_le_s, r * stab_span_p / 2, z_s],
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

def von_mises_root(chord_root: float, span: float,
                   load_factor: float = 1.0) -> float:
    """
    Von Mises à l'emplanture — coque carbone + âme polystyrène (négligée).
    `load_factor` multiplie la charge statique pour estimer le PIC dynamique
    (maneuvering, vagues, pumping). Le résultat est ensuite comparé à
    SIGMA_ADMISSIBLE (= fatigue_ratio × σ_ult, ~120 MPa) — pas à σ_ult.
    """
    weight_eff = WEIGHT * load_factor
    span_semi  = span / 2.0
    t_max      = WING_THICKNESS_REL * chord_root
    a, b       = chord_root / 2.0, t_max / 2.0

    t_skin     = min(WING_SKIN_THICK, 0.9 * b)
    a_in, b_in = max(a - t_skin, 0.0), max(b - t_skin, 0.0)
    I_xx  = (np.pi / 4.0) * (a * b ** 3 - a_in * b_in ** 3)
    A_enc = np.pi * a * b

    M_flex = (weight_eff / 2.0) * (span_semi / 4.0)
    M_tors = (weight_eff / 2.0) * (0.05 * chord_root) * (span_semi / 2.0)

    sigma_flex = M_flex * b / (I_xx + 1e-12)
    tau        = M_tors / (2.0 * A_enc * t_skin + 1e-12)
    return float(np.sqrt(sigma_flex ** 2 + 3 * tau ** 2))

# ─────────────────────────────────────────────────────────────────────────────
# 6. DYNAMIQUE DE TANGAGE — Cm_α, SM et ω_n issus directement d'AeroBuildup.
# ─────────────────────────────────────────────────────────────────────────────
# Toutes les métriques de stabilité (Cm_α, SM/c̄, SM/l_t, ω_n) sont calculées
# par différence finie sur les deux points AeroBuildup déjà obtenus pendant le
# trim de croisière (alpha_lo, alpha_hi) — donc à coût zéro et avec ~3 % près
# de la vérité VLM. L'ancienne formule analytique (Helmbold + V_H + de_da
# calibré linéairement) a été supprimée : elle introduisait 15-25 % d'erreur
# sur Cm_α et donc sur ω_n, ce qui faisait que la cible pilotability était
# souvent ratée silencieusement.

# Inertie de tangage du système pilote + board + rig
_M_TOTAL    = phy["pilot"]["mass_kg"] + phy["board"]["mass_kg"] + rig_mass
_R_GYR      = phy["pilot"].get("gyration_radius_m", 0.30)
I_YY_SYSTEM = _M_TOTAL * _R_GYR ** 2

# Plage ω_n cible pour la discipline courante (calibrée freeride par scénario
# dans scenarios.yaml). Un seul scope — pas de subdivision par niveau pilote.
PILOT_FREQ_LO, PILOT_FREQ_HI = cfg["pilotability_freq"]

# Tolérance sur le moment résiduel — le pilote compense via sa stance, donc
# la grandeur de référence n'est PAS la corde aile (héritage aviation aberrant
# pour un hydrofoil) mais la capacité de trim pilote, exprimée en N·m.
# Cf. parameters.yaml#pilot.trim_moment_tolerance_N_m.
M_TOL_TRIM = phy["pilot"].get("trim_moment_tolerance_N_m", 25.0)

# Pente CL_α du stab (NACA 0012-like), utilisée pour l'autorité de contrôle
# dF_stab/dα = CL_α × q × S_stab. Cible définie par scénario.
CL_ALPHA_STAB_PER_DEG = phy["stab"].get("cl_alpha_per_deg", 0.10)


def get_pilot_freq_range() -> tuple:
    """Retourne (f_lo, f_hi) — la cible ω_n du scénario courant."""
    return PILOT_FREQ_LO, PILOT_FREQ_HI


def pitch_frequency_hz(Cm_alpha: float, q: float, S: float, c_ref: float) -> float:
    """
    Fréquence naturelle du mode short-period (Hz).
        ω_n² = -Cm_α × q × S × c̄ / I_yy        [I_yy ≈ m_total × r_gyr²]
    Renvoie NaN si Cm_α > 0 (foil instable, ω_n imaginaire).
    """
    omega_n_sq = -Cm_alpha * q * S * c_ref / max(I_YY_SYSTEM, 1e-9)
    if omega_n_sq <= 0:
        return float("nan")
    return float(np.sqrt(omega_n_sq) / (2.0 * np.pi))


# ─────────────────────────────────────────────────────────────────────────────
# 7. FONCTION OBJECTIF
# ─────────────────────────────────────────────────────────────────────────────

K1 = 4000.0   # contraintes critiques (portance, équilibre, structure)
K2 = 2000.0   # contraintes de pilotage (SM, surface)
K3 = 1000.0    # contraintes de confort  (Vh, force stab, cavitation)

_run_counter = {"n": 0}


def soft_penalty(val: float, lo: float, hi: float, ref: float) -> float:
    """Pénalité linéaire normalisée continue.

    P = |violation| / ref (exp=1). Gradient constant — pas d'écrasement près
    du bord. Contrairement au quadratique (∂P/∂x → 0 à la frontière, qui
    laissait les petites violations « gratuites »), ici une violation de 10%
    coûte 0.1 vraies unités, pas 0.01.
    """
    ref_safe = abs(ref) + 1e-9
    if val < lo:
        return (lo - val) / ref_safe
    if val > hi:
        return (val - hi) / ref_safe
    return 0.0


def trim_alpha_for_lift(airplane, target_L: float,
                        alpha_lo: float = 0.0, alpha_hi: float = 3.0) -> tuple:
    """
    Trouve alpha_cruise tel que L(alpha) ≈ target_L par interpolation linéaire
    (L est quasi-linéaire en α dans le régime AeroBuildup pré-stall).

    Retourne (alpha_trim, aero_at_trim, bracket) où bracket = dict avec aero_lo,
    aero_hi, alpha_lo, alpha_hi. Le bracket donne gratuitement dCm/dα = Cm_α
    réel (issu de 2 AeroBuildup déjà calculés) — utilisé par pitch_dynamics_from_aero.
    """
    op_lo = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_lo, atmosphere=atmosphere)
    op_hi = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_hi, atmosphere=atmosphere)
    aero_lo = asb.AeroBuildup(airplane, op_lo).run()
    aero_hi = asb.AeroBuildup(airplane, op_hi).run()
    bracket = {"aero_lo": aero_lo, "aero_hi": aero_hi,
               "alpha_lo": alpha_lo, "alpha_hi": alpha_hi}
    L_lo, L_hi = float(aero_lo["L"]), float(aero_hi["L"])
    if abs(L_hi - L_lo) < 1.0:
        return float(alpha_lo), aero_lo, bracket  # plateau (rare, foil saturé)
    alpha_trim = alpha_lo + (target_L - L_lo) / (L_hi - L_lo) * (alpha_hi - alpha_lo)
    # Clamp aux bornes physiques (α_cruise raisonnable)
    alpha_trim = max(-3.0, min(12.0, alpha_trim))
    op_trim = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_trim, atmosphere=atmosphere)
    aero_trim = asb.AeroBuildup(airplane, op_trim).run()
    return float(alpha_trim), aero_trim, bracket


def pitch_dynamics_from_aero(bracket: dict, mean_chord: float,
                             l_t: float) -> dict:
    """
    Dynamique de tangage calculée DIRECTEMENT depuis les 2 points AeroBuildup
    du bracket de trim (déjà calculés par trim_alpha_for_lift — coût zéro).

        Cm_α (rad⁻¹) = dCm / dα
        SM/c̄         = -dCm / dCL    (négatif si stable, on retourne |·|)
        SM (m)       = (SM/c̄) × c̄
        SM/l_t       = SM / l_t

    Précision : ~3 % près de VLM, vs ~15-20 % pour la formule analytique
    avec de_da calibré linéairement.
    """
    aero_lo = bracket["aero_lo"]
    aero_hi = bracket["aero_hi"]
    dCm   = float(aero_hi["Cm"]) - float(aero_lo["Cm"])
    dCL   = float(aero_hi["CL"]) - float(aero_lo["CL"])
    dα_rad = np.radians(bracket["alpha_hi"] - bracket["alpha_lo"])

    Cm_alpha = dCm / dα_rad if abs(dα_rad) > 1e-9 else 0.0
    sm_c     = -dCm / dCL   if abs(dCL)    > 1e-6 else 0.0
    sm_len   = sm_c * mean_chord
    sm_lt    = sm_len / l_t if abs(l_t)    > 1e-6 else 0.0

    return {"Cm_alpha": Cm_alpha, "SM_chord": sm_c,
            "SM_abs":   sm_len,   "SM_lt":    sm_lt}


def objective(x: np.ndarray) -> float:
    """
    Minimise la traînée totale (D_aero + D_mât) sous contraintes douces.
    α_cruise est DÉRIVÉ (solveur trim L=WEIGHT), pas une variable d'opti.
    """
    x = np.clip(x, LB, UB)
    p = decode(x)

    try:
        airplane, wing, stab, mean_chord, _, _ = build_airplane(p)
    except Exception:
        return 1e6  # exception structurelle AeroSandBox

    # ── Croisière : α_cruise solvé en interne pour avoir L = WEIGHT ─────────
    try:
        alpha_trim, aero_c, bracket = trim_alpha_for_lift(airplane, target_L=WEIGHT)
        p["alpha_cruise"] = alpha_trim
        L  = float(aero_c["L"])
        D  = float(aero_c["D"])
        Cm = float(aero_c["Cm"])
        CL = float(aero_c["CL"])
    except Exception:
        return 1e6

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

    # Portance croisière = poids
    penalty += K1 * soft_penalty(L, WEIGHT, WEIGHT, ref=WEIGHT)

    # Portance décollage : L_to ≥ poids + CL_to ≤ marge × CL_max_to.
    penalty += K1 * soft_penalty(L_to, WEIGHT, np.inf, ref=WEIGHT)
    cl_to_target = cfg["takeoff_cl_margin"] * CL_MAX_TO
    penalty += K1 * soft_penalty(CL_to, -np.inf, cl_to_target, ref=CL_MAX_TO)

    # Régime d'opération sain : α_cruise ≥ -1° sur profil cambré
    # (en α<-1° on tombe en zone non-linéaire négative pour NACA modérément cambré, Cm(α) devient erratique
    penalty += K2 * soft_penalty(p["alpha_cruise"], -1.0, np.inf, ref=2.0)

    # Équilibre de tangage — tolérance = capacité de trim du pilote (stance),
    # cf. M_TOL_TRIM dans parameters.yaml. Pas de tolérance aviation aberrante.
    X_cg    = p["cg_ratio"] * mean_chord
    M_wing  = Cm * q_cruise * S_wing * mean_chord
    M_rig = rig_mass * 9.81 * (-(X_cg - x_mast))
    M_total = M_wing + M_MAST + M_rig
    penalty += K1 * soft_penalty(M_total, -M_TOL_TRIM, +M_TOL_TRIM, ref=M_TOL_TRIM)

    # Force stab — éviter la solution tandem
    try:
        F_stab = float(aero_c["wing_aero_components"][1].L)
        f_lo, f_hi = cfg["stab_load_range"]   # négatif = déportance
        penalty += K2 * soft_penalty(F_stab, f_lo, f_hi, ref=abs(f_lo))
    except Exception:
        pass  # pas de pénalité si AeroBuildup n'a pas la décomposition par surface

    # Autorité de contrôle stab : dF_stab/dα = CL_α × q × S_stab. C'est la force
    # aero modulée par degré de variation d'α — ce que ressent le pilote quand
    # une vague tilt le foil ou qu'il change de stance. Contrainte non capturée
    # par ω_n (qui n'impose que la stiffness totale, pas la taille stab). Sans
    # cette borne, l'opti garde un stab minuscule à CL élevé → force OK en
    # steady-state mais autorité de correction insuffisante en pratique.
    dFda_target = cfg.get("stab_control_authority_min_N_per_deg", 0.0)
    if dFda_target > 0:
        dFda_stab = CL_ALPHA_STAB_PER_DEG * q_cruise * S_stab
        penalty += K2 * soft_penalty(dFda_stab, dFda_target, np.inf, ref=dFda_target)

    # Pilotabilité : Cm_a dérivé DIRECTEMENT du bracket AeroBuildup (coût zéro,
    try:
        c_stab_mean = 0.5 * (p["stab_root_chord"] + p["stab_tip_chord"])
        l_t = (x_fuselage_start + p["fuselage_length"] - STAB_FUSE_OFFSET
               + 0.25 * c_stab_mean) - 0.25 * mean_chord
        pd = pitch_dynamics_from_aero(bracket, mean_chord, l_t)
        omega_n = pitch_frequency_hz(pd["Cm_alpha"], q_cruise, S_wing, mean_chord)
        f_lo, f_hi = get_pilot_freq_range()
        if np.isfinite(omega_n):
            penalty += K1 * soft_penalty(omega_n, f_lo, f_hi, ref=0.5*(f_lo+f_hi))
        else:
            penalty += K1 * 4.0  # foil instable → forte pénalité
    except Exception:
        penalty += K1

    # ── Pénalités auxiliaires ────────────────────────────────

    # Contrainte structurelle : von Mises emplanture (coque carbone)
    # Contrainte fatigue : pic dynamique vs limite admissible (≠ rupture statique).
    sigma_vm_peak = von_mises_root(p["wing_root_chord"], p["wing_span"],
                                    load_factor=LOAD_PEAK_FACTOR)
    penalty += K1 * soft_penalty(sigma_vm_peak, 0.0, SIGMA_ADMISSIBLE, ref=SIGMA_ADMISSIBLE)

    # Cavitation : Cp_min ≥ −σ_v
    Cp_min = -(1.2 * abs(CL) + 3.0 * WING_THICKNESS_REL)
    penalty += K3 * soft_penalty(Cp_min, -SIGMA_CAV, np.inf, ref=SIGMA_CAV)

    # Garde-fou sur l'AR du stab (évite stabs trop carrés ou trop fins).
    AR_stab_phys = (p["stab_span"] ** 2) / max(S_stab, 1e-6)
    ar_lo, ar_hi = STAB_AR_RANGE
    penalty += K2 * soft_penalty(AR_stab_phys, ar_lo, ar_hi, ref=0.5*(ar_lo+ar_hi))

    # Cap AR aile — raisons non-hydro (fabrication carbone, stiffness, impact,
    # roll inertia, pilotabilité freeride). Référence industrielle par discipline.
    ar_wing_max = cfg.get("wing_aspect_ratio_max", np.inf)
    if np.isfinite(ar_wing_max):
        AR_wing_phys = (p["wing_span"] ** 2) / max(S_wing, 1e-6)
        penalty += K2 * soft_penalty(AR_wing_phys, -np.inf, ar_wing_max, ref=ar_wing_max)

    # Objectif multi-point : D_cruise + W_TO × D_takeoff
    W_TAKEOFF = 0.3
    return D_total + W_TAKEOFF * D_to + penalty


# ─────────────────────────────────────────────────────────────────────────────
# 8. OPTIMISATION — DIFFERENTIAL EVOLUTION + RAFFINAGE 3D (LiftingLine)
# ─────────────────────────────────────────────────────────────────────────────

DE_PARAMS = {
    "strategy":      "best1bin",
    "maxiter":       40,
    "popsize":       25,           # 25 × 7 = 175 individus — suffisant pour 7 var
    "tol":           1e-4,
    "atol":          1e-3,
    "mutation":      (0.5, 1.0),
    "recombination": 0.85,
    "workers":       -1,
    "polish":        False,
    "updating":      "deferred",
}


def _de_callback(_xk: np.ndarray, convergence: float) -> bool:
    """Affichage léger toutes les 5 générations (sans ré-évaluer xk)."""
    _run_counter["n"] += 1
    gen = _run_counter["n"]
    if gen % 5 != 0:
        return False
    pct = min(gen / DE_PARAMS["maxiter"] * 100.0, 100.0)
    n_full = int(pct / 10)
    bar = "█" * n_full + "░" * (10 - n_full)
    print(f"    G{gen:3d} {C.sec('[' + bar + ']')} {pct:4.0f}%   "
          f"{C.dim(f'conv = {convergence:.2e}')}")
    return False


def _heuristic_starts() -> list:
    """
    Individus heuristiques DE — couvrent fuselage moyen + diverses planforms
    aile/stab + 2 CG (avancé/centré). Cordes au saumon dérivées (TIP_RATIO).
    """
    seeds = []
    fl_mid = 0.5 * (BOUNDS[0][0] + BOUNDS[0][1])
    # Planform aile : référence + variante 15 % plus petite
    wing_planforms = [
        (WING_SPAN, WING_ROOT_CHORD),
        (0.85 * WING_SPAN, 0.85 * WING_ROOT_CHORD),
    ]
    # Stab : référence + variante un peu plus grosse (autorité tangage)
    stab_planforms = [
        (STAB_SPAN, STAB_ROOT_CHORD),
        (1.10 * STAB_SPAN, 1.10 * STAB_ROOT_CHORD),
    ]
    for cg in (0.35 * (BOUNDS[1][0] + BOUNDS[1][1]),
               0.60 * (BOUNDS[1][0] + BOUNDS[1][1])):
        for w_sp, w_rc in wing_planforms:
            for s_sp, s_rc in stab_planforms:
                # x = [fl, cg, calage, twist, s_twist, α_to,
                #      w_span, w_root, s_span, s_root]  (tips dérivés)
                seeds.append(np.array([fl_mid, cg, 0.0, -1.0, -2.0, 7.0,
                                       w_sp, w_rc, s_sp, s_rc]))
    return [np.clip(s, LB, UB) for s in seeds]


def run_multistart(n_starts: int = 1) -> np.ndarray:
    HBAR = "═" * 70
    print()
    print(C.head(HBAR))
    print(C.head(f"  HYDROFOIL OPTIMIZATION  ·  {CASE.upper()} freeride"))
    print(f"  {C.dim('Aile')}  {WING_AIRFOIL_NAME:<10}  warm-start "
          f"{WING_SPAN*100:5.1f} cm  /  {WING_ROOT_CHORD*1000:.0f} mm root")
    print(f"  {C.dim('Stab')}  {STAB_AIRFOIL_NAME:<10}  warm-start "
          f"{STAB_SPAN*100:5.1f} cm  /  {STAB_ROOT_CHORD*1000:.0f} mm root")
    _pop  = DE_PARAMS["popsize"] * N_VAR
    _gen  = DE_PARAMS["maxiter"]
    _runs = f"{n_starts} run" + ("s" if n_starts > 1 else "")
    print(f"  {C.dim(f'{N_VAR} vars · pop={_pop} · gen={_gen} · {_runs}')}")
    print(C.head(HBAR))

    val_ref = objective(X_REF)
    print(f"\n  {C.dim('Référence (centre des bornes) :')}  obj = {val_ref:.1f} N\n")

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

        print(C.sec(f"  ── Run {run_idx + 1}/{n_starts} " + "─" * (55 - len(str(run_idx+1)) - len(str(n_starts)))))
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

        mark = C.ok("✓") if result.success else C.warn("~")
        line = f"  {mark} Run {run_idx+1} terminé   obj = {result.fun:.1f} N"
        if result.fun < best_val:
            best_val = result.fun
            best_x   = result.x.copy()
            print(line + "   " + C.bold(C.ok("★ nouveau meilleur")))
        else:
            print(line)

    # ── Affinage 3D (LiftingLine + Nelder-Mead borné) ────────────────────────
    # LiftingLine capture le downwash et le induced drag, ce qui re-trimme correctement les angles
    print()
    print(C.sec("  ── Raffinage 3D " + "─" * 53))
    try:
        import optFixedProfileRefine3d as R3
        print(f"  LiftingLine (planform fixée) depuis le meilleur DE  "
              f"{C.dim(f'(obj DE = {best_val:.1f} N)')}")
        best_x_clipped = np.clip(best_x, LB, UB)
        x_refined, J_refined, n_iter = R3.refine_trim_3d(best_x_clipped, maxiter=80)
        # Comparer : on évalue x_refined en cohérence avec l'objective DE (AB),
        # pour décider si on garde le raffinage 3D ou la solution DE pure.
        val_refined_ab = objective(x_refined)
        print(f"  {C.ok('✓')} {n_iter} itérations   J_3D = {J_refined:.1f} N   "
              f"{C.dim(f'(2D-équivalent = {val_refined_ab:.1f} N)')}")
        # On garde TOUJOURS x_refined : c'est la solution 3D physiquement correcte,
        # même si son score "AeroBuildup" est moins bon (AB sous-estime L à α bas).
        best_x = x_refined
    except Exception as e:
        print(f"  {C.warn('~')} échec ({type(e).__name__}: {str(e)[:60]}) — fallback solution DE")
        best_x = np.clip(best_x, LB, UB)

    return best_x


# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPORT & REPORT
# ─────────────────────────────────────────────────────────────────────────────

def next_output_dir(suffix: str = "", out_root: str = "outputs") -> str:
    """
    Construit le prochain dossier de sortie au format :
        outputs/{case}_{YYYYMMDD}_{NN}/
    NN s'incrémente automatiquement par jour
    """
    today = dt.datetime.now().strftime("%Y%m%d")
    prefix = f"{CASE}_{today}"
    os.makedirs(out_root, exist_ok=True)
    suffix_re = re.escape("_" + suffix) if suffix else ""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+){suffix_re}$")
    nums = [int(m.group(1)) for d in os.listdir(out_root) if (m := pat.match(d))]
    n = max(nums) + 1 if nums else 1
    name = f"{prefix}_{n:02d}" + (f"_{suffix}" if suffix else "")
    return os.path.join(out_root, name)


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

    # Croisière : α_cruise dérivé par trim L = WEIGHT (cohérent avec objective).
    alpha_trim, aero_c, bracket = trim_alpha_for_lift(airplane, target_L=WEIGHT)
    p["alpha_cruise"] = alpha_trim
    L, D   = float(aero_c["L"]), float(aero_c["D"])
    Cm, CL = float(aero_c["Cm"]), float(aero_c["CL"])
    D_total = D + D_MAST

    # Décollage
    op_to   = asb.OperatingPoint(velocity=cfg["v_takeoff"], alpha=p["alpha_to"], atmosphere=atmosphere)
    aero_to = asb.AeroBuildup(airplane, op_to).run()
    L_to, D_to, CL_to = float(aero_to["L"]), float(aero_to["D"]), float(aero_to["CL"])

    # Pitch dynamics RÉELLE issue du bracket AeroBuildup. Plus de comparaison
    # analytique vs VLM — AeroBuildup est désormais la source de vérité
    # (~3 % près de VLM, ce qui suffit largement).
    c_stab_mean = 0.5 * (p["stab_root_chord"] + p["stab_tip_chord"])
    l_t = (x_fuselage_start + p["fuselage_length"] - STAB_FUSE_OFFSET
           + 0.25 * c_stab_mean) - 0.25 * mean_chord
    pd = pitch_dynamics_from_aero(bracket, mean_chord, l_t)
    omega_n = pitch_frequency_hz(pd["Cm_alpha"], q_cruise, wing.area(), mean_chord)

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

    # Von Mises emplanture — statique cruise (1g) ET pic dynamique (n_g équivalent).
    # Le critère de design est sur le PIC : σ_peak < σ_admissible (fatigue).
    sigma_vm_static = von_mises_root(p["wing_root_chord"], p["wing_span"], load_factor=1.0)
    sigma_vm        = von_mises_root(p["wing_root_chord"], p["wing_span"],
                                     load_factor=LOAD_PEAK_FACTOR)

    # ── Métriques dérivées pour l'affichage ─────────────────────────────────
    f_lo, f_hi   = get_pilot_freq_range()
    Cp_min       = -(1.2 * abs(CL) + 3.0 * WING_THICKNESS_REL)
    M_tol        = M_TOL_TRIM
    cl_to_target = cfg["takeoff_cl_margin"] * CL_MAX_TO
    Re_root      = rho * cfg["v_cruise"] * p["wing_root_chord"] / mu
    Re_tip       = rho * cfg["v_cruise"] * p["wing_tip_chord"]  / mu

    HBAR = "═" * 70
    def SEC(title):
        return C.sec(f"  ── {title} " + "─" * max(0, 60 - len(title)))

    print()
    print(C.head(HBAR))
    print(C.head(f"  RÉSULTAT  ·  {CASE.upper()} freeride"))
    print(C.head(HBAR))

    # ── Géométrie ──
    print(SEC("Géométrie"))
    ar_wing_cap = cfg.get("wing_aspect_ratio_max", float("inf"))
    ar_w_mark = C.ok("✓") if AR_w <= ar_wing_cap else C.warn("⚠")
    ar_dim = f" {C.dim(f'(cap {ar_wing_cap:.1f})')}" if np.isfinite(ar_wing_cap) else ""
    print(f"    Aile      {WING_AIRFOIL_NAME:<10}  span {p['wing_span']*100:5.1f} cm   "
          f"root/tip {p['wing_root_chord']*1000:3.0f}/{p['wing_tip_chord']*1000:2.0f} mm   "
          f"AR {AR_w:5.2f}{ar_dim} {ar_w_mark}")
    print(f"    Stab      {STAB_AIRFOIL_NAME:<10}  span {p['stab_span']*100:5.1f} cm   "
          f"root/tip {p['stab_root_chord']*1000:3.0f}/{p['stab_tip_chord']*1000:2.0f} mm   AR {AR_s:5.2f}")
    print(f"    Fuselage  {p['fuselage_length']*100:5.1f} cm    CG {p['cg_ratio']*100:5.1f} % c̄    "
          f"V_h {v_h:.3f}  {C.dim('(info)')}")

    # ── Trim ──
    print(SEC("Trim"))
    print(f"    α cruise    {p['alpha_cruise']:+6.2f}°    "
          f"α décollage   {p['alpha_to']:+6.2f}°")
    print(f"    Calage aile {p['wing_setting_angle']:+6.2f}°    Twist        {p['twist']:+6.2f}°    "
          f"Calage stab {p['s_twist']:+6.2f}°")

    # ── Performances ──
    print(SEC("Performances"))
    print(f"    L cruise        {L:7.1f} N   {C.dim(f'/ {WEIGHT:.1f} N poids')}")
    print(f"    D aero          {D:7.2f} N   {C.dim(f'+ D_mât {D_MAST:5.2f} N = {D_total:6.2f} N')}")
    print(f"    {C.bold(f'L/D total       {L/D_total:7.2f}')}   {C.dim(f'(wing-only {L/D:.2f})')}")
    cl_to_info = f"≤ {cl_to_target:.3f} cible / {CL_MAX_TO} stall"
    print(f"    L décollage     {L_to:7.1f} N   D_to {D_to:5.2f} N   "
          f"CL_to {CL_to:.3f}  {C.dim(cl_to_info)}")
    print(f"    Re emplanture   {Re_root:.2e}   {C.dim(f'Re saumon {Re_tip:.2e}')}")

    # ── Pilotabilité & stabilité ──
    print(SEC("Pilotabilité & stabilité"))
    omega_str  = f"{omega_n:5.2f} Hz" if np.isfinite(omega_n) else "UNSTABLE"
    omega_ok   = np.isfinite(omega_n) and (f_lo <= omega_n <= f_hi)
    omega_mark = C.ok("✓") if omega_ok else C.warn("⚠")
    omega_dim  = f"cible freeride {f_lo:.1f}-{f_hi:.1f} Hz"
    print(f"    ω_n           {omega_str}   {C.dim(omega_dim)}   {omega_mark}")
    np_gap = pd["SM_abs"] * 1000
    print(f"    Cm_α          {pd['Cm_alpha']:+6.2f} rad⁻¹    SM/l_t  {pd['SM_lt']*100:5.1f} %    "
          f"{C.dim(f'(gap NP-CG {np_gap:.1f} mm)')}")
    m_mark = C.ok("✓") if abs(M_total) <= M_tol else C.warn("⚠")
    print(f"    Moment résid. {M_total:+6.2f} N·m   {C.dim(f'tol ±{M_tol:.2f}')}   {m_mark}")
    print(f"    Force stab    {F_stab:+6.1f} N")
    # Autorité de contrôle stab — pas redondant avec ω_n (cf. README).
    dFda_stab = CL_ALPHA_STAB_PER_DEG * q_cruise * stab.area()
    dFda_target = cfg.get("stab_control_authority_min_N_per_deg", 0.0)
    if dFda_target > 0:
        dFda_ok = dFda_stab >= dFda_target
        dFda_mark = C.ok("✓") if dFda_ok else C.warn("⚠")
        print(f"    dF_stab/dα    {dFda_stab:+6.1f} N/°   "
              f"{C.dim(f'≥ {dFda_target:.0f} (autorité contrôle)')}   {dFda_mark}")

    # ── Structure & cavitation ──
    print(SEC("Structure & cavitation"))
    s_mark = C.ok("✓") if sigma_vm <= SIGMA_ADMISSIBLE else C.warn("⚠")
    print(f"    σ_VM pic      {sigma_vm/1e6:6.1f} MPa   "
          f"{C.dim(f'≤ {SIGMA_ADMISSIBLE/1e6:.0f} fatigue')}   {s_mark}")
    print(f"    σ_VM statique {sigma_vm_static/1e6:6.1f} MPa   "
          f"{C.dim(f'≤ {SIGMA_ULTIMATE/1e6:.0f} rupture (σ_ult)')}")
    cav_mark = C.ok("✓") if Cp_min >= -SIGMA_CAV else C.warn("⚠")
    print(f"    Cp_min        {Cp_min:+6.2f}        "
          f"{C.dim(f'≥ {-SIGMA_CAV:.2f} σ_v cavitation')}   {cav_mark}")

    # ── Contraintes ──
    sl_lo, sl_hi = cfg["stab_load_range"]
    stab_ok = np.isfinite(F_stab) and (sl_lo <= F_stab <= sl_hi)
    dFda_target_chk = cfg.get("stab_control_authority_min_N_per_deg", 0.0)
    dFda_stab_chk = CL_ALPHA_STAB_PER_DEG * q_cruise * stab.area()
    dFda_ok_chk = (dFda_target_chk == 0) or (dFda_stab_chk >= dFda_target_chk)
    checks = [
        ("L cruise ≥ poids",                 L     >= 0.99 * WEIGHT, ""),
        ("L décollage ≥ poids",              L_to  >= 0.99 * WEIGHT, f"{L_to:.1f} / {WEIGHT:.1f} N"),
        ("CL_to ≤ CL_max (stall)",           CL_to <= CL_MAX_TO + 1e-3, f"{CL_to:.3f} / {CL_MAX_TO}"),
        (f"CL_to ≤ {cfg['takeoff_cl_margin']:.0%} CL_max ({CASE})",
                                              CL_to <= cl_to_target + 1e-3, f"{CL_to:.3f} / {cl_to_target:.3f}"),
        ("|M_total| ≤ trim authority pilote",abs(M_total) <= M_tol, f"{M_total:+.2f} vs ±{M_tol:.2f} N·m"),
        (f"Stab déportant dans [{sl_lo:.0f},{sl_hi:.0f}] N",
                                              stab_ok, f"{F_stab:+.1f} N"),
        (f"Autorité stab dF/dα ≥ {dFda_target_chk:.0f} N/°",
                                              dFda_ok_chk, f"{dFda_stab_chk:.1f} N/°"),
        (f"AR aile ≤ {ar_wing_cap:.1f}" if np.isfinite(ar_wing_cap) else "AR aile libre",
                                              (AR_w <= ar_wing_cap + 0.05), f"{AR_w:.2f}"),
        ("ω_n dans cible freeride",          omega_ok, f"{omega_n:.2f} vs [{f_lo:.1f}-{f_hi:.1f}] Hz"),
        ("Cavitation OK",                    Cp_min >= -SIGMA_CAV, ""),
        ("σ_VM pic ≤ σ_fatigue",             sigma_vm <= SIGMA_ADMISSIBLE, f"{sigma_vm/1e6:.0f} / {SIGMA_ADMISSIBLE/1e6:.0f} MPa"),
        ("α_cruise ≥ -1°",                   p["alpha_cruise"] >= -1.0, f"{p['alpha_cruise']:+.2f}°"),
    ]
    n_ok = sum(1 for _, ok, _ in checks if ok)
    print(SEC(f"Contraintes  ({n_ok}/{len(checks)})"))
    for name, ok, detail in checks:
        mark = C.ok("✓") if ok else C.warn("⚠")
        suffix = f"   {C.dim(detail)}" if (not ok and detail) else ""
        print(f"    {mark}  {name}{suffix}")

    # ── Export ───────────────────────────────────────────────────────────────
    # Naming : outputs/{case}_{level}_{date}_{NN}/  (compteur quotidien)
    out_dir = next_output_dir()
    run_tag = os.path.basename(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    _export_md(out_dir, p, wing, stab, mean_chord, D_total, L, D,
               F_stab, v_h, M_total, L_to, CL_to, rho, mu, X_cg, AR_w, AR_s,
               sigma_vm, sigma_vm_static, pd, omega_n, f_lo, f_hi)

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

    # IMPORTANT → on COPIE l'airfoil par section avant de renommer.
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

    # ── Bloc Exports ─────────────────────────────────────────────────────────
    print(SEC("Exports"))
    print(f"    {C.ok('✓')} Fiche      {C.dim(out_dir + '/fiche_technique.md')}")
    print(f"    {C.ok('✓')} Profils    {C.dim(airfoils_dir + '/')}")

    # XML XFLR5 — les noms d'airfoils des xsecs viennent d'être renommés
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
        print(f"    {C.ok('✓')} XML XFLR5  {C.dim(xml_path)}")
    except Exception as e:
        print(f"    {C.warn('~')} XML XFLR5 non exporté : {e}")

    # Sauvegarde du X optimal pour refine_3d.py (load auto)
    np.save(os.path.join(out_dir, "x_best.npy"), x)
    print(f"    {C.ok('✓')} x_best     {C.dim(out_dir + '/x_best.npy')}")

    # ── Rapport des bornes saturées ─────────────────────────────────────────
    VAR_NAMES = ["fuselage_length", "cg_ratio", "wing_setting_angle", "twist",
                 "s_twist", "alpha_to",
                 "wing_span", "wing_root_chord",
                 "stab_span", "stab_root_chord"]
    TOL = 0.02  # 2% de la largeur de bornes
    saturated = []
    for i, (lo, hi) in enumerate(BOUNDS):
        width = hi - lo
        if x[i] - lo < TOL * width:
            saturated.append(f"{VAR_NAMES[i]} ↓ {lo:.3g}")
        elif hi - x[i] < TOL * width:
            saturated.append(f"{VAR_NAMES[i]} ↑ {hi:.3g}")
    if saturated:
        print()
        print(SEC(f"Bornes saturées  ({len(saturated)})"))
        for s in saturated:
            print(f"    {C.warn('⚠')}  {s}")
        print(C.dim("       → relâcher ces bornes pour explorer plus loin"))
    print(C.head(HBAR))
    print()


def _export_md(out_dir, p, wing, stab, mean_chord, D_total, L, D,
               F_stab, v_h, M_total, L_to, CL_to, rho, mu, X_cg, AR_w, AR_s,
               sigma_vm, sigma_vm_static, pd, omega_n, f_lo, f_hi):
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Re_root = rho * cfg["v_cruise"] * p["wing_root_chord"] / mu
    Re_tip  = rho * cfg["v_cruise"] * p["wing_tip_chord"]  / mu
    CL_c    = L / (0.5 * rho * cfg["v_cruise"] ** 2 * wing.area())
    CD_c    = D / (0.5 * rho * cfg["v_cruise"] ** 2 * wing.area())

    lines = [
        f"# Fiche Technique — {CASE.upper()} freeride",
        "", f"*Générée le {now_str}*", "", "---", "",
        f"## 1. Variables optimisées ({N_VAR})", "",
        "| Variable | Valeur | Bornes |", "|:---|:---|:---|",
        f"| Fuselage length | {p['fuselage_length']*100:.1f} cm | [{BOUNDS[0][0]*100:.0f}–{BOUNDS[0][1]*100:.0f}] cm |",
        f"| CG ratio | {p['cg_ratio']*100:.1f}% c̄ | [{BOUNDS[1][0]*100:.0f}–{BOUNDS[1][1]*100:.0f}]% |",
        f"| Calage aile | {p['wing_setting_angle']:.2f}° | [{BOUNDS[2][0]}–{BOUNDS[2][1]}]° |",
        f"| Twist | {p['twist']:.2f}° | [{BOUNDS[3][0]}–{BOUNDS[3][1]}]° |",
        f"| Calage stab | {p['s_twist']:.2f}° | [{BOUNDS[4][0]}–{BOUNDS[4][1]}]° |",
        f"| α décollage | {p['alpha_to']:.2f}° | [{BOUNDS[5][0]}–{BOUNDS[5][1]}]° |",
        f"| Wing span | {p['wing_span']*100:.1f} cm | [{BOUNDS[6][0]*100:.0f}–{BOUNDS[6][1]*100:.0f}] cm |",
        f"| Wing root chord | {p['wing_root_chord']*1000:.0f} mm | [{BOUNDS[7][0]*1000:.0f}–{BOUNDS[7][1]*1000:.0f}] mm |",
        f"| Stab span | {p['stab_span']*100:.1f} cm | [{BOUNDS[8][0]*100:.0f}–{BOUNDS[8][1]*100:.0f}] cm |",
        f"| Stab root chord | {p['stab_root_chord']*1000:.0f} mm | [{BOUNDS[9][0]*1000:.0f}–{BOUNDS[9][1]*1000:.0f}] mm |",
        f"| α croisière (dérivé L=W) | {p['alpha_cruise']:.2f}° | — |",
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
        f"| Profil  | {WING_AIRFOIL_NAME} | {STAB_AIRFOIL_NAME} |",
        f"| Surface (cm²) | {wing.area()*1e4:.0f} | {stab.area()*1e4:.0f} |",
        f"| Allongement | {AR_w:.2f} | {AR_s:.2f} |",
        f"| Corde moyenne (mm) | {mean_chord*1000:.0f} | {(STAB_ROOT_CHORD+STAB_TIP_CHORD)*500:.0f} |",
        f"| Corde R/T (mm) | {p['wing_root_chord']*1000:.0f}/{p['wing_tip_chord']*1000:.0f} | {STAB_ROOT_CHORD*1000:.0f} / {STAB_TIP_CHORD*1000:.0f} |",
        "", "---", "",
        "## 5. Pilotabilité, Stabilité & Structure", "",
        "| Paramètre | Valeur | Cible |", "|:---|:---|:---|",
        f"| **ω_n** (fréquence pitch, Hz) | {omega_n:.2f} Hz | freeride [{f_lo:.1f}–{f_hi:.1f}] Hz |",
        f"| Cm_α (raideur tangage, rad⁻¹) | {pd['Cm_alpha']:.2f} | (<0 = stable) |",
        f"| SM/l_t (scale-invariant) | {pd['SM_lt']*100:.1f}% | typique aviation 10-25% |",
        f"| Gap NP-CG (absolu) | {pd['SM_abs']*1000:.1f} mm | — |",
        f"| SM/c̄ | {pd['SM_chord']*100:.1f}% | — (chord-normalisé) |",
        f"| CG | {p['cg_ratio']*100:.1f}% c̄ ({X_cg*100:.1f} cm) | [{cfg['cg_range'][0]*100:.0f}–{cfg['cg_range'][1]*100:.0f}]% |",
        f"| Moment résiduel | {M_total:.3f} N·m | < {M_TOL_TRIM:.1f} N·m (trim authority pilote) |",
        f"| Force stab | {F_stab:.1f} N | (<0 = stable) |",
        (lambda d, t: f"| Autorité contrôle stab dF/dα | {d:.1f} N/° | ≥ {t:.0f} (cible) |")(
            CL_ALPHA_STAB_PER_DEG * 0.5 * rho * cfg['v_cruise']**2 * stab.area(),
            cfg.get('stab_control_authority_min_N_per_deg', 0.0)),
        f"| Volume de queue (info) | {v_h:.3f} | — |",
        f"| Von Mises root (pic ×{LOAD_PEAK_FACTOR:.1f}g) | {sigma_vm/1e6:.1f} MPa | < {SIGMA_ADMISSIBLE/1e6:.0f} MPa (fatigue) |",
        f"| Von Mises root (statique 1g) | {sigma_vm_static/1e6:.1f} MPa | < {SIGMA_ULTIMATE/1e6:.0f} MPa (rupture) |",
        f"| σ_v cavitation | {SIGMA_CAV:.2f} | — |",
        "",
    ]

    warn = []
    if not np.isfinite(omega_n):
        warn.append(f"⚠️ Foil INSTABLE (Cm_α > 0) — pilote ne peut pas le maintenir au trim")
    elif not (f_lo <= omega_n <= f_hi):
        warn.append(f"⚠️ ω_n {omega_n:.2f} Hz hors cible freeride [{f_lo:.1f}–{f_hi:.1f}] Hz")
    if L_to < WEIGHT * 0.98:
        warn.append(f"⚠️ Portance décollage insuffisante : {L_to:.1f} / {WEIGHT:.1f} N")
    if CL_to > CL_MAX_TO:
        warn.append(f"⚠️ CL_to {CL_to:.2f} > CL_max {CL_MAX_TO}")
    if abs(M_total) > M_TOL_TRIM:
        warn.append(f"⚠️ Moment résiduel {M_total:.2f} N·m > trim authority pilote ±{M_TOL_TRIM:.1f} N·m")
    if sigma_vm > SIGMA_ADMISSIBLE:
        warn.append(f"⚠️ Von Mises pic dyn {sigma_vm/1e6:.0f} MPa > σ_fatigue {SIGMA_ADMISSIBLE/1e6:.0f} MPa "
                    f"— ratio {sigma_vm/SIGMA_ADMISSIBLE:.2f}, foil sous-dimensionné en fatigue")
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