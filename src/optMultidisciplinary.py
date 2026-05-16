# =================================================================================
# Hydrofoil Optimization — Kulfan CST + Differential Evolution
# =================================================================================

import os
import sys
import datetime as dt
import warnings
import numpy as np
import yaml
from pathlib import Path
from scipy.optimize import differential_evolution

import aerosandbox as asb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from config.water_atmosphere import Water as Atmosphere

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

with open(ROOT / "config" / "parameters.yaml") as f:
    phy = yaml.safe_load(f)
with open(ROOT / "config" / "scenarios.yaml") as f:
    SCENARIOS = yaml.safe_load(f)

CASE = phy["case"]
if CASE not in SCENARIOS:
    raise ValueError(f"Cas '{CASE}' introuvable. Options : {list(SCENARIOS.keys())}")
cfg = SCENARIOS[CASE]

atmosphere = Atmosphere()

# Masses
mass      = phy["pilot"]["mass_kg"] + phy["board"]["mass_kg"]
rig_mass  = cfg["rig_mass_kg"]
WEIGHT    = (mass + rig_mass) * 9.81

# Géométrie fixe
sweep_deg          = phy["wing"]["sweep_deg"]
wing_anhedral_deg  = phy["wing"]["anhedral_deg"]
N_WING             = phy["wing"]["n_sections"]
CL_MAX_TO          = phy["wing"]["cl_max_takeoff"]
fuselage_diameter  = phy["fuselage"]["diameter"]
x_fuselage_start   = phy["fuselage"]["x_start"]
N_FUSE             = phy["fuselage"]["n_sections"]
mast_length        = phy["mast"]["length"]
profondeur_imm     = phy["mast"]["immersion_depth"]
mast_chord_top     = phy["mast"]["chord_top"]
mast_chord_bot     = phy["mast"]["chord_bot"]
mast_profile       = phy["mast"]["profile"]
x_mast             = phy["mast"]["x_position"]
N_MAST             = phy["mast"]["n_sections"]
stab_dihedral_deg  = phy["stab"]["dihedral_deg"]
s_sweep_deg        = phy["stab"]["sweep_deg"]
N_STAB             = phy["stab"]["n_sections"]

chord_wl = mast_chord_bot * (1 - profondeur_imm / mast_length) \
         + mast_chord_top * (profondeur_imm / mast_length)

# Traînée & moment du mât (constants pour un cfg donné)
q_cruise    = 0.5 * atmosphere.density() * cfg["v_cruise"] ** 2
D_MAST      = q_cruise * ((mast_chord_bot + chord_wl) / 2) * profondeur_imm * 0.011
M_MAST      = -D_MAST * (profondeur_imm / 2)

# ─────────────────────────────────────────────────────────────────────────────
# 2. PARAMÉTRAGE KULFAN (CST)
# ─────────────────────────────────────────────────────────────────────────────

N_CST = 4          # Coefficients par surface (extrados + intrados)
DELTA = 0.12       # Demi-amplitude des bornes autour des valeurs de référence

# Valeurs de référence centrées sur NACA 2412 (cambrure légère, portance/traînée équilibrée)
# Obtenues par fit CST d'ordre 4 sur le profil NACA 2412.
_REF_UPPER = np.array([0.195, 0.165, 0.180, 0.160])
_REF_LOWER = np.array([0.095, 0.040, 0.028, 0.030])

# Bornes Kulfan : [lower_bound, upper_bound] par coefficient
# L'extrados reste positif (surface portante), l'intrados peut être légèrement négatif
BOUNDS_Au = [(_REF_UPPER[i] - DELTA,      _REF_UPPER[i] + DELTA)      for i in range(N_CST)]
BOUNDS_Al = [(_REF_LOWER[i] - DELTA + 0.02, _REF_LOWER[i] + DELTA)    for i in range(N_CST)]
# Root et Tip partagent les mêmes bornes (le morphing interpole entre les deux)
BOUNDS_AIRFOIL = BOUNDS_Au + BOUNDS_Al  # 8 bornes par profil

# Bornes géométriques (depuis physics.yaml)
BOUNDS_GEOM = [
    tuple(phy["wing"]["span_bounds"]),
    tuple(phy["wing"]["root_chord_bounds"]),
    tuple(phy["wing"]["tip_chord_bounds"]),
    tuple(phy["wing"]["twist_bounds"]),
    tuple(phy["stab"]["span_bounds"]),
    tuple(phy["stab"]["root_chord_bounds"]),
    tuple(phy["stab"]["tip_chord_bounds"]),
    tuple(phy["stab"]["twist_bounds"]),
    tuple(phy["fuselage"]["length_bounds"]),
    tuple(cfg["cg_range"]),
    tuple(phy["alpha"]["bounds"]),
]

# Vecteur complet : [Au_root(4), Al_root(4), Au_tip(4), Al_tip(4), géom(11)]
# Index :            0:4         4:8          8:12        12:16       16:27
BOUNDS = BOUNDS_AIRFOIL + BOUNDS_AIRFOIL + BOUNDS_GEOM
N_VAR  = len(BOUNDS)  # 27

# ─────────────────────────────────────────────────────────────────────────────
# 3. FILTRE GÉOMÉTRIQUE RAPIDE
# ─────────────────────────────────────────────────────────────────────────────

TE_MIN_M         = 0.001   # Épaisseur bord de fuite minimale (1 mm)
THICKNESS_MIN    = 0.06    # Épaisseur relative minimale (6 %)
THICKNESS_SAMPLE = np.linspace(0.05, 0.95, 15)


def geometric_penalty(af: asb.KulfanAirfoil, chord: float) -> float:
    """
    Retourne 0.0 si le profil est valide, sinon une pénalité proportionnelle
    à la sévérité du défaut. Appelé avant tout calcul fluide.
    """
    try:
        coords = af.coordinates
        if np.any(~np.isfinite(coords)):
            return 1e6

        # Épaisseur max
        t_max = af.max_thickness()
        if t_max < THICKNESS_MIN:
            return 1e6 + 1e4 * (THICKNESS_MIN - t_max)

        # Épaisseur locale — détecte les croisements extrados/intrados
        # On sépare les coordonnées manuellement pour robustesse
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        idx_le   = int(np.argmin(x_coords))
        upper_y  = np.interp(THICKNESS_SAMPLE, x_coords[idx_le::-1], y_coords[idx_le::-1])
        lower_y  = np.interp(THICKNESS_SAMPLE, x_coords[idx_le:],    y_coords[idx_le:])
        thickness = upper_y - lower_y
        if np.any(thickness < 0):
            return 1e6 + 1e4 * float(-np.min(thickness))

        # Bord de fuite
        te_thick = float(af.TE_thickness) * chord
        if te_thick < TE_MIN_M:
            return 5e4 * (TE_MIN_M - te_thick) / TE_MIN_M

    except Exception:
        return 1e6

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANALYSE STRUCTURELLE — FLEXION À L'EMPLANTURE
# ─────────────────────────────────────────────────────────────────────────────

SIGMA_CARBONE = 400e6   # Limite admissible carbone/époxy (Pa) — valeur conservative


def section_inertia(af: asb.KulfanAirfoil, chord: float) -> tuple:
    """
    Calcule le moment quadratique I_xx et y_max d'une section de profil
    par la formule de Green sur le polygone de coordonnées.
    Hypothèse section pleine (conservative pour une coque).
    """
    coords = af.coordinates
    xp = coords[:, 0] * chord
    yp = coords[:, 1] * chord
    n  = len(xp)

    I_xx = 0.0
    for i in range(n - 1):
        cross = xp[i] * yp[i + 1] - xp[i + 1] * yp[i]
        I_xx += (yp[i] ** 2 + yp[i] * yp[i + 1] + yp[i + 1] ** 2) * cross
    I_xx  = abs(I_xx) / 12.0
    y_max = float(np.max(np.abs(yp)))
    return I_xx, y_max


def bending_stress(I_xx: float, y_max: float, lift_semi: float, span_semi: float) -> float:
    """
    Contrainte de flexion à l'emplanture.
    Charge appliquée au centre de portance d'une distribution elliptique (span/4).
    """
    M_flex = lift_semi * (span_semi / 4.0)
    return M_flex * y_max / (I_xx + 1e-15)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONTRAINTE DE CAVITATION
# ─────────────────────────────────────────────────────────────────────────────

P_ATM   = 101_325.0    # Pa
P_VAPOR = 2_338.0      # Pa à 20 °C
RHO     = 1_000.0      # kg/m³


def cavitation_number(v: float, depth: float) -> float:
    """σ_v = (P_atm + ρgh - P_vapor) / (½ρV²). Cavitation si Cp_min < -σ_v."""
    return (P_ATM + RHO * 9.81 * depth - P_VAPOR) / (0.5 * RHO * v ** 2)


# À 9.5 m/s : σ_v ≈ 2.3 | À 13 m/s : σ_v ≈ 1.2
# Les Cp_min typiques à ces Reynolds (~1–2×10⁶) restent autour de −0.5 à −0.8 :
# la cavitation n'est pas critique en freeride. Contrainte incluse mais pondérée légèrement.
SIGMA_CAV = cavitation_number(cfg["v_cruise"], profondeur_imm)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONSTRUCTION DE L'AVION (forward pass, sans CasADi)
# ─────────────────────────────────────────────────────────────────────────────

def decode(x: np.ndarray) -> dict:
    """Découpe le vecteur DE en sous-ensembles nommés."""
    return {
        "Au_root":        x[0:N_CST],
        "Al_root":        x[N_CST:2*N_CST],
        "Au_tip":         x[2*N_CST:3*N_CST],
        "Al_tip":         x[3*N_CST:4*N_CST],
        "span":           float(x[16]),
        "root_chord":     float(x[17]),
        "tip_chord":      float(x[18]),
        "twist":          float(x[19]),
        "s_span":         float(x[20]),
        "s_root_chord":   float(x[21]),
        "s_tip_chord":    float(x[22]),
        "s_twist":        float(x[23]),
        "fuselage_length":float(x[24]),
        "cg_ratio":       float(x[25]),
        "alpha":          float(x[26]),
    }


def interpolate_kulfan(af1: asb.KulfanAirfoil, af2: asb.KulfanAirfoil, r: float) -> asb.KulfanAirfoil:
    """Interpolation linéaire des coefficients CST entre root (r=0) et tip (r=1)."""
    return asb.KulfanAirfoil(
        upper_weights       = (1 - r) * af1.upper_weights       + r * af2.upper_weights,
        lower_weights       = (1 - r) * af1.lower_weights       + r * af2.lower_weights,
        leading_edge_weight = (1 - r) * af1.leading_edge_weight + r * af2.leading_edge_weight,
        TE_thickness        = (1 - r) * af1.TE_thickness        + r * af2.TE_thickness,
    )


def build_airplane(p: dict) -> tuple:
    """
    Construit l'avion AeroSandBox depuis le dictionnaire de paramètres.
    Retourne (airplane, wing, stab, mean_chord, af_root) ou lève une exception.
    """
    af_root = asb.KulfanAirfoil(upper_weights=p["Au_root"], lower_weights=p["Al_root"])
    af_tip  = asb.KulfanAirfoil(upper_weights=p["Au_tip"],  lower_weights=p["Al_tip"])

    span        = p["span"]
    root_chord  = p["root_chord"]
    tip_chord   = p["tip_chord"]
    twist       = p["twist"]
    cg_ratio    = p["cg_ratio"]

    # ── Aile principale (morphing CST) ──────────────────────────────────────
    wing_xsecs = []
    for i in range(N_WING):
        r        = i / (N_WING - 1)
        af_blend = interpolate_kulfan(af_root, af_tip, r)

        elliptic = np.sqrt(max(1 - r ** 2, 0))
        c_dist   = tip_chord + (root_chord - tip_chord) * (0.6 * elliptic + 0.4 * (1 - r))

        # Fermeture progressive au saumon (évite une fin de corde abrupte)
        r0 = 0.88
        if r > r0:
            rt      = (r - r0) / (1 - r0)
            closure = np.sqrt(max(1 - rt ** 2, 0))
            c_dist  = (c_dist - 0.012) * closure + 0.012

        z_pos   = -((r ** 2) * span / 2) * np.tan(np.radians(wing_anhedral_deg)) \
                  - 0.025 * (r ** 5)
        x_le    = ((r ** 2.5) * span / 2) * np.tan(np.radians(sweep_deg))

        wing_xsecs.append(asb.WingXSec(
            xyz_le=[x_le, r * span / 2, z_pos],
            chord=c_dist,
            twist=twist * r,
            airfoil=af_blend,
        ))

    wing       = asb.Wing(symmetric=True, name="MainWing", xsecs=wing_xsecs)
    mean_chord = wing.mean_geometric_chord()
    X_cg       = cg_ratio * mean_chord

    # ── Fuselage ─────────────────────────────────────────────────────────────
    fuse_xsecs = []
    for i in range(N_FUSE):
        xi_rel = i / (N_FUSE - 1)
        xi     = x_fuselage_start + xi_rel * p["fuselage_length"]
        width  = (0.001 if i == 0
                  else fuselage_diameter * (i / 3) if i < 3
                  else fuselage_diameter * (1 - 0.5 * xi_rel ** 3))
        fuse_xsecs.append(asb.FuselageXSec(xyz_c=[xi, 0, 0], radius=width))
    fuselage_obj = asb.Fuselage(name="Fuselage", xsecs=fuse_xsecs)

    # ── Mât ──────────────────────────────────────────────────────────────────
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

    # ── Stabilisateur (NACA 0012 symétrique — pas de CST nécessaire) ─────────
    s_span       = p["s_span"]
    s_root_chord = p["s_root_chord"]
    s_tip_chord  = p["s_tip_chord"]
    s_twist      = p["s_twist"]
    x_stab_root  = x_fuselage_start + p["fuselage_length"] - 0.10

    stab_xsecs = []
    for i in range(N_STAB):
        r      = i / (N_STAB - 1)
        # Distribution de corde propre au stab (bug corrigé vs version NACA)
        c_s    = s_tip_chord + (s_root_chord - s_tip_chord) * (0.8 * (1 - r) + 0.2 * np.sqrt(max(1 - r**2, 0)))
        r0s    = 0.80
        if r > r0s:
            rt  = (r - r0s) / (1 - r0s)
            c_s = (c_s - 0.01) * np.sqrt(max(1 - rt ** 2, 0)) + 0.01

        z_s    = (r ** 1.5 * s_span / 2) * np.tan(np.radians(stab_dihedral_deg))
        x_s    = 0.9 * (s_root_chord - c_s) + (r ** 2.5 * s_span / 2) * np.tan(np.radians(s_sweep_deg))

        stab_xsecs.append(asb.WingXSec(
            xyz_le=[x_stab_root + x_s, r * s_span / 2, z_s],
            chord=c_s, twist=s_twist,
            airfoil=asb.Airfoil("naca0012"),
        ))
    stab = asb.Wing(symmetric=True, name="Stab", xsecs=stab_xsecs)

    # ── Avion complet ─────────────────────────────────────────────────────────
    airplane = asb.Airplane(
        wings=[wing, stab],
        fuselages=[fuselage_obj],
        xyz_ref=np.array([X_cg, 0.0, 0.0]),
        s_ref=wing.area(), c_ref=mean_chord, b_ref=wing.span(),
    )

    return airplane, wing, stab, mean_chord, af_root, mast_obj, fuselage_obj


# ─────────────────────────────────────────────────────────────────────────────
# 7. FONCTION OBJECTIF — TRAÎNÉE + PÉNALITÉS
# ─────────────────────────────────────────────────────────────────────────────

K = 500.0    # Coefficient de pénalité (empiriquement calibré)


def objective(x: np.ndarray) -> float:
    """
    Minimise D_total = D_aéro + D_mât.
    Les contraintes sont gérées par pénalité quadratique (sauf géométrie = fail-fast).

    Contraintes :
      Hard (filtre géométrique) : épaisseur, croisement, TE
      Soft (pénalité K×violation²) :
        - L ≥ Weight
        - |M_total| / (W·c̄) ≤ 5%   [moment d'équilibre]
        - Surfaces aile & stab dans la plage cible
        - Force stab dans la plage cible
        - Marge statique dans sm_range
        - σ_flexion ≤ σ_carbone
        - CL décollage ≤ CL_max
        - Volume de queue dans vh_range
        - Cp_min ≥ -σ_cavitation  [faible poids]
    """
    p = decode(x)

    # ── Filtre géométrique (fast-fail) ───────────────────────────────────────
    af_root_tmp = asb.KulfanAirfoil(upper_weights=p["Au_root"], lower_weights=p["Al_root"])
    af_tip_tmp  = asb.KulfanAirfoil(upper_weights=p["Au_tip"],  lower_weights=p["Al_tip"])

    geo_pen = geometric_penalty(af_root_tmp, p["root_chord"]) \
            + geometric_penalty(af_tip_tmp,  p["tip_chord"])
    if geo_pen > 0:
        return 1e6 + geo_pen

    # ── Construction de l'avion ───────────────────────────────────────────────
    try:
        airplane, wing, stab, mean_chord, af_root, _, _ = build_airplane(p)
    except Exception:
        return 1e6

    # ── Évaluation aérodynamique ─────────────────────────────────────────────
    try:
        op   = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=p["alpha"], atmosphere=atmosphere)
        aero = asb.AeroBuildup(airplane, op).run()
        L    = float(aero["L"])
        D    = float(aero["D"])
        Cm   = float(aero["Cm"])
        CL   = float(aero["CL"])
    except Exception:
        return 1e6

    D_total = D + D_MAST

    # ── Pénalités ─────────────────────────────────────────────────────────────
    penalty = 0.0

    def soft(violation: float) -> float:
        """Pénalité quadratique — nulle si violation ≤ 0."""
        return K * max(violation, 0.0) ** 2

    # 1. Portance = Poids
    penalty += soft((WEIGHT - L) / WEIGHT)

    # 2. Équilibre de tangage (soft, ±5 % c̄)
    X_cg      = p["cg_ratio"] * mean_chord
    M_wing    = Cm * q_cruise * wing.area() * mean_chord
    M_greem   = rig_mass * 9.81 * (-(X_cg - x_mast))
    M_total   = M_wing + M_MAST + M_greem
    M_norm    = abs(M_total) / (WEIGHT * mean_chord)
    if M_norm > 0.05:
        penalty += K * (M_norm / 0.05) ** 2

    # 3. Surface aile
    S_wing = wing.area()
    penalty += soft(cfg["area_target_range"][0] - S_wing)
    penalty += soft(S_wing - cfg["area_target_range"][1])

    # 4. Surface stab
    S_stab = stab.area()
    penalty += soft(cfg["stab_area_range"][0] - S_stab)
    penalty += soft(S_stab - cfg["stab_area_range"][1])

    # 5. Force stab
    try:
        F_stab = float(aero["wing_aero_components"][1].L)
        penalty += soft(F_stab - cfg["stab_load_range"][1])
        penalty += soft(cfg["stab_load_range"][0] - F_stab)
    except Exception:
        penalty += K

    # 6. Marge statique — perturbation numérique +0.1°
    try:
        op2   = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=p["alpha"] + 0.1, atmosphere=atmosphere)
        aero2 = asb.AeroBuildup(airplane, op2).run()
        dCL   = float(aero2["CL"]) - CL
        dCm   = float(aero2["Cm"]) - float(aero["Cm"])
        SM    = -(dCm / (dCL + 1e-9))
        penalty += soft(cfg["sm_range"][0] - SM)
        penalty += soft(SM - cfg["sm_range"][1])
    except Exception:
        penalty += K

    # 7. Contrainte structurelle — flexion à l'emplanture
    try:
        I_xx, y_max = section_inertia(af_root, p["root_chord"])
        sigma       = bending_stress(I_xx, y_max, WEIGHT / 2, p["span"] / 2)
        penalty    += soft((sigma - SIGMA_CARBONE) / SIGMA_CARBONE)
    except Exception:
        penalty += K * 0.5

    # 8. CL décollage
    q_to    = 0.5 * atmosphere.density() * cfg["v_takeoff"] ** 2
    CL_to   = WEIGHT / (q_to * S_wing + 1e-9)
    penalty += soft(CL_to - CL_MAX_TO)

    # 9. Volume de queue
    v_h = (S_stab * p["fuselage_length"]) / (S_wing * mean_chord + 1e-9)
    penalty += soft(cfg["vh_range"][0] - v_h)
    penalty += soft(v_h - cfg["vh_range"][1])

    # 10. Cavitation (poids faible — non critique en freeride)
    # Cp_min ≈ -2·CL (estimation conservative profil mince à Re > 10^6)
    Cp_min_est = -2.0 * abs(CL)
    if Cp_min_est < -SIGMA_CAV:
        penalty += K * 0.05 * ((-Cp_min_est - SIGMA_CAV) / SIGMA_CAV) ** 2

    return D_total + penalty


# ─────────────────────────────────────────────────────────────────────────────
# 8. OPTIMISATION — DIFFERENTIAL EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

DE_PARAMS = {
    "strategy":   "best1bin",    # Bonne convergence sur problèmes continus
    "maxiter":    300,
    "popsize":    12,            # 12 × 27 vars = 324 individus par génération
    "tol":        1e-5,
    "mutation":   (0.5, 1.0),   # Plage de mutation adaptative
    "recombination": 0.85,
    "seed":       42,
    "workers":    -1,            # Parallèle sur tous les cœurs disponibles
    "polish":     True,          # Affinage L-BFGS-B sur le meilleur individu
    "updating":   "deferred",    # Nécessaire pour workers=-1
    "disp":       True,
}


def run_optimization() -> np.ndarray:
    print(f"\n{'='*65}")
    print(f"  DIFFERENTIAL EVOLUTION — CAS : {CASE.upper()}")
    print(f"{'='*65}")
    print(f"  Variables       : {N_VAR} ({N_CST*4} CST + 11 géom.)")
    print(f"  Population      : {DE_PARAMS['popsize'] * N_VAR} individus")
    print(f"  Générations max : {DE_PARAMS['maxiter']}")
    print(f"  Profilage       : Kulfan CST d'ordre {N_CST}")
    print(f"  σ_cav cible     : {SIGMA_CAV:.2f} (Cp_min critique)\n")

    result = differential_evolution(objective, BOUNDS, **DE_PARAMS)

    print(f"\n  Convergence : {'✓' if result.success else '✗ (partielle)'}")
    print(f"  D_total     : {result.fun:.3f} N")
    print(f"  Évaluations : {result.nfev}")
    return result.x


# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPORT — FICHE TECHNIQUE & FICHIERS CAO
# ─────────────────────────────────────────────────────────────────────────────

def full_report(x: np.ndarray) -> None:
    """Évalue le meilleur individu, affiche le bilan et exporte les fichiers."""
    p = decode(x)
    try:
        airplane, wing, stab, mean_chord, af_root, mast_obj, fuselage_obj = build_airplane(p)
    except Exception as e:
        print(f"Erreur construction avion : {e}")
        return

    rho = atmosphere.density()
    mu  = atmosphere.dynamic_viscosity()

    op    = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=p["alpha"], atmosphere=atmosphere)
    aero  = asb.AeroBuildup(airplane, op).run()
    L, D  = float(aero["L"]), float(aero["D"])
    Cm, CL = float(aero["Cm"]), float(aero["CL"])

    D_total   = D + D_MAST
    X_cg      = p["cg_ratio"] * mean_chord
    M_wing    = Cm * q_cruise * wing.area() * mean_chord
    M_greem   = rig_mass * 9.81 * (-(X_cg - x_mast))
    M_total   = M_wing + M_MAST + M_greem

    op2   = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=p["alpha"] + 0.1, atmosphere=atmosphere)
    aero2 = asb.AeroBuildup(airplane, op2).run()
    SM    = -((float(aero2["Cm"]) - Cm) / (float(aero2["CL"]) - CL + 1e-9))

    F_stab  = float(aero["wing_aero_components"][1].L)
    v_h     = (stab.area() * p["fuselage_length"]) / (wing.area() * mean_chord)
    I_xx, y_max = section_inertia(af_root, p["root_chord"])
    sigma   = bending_stress(I_xx, y_max, WEIGHT / 2, p["span"] / 2)

    # ── Affichage console ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RÉSULTAT FINAL — {CASE.upper()}")
    print(f"{'='*65}")
    print(f"  Traînée totale   : {D_total:.2f} N      Finesse : {L/D_total:.2f}")
    print(f"  Incidence trim   : {p['alpha']:.2f}°")
    print(f"  Surface aile     : {wing.area()*1e4:.0f} cm²    AR : {wing.aspect_ratio():.2f}")
    print(f"  Envergure        : {p['span']*100:.1f} cm")
    print(f"  Corde R/T        : {p['root_chord']*1000:.0f} / {p['tip_chord']*1000:.0f} mm")
    print(f"  Surface stab     : {stab.area()*1e4:.0f} cm²")
    print(f"  Fuselage         : {p['fuselage_length']*100:.0f} cm")
    print(f"  Marge statique   : {SM*100:.1f} %")
    print(f"  Moment résiduel  : {M_total:.4f} N·m")
    print(f"  Force stab       : {F_stab:.1f} N")
    print(f"  Volume de queue  : {v_h:.3f}")
    print(f"  CG               : {p['cg_ratio']*100:.1f} % c̄")
    print(f"  σ flexion root   : {sigma/1e6:.1f} MPa / {SIGMA_CARBONE/1e6:.0f} MPa admis.")
    print(f"  Cavitation σ_v   : {SIGMA_CAV:.2f}  |  Cp_min ≈ {-2*abs(CL):.2f}")
    print(f"{'='*65}\n")

    # ── Export fichiers ───────────────────────────────────────────────────────
    now_str      = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir      = os.path.join("outputs", f"{CASE}_kulfan_{now_str}")
    airfoils_dir = os.path.join(out_dir, "airfoils")
    os.makedirs(airfoils_dir, exist_ok=True)

    # Fiche technique Markdown
    _export_md(out_dir, p, wing, stab, mean_chord, D_total, L, D, CL, Cm,
               SM, F_stab, v_h, M_total, sigma, rho, mu, X_cg)

    # Profils .dat (format Selig pour XFLR5)
    airplane_sol = airplane
    for i, xsec in enumerate(airplane_sol.wings[0].xsecs):
        _export_dat(xsec.airfoil, os.path.join(airfoils_dir, f"wing_sec_{i}.dat"), f"wing_sec_{i}")
    for i, xsec in enumerate(airplane_sol.wings[1].xsecs):
        _export_dat(xsec.airfoil, os.path.join(airfoils_dir, f"stab_sec_{i}.dat"), f"stab_sec_{i}")

    # XML XFLR5
    xml_path = os.path.join(out_dir, f"{CASE}_kulfan_{now_str}_plane.xml")
    asb.Airplane(
        wings=[
            asb.Wing(symmetric=True, name="mainwing",  xsecs=airplane_sol.wings[0].xsecs),
            asb.Wing(symmetric=True, name="elevator",  xsecs=airplane_sol.wings[1].xsecs),
            mast_obj,
        ],
        fuselages=[fuselage_obj],
        xyz_ref=airplane_sol.xyz_ref,
    ).export_XFLR5_xml(xml_path)

    print(f"  ✓ Fiche technique : {out_dir}/fiche_technique.md")
    print(f"  ✓ Profils .dat    : {airfoils_dir}/")
    print(f"  ✓ XML XFLR5       : {xml_path}")


def _export_dat(af, filepath, name):
    af  = af.repanel(n_points_per_side=50)
    c   = af.coordinates
    x_min, x_max = np.min(c[:, 0]), np.max(c[:, 0])
    c[:, 0] = (c[:, 0] - x_min) / (x_max - x_min)
    c[:, 1] =  c[:, 1] / (x_max - x_min)
    idx  = np.argmin(c[:, 0])
    up   = c[:idx + 1][::-1] if c[0, 0] > c[-1, 0] else c[:idx + 1]
    lo   = c[idx:]
    if lo[0, 0] > lo[-1, 0]:
        lo = lo[::-1]
    coords = np.concatenate([up, lo[1:]])
    with open(filepath, "w") as f:
        f.write(f"{name}\n")
        for xi, yi in coords:
            f.write(f" {xi:.6f} {yi:.6f}\n")


def _export_md(out_dir, p, wing, stab, mean_chord, D_total, L, D, CL, Cm,
               SM, F_stab, v_h, M_total, sigma, rho, mu, X_cg):
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Re_root = rho * cfg["v_cruise"] * p["root_chord"] / mu
    Re_tip  = rho * cfg["v_cruise"] * p["tip_chord"]  / mu
    CL_c    = L / (0.5 * rho * cfg["v_cruise"] ** 2 * wing.area())
    CD_c    = D / (0.5 * rho * cfg["v_cruise"] ** 2 * wing.area())

    lines = [
        f"# Fiche Technique — {CASE.upper()} | Kulfan CST",
        f"", f"*Générée le {now_str}*", f"", f"---", f"",
        f"## 0. Paramètres du run", f"",
        f"| Variable | Min | Max |", f"|:---|:---|:---|",
        f"| Span (m) | {phy['wing']['span_bounds'][0]} | {phy['wing']['span_bounds'][1]} |",
        f"| Root chord (m) | {phy['wing']['root_chord_bounds'][0]} | {phy['wing']['root_chord_bounds'][1]} |",
        f"| Fuselage (m) | {phy['fuselage']['length_bounds'][0]} | {phy['fuselage']['length_bounds'][1]} |",
        f"| CG ratio | {cfg['cg_range'][0]} | {cfg['cg_range'][1]} |",
        f"", f"| Contrainte scénario | Min | Max |", f"|:---|:---|:---|",
        f"| Surface aile (m²) | {cfg['area_target_range'][0]} | {cfg['area_target_range'][1]} |",
        f"| SM | {cfg['sm_range'][0]} | {cfg['sm_range'][1]} |",
        f"| Stab load (N) | {cfg['stab_load_range'][0]} | {cfg['stab_load_range'][1]} |",
        f"| Vh | {cfg['vh_range'][0]} | {cfg['vh_range'][1]} |",
        f"", f"---", f"",
        f"## 1. Conditions de vol", f"",
        f"| Paramètre | Valeur |", f"|:---|:---|",
        f"| Poids total | {WEIGHT:.1f} N ({WEIGHT/9.81:.0f} kg) |",
        f"| V décollage | {cfg['v_takeoff']} m/s |",
        f"| V croisière | {cfg['v_cruise']} m/s |",
        f"| Re emplanture | {Re_root:.2e} |",
        f"| Re saumon | {Re_tip:.2e} |",
        f"", f"---", f"",
        f"## 2. Géométrie", f"",
        f"| Paramètre | Valeur |", f"|:---|:---|",
        f"| Surface aile | {wing.area()*1e4:.0f} cm² |",
        f"| Envergure | {p['span']*100:.1f} cm |",
        f"| Allongement | {wing.aspect_ratio():.2f} |",
        f"| Corde emplanture | {p['root_chord']*1000:.0f} mm |",
        f"| Corde saumon | {p['tip_chord']*1000:.0f} mm |",
        f"| Vrillage | {p['twist']:.2f}° |",
        f"| Surface stab | {stab.area()*1e4:.0f} cm² |",
        f"| Envergure stab | {p['s_span']*100:.1f} cm |",
        f"| Fuselage | {p['fuselage_length']*100:.0f} cm |",
        f"", f"---", f"",
        f"## 3. Performances", f"",
        f"| Paramètre | Valeur |", f"|:---|:---|",
        f"| Finesse (L/D) | {L/D_total:.2f} |",
        f"| Traînée totale | {D_total:.2f} N |",
        f"| Incidence | {p['alpha']:.2f}° |",
        f"| CL croisière | {CL_c:.3f} |",
        f"| CD croisière | {CD_c:.4f} |",
        f"", f"---", f"",
        f"## 4. Stabilité & Structure", f"",
        f"| Paramètre | Valeur |", f"|:---|:---|",
        f"| Marge statique | {SM*100:.1f} % |",
        f"| Position CG | {p['cg_ratio']*100:.1f} % c̄ ({X_cg*100:.1f} cm du BA) |",
        f"| Moment résiduel | {M_total:.4f} N·m |",
        f"| Force stab | {F_stab:.1f} N |",
        f"| Volume de queue | {v_h:.3f} |",
        f"| σ flexion root | {sigma/1e6:.1f} MPa / {SIGMA_CARBONE/1e6:.0f} MPa |",
        f"| σ_v cavitation | {SIGMA_CAV:.2f} |",
        f"",
    ]

    warnings_list = []
    if SM * 100 > 70:
        warnings_list.append(f"⚠️ SM élevée ({SM*100:.1f}%) — maniabilité réduite.")
    if SM * 100 < 5:
        warnings_list.append(f"⚠️ SM très faible ({SM*100:.1f}%) — risque d'instabilité.")
    if abs(M_total) > 5:
        warnings_list.append(f"⚠️ Moment résiduel important ({M_total:.2f} N·m).")
    if sigma > SIGMA_CARBONE:
        warnings_list.append(f"⚠️ Contrainte flexion ({sigma/1e6:.0f} MPa) dépasse la limite admissible.")
    if warnings_list:
        lines += [f"## ⚠️ Avertissements", f""] + [f"- {w}" for w in warnings_list] + [f""]

    with open(os.path.join(out_dir, "fiche_technique.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 10. POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    x_best = run_optimization()
    full_report(x_best)