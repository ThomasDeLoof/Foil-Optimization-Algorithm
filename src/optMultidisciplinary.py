# =================================================================================
# ====== Hydrofoil Optimization — Kulfan CST - Solver Differential Evolution ======
# =================================================================================

import os
import sys
import datetime as dt
import warnings
import numpy as np
import yaml
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.special import comb

import aerosandbox as asb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from config.water_atmosphere import Water as Atmosphere

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

with open("/Users/thomas/Documents/Dossier Supaero/clubs/Foil/Foil-Optimization-Algorithm/src/config/parameters.yaml") as f:
    phy = yaml.safe_load(f)
with open("/Users/thomas/Documents/Dossier Supaero/clubs/Foil/Foil-Optimization-Algorithm/src/config/scenarios.yaml") as f:
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
cd_mast            = phy["mast"]["cd"]
N_MAST             = phy["mast"]["n_sections"]
stab_dihedral_deg  = phy["stab"]["dihedral_deg"]
s_sweep_deg        = phy["stab"]["sweep_deg"]
N_STAB             = phy["stab"]["n_sections"]
N_CST              = phy["Kulfan"]["N_CST"]          
DELTA              = phy["Kulfan"]["DELTA"]   
NACA_REF_NAME      = phy["Kulfan"]["NACA_REF_NAME"] 

chord_wl = mast_chord_bot * (1 - profondeur_imm / mast_length) \
         + mast_chord_top * (profondeur_imm / mast_length)

# Traînée & moment du mât (constants pour un cfg donné)
q_cruise    = 0.5 * atmosphere.density() * cfg["v_cruise"] ** 2
D_MAST      = q_cruise * ((mast_chord_bot + chord_wl) / 2) * profondeur_imm * cd_mast
M_MAST      = -D_MAST * (profondeur_imm / 2)

# ─────────────────────────────────────────────────────────────────────────────
# 2. PARAMÉTRAGE KULFAN (CST)
# ─────────────────────────────────────────────────────────────────────────────

ref_airfoil = asb.Airfoil(NACA_REF_NAME)

# ------------ Creation des coordonnées de ref ----------------------------
def fit_cst_coefficients(airfoil_obj: asb.Airfoil, order: int) -> tuple:
    """
    Échantillonne un profil et trouve par moindres carrés les coefficients Kulfan 
    optimaux pour l'ordre (order) demandé.
    """
    # Échantillonnage dense pour un fit précis
    af_rep = airfoil_obj.repanel(n_points_per_side=100)
    coords = af_rep.coordinates
    
    # Séparation Extrados / Intrados
    idx_le = np.argmin(coords[:, 0])
    up_coords = coords[:idx_le + 1][::-1] if coords[0, 0] > coords[-1, 0] else coords[:idx_le + 1]
    lo_coords = coords[idx_le:]
    if lo_coords[0, 0] > lo_coords[-1, 0]:
        lo_coords = lo_coords[::-1]

    x_u, y_u = up_coords[:, 0], up_coords[:, 1]
    x_l, y_l = lo_coords[:, 0], lo_coords[:, 1]

    # Exclusion des points singuliers (x=0, x=1)
    eps = 0.02
    mask_u = (x_u > eps) & (x_u < 1 - eps)
    mask_l = (x_l > eps) & (x_l < 1 - eps)
    x_u, y_u = x_u[mask_u], y_u[mask_u]
    x_l, y_l = x_l[mask_l], y_l[mask_l]
    
    # Fonctions de classe classiques pour profil à bord d'attaque rond et BA pointu
    def class_function(x):
        return np.sqrt(np.maximum(x, 1e-10)) * (1 - x)
    
    def bernstein_matrix(x, order):
        matrix = np.zeros((len(x), order))
        for i in range(order):
            matrix[:, i] = comb(order - 1, i) * (x ** i) * ((1 - x) ** (order - 1 - i))
        return matrix

    # Résolution par moindres carrés : y / C(x) = B(x) * W
    # Extrados
    A_u = bernstein_matrix(x_u, order)
    rhs_u = y_u / (class_function(x_u) + 1e-12)
    w_u, _, _, _ = np.linalg.lstsq(A_u, rhs_u, rcond=None)
    
    # Intrados (Inversé par convention KulfanAirfoil d'AeroSandBox)
    A_l = bernstein_matrix(x_l, order)
    rhs_l = -y_l / (class_function(x_l) + 1e-12)
    w_l, _, _, _ = np.linalg.lstsq(A_l, rhs_l, rcond=None)
    
    # Vérification de validité
    if not (np.all(np.isfinite(w_u)) and np.all(np.isfinite(w_l))):
        raise ValueError(f"Fit CST non convergé pour {airfoil_obj.name}")

    return w_u, w_l

# Génération dynamiqe des coefficients de référence
try:
    _REF_UPPER, _REF_LOWER = fit_cst_coefficients(ref_airfoil, N_CST)
except Exception as e:
    print(f"Erreur lors du fit CST automatique : {e}")
    _REF_UPPER = np.zeros(N_CST)
    _REF_LOWER = np.zeros(N_CST)

# -------- Calcul dynamique des bornes Kulfan -----------------------
BOUNDS_Au = [(_REF_UPPER[i] - DELTA, _REF_UPPER[i] + DELTA) for i in range(N_CST)]
BOUNDS_Al = [(_REF_LOWER[i] - DELTA, _REF_LOWER[i] + DELTA) for i in range(N_CST)]
BOUNDS_AIRFOIL = BOUNDS_Au + BOUNDS_Al  # 2 * N_CST bornes par profil

# Bornes géométriques 
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

# Vecteur complet 
BOUNDS = BOUNDS_AIRFOIL + BOUNDS_AIRFOIL + BOUNDS_GEOM
N_VAR  = len(BOUNDS) 

# ─────────────────────────────────────────────────────────────────────────────
# 3. FILTRE GÉOMÉTRIQUE BAS COÛT
# ─────────────────────────────────────────────────────────────────────────────

TE_MIN_M         = 0.001   # Épaisseur bord de fuite minimale (1 mm)
THICKNESS_MIN    = 0.06    # Épaisseur relative minimale (6 %)

def _cst_eval(x: np.ndarray, weights: np.ndarray, upper: bool) -> np.ndarray:
    """Évalue la formule CST directement depuis les poids de Bernstein."""
    C = np.sqrt(np.maximum(x, 1e-10)) * (1 - x)
    n = len(weights) - 1
    B = np.zeros_like(x)
    for i, w in enumerate(weights):
        B += w * comb(n, i, exact=False) * (x ** i) * ((1 - x) ** (n - i))
    return C * B 


def geometric_penalty(au_weights: np.ndarray, al_weights: np.ndarray, chord: float) -> float:
    """
    Vérifie la validité géométrique en évaluant directement la formule CST.
    Distribution cosinus : dense près du BA et du BF, pas d'angle mort.
    """
    try:
        # Distribution cosinus — 80 points, couvre x ∈ [0.001, 0.999]
        theta   = np.linspace(0.01, np.pi - 0.01, 80)
        x_check = 0.5 * (1 - np.cos(theta))

        y_upper = _cst_eval(x_check, np.array(au_weights), upper=True)
        y_lower = -_cst_eval(x_check, np.array(al_weights), upper=False)

        thickness = y_upper - y_lower  # doit être > 0 partout

        # Croisement extrados/intrados
        if np.any(thickness < 0):
            return 1e6 + 1e4 * float(-np.min(thickness))

        # Épaisseur relative max
        t_max = float(np.max(thickness))
        if t_max < THICKNESS_MIN:
            return 1e6 + 1e4 * (THICKNESS_MIN - t_max)

        # Bord de fuite absolu
        te_thick = float(au_weights[-1] + al_weights[-1]) * 0.5 * chord 
        if te_thick < TE_MIN_M:
            return 5e4 * (TE_MIN_M - te_thick) / TE_MIN_M

        # Épaisseur minimale absolue au BA (x < 2%)
        x_le    = np.array([0.005, 0.010, 0.015, 0.020])
        y_u_le  = _cst_eval(x_le, np.array(au_weights), upper=True)
        y_l_le  = -_cst_eval(x_le, np.array(al_weights), upper=False)
        if np.any(y_u_le - y_l_le < 1e-4):   # épaisseur < 0.01% corde au BA
            return 1e6

    except Exception:
        return 1e6

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANALYSE STRUCTURELLE 
# ─────────────────────────────────────────────────────────────────────────────

SIGMA_CARBONE = 400e6   # Pa, Limite admissible carbone/époxy (valeur conservative)


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
    return M_flex * y_max / (I_xx + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANALYSE FLUIDE
# ─────────────────────────────────────────────────────────────────────────────

P  = atmosphere.pressure()
P_VAPOR = atmosphere.vapor_pressure()
RHO     = atmosphere.density()

# σ_v = (P_atm + ρgh - P_vapor) / (½ρV²). Cavitation si Cp_min < -σ_v 

SIGMA_CAV = (P - P_VAPOR) / (0.5 * RHO * cfg["v_cruise"] ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONSTRUCTION DE L'AVION (forward pass, sans CasADi)
# ─────────────────────────────────────────────────────────────────────────────

def decode(x: np.ndarray) -> dict:
    """Découpe le vecteur DE en sous-ensembles"""
    idx = 0
    Au_root = x[idx : idx + N_CST]; idx += N_CST
    Al_root = x[idx : idx + N_CST]; idx += N_CST
    Au_tip  = x[idx : idx + N_CST]; idx += N_CST
    Al_tip  = x[idx : idx + N_CST]; idx += N_CST
    
    return {
        "Au_root":        Au_root,
        "Al_root":        Al_root,
        "Au_tip":         Au_tip,
        "Al_tip":         Al_tip,
        "span":           float(x[idx]),
        "root_chord":     float(x[idx+1]),
        "tip_chord":      float(x[idx+2]),
        "twist":          float(x[idx+3]),
        "s_span":         float(x[idx+4]),
        "s_root_chord":   float(x[idx+5]),
        "s_tip_chord":    float(x[idx+6]),
        "s_twist":        float(x[idx+7]),
        "fuselage_length":float(x[idx+8]),
        "cg_ratio":       float(x[idx+9]),
        "alpha":          float(x[idx+10]),
    }


def interpolate_kulfan(af1: asb.KulfanAirfoil, af2: asb.KulfanAirfoil, r: float) -> asb.KulfanAirfoil:
    """Interpolation linéaire des coefficients CST entre root et tip"""
    return asb.KulfanAirfoil(
        upper_weights       = (1 - r) * af1.upper_weights       + r * af2.upper_weights,
        lower_weights       = (1 - r) * af1.lower_weights       + r * af2.lower_weights,
        leading_edge_weight = (1 - r) * af1.leading_edge_weight + r * af2.leading_edge_weight,
        TE_thickness        = (1 - r) * af1.TE_thickness        + r * af2.TE_thickness,
    )


def build_airplane(p: dict) -> tuple:
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

def objective(x: np.ndarray) -> float:
    """
    Minimise la trainée
    Contraintes :
      Hard (filtre géométrique) : épaisseur, croisement, TE
      Soft (pénalité K×violation²) : équilibres et contraintes fluides/structurelles
    """
    p = decode(x)

    # ── Filtre géométrique (fast-fail) ───────────────────────────────────────
    geo_pen = geometric_penalty(p["Au_root"], p["Al_root"], p["root_chord"]) \
            + geometric_penalty(p["Au_tip"],  p["Al_tip"],  p["tip_chord"]) 
    if geo_pen > 0:
        return 1e6 + geo_pen # Arrêt immédiat si c'est un vrai monstre
    
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
    
    # Coefficients de pénalité (par niveau d'importance)
    K1 = 2000      # Contraintes critiques
    K2 = 1000      # Contraintes de pilotage
    K3 = 500       # Contraintes de confort
    '''
    Calibrage empirique: pour que l'opti ne passe pas outre les contraintes soft, 
    il faut que K soit d'un ordre de grandeur supérieur à la fonction objectif,
    içi F = 80N environ, donc K entre 500 et 2000
    ATTENTION : normaliser les contraintes
    '''


    # Fonction générique pour appliquer le carré uniquement en cas de dépassement strict
    def soft_penalty(valeur: float, limite_basse: float, limite_haute: float, ref: float) -> float:
        if valeur < limite_basse:
            return ((limite_basse - valeur) / (abs(ref) + 1e-9)) ** 2
        elif valeur > limite_haute:
            return ((valeur - limite_haute) / (abs(ref) + 1e-9)) ** 2
        return 0.0

    # Portance = Poids 
    penalty += K1 * ((WEIGHT - L) / WEIGHT) ** 2

    # Équilibre de tangage 
    X_cg      = p["cg_ratio"] * mean_chord
    M_wing    = Cm * q_cruise * wing.area() * mean_chord
    M_greem   = rig_mass * 9.81 * (-(X_cg - x_mast))
    M_total   = M_wing + M_MAST + M_greem
    M_reference = WEIGHT * mean_chord
    M_tol_brute = 0.05 * M_reference  # On tolère un moment résiduel 
    if abs(M_total) > M_tol_brute:
        penalty += K1 * ((abs(M_total) - M_tol_brute) / M_reference) ** 2

    # Contrainte structurelle 
    try:
        I_xx, y_max = section_inertia(af_root, p["root_chord"])
        sigma       = bending_stress(I_xx, y_max, WEIGHT / 2, p["span"] / 2)
        if sigma > SIGMA_CARBONE:
            penalty += K1 * ((sigma - SIGMA_CARBONE) / SIGMA_CARBONE) ** 2
    except Exception:
        penalty += K1

    # CL décollage 
    S_wing  = wing.area()
    q_to    = 0.5 * float(atmosphere.density()) * cfg["v_takeoff"] ** 2
    CL_to   = WEIGHT / (q_to * S_wing + 1e-9)
    if CL_to > CL_MAX_TO:
        penalty += K1 * ((CL_to - CL_MAX_TO) / CL_MAX_TO) ** 2

    # Marge statique (version analytique bas coût avec le volume de queue)
    try:
        # approximation foyer à 25% puis décalage dû au stab de 80% du point neutre
        x_ac_wing = float(wing.xsecs[0].xyz_le[0]) + 0.25 * mean_chord
        v_h       = (stab.area() * p["fuselage_length"]) / (S_wing * mean_chord + 1e-9)
        x_neutral_point_ratio = (x_ac_wing / mean_chord) + (0.80 * v_h)
        SM        = x_neutral_point_ratio - p["cg_ratio"]

        sm_low, sm_high = cfg["sm_range"][0], cfg["sm_range"][1]
        penalty += K2 * soft_penalty(SM, sm_low, sm_high, ref=sm_low)
    except Exception:
        penalty += K2

    # Surface aile 
    sw_low, sw_high = cfg["area_target_range"][0], cfg["area_target_range"][1]
    S_ref = (sw_low + sw_high) / 2
    penalty += K2 * soft_penalty(S_wing, sw_low, sw_high, ref=S_ref)

    # Surface stab
    S_stab = stab.area()
    ss_low, ss_high = cfg["stab_area_range"][0], cfg["stab_area_range"][1]
    S_stab_ref = (ss_low + ss_high) / 2
    penalty += K2 * soft_penalty(S_stab, ss_low, ss_high, ref=S_stab_ref)

    # Force stab 
    try:
        F_stab = float(aero["wing_aero_components"][1].L)
        f_low, f_high = cfg["stab_load_range"][0], cfg["stab_load_range"][1]
        F_ref  = abs(f_high - f_low) if abs(f_high - f_low) > 1e-3 else 50.0
        penalty += K3 * soft_penalty(F_stab, f_low, f_high, ref=F_ref)
    except Exception:
        penalty += K3

    # Volume de queue Vh (Vérification géométrique pour le comportement de l'assiette)
    try:
        vh_low, vh_high = cfg["vh_range"][0], cfg["vh_range"][1]
        penalty += K3 * soft_penalty(v_h, vh_low, vh_high, ref=vh_low)
    except Exception:
        penalty += K3

    # Cavitation 
    try:
        t_max = af_root.max_thickness() 
    except Exception:
        t_max = 0.12

    Cp_min_est = -(1.2 * abs(CL) + 3.0 * t_max)
    if Cp_min_est < SIGMA_CAV:
        # Correction du bug syntaxique (ajout du signe + et du multiplicateur *)
        penalty += K3 * ((-Cp_min_est - SIGMA_CAV) / SIGMA_CAV) ** 2

    return D_total + penalty

# ─────────────────────────────────────────────────────────────────────────────
# 8. OPTIMISATION — DIFFERENTIAL EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

DE_PARAMS = {
    "strategy":   "best1bin",    # Bonne convergence sur problèmes continus
    "maxiter":    300,
    "popsize":    12,            
    "tol":        1e-5,
    "mutation":   (0.5, 1.0),   # Plage de mutation adaptative
    "recombination": 0.85,
    "seed":       42,
    "workers":    -1,            # Parallel processing sur tous les cœurs disponibles
    "polish":     False,          # Affinage L-BFGS-B sur le meilleur individu
    "updating":   "deferred",    # Nécessaire pour workers=-1
    "disp":       True,
}

# Run à un seul point d'entrée
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
    print(f"  Fonction objectif     : {result.fun:.3f} N")
    print(f"  Évaluations : {result.nfev}")
    return result.x

from scipy.stats import qmc
from scipy.optimize import minimize

# Run à plusieurs points d'entrée distribués uniformément
def run_multistart(n_starts: int = 5) -> np.ndarray:
    """
    Lance n_starts runs DE avec des initialisations Sobol décalées.
    Chaque run couvre une région différente de l'espace de recherche.
    Retourne le meilleur individu global.
    """
    print(f"\n{'='*65}")
    print(f"  MULTI-START DE — {n_starts} runs | CAS : {CASE.upper()}")
    print(f"{'='*65}\n")

    lb = np.array([b[0] for b in BOUNDS])
    ub = np.array([b[1] for b in BOUNDS])

    best_x   = None
    best_val = np.inf

    for run_idx in range(n_starts):
        seed = 42 + run_idx * 137   # Seeds décorrélées

        # Population initiale Sobol — couvre l'espace plus uniformément que random
        # Chaque run utilise un scramble différent pour diversifier
        pop_size  = DE_PARAMS["popsize"] * N_VAR
        sampler   = qmc.Sobol(d=N_VAR, scramble=True, seed=seed)
        init_pop  = qmc.scale(sampler.random(pop_size), lb, ub)

        print(f"  ─── Run {run_idx + 1}/{n_starts} (seed={seed}) ───")

        result = differential_evolution(
            objective,
            BOUNDS,
            init=init_pop,            # Population Sobol
            seed=seed,
            strategy=DE_PARAMS["strategy"],
            maxiter=DE_PARAMS["maxiter"],
            popsize=DE_PARAMS["popsize"],
            tol=DE_PARAMS["tol"],
            mutation=DE_PARAMS["mutation"],
            recombination=DE_PARAMS["recombination"],
            workers=DE_PARAMS["workers"],
            polish=False,             # On polit séparément après
            updating=DE_PARAMS["updating"],
            disp=True,
            callback=_de_callback,
        )

        print(f"  Run {run_idx+1} → Fonction Objectif = {result.fun:.3f} N | "
              f"{'OK' if result.success else 'Failed'}")

        if result.fun < best_val:
            best_val = result.fun
            best_x   = result.x.copy()
            print(f"  - Nouveau meilleur global : {best_val:.3f} N\n")

    # Affinage final Nelder-Mead sur le meilleur individu
    print(f"\n  Affinage final (Nelder-Mead) depuis le meilleur global...")
    refined = minimize(
        objective,
        best_x,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-7, "disp": True},
    )
    if refined.fun < best_val:
        print(f"  ✓ Affinage réussi : {best_val:.3f} → {refined.fun:.3f} N")
        best_x = refined.x

    return best_x


_run_counter = {"n": 0}

def _de_callback(xk: np.ndarray, convergence: float) -> bool:
    """Affiche la progression tous les 20 appels"""
    _run_counter["n"] += 1
    if _run_counter["n"] % 20 == 0:
        val = objective(xk)
        print(f"    gen ~{_run_counter['n']} | convergence={convergence:.4f} | D={val:.2f} N")
    return False   # False = ne pas arrêter


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
    SM_true    = -((float(aero2["Cm"]) - Cm) / (float(aero2["CL"]) - CL + 1e-9))

    x_ac_wing = float(wing.xsecs[0].xyz_le[0]) + 0.25 * mean_chord
    v_h       = (stab.area() * p["fuselage_length"]) / (wing.area() * mean_chord + 1e-9)
    x_neutral_point_ratio = (x_ac_wing / mean_chord) + (0.80 * v_h)
    SM_approx       = x_neutral_point_ratio - p["cg_ratio"]

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
    print(f"  Marge statique théorique   : {SM_true*100:.1f} %")
    print(f"  Marge statique approximée pour l'optimisation   : {SM_approx*100:.1f} %")
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
               SM_true, SM_approx, F_stab, v_h, M_total, sigma, rho, mu, X_cg)

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
               SM_true, SM_approx, F_stab, v_h, M_total, sigma, rho, mu, X_cg):
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
        f"| Marge statique théorique | {SM_true*100:.1f} % |",
        f"| Marge statique approximée pour l'optimisation | {SM_approx*100:.1f} % |",
        f"| Position CG | {p['cg_ratio']*100:.1f} % c̄ ({X_cg*100:.1f} cm du BA) |",
        f"| Moment résiduel | {M_total:.4f} N·m |",
        f"| Force stab | {F_stab:.1f} N |",
        f"| Volume de queue | {v_h:.3f} |",
        f"| σ flexion root | {sigma/1e6:.1f} MPa / {SIGMA_CARBONE/1e6:.0f} MPa |",
        f"| σ_v cavitation | {SIGMA_CAV:.2f} |",
        f"",
    ]

    warnings_list = []
    if SM_true * 100 > 70:
        warnings_list.append(f"⚠️ SM élevée ({SM_true*100:.1f}%) — maniabilité réduite.")
    if SM_true * 100 < 5:
        warnings_list.append(f"⚠️ SM très faible ({SM_true*100:.1f}%) — risque d'instabilité.")
    if abs(M_total) > 5:
        warnings_list.append(f"⚠️ Moment résiduel important ({M_total:.2f} N·m).")
    if sigma > SIGMA_CARBONE:
        warnings_list.append(f"⚠️ Contrainte flexion ({sigma/1e6:.0f} MPa) dépasse la limite admissible.")
    if warnings_list:
        lines += [f"## ⚠️ Avertissements", f""] + [f"- {w}" for w in warnings_list] + [f""]

    with open(os.path.join(out_dir, "fiche_technique.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 10. POINT D'ENTRÉE ET AFFINAGE FINAL
# ─────────────────────────────────────────────────────────────────────────────

from scipy.optimize import minimize

N_starts = phy["search_space"]["N_starts"]

if __name__ == "__main__":
    x_best   = run_multistart(n_starts=N_starts)
    full_report(x_best)