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

with open(ROOT / "config" / "parameters.yaml") as f:
    phy = yaml.safe_load(f)
with open(ROOT / "config" / "scenarios.yaml") as f:
    SCENARIOS = yaml.safe_load(f)

CASE = phy["case"]
if CASE not in SCENARIOS:
    raise ValueError(f"Case '{CASE}' not found. Options: {list(SCENARIOS.keys())}")
cfg = SCENARIOS[CASE]

atmosphere = Atmosphere()

# Masses
mass      = phy["pilot"]["mass_kg"] + phy["board"]["mass_kg"]
rig_mass  = cfg["rig_mass_kg"]
WEIGHT    = (mass + rig_mass) * 9.81

# Fixed geometry
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

# Mast drag & moment (constants for a given cfg)
q_cruise    = 0.5 * atmosphere.density() * cfg["v_cruise"] ** 2
D_MAST      = q_cruise * ((mast_chord_bot + chord_wl) / 2) * profondeur_imm * cd_mast
M_MAST      = -D_MAST * (profondeur_imm / 2)

# ─────────────────────────────────────────────────────────────────────────────
# 2. KULFAN (CST) PARAMETERIZATION
# ─────────────────────────────────────────────────────────────────────────────

ref_airfoil = asb.Airfoil(NACA_REF_NAME)

# ------------ Reference coordinates creation ----------------------------
def fit_cst_coefficients(airfoil_obj: asb.Airfoil, order: int) -> tuple:
    """
    Samples an airfoil and finds by least squares the optimal Kulfan
    coefficients for the requested (order).
    """
    # Dense sampling for an accurate fit
    af_rep = airfoil_obj.repanel(n_points_per_side=100)
    coords = af_rep.coordinates

    # Upper / Lower separation
    idx_le = np.argmin(coords[:, 0])
    up_coords = coords[:idx_le + 1][::-1] if coords[0, 0] > coords[-1, 0] else coords[:idx_le + 1]
    lo_coords = coords[idx_le:]
    if lo_coords[0, 0] > lo_coords[-1, 0]:
        lo_coords = lo_coords[::-1]

    x_u, y_u = up_coords[:, 0], up_coords[:, 1]
    x_l, y_l = lo_coords[:, 0], lo_coords[:, 1]

    # Exclude singular points (x=0, x=1)
    eps = 0.02
    mask_u = (x_u > eps) & (x_u < 1 - eps)
    mask_l = (x_l > eps) & (x_l < 1 - eps)
    x_u, y_u = x_u[mask_u], y_u[mask_u]
    x_l, y_l = x_l[mask_l], y_l[mask_l]

    # Classical class functions for an airfoil with round leading edge and sharp trailing edge
    def class_function(x):
        return np.sqrt(np.maximum(x, 1e-10)) * (1 - x)

    def bernstein_matrix(x, order):
        matrix = np.zeros((len(x), order))
        for i in range(order):
            matrix[:, i] = comb(order - 1, i) * (x ** i) * ((1 - x) ** (order - 1 - i))
        return matrix

    # Least-squares resolution: y / C(x) = B(x) * W
    # Upper surface
    A_u = bernstein_matrix(x_u, order)
    rhs_u = y_u / (class_function(x_u) + 1e-12)
    w_u, _, _, _ = np.linalg.lstsq(A_u, rhs_u, rcond=None)

    # Lower surface (inverted by AeroSandBox KulfanAirfoil convention)
    A_l = bernstein_matrix(x_l, order)
    rhs_l = -y_l / (class_function(x_l) + 1e-12)
    w_l, _, _, _ = np.linalg.lstsq(A_l, rhs_l, rcond=None)

    # Validity check
    if not (np.all(np.isfinite(w_u)) and np.all(np.isfinite(w_l))):
        raise ValueError(f"CST fit did not converge for {airfoil_obj.name}")

    return w_u, w_l

# Dynamic generation of reference coefficients
try:
    _REF_UPPER, _REF_LOWER = fit_cst_coefficients(ref_airfoil, N_CST)
except Exception as e:
    print(f"Error during automatic CST fit: {e}")
    _REF_UPPER = np.zeros(N_CST)
    _REF_LOWER = np.zeros(N_CST)

# -------- Dynamic computation of Kulfan bounds -----------------------
BOUNDS_Au = [(_REF_UPPER[i] - DELTA, _REF_UPPER[i] + DELTA) for i in range(N_CST)]
BOUNDS_Al = [(_REF_LOWER[i] - DELTA, _REF_LOWER[i] + DELTA) for i in range(N_CST)]
BOUNDS_AIRFOIL = BOUNDS_Au + BOUNDS_Al  # 2 * N_CST bounds per airfoil

# Geometric bounds
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

# Complete vector
BOUNDS = BOUNDS_AIRFOIL + BOUNDS_AIRFOIL + BOUNDS_GEOM
N_VAR  = len(BOUNDS)

# ─────────────────────────────────────────────────────────────────────────────
# 3. FREE AIRFOIL VALIDITY
# ─────────────────────────────────────────────────────────────────────────────

def _cst_eval(x: np.ndarray, weights: np.ndarray, upper: bool) -> np.ndarray:
    """Evaluates the CST formula directly from the Bernstein weights."""
    C = np.sqrt(np.maximum(x, 1e-10)) * (1 - x)
    n = len(weights) - 1
    B = np.zeros_like(x)
    for i, w in enumerate(weights):
        B += w * comb(n, i, exact=False) * (x ** i) * ((1 - x) ** (n - i))
    return C * B

def geometric_penalty(au_weights: np.ndarray, al_weights: np.ndarray, chord: float) -> float:
    """
    Geometric validity of a Kulfan (CST) airfoil
    """
    try:
        # thickness positivity (no crossing)
        x_check = np.linspace(0.05, 0.95, 100) # exclude x=0 and x=1
        y_upper = _cst_eval(x_check, au_weights, upper=True)
        y_lower = -_cst_eval(x_check, al_weights, upper=False)
        thickness =  y_upper - y_lower

        if np.any(thickness <= 0.0):
            return 1e6 + 1e4 * float(-np.min(thickness))

        # LE radius of curvature: R_le = 0.5 * w0^2
        # Enforcing w0 >= 0.10 guarantees a radius of curvature > 0.5% of the chord

    except Exception:
        return 1e6

    return 0.0  # The airfoil is valid

# ─────────────────────────────────────────────────────────────────────────────
# 4. STRUCTURAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

SIGMA_CARBONE = 300e6   # Pa, carbon/epoxy admissible limit (very conservative value to limit AR)

def _section_bending_inertia(au_weights: np.ndarray, al_weights: np.ndarray, chord: float) -> tuple:
    """
    Computes the second moment of area I_xx and y_max of an airfoil section
    via Green's formula on the coordinate polygon.
    Solid section assumption (conservative for a shell).
    """
    x_check = np.linspace(0.0, 1.0, 100)
    xp = x_check * chord
    yp_upper = _cst_eval(x_check, au_weights, upper=True) * chord
    yp_lower = -_cst_eval(x_check, al_weights, upper=False) * chord

    # Closed polygon reconstruction (upper then reversed lower)
    x_poly = np.concatenate([xp, xp[::-1]])
    y_poly = np.concatenate([yp_upper, yp_lower[::-1]])
    n  = len(xp)

    I_xx = 0.0
    for i in range(n - 1):
        # Green's formula for the inertia of a polygon
        cross = x_poly[i] * y_poly[i + 1] - x_poly[i + 1] * y_poly[i]
        I_xx += (y_poly[i] ** 2 + y_poly[i] * y_poly[i + 1] + y_poly[i + 1] ** 2) * cross
    I_xx  = abs(I_xx) / 12.0
    if I_xx < 1e-12:
        I_xx = 1e-12

    y_max = float(np.max(np.abs(y_poly)))
    return I_xx, y_max

def _section_torsional_inertia(au_weights: np.ndarray, al_weights: np.ndarray, chord: float) -> float:
    """
    Computes the torsion constant J (m^4) of a solid airfoil section.
    Modeled via an elliptical approximation.
    """
    # Clean sampling to extract the actual thickness
    x_check = np.linspace(0.0, 1.0, 100)
    yp_upper = _cst_eval(x_check, au_weights, upper=True) * chord
    yp_lower = -_cst_eval(x_check, al_weights, upper= False) * chord
    thickness = (yp_upper - yp_lower) * chord
    t_max = float(np.max(thickness))

    # Semi-axes of the equivalent ellipse
    c = chord
    a = c / 2.0
    b = t_max / 2.0

    # Analytical formula of the torsion inertia of a solid elliptical section
    J = (np.pi * (a**3) * (b**3)) / (a**2 + b**2 + 1e-15)
    return J


def _bending_stress(I_xx: float, y_max: float, lift_semi: float, span_semi: float) -> float:
    """
    Bending stress at the root.
    Load applied at the lift centroid of an elliptical distribution (span/4).
    """
    M_flex = lift_semi * (span_semi / 4.0)
    return M_flex * y_max / (I_xx+1e-10)


def _torsional_stress(J: float, chord: float, au_weights: np.ndarray, al_weights: np.ndarray, span_semi: float) -> float:
    """
    Computes the maximum shear stress (Tau) caused by a normalized torsion
    load case (independent of the immediate CL).
    """
    x_check = np.linspace(0.0, 1.0, 100)
    yp_upper = _cst_eval(x_check, au_weights, upper=True) * chord
    yp_lower = -_cst_eval(x_check, al_weights, upper=False) * chord
    thickness = (yp_upper - yp_lower) * chord
    t_max = float(np.max(thickness))

    # ---- NORMALIZED LOAD CASE (GENERIC STRESS TEST) ----
    # half the weight (WEIGHT/2) acts with a standard
    # hydrodynamic eccentricity (5% of the chord)
    M_torsion_ref = (WEIGHT / 2.0) * (0.05 * chord) * (span_semi / 2.0)

    # For a solid airfoil, max Tau stress is at the middle of the long face (max thickness)
    # Formula: Tau = Mt / J * max_thickness (with approximate shape coefficient)
    tau_torsion = (M_torsion_ref / (J + 1e-15)) * (t_max / 2.0)
    return tau_torsion

def VonMisesStress(au_weights: np.ndarray, al_weights: np.ndarray, chord: float, span: float) -> float:
    """Global Von Mises synthesis on the main wing averages"""
    lift_semi = WEIGHT / 2.0
    span_semi = span / 2.0

    # Pure Bending computation (Sigma)
    I_xx, y_max = _section_bending_inertia(au_weights, al_weights, chord)
    sigma_flex = _bending_stress(I_xx, y_max, lift_semi, span_semi)

    # Pure Torsion computation (Tau)
    J = _section_torsional_inertia(au_weights, al_weights, chord)
    tau_torsion = _torsional_stress(J, chord, au_weights, al_weights, span_semi)

    # Von Mises criterion (total equivalent stress)
    sigma_von_mises = np.sqrt(sigma_flex**2 + 3 * tau_torsion**2)
    return sigma_von_mises

# ─────────────────────────────────────────────────────────────────────────────
# 5. FLUID ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

P  = atmosphere.pressure()
P_VAPOR = atmosphere.vapor_pressure()
RHO     = atmosphere.density()

# σ_v = (P_atm + ρgh - P_vapor) / (½ρV²). Cavitation if Cp_min < -σ_v

SIGMA_CAV = (P - P_VAPOR) / (0.5 * RHO * cfg["v_cruise"] ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# 6. AIRPLANE CONSTRUCTION (forward pass, without CasADi)
# ─────────────────────────────────────────────────────────────────────────────

def decode(x: np.ndarray) -> dict:
    """Splits the DE vector into subsets"""
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
    """Linear interpolation of CST coefficients between root and tip"""
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

    # ── Main wing (CST morphing) ────────────────────────────────────────────
    wing_xsecs = []
    for i in range(N_WING):
        r        = i / (N_WING - 1)
        af_blend = interpolate_kulfan(af_root, af_tip, r)

        elliptic = np.sqrt(max(1 - r ** 2, 0))
        c_dist   = tip_chord + (root_chord - tip_chord) * (0.6 * elliptic + 0.4 * (1 - r))

        # Progressive closure at the tip (avoids an abrupt chord termination)
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

    # ── Mast ─────────────────────────────────────────────────────────────────
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

    # ── Stabilizer (symmetric NACA 0012 — no CST needed) ─────────────────────
    s_span       = p["s_span"]
    s_root_chord = p["s_root_chord"]
    s_tip_chord  = p["s_tip_chord"]
    s_twist      = p["s_twist"]
    x_stab_root  = x_fuselage_start + p["fuselage_length"] - 0.10

    stab_xsecs = []
    for i in range(N_STAB):
        r      = i / (N_STAB - 1)
        # Chord distribution specific to the stab (bug fixed vs NACA version)
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

    # ── Complete airplane ────────────────────────────────────────────────────
    airplane = asb.Airplane(
        wings=[wing, stab],
        fuselages=[fuselage_obj],
        xyz_ref=np.array([X_cg, 0.0, 0.0]),
        s_ref=wing.area(), c_ref=mean_chord, b_ref=wing.span(),
    )

    return airplane, wing, stab, mean_chord, af_root, mast_obj, fuselage_obj


# ─────────────────────────────────────────────────────────────────────────────
# 7. OBJECTIVE FUNCTION — DRAG + PENALTIES
# ─────────────────────────────────────────────────────────────────────────────

def objective(x: np.ndarray) -> float:
    """
    Minimizes drag.
    Constraints:
      Hard: geometric validity of airfoils
      Soft (penalty K*violation^2): trim and fluid/structural constraints
    """
    p = decode(x)

    # ── Geometric filter (fast-fail) ─────────────────────────────────────────
    geo_pen = geometric_penalty(p["Au_root"], p["Al_root"], p["root_chord"]) \
            + geometric_penalty(p["Au_tip"],  p["Al_tip"],  p["tip_chord"])
    if geo_pen > 0:
        return 1e6 + geo_pen # Immediate stop if it's a real monster

    # ── Airplane construction ────────────────────────────────────────────────
    try:
        airplane, wing, stab, mean_chord, af_root, _, _ = build_airplane(p)
    except Exception:
        return 1e6

    # ── Aerodynamic evaluation ───────────────────────────────────────────────
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

    # ── Penalties ────────────────────────────────────────────────────────────
    penalty = 0.0

    # Penalty coefficients (by importance level)
    K1 = 2000      # Critical constraints
    K2 = 1000      # Handling constraints
    K3 = 500       # Comfort constraints
    '''
    Empirical calibration: so the optimizer does not bypass soft constraints,
    K must be an order of magnitude larger than the objective function;
    here F ≈ 80 N, so K between 500 and 2000.
    WARNING: normalize the constraints.
    '''

    # Generic function applying the square only on strict overshoot
    def soft_penalty(valeur: float, limite_basse: float, limite_haute: float, ref: float) -> float:
        if valeur < limite_basse:
            return ((limite_basse - valeur) / (abs(ref) + 1e-9)) ** 2
        elif valeur > limite_haute:
            return ((valeur - limite_haute) / (abs(ref) + 1e-9)) ** 2
        return 0.0

    # Lift = Weight
    penalty += K1 * ((WEIGHT - L) / WEIGHT) ** 2

    # Pitch trim
    X_cg      = p["cg_ratio"] * mean_chord
    M_wing    = Cm * q_cruise * wing.area() * mean_chord
    M_greem   = rig_mass * 9.81 * (-(X_cg - x_mast))
    M_total   = M_wing + M_MAST + M_greem
    M_reference = WEIGHT * mean_chord
    M_tol_brute = 0.05 * M_reference  # Tolerate a residual moment
    if abs(M_total) > M_tol_brute:
        penalty += K1 * ((abs(M_total) - M_tol_brute) / M_reference) ** 2

    # Structural constraint
    try:
        sigma_total = VonMisesStress(p["Au_root"], p["Al_root"], p["root_chord"], p["span"])
        if sigma_total > SIGMA_CARBONE:
            penalty += K1 * ((sigma_total - SIGMA_CARBONE) / SIGMA_CARBONE) ** 2
    except Exception:
        penalty += K1

    # Takeoff CL
    S_wing  = wing.area()
    q_to    = 0.5 * float(atmosphere.density()) * cfg["v_takeoff"] ** 2
    CL_to   = WEIGHT / (q_to * S_wing + 1e-9)
    if CL_to > CL_MAX_TO:
        penalty += K1 * ((CL_to - CL_MAX_TO) / CL_MAX_TO) ** 2

    # Static margin
    try:
        op2   = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=p["alpha"] + 0.1, atmosphere=atmosphere)
        aero2 = asb.AeroBuildup(airplane, op2).run()
        SM    = -((float(aero2["Cm"]) - Cm) / (float(aero2["CL"]) - CL + 1e-9))

        sm_low, sm_high = cfg["sm_range"][0], cfg["sm_range"][1]
        penalty += K2 * soft_penalty(SM, sm_low, sm_high, ref=sm_low)
    except Exception:
        penalty += K2


    # Wing area
    sw_low, sw_high = cfg["area_target_range"][0], cfg["area_target_range"][1]
    S_ref = (sw_low + sw_high) / 2
    penalty += K2 * soft_penalty(S_wing, sw_low, sw_high, ref=S_ref)

    # Stab area
    S_stab = stab.area()
    ss_low, ss_high = cfg["stab_area_range"][0], cfg["stab_area_range"][1]
    S_stab_ref = (ss_low + ss_high) / 2
    penalty += K2 * soft_penalty(S_stab, ss_low, ss_high, ref=S_stab_ref)

    # Stab force
    try:
        F_stab = float(aero["wing_aero_components"][1].L)
        f_low, f_high = cfg["stab_load_range"][0], cfg["stab_load_range"][1]
        F_ref  = abs(f_high - f_low) if abs(f_high - f_low) > 1e-3 else 50.0
        penalty += K3 * soft_penalty(F_stab, f_low, f_high, ref=F_ref)
    except Exception:
        penalty += K3

    # Tail volume Vh (geometric check for pitch attitude behavior)
    try:
        v_h     = (stab.area() * p["fuselage_length"]) / (wing.area() * mean_chord)
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
    if Cp_min_est < - SIGMA_CAV:
        penalty += K3 * ((-Cp_min_est - SIGMA_CAV) / SIGMA_CAV) ** 2

    # If the current individual is better than the run record, memorize its components
    _run_score["D_total"] = D_total
    _run_score["penalty"] = penalty

    return D_total + penalty

# ─────────────────────────────────────────────────────────────────────────────
# 8. OPTIMISATION — DIFFERENTIAL EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

DE_PARAMS = {
    "strategy":   "best1bin",    # Good convergence on continuous problems
    "maxiter":    30,
    "popsize":    12,
    "tol":        1e-3,
    "mutation":   (0.5, 1.0),   # Adaptive mutation range
    "recombination": 0.85,
    "seed":       42,
    "workers":    -1,            # Parallel processing on all available cores
    "polish":     False,          # L-BFGS-B polish on the best individual
    "updating":   "deferred",    # Required for workers=-1
    "disp":       True,
    "maxiter": 100,            # Limit to 100 generations max per trial (instead of 300+)
    "tol": 1e-3,               # Looser relative tolerance to stop earlier
    "atol": 0.1,               # Stop as soon as the population varies by less than 0.1 Newton

}

# Single-entry-point run
def run_optimization() -> np.ndarray:
    print(f"\n{'='*65}")
    print(f"  DIFFERENTIAL EVOLUTION — CASE: {CASE.upper()}")
    print(f"{'='*65}")
    print(f"  Variables       : {N_VAR} ({N_CST*4} CST + 11 geom.)")
    print(f"  Population      : {DE_PARAMS['popsize'] * N_VAR} individuals")
    print(f"  Max generations : {DE_PARAMS['maxiter']}")
    print(f"  Parameterization: Kulfan CST of order {N_CST}")
    print(f"  σ_cav target    : {SIGMA_CAV:.2f} (critical Cp_min)\n")

    result = differential_evolution(objective, BOUNDS, **DE_PARAMS)

    print(f"\n  Convergence: {'✓' if result.success else '✗ (partial)'}")
    print(f"  Objective function    : {result.fun:.3f} N")
    print(f"  Evaluations: {result.nfev}")
    return result.x

from scipy.stats import qmc
from scipy.optimize import minimize

# Run with multiple uniformly distributed entry points
def run_multistart(n_starts: int = 5) -> np.ndarray:
    """
    Runs n_starts DE runs with offset Sobol initializations.
    """
    print(f"\n{'='*65}")
    print(f"  MULTI-START DE — {n_starts} runs | CASE: {CASE.upper()}")
    print(f"{'='*65}\n")

    lb = np.array([b[0] for b in BOUNDS])
    ub = np.array([b[1] for b in BOUNDS])

    pop_size = DE_PARAMS["popsize"] * N_VAR

    # single global sampler before the loop
    sampler = qmc.Sobol(d=N_VAR, scramble=True, seed=42)

    best_x   = None
    best_val = np.inf

    for run_idx in range(n_starts):
        seed = 42 + run_idx * 137   # Decorrelated seeds
        raw_samples = sampler.random(pop_size)
        init_pop    = qmc.scale(raw_samples, lb, ub)

        print(f"  ─── Run {run_idx + 1}/{n_starts} (seed={seed}) ───")

        result = differential_evolution(
            objective,
            BOUNDS,
            init=init_pop,
            seed=seed,
            strategy=DE_PARAMS["strategy"],

            # --- COMPUTATIONAL EFFORT CONTROL ---
            maxiter=DE_PARAMS["maxiter"],
            tol=DE_PARAMS["tol"],
            atol=DE_PARAMS["atol"],
            # ──────────────────────────────────────

            popsize=DE_PARAMS["popsize"],
            mutation=DE_PARAMS["mutation"],
            recombination=DE_PARAMS["recombination"],
            workers=DE_PARAMS["workers"],
            polish=False,
            updating=DE_PARAMS["updating"],
            disp=False,
            callback=_de_callback,
        )

        print(f"  Run {run_idx+1} → Objective function = {result.fun:.3f} N | "
              f"{'OK' if result.success else 'Convergence failed'}")

        if result.fun < best_val:
            best_val = result.fun
            best_x   = result.x.copy()
            print(f"  - New global best: {best_val:.3f} N\n")

    # Final polish on the best individual
    print(f"\n  Final polish (L-BFGS-B) from the global best...")
    refined = minimize(
        objective,
        best_x,
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-6, "disp": True},
    )
    if refined.fun < best_val:
        print(f"  ✓ Polish successful: {best_val:.3f} → {refined.fun:.3f} N")
        best_x = refined.x

    return best_x


_run_counter = {"n": 0}
_run_score = {
    "D_total": np.inf,
    "penalty": np.inf
    } # Objective function

def _de_callback(xk: np.ndarray, convergence: float) -> bool:
    """Displays progress every 20 actual generations"""
    _run_counter["n"] += 1

    if (_run_counter["n"]-1) % 5 == 0:
        progress = min((_run_counter["n"] / DE_PARAMS["maxiter"]) * 100, 100.0)

        # Safe retrieval of scores from the global dictionary
        D_total = _run_score["D_total"]
        penalty = _run_score["penalty"]

        D_str = f"{D_total:.2f} N" if D_total < np.inf else "Filters..."
        Penalty_str = f"{penalty:.2f} N" if penalty < np.inf else "..."

        # Visual progress bar
        bar_length = 10
        filled = int(round(progress / 10))
        bar = "█" * filled + "░" * (bar_length - filled)

        # Perfect column alignment for the console
        print(f"    Gen {_run_counter['n']:3d} | D_total = {D_str:<9} | Penalty = {Penalty_str:<13} | Population: [{bar}] {progress:5.1f}%")
    return False   # False = do not stop


# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPORT — TECHNICAL SHEET & CAD FILES
# ─────────────────────────────────────────────────────────────────────────────

def full_report(x: np.ndarray) -> None:
    """Evaluates the best individual, prints the summary and exports the files."""
    p = decode(x)
    try:
        airplane, wing, stab, mean_chord, af_root, mast_obj, fuselage_obj = build_airplane(p)
    except Exception as e:
        print(f"Airplane construction error: {e}")
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
    sigma   = VonMisesStress(p["Au_root"], p["Al_root"], (p["root_chord"]+p["tip_chord"])/2 ,p["span"]/ 2)

    # ── Console output ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL RESULT — {CASE.upper()}")
    print(f"{'='*65}")
    print(f"  Total drag       : {D_total:.2f} N      L/D : {L/D_total:.2f}")
    print(f"  Trim incidence   : {p['alpha']:.2f}°")
    print(f"  Wing area        : {wing.area()*1e4:.0f} cm²    AR : {wing.aspect_ratio():.2f}")
    print(f"  Span             : {p['span']*100:.1f} cm")
    print(f"  Chord R/T        : {p['root_chord']*1000:.0f} / {p['tip_chord']*1000:.0f} mm")
    print(f"  Stab area        : {stab.area()*1e4:.0f} cm²")
    print(f"  Fuselage         : {p['fuselage_length']*100:.0f} cm")
    print(f"  Static margin    : {SM*100:.1f} %")
    print(f"  Residual moment  : {M_total:.4f} N·m")
    print(f"  Stab force       : {F_stab:.1f} N")
    print(f"  Tail volume      : {v_h:.3f}")
    print(f"  CG               : {p['cg_ratio']*100:.1f} % c̄")
    print(f"  Von Mises stress : {sigma/1e6:.1f} MPa / {SIGMA_CARBONE/1e6:.0f} MPa admissible.")
    print(f"  Cavitation σ_v   : {SIGMA_CAV:.2f}  |  Cp_min ≈ {-2*abs(CL):.2f}")
    print(f"{'='*65}\n")

    # ── File export ──────────────────────────────────────────────────────────
    now_str      = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir      = os.path.join("outputs", f"{CASE}_kulfan_{now_str}")
    airfoils_dir = os.path.join(out_dir, "airfoils")
    os.makedirs(airfoils_dir, exist_ok=True)

    # Markdown technical sheet
    _export_md(out_dir, p, wing, stab, mean_chord, D_total, L, D, CL, Cm,
               SM, F_stab, v_h, M_total, sigma, rho, mu, X_cg)

    # .dat airfoils (Selig format for XFLR5)
    airplane_sol = airplane
    for i, xsec in enumerate(airplane_sol.wings[0].xsecs):
        _export_dat(xsec.airfoil, os.path.join(airfoils_dir, f"wing_sec_{i}.dat"), f"wing_sec_{i}")
    for i, xsec in enumerate(airplane_sol.wings[1].xsecs):
        _export_dat(xsec.airfoil, os.path.join(airfoils_dir, f"stab_sec_{i}.dat"), f"stab_sec_{i}")

    # XFLR5 XML
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

    print(f"  ✓ Technical sheet : {out_dir}/fiche_technique.md")
    print(f"  ✓ .dat airfoils   : {airfoils_dir}/")
    print(f"  ✓ XFLR5 XML       : {xml_path}")


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
        f"# Technical Sheet — {CASE.upper()} | Kulfan CST",
        f"", f"*Generated on {now_str}*", f"", f"---", f"",
        f"## 0. Run Parameters", f"",
        f"| Variable | Min | Max |", f"|:---|:---|:---|",
        f"| Span (m) | {phy['wing']['span_bounds'][0]} | {phy['wing']['span_bounds'][1]} |",
        f"| Root chord (m) | {phy['wing']['root_chord_bounds'][0]} | {phy['wing']['root_chord_bounds'][1]} |",
        f"| Fuselage (m) | {phy['fuselage']['length_bounds'][0]} | {phy['fuselage']['length_bounds'][1]} |",
        f"| CG ratio | {cfg['cg_range'][0]} | {cfg['cg_range'][1]} |",
        f"", f"| Scenario constraint | Min | Max |", f"|:---|:---|:---|",
        f"| Wing area (m²) | {cfg['area_target_range'][0]} | {cfg['area_target_range'][1]} |",
        f"| SM | {cfg['sm_range'][0]} | {cfg['sm_range'][1]} |",
        f"| Stab load (N) | {cfg['stab_load_range'][0]} | {cfg['stab_load_range'][1]} |",
        f"| Vh | {cfg['vh_range'][0]} | {cfg['vh_range'][1]} |",
        f"", f"---", f"",
        f"## 1. Flight conditions", f"",
        f"| Parameter | Value |", f"|:---|:---|",
        f"| Total weight | {WEIGHT:.1f} N ({WEIGHT/9.81:.0f} kg) |",
        f"| V takeoff | {cfg['v_takeoff']} m/s |",
        f"| V cruise | {cfg['v_cruise']} m/s |",
        f"| Re root | {Re_root:.2e} |",
        f"| Re tip | {Re_tip:.2e} |",
        f"", f"---", f"",
        f"## 2. Geometry", f"",
        f"| Parameter | Value |", f"|:---|:---|",
        f"| Wing area | {wing.area()*1e4:.0f} cm² |",
        f"| Span | {p['span']*100:.1f} cm |",
        f"| Aspect ratio | {wing.aspect_ratio():.2f} |",
        f"| Root chord | {p['root_chord']*1000:.0f} mm |",
        f"| Tip chord | {p['tip_chord']*1000:.0f} mm |",
        f"| Twist | {p['twist']:.2f}° |",
        f"| Stab area | {stab.area()*1e4:.0f} cm² |",
        f"| Stab span | {p['s_span']*100:.1f} cm |",
        f"| Fuselage | {p['fuselage_length']*100:.0f} cm |",
        f"", f"---", f"",
        f"## 3. Performance", f"",
        f"| Parameter | Value |", f"|:---|:---|",
        f"| L/D ratio | {L/D_total:.2f} |",
        f"| Total drag | {D_total:.2f} N |",
        f"| Incidence | {p['alpha']:.2f}° |",
        f"| CL cruise | {CL_c:.3f} |",
        f"| CD cruise | {CD_c:.4f} |",
        f"", f"---", f"",
        f"## 4. Stability & Structure", f"",
        f"| Parameter | Value |", f"|:---|:---|",
        f"| Theoretical static margin | {SM*100:.1f} % |",
        f"| CG position | {p['cg_ratio']*100:.1f} % c̄ ({X_cg*100:.1f} cm from LE) |",
        f"| Residual moment | {M_total:.4f} N·m |",
        f"| Stab force | {F_stab:.1f} N |",
        f"| Tail volume | {v_h:.3f} |",
        f"| σ bending root | {sigma/1e6:.1f} MPa / {SIGMA_CARBONE/1e6:.0f} MPa |",
        f"| σ_v cavitation | {SIGMA_CAV:.2f} |",
        f"",
    ]

    warnings_list = []
    if SM * 100 > 70:
        warnings_list.append(f"⚠️ High SM ({SM*100:.1f}%) — reduced maneuverability.")
    if SM * 100 < 5:
        warnings_list.append(f"⚠️ Very low SM ({SM*100:.1f}%) — risk of instability.")
    if abs(M_total) > 5:
        warnings_list.append(f"⚠️ Significant residual moment ({M_total:.2f} N·m).")
    if sigma > SIGMA_CARBONE:
        warnings_list.append(f"⚠️ Bending stress ({sigma/1e6:.0f} MPa) exceeds the admissible limit.")
    if warnings_list:
        lines += [f"## ⚠️ Warnings", f""] + [f"- {w}" for w in warnings_list] + [f""]

    with open(os.path.join(out_dir, "fiche_technique.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 10. ENTRY POINT AND FINAL POLISH
# ─────────────────────────────────────────────────────────────────────────────

from scipy.optimize import minimize

N_starts = phy["search_space"]["N_starts"]

if __name__ == "__main__":
    x_best   = run_multistart(n_starts=N_starts)
    full_report(x_best)