# =================================================================================
# Hydrofoil Optimization V3 — Macroscopic Parametric
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

# ─────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

with open(ROOT / "config/parameters.yaml") as f:
    phy = yaml.safe_load(f)
with open(ROOT / "config/scenarios.yaml") as f:
    SCENARIOS = yaml.safe_load(f)

CASE = phy["case"]
if CASE not in SCENARIOS:
    raise ValueError(f"Case '{CASE}' not found. Options: {list(SCENARIOS.keys())}")
cfg = SCENARIOS[CASE]

atmosphere = Atmosphere()

# ── Masses ────────────────────────────────────────────────────────────────────
mass     = phy["pilot"]["mass_kg"] + phy["board"]["mass_kg"]
rig_mass = cfg["rig_mass_kg"]
WEIGHT   = (mass + rig_mass) * 9.81

# ── Fixed geometries inherited from YAML (mast, fuselage, wing sweep, etc.) ──
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

# Geometric sanity check: the stab LE must remain inside the fuselage
_fuse_len_min = cfg["fuselage_length_bounds"][0]
if STAB_FUSE_OFFSET > _fuse_len_min:
    raise ValueError(
        f"stab.fuselage_offset ({STAB_FUSE_OFFSET*100:.1f} cm) > "
        f"fuselage_length_bounds[0] ({_fuse_len_min*100:.1f} cm) — "
        f"the stab LE would lie ahead of the fuselage nose."
    )

# ── Mast (constants) ──────────────────────────────────────────────────────────
chord_wl = mast_chord_bot * (1 - profondeur_imm / mast_length) \
         + mast_chord_top * (profondeur_imm / mast_length)
q_cruise = 0.5 * atmosphere.density() * cfg["v_cruise"] ** 2
D_MAST   = q_cruise * ((mast_chord_bot + chord_wl) / 2) * profondeur_imm * cd_mast
M_MAST   = -D_MAST * (profondeur_imm / 2)

# ── Mechanical limits — fatigue sizing ──
# σ_ult = allowable "design" carbon-epoxy laminate (~400 MPa, implicit knockdowns).
# σ_admissible = σ_ult × fatigue_ratio (cyclic >10⁶ cycles) ≈ 160 MPa.
# The design constraint compares σ_peak (dynamic load × LOAD_PEAK_FACTOR) to σ_admissible.
SIGMA_ULTIMATE     = phy["wing"]["ultimate_stress_mpa"] * 1e6
FATIGUE_RATIO      = phy["wing"]["fatigue_allowable_ratio"]
LOAD_PEAK_FACTOR   = phy["wing"]["load_peak_factor"]
SIGMA_ADMISSIBLE   = FATIGUE_RATIO * SIGMA_ULTIMATE          # e.g.: 0.40 × 400 = 160 MPa
P_ATM         = atmosphere.pressure()
P_VAPOR       = atmosphere.vapor_pressure()
SIGMA_CAV     = (P_ATM + atmosphere.density() * 9.81 * profondeur_imm - P_VAPOR) \
              / (0.5 * atmosphere.density() * cfg["v_cruise"] ** 2)

# ─────────────────────────────────────────────────────────────────────────────
# 2. GEOMETRY — airfoils and dimensions
# ─────────────────────────────────────────────────────────────────────────────

WING_AIRFOIL_NAME = cfg["wing_airfoil"]
STAB_AIRFOIL_NAME = phy["stab"]["airfoil"]

# Tip ratio — tip = TIP_RATIO × root (tip chord is derived, not a variable)
TIP_RATIO_W = phy["wing"]["tip_chord_ratio"]
TIP_RATIO_S = phy["stab"]["tip_chord_ratio"]

# Warm-start of the wing planform: use the scenario reference if provided
# (scenarios.yaml), otherwise the default values from parameters.yaml.
WING_SPAN         = cfg.get("wing_span_init",       phy["wing"]["span_init"])
WING_ROOT_CHORD   = cfg.get("wing_root_chord_init", phy["wing"]["root_chord_init"])
WING_TIP_CHORD    = WING_ROOT_CHORD * TIP_RATIO_W

# Stab warm-start
STAB_SPAN         = cfg["stab_span"]
STAB_ROOT_CHORD   = cfg["stab_root_chord"]
STAB_TIP_CHORD    = STAB_ROOT_CHORD * TIP_RATIO_S

# Airfoil precomputation, with robust import (ASB does NOT have all NACA 6-series).
def _load_airfoil(name: str, fallback: str = "naca2410"):
    """Load an ASB airfoil, fall back to `fallback` if the library lacks the coords."""
    try:
        af = asb.Airfoil(name)
        if af.coordinates is not None and len(af.coordinates) >= 30:
            return af
    except Exception:
        pass
    print(f"  ⚠ Airfoil '{name}' unavailable in ASB → fallback '{fallback}'")
    return asb.Airfoil(fallback)


WING_AIRFOIL = _load_airfoil(WING_AIRFOIL_NAME, fallback="naca2410")
STAB_AIRFOIL = _load_airfoil(STAB_AIRFOIL_NAME, fallback="naca0012")
# If a fallback occurred, the official name used downstream remains the requested
# name, but the exported .dat will correspond to the fallback (visible in XFLR5).
WING_AIRFOIL_NAME = WING_AIRFOIL.name
STAB_AIRFOIL_NAME = STAB_AIRFOIL.name

try:
    WING_THICKNESS_REL = float(WING_AIRFOIL.max_thickness())
except Exception:
    WING_THICKNESS_REL = 0.12

# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTIMIZATION VECTOR
# ─────────────────────────────────────────────────────────────────────────────
# NB: α_cruise is NOT an opti variable — it is derived by internal solve
# (L = WEIGHT) at each evaluation, which guarantees vertical equilibrium.
# Tip chords are derived: tip = TIP_RATIO × root (no dedicated opti variable,
# see TIP_RATIO_W / TIP_RATIO_S).
# x = [fuselage_length, cg_ratio, wing_setting_angle, twist, s_twist, alpha_to,
#      wing_span, wing_root_chord, stab_span, stab_root_chord]

# Wing bounds: scenario override (discipline-specific incidence) with fallback
# on the global defaults from parameters.yaml.
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

STAB_AR_RANGE = tuple(cfg.get("stab_aspect_ratio_range",
                              phy["stab"].get("aspect_ratio_range", [4.0, 14.0])))


def decode(x: np.ndarray) -> dict:
    """Split the DE vector → dictionary of macroscopic parameters.
    Tip chords are derived (root × TIP_RATIO) — not opti variables."""
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


# Warm-start: midpoint of the bounds, wing + stab planforms initialized on the
# reference geometry of the scenario.
X_REF = np.array([(b[0] + b[1]) / 2 for b in BOUNDS])
X_REF[6]  = WING_SPAN
X_REF[7]  = WING_ROOT_CHORD
X_REF[8]  = STAB_SPAN
X_REF[9]  = STAB_ROOT_CHORD
X_REF = np.clip(X_REF, LB, UB)

# ─────────────────────────────────────────────────────────────────────────────
# 4. AIRPLANE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_airplane(p: dict) -> tuple:
    """
    Assemble the wing, fuselage, mast and stabilizer AeroSandBox.

    Geometry anchored on the TRAILING EDGE:
        x_te(r) = x_te_root + r^sweep_power × (span/2 × tan(sweep_deg))
        x_le(r) = x_te(r) − c(r)
        c(r)   = elliptic chord
    The TE is therefore monotonic by construction, regardless of the chord law
    (the elliptic distribution has dc/dr → −∞ at the tip, which used to push
    the TE back up near the tip with the old QC-anchored parametrization).
    """
    SWEEP_POWER_W = 1.5   # wing TE curvature (1=linear, >1=curved)
    SWEEP_POWER_S = 1.5   # same for stab

    # ── Main wing ────────────────────────────────────────────────────────────
    span_w  = p["wing_span"]
    root_w  = p["wing_root_chord"]
    tip_w   = min(p["wing_tip_chord"], root_w * 0.95)  # safety tip < root
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

    # ── Mast ──────────────────────────────────────────────────────
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

    # ── Stabilizer ───────────────────────────────────────────────────────────
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
# 5. STRUCTURAL CONSTRAINT
# ─────────────────────────────────────────────────────────────────────────────

def von_mises_root(chord_root: float, span: float,
                   load_factor: float = 1.0) -> float:
    """
    Von Mises at the root — carbon shell + polystyrene core (neglected).
    `load_factor` multiplies the static load to estimate the dynamic PEAK
    (maneuvering, waves, pumping). The result is then compared against
    SIGMA_ADMISSIBLE (= fatigue_ratio × σ_ult, ~120 MPa) — not against σ_ult.
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
# 6. PITCH DYNAMICS — Cm_α, SM and ω_n derived directly from AeroBuildup.
# ─────────────────────────────────────────────────────────────────────────────
# All stability metrics (Cm_α, SM/c̄, SM/l_t, ω_n) are computed by finite
# difference over the two AeroBuildup points already obtained during the
# cruise trim (alpha_lo, alpha_hi) — therefore at zero cost and within ~3% of
# the VLM truth. The former analytical formula (Helmbold + V_H + linearly-
# calibrated de_da) was removed: it introduced 15-25% error on Cm_α and
# therefore on ω_n, which caused the pilotability target to be silently
# missed quite often.

# Pitch inertia of the pilot + board + rig system
_M_TOTAL    = phy["pilot"]["mass_kg"] + phy["board"]["mass_kg"] + rig_mass
_R_GYR      = phy["pilot"].get("gyration_radius_m", 0.30)
I_YY_SYSTEM = _M_TOTAL * _R_GYR ** 2

# ω_n target range for the current discipline (calibrated freeride per scenario
# in scenarios.yaml). A single scope — no subdivision by pilot skill level.
PILOT_FREQ_LO, PILOT_FREQ_HI = cfg["pilotability_freq"]

# Tolerance on the residual moment — the pilot compensates via stance, so the
# reference quantity is NOT the wing chord (aviation legacy, aberrant for a
# hydrofoil) but the pilot trim authority, expressed in N·m.
# Cf. parameters.yaml#pilot.trim_moment_tolerance_N_m.
M_TOL_TRIM = phy["pilot"].get("trim_moment_tolerance_N_m", 25.0)

# Stab CL_α slope (NACA 0012-like), used for the control authority
# dF_stab/dα = CL_α × q × S_stab. Target defined per scenario.
CL_ALPHA_STAB_PER_DEG = phy["stab"].get("cl_alpha_per_deg", 0.10)


def get_pilot_freq_range() -> tuple:
    """Return (f_lo, f_hi) — the ω_n target for the current scenario."""
    return PILOT_FREQ_LO, PILOT_FREQ_HI


def pitch_frequency_hz(Cm_alpha: float, q: float, S: float, c_ref: float) -> float:
    """
    Natural frequency of the short-period mode (Hz).
        ω_n² = -Cm_α × q × S × c̄ / I_yy        [I_yy ≈ m_total × r_gyr²]
    Returns NaN if Cm_α > 0 (unstable foil, imaginary ω_n).
    """
    omega_n_sq = -Cm_alpha * q * S * c_ref / max(I_YY_SYSTEM, 1e-9)
    if omega_n_sq <= 0:
        return float("nan")
    return float(np.sqrt(omega_n_sq) / (2.0 * np.pi))


# ─────────────────────────────────────────────────────────────────────────────
# 7. OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

K1 = 4000.0   # critical constraints (lift, equilibrium, structure)
K2 = 2000.0   # piloting constraints (SM, surface)
K3 = 1000.0    # comfort constraints  (Vh, stab force, cavitation)

_run_counter = {"n": 0}


def soft_penalty(val: float, lo: float, hi: float, ref: float) -> float:
    """Continuous normalized linear penalty.

    P = |violation| / ref (exp=1). Constant gradient — no flattening near the
    boundary. Unlike the quadratic form (∂P/∂x → 0 at the frontier, which
    made small violations "free"), here a 10% violation truly costs 0.1
    units, not 0.01.
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
    Find alpha_cruise such that L(alpha) ≈ target_L by linear interpolation
    (L is quasi-linear in α in the pre-stall AeroBuildup regime).

    Returns (alpha_trim, aero_at_trim, bracket) where bracket = dict with aero_lo,
    aero_hi, alpha_lo, alpha_hi. The bracket yields the real dCm/dα = Cm_α
    for free (from 2 AeroBuildup runs already computed) — used by
    pitch_dynamics_from_aero.
    """
    op_lo = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_lo, atmosphere=atmosphere)
    op_hi = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_hi, atmosphere=atmosphere)
    aero_lo = asb.AeroBuildup(airplane, op_lo).run()
    aero_hi = asb.AeroBuildup(airplane, op_hi).run()
    bracket = {"aero_lo": aero_lo, "aero_hi": aero_hi,
               "alpha_lo": alpha_lo, "alpha_hi": alpha_hi}
    L_lo, L_hi = float(aero_lo["L"]), float(aero_hi["L"])
    if abs(L_hi - L_lo) < 1.0:
        return float(alpha_lo), aero_lo, bracket  # plateau (rare, saturated foil)
    alpha_trim = alpha_lo + (target_L - L_lo) / (L_hi - L_lo) * (alpha_hi - alpha_lo)
    # Clamp to physical bounds (reasonable α_cruise)
    alpha_trim = max(-3.0, min(12.0, alpha_trim))
    op_trim = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_trim, atmosphere=atmosphere)
    aero_trim = asb.AeroBuildup(airplane, op_trim).run()
    return float(alpha_trim), aero_trim, bracket


def pitch_dynamics_from_aero(bracket: dict, mean_chord: float,
                             l_t: float) -> dict:
    """
    Pitch dynamics computed DIRECTLY from the 2 AeroBuildup points of the trim
    bracket (already computed by trim_alpha_for_lift — zero cost).

        Cm_α (rad⁻¹) = dCm / dα
        SM/c̄         = -dCm / dCL    (negative if stable; we return |·|)
        SM (m)       = (SM/c̄) × c̄
        SM/l_t       = SM / l_t

    Accuracy: within ~3% of VLM, vs ~15-20% for the analytical formula with
    linearly-calibrated de_da.
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
    Minimize the total drag (D_aero + D_mast) under soft constraints.
    α_cruise is DERIVED (trim solver L=WEIGHT), not an opti variable.
    """
    x = np.clip(x, LB, UB)
    p = decode(x)

    try:
        airplane, wing, stab, mean_chord, _, _ = build_airplane(p)
    except Exception:
        return 1e6  # AeroSandBox structural exception

    # ── Cruise: α_cruise solved internally so that L = WEIGHT ───────────────
    try:
        alpha_trim, aero_c, bracket = trim_alpha_for_lift(airplane, target_L=WEIGHT)
        p["alpha_cruise"] = alpha_trim
        L  = float(aero_c["L"])
        D  = float(aero_c["D"])
        Cm = float(aero_c["Cm"])
        CL = float(aero_c["CL"])
    except Exception:
        return 1e6

    # ── Aerodynamic evaluation — takeoff point ──────────────────────────────
    try:
        op_to   = asb.OperatingPoint(velocity=cfg["v_takeoff"],
                                     alpha=p["alpha_to"],
                                     atmosphere=atmosphere)
        aero_to = asb.AeroBuildup(airplane, op_to).run()
        L_to    = float(aero_to["L"])
        D_to    = float(aero_to["D"])
        CL_to   = float(aero_to["CL"])
    except Exception:
        return 1e6  # AeroBuildup matrix exception

    D_total = D + D_MAST
    S_wing  = wing.area()
    S_stab  = stab.area()
    penalty = 0.0

    # Cruise lift = weight
    penalty += K1 * soft_penalty(L, WEIGHT, WEIGHT, ref=WEIGHT)

    # Takeoff lift: L_to ≥ weight + CL_to ≤ margin × CL_max_to.
    penalty += K1 * soft_penalty(L_to, WEIGHT, np.inf, ref=WEIGHT)
    cl_to_target = cfg["takeoff_cl_margin"] * CL_MAX_TO
    penalty += K1 * soft_penalty(CL_to, -np.inf, cl_to_target, ref=CL_MAX_TO)

    # Healthy operating regime: α_cruise ≥ -1° on cambered airfoil
    # (at α<-1° we enter the negative non-linear zone for moderately cambered NACA, Cm(α) becomes erratic)
    penalty += K2 * soft_penalty(p["alpha_cruise"], -1.0, np.inf, ref=2.0)

    # Pitch equilibrium — tolerance = pilot trim authority (stance),
    # cf. M_TOL_TRIM in parameters.yaml. No aberrant aviation tolerance.
    X_cg    = p["cg_ratio"] * mean_chord
    M_wing  = Cm * q_cruise * S_wing * mean_chord
    M_rig = rig_mass * 9.81 * (-(X_cg - x_mast))
    M_total = M_wing + M_MAST + M_rig
    penalty += K1 * soft_penalty(M_total, -M_TOL_TRIM, +M_TOL_TRIM, ref=M_TOL_TRIM)

    # Stab force — avoid the tandem solution
    try:
        F_stab = float(aero_c["wing_aero_components"][1].L)
        f_lo, f_hi = cfg["stab_load_range"]   # negative = downforce
        penalty += K2 * soft_penalty(F_stab, f_lo, f_hi, ref=abs(f_lo))
    except Exception:
        pass  # no penalty if AeroBuildup lacks the per-surface decomposition

    # Stab control authority: dF_stab/dα = CL_α × q × S_stab. This is the
    # aero force modulated per degree of α variation — what the pilot feels
    # when a wave tilts the foil or when changing stance. A constraint not
    # captured by ω_n (which only enforces total stiffness, not stab size).
    # Without this bound, the opti keeps a tiny stab at high CL → steady-state
    # force OK but insufficient correction authority in practice.
    dFda_target = cfg.get("stab_control_authority_min_N_per_deg", 0.0)
    if dFda_target > 0:
        dFda_stab = CL_ALPHA_STAB_PER_DEG * q_cruise * S_stab
        penalty += K2 * soft_penalty(dFda_stab, dFda_target, np.inf, ref=dFda_target)

    # Pilotability: Cm_a derived DIRECTLY from the AeroBuildup bracket (zero cost,
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
            penalty += K1 * 4.0  # unstable foil → strong penalty
    except Exception:
        penalty += K1

    # ── Auxiliary penalties ────────────────────────────────

    # Structural constraint: root von Mises (carbon shell).
    # Fatigue constraint: dynamic peak vs allowable limit (≠ static rupture).
    sigma_vm_peak = von_mises_root(p["wing_root_chord"], p["wing_span"],
                                    load_factor=LOAD_PEAK_FACTOR)
    penalty += K1 * soft_penalty(sigma_vm_peak, 0.0, SIGMA_ADMISSIBLE, ref=SIGMA_ADMISSIBLE)

    # Cavitation: Cp_min ≥ −σ_v
    Cp_min = -(1.2 * abs(CL) + 3.0 * WING_THICKNESS_REL)
    penalty += K3 * soft_penalty(Cp_min, -SIGMA_CAV, np.inf, ref=SIGMA_CAV)

    # Safety bound on the stab AR (prevents stabs that are too square or too thin).
    AR_stab_phys = (p["stab_span"] ** 2) / max(S_stab, 1e-6)
    ar_lo, ar_hi = STAB_AR_RANGE
    penalty += K2 * soft_penalty(AR_stab_phys, ar_lo, ar_hi, ref=0.5*(ar_lo+ar_hi))

    # Wing AR cap — non-hydro reasons (carbon manufacturing, stiffness, impact,
    # roll inertia, freeride pilotability). Industry reference per discipline.
    ar_wing_max = cfg.get("wing_aspect_ratio_max", np.inf)
    if np.isfinite(ar_wing_max):
        AR_wing_phys = (p["wing_span"] ** 2) / max(S_wing, 1e-6)
        penalty += K2 * soft_penalty(AR_wing_phys, -np.inf, ar_wing_max, ref=ar_wing_max)

    # Multi-point objective: D_cruise + W_TO × D_takeoff
    W_TAKEOFF = 0.3
    return D_total + W_TAKEOFF * D_to + penalty


# ─────────────────────────────────────────────────────────────────────────────
# 8. OPTIMIZATION — DIFFERENTIAL EVOLUTION + 3D REFINEMENT (LiftingLine)
# ─────────────────────────────────────────────────────────────────────────────

# ── ANSI colors for the console (auto-disabled if non-TTY) ─────────────
class C:
    _ON = sys.stdout.isatty()
    @classmethod
    def _w(cls, code, t): return f"\033[{code}m{t}\033[0m" if cls._ON else t
    @classmethod
    def head(cls, t): return cls._w("1;36", t)    # cyan bold (titles)
    @classmethod
    def sec(cls, t):  return cls._w("36",   t)    # cyan (sections)
    @classmethod
    def ok(cls, t):   return cls._w("32",   t)    # green
    @classmethod
    def warn(cls, t): return cls._w("33",   t)    # yellow
    @classmethod
    def dim(cls, t):  return cls._w("2",    t)    # gray/dim
    @classmethod
    def bold(cls, t): return cls._w("1",    t)    # bold


DE_PARAMS = {
    "strategy":      "best1bin",
    "maxiter":       40,
    "popsize":       25,           # 25 × 7 = 175 individuals — sufficient for 7 vars
    "tol":           1e-4,
    "atol":          1e-3,
    "mutation":      (0.5, 1.0),
    "recombination": 0.85,
    "workers":       -1,
    "polish":        False,
    "updating":      "deferred",
}


def _de_callback(_xk: np.ndarray, convergence: float) -> bool:
    """Light display every 5 generations (without re-evaluating xk)."""
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
    DE heuristic individuals — cover a mid fuselage + various wing/stab
    planforms + 2 CGs (forward/centered). Tip chords are derived (TIP_RATIO).
    """
    seeds = []
    fl_mid = 0.5 * (BOUNDS[0][0] + BOUNDS[0][1])
    # Wing planform: reference + 15% smaller variant
    wing_planforms = [
        (WING_SPAN, WING_ROOT_CHORD),
        (0.85 * WING_SPAN, 0.85 * WING_ROOT_CHORD),
    ]
    # Stab: reference + slightly larger variant (pitch authority)
    stab_planforms = [
        (STAB_SPAN, STAB_ROOT_CHORD),
        (1.10 * STAB_SPAN, 1.10 * STAB_ROOT_CHORD),
    ]
    for cg in (0.35 * (BOUNDS[1][0] + BOUNDS[1][1]),
               0.60 * (BOUNDS[1][0] + BOUNDS[1][1])):
        for w_sp, w_rc in wing_planforms:
            for s_sp, s_rc in stab_planforms:
                # x = [fl, cg, incidence, twist, s_twist, α_to,
                #      w_span, w_root, s_span, s_root]  (tips derived)
                seeds.append(np.array([fl_mid, cg, 0.0, -1.0, -2.0, 7.0,
                                       w_sp, w_rc, s_sp, s_rc]))
    return [np.clip(s, LB, UB) for s in seeds]


def run_multistart(n_starts: int = 1) -> np.ndarray:
    HBAR = "═" * 70
    print()
    print(C.head(HBAR))
    print(C.head(f"  HYDROFOIL OPTIMIZATION  ·  {CASE.upper()} freeride"))
    print(f"  {C.dim('Wing')}  {WING_AIRFOIL_NAME:<10}  warm-start "
          f"{WING_SPAN*100:5.1f} cm  /  {WING_ROOT_CHORD*1000:.0f} mm root")
    print(f"  {C.dim('Stab')}  {STAB_AIRFOIL_NAME:<10}  warm-start "
          f"{STAB_SPAN*100:5.1f} cm  /  {STAB_ROOT_CHORD*1000:.0f} mm root")
    _pop  = DE_PARAMS["popsize"] * N_VAR
    _gen  = DE_PARAMS["maxiter"]
    _runs = f"{n_starts} run" + ("s" if n_starts > 1 else "")
    print(f"  {C.dim(f'{N_VAR} vars · pop={_pop} · gen={_gen} · {_runs}')}")
    print(C.head(HBAR))

    val_ref = objective(X_REF)
    print(f"\n  {C.dim('Reference (center of bounds):')}  obj = {val_ref:.1f} N\n")

    heuristics = _heuristic_starts()
    best_x     = X_REF.copy()
    best_val   = val_ref

    for run_idx in range(n_starts):
        seed = 42 + run_idx * 137
        _run_counter["n"] = 0

        # Sampler re-created at each run for truly independent populations
        sampler     = qmc.Sobol(d=N_VAR, scramble=True, seed=seed)
        pop_size    = DE_PARAMS["popsize"] * N_VAR
        init_pop    = qmc.scale(sampler.random(pop_size), LB, UB)

        # Warm-start: X_REF + up to 4 heuristic individuals at the top
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
        line = f"  {mark} Run {run_idx+1} done   obj = {result.fun:.1f} N"
        if result.fun < best_val:
            best_val = result.fun
            best_x   = result.x.copy()
            print(line + "   " + C.bold(C.ok("★ new best")))
        else:
            print(line)

    # ── 3D refinement (LiftingLine + bounded Nelder-Mead) ────────────────────
    # LiftingLine captures the downwash and induced drag, which re-trims the angles correctly
    print()
    print(C.sec("  ── 3D refinement " + "─" * 53))
    try:
        import optFixedProfileRefine3d as R3
        print(f"  LiftingLine (fixed planform) from the best DE  "
              f"{C.dim(f'(DE obj = {best_val:.1f} N)')}")
        best_x_clipped = np.clip(best_x, LB, UB)
        x_refined, J_refined, n_iter = R3.refine_trim_3d(best_x_clipped, maxiter=80)
        # Compare: evaluate x_refined consistently with the DE objective (AB),
        # to decide whether to keep the 3D refinement or the pure DE solution.
        val_refined_ab = objective(x_refined)
        print(f"  {C.ok('✓')} {n_iter} iterations   J_3D = {J_refined:.1f} N   "
              f"{C.dim(f'(2D-equivalent = {val_refined_ab:.1f} N)')}")
        # We ALWAYS keep x_refined: it is the physically correct 3D solution,
        # even if its "AeroBuildup" score is worse (AB underestimates L at low α).
        best_x = x_refined
    except Exception as e:
        print(f"  {C.warn('~')} failure ({type(e).__name__}: {str(e)[:60]}) — falling back to DE solution")
        best_x = np.clip(best_x, LB, UB)

    return best_x


# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPORT & REPORT
# ─────────────────────────────────────────────────────────────────────────────

def next_output_dir(suffix: str = "", out_root: str = "outputs") -> str:
    """
    Build the next output directory in the format:
        outputs/{case}_{YYYYMMDD}_{NN}/
    NN increments automatically per day.
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
    """Console summary + markdown report + XFLR5 XML."""
    x = np.clip(x, LB, UB)
    p = decode(x)

    try:
        airplane, wing, stab, mean_chord, mast_obj, fuselage_obj = build_airplane(p)
    except Exception as e:
        print(f"Airplane construction error: {e}")
        return

    rho = atmosphere.density()
    mu  = atmosphere.dynamic_viscosity()

    # Cruise: α_cruise derived by trim L = WEIGHT (consistent with objective).
    alpha_trim, aero_c, bracket = trim_alpha_for_lift(airplane, target_L=WEIGHT)
    p["alpha_cruise"] = alpha_trim
    L, D   = float(aero_c["L"]), float(aero_c["D"])
    Cm, CL = float(aero_c["Cm"]), float(aero_c["CL"])
    D_total = D + D_MAST

    # Takeoff
    op_to   = asb.OperatingPoint(velocity=cfg["v_takeoff"], alpha=p["alpha_to"], atmosphere=atmosphere)
    aero_to = asb.AeroBuildup(airplane, op_to).run()
    L_to, D_to, CL_to = float(aero_to["L"]), float(aero_to["D"]), float(aero_to["CL"])

    # REAL pitch dynamics derived from the AeroBuildup bracket. No more analytical
    # vs VLM comparison — AeroBuildup is henceforth the source of truth (~3% of
    # VLM, which is largely sufficient).
    c_stab_mean = 0.5 * (p["stab_root_chord"] + p["stab_tip_chord"])
    l_t = (x_fuselage_start + p["fuselage_length"] - STAB_FUSE_OFFSET
           + 0.25 * c_stab_mean) - 0.25 * mean_chord
    pd = pitch_dynamics_from_aero(bracket, mean_chord, l_t)
    omega_n = pitch_frequency_hz(pd["Cm_alpha"], q_cruise, wing.area(), mean_chord)

    # Equilibrium
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

    # Root von Mises — static cruise (1g) AND dynamic peak (equivalent n_g).
    # The design criterion is on the PEAK: σ_peak < σ_admissible (fatigue).
    sigma_vm_static = von_mises_root(p["wing_root_chord"], p["wing_span"], load_factor=1.0)
    sigma_vm        = von_mises_root(p["wing_root_chord"], p["wing_span"],
                                     load_factor=LOAD_PEAK_FACTOR)

    # ── Derived metrics for display ─────────────────────────────────────────
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
    print(C.head(f"  RESULT  ·  {CASE.upper()} freeride"))
    print(C.head(HBAR))

    # ── Geometry ──
    print(SEC("Geometry"))
    ar_wing_cap = cfg.get("wing_aspect_ratio_max", float("inf"))
    ar_w_mark = C.ok("✓") if AR_w <= ar_wing_cap else C.warn("⚠")
    ar_dim = f" {C.dim(f'(cap {ar_wing_cap:.1f})')}" if np.isfinite(ar_wing_cap) else ""
    print(f"    Wing      {WING_AIRFOIL_NAME:<10}  span {p['wing_span']*100:5.1f} cm   "
          f"root/tip {p['wing_root_chord']*1000:3.0f}/{p['wing_tip_chord']*1000:2.0f} mm   "
          f"AR {AR_w:5.2f}{ar_dim} {ar_w_mark}")
    print(f"    Stab      {STAB_AIRFOIL_NAME:<10}  span {p['stab_span']*100:5.1f} cm   "
          f"root/tip {p['stab_root_chord']*1000:3.0f}/{p['stab_tip_chord']*1000:2.0f} mm   AR {AR_s:5.2f}")
    print(f"    Fuselage  {p['fuselage_length']*100:5.1f} cm    CG {p['cg_ratio']*100:5.1f} % c̄    "
          f"V_h {v_h:.3f}  {C.dim('(info)')}")

    # ── Trim ──
    print(SEC("Trim"))
    print(f"    α cruise    {p['alpha_cruise']:+6.2f}°    "
          f"α takeoff     {p['alpha_to']:+6.2f}°")
    print(f"    Wing incid. {p['wing_setting_angle']:+6.2f}°    Twist        {p['twist']:+6.2f}°    "
          f"Stab incid. {p['s_twist']:+6.2f}°")

    # ── Performance ──
    print(SEC("Performance"))
    print(f"    L cruise        {L:7.1f} N   {C.dim(f'/ {WEIGHT:.1f} N weight')}")
    print(f"    D aero          {D:7.2f} N   {C.dim(f'+ D_mast {D_MAST:5.2f} N = {D_total:6.2f} N')}")
    print(f"    {C.bold(f'L/D total       {L/D_total:7.2f}')}   {C.dim(f'(wing-only {L/D:.2f})')}")
    cl_to_info = f"≤ {cl_to_target:.3f} target / {CL_MAX_TO} stall"
    print(f"    L takeoff       {L_to:7.1f} N   D_to {D_to:5.2f} N   "
          f"CL_to {CL_to:.3f}  {C.dim(cl_to_info)}")
    print(f"    Re root         {Re_root:.2e}   {C.dim(f'Re tip {Re_tip:.2e}')}")

    # ── Pilotability & stability ──
    print(SEC("Pilotability & stability"))
    omega_str  = f"{omega_n:5.2f} Hz" if np.isfinite(omega_n) else "UNSTABLE"
    omega_ok   = np.isfinite(omega_n) and (f_lo <= omega_n <= f_hi)
    omega_mark = C.ok("✓") if omega_ok else C.warn("⚠")
    omega_dim  = f"freeride target {f_lo:.1f}-{f_hi:.1f} Hz"
    print(f"    ω_n           {omega_str}   {C.dim(omega_dim)}   {omega_mark}")
    np_gap = pd["SM_abs"] * 1000
    print(f"    Cm_α          {pd['Cm_alpha']:+6.2f} rad⁻¹    SM/l_t  {pd['SM_lt']*100:5.1f} %    "
          f"{C.dim(f'(NP-CG gap {np_gap:.1f} mm)')}")
    m_mark = C.ok("✓") if abs(M_total) <= M_tol else C.warn("⚠")
    print(f"    Residual mom. {M_total:+6.2f} N·m   {C.dim(f'tol ±{M_tol:.2f}')}   {m_mark}")
    print(f"    Stab force    {F_stab:+6.1f} N")
    # Stab control authority — not redundant with ω_n (cf. README).
    dFda_stab = CL_ALPHA_STAB_PER_DEG * q_cruise * stab.area()
    dFda_target = cfg.get("stab_control_authority_min_N_per_deg", 0.0)
    if dFda_target > 0:
        dFda_ok = dFda_stab >= dFda_target
        dFda_mark = C.ok("✓") if dFda_ok else C.warn("⚠")
        print(f"    dF_stab/dα    {dFda_stab:+6.1f} N/°   "
              f"{C.dim(f'≥ {dFda_target:.0f} (control authority)')}   {dFda_mark}")

    # ── Structure & cavitation ──
    print(SEC("Structure & cavitation"))
    s_mark = C.ok("✓") if sigma_vm <= SIGMA_ADMISSIBLE else C.warn("⚠")
    print(f"    σ_VM peak     {sigma_vm/1e6:6.1f} MPa   "
          f"{C.dim(f'≤ {SIGMA_ADMISSIBLE/1e6:.0f} fatigue')}   {s_mark}")
    print(f"    σ_VM static   {sigma_vm_static/1e6:6.1f} MPa   "
          f"{C.dim(f'≤ {SIGMA_ULTIMATE/1e6:.0f} rupture (σ_ult)')}")
    cav_mark = C.ok("✓") if Cp_min >= -SIGMA_CAV else C.warn("⚠")
    print(f"    Cp_min        {Cp_min:+6.2f}        "
          f"{C.dim(f'≥ {-SIGMA_CAV:.2f} σ_v cavitation')}   {cav_mark}")

    # ── Constraints ──
    sl_lo, sl_hi = cfg["stab_load_range"]
    stab_ok = np.isfinite(F_stab) and (sl_lo <= F_stab <= sl_hi)
    dFda_target_chk = cfg.get("stab_control_authority_min_N_per_deg", 0.0)
    dFda_stab_chk = CL_ALPHA_STAB_PER_DEG * q_cruise * stab.area()
    dFda_ok_chk = (dFda_target_chk == 0) or (dFda_stab_chk >= dFda_target_chk)
    checks = [
        ("L cruise ≥ weight",                L     >= 0.99 * WEIGHT, ""),
        ("L takeoff ≥ weight",               L_to  >= 0.99 * WEIGHT, f"{L_to:.1f} / {WEIGHT:.1f} N"),
        ("CL_to ≤ CL_max (stall)",           CL_to <= CL_MAX_TO + 1e-3, f"{CL_to:.3f} / {CL_MAX_TO}"),
        (f"CL_to ≤ {cfg['takeoff_cl_margin']:.0%} CL_max ({CASE})",
                                              CL_to <= cl_to_target + 1e-3, f"{CL_to:.3f} / {cl_to_target:.3f}"),
        ("|M_total| ≤ pilot trim authority", abs(M_total) <= M_tol, f"{M_total:+.2f} vs ±{M_tol:.2f} N·m"),
        (f"Stab downforce in [{sl_lo:.0f},{sl_hi:.0f}] N",
                                              stab_ok, f"{F_stab:+.1f} N"),
        (f"Stab authority dF/dα ≥ {dFda_target_chk:.0f} N/°",
                                              dFda_ok_chk, f"{dFda_stab_chk:.1f} N/°"),
        (f"Wing AR ≤ {ar_wing_cap:.1f}" if np.isfinite(ar_wing_cap) else "Wing AR free",
                                              (AR_w <= ar_wing_cap + 0.05), f"{AR_w:.2f}"),
        ("ω_n in freeride target",           omega_ok, f"{omega_n:.2f} vs [{f_lo:.1f}-{f_hi:.1f}] Hz"),
        ("Cavitation OK",                    Cp_min >= -SIGMA_CAV, ""),
        ("σ_VM peak ≤ σ_fatigue",            sigma_vm <= SIGMA_ADMISSIBLE, f"{sigma_vm/1e6:.0f} / {SIGMA_ADMISSIBLE/1e6:.0f} MPa"),
        ("α_cruise ≥ -1°",                   p["alpha_cruise"] >= -1.0, f"{p['alpha_cruise']:+.2f}°"),
    ]
    n_ok = sum(1 for _, ok, _ in checks if ok)
    print(SEC(f"Constraints  ({n_ok}/{len(checks)})"))
    for name, ok, detail in checks:
        mark = C.ok("✓") if ok else C.warn("⚠")
        suffix = f"   {C.dim(detail)}" if (not ok and detail) else ""
        print(f"    {mark}  {name}{suffix}")

    # ── Export ───────────────────────────────────────────────────────────────
    # Naming: outputs/{case}_{level}_{date}_{NN}/  (daily counter)
    out_dir = next_output_dir()
    run_tag = os.path.basename(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    _export_md(out_dir, p, wing, stab, mean_chord, D_total, L, D,
               F_stab, v_h, M_total, L_to, CL_to, rho, mu, X_cg, AR_w, AR_s,
               sigma_vm, sigma_vm_static, pd, omega_n, f_lo, f_hi)

    # ── XFLR5 + .dat airfoils export — V1 method (the only one that works in practice).
    # We RENAME each section with a unique name (wing_sec_0, ...) and write the
    # corresponding .dat into airfoils/. When loading the XML, XFLR5 retrieves
    # each section by name from the subfolder.
    airfoils_dir = os.path.join(out_dir, "airfoils")
    os.makedirs(airfoils_dir, exist_ok=True)

    def _export_profile_dat(af_obj, filename, name_internal):
        """Normalized Selig format, repaneled at 50 pts/side (cf. V1)."""
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

    # IMPORTANT → COPY the airfoil per section before renaming.
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

    # ── Exports block ────────────────────────────────────────────────────────
    print(SEC("Exports"))
    print(f"    {C.ok('✓')} Sheet      {C.dim(out_dir + '/fiche_technique.md')}")
    print(f"    {C.ok('✓')} Airfoils   {C.dim(airfoils_dir + '/')}")

    # XFLR5 XML — the xsec airfoil names have just been renamed
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
        print(f"    {C.ok('✓')} XFLR5 XML  {C.dim(xml_path)}")
    except Exception as e:
        print(f"    {C.warn('~')} XFLR5 XML not exported: {e}")

    # Save optimal X for refine_3d.py (auto load)
    np.save(os.path.join(out_dir, "x_best.npy"), x)
    print(f"    {C.ok('✓')} x_best     {C.dim(out_dir + '/x_best.npy')}")

    # ── Saturated bounds report ─────────────────────────────────────────────
    VAR_NAMES = ["fuselage_length", "cg_ratio", "wing_setting_angle", "twist",
                 "s_twist", "alpha_to",
                 "wing_span", "wing_root_chord",
                 "stab_span", "stab_root_chord"]
    TOL = 0.02  # 2% of the bound width
    saturated = []
    for i, (lo, hi) in enumerate(BOUNDS):
        width = hi - lo
        if x[i] - lo < TOL * width:
            saturated.append(f"{VAR_NAMES[i]} ↓ {lo:.3g}")
        elif hi - x[i] < TOL * width:
            saturated.append(f"{VAR_NAMES[i]} ↑ {hi:.3g}")
    if saturated:
        print()
        print(SEC(f"Saturated bounds  ({len(saturated)})"))
        for s in saturated:
            print(f"    {C.warn('⚠')}  {s}")
        print(C.dim("       → loosen these bounds to explore further"))
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
        f"# Technical Sheet — {CASE.upper()} freeride",
        "", f"*Generated on {now_str}*", "", "---", "",
        f"## 1. Optimized variables ({N_VAR})", "",
        "| Variable | Value | Bounds |", "|:---|:---|:---|",
        f"| Fuselage length | {p['fuselage_length']*100:.1f} cm | [{BOUNDS[0][0]*100:.0f}–{BOUNDS[0][1]*100:.0f}] cm |",
        f"| CG ratio | {p['cg_ratio']*100:.1f}% c̄ | [{BOUNDS[1][0]*100:.0f}–{BOUNDS[1][1]*100:.0f}]% |",
        f"| Wing incidence | {p['wing_setting_angle']:.2f}° | [{BOUNDS[2][0]}–{BOUNDS[2][1]}]° |",
        f"| Twist | {p['twist']:.2f}° | [{BOUNDS[3][0]}–{BOUNDS[3][1]}]° |",
        f"| Stab incidence | {p['s_twist']:.2f}° | [{BOUNDS[4][0]}–{BOUNDS[4][1]}]° |",
        f"| α takeoff | {p['alpha_to']:.2f}° | [{BOUNDS[5][0]}–{BOUNDS[5][1]}]° |",
        f"| Wing span | {p['wing_span']*100:.1f} cm | [{BOUNDS[6][0]*100:.0f}–{BOUNDS[6][1]*100:.0f}] cm |",
        f"| Wing root chord | {p['wing_root_chord']*1000:.0f} mm | [{BOUNDS[7][0]*1000:.0f}–{BOUNDS[7][1]*1000:.0f}] mm |",
        f"| Stab span | {p['stab_span']*100:.1f} cm | [{BOUNDS[8][0]*100:.0f}–{BOUNDS[8][1]*100:.0f}] cm |",
        f"| Stab root chord | {p['stab_root_chord']*1000:.0f} mm | [{BOUNDS[9][0]*1000:.0f}–{BOUNDS[9][1]*1000:.0f}] mm |",
        f"| α cruise (derived L=W) | {p['alpha_cruise']:.2f}° | — |",
        "", "---", "",
        "## 2. Flight conditions", "",
        "| Parameter | Value |", "|:---|:---|",
        f"| Total weight | {WEIGHT:.1f} N ({WEIGHT/9.81:.0f} kg) |",
        f"| V takeoff | {cfg['v_takeoff']} m/s |",
        f"| V cruise | {cfg['v_cruise']} m/s |",
        f"| Re root | {Re_root:.2e} |",
        f"| Re tip | {Re_tip:.2e} |",
        "", "---", "",
        "## 3. Performance (2 flight points)", "",
        "| Parameter | Cruise | Takeoff |", "|:---|:---|:---|",
        f"| Speed (m/s) | {cfg['v_cruise']} | {cfg['v_takeoff']} |",
        f"| α (°) | {p['alpha_cruise']:.2f} | {p['alpha_to']:.2f} |",
        f"| L (N) | {L:.1f} | {L_to:.1f} |",
        f"| D (N) | {D:.2f} | — |",
        f"| CL | {CL_c:.3f} | {CL_to:.3f} |",
        f"| CD | {CD_c:.4f} | — |",
        f"| D total (+ mast) | {D_total:.2f} | — |",
        f"| L/D ratio | {L/D_total:.2f} | — |",
        "", "---", "",
        "## 4. Geometry from opti", "",
        "| Parameter | Wing | Stab |", "|:---|:---|:---|",
        f"| Airfoil  | {WING_AIRFOIL_NAME} | {STAB_AIRFOIL_NAME} |",
        f"| Area (cm²) | {wing.area()*1e4:.0f} | {stab.area()*1e4:.0f} |",
        f"| Aspect ratio | {AR_w:.2f} | {AR_s:.2f} |",
        f"| Mean chord (mm) | {mean_chord*1000:.0f} | {(STAB_ROOT_CHORD+STAB_TIP_CHORD)*500:.0f} |",
        f"| Chord R/T (mm) | {p['wing_root_chord']*1000:.0f}/{p['wing_tip_chord']*1000:.0f} | {STAB_ROOT_CHORD*1000:.0f} / {STAB_TIP_CHORD*1000:.0f} |",
        "", "---", "",
        "## 5. Pilotability, Stability & Structure", "",
        "| Parameter | Value | Target |", "|:---|:---|:---|",
        f"| **ω_n** (pitch frequency, Hz) | {omega_n:.2f} Hz | freeride [{f_lo:.1f}–{f_hi:.1f}] Hz |",
        f"| Cm_α (pitch stiffness, rad⁻¹) | {pd['Cm_alpha']:.2f} | (<0 = stable) |",
        f"| SM/l_t (scale-invariant) | {pd['SM_lt']*100:.1f}% | typical aviation 10-25% |",
        f"| NP-CG gap (absolute) | {pd['SM_abs']*1000:.1f} mm | — |",
        f"| SM/c̄ | {pd['SM_chord']*100:.1f}% | — (chord-normalized) |",
        f"| CG | {p['cg_ratio']*100:.1f}% c̄ ({X_cg*100:.1f} cm) | [{cfg['cg_range'][0]*100:.0f}–{cfg['cg_range'][1]*100:.0f}]% |",
        f"| Residual moment | {M_total:.3f} N·m | < {M_TOL_TRIM:.1f} N·m (pilot trim authority) |",
        f"| Stab force | {F_stab:.1f} N | (<0 = stable) |",
        (lambda d, t: f"| Stab control authority dF/dα | {d:.1f} N/° | ≥ {t:.0f} (target) |")(
            CL_ALPHA_STAB_PER_DEG * 0.5 * rho * cfg['v_cruise']**2 * stab.area(),
            cfg.get('stab_control_authority_min_N_per_deg', 0.0)),
        f"| Tail volume (info) | {v_h:.3f} | — |",
        f"| Von Mises root (peak ×{LOAD_PEAK_FACTOR:.1f}g) | {sigma_vm/1e6:.1f} MPa | < {SIGMA_ADMISSIBLE/1e6:.0f} MPa (fatigue) |",
        f"| Von Mises root (static 1g) | {sigma_vm_static/1e6:.1f} MPa | < {SIGMA_ULTIMATE/1e6:.0f} MPa (rupture) |",
        f"| σ_v cavitation | {SIGMA_CAV:.2f} | — |",
        "",
    ]

    warn = []
    if not np.isfinite(omega_n):
        warn.append(f"⚠️ UNSTABLE foil (Cm_α > 0) — the pilot cannot hold it at trim")
    elif not (f_lo <= omega_n <= f_hi):
        warn.append(f"⚠️ ω_n {omega_n:.2f} Hz outside freeride target [{f_lo:.1f}–{f_hi:.1f}] Hz")
    if L_to < WEIGHT * 0.98:
        warn.append(f"⚠️ Insufficient takeoff lift: {L_to:.1f} / {WEIGHT:.1f} N")
    if CL_to > CL_MAX_TO:
        warn.append(f"⚠️ CL_to {CL_to:.2f} > CL_max {CL_MAX_TO}")
    if abs(M_total) > M_TOL_TRIM:
        warn.append(f"⚠️ Residual moment {M_total:.2f} N·m > pilot trim authority ±{M_TOL_TRIM:.1f} N·m")
    if sigma_vm > SIGMA_ADMISSIBLE:
        warn.append(f"⚠️ Dynamic peak Von Mises {sigma_vm/1e6:.0f} MPa > σ_fatigue {SIGMA_ADMISSIBLE/1e6:.0f} MPa "
                    f"— ratio {sigma_vm/SIGMA_ADMISSIBLE:.2f}, foil undersized in fatigue")
    if warn:
        lines += ["## ⚠️ Warnings", ""] + [f"- {w}" for w in warn] + [""]

    with open(os.path.join(out_dir, "fiche_technique.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 10. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_starts = phy["search_space"]["N_starts"]
    x_best   = run_multistart(n_starts=N_starts)
    full_report(x_best)