# =================================================================================
# Hydrofoil Optimization V4 — Shape optimization (differentiable, gradient-based)
# ---------------------------------------------------------------------------------
#   3. OPTIMIZER
#        - a single gradient-based `asb.Opti` (IPOPT) with REAL constraints instead
#          of DE + Nelder-Mead + weighted penalties. alpha_cruise is a variable
#          trimmed by the equality constraint L = WEIGHT (no interpolation solver).
#        - MULTI-START: the (expensive) symbolic graph is traced ONCE and re-solved
#          from several Sobol-diversified initial guesses via set_initial — cheap.
#        - AeroBuildup (NeuralFoil sections) is the in-loop model because it is
#          CasADi-differentiable; the final optimum is VERIFIED (not re-frozen) on
#          LiftingLine, which captures the exact wing->stab downwash & induced drag.
#
# Config, exports (.dat / XFLR5 / markdown) and scenario/airfoil conventions are
# kept identical to V3 so the rest of the workspace stays coherent.
# =================================================================================

import os
import re
import sys
import copy
import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import aerosandbox as asb
import aerosandbox.numpy as anp          # dispatches numpy <-> CasADi (symbolic)
import yaml
from scipy.stats import qmc

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from config.water_atmosphere import Water as Atmosphere

# ─────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL CONFIGURATION  (same files / keys as V3)
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
RHO = atmosphere.density()
MU  = atmosphere.dynamic_viscosity()
GRAV = phy.get("hydro", {}).get("gravity", 9.81)

# ── Masses & weight (nominal for cruise trim, design-max for takeoff & structure) ──
_pilot_nom = phy["pilot"]["mass_kg"]
_pilot_max = phy["pilot"].get("mass_design_max_kg", _pilot_nom)
board_mass = phy["board"]["mass_kg"]
rig_mass   = cfg["rig_mass_kg"]
mass       = _pilot_nom + board_mass
WEIGHT     = (mass + rig_mass) * GRAV                       # nominal (cruise trim target)
WEIGHT_MAX = (_pilot_max + board_mass + rig_mass) * GRAV    # heaviest rider (takeoff/struct)

# ── Fixed geometry inherited from YAML ──
sweep_deg          = phy["wing"]["sweep_deg"]
wing_anhedral_deg  = phy["wing"]["anhedral_deg"]
N_WING             = 8                         # spanwise sections (cost/accuracy)
CL_MAX_TO          = phy["wing"]["cl_max_takeoff"]
WING_SKIN_THICK    = phy["wing"]["skin_thickness"]
E_CARBON           = phy["wing"].get("youngs_modulus_gpa", 70.0) * 1e9
TIP_DEFL_MAX       = phy["wing"].get("tip_deflection_max_ratio", 0.15)
fuselage_diameter  = phy["fuselage"]["diameter"]
x_fuselage_start   = phy["fuselage"]["x_start"]
N_FUSE             = phy["fuselage"]["n_sections"]
mast_length        = phy["mast"]["length"]
profondeur_imm     = phy["mast"]["immersion_depth"]      # wing depth below surface
mast_chord_top     = phy["mast"]["chord_top"]
mast_chord_bot     = phy["mast"]["chord_bot"]
mast_profile       = phy["mast"]["profile"]
x_mast             = phy["mast"]["x_position"]
cd_mast            = phy["mast"]["cd"]
N_MAST             = phy["mast"]["n_sections"]
N_STAB             = 5
STAB_SWEEP_DEG     = phy["stab"].get("sweep_deg", 5.0)
STAB_DIHEDRAL_DEG  = phy["stab"].get("dihedral_deg", 3.0)
STAB_FUSE_OFFSET   = phy["stab"].get("fuselage_offset", 0.10)
TIP_RATIO_S        = phy["stab"]["tip_chord_ratio"]
CL_ALPHA_STAB_PER_DEG = phy["stab"].get("cl_alpha_per_deg", 0.10)

# ── Hydro / free-surface ──
HYD = phy.get("hydro", {})
FREE_SURFACE   = bool(HYD.get("free_surface", True))
LIFT_KNOCKDOWN = bool(HYD.get("lift_knockdown", True))
WAVE_DRAG_COEF = float(HYD.get("wave_drag_coeff", 0.5))

# ── Mast: chord is now a design variable, drag & structure from a real model ──
q_cruise   = 0.5 * RHO * cfg["v_cruise"] ** 2
q_takeoff  = 0.5 * RHO * cfg["v_takeoff"] ** 2
MAST_TC        = float(phy["mast"].get("thickness_ratio", 0.12))
MAST_TURN_G    = float(phy["mast"].get("turn_load_g", 1.0))
MAST_ALLOW     = float(phy["mast"].get("allowable_mpa", 300)) * 1e6
MAST_DEFL_MAX  = float(phy["mast"].get("deflection_max_ratio", 0.10))
E_MAST         = float(phy["mast"].get("youngs_modulus_gpa", 110)) * 1e9
_mast_tc_pct   = max(6, min(24, int(round(MAST_TC * 100))))
MAST_AIRFOIL   = asb.Airfoil(f"naca00{_mast_tc_pct:02d}")   # fixed symmetric mast section
# Legacy constants (fallback only; V4 computes mast drag/moment per design & speed):
chord_wl = 0.5 * (mast_chord_bot + mast_chord_top)
D_MAST   = q_cruise * chord_wl * profondeur_imm * cd_mast
M_MAST   = -D_MAST * (profondeur_imm / 2)

# ── Structural / cavitation limits ──
# Amateur, hand-built realism: the allowable stress is the wet-layup allowable
# (manufacturing.carbon_allowable_mpa, ~100 MPa) rather than the aerospace-grade
# fatigue allowable (~160 MPa). This sizes the shell thicker/stronger → buildable.
SIGMA_ULTIMATE   = phy["wing"]["ultimate_stress_mpa"] * 1e6
FATIGUE_RATIO    = phy["wing"]["fatigue_allowable_ratio"]
LOAD_PEAK_FACTOR = phy["wing"]["load_peak_factor"]
MFG = phy.get("manufacturing", {})
SIGMA_ADMISSIBLE = MFG.get("carbon_allowable_mpa", FATIGUE_RATIO * SIGMA_ULTIMATE / 1e6) * 1e6
P_ATM   = atmosphere.pressure()
P_VAPOR = atmosphere.vapor_pressure()
SIGMA_CAV = (P_ATM + RHO * GRAV * profondeur_imm - P_VAPOR) \
          / (0.5 * RHO * cfg["v_cruise"] ** 2)

# ── Stability targets — industry metrics (tail volume), NOT ω_n ──
# V_h band is AUTO-CALIBRATED to the scenario reference geometry (see REF_VH in
# section 3): each discipline's real kit has its own tail volume, there is no
# universal aircraft-like target. Static margin is only floored (foils are very
# pitch-stable; SM/c̄ > 100% is normal) and otherwise reported.
STAB_CFG        = phy.get("stability", {})
VH_TOL          = float(STAB_CFG.get("tail_volume_ref_tolerance", 0.35))
SM_FLOOR        = float(STAB_CFG.get("static_margin_floor", 0.03))
CG_NOM          = float(cfg.get("cg_ratio_init", 0.35))   # nominal rider stance (reference)
PILOT_MASS      = float(phy["pilot"]["mass_kg"])          # weight-shift authority mass
PILOT_WEIGHT    = PILOT_MASS * GRAV

# ── Rider LEVEL (beginner→race): scales trim effort, tip forgiveness, stall margin ──
LEVEL       = phy.get("level", "intermediate")
_lvl        = phy.get("levels", {}).get(LEVEL, {})
STANCE_TRIM_MAX = float(_lvl.get("trim_budget_m", STAB_CFG.get("stance_trim_max_m", 0.04)))
STANCE_FULL     = float(_lvl.get("stance_full_m",  STAB_CFG.get("stance_full_m", 0.08)))
LVL_TIP_CAMBER_MIN = _lvl.get("tip_camber_min", None)     # overrides safety floor if set
LVL_TIP_STALL_MARGIN = _lvl.get("tip_stall_margin_deg", None)

# ── Multi-point speed optimization ──
_OPT          = phy.get("optimization", {})
SPEED_POINTS  = list(_OPT.get("speed_points", [1.0]))
SPEED_WEIGHTS = list(_OPT.get("speed_weights", [1.0]))
TAKEOFF_WEIGHT = float(_OPT.get("takeoff_weight", 0.3))

# ── Pitch inertia of the rider system (used only for the REPORTED ω_n now) ──
_M_TOTAL    = phy["pilot"]["mass_kg"] + phy["board"]["mass_kg"] + rig_mass
_R_GYR      = phy["pilot"].get("gyration_radius_m", 0.30)
I_YY_RIDER  = _M_TOTAL * _R_GYR ** 2
PILOT_FREQ_LO, PILOT_FREQ_HI = cfg["pilotability_freq"]
M_TOL_TRIM  = phy["pilot"].get("trim_moment_tolerance_N_m", 25.0)

DALPHA = 1.0          # deg — finite-difference step for Cm_alpha (short period)

# ─────────────────────────────────────────────────────────────────────────────
# 2. AIRFOIL BASIS — reference Kulfan/CST weights (spanwise morphing)
# ─────────────────────────────────────────────────────────────────────────────
# The wing airfoil per section is generated from the scenario's reference airfoil
# by scaling its CAMBER line and its THICKNESS distribution (both in the Kulfan
# basis). camber_scale=thickness_scale=1 reproduces the reference exactly.
#   upper = camber_scale * mean + thickness_scale * half
#   lower = camber_scale * mean - thickness_scale * half
# where mean=(u+l)/2 (camber modes), half=(u-l)/2 (thickness modes).
WING_AIRFOIL_NAME = cfg["wing_airfoil"]
STAB_AIRFOIL_NAME = phy["stab"]["airfoil"]

def _load_airfoil(name, fallback):
    try:
        af = asb.Airfoil(name)
        if af.coordinates is not None and len(af.coordinates) >= 30:
            return af
    except Exception:
        pass
    print(f"  ⚠ Airfoil '{name}' unavailable in ASB → fallback '{fallback}'")
    return asb.Airfoil(fallback)

_ref_af  = _load_airfoil(WING_AIRFOIL_NAME, "naca2412").to_kulfan_airfoil()
WING_AIRFOIL_NAME = _load_airfoil(WING_AIRFOIL_NAME, "naca2412").name
_UW = np.array(_ref_af.upper_weights, dtype=float)
_LW = np.array(_ref_af.lower_weights, dtype=float)
KULFAN_MEAN = 0.5 * (_UW + _LW)          # camber modes
KULFAN_HALF = 0.5 * (_UW - _LW)          # thickness modes
KULFAN_LE   = float(_ref_af.leading_edge_weight)
KULFAN_TE   = float(_ref_af.TE_thickness)
REF_TC      = float(_load_airfoil(WING_AIRFOIL_NAME, "naca2412").max_thickness())  # ref t/c
STAB_AIRFOIL = _load_airfoil(STAB_AIRFOIL_NAME, "naca0012")

def make_wing_airfoil(camber_scale, thickness_scale, name="wing_sec"):
    """Kulfan airfoil = camber_scale*camber_modes ± thickness_scale*thickness_modes.
    Works with numeric floats OR CasADi symbolics (fed to NeuralFoil)."""
    uw = camber_scale * KULFAN_MEAN + thickness_scale * KULFAN_HALF
    lw = camber_scale * KULFAN_MEAN - thickness_scale * KULFAN_HALF
    return asb.KulfanAirfoil(name=name, lower_weights=lw, upper_weights=uw,
                             leading_edge_weight=KULFAN_LE, TE_thickness=KULFAN_TE)

# Stab morph basis: NACA0012 thickness modes + the wing camber basis for a variable
# (usually negative → download-efficient) camber. camber=0 reproduces NACA0012.
_stab_kf   = STAB_AIRFOIL.to_kulfan_airfoil()
_S_UW, _S_LW = np.array(_stab_kf.upper_weights), np.array(_stab_kf.lower_weights)
STAB_KULFAN_HALF = 0.5 * (_S_UW - _S_LW)
STAB_KULFAN_LE   = float(_stab_kf.leading_edge_weight)
STAB_KULFAN_TE   = float(_stab_kf.TE_thickness)

def make_stab_airfoil(camber_scale, name="stab_sec"):
    """Symmetric NACA0012 thickness + `camber_scale`×wing-camber modes (reflex/download)."""
    uw = camber_scale * KULFAN_MEAN + STAB_KULFAN_HALF
    lw = camber_scale * KULFAN_MEAN - STAB_KULFAN_HALF
    return asb.KulfanAirfoil(name=name, lower_weights=lw, upper_weights=uw,
                             leading_edge_weight=STAB_KULFAN_LE, TE_thickness=STAB_KULFAN_TE)

# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTIMIZATION VECTOR  (16 design vars + alpha_cruise trimmed as a variable)
# ─────────────────────────────────────────────────────────────────────────────
# Only shape/trim DOF that move the objective significantly are kept.
# CG is NOT a variable: the rider is an active controller who trims by shifting
# weight. We evaluate at the nominal stance CG_NOM and constrain the physical
# weight-shift needed to trim (see compute_metrics / _build_opti). The stab is
# sized by the tail-volume band, not by sliding a fictitious CG.
VAR_NAMES = [
    "wing_span", "wing_root_chord", "taper", "chord_shape",     # planform  (4)
    "wing_setting_angle", "tip_washout",                        # twist     (2)
    "camber_root", "thick_root", "camber_tip", "thick_tip",     # airfoils  (4)
    "stab_span", "stab_root_chord", "s_twist", "stab_camber",   # stab      (4)
    "mast_chord",                                               # mast      (1)
    "fuselage_length", "alpha_to",                              # system    (2)
]
N_VAR = len(VAR_NAMES)

_wing_span_bounds = tuple(cfg.get("wing_span_bounds", phy["wing"].get("span_bounds", [0.75, 1.30])))
_wing_root_bounds = tuple(cfg.get("wing_root_chord_bounds", phy["wing"].get("root_chord_bounds", [0.10, 0.30])))

# Thickness bounds are set from ABSOLUTE t/c manufacturing targets, converted to
# multipliers of the reference airfoil's t/c (REF_TC). This is scenario-robust:
# a thin laminar ref (e387) gets thickened to a hand-buildable t/c, a thick ref
# (naca4412) is only mildly scaled.
_min_tc_root = float(MFG.get("min_tc_root", 0.11))
_min_tc_tip  = float(MFG.get("min_tc_tip", 0.085))
_max_tc_root = float(MFG.get("max_tc_root", 0.16))
_thick_root_bounds = (_min_tc_root / REF_TC, _max_tc_root / REF_TC)
_thick_tip_bounds  = (_min_tc_tip / REF_TC, (_max_tc_root - 0.01) / REF_TC)
# Taper floor from the min absolute tip chord (buildable, high-enough tip Re).
_min_tip_chord = float(MFG.get("min_tip_chord_m", 0.045))
_taper_lo = max(0.20, _min_tip_chord / _wing_root_bounds[1])

# Handling-safety bounds (forgiving tip). The tip-camber floor is set by the rider
# LEVEL if given (beginner = more camber = gentler), else the global safety default.
SAFETY          = phy.get("safety", {})
_tip_camber_min = float(LVL_TIP_CAMBER_MIN if LVL_TIP_CAMBER_MIN is not None
                        else SAFETY.get("tip_camber_min_ratio", 0.50))
TIP_STALL_MARGIN = float(LVL_TIP_STALL_MARGIN if LVL_TIP_STALL_MARGIN is not None
                         else SAFETY.get("tip_stall_margin_deg", 4.0))
_max_washin     = float(SAFETY.get("max_washin_deg", 0.0))
_tw_lo, _tw_hi  = tuple(phy["wing"].get("twist_bounds", [-6.0, 1.0]))
_tip_washout_bounds = (_tw_lo, min(_tw_hi, _max_washin))    # forbid wash-in (upper ≤ 0)
_mast_chord_bounds  = tuple(phy["mast"].get("chord_bounds", [0.075, 0.145]))
_stab_camber_bounds = tuple(phy["stab"].get("camber_bounds", [-0.8, 0.1]))

BOUNDS = {
    "wing_span":          _wing_span_bounds,
    "wing_root_chord":    _wing_root_bounds,
    "taper":              (_taper_lo, 0.45),
    "chord_shape":        (0.0, 1.0),                       # 0 linear · 1 elliptic
    "wing_setting_angle": tuple(phy["wing"].get("calage_bounds", [-3.0, 5.0])),
    "tip_washout":        _tip_washout_bounds,              # no wash-in → root stalls first
    "camber_root":        (0.2, 1.8),                       # ×ref camber (keep some)
    "thick_root":         _thick_root_bounds,               # → min_tc_root .. max_tc_root
    "camber_tip":         (_tip_camber_min, 1.5),           # keep tip camber → gentle stall
    "thick_tip":          _thick_tip_bounds,                # → min_tc_tip ..
    "stab_span":          tuple(phy["stab"]["span_bounds"]),
    "stab_root_chord":    tuple(phy["stab"]["root_chord_bounds"]),
    "s_twist":            tuple(phy["stab"]["twist_bounds"]),
    "stab_camber":        _stab_camber_bounds,              # download-efficient reflex
    "mast_chord":         _mast_chord_bounds,               # drag vs bending/stiffness
    "fuselage_length":    tuple(cfg["fuselage_length_bounds"]),
    "alpha_to":           tuple(phy["alpha"]["bounds"]),
}
ALPHA_CRUISE_BOUNDS = (-2.0, 10.0)

# Warm start on the scenario's industrial reference geometry.
WING_SPAN       = cfg.get("wing_span_init",       phy["wing"]["span_init"])
WING_ROOT_CHORD = cfg.get("wing_root_chord_init", phy["wing"]["root_chord_init"])
STAB_SPAN       = cfg["stab_span"]
STAB_ROOT_CHORD = cfg["stab_root_chord"]

INIT = {
    "wing_span":          WING_SPAN,
    "wing_root_chord":    WING_ROOT_CHORD,
    "taper":              phy["wing"].get("tip_chord_ratio", 0.25),
    "chord_shape":        0.7,
    "wing_setting_angle": 1.0,
    "tip_washout":        -2.0,
    "camber_root":        1.0,
    "thick_root":         1.0,
    "camber_tip":         0.6,
    "thick_tip":          0.8,
    "stab_span":          STAB_SPAN,
    "stab_root_chord":    STAB_ROOT_CHORD,
    "s_twist":            -0.7,   # small decalage → stab download near target (feasible start)
    "stab_camber":        -0.2,   # mild reflex for download efficiency
    "mast_chord":         float(phy["mast"].get("chord_init", 0.120)),
    "fuselage_length":    cfg.get("fuselage_length_init",
                                  0.5 * (BOUNDS["fuselage_length"][0] + BOUNDS["fuselage_length"][1])),
    "alpha_to":           7.0,
}

def _clip_init(d):
    return {k: float(np.clip(v, BOUNDS[k][0], BOUNDS[k][1])) for k, v in d.items()}
INIT = _clip_init(INIT)

# Wing→stab downwash angle (reported for information; the exact downwash is
# captured by the final LiftingLine verification). It is NOT baked into the stab
# geometry any more — doing so double-counted and inflated the stab download,
# pushing the warm start deep into the infeasible region.
_S_ref_nom  = np.pi / 4 * WING_SPAN * WING_ROOT_CHORD * (1 + INIT["taper"]) / 2
_CL_nom     = WEIGHT / (q_cruise * max(_S_ref_nom, 1e-3))
_AR_nom     = WING_SPAN ** 2 / max(_S_ref_nom, 1e-3)
DOWNWASH_DEG = float(np.degrees(2.0 * _CL_nom / (np.pi * max(_AR_nom, 3.0))))

# ── Reference tail volume V_h → auto-calibrated V_h band (per discipline) ──
def _elliptic_area(span, root, tip):
    r = np.linspace(0, 1, 25)
    c = tip + (root - tip) * np.sqrt(np.clip(1 - r ** 2, 0, 1))
    return 2.0 * float(np.trapz(c, r * span / 2))

_Sw_ref   = _elliptic_area(WING_SPAN, WING_ROOT_CHORD, INIT["taper"] * WING_ROOT_CHORD)
_cbar_ref = _Sw_ref / WING_SPAN
_Ss_ref   = _elliptic_area(STAB_SPAN, STAB_ROOT_CHORD, TIP_RATIO_S * STAB_ROOT_CHORD)
_cs_ref   = 0.5 * (STAB_ROOT_CHORD + TIP_RATIO_S * STAB_ROOT_CHORD)
_lt_ref   = (x_fuselage_start + INIT["fuselage_length"] - STAB_FUSE_OFFSET
             + 0.25 * _cs_ref) - 0.25 * _cbar_ref
REF_VH    = _Ss_ref * _lt_ref / (_Sw_ref * _cbar_ref)
VH_RANGE  = ((1.0 - VH_TOL) * REF_VH, (1.0 + VH_TOL) * REF_VH)

# ─────────────────────────────────────────────────────────────────────────────
# 4. GEOMETRY  (differentiable — numeric floats OR CasADi symbolics)
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_POWER_W = 1.5      # TE curvature (kept from V3; TE monotonic by construction)
SWEEP_POWER_S = 1.5

def _wing_chords_twists(p):
    """Return spanwise (r, chord, twist) lists for the wing (works symbolic)."""
    root = p["wing_root_chord"]
    tip  = p["taper"] * root
    s    = p["chord_shape"]
    rs, chords, twists = [], [], []
    for i in range(N_WING):
        r = i / (N_WING - 1)
        ell = np.sqrt(max(1.0 - r ** 2, 0.0))     # r is a constant → np ok
        lin = 1.0 - r
        shape = s * ell + (1.0 - s) * lin
        c = tip + (root - tip) * shape
        tw = p["wing_setting_angle"] + p["tip_washout"] * (r ** SWEEP_POWER_W)
        rs.append(r); chords.append(c); twists.append(tw)
    return rs, chords, twists

def _planform_area_mean(rs, chords, span):
    """Trapezoidal area (full, symmetric) and mean chord — symbolic-safe."""
    dy = (span / 2.0) / (N_WING - 1)
    area_semi = 0.0
    for i in range(N_WING - 1):
        area_semi = area_semi + 0.5 * (chords[i] + chords[i + 1]) * dy
    area_full = 2.0 * area_semi
    mean_chord = area_full / span
    return area_full, mean_chord

def build_airplane(p):
    """Assemble wing + fuselage + mast + stab. Returns airplane and key metrics.
    Airfoils are morphed Kulfan (wing) — symbolic-compatible."""
    span_w = p["wing_span"]
    rs, chords, twists = _wing_chords_twists(p)
    S_wing, mean_chord = _planform_area_mean(rs, chords, span_w)
    K_sweep_w = (span_w / 2) * anp.tan(anp.radians(sweep_deg))

    af_root = make_wing_airfoil(p["camber_root"], p["thick_root"], "wing_root")
    af_tip  = make_wing_airfoil(p["camber_tip"],  p["thick_tip"],  "wing_tip")

    wing_xsecs = []
    for i, r in enumerate(rs):
        c = chords[i]
        x_te = p["wing_root_chord"] + (r ** SWEEP_POWER_W) * K_sweep_w
        x_le = x_te - c
        z    = -((r ** 2) * span_w / 2) * anp.tan(anp.radians(wing_anhedral_deg))
        # Morphed section airfoil: blend root/tip Kulfan weights linearly in r.
        uw = (p["camber_root"] * KULFAN_MEAN + p["thick_root"] * KULFAN_HALF) * (1 - r) \
           + (p["camber_tip"]  * KULFAN_MEAN + p["thick_tip"]  * KULFAN_HALF) * r
        lw = (p["camber_root"] * KULFAN_MEAN - p["thick_root"] * KULFAN_HALF) * (1 - r) \
           + (p["camber_tip"]  * KULFAN_MEAN - p["thick_tip"]  * KULFAN_HALF) * r
        af = asb.KulfanAirfoil(name=f"wing_sec_{i}", lower_weights=lw, upper_weights=uw,
                               leading_edge_weight=KULFAN_LE, TE_thickness=KULFAN_TE)
        wing_xsecs.append(asb.WingXSec(xyz_le=[x_le, r * span_w / 2, z],
                                       chord=c, twist=twists[i], airfoil=af))
    wing = asb.Wing(symmetric=True, name="MainWing", xsecs=wing_xsecs)

    # CG = rider stance (fixed nominal), not a design variable.
    cg_ratio = p.get("cg_ratio", CG_NOM)
    X_cg = cg_ratio * mean_chord

    # ── Fuselage (same shape law as V3) ──
    fuse_xsecs = []
    for i in range(N_FUSE):
        xi_rel = i / (N_FUSE - 1)
        xi     = x_fuselage_start + xi_rel * p["fuselage_length"]
        width  = (0.001 if i == 0
                  else fuselage_diameter * (i / 3) if i < 3
                  else fuselage_diameter * (1 - 0.5 * xi_rel ** 3))
        fuse_xsecs.append(asb.FuselageXSec(xyz_c=[xi, 0, 0], radius=width))
    fuselage_obj = asb.Fuselage(name="Fuselage", xsecs=fuse_xsecs)

    # ── Mast (constant chord = design variable) ──
    mast_c = p.get("mast_chord", INIT["mast_chord"])
    mast_xsecs = []
    for i in range(N_MAST):
        r = i / (N_MAST - 1)
        mast_xsecs.append(asb.WingXSec(
            xyz_le=[x_mast, 0, -r * profondeur_imm],
            chord=mast_c, twist=0, airfoil=MAST_AIRFOIL))
    mast_obj = asb.Wing(name="Mast", symmetric=False, xsecs=mast_xsecs)

    # ── Stabilizer (morphed camber for download efficiency) ──
    stab_af     = make_stab_airfoil(p.get("stab_camber", -0.2), "stab_sec")
    stab_span_p = p["stab_span"]
    stab_root_p = p["stab_root_chord"]
    stab_tip_p  = TIP_RATIO_S * stab_root_p
    x_stab_root = x_fuselage_start + p["fuselage_length"] - STAB_FUSE_OFFSET
    K_sweep_s   = (stab_span_p / 2) * anp.tan(anp.radians(STAB_SWEEP_DEG))
    stab_xsecs, stab_chords = [], []
    for i in range(N_STAB):
        r = i / (N_STAB - 1)
        ell = np.sqrt(max(1 - r ** 2, 0.0))
        c_s = stab_tip_p + (stab_root_p - stab_tip_p) * ell
        x_te_s = stab_root_p + (r ** SWEEP_POWER_S) * K_sweep_s
        x_le_s = x_te_s - c_s
        z_s = (r ** 1.5 * stab_span_p / 2) * anp.tan(anp.radians(STAB_DIHEDRAL_DEG))
        stab_xsecs.append(asb.WingXSec(
            xyz_le=[x_stab_root + x_le_s, r * stab_span_p / 2, z_s],
            chord=c_s, twist=p["s_twist"], airfoil=stab_af))
        stab_chords.append(c_s)
    stab = asb.Wing(symmetric=True, name="Stab", xsecs=stab_xsecs)
    dy_s = (stab_span_p / 2) / (N_STAB - 1)
    S_stab = 2.0 * sum(0.5 * (stab_chords[i] + stab_chords[i + 1]) * dy_s
                       for i in range(N_STAB - 1))

    airplane = asb.Airplane(
        wings=[wing, stab], fuselages=[fuselage_obj],
        xyz_ref=[X_cg, 0.0, 0.0],
        s_ref=S_wing, c_ref=mean_chord, b_ref=span_w)

    geo = dict(S_wing=S_wing, mean_chord=mean_chord, chords=chords, rs=rs,
               S_stab=S_stab, X_cg=X_cg, mast_obj=mast_obj, fuselage_obj=fuselage_obj,
               wing=wing, stab=stab)
    return airplane, geo

# ─────────────────────────────────────────────────────────────────────────────
# 5. HYDRO CORRECTIONS  (free surface) — differentiable
# ─────────────────────────────────────────────────────────────────────────────
def free_surface_lift_factor(mean_chord):
    """Image-vortex lift knockdown near the free surface (pressure-release image
    of opposite sign at 2h). Δα_img = Γ/(4π h U) → ΔCL/CL ≈ c/(4h) for a thin
    section. Depth h = wing immersion. Returns a factor in [~0.7, 1]."""
    if not (FREE_SURFACE and LIFT_KNOCKDOWN):
        return 1.0
    h = max(profondeur_imm, 1e-3)
    return anp.maximum(0.7, 1.0 - mean_chord / (4.0 * h))

def wave_drag(CL, mean_chord, span, velocity):
    """Linearised wave drag of a submerged lifting foil (Froude). Per-span
    submerged-vortex result dDw/dy = C·ρ·Γ²·(g/U²)·exp(-2gh/U²), Γ=½ U c CL,
    integrated over span. Decays exponentially with depth·g/U² (deeper/faster →
    negligible). C = WAVE_DRAG_COEF folds the O(1) prefactor (calibrate on data)."""
    if not FREE_SURFACE:
        return 0.0
    h  = max(profondeur_imm, 1e-3)
    k0 = GRAV / velocity ** 2
    gamma = 0.5 * velocity * mean_chord * CL
    return WAVE_DRAG_COEF * RHO * gamma ** 2 * k0 * anp.exp(-2.0 * k0 * h) * span

# ─────────────────────────────────────────────────────────────────────────────
# 6. STRUCTURE  (improved beam) — differentiable
# ─────────────────────────────────────────────────────────────────────────────
def structure_root(chord_root, span, thick_root_rel, load_factor):
    """Root von Mises + tip deflection. Sized for the HEAVIEST design rider
    (WEIGHT_MAX) so the foil works across a weight range. Elliptic spanwise load →
    root bending moment at the load centroid 4s/3π. Elliptical carbon shell."""
    weight_eff = WEIGHT_MAX * load_factor
    s = span / 2.0
    t_max = thick_root_rel * chord_root
    a, b = chord_root / 2.0, t_max / 2.0
    t_skin = anp.minimum(WING_SKIN_THICK, 0.9 * b)
    a_in, b_in = a - t_skin, b - t_skin
    I_xx = (np.pi / 4.0) * (a * b ** 3 - a_in * b_in ** 3)
    A_enc = np.pi * a * b

    M_flex = (weight_eff / 2.0) * (4.0 * s / (3.0 * np.pi))       # elliptic centroid
    M_tors = (weight_eff / 2.0) * (0.05 * chord_root) * (s / 2.0)
    sigma_flex = M_flex * b / (I_xx + 1e-12)
    tau = M_tors / (2.0 * A_enc * t_skin + 1e-12)
    sigma_vm = anp.sqrt(sigma_flex ** 2 + 3 * tau ** 2)

    # Static tip deflection (cantilever, uniform-EI approximation) at 1 g, heaviest.
    M_flex_1g = (WEIGHT_MAX / 2.0) * (4.0 * s / (3.0 * np.pi))
    tip_defl = M_flex_1g * s ** 2 / (2.0 * E_CARBON * I_xx + 1e-12)
    return sigma_vm, tip_defl / s


def mast_drag_structure(mast_chord, velocity):
    """Mast profile drag (from the real section) + a cantilever beam check.
    Drag: CD of the fixed symmetric section from NeuralFoil at α=0, Re(V) → the
    optimizer wants a thin chord. Structure: a lateral maneuver load F=turn_g×W_max
    at the foil bends the mast; stress & tip deflection floor the chord so it can't
    become a wafer. Returns (D_mast, sigma_mast, defl_ratio)."""
    Re_m = RHO * velocity * mast_chord / MU
    cd0  = MAST_AIRFOIL.get_aero_from_neuralfoil(alpha=0.0, Re=Re_m, model_size="xxsmall")["CD"]
    q    = 0.5 * RHO * velocity ** 2
    D_mast = q * mast_chord * profondeur_imm * cd0            # profile drag over immersed span

    # Beam: solid-ish elliptical section (a=chord/2, b=t/2), lateral load at foil.
    a  = mast_chord / 2.0
    b  = MAST_TC * mast_chord / 2.0
    I  = (np.pi / 4.0) * a * b ** 3
    F_lat = MAST_TURN_G * WEIGHT_MAX
    M     = F_lat * mast_length                              # moment at board joint
    sigma = M * b / (I + 1e-12)
    defl  = F_lat * mast_length ** 3 / (3.0 * E_MAST * I + 1e-12)
    return D_mast, sigma, defl / mast_length

def added_pitch_inertia(chords, mean_chord, span, cg_ratio):
    """Hydrodynamic added inertia contribution to I_yy (2D added mass of the wing
    strips). Heave added mass ρπc²/4 acts through the fore/aft CG offset; plus the
    sectional pitch added inertia ρπc⁴/128. Small but physically correct: it
    lowers omega_n. Stab neglected (small chord)."""
    c2 = sum(c ** 2 for c in chords) / len(chords)
    c4 = sum(c ** 4 for c in chords) / len(chords)
    m_heave = RHO * np.pi / 4.0 * c2 * span            # ∫ ρπc²/4 dy over full span
    I_pitch = RHO * np.pi / 128.0 * c4 * span
    d_x = anp.fabs(cg_ratio - 0.25) * mean_chord       # CG to wing AC offset
    return m_heave * d_x ** 2 + I_pitch

# ─────────────────────────────────────────────────────────────────────────────
# 7. AERO + METRICS  (one call → everything; numeric or symbolic)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(p, alpha_cruise, model_size="xsmall"):
    """Build the airplane and evaluate every quantity needed for the objective and
    the constraints, at cruise (α trimmed), cruise+Δα (for Cm_α), and takeoff.
    Returns a dict of (possibly symbolic) scalars. Same code path numeric/symbolic."""
    airplane, geo = build_airplane(p)
    S_wing, mean_chord = geo["S_wing"], geo["mean_chord"]
    S_stab, X_cg       = geo["S_stab"], geo["X_cg"]

    # --- AeroBuildup at the three operating points (NeuralFoil sections) ---
    op_c  = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_cruise, atmosphere=atmosphere)
    op_c2 = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_cruise + DALPHA, atmosphere=atmosphere)
    op_to = asb.OperatingPoint(velocity=cfg["v_takeoff"], alpha=p["alpha_to"], atmosphere=atmosphere)
    ab_c  = asb.AeroBuildup(airplane, op_c ).run()
    ab_c2 = asb.AeroBuildup(airplane, op_c2).run()
    ab_to = asb.AeroBuildup(airplane, op_to).run()

    # --- Free-surface lift knockdown (affects trim & lift constraints) ---
    kL   = free_surface_lift_factor(mean_chord)
    L    = kL * ab_c["L"];   CL = ab_c["CL"];   D = ab_c["D"];   Cm = ab_c["Cm"]
    L_to = kL * ab_to["L"];  CL_to = ab_to["CL"];  D_to = ab_to["D"]

    # --- Drag: aero + mast (real model) + free-surface wave drag ---
    D_wave  = wave_drag(CL, mean_chord, p["wing_span"], cfg["v_cruise"])
    D_mast_c, mast_sigma, mast_defl = mast_drag_structure(p.get("mast_chord", INIT["mast_chord"]),
                                                          cfg["v_cruise"])
    D_total = D + D_mast_c + D_wave

    # --- Pitch stiffness (finite diff over the two cruise points) ---
    dCm    = ab_c2["Cm"] - ab_c["Cm"]
    dCL    = ab_c2["CL"] - ab_c["CL"]
    dalpha = anp.radians(DALPHA)
    Cm_alpha = dCm / dalpha
    SM_chord = -dCm / (dCL + 1e-9)

    # --- Trim moment about the nominal stance CG (wing + mast + rig) ---
    cg_ratio = p.get("cg_ratio", CG_NOM)
    M_mast_moment = -D_mast_c * (profondeur_imm / 2.0)       # mast drag pitching moment
    M_wing  = Cm * q_cruise * S_wing * mean_chord
    M_rig   = rig_mass * GRAV * (-(X_cg - x_mast))
    M_total = M_wing + M_mast_moment + M_rig

    # --- Active-pilot trim: the rider nulls M_total by shifting weight fore/aft.
    #   shift_to_trim = M_total / pilot_weight   [m of weight shift]
    # Bounded effort (|shift| ≤ STANCE_TRIM_MAX) and control reserve
    #   authority_reserve = pilot_weight × (STANCE_FULL − |shift|)   [N·m]
    # replace the old aviation-style "residual moment ≤ ±25 N·m" proxy.
    shift_trim_m = M_total / PILOT_WEIGHT
    authority_reserve = PILOT_WEIGHT * (STANCE_FULL - anp.fabs(shift_trim_m))

    # --- Stab force (per-surface decomposition) ---
    try:
        F_stab = ab_c["wing_aero_components"][1].L * kL
    except Exception:
        F_stab = 0.0

    # --- Static stability: tail volume + neutral point (industry sizing) ---
    # Tail arm l_t = stab AC (¼ c_stab) to wing AC (¼ c̄).
    c_stab_mean = 0.5 * (p["stab_root_chord"] + TIP_RATIO_S * p["stab_root_chord"])
    l_t   = (x_fuselage_start + p["fuselage_length"] - STAB_FUSE_OFFSET
             + 0.25 * c_stab_mean) - 0.25 * mean_chord
    V_h   = (S_stab * l_t) / (S_wing * mean_chord)
    # Neutral point behind CG = SM·c̄. x_NP measured from wing LE datum.
    SM_abs = SM_chord * mean_chord
    x_NP   = X_cg + SM_abs

    # --- Ride-height (heave) stability from the free-surface model ---
    # dL/dh > 0 (lift grows with depth) ⇒ a rise loses lift ⇒ self-correcting.
    # kL = 1 - c/(4h) ⇒ dkL/dh = c/(4h²); report the restoring stiffness per cm.
    h_dep = max(profondeur_imm, 1e-3)
    dL_dh = (ab_c["L"]) * (mean_chord / (4.0 * h_dep ** 2)) if (FREE_SURFACE and LIFT_KNOCKDOWN) else 0.0
    heave_stiff_N_per_cm = dL_dh * 0.01

    # --- omega_n with hydrodynamic added mass (REPORTED only, not a constraint) ---
    I_added = added_pitch_inertia(geo["chords"], mean_chord, p["wing_span"], cg_ratio)
    I_yy    = I_YY_RIDER + I_added
    stiffness  = -Cm_alpha * q_cruise * S_wing * mean_chord
    omega_n_sq = stiffness / I_yy

    # --- Cavitation: real Cpmin of ROOT and TIP sections at the fast point ---
    # Evaluated at the highest-speed operating point (cruise here) and at each
    # section's own local AoA & Reynolds. Cavitation number σ is lowest at high
    # speed, so this is where -Cpmin ≥ -σ bites (windsurf/downwind race foils).
    tip_chord = p["taper"] * p["wing_root_chord"]
    Re_root = RHO * cfg["v_cruise"] * p["wing_root_chord"] / MU
    Re_tip  = RHO * cfg["v_cruise"] * tip_chord / MU
    af_root = make_wing_airfoil(p["camber_root"], p["thick_root"], "cav_root")
    af_tip  = make_wing_airfoil(p["camber_tip"],  p["thick_tip"],  "cav_tip")
    alpha_root = alpha_cruise + p["wing_setting_angle"]
    alpha_tip  = alpha_cruise + p["wing_setting_angle"] + p["tip_washout"]   # r=1 twist
    Cpmin      = af_root.get_aero_from_neuralfoil(alpha=alpha_root, Re=Re_root, model_size=model_size)["Cpmin"]
    Cpmin_tip  = af_tip.get_aero_from_neuralfoil(alpha=alpha_tip, Re=Re_tip, model_size=model_size)["Cpmin"]

    # --- Structure ---
    sigma_vm, tip_defl_ratio = structure_root(
        p["wing_root_chord"], p["wing_span"], REF_TC * p["thick_root"], LOAD_PEAK_FACTOR)

    AR_wing = p["wing_span"] ** 2 / S_wing
    AR_stab = p["stab_span"] ** 2 / S_stab
    dFda_stab = CL_ALPHA_STAB_PER_DEG * q_cruise * S_stab

    return dict(
        L=L, D=D, D_total=D_total, D_wave=D_wave, Cm=Cm, CL=CL,
        L_to=L_to, CL_to=CL_to, D_to=D_to, kL=kL,
        Cm_alpha=Cm_alpha, SM_chord=SM_chord, omega_n_sq=omega_n_sq, stiffness=stiffness,
        V_h=V_h, l_t=l_t, x_NP=x_NP, SM_abs=SM_abs, heave_stiff=heave_stiff_N_per_cm,
        M_total=M_total, shift_trim_m=shift_trim_m, authority_reserve=authority_reserve,
        F_stab=F_stab, Cpmin=Cpmin, Cpmin_tip=Cpmin_tip, tip_chord=tip_chord, sigma_vm=sigma_vm,
        D_mast=D_mast_c, mast_sigma=mast_sigma, mast_defl=mast_defl,
        tip_defl_ratio=tip_defl_ratio, S_wing=S_wing, S_stab=S_stab, mean_chord=mean_chord,
        AR_wing=AR_wing, AR_stab=AR_stab, dFda_stab=dFda_stab, X_cg=X_cg,
        I_yy=I_yy, airplane=airplane, geo=geo,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 8. OPTIMIZATION — gradient-based Opti (IPOPT) with REAL constraints, multi-start
# ─────────────────────────────────────────────────────────────────────────────
class C:
    _ON = sys.stdout.isatty()
    @classmethod
    def _w(cls, code, t): return f"\033[{code}m{t}\033[0m" if cls._ON else t
    @classmethod
    def head(cls, t): return cls._w("1;36", t)
    @classmethod
    def sec(cls, t):  return cls._w("36", t)
    @classmethod
    def ok(cls, t):   return cls._w("32", t)
    @classmethod
    def warn(cls, t): return cls._w("33", t)
    @classmethod
    def dim(cls, t):  return cls._w("2", t)
    @classmethod
    def bold(cls, t): return cls._w("1", t)


def _build_opti(model_size="xsmall"):
    """Trace the symbolic NLP ONCE. Returns (opti, var_dict, alpha_cruise, obj, cons)."""
    opti = asb.Opti()
    v = {}
    for name in VAR_NAMES:
        lo, hi = BOUNDS[name]
        v[name] = opti.variable(init_guess=INIT[name], lower_bound=lo, upper_bound=hi)
    alpha_cruise = opti.variable(init_guess=2.0,
                                 lower_bound=ALPHA_CRUISE_BOUNDS[0],
                                 upper_bound=ALPHA_CRUISE_BOUNDS[1])

    m = compute_metrics(v, alpha_cruise, model_size=model_size)

    # ── REAL constraints (no weighted penalties) ──
    opti.subject_to(m["L"] == WEIGHT)                        # cruise trim (α_cruise, nominal)
    opti.subject_to(m["L_to"] >= WEIGHT_MAX)                 # takeoff lift for HEAVIEST rider
    opti.subject_to(m["CL_to"] <= cfg["takeoff_cl_margin"] * CL_MAX_TO)
    opti.subject_to(alpha_cruise >= -1.0)                    # healthy regime
    # Active-pilot trim: the rider holds trim with a bounded fore/aft weight shift,
    # keeping a control reserve. Replaces the aviation-style residual-moment band.
    opti.subject_to(m["shift_trim_m"] <=  STANCE_TRIM_MAX)
    opti.subject_to(m["shift_trim_m"] >= -STANCE_TRIM_MAX)
    # Static stability — industry metric: tail volume V_h, auto-calibrated to the
    # scenario reference kit; static margin only needs a positive floor (foils are
    # inherently very pitch-stable, and the rider actively controls).
    opti.subject_to(m["V_h"] >= VH_RANGE[0])
    opti.subject_to(m["V_h"] <= VH_RANGE[1])
    opti.subject_to(m["SM_chord"] >= SM_FLOOR)
    sl_lo, sl_hi = cfg["stab_load_range"]
    opti.subject_to(m["F_stab"] >= sl_lo)
    opti.subject_to(m["F_stab"] <= sl_hi)
    opti.subject_to(m["Cpmin"] >= -SIGMA_CAV)                # cavitation root (fast pt)
    opti.subject_to(m["Cpmin_tip"] >= -SIGMA_CAV)            # cavitation tip  (fast pt)
    opti.subject_to(m["sigma_vm"] <= SIGMA_ADMISSIBLE)       # wing fatigue (heaviest)
    opti.subject_to(m["tip_defl_ratio"] <= TIP_DEFL_MAX)     # wing stiffness / aeroelastic
    opti.subject_to(m["tip_chord"] >= _min_tip_chord)        # tip Re + buildable/robust
    opti.subject_to(m["mast_sigma"] <= MAST_ALLOW)           # mast bending strength
    opti.subject_to(m["mast_defl"]  <= MAST_DEFL_MAX)        # mast lateral stiffness
    ar_lo, ar_hi = tuple(cfg.get("stab_aspect_ratio_range",
                                 phy["stab"].get("aspect_ratio_range", [5.0, 12.0])))
    opti.subject_to(m["AR_stab"] >= ar_lo)
    opti.subject_to(m["AR_stab"] <= ar_hi)
    ar_wing_max = cfg.get("wing_aspect_ratio_max", None)
    if ar_wing_max is not None:
        opti.subject_to(m["AR_wing"] <= ar_wing_max)
    dFda_target = cfg.get("stab_control_authority_min_N_per_deg", 0.0)
    if dFda_target > 0:
        opti.subject_to(m["dFda_stab"] >= dFda_target)

    # ── Objective: MULTI-POINT trimmed drag across speeds + weighted takeoff ──
    # Point 0 (= v_cruise) is m["D_total"]; extra speed points get their own trim α.
    obj = SPEED_WEIGHTS[0] * m["D_total"] + TAKEOFF_WEIGHT * m["D_to"]
    airplane = m["airplane"]
    extra_alphas = []
    for frac, w in zip(SPEED_POINTS[1:], SPEED_WEIGHTS[1:]):
        V = cfg["v_cruise"] * frac
        a_s = opti.variable(init_guess=0.0, lower_bound=-3.0, upper_bound=10.0)
        extra_alphas.append(a_s)
        ab_s = asb.AeroBuildup(airplane, asb.OperatingPoint(velocity=V, alpha=a_s,
                                                            atmosphere=atmosphere)).run()
        opti.subject_to(m["kL"] * ab_s["L"] == WEIGHT)
        D_mast_s, _, _ = mast_drag_structure(v["mast_chord"], V)
        D_s = ab_s["D"] + D_mast_s + wave_drag(ab_s["CL"], m["mean_chord"], v["wing_span"], V)
        obj = obj + w * D_s
    opti.minimize(obj)
    return opti, v, alpha_cruise, obj, m


def run_optimization(n_starts=6, model_size="xsmall"):
    HBAR = "═" * 70
    print()
    print(C.head(HBAR))
    print(C.head(f"  HYDROFOIL OPTIMIZATION V4  ·  {CASE.upper()} freeride  (gradient / IPOPT)"))
    print(f"  {C.dim('Wing')}  {WING_AIRFOIL_NAME:<10} morphable (camber×thick, root→tip)   "
          f"warm-start {WING_SPAN*100:.0f} cm / {WING_ROOT_CHORD*1000:.0f} mm")
    print(f"  {C.dim('Model')} AeroBuildup+NeuralFoil({model_size}) · mast beam+drag · "
          f"stab morph · multi-point {SPEED_POINTS}×Vc")
    print(f"  {C.dim('Level')} {LEVEL}  (trim budget ±{STANCE_TRIM_MAX*100:.0f} cm, tip camber ≥{_tip_camber_min:.2f})   "
          f"riders {WEIGHT/GRAV:.0f}–{WEIGHT_MAX/GRAV:.0f} kg")
    print(f"  {C.dim(f'{N_VAR} vars + α_cruise · IPOPT · {n_starts} starts')}")
    print(C.head(HBAR))

    print(f"\n  {C.dim('Tracing symbolic NLP (once)…')}", flush=True)
    opti, v, alpha_cruise, obj, m = _build_opti(model_size=model_size)

    # Sobol-diversified starts (start 0 = industrial warm start).
    sampler = qmc.Sobol(d=N_VAR, scramble=True, seed=42)
    lo = np.array([BOUNDS[n][0] for n in VAR_NAMES])
    hi = np.array([BOUNDS[n][1] for n in VAR_NAMES])
    sob = qmc.scale(sampler.random(max(n_starts - 1, 1)), lo, hi)

    best = {"obj": np.inf, "x": None, "alpha": None, "feasible": False}
    for k in range(n_starts):
        if k == 0:
            guess = np.array([INIT[n] for n in VAR_NAMES]); a0 = 2.0
        else:
            guess = sob[k - 1]; a0 = 3.0
        for n, val in zip(VAR_NAMES, guess):
            opti.set_initial(v[n], float(val))
        opti.set_initial(alpha_cruise, a0)

        try:
            sol = opti.solve(max_iter=200, behavior_on_failure="return_last", verbose=False)
            x   = {n: float(sol.value(v[n])) for n in VAR_NAMES}
            ac  = float(sol.value(alpha_cruise))
            ok, viol, obj_num = _feasibility(x, ac)
            tag = C.ok("feasible") if ok else C.warn(f"viol {viol:.2f}")
            better = (ok and (not best["feasible"] or obj_num < best["obj"])) or \
                     (not best["feasible"] and not ok and obj_num < best["obj"])
            mark = ""
            if (ok and (not best["feasible"] or obj_num < best["obj"])) or \
               (not best["feasible"] and (best["x"] is None or viol < best.get("viol", np.inf))):
                best = {"obj": obj_num, "x": x, "alpha": ac, "feasible": ok, "viol": viol}
                mark = C.bold(C.ok("  ★ new best"))
            print(f"  start {k+1}/{n_starts}   obj = {obj_num:7.1f} N   {tag}{mark}")
        except Exception as e:
            print(f"  start {k+1}/{n_starts}   {C.warn('solve failed')} ({type(e).__name__})")

    if best["x"] is None:
        print(C.warn("  All starts failed — returning warm start."))
        best = {"x": dict(INIT), "alpha": 2.0, "feasible": False}
    return best


def _feasibility(x, alpha_cruise):
    """Numeric re-evaluation of objective + max relative constraint violation."""
    m = compute_metrics(x, alpha_cruise, model_size="medium")
    g = lambda k: float(np.array(m[k]))
    viols = []
    viols.append(abs(g("L") - WEIGHT) / WEIGHT)
    viols.append(max(0.0, (WEIGHT_MAX - g("L_to")) / WEIGHT_MAX))
    viols.append(max(0.0, (g("CL_to") - cfg["takeoff_cl_margin"] * CL_MAX_TO) / CL_MAX_TO))
    # Active-pilot trim: weight-shift to hold trim within the comfortable limit.
    viols.append(max(0.0, (abs(g("shift_trim_m")) - STANCE_TRIM_MAX) / STANCE_TRIM_MAX))
    # Static stability: tail volume band (auto-calibrated) + positive SM floor.
    vh = g("V_h")
    viols.append(max(0.0, (VH_RANGE[0] - vh) / VH_RANGE[0], (vh - VH_RANGE[1]) / VH_RANGE[1]))
    viols.append(max(0.0, (SM_FLOOR - g("SM_chord")) / max(SM_FLOOR, 1e-3)))
    viols.append(max(0.0, (g("sigma_vm") - SIGMA_ADMISSIBLE) / SIGMA_ADMISSIBLE))
    viols.append(max(0.0, (-SIGMA_CAV - g("Cpmin")) / SIGMA_CAV))
    viols.append(max(0.0, (-SIGMA_CAV - g("Cpmin_tip")) / SIGMA_CAV))
    viols.append(max(0.0, (g("tip_defl_ratio") - TIP_DEFL_MAX) / TIP_DEFL_MAX))
    viols.append(max(0.0, (_min_tip_chord - g("tip_chord")) / _min_tip_chord))
    viols.append(max(0.0, (g("mast_sigma") - MAST_ALLOW) / MAST_ALLOW))
    viols.append(max(0.0, (g("mast_defl") - MAST_DEFL_MAX) / MAST_DEFL_MAX))
    viol = max(viols)
    obj  = SPEED_WEIGHTS[0] * float(np.array(m["D_total"])) + TAKEOFF_WEIGHT * float(np.array(m["D_to"]))
    return (viol < 0.02), viol, obj

# ─────────────────────────────────────────────────────────────────────────────
# 9. LIFTINGLINE VERIFICATION  (exact downwash + induced drag on the optimum)
# ─────────────────────────────────────────────────────────────────────────────
def verify_liftingline(x, alpha_cruise):
    """Re-trim α in LiftingLine at cruise; report exact 3D L/D. Not a re-optimization
    — an honest check that the AeroBuildup optimum survives the exact physics."""
    m = compute_metrics(x, alpha_cruise, model_size="medium")
    airplane = m["airplane"]
    plane = asb.Airplane(wings=airplane.wings, xyz_ref=airplane.xyz_ref,
                         s_ref=float(np.array(m["S_wing"])),
                         c_ref=float(np.array(m["mean_chord"])), b_ref=x["wing_span"])
    def L_at(a):
        r = asb.LiftingLine(plane, asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=a,
                                                      atmosphere=atmosphere)).run()
        return float(r["L"]), r
    try:
        a_lo, a_hi = 0.0, 5.0
        (L_lo, _), (L_hi, _) = L_at(a_lo), L_at(a_hi)
        kL = float(np.array(m["kL"]))
        if abs(L_hi - L_lo) < 1.0:
            a_tr = alpha_cruise
        else:
            a_tr = a_lo + (WEIGHT / kL - L_lo) / (L_hi - L_lo) * (a_hi - a_lo)
            a_tr = float(np.clip(a_tr, -2, 10))
        L_ll, r = L_at(a_tr)
        D_ll = float(r["D"]) * 1.0
        return dict(alpha=a_tr, L=kL * L_ll,
                    D=D_ll + float(np.array(m["D_mast"])) + float(np.array(m["D_wave"])),
                    CL=float(r["CL"]), ok=True)
    except Exception as e:
        return dict(ok=False, err=str(e))


def performance_envelope(x):
    """Report-only, numeric. Two all-conditions checks the 2-point optimum can hide:
       (1) SPEED SWEEP — trimmed L/D and CL from 0.7× to 1.3× v_cruise;
       (2) TIP-STALL MARGIN — the tip section's CL_max angle (NeuralFoil sweep) vs
           its take-off operating angle. A small margin = ventilation/spin-out risk
           in turns & pumping."""
    p = {n: x[n] for n in VAR_NAMES}
    airplane, geo = build_airplane(p)
    mean_chord = float(np.array(geo["mean_chord"]))
    kL = float(np.array(free_surface_lift_factor(mean_chord)))
    span = p["wing_span"]

    def _ab(V, a):
        return asb.AeroBuildup(airplane, asb.OperatingPoint(velocity=V, alpha=a,
                                                            atmosphere=atmosphere)).run()
    rows = []
    for frac in (0.7, 0.85, 1.0, 1.15, 1.3):
        V = cfg["v_cruise"] * frac
        r0, r1 = _ab(V, 0.0), _ab(V, 4.0)
        L0, L1 = float(r0["L"]) * kL, float(r1["L"]) * kL
        if abs(L1 - L0) < 1.0:
            continue
        a_tr = float(np.clip((WEIGHT - L0) / (L1 - L0) * 4.0, -3, 12))
        rt = _ab(V, a_tr)
        CL = float(rt["CL"])
        D_mast_s = float(mast_drag_structure(p.get("mast_chord", INIT["mast_chord"]), V)[0])
        D = float(rt["D"]) + D_mast_s + float(wave_drag(CL, mean_chord, span, V))
        rows.append(dict(V=V, alpha=a_tr, CL=CL, LD=WEIGHT / D))

    # Tip-stall margin
    af_tip = make_wing_airfoil(p["camber_tip"], p["thick_tip"], "tipstall")
    tip_chord = p["taper"] * p["wing_root_chord"]
    Re_tip = RHO * cfg["v_cruise"] * tip_chord / MU
    sweep = np.linspace(0, 16, 33)
    pol = af_tip.get_aero_from_neuralfoil(alpha=sweep, Re=Re_tip, model_size="medium")
    CLt = np.array(pol["CL"], dtype=float)
    istall = int(np.argmax(CLt))
    a_stall, clmax_tip = float(sweep[istall]), float(CLt[istall])
    tip_aoa_to = x["alpha_to"] + x["wing_setting_angle"] + x["tip_washout"]
    tip = dict(clmax=clmax_tip, a_stall=a_stall, aoa_to=tip_aoa_to,
               margin=a_stall - tip_aoa_to, Re=Re_tip,
               want=float(SAFETY.get("tip_stall_margin_deg", 4.0)))
    return rows, tip


# ─────────────────────────────────────────────────────────────────────────────
# 10. REPORT & EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def next_output_dir(suffix="v4", out_root=None):
    # Anchor on the PROJECT ROOT (src/..) so results always land in the canonical
    # <project>/outputs/ regardless of the working directory — same convention as
    # optFixedProfileRefine3d.py (ROOT.parent / "outputs"), keeping V3 and V4
    # outputs side by side.
    if out_root is None:
        out_root = str(ROOT.parent / "outputs")
    today = dt.datetime.now().strftime("%Y%m%d")
    prefix = f"{CASE}_{today}"
    os.makedirs(out_root, exist_ok=True)
    suffix_re = re.escape("_" + suffix) if suffix else ""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+){suffix_re}$")
    nums = [int(mm.group(1)) for d in os.listdir(out_root) if (mm := pat.match(d))]
    n = max(nums) + 1 if nums else 1
    name = f"{prefix}_{n:02d}" + (f"_{suffix}" if suffix else "")
    return os.path.join(out_root, name)


def _export_profile_dat(af_obj, airfoils_dir, filename, name_internal):
    af_obj = af_obj.repanel(n_points_per_side=50)
    coords = np.array(af_obj.coordinates, dtype=float).copy()
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
        for xx, yy in final:
            f.write(f" {xx:.6f} {yy:.6f}\n")


def full_report(best):
    x, alpha_cruise = best["x"], best["alpha"]
    m = compute_metrics(x, alpha_cruise, model_size="medium")
    g = lambda k: float(np.array(m[k]))

    L, D_total = g("L"), g("D_total")
    ll = verify_liftingline(x, alpha_cruise)

    omega_n = np.sqrt(max(g("omega_n_sq"), 0.0)) / (2 * np.pi)
    f_lo, f_hi = PILOT_FREQ_LO, PILOT_FREQ_HI
    sigma_static, _ = structure_root(x["wing_root_chord"], x["wing_span"],
                                     REF_TC * x["thick_root"], 1.0)
    sigma_static = float(np.array(sigma_static))
    Re_root = RHO * cfg["v_cruise"] * x["wing_root_chord"] / MU
    tip_chord = x["taper"] * x["wing_root_chord"]

    HBAR = "═" * 70
    def SEC(t): return C.sec(f"  ── {t} " + "─" * max(0, 58 - len(t)))
    print(); print(C.head(HBAR)); print(C.head(f"  RESULT V4  ·  {CASE.upper()} freeride")); print(C.head(HBAR))

    print(SEC("Geometry"))
    print(f"    Wing   span {x['wing_span']*100:5.1f} cm   root/tip "
          f"{x['wing_root_chord']*1000:3.0f}/{tip_chord*1000:2.0f} mm   "
          f"AR {g('AR_wing'):4.2f}   taper {x['taper']:.2f}   shape {x['chord_shape']:.2f}")
    print(f"    Airfoil root  camber×{x['camber_root']:.2f} thick×{x['thick_root']:.2f} "
          f"(t/c {REF_TC*x['thick_root']*100:.1f}%)   "
          f"tip camber×{x['camber_tip']:.2f} thick×{x['thick_tip']:.2f} "
          f"(t/c {REF_TC*x['thick_tip']*100:.1f}%)")
    print(f"    Twist  root {x['wing_setting_angle']:+.2f}°  tip washout {x['tip_washout']:+.2f}°")
    print(f"    Stab   span {x['stab_span']*100:5.1f} cm   root {x['stab_root_chord']*1000:.0f} mm   "
          f"AR {g('AR_stab'):4.2f}   incid {x['s_twist']:+.2f}°   camber {x['stab_camber']:+.2f}")
    mast_struct = f"σ {g('mast_sigma')/1e6:.0f}/{MAST_ALLOW/1e6:.0f} MPa · flex {g('mast_defl')*100:.1f}/{MAST_DEFL_MAX*100:.0f}%"
    print(f"    Mast   chord {x['mast_chord']*1000:.0f} mm (t/c {MAST_TC*100:.0f}%)   {C.dim(mast_struct)}")
    print(f"    Fuse   {x['fuselage_length']*100:5.1f} cm   nominal CG {CG_NOM*100:.0f}% c̄ "
          f"{C.dim(f'(level: {LEVEL})')}")

    print(SEC("Performance"))
    D_aero_v, D_wave_v, kL_v, D_mast_v = g("D"), g("D_wave"), g("kL"), g("D_mast")
    cl_to_tgt = cfg["takeoff_cl_margin"] * CL_MAX_TO
    print(f"    α cruise {alpha_cruise:+.2f}°   α takeoff {x['alpha_to']:+.2f}°")
    print(f"    L cruise {L:7.1f} N  {C.dim(f'/ {WEIGHT:.1f} N nominal, takeoff sized {WEIGHT_MAX:.0f} N')}")
    breakdown = f"(aero {D_aero_v:.2f} + mast {D_mast_v:.2f} + wave {D_wave_v:.2f})"
    print(f"    D total  {D_total:7.2f} N  {C.dim(breakdown)}")
    print(f"    {C.bold(f'L/D total {L/D_total:6.2f}')}   {C.dim(f'(free-surface kL={kL_v:.3f})')}")
    if ll.get("ok"):
        ll_str = f"LiftingLine check: L {ll['L']:.1f} N  D {ll['D']:.2f} N  L/D {ll['L']/ll['D']:.2f}  @ α {ll['alpha']:.2f}°"
        print(f"    {C.dim(ll_str)}")
    print(f"    L takeoff {g('L_to'):7.1f} N   CL_to {g('CL_to'):.3f} "
          f"{C.dim(f'≤ {cl_to_tgt:.3f}')}   Re_root {Re_root:.2e}")

    print(SEC("Stability (industry metric: tail volume, ref-calibrated)"))
    vh_v, sm_v = g("V_h"), g("SM_chord")
    vh_mark = C.ok("✓") if (VH_RANGE[0] - 0.005 <= vh_v <= VH_RANGE[1] + 0.005) else C.warn("⚠")
    sm_mark = C.ok("✓") if sm_v >= SM_FLOOR else C.warn("⚠")
    vh_tgt = f"ref {REF_VH:.2f}, band {VH_RANGE[0]:.2f}-{VH_RANGE[1]:.2f}"
    print(f"    V_h  {vh_v:5.3f} {C.dim(vh_tgt)} {vh_mark}   tail arm l_t {g('l_t')*100:4.1f} cm")
    np_gap_mm = g("SM_abs") * 1000
    print(f"    Static margin {sm_v*100:4.1f}% c̄ {C.dim(f'> {SM_FLOOR*100:.0f}% (stable)')} {sm_mark}   "
          f"{C.dim(f'(NP {np_gap_mm:.0f} mm aft of CG — foils run high)')}")
    shift_cm = g("shift_trim_m") * 100
    reserve  = g("authority_reserve")
    fstab_v  = g("F_stab")
    tr_mark  = C.ok("✓") if abs(g("shift_trim_m")) <= STANCE_TRIM_MAX + 1e-4 else C.warn("⚠")
    print(f"    Pilot trim: shift {shift_cm:+.1f} cm {C.dim(f'≤ ±{STANCE_TRIM_MAX*100:.0f} cm')} {tr_mark}   "
          f"reserve authority {reserve:+.0f} N·m   {C.dim(f'stab {fstab_v:+.0f} N')}")
    print(f"    Ride-height stiffness {g('heave_stiff'):+.2f} N/cm {C.dim('(>0 = self-correcting depth)')}   "
          f"{C.dim(f'ω_n {omega_n:.2f} Hz info')}")

    print(SEC("Structure & cavitation (hand-built limits)"))
    sm2 = C.ok("✓") if g("sigma_vm") <= SIGMA_ADMISSIBLE else C.warn("⚠")
    print(f"    σ_VM peak {g('sigma_vm')/1e6:5.1f} MPa {C.dim(f'≤ {SIGMA_ADMISSIBLE/1e6:.0f} hand-layup allow.')} {sm2}   "
          f"{C.dim(f'static {sigma_static/1e6:.0f} MPa')}")
    td = C.ok("✓") if g("tip_defl_ratio") <= TIP_DEFL_MAX else C.warn("⚠")
    cav_r = C.ok("✓") if g("Cpmin") >= -SIGMA_CAV else C.warn("⚠")
    cav_t = C.ok("✓") if g("Cpmin_tip") >= -SIGMA_CAV else C.warn("⚠")
    print(f"    Tip deflection {g('tip_defl_ratio')*100:4.1f}% semi-span {C.dim(f'≤ {TIP_DEFL_MAX*100:.0f}%')} {td}")
    print(f"    Cavitation σ={SIGMA_CAV:.2f} @ {cfg['v_cruise']} m/s   "
          f"Cpmin root {g('Cpmin'):+.2f} {cav_r}  tip {g('Cpmin_tip'):+.2f} {cav_t}")

    # ── All-conditions envelope: speed sweep + tip-stall margin ──
    try:
        rows, tip = performance_envelope(x)
    except Exception as e:
        rows, tip = [], None
        print(C.warn(f"  envelope failed: {e}"))
    print(SEC("All-conditions envelope"))
    if rows:
        hdr = "  ".join(f"{r['V']:4.1f}" for r in rows)
        cls = "  ".join(f"{r['CL']:4.2f}" for r in rows)
        lds = "  ".join(f"{r['LD']:4.1f}" for r in rows)
        print(f"    V (m/s)   {hdr}")
        print(f"    CL trim   {cls}   {C.dim('(low CL = fast/efficient, high = slow/grunty)')}")
        print(f"    L/D tot   {lds}")
    if tip:
        tmark = C.ok("✓") if tip["margin"] >= tip["want"] else C.warn("⚠")
        want_s = f"want ≥ {tip['want']:.0f}°"
        re_s   = f"Re_tip {tip['Re']:.1e}"
        print(f"    Tip stall: CL_max {tip['clmax']:.2f} @ {tip['a_stall']:.1f}°   "
              f"takeoff tip AoA {tip['aoa_to']:+.1f}°   margin {tip['margin']:.1f}° "
              f"{C.dim(want_s)} {tmark}   {C.dim(re_s)}")

    # ── Export ──
    out_dir = next_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    airfoils_dir = os.path.join(out_dir, "airfoils")
    os.makedirs(airfoils_dir, exist_ok=True)
    airplane = m["airplane"]; geo = m["geo"]

    def _rename_and_export(xsecs, prefix):
        for i, xs in enumerate(xsecs):
            n = f"{prefix}_sec_{i}"
            af_copy = copy.deepcopy(xs.airfoil)
            try:
                af_copy = af_copy.to_airfoil() if hasattr(af_copy, "to_airfoil") else af_copy
            except Exception:
                pass
            af_copy.name = n
            xs.airfoil = af_copy
            _export_profile_dat(af_copy, airfoils_dir, f"{n}.dat", n)

    _rename_and_export(airplane.wings[0].xsecs, "wing")
    _rename_and_export(airplane.wings[1].xsecs, "stab")
    _rename_and_export(geo["mast_obj"].xsecs,   "mast")

    xml_path = os.path.join(out_dir, f"{os.path.basename(out_dir)}_plane.xml")
    try:
        asb.Airplane(
            wings=[asb.Wing(symmetric=True, name="mainwing", xsecs=airplane.wings[0].xsecs),
                   asb.Wing(symmetric=True, name="elevator", xsecs=airplane.wings[1].xsecs),
                   geo["mast_obj"]],
            fuselages=[geo["fuselage_obj"]], xyz_ref=airplane.xyz_ref,
        ).export_XFLR5_xml(xml_path)
        xml_ok = True
    except Exception as e:
        xml_ok = False
        print(f"    {C.warn('~')} XFLR5 XML not exported: {e}")

    _export_md(out_dir, x, alpha_cruise, m, omega_n, sigma_static, ll, rows, tip)
    np.save(os.path.join(out_dir, "x_best_v4.npy"),
            np.array([x[n] for n in VAR_NAMES] + [alpha_cruise]))

    # ── Saturated-bounds report (which DOF want to go further) ──
    saturated = []
    for n in VAR_NAMES:
        lob, hib = BOUNDS[n]
        width = hib - lob
        if width <= 0:
            continue
        if x[n] - lob < 0.02 * width:
            saturated.append(f"{n} ↓ {lob:.3g}")
        elif hib - x[n] < 0.02 * width:
            saturated.append(f"{n} ↑ {hib:.3g}")
    if saturated:
        print(SEC(f"Saturated bounds ({len(saturated)})"))
        for s in saturated:
            print(f"    {C.warn('⚠')}  {s}")
        print(C.dim("       → loosen these bounds to let the shape explore further"))

    print(SEC("Exports"))
    print(f"    {C.ok('✓')} Sheet     {C.dim(out_dir + '/fiche_technique_v4.md')}")
    print(f"    {C.ok('✓')} Airfoils  {C.dim(airfoils_dir + '/')}")
    if xml_ok:
        print(f"    {C.ok('✓')} XFLR5 XML {C.dim(xml_path)}")
    if not best["feasible"]:
        print(C.warn("  ⚠ Best solution is not fully feasible — inspect saturated constraints."))
    print(C.head(HBAR)); print()


def _export_md(out_dir, x, alpha_cruise, m, omega_n, sigma_static, ll, rows=None, tip=None):
    g = lambda k: float(np.array(m[k]))
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tip_chord = x["taper"] * x["wing_root_chord"]
    ll_line = (f"| LiftingLine (3D check) | L {ll['L']:.1f} N · D {ll['D']:.2f} N · "
               f"L/D {ll['L']/ll['D']:.2f} @ α {ll['alpha']:.2f}° |" if ll.get("ok")
               else "| LiftingLine (3D check) | failed |")
    lines = [
        f"# Technical Sheet V4 — {CASE.upper()} freeride  ·  level: {LEVEL}", "", f"*Generated on {now}*",
        "", "Shape-optimized (gradient/IPOPT) on AeroBuildup+NeuralFoil: free-surface,",
        "real Cpmin cavitation, morphed wing+stab sections, a mast beam+drag model, and",
        f"multi-point speed trim ({', '.join(f'{s:g}×' for s in SPEED_POINTS)} v_cruise); LiftingLine-verified.",
        "", "---", "", "## 1. Optimized geometry", "",
        "| Variable | Value |", "|:---|:---|",
        f"| Wing span | {x['wing_span']*100:.1f} cm |",
        f"| Wing root / tip chord | {x['wing_root_chord']*1000:.0f} / {tip_chord*1000:.0f} mm |",
        f"| Taper (tip/root) | {x['taper']:.3f} |",
        f"| Chord shape (0 lin · 1 ell) | {x['chord_shape']:.3f} |",
        f"| Wing AR | {g('AR_wing'):.2f} |",
        f"| Root airfoil | {WING_AIRFOIL_NAME} · camber×{x['camber_root']:.2f} · thick×{x['thick_root']:.2f} (t/c {REF_TC*x['thick_root']*100:.1f}%) |",
        f"| Tip airfoil | camber×{x['camber_tip']:.2f} · thick×{x['thick_tip']:.2f} (t/c {REF_TC*x['thick_tip']*100:.1f}%) |",
        f"| Root incidence | {x['wing_setting_angle']:.2f}° |",
        f"| Tip washout | {x['tip_washout']:.2f}° |",
        f"| Stab span / root | {x['stab_span']*100:.1f} cm / {x['stab_root_chord']*1000:.0f} mm |",
        f"| Stab incidence / camber | {x['s_twist']:.2f}° / {x['stab_camber']:+.2f} (reflex) |",
        f"| Mast chord (t/c {MAST_TC*100:.0f}%) | {x['mast_chord']*1000:.0f} mm |",
        f"| Fuselage length | {x['fuselage_length']*100:.1f} cm |",
        f"| Nominal CG (rider stance) | {CG_NOM*100:.0f}% c̄ |",
        f"| α cruise (trim L=W) | {alpha_cruise:.2f}° |",
        f"| α takeoff | {x['alpha_to']:.2f}° |",
        "", "---", "", "## 2. Performance", "",
        "| Parameter | Value |", "|:---|:---|",
        f"| Weight (nominal / takeoff-sized) | {WEIGHT:.0f} N ({WEIGHT/GRAV:.0f} kg) / {WEIGHT_MAX:.0f} N ({WEIGHT_MAX/GRAV:.0f} kg) |",
        f"| V cruise / takeoff | {cfg['v_cruise']} / {cfg['v_takeoff']} m/s |",
        f"| L cruise | {g('L'):.1f} N |",
        f"| D total (aero+mast+wave) | {g('D_total'):.2f} N |",
        f"| D mast (beam+NeuralFoil) | {g('D_mast'):.2f} N |",
        f"| D wave (free surface) | {g('D_wave'):.2f} N |",
        f"| Free-surface lift factor kL | {g('kL'):.3f} |",
        f"| **L/D total** | **{g('L')/g('D_total'):.2f}** |",
        ll_line,
        f"| L takeoff | {g('L_to'):.1f} N |",
        f"| CL takeoff | {g('CL_to'):.3f} (≤ {cfg['takeoff_cl_margin']*CL_MAX_TO:.3f}) |",
        "", "---", "", "## 3. Stability & pilotability (foil-specific)", "",
        "The rider is an **active controller**: they trim by shifting weight fore/aft,",
        "so CG is not optimized. We bound the **weight-shift to hold trim** and keep a",
        "control **reserve**. The stab is sized by the **tail volume** (calibrated to",
        "this discipline's reference kit). Static margin is only floored — foils run",
        "SM/c̄ > 100% (huge tail arm vs small chord), which is normal and stable.",
        "", "| Parameter | Value | Target |", "|:---|:---|:---|",
        f"| **Pilot weight-shift to trim** | **{g('shift_trim_m')*100:+.1f} cm** | \\|·\\| ≤ {STANCE_TRIM_MAX*100:.0f} cm |",
        f"| Control reserve authority | {g('authority_reserve'):.0f} N·m | > 0 |",
        f"| **Tail volume V_h** | **{g('V_h'):.3f}** | ref {REF_VH:.2f}, band [{VH_RANGE[0]:.2f}–{VH_RANGE[1]:.2f}] |",
        f"| Tail arm l_t | {g('l_t')*100:.1f} cm | — |",
        f"| Static margin SM/c̄ | {g('SM_chord')*100:.1f}% | > {SM_FLOOR*100:.0f}% (stable) |",
        f"| Neutral point aft of CG | {g('SM_abs')*1000:.0f} mm | > 0 |",
        f"| Stab force | {g('F_stab'):.1f} N | [{cfg['stab_load_range'][0]:.0f},{cfg['stab_load_range'][1]:.0f}] N |",
        f"| Ride-height (heave) stiffness | {g('heave_stiff'):+.2f} N/cm | > 0 (self-correcting) |",
        f"| Cm_α (info) | {g('Cm_alpha'):.2f} rad⁻¹ | < 0 |",
        f"| ω_n (info, not a driver) | {omega_n:.2f} Hz | ([{PILOT_FREQ_LO:.1f}–{PILOT_FREQ_HI:.1f}] Hz ref) |",
        "", "## 4. Structure & cavitation (hand-built limits)", "",
        "| Parameter | Value | Target |", "|:---|:---|:---|",
        f"| σ_VM peak (×{LOAD_PEAK_FACTOR:.1f}g) | {g('sigma_vm')/1e6:.1f} MPa | < {SIGMA_ADMISSIBLE/1e6:.0f} MPa (hand wet-layup) |",
        f"| σ_VM static (1g) | {sigma_static/1e6:.1f} MPa | < {SIGMA_ULTIMATE/1e6:.0f} MPa (ultimate) |",
        f"| Tip deflection | {g('tip_defl_ratio')*100:.1f}% semi-span | < {TIP_DEFL_MAX*100:.0f}% |",
        f"| Root t/c | {REF_TC*x['thick_root']*100:.1f}% | ≥ {MFG.get('min_tc_root',0.11)*100:.0f}% (buildable) |",
        f"| Tip t/c | {REF_TC*x['thick_tip']*100:.1f}% | ≥ {MFG.get('min_tc_tip',0.085)*100:.1f}% (buildable) |",
        f"| Mast bending σ (@ {MAST_TURN_G:.1f}g carve) | {g('mast_sigma')/1e6:.0f} MPa | < {MAST_ALLOW/1e6:.0f} MPa (UD) |",
        f"| Mast lateral deflection | {g('mast_defl')*100:.1f}% length | < {MAST_DEFL_MAX*100:.0f}% |",
        f"| Cavitation number σ @ {cfg['v_cruise']} m/s | {SIGMA_CAV:.2f} | — |",
        f"| Cpmin root / tip (real) | {g('Cpmin'):.2f} / {g('Cpmin_tip'):.2f} | ≥ {-SIGMA_CAV:.2f} |",
        "",
    ]
    if rows:
        lines += [
            "---", "", "## 5. All-conditions envelope", "",
            "Trimmed (L=W) across the speed range — the 2-point optimum can hide off-design",
            "behaviour. Low CL = fast/efficient; high CL = slow/grunty (early lift).",
            "", "| V (m/s) | " + " | ".join(f"{r['V']:.1f}" for r in rows) + " |",
            "|:---|" + "---|" * len(rows),
            "| α trim (°) | " + " | ".join(f"{r['alpha']:+.1f}" for r in rows) + " |",
            "| CL | " + " | ".join(f"{r['CL']:.2f}" for r in rows) + " |",
            "| L/D total | " + " | ".join(f"{r['LD']:.1f}" for r in rows) + " |",
            "",
        ]
    if tip:
        ok = "✓ OK" if tip["margin"] >= tip["want"] else "⚠ marginal — soften the tip"
        lines += [
            "**Tip-stall margin** (turns / pumping safety):", "",
            f"- Tip section CL_max **{tip['clmax']:.2f} @ {tip['a_stall']:.1f}°** (Re {tip['Re']:.1e})",
            f"- Take-off tip local AoA **{tip['aoa_to']:+.1f}°** → margin **{tip['margin']:.1f}°** "
            f"(want ≥ {tip['want']:.0f}°) — {ok}",
            "",
        ]
    with open(os.path.join(out_dir, "fiche_technique_v4.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 11. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # V4_STARTS: number of multi-start IPOPT solves (graph traced once, then reused).
    # V4_MODEL:  in-loop NeuralFoil size (xsmall=fast, medium=more accurate).
    n_starts = int(os.environ.get("V4_STARTS", 4))
    best = run_optimization(n_starts=n_starts, model_size=os.environ.get("V4_MODEL", "xsmall"))
    full_report(best)
