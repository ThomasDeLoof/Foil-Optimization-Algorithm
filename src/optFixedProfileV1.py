# ==================================================================================
# === Hydrofoil Optimization Script - NACA Parametric Search - IPOPT Solver ========
# ==================================================================================

import os
import datetime as dt
import aerosandbox as asb
import aerosandbox.numpy as np
import sys
from pathlib import Path
import yaml

# =========================================================================================
# ==============================   CONFIGURATION    =======================================
# =========================================================================================

# --- Paths & Atmosphere configuration ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from config.water_atmosphere import Water as Atmosphere

# --- Load global physical configuration ---
with open(ROOT / "config" / "parameters.yaml") as f:
    phy = yaml.safe_load(f)

CASE = phy["case"]

# Masses & Weights
mass = phy["pilot"]["mass_kg"] + phy["board"]["mass_kg"]

# Search spaces
cambrures       = phy["search_space"]["cambrures"]
epaisseurs_root = phy["search_space"]["epaisseurs_root"]
angle_reflex    = phy["search_space"]["angle_reflex"]

# Alpha
alpha_bounds = phy["alpha"]["bounds"]
alpha_init   = phy["alpha"]["init"]

# Wing
wing_span_bounds      = phy["wing"]["span_bounds"]
wing_span_init        = phy["wing"]["span_init"]
root_chord_bounds     = phy["wing"]["root_chord_bounds"]
root_chord_init       = phy["wing"]["root_chord_init"]
tip_chord_bounds      = phy["wing"]["tip_chord_bounds"]
tip_chord_init        = phy["wing"]["tip_chord_init"]
twist_bounds          = phy["wing"]["twist_bounds"]
twist_init            = phy["wing"]["twist_init"]
sweep_deg             = phy["wing"]["sweep_deg"]
wing_anhedral_deg     = phy["wing"]["anhedral_deg"]
N_sections_wing       = phy["wing"]["n_sections"]
CL_max_takeoff_approx = phy["wing"]["cl_max_takeoff"]

# Fuselage
fuselage_bounds     = phy["fuselage"]["length_bounds"]
fuselage_init       = phy["fuselage"]["length_init"]
fuselage_diameter   = phy["fuselage"]["diameter"]
x_fuselage_start    = phy["fuselage"]["x_start"]
N_sections_fuselage = phy["fuselage"]["n_sections"]

# Mast
mast_length          = phy["mast"]["length"]
profondeur_immersion = phy["mast"]["immersion_depth"]
mast_chord_top       = phy["mast"]["chord_top"]
mast_chord_bot       = phy["mast"]["chord_bot"]
mast_profile         = phy["mast"]["profile"]
x_mast               = phy["mast"]["x_position"]
N_sections_mast      = phy["mast"]["n_sections"]
ratio_immersion      = profondeur_immersion / mast_length
chord_at_water_line  = mast_chord_bot * (1 - ratio_immersion) + mast_chord_top * ratio_immersion

# Stabilizer
stab_span_bounds       = phy["stab"]["span_bounds"]
stab_span_init         = phy["stab"]["span_init"]
stab_root_chord_bounds = phy["stab"]["root_chord_bounds"]
stab_root_chord_init   = phy["stab"]["root_chord_init"]
stab_tip_chord_bounds  = phy["stab"]["tip_chord_bounds"]
stab_tip_chord_init    = phy["stab"]["tip_chord_init"]
s_twist_bounds         = phy["stab"]["twist_bounds"]
s_twist_init           = phy["stab"]["twist_init"]
s_sweep_deg            = phy["stab"]["sweep_deg"]
stab_dihedral_deg      = phy["stab"]["dihedral_deg"]
N_sections_stab        = phy["stab"]["n_sections"]

# --- Load active scenario ---
with open(ROOT / "config" / "scenarios.yaml") as f:
    SCENARIOS = yaml.safe_load(f)

if CASE not in SCENARIOS:
    raise ValueError(f"Case '{CASE}' not found in scenarios.yaml. Options: {list(SCENARIOS.keys())}")
cfg = SCENARIOS[CASE]

rig_mass = cfg["rig_mass_kg"]  # Rig mass
total_mass = mass + rig_mass  # Total system mass (pilot + board + rig)
weight     = total_mass * 9.81  # Total weight (N)


# =========================================================================================
# ==============================   NACA AIRFOILS    =======================================
# =========================================================================================

def get_airfoil(name: str) -> asb.Airfoil:
    try:
        return asb.Airfoil(name).repanel(50)
    except Exception:
        return asb.Airfoil("naca0012").repanel(50)


def interpolate_airfoil(af1: asb.Airfoil, af2: asb.Airfoil, r: float) -> asb.Airfoil:
    coords = (1 - r) * af1.coordinates + r * af2.coordinates
    return asb.Airfoil("blend", coordinates=coords).repanel(50)


def apply_reflex_to_naca(naca_name: str, reflex_angle_deg: float) -> asb.Airfoil:
    """
    Applies a 'Reflex' deformation (S-Shape) on the rear 30% of the airfoil.
    Allows control of the pitching moment without modifying the overall camber.
    """
    base_af = asb.Airfoil(naca_name)
    x = base_af.coordinates[:, 0]
    y = base_af.coordinates[:, 1]
    x_start_bend = 0.7
    mask = x > x_start_bend
    if reflex_angle_deg > 0:
        bend_strength = np.radians(reflex_angle_deg) * 1.5
        y[mask] += (x[mask] - x_start_bend) ** 2 * bend_strength
    return asb.Airfoil(
        name=f"{naca_name}_Reflex_{reflex_angle_deg}deg",
        coordinates=np.stack((x, y), axis=1)
    ).repanel(50)


# =========================================================================================
# ==============================   TECHNICAL SHEET EXPORT   ===============================
# =========================================================================================

def export_fiche_technique(
    out_dir, root_naca_profile,
    # Wing geometry
    val_surface, val_span, val_ar, val_root_c, val_tip_c, val_twist, wing_loading, reflex_angle_deg,
    # Stabilizer geometry
    val_s_surface, val_s_span, val_s_ar, val_s_root_c, val_s_twist, val_fuselage_length,
    # Performance
    val_finesse, val_trainee, val_alpha, Re_root, Re_tip, CL_cruise, CD_cruise, CL_stab,
    # Stability
    val_sm, val_cg, val_xcg, val_force_stab, val_moment, val_vh,
):
    """
    Exports the technical sheet of the optimized design to a Markdown file.
    """
    now_str  = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filepath = os.path.join(out_dir, "fiche_technique.md")
    lines    = []

    # Header
    lines += [
        f"# Technical Sheet — {CASE.upper()} | {root_naca_profile.upper()}",
        f"",
        f"*Generated on {now_str}*",
        f"",
        f"---",
        f"",
    ]

    # Snapshot of input parameters
    lines += [
        f"## 0. Run Parameters",
        f"",
        f"### Parametric search",
        f"| Parameter | Value |",
        f"|:---|:---|",
        f"| Cambers tested | {cambrures} |",
        f"| Root thicknesses tested | {epaisseurs_root} |",
        f"| Reflex angles tested | {angle_reflex} |",
        f"| Selected root airfoil | {root_naca_profile.upper()} |",
        f"| Selected reflex angle | {reflex_angle_deg}° |",
        f"",
        f"### Optimization bounds — Wing",
        f"| Variable | Min bound | Max bound | Init |",
        f"|:---|:---|:---|:---|",
        f"| Span (m) | {wing_span_bounds[0]} | {wing_span_bounds[1]} | {wing_span_init} |",
        f"| Root chord (m) | {root_chord_bounds[0]} | {root_chord_bounds[1]} | {root_chord_init} |",
        f"| Tip chord (m) | {tip_chord_bounds[0]} | {tip_chord_bounds[1]} | {tip_chord_init} |",
        f"| Twist (°) | {twist_bounds[0]} | {twist_bounds[1]} | {twist_init} |",
        f"",
        f"### Optimization bounds — Stabilizer",
        f"| Variable | Min bound | Max bound | Init |",
        f"|:---|:---|:---|:---|",
        f"| Span (m) | {stab_span_bounds[0]} | {stab_span_bounds[1]} | {stab_span_init} |",
        f"| Root chord (m) | {stab_root_chord_bounds[0]} | {stab_root_chord_bounds[1]} | {stab_root_chord_init} |",
        f"| Tip chord (m) | {stab_tip_chord_bounds[0]} | {stab_tip_chord_bounds[1]} | {stab_tip_chord_init} |",
        f"| Incidence angle (°) | {s_twist_bounds[0]} | {s_twist_bounds[1]} | {s_twist_init} |",
        f"",
        f"### Optimization bounds — Fuselage & CG",
        f"| Variable | Min bound | Max bound | Init |",
        f"|:---|:---|:---|:---|",
        f"| Fuselage length (m) | {fuselage_bounds[0]} | {fuselage_bounds[1]} | {fuselage_init} |",
        f"| CG ratio | {cfg['cg_range'][0]} | {cfg['cg_range'][1]} | {cfg['cg_ratio_init']} |",
        f"",
        f"### Scenario constraints — {CASE.upper()}",
        f"| Constraint | Min | Max |",
        f"|:---|:---|:---|",
        f"| Wing area (m²) | {cfg['area_target_range'][0]} | {cfg['area_target_range'][1]} |",
        f"| Stab area (m²) | {cfg['stab_area_range'][0]} | {cfg['stab_area_range'][1]} |",
        f"| Static margin | {cfg['sm_range'][0]} | {cfg['sm_range'][1]} |",
        f"| Stab force (N) | {cfg['stab_load_range'][0]} | {cfg['stab_load_range'][1]} |",
        f"| Tail volume | {cfg['vh_range'][0]} | {cfg['vh_range'][1]} |",
        f"",
        f"---",
        f"",
    ]

    # Flight conditions
    lines += [
        f"## Flight conditions",
        f"",
        f"| Parameter | Value |",
        f"|:---|:---|",
        f"| Case | {CASE} |",
        f"| Total weight | {weight:.1f} N ({weight/9.81:.0f} kg) |",
        f"| Takeoff speed | {cfg['v_takeoff']} m/s |",
        f"| Cruise speed | {cfg['v_cruise']} m/s |",
        f"",
        f"---",
        f"",
    ]

    # Wing geometry
    lines += [
        f"## 1. Wing Geometry",
        f"",
        f"| Parameter | Value |",
        f"|:---|:---|",
        f"| Airfoil | {root_naca_profile.upper()} |",
        f"| Area | {val_surface:.1f} cm² |",
        f"| Span | {val_span:.1f} cm |",
        f"| Aspect ratio | {val_ar:.2f} |",
        f"| Root chord | {val_root_c:.1f} mm |",
        f"| Tip chord | {val_tip_c:.1f} mm |",
        f"| Twist | {val_twist:.2f}° |",
        f"| Wing loading | {wing_loading:.2f} N/m² |",
        f"| Re root | {Re_root:.2e} |",
        f"| Re tip | {Re_tip:.2e} |",
        f"",
        f"---",
        f"",
    ]

    # Stabilizer geometry
    lines += [
        f"## 2. Stabilizer Geometry",
        f"",
        f"| Parameter | Value |",
        f"|:---|:---|",
        f"| Area | {val_s_surface:.1f} cm² |",
        f"| Span | {val_s_span:.1f} cm |",
        f"| Aspect ratio | {val_s_ar:.2f} |",
        f"| Root chord | {val_s_root_c:.1f} mm |",
        f"| Incidence angle | {val_s_twist:.2f}° |",
        f"| Fuselage length | {val_fuselage_length:.1f} cm |",
        f"",
        f"---",
        f"",
    ]

    # Performance
    lines += [
        f"## 3. Performance",
        f"",
        f"| Parameter | Value |",
        f"|:---|:---|",
        f"| L/D ratio | {val_finesse:.2f} |",
        f"| Drag | {val_trainee:.2f} N |",
        f"| Cruise incidence | {val_alpha:.2f}° |",
        f"| CL cruise | {CL_cruise:.3f} |",
        f"| CD cruise | {CD_cruise:.4f} |",
        f"| CL stabilizer | {CL_stab:.3f} |",
        f"",
        f"---",
        f"",
    ]

    # Stability
    lines += [
        f"## 4. Stability & Trim",
        f"",
        f"| Parameter | Value |",
        f"|:---|:---|",
        f"| Static margin | {val_sm:.2f} % |",
        f"| CG position | {val_cg:.1f}% ({val_xcg:.1f} cm from LE) |",
        f"| Stabilizer force | {val_force_stab:.2f} N |",
        f"| Residual moment | {val_moment:.4f} N·m |",
        f"| Tail volume | {val_vh:.4f} |",
        f"",
        f"---",
        f"",
    ]

    # Automatic warnings
    warnings_list = []
    if val_sm > 70:
        warnings_list.append(f"⚠️ High static margin ({val_sm:.1f}%) - risk of reduced maneuverability.")
    if val_sm < 5:
        warnings_list.append(f"⚠️ Very low static margin ({val_sm:.1f}%) - risk of instability.")
    if abs(val_moment) > 5:
        warnings_list.append(f"⚠️ Significant residual moment ({val_moment:.2f} N·m) - check trim.")
    if CL_cruise < 0.08:
        warnings_list.append(f"⚠️ Very low cruise CL ({CL_cruise:.3f}) - area possibly underused.")

    if warnings_list:
        lines += [f"## ⚠️ Warnings", f""]
        for w in warnings_list:
            lines.append(f"- {w}")
        lines.append(f"")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✓ Technical sheet exported: {filepath}")
    return filepath


# =========================================================================================
# ====================   PARAMETRIC OPTIMIZATION - OBJECTIVE FUNCTION   ===================
# =========================================================================================

def evaluate_design(root_naca_profile: str, tip_naca_profile: str,
                    reflex_angle_deg: float, perform_export: bool = False) -> dict:
    """
    Runs IPOPT optimization for a given combination of NACA airfoils.
    Returns a dict with the key metrics, or {"success": False} if infeasible.
    """
    # Reset the environment at each call
    opti       = asb.Opti()
    atmosphere = Atmosphere()

    # ===================== OPTIMIZATION VARIABLES =====================

    # Wing
    span       = opti.variable(init_guess=wing_span_init,   lower_bound=wing_span_bounds[0],   upper_bound=wing_span_bounds[1])
    root_chord = opti.variable(init_guess=root_chord_init,  lower_bound=root_chord_bounds[0],  upper_bound=root_chord_bounds[1])
    tip_chord  = opti.variable(init_guess=tip_chord_init,   lower_bound=tip_chord_bounds[0],   upper_bound=tip_chord_bounds[1])
    twist      = opti.variable(init_guess=twist_init,       lower_bound=twist_bounds[0],       upper_bound=twist_bounds[1])

    # Stabilizer
    s_span       = opti.variable(init_guess=stab_span_init,       lower_bound=stab_span_bounds[0],       upper_bound=stab_span_bounds[1])
    s_root_chord = opti.variable(init_guess=stab_root_chord_init, lower_bound=stab_root_chord_bounds[0], upper_bound=stab_root_chord_bounds[1])
    s_tip_chord  = opti.variable(init_guess=stab_tip_chord_init,  lower_bound=stab_tip_chord_bounds[0],  upper_bound=stab_tip_chord_bounds[1])
    s_twist      = opti.variable(init_guess=s_twist_init,         lower_bound=s_twist_bounds[0],         upper_bound=s_twist_bounds[1])

    # Fuselage & CG
    fuselage_length = opti.variable(init_guess=fuselage_init,       lower_bound=fuselage_bounds[0],  upper_bound=fuselage_bounds[1])
    cg_ratio        = opti.variable(init_guess=cfg["cg_ratio_init"], lower_bound=cfg["cg_range"][0], upper_bound=cfg["cg_range"][1])

    # ===================== GEOMETRIC CONSTRUCTION =====================

    # Airfoils
    af_root = apply_reflex_to_naca(root_naca_profile, reflex_angle_deg) if reflex_angle_deg > 0 else get_airfoil(root_naca_profile)
    af_tip  = apply_reflex_to_naca(tip_naca_profile,  reflex_angle_deg) if reflex_angle_deg > 0 else get_airfoil(tip_naca_profile)

    # --- Main wing (morphing) ---
    wing_xsecs = []
    for i in range(N_sections_wing):
        r        = i / (N_sections_wing - 1)
        af_blend = interpolate_airfoil(af_root, af_tip, r)

        # Elliptic/linear chord distribution
        elliptic_dist  = np.sqrt(1 - r ** 2)
        linear_dist    = 1 - r
        combined_factor = 0.6 * elliptic_dist + 0.4 * linear_dist
        c_dist         = tip_chord + (root_chord - tip_chord) * combined_factor

        # Z position (anhedral + winglet)
        z_base = - (r * (span / 2)) * np.tan(np.radians(wing_anhedral_deg))
        # The winglet only activates near the tip
        if r > 0.8:
            r_winglet = (r - 0.8) / 0.2
            z_winglet = 0.05 * (r_winglet ** 2) * (span / 2)
        else:
            z_winglet = 0.0
        z_pos = z_base + z_winglet

        # X position (sweep + rear tip)
        x_mid_sweep = ((r ** 2) * (span / 2)) * np.tan(np.radians(sweep_deg))
        # small parabolic dynamic tip at wing extremity
        x_mid_tip   = 0.03 * (r ** 3) * (span / 2)
        x_mid_total = x_mid_sweep + x_mid_tip

        # chord centered on this line
        x_le = x_mid_total - 0.5 * c_dist

        wing_xsecs.append(asb.WingXSec(
            xyz_le=[x_le, r * (span / 2), z_pos],
            chord=c_dist,
            twist=twist * r,
            airfoil=af_blend,
        ))

    wing       = asb.Wing(symmetric=True, name="MainWing", xsecs=wing_xsecs)
    mean_chord = wing.mean_geometric_chord()
    X_cg       = cg_ratio * mean_chord

    # --- Fuselage ---
    fuse_xsecs = []
    for i in range(N_sections_fuselage):
        xi_rel = i / (N_sections_fuselage - 1)
        xi     = x_fuselage_start + xi_rel * fuselage_length
        if i == 0:
            width = 0.001
        elif i < 3:
            width = fuselage_diameter * (i / 3)
        else:
            width = fuselage_diameter * (1 - 0.5 * xi_rel ** 3)
        fuse_xsecs.append(asb.FuselageXSec(xyz_c=[xi, 0, 0], radius=width))

    fuselage_obj = asb.Fuselage(name="Fuselage", xsecs=fuse_xsecs)

    # --- Mast ---
    mast_xsecs = []
    for i in range(N_sections_mast):
        r       = i / (N_sections_mast - 1)
        c_local = mast_chord_bot * (1 - r) + chord_at_water_line * r
        z_pos   = -r * profondeur_immersion
        mast_xsecs.append(asb.WingXSec(
            xyz_le=[x_mast, 0, z_pos],
            chord=c_local, twist=0,
            airfoil=asb.Airfoil(mast_profile),
        ))

    mast_obj = asb.Wing(name="Mast_Immersed", symmetric=False, xsecs=mast_xsecs)

    # Mast drag and moment
    c_mast_mean     = (mast_chord_bot + chord_at_water_line) / 2
    S_mast_immersed = c_mast_mean * profondeur_immersion
    q               = 0.5 * atmosphere.density() * cfg["v_cruise"] ** 2
    D_mast          = q * S_mast_immersed * 0.011
    M_mast          = -D_mast * (profondeur_immersion / 2)

    # --- Stabilizer ---
    stab_xsecs = []
    for i in range(N_sections_stab):
        r             = i / (N_sections_stab - 1)
        s_elliptic     = np.sqrt(1 - r ** 2)
        s_c_dist       = s_tip_chord + (s_root_chord - s_tip_chord) * s_elliptic

        # Geometry (dihedral and sweep)
        z_dihedral = (r * (s_span / 2)) * np.tan(np.radians(stab_dihedral_deg))

        # Sweep referenced to the mid-chord of the stab
        x_s_mid_sweep = ((r ** 2) * (s_span / 2)) * np.tan(np.radians(s_sweep_deg))
        x_stab_base   = x_fuselage_start + fuselage_length - 0.10

        # Recomputed stab leading edge
        x_s_le = (x_stab_base + x_s_mid_sweep) - 0.5 * s_c_dist

        stab_xsecs.append(asb.WingXSec(
            xyz_le=[x_s_le, r * (s_span / 2), z_dihedral],
            chord=s_c_dist,
            twist=s_twist,
            airfoil=asb.Airfoil("naca0012"),
        ))

    stab = asb.Wing(symmetric=True, name="Stab", xsecs=stab_xsecs)

    # --- Complete airplane ---
    airplane = asb.Airplane(
        wings=[wing, stab],
        fuselages=[fuselage_obj],
        xyz_ref=np.array([X_cg, 0, 0]),
        s_ref=wing.area(), c_ref=mean_chord, b_ref=wing.span(),
    )

    # ===================== AERODYNAMIC ANALYSIS =====================

    alpha=opti.variable(init_guess=alpha_init, lower_bound=alpha_bounds[0], upper_bound=alpha_bounds[1])

    op_point = asb.OperatingPoint(
        velocity=cfg["v_cruise"],
        alpha=alpha,
        atmosphere=atmosphere,
    )

    aero = asb.AeroBuildup(airplane, op_point).run()

    # Static margin computation
    op_2         = asb.OperatingPoint(
        velocity=cfg["v_cruise"],
        alpha=alpha*1.01,  # Slight increase in angle of attack for the derivative computation
        atmosphere=atmosphere)
    aero_2       = asb.AeroBuildup(airplane, op_2).run()
    dCL          = aero_2["CL"] - aero["CL"]
    dCm          = aero_2["Cm"] - aero["Cm"]
    static_margin = -(dCm / (dCL + 1e-9))

    L  = aero["L"]
    D  = aero["D"]
    M  = aero["Cm"] * (0.5 * atmosphere.density() * cfg["v_cruise"] ** 2 * wing.area() * mean_chord)

    # Rig moment (arm computed dynamically from the optimized CG)
    bras_greement = X_cg - x_mast
    M_greement    = rig_mass * 9.81 * (-bras_greement)

    D_total = D + D_mast
    M_total = M + M_mast + M_greement

    lever_arm  = fuselage_length
    v_h        = (stab.area() * lever_arm) / (wing.area() * mean_chord)
    q_takeoff  = 0.5 * atmosphere.density() * cfg["v_takeoff"] ** 2

    # ===================== CONSTRAINTS =====================

    opti.subject_to([
        # Lift = Weight
        L >= weight,

        # Pitch trim
        M_total / (weight * mean_chord) >= -0.01,
        M_total / (weight * mean_chord) <=  0.01, # Allow 1% residual moment margin to avoid over-constrained designs

        # Areas (scenario targets)
        wing.area() >= cfg["area_target_range"][0],
        wing.area() <= cfg["area_target_range"][1],
        stab.area() >= cfg["stab_area_range"][0],
        stab.area() <= cfg["stab_area_range"][1],

        # Geometry (tip < root)
        tip_chord   <= root_chord   * 0.2,
        s_tip_chord <= s_root_chord * 0.3,

        # Stabilizer load
        aero["wing_aero_components"][1].L <= cfg["stab_load_range"][1],
        aero["wing_aero_components"][1].L >= cfg["stab_load_range"][0],

        # Takeoff
        aero["wing_aero_components"][0].L <= q_takeoff * wing.area() * CL_max_takeoff_approx,

        # Static margin
        static_margin >= cfg["sm_range"][0],
        static_margin <= cfg["sm_range"][1],
    ])

    # Objective: minimize total drag
    opti.minimize(D_total)

    # ===================== SOLVING =====================

    try:
        sol = opti.solve(verbose=False)

        # --- Static margin computation (post-processing) ---
        static_margin = -9.99  # Default error code
        try:
            airplane_sol = sol(airplane)
            alpha_val    = sol(op_point.alpha)
            op_1         = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_val,       atmosphere=atmosphere)
            op_2         = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_val + 0.1, atmosphere=atmosphere)
            aero_1       = asb.AeroBuildup(airplane_sol, op_1).run()
            aero_2       = asb.AeroBuildup(airplane_sol, op_2).run()
            dCL          = aero_2["CL"] - aero_1["CL"]
            dCm          = aero_2["Cm"] - aero_1["Cm"]
            static_margin = -(dCm / (dCL + 1e-9))
        except Exception as e_sm:
            print(f"  Warning SM: {e_sm}")

        # --- Export & detailed report ---
        if perform_export:
            try:
                rho = atmosphere.density()
                mu  = atmosphere.dynamic_viscosity()

                # Value extraction
                val_surface         = float(sol(wing.area())) * 10000
                val_span            = float(sol(span)) * 100
                val_ar              = float(sol(wing.aspect_ratio()))
                val_root_c          = float(sol(root_chord)) * 1000
                val_tip_c           = float(sol(wing_xsecs[-1].chord)) * 1000
                val_twist           = float(sol(twist))
                val_s_surface       = float(sol(stab.area())) * 10000
                val_s_span          = float(sol(s_span)) * 100
                val_s_ar            = float(sol(stab.aspect_ratio()))
                val_s_root_c        = float(sol(s_root_chord)) * 1000
                val_s_twist         = float(sol(s_twist))
                val_fuselage_length = float(sol(fuselage_length)) * 100
                val_finesse         = float(sol(L / D))
                val_trainee         = float(sol(D_total))
                val_alpha           = float(sol(op_point.alpha))
                val_sm              = float(static_margin) * 100
                val_cg              = float(sol(cg_ratio)) * 100
                val_xcg             = float(sol(X_cg)) * 100
                val_force_stab      = float(sol(aero["wing_aero_components"][1].L))
                val_moment          = float(sol(M_total))
                val_vh              = float(sol(v_h))
                wing_loading        = weight / float(sol(wing.area()))
                Re_root             = (rho * cfg["v_cruise"] * float(sol(root_chord))) / mu
                Re_tip              = (rho * cfg["v_cruise"] * float(sol(tip_chord))) / mu
                CL_cruise           = float(sol(L)) / (0.5 * rho * cfg["v_cruise"] ** 2 * float(sol(wing.area())))
                CD_cruise           = float(sol(D)) / (0.5 * rho * cfg["v_cruise"] ** 2 * float(sol(wing.area())))
                CL_stab             = val_force_stab / (0.5 * rho * cfg["v_cruise"] ** 2 * float(sol(stab.area())))

                # Output folder
                now_str      = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir      = os.path.join("outputs", f"{CASE}_{root_naca_profile}_{now_str}")
                airfoils_dir = os.path.join(out_dir, "airfoils")
                os.makedirs(airfoils_dir, exist_ok=True)

                # Markdown technical sheet export
                export_fiche_technique(
                    out_dir=out_dir, root_naca_profile=root_naca_profile,
                    val_surface=val_surface, val_span=val_span, val_ar=val_ar,
                    val_root_c=val_root_c, val_tip_c=val_tip_c, val_twist=val_twist,
                    wing_loading=wing_loading, reflex_angle_deg=reflex_angle_deg,
                    val_s_surface=val_s_surface, val_s_span=val_s_span, val_s_ar=val_s_ar,
                    val_s_root_c=val_s_root_c, val_s_twist=val_s_twist,
                    val_fuselage_length=val_fuselage_length,
                    val_finesse=val_finesse, val_trainee=val_trainee, val_alpha=val_alpha,
                    Re_root=Re_root, Re_tip=Re_tip,
                    CL_cruise=CL_cruise, CD_cruise=CD_cruise, CL_stab=CL_stab,
                    val_sm=val_sm, val_cg=val_cg, val_xcg=val_xcg,
                    val_force_stab=val_force_stab, val_moment=val_moment, val_vh=val_vh,
                )

                # Export .dat airfoils
                def export_profile_dat(af_obj, filename, name_internal):
                    af_obj  = af_obj.repanel(n_points_per_side=50)
                    coords  = af_obj.coordinates
                    x_min, x_max  = np.min(coords[:, 0]), np.max(coords[:, 0])
                    coords[:, 0]  = (coords[:, 0] - x_min) / (x_max - x_min)
                    coords[:, 1]  = coords[:, 1] / (x_max - x_min)
                    idx_le  = np.argmin(coords[:, 0])
                    upper   = coords[:idx_le + 1]
                    lower   = coords[idx_le:]
                    if upper[0, 0] < upper[-1, 0]: upper = upper[::-1]
                    if lower[0, 0] > lower[-1, 0]: lower = lower[::-1]
                    final_coords = np.concatenate([upper, lower[1:]])
                    with open(os.path.join(airfoils_dir, filename), "w") as f:
                        f.write(f"{name_internal}\n")
                        for x, y in final_coords:
                            f.write(f" {x:.6f} {y:.6f}\n")

                wing_optim = sol(wing)
                for i, xsec in enumerate(wing_optim.xsecs):
                    af      = xsec.airfoil
                    af_name = f"wing_sec_{i}"
                    af.name = af_name
                    export_profile_dat(af, f"{af_name}.dat", af_name)

                stab_optim = sol(stab)
                for i, xsec in enumerate(stab_optim.xsecs):
                    af      = xsec.airfoil
                    af_name = f"stab_sec_{i}"
                    af.name = af_name
                    export_profile_dat(af, f"{af_name}.dat", af_name)

                # XFLR5 XML export
                airplane_export = asb.Airplane(
                    wings=[
                        asb.Wing(symmetric=True, name="mainwing",  xsecs=wing_optim.xsecs),
                        asb.Wing(symmetric=True, name="elevator",  xsecs=stab_optim.xsecs),
                        mast_obj,
                    ],
                    fuselages=[fuselage_obj],
                    xyz_ref=sol(airplane).xyz_ref,
                )
                xml_path = os.path.join(out_dir, f"{CASE}_{root_naca_profile}_{now_str}_plane.xml")
                airplane_export.export_XFLR5_xml(xml_path)
                print(f"✓ XFLR5 XML exported: {xml_path}")
                print(f"✓ .dat airfoils generated: {airfoils_dir}")

            except Exception as e_print:
                import traceback
                print(f"\nExport error: {e_print}")
                traceback.print_exc()

        return {
            "success":    True,
            "finesse":    sol(L / D),
            "surface":    sol(wing.area()) * 10000,
            "stab_force": sol(aero["wing_aero_components"][1].L),
            "moment":     sol(M_total),
            "root_chord": sol(root_chord),
            "span":       sol(span),
        }

    except Exception:
        return {"success": False}


# =========================================================================================
# =======================   MAIN EVALUATION LOOP   ========================================
# =========================================================================================

print("\n" + "=" * 60)
print(f"     PARAMETRIC OPTIMIZATION — CASE: {CASE.upper()}")
print("=" * 60 + "\n")

best_run    = None
best_config = None

for c in cambrures:
    for t_root in epaisseurs_root:
        for reflex in angle_reflex:
            t_tip     = t_root - 3
            root_name = f"naca{c}4{t_root:02d}"
            tip_name  = f"naca{c}4{t_tip:02d}"

            print(f"Test: {root_name} → {tip_name} | reflex {reflex}° | ", end="", flush=True)

            res = evaluate_design(root_name, tip_name, reflex, perform_export=False)

            if res["success"]:
                print(f"Convergence: OK | L/D: {res['finesse']:.2f} | Area: {res['surface']:.0f} cm² | Stab: {res['stab_force']:.1f} N")
                if best_run is None or res["finesse"] > best_run["finesse"]:
                    best_run    = res
                    best_config = (root_name, tip_name, reflex)
            else:
                print("Infeasible")

# =========================================================================================
# ================================   FINAL RESULT   =======================================
# =========================================================================================

print("\n" + "=" * 60)
if best_run:
    r_name, t_name, reflex = best_config
    print(f"SELECTED CONFIGURATION:")
    print(f"  Root         : {r_name.upper()}")
    print(f"  Tip          : {t_name.upper()}")
    print(f"  Reflex angle : {reflex}°")
    print("-" * 60)
    print("\n[Generating final files...]\n")
    evaluate_design(r_name, t_name, reflex, perform_export=True)
else:
    print("No airfoil satisfied all constraints.")
print("=" * 60)