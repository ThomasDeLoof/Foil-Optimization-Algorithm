# ==================================================================================
# === Hydrofoil Optimization Script - Recherche Paramétrique NACA - Solver IPOPT ===
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

# --- Configuration Chemins & Atmosphère ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from config.water_atmosphere import Water as Atmosphere

# --- Chargement de la configuration physique globale ---
with open("config/parameters.yaml") as f:
    phy = yaml.safe_load(f)

CASE = phy["case"]

# Masses & Poids
rig_mass   = phy["rig"][f"{CASE}_mass_kg"]
total_mass = phy["pilot"]["mass_kg"] + phy["board"]["mass_kg"] + rig_mass
weight     = total_mass * 9.81  # Constante — jamais modifiée dans les fonctions

# Espaces de recherche
cambrures       = phy["search_space"]["cambrures"]
epaisseurs_root = phy["search_space"]["epaisseurs_root"]
angle_reflex    = phy["search_space"]["angle_reflex"]

# Alpha
alpha_bounds = phy["alpha"]["bounds"]
alpha_init   = phy["alpha"]["init"]

# Aile
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

# Mât
mast_length          = phy["mast"]["length"]
profondeur_immersion = phy["mast"]["immersion_depth"]
mast_chord_top       = phy["mast"]["chord_top"]
mast_chord_bot       = phy["mast"]["chord_bot"]
mast_profile         = phy["mast"]["profile"]
x_mast               = phy["mast"]["x_position"]
N_sections_mast      = phy["mast"]["n_sections"]
ratio_immersion      = profondeur_immersion / mast_length
chord_at_water_line  = mast_chord_bot * (1 - ratio_immersion) + mast_chord_top * ratio_immersion

# Stab
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

# --- Chargement du scénario actif ---
with open("config/scenarios.yaml") as f:
    SCENARIOS = yaml.safe_load(f)

if CASE not in SCENARIOS:
    raise ValueError(f"Cas '{CASE}' introuvable dans scenarios.yaml. Options : {list(SCENARIOS.keys())}")
cfg = SCENARIOS[CASE]


# =========================================================================================
# ==============================   PROFILS NACA    ========================================
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
    Applique une déformation 'Reflex' (S-Shape) sur les 30% arrière du profil.
    Permet de contrôler le moment de tangage sans modifier la cambrure globale.
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
# ==============================   EXPORT FICHE TECHNIQUE   ===============================
# =========================================================================================

def export_fiche_technique(
    out_dir, root_naca_profile,
    # Géométrie aile
    val_surface, val_span, val_ar, val_root_c, val_tip_c, val_twist, wing_loading,
    # Géométrie stab
    val_s_surface, val_s_span, val_s_ar, val_s_root_c, val_s_twist, val_fuselage_length,
    # Performances
    val_finesse, val_trainee, val_alpha, Re_root, Re_tip, CL_cruise, CD_cruise, CL_stab,
    # Stabilité
    val_sm, val_cg, val_xcg, val_force_stab, val_moment, val_vh,
):
    """
    Exporte la fiche technique du design optimisé dans un fichier Markdown.
    """
    now_str  = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filepath = os.path.join(out_dir, "fiche_technique.md")
    lines    = []

    # En-tête
    lines += [
        f"# Fiche Technique — {CASE.upper()} | {root_naca_profile.upper()}",
        f"",
        f"*Générée le {now_str}*",
        f"",
        f"---",
        f"",
    ]

    # Conditions de vol
    lines += [
        f"## Conditions de vol",
        f"",
        f"| Paramètre | Valeur |",
        f"|:---|:---|",
        f"| Cas | {CASE} |",
        f"| Poids total | {weight:.1f} N ({weight/9.81:.0f} kg) |",
        f"| Vitesse décollage | {cfg['v_takeoff']} m/s |",
        f"| Vitesse croisière | {cfg['v_cruise']} m/s |",
        f"",
        f"---",
        f"",
    ]

    # Géométrie aile
    lines += [
        f"## 1. Géométrie Aile",
        f"",
        f"| Paramètre | Valeur |",
        f"|:---|:---|",
        f"| Profil | {root_naca_profile.upper()} |",
        f"| Surface | {val_surface:.1f} cm² |",
        f"| Envergure | {val_span:.1f} cm |",
        f"| Allongement | {val_ar:.2f} |",
        f"| Corde emplanture | {val_root_c:.1f} mm |",
        f"| Corde saumon | {val_tip_c:.1f} mm |",
        f"| Vrillage | {val_twist:.2f}° |",
        f"| Charge alaire | {wing_loading:.2f} N/m² |",
        f"| Re emplanture | {Re_root:.2e} |",
        f"| Re saumon | {Re_tip:.2e} |",
        f"",
        f"---",
        f"",
    ]

    # Géométrie stab
    lines += [
        f"## 2. Géométrie Stabilisateur",
        f"",
        f"| Paramètre | Valeur |",
        f"|:---|:---|",
        f"| Surface | {val_s_surface:.1f} cm² |",
        f"| Envergure | {val_s_span:.1f} cm |",
        f"| Allongement | {val_s_ar:.2f} |",
        f"| Corde emplanture | {val_s_root_c:.1f} mm |",
        f"| Calage | {val_s_twist:.2f}° |",
        f"| Longueur fuselage | {val_fuselage_length:.1f} cm |",
        f"",
        f"---",
        f"",
    ]

    # Performances
    lines += [
        f"## 3. Performances",
        f"",
        f"| Paramètre | Valeur |",
        f"|:---|:---|",
        f"| Finesse (L/D) | {val_finesse:.2f} |",
        f"| Traînée | {val_trainee:.2f} N |",
        f"| Incidence croisière | {val_alpha:.2f}° |",
        f"| CL croisière | {CL_cruise:.3f} |",
        f"| CD croisière | {CD_cruise:.4f} |",
        f"| CL stabilisateur | {CL_stab:.3f} |",
        f"",
        f"---",
        f"",
    ]

    # Stabilité
    lines += [
        f"## 4. Stabilité & Équilibre",
        f"",
        f"| Paramètre | Valeur |",
        f"|:---|:---|",
        f"| Marge statique | {val_sm:.2f} % |",
        f"| Position CG | {val_cg:.1f}% ({val_xcg:.1f} cm du BA) |",
        f"| Force stabilisateur | {val_force_stab:.2f} N |",
        f"| Moment résiduel | {val_moment:.4f} N·m |",
        f"| Volume de queue | {val_vh:.4f} |",
        f"",
        f"---",
        f"",
    ]

    # Avertissements automatiques
    warnings_list = []
    if val_sm > 70:
        warnings_list.append(f"⚠️ Marge statique élevée ({val_sm:.1f}%) - risque de manœuvrabilité réduite.")
    if val_sm < 5:
        warnings_list.append(f"⚠️ Marge statique très faible ({val_sm:.1f}%) - risque d'instabilité.")
    if abs(val_moment) > 5:
        warnings_list.append(f"⚠️ Moment résiduel important ({val_moment:.2f} N·m) - vérifier l'équilibre.")
    if CL_cruise < 0.08:
        warnings_list.append(f"⚠️ CL de croisière très bas ({CL_cruise:.3f}) - surface peut-être sous-utilisée.")

    if warnings_list:
        lines += [f"## ⚠️ Avertissements", f""]
        for w in warnings_list:
            lines.append(f"- {w}")
        lines.append(f"")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✓ Fiche technique exportée : {filepath}")
    return filepath


# =========================================================================================
# ====================   OPTIMISATION PARAMÉTRIQUE - FONCTION OBJECTIF   ==================
# =========================================================================================

def evaluate_design(root_naca_profile: str, tip_naca_profile: str,
                    reflex_angle_deg: float, perform_export: bool = False) -> dict:
    """
    Lance l'optimisation IPOPT pour une combinaison de profils NACA donnée.
    Retourne un dict avec les métriques clés, ou {"success": False} si infaisable.
    """
    # Reset de l'environnement à chaque appel
    opti       = asb.Opti()
    atmosphere = Atmosphere()

    # ===================== VARIABLES D'OPTIMISATION =====================

    # Aile
    span       = opti.variable(init_guess=wing_span_init,   lower_bound=wing_span_bounds[0],   upper_bound=wing_span_bounds[1])
    root_chord = opti.variable(init_guess=root_chord_init,  lower_bound=root_chord_bounds[0],  upper_bound=root_chord_bounds[1])
    tip_chord  = opti.variable(init_guess=tip_chord_init,   lower_bound=tip_chord_bounds[0],   upper_bound=tip_chord_bounds[1])
    twist      = opti.variable(init_guess=twist_init,       lower_bound=twist_bounds[0],       upper_bound=twist_bounds[1])

    # Stabilisateur
    s_span       = opti.variable(init_guess=stab_span_init,       lower_bound=stab_span_bounds[0],       upper_bound=stab_span_bounds[1])
    s_root_chord = opti.variable(init_guess=stab_root_chord_init, lower_bound=stab_root_chord_bounds[0], upper_bound=stab_root_chord_bounds[1])
    s_tip_chord  = opti.variable(init_guess=stab_tip_chord_init,  lower_bound=stab_tip_chord_bounds[0],  upper_bound=stab_tip_chord_bounds[1])
    s_twist      = opti.variable(init_guess=s_twist_init,         lower_bound=s_twist_bounds[0],         upper_bound=s_twist_bounds[1])

    # Fuselage & CG
    fuselage_length = opti.variable(init_guess=fuselage_init,       lower_bound=fuselage_bounds[0],  upper_bound=fuselage_bounds[1])
    cg_ratio        = opti.variable(init_guess=cfg["cg_ratio_init"], lower_bound=cfg["cg_range"][0], upper_bound=cfg["cg_range"][1])

    # ===================== CONSTRUCTION GÉOMÉTRIQUE =====================

    # Profils
    af_root = apply_reflex_to_naca(root_naca_profile, reflex_angle_deg) if reflex_angle_deg > 0 else get_airfoil(root_naca_profile)
    af_tip  = apply_reflex_to_naca(tip_naca_profile,  reflex_angle_deg) if reflex_angle_deg > 0 else get_airfoil(tip_naca_profile)

    # --- Aile principale (morphing) ---
    wing_xsecs = []
    for i in range(N_sections_wing):
        r        = i / (N_sections_wing - 1)
        af_blend = interpolate_airfoil(af_root, af_tip, r)

        # Distribution de corde elliptique/linéaire
        elliptic_dist  = np.sqrt(1 - r ** 2)
        linear_dist    = 1 - r
        combined_factor = 0.6 * elliptic_dist + 0.4 * linear_dist
        c_dist         = tip_chord + (root_chord - tip_chord) * combined_factor

        # Fermeture progressive au saumon
        r_start_tip = 0.90
        if r > r_start_tip:
            rt             = (r - r_start_tip) / (1 - r_start_tip)
            closure_factor = np.sqrt(1 - rt ** 2)
            c_dist         = (c_dist - 0.01) * closure_factor + 0.01

        # Position Z (anhedral + winglet)
        z_base    = -((r ** 2) * span / 2) * np.tan(np.radians(wing_anhedral_deg))
        z_winglet = -0.035 * (r ** 6)
        z_pos     = z_base + z_winglet

        # Position X (sweep + pointe arrière)
        x_sweep_base = ((r ** 2.5) * span / 2) * np.tan(np.radians(sweep_deg))
        x_sweep_tip  = (root_chord - c_dist) if r > r_start_tip else 0
        x_total      = x_sweep_base + x_sweep_tip

        wing_xsecs.append(asb.WingXSec(
            xyz_le=[x_total, r * (span / 2), z_pos],
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

    # --- Mât ---
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

    # Traînée et moment du mât
    c_mast_mean     = (mast_chord_bot + chord_at_water_line) / 2
    S_mast_immersed = c_mast_mean * profondeur_immersion
    q               = 0.5 * atmosphere.density() * cfg["v_cruise"] ** 2
    D_mast          = q * S_mast_immersed * 0.011
    M_mast          = -D_mast * (profondeur_immersion / 2)

    # --- Stabilisateur ---
    stab_xsecs = []
    for i in range(N_sections_stab):
        r             = i / (N_sections_stab - 1)
        elliptic_dist = np.sqrt(1 - r ** 2)
        linear_dist   = 1 - r
        c_dist        = s_tip_chord + (s_root_chord - s_tip_chord) * (0.8 * linear_dist + 0.2 * elliptic_dist)

        r_start_tip = 0.80
        if r > r_start_tip:
            rt             = (r - r_start_tip) / (1 - r_start_tip)
            closure_factor = np.sqrt(1 - rt ** 2)
            c_dist         = (c_dist - 0.01) * closure_factor + 0.01

        z_dihedral = (r ** (3 / 2) * s_span / 2) * np.tan(np.radians(stab_dihedral_deg))
        x_sweep    = 0.9 * (s_root_chord - c_dist) + ((r ** 2.5) * s_span / 2) * np.tan(np.radians(s_sweep_deg))
        x_stab     = x_fuselage_start + fuselage_length - 0.10

        stab_xsecs.append(asb.WingXSec(
            xyz_le=[x_stab + x_sweep, r * (s_span / 2), z_dihedral],
            chord=c_dist, twist=s_twist,
            airfoil=asb.Airfoil("naca0012"),
        ))

    stab = asb.Wing(symmetric=True, name="Stab", xsecs=stab_xsecs)

    # --- Avion complet ---
    airplane = asb.Airplane(
        wings=[wing, stab],
        fuselages=[fuselage_obj],
        xyz_ref=np.array([X_cg, 0, 0]),
        s_ref=wing.area(), c_ref=mean_chord, b_ref=wing.span(),
    )

    # ===================== ANALYSE AÉRODYNAMIQUE =====================

    op_point = asb.OperatingPoint(
        velocity=cfg["v_cruise"],
        alpha=opti.variable(init_guess=alpha_init, lower_bound=alpha_bounds[0], upper_bound=alpha_bounds[1]),
        atmosphere=atmosphere,
    )

    aero = asb.AeroBuildup(airplane, op_point).run()

    L  = aero["L"]
    D  = aero["D"]
    M  = aero["Cm"] * (0.5 * atmosphere.density() * cfg["v_cruise"] ** 2 * wing.area() * mean_chord)

    # Moment du gréement (bras calculé dynamiquement depuis le CG optimisé)
    bras_greement = X_cg - x_mast
    M_greement    = rig_mass * 9.81 * (-bras_greement)

    D_total = D + D_mast
    M_total = M + M_mast + M_greement  # CORRIGÉ : M_mast réintégré

    lever_arm  = fuselage_length
    v_h        = (stab.area() * lever_arm) / (wing.area() * mean_chord)
    q_takeoff  = 0.5 * atmosphere.density() * cfg["v_takeoff"] ** 2

    # ===================== CONTRAINTES =====================

    opti.subject_to([
        # Portance = Poids
        L >= weight,

        # Équilibre de tangage
        M_total / (weight * mean_chord) >= -0.1,
        M_total / (weight * mean_chord) <=  0.1,

        # Surfaces (cibles scénario)
        wing.area() >= cfg["area_target_range"][0],
        wing.area() <= cfg["area_target_range"][1],
        stab.area() >= cfg["stab_area_range"][0],
        stab.area() <= cfg["stab_area_range"][1],

        # Géométrie (saumon < root)
        tip_chord   <= root_chord   * 0.2,
        s_tip_chord <= s_root_chord * 0.3,

        # Charge stabilisateur
        aero["wing_aero_components"][1].L <= cfg["stab_load_range"][1],
        aero["wing_aero_components"][1].L >= cfg["stab_load_range"][0],

        # Décollage
        aero["wing_aero_components"][0].L <= q_takeoff * wing.area() * CL_max_takeoff_approx,

        # Volume de queue
        v_h >= cfg["vh_range"][0],
        v_h <= cfg["vh_range"][1],
    ])

    # Objectif : minimiser la traînée totale
    opti.minimize(D_total)

    # ===================== RÉSOLUTION =====================

    try:
        sol = opti.solve(verbose=False)

        # --- Calcul de la Marge Statique (post-traitement) ---
        static_margin = -9.99  # Code d'erreur par défaut
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
            print(f"  Warning SM : {e_sm}")

        # --- Export & Bilan détaillé ---
        if perform_export:
            try:
                rho = atmosphere.density()
                mu  = atmosphere.dynamic_viscosity()

                # Extraction des valeurs
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

                # Dossier de sortie
                now_str      = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir      = os.path.join("outputs", f"{CASE}_{root_naca_profile}_{now_str}")
                airfoils_dir = os.path.join(out_dir, "airfoils")
                os.makedirs(airfoils_dir, exist_ok=True)

                # Export fiche technique Markdown
                export_fiche_technique(
                    out_dir=out_dir, root_naca_profile=root_naca_profile,
                    val_surface=val_surface, val_span=val_span, val_ar=val_ar,
                    val_root_c=val_root_c, val_tip_c=val_tip_c, val_twist=val_twist,
                    wing_loading=wing_loading,
                    val_s_surface=val_s_surface, val_s_span=val_s_span, val_s_ar=val_s_ar,
                    val_s_root_c=val_s_root_c, val_s_twist=val_s_twist,
                    val_fuselage_length=val_fuselage_length,
                    val_finesse=val_finesse, val_trainee=val_trainee, val_alpha=val_alpha,
                    Re_root=Re_root, Re_tip=Re_tip,
                    CL_cruise=CL_cruise, CD_cruise=CD_cruise, CL_stab=CL_stab,
                    val_sm=val_sm, val_cg=val_cg, val_xcg=val_xcg,
                    val_force_stab=val_force_stab, val_moment=val_moment, val_vh=val_vh,
                )

                # Export profils .dat
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

                # Export XML XFLR5
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
                print(f"✓ XML XFLR5 exporté   : {xml_path}")
                print(f"✓ Profils .dat générés : {airfoils_dir}")

            except Exception as e_print:
                import traceback
                print(f"\nErreur export : {e_print}")
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
# =======================   BOUCLE D'ÉVALUATION PRINCIPALE   ==============================
# =========================================================================================

print("\n" + "=" * 60)
print(f"     OPTIMISATION PARAMÉTRIQUE — CAS : {CASE.upper()}")
print("=" * 60 + "\n")

best_run    = None
best_config = None

for c in cambrures:
    for t_root in epaisseurs_root:
        for reflex in angle_reflex:
            t_tip     = t_root - 3
            root_name = f"naca{c}4{t_root:02d}"
            tip_name  = f"naca{c}4{t_tip:02d}"

            print(f"Test : {root_name} → {tip_name} | reflex {reflex}° | ", end="", flush=True)

            res = evaluate_design(root_name, tip_name, reflex, perform_export=False)

            if res["success"]:
                print(f"OK | Finesse : {res['finesse']:.2f} | Surf : {res['surface']:.0f} cm² | Stab : {res['stab_force']:.1f} N")
                if best_run is None or res["finesse"] > best_run["finesse"]:
                    best_run    = res
                    best_config = (root_name, tip_name, reflex)
            else:
                print("Infaisable")

# =========================================================================================
# ================================   RÉSULTAT FINAL   =====================================
# =========================================================================================

print("\n" + "=" * 60)
if best_run:
    r_name, t_name, reflex = best_config
    print(f"CONFIGURATION RETENUE :")
    print(f"  Emplanture   : {r_name.upper()}")
    print(f"  Saumon       : {t_name.upper()}")
    print(f"  Angle Reflex : {reflex}°")
    print("-" * 60)
    print("\n[Génération des fichiers finaux...]\n")
    evaluate_design(r_name, t_name, reflex, perform_export=True)
else:
    print("Aucun profil n'a satisfait toutes les contraintes.")
print("=" * 60)