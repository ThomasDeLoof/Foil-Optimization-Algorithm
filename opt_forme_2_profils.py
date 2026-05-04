# === Hydrofoil Optimization Script - 2 Profils libres par aile ===
import os
import datetime as dt
import aerosandbox as asb
import aerosandbox.numpy as np
import sys
from pathlib import Path

# --- Configuration Chemins & Atmosphère ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from common.water_atmosphere import Water as Atmosphere


# ========================================================================================
# 1. PARAMÈTRES GÉNÉRAUX ET BORNES (CONTRÔLE TOTAL)
# ========================================================================================
CASE = "wingfoil"  # Options: "wingfoil", "windsurf", "downwind", "pumping"
mass = 78
weight = mass * 9.81

# --- Bornes Géométriques Aile ---
wing_ar_bounds = (8.5, 11.5)
wing_span_bounds = (0.8, 1.2)
wing_span_init_guess = 1
wing_chord_bounds = (0.08, 0.22)
root_chord_init_guess = 0.16
twist_bounds = (-2, 5)
twist_init_guess = 0.0
N_sections_wing = 5  

# --- Stabilisateur ---
stab_span_bounds = (0.30, 0.45)
stab_span_init_guess = 0.38
stab_chord_bounds = (0.05, 0.12)
stab_chord_init_guess = 0.09
fuselage_length_bounds = (0.60, 0.70) # s_xpos
s_xpos_init_guess = 0.60
s_twist_bounds = (-6, 2)
s_twist_init_guess = -1.0
l_stab_min_deportance = -5  # Newtons (Force de déportance minimale du stab en croisière)
N_sections_stab = 4

# --- Fabrication & Sécurité ---
wing_min_thickness_ratio = 0.11
stab_min_thickness_ratio = 0.09
min_thickness_root_abs = 0.018 
CL_max_takeoff = 1.25    

# ========================================================================================
# 2. CONFIGURATION DES SCÉNARIOS
# ========================================================================================
SCENARIOS = {
    "wingfoil":     
        {"v_takeoff": 5.5,
        "v_cruise": 9.5,  
        "area_target": 0.10, 
        "area_margin": 0.015, 
        "alpha_bounds": (-2, 7),
        "sm_range": (0.10, 0.15),      # Stable et rassurant pour les transitions
        "l_stab_range": (-20.0,-8),    # Pilotage calé (pied avant présent)
        "cg_range": (0.22, 0.28),      # Position pieds standard
        "vh_range": (0.38, 0.50),      # Fuselage moyen},
        },
    "windsurf": 
        {"v_takeoff": 7.5, 
        "v_cruise": 13.5, 
        "area_target": 0.085, 
        "area_margin": 0.02, 
        "alpha_bounds": (-2, 6),
        "sm_range": (0.12, 0.18),      # Très stable à haute vitesse (bloqué sur un rail)
        "l_stab_range": (-28.0,-12),   # Forte pression nécessaire pour contrer la voile
        "cg_range": (0.24, 0.30),      # CG légèrement plus reculé
        "vh_range": (0.45, 0.60),      # Long fuselage pour la stabilité directionnelle
        },
    "downwind": 
        {"v_takeoff": 6.5, 
        "v_cruise": 13.0, 
        "area_target": 0.085, 
        "area_margin": 0.03, 
        "alpha_bounds": (-1, 5),
        "sm_range": (0.07, 0.12),      # Plus maniable pour surfer la houle
        "l_stab_range": (-12.0, -2),   # Neutre et réactif (glisse maximale)
        "cg_range": (0.20, 0.26),      # Position pieds vers l'avant
        "vh_range": (0.35, 0.45),      # Fuselage court pour la maniabilité
        },
    "pumping":  
        {"v_takeoff": 3.5, 
        "v_cruise": 6.0,  
        "area_target": 0.160, 
        "area_margin": 0.03, 
        "alpha_bounds": (-2, 5),
        "sm_range": (0.04, 0.08),      # Très réactif au pumping (bascule facile)
        "l_stab_range": (-8, -1),      # Très peu de déportance pour ne pas freiner
        "cg_range": (0.22, 0.28),      # Centrage standard
        "vh_range": (0.30, 0.40),      # Fuselage très court pour l'efficacité du pump
        },

}

cfg = SCENARIOS[CASE]
opti = asb.Opti()
atmosphere = Atmosphere()

# --- Bornes de Centrage (CG) ---
# Le CG est exprimé en ratio de la corde moyenne
cg_ratio_bounds = cfg["cg_range"]
cg_ratio_init_guess = 0.25

# ========================================================================================
# 3. DÉFINITION DES VARIABLES (COMPLEXITÉ PARTAGÉE)
# ========================================================================================

def make_af_vars(name, base_name="naca0012"):
    base_af = asb.KulfanAirfoil(base_name)
    upper_w = opti.variable(
        init_guess=base_af.upper_weights, 
        lower_bound=0.10,  # Empêche le profil de devenir plat ou concave sur le dessus
        upper_bound=0.25   # Empêche les bosses de "baleine"
    )
    lower_w = opti.variable(
        init_guess=base_af.lower_weights, 
        lower_bound=-0.20, # Empêche un ventre trop creux (banane)
        upper_bound=0.05   # Empêche un profil trop épais par le bas
    )
    af = asb.KulfanAirfoil(name=name, upper_weights=upper_w, lower_weights=lower_w)

    dev_penalty = np.sum((upper_w - base_af.upper_weights)**2) + \
                  np.sum((lower_w - base_af.lower_weights)**2)
    return af, dev_penalty

# Profils Maîtres (Emplanture / Saumon)
af_w_root, p1 = make_af_vars("WingRoot", "naca0015")
af_w_tip,  p2 = make_af_vars("WingTip", "naca0010")
af_s_root, p3 = make_af_vars("StabRoot", "naca0012")
af_s_tip,  p4 = make_af_vars("StabTip", "naca0008")
total_smoothness_penalty = p1 + p2 + p3 + p4

# --- Géométrie Aile ---
span = opti.variable(init_guess=wing_span_init_guess, lower_bound=wing_span_bounds[0], upper_bound=wing_span_bounds[1])
root_chord = opti.variable(init_guess=root_chord_init_guess, lower_bound=wing_chord_bounds[0], upper_bound=wing_chord_bounds[1])
twist = opti.variable(init_guess=twist_init_guess, lower_bound=twist_bounds[0], upper_bound=twist_bounds[1])


wing_xsecs = []
for y in range(N_sections_wing):
    pos_ratio = y / (N_sections_wing - 1)
    w_name = f"WING_SEC_{y}"
    base_af = af_w_root if pos_ratio < 0.5 else af_w_tip
    af_section = asb.KulfanAirfoil(
        upper_weights=base_af.upper_weights,
        lower_weights=base_af.lower_weights,
        leading_edge_weight=base_af.leading_edge_weight,
        name=w_name
    )
    wing_xsecs.append(
        asb.WingXSec(
            xyz_le=[0.05 * pos_ratio**2, pos_ratio * (span/2), 0],
            chord=root_chord * np.sqrt(np.maximum(1 - pos_ratio**2, 0.05)),
            twist=twist,
            airfoil=af_section
        )
    )
wing = asb.Wing(symmetric=True, name="MainWing", xsecs=wing_xsecs)

# --- Géometrie stabilisateur ---
s_span = opti.variable(init_guess=stab_span_init_guess, lower_bound=stab_span_bounds[0], upper_bound=stab_span_bounds[1])
s_chord = opti.variable(init_guess=stab_chord_init_guess, lower_bound=stab_chord_bounds[0], upper_bound=stab_chord_bounds[1])
s_xpos = opti.variable(init_guess=s_xpos_init_guess, lower_bound=fuselage_length_bounds[0], upper_bound=fuselage_length_bounds[1])
s_twist = opti.variable(init_guess=s_twist_init_guess, lower_bound=s_twist_bounds[0], upper_bound=s_twist_bounds[1])

stab_xsecs = []
for y in range(N_sections_stab):
    pos_ratio = y / (N_sections_stab - 1)
    s_name = f"STAB_SEC_{y}"
    base_af = af_s_root if pos_ratio < 0.5 else af_s_tip
    af_section = asb.KulfanAirfoil(
        upper_weights=base_af.upper_weights,
        lower_weights=base_af.lower_weights,
        leading_edge_weight=base_af.leading_edge_weight,
        name=s_name
    )
    stab_xsecs.append(
        asb.WingXSec(
            xyz_le=[s_xpos + 0.05 * pos_ratio**2, pos_ratio * (s_span/2), 0],
            chord=s_chord * np.sqrt(np.maximum(1 - pos_ratio**2, 0.05)),
            twist=s_twist,
            airfoil=af_section
        )
    )
stab = asb.Wing(symmetric=True, name="Stab", xsecs=stab_xsecs)

# --- Centre de Gravité ---
cg_ratio = opti.variable(init_guess=cg_ratio_init_guess, lower_bound=cg_ratio_bounds[0], upper_bound=cg_ratio_bounds[1])
mean_chord = wing.mean_geometric_chord()
X_cg = cg_ratio * mean_chord

airplane = asb.Airplane(wings=[wing, stab], xyz_ref=np.array([X_cg, 0, 0]))

# ========================================================================================
# 4. ANALYSE ET CONTRAINTES
# ========================================================================================
v_points = np.array([cfg["v_takeoff"], cfg["v_cruise"]])
alpha = opti.variable(init_guess=np.ones(2)*2, lower_bound=cfg["alpha_bounds"][0], upper_bound=cfg["alpha_bounds"][1])
op_point=asb.OperatingPoint(velocity=v_points, alpha=alpha, atmosphere=atmosphere)
analysis = asb.AeroBuildup(airplane=airplane, op_point=op_point).run()

l_total = analysis["L"]
m_pitch = np.stack(analysis["M_g"])[:, 1]
lever_arm = (s_xpos - X_cg)
wing_results = analysis["wing_aero_components"]
l_wing = wing_results[0].L
l_stab = wing_results[1].L

# Marge Statique
op_plus = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha[1] + 1.0, atmosphere=atmosphere)
analysis_plus = asb.AeroBuildup(airplane=airplane, op_point=op_plus).run()

dCL = (analysis_plus["CL"] - analysis["CL"][1])
dCM = (analysis_plus["Cm"] - analysis["Cm"][1])
static_margin = - dCM / dCL

# Volume de Queue
v_h = (stab.area() * lever_arm) / (wing.area() * mean_chord)

# CONTRAINTES D'OPTIMISATION
opti.subject_to([
    l_total == weight,              # Équilibre de portance
    m_pitch == 0,                   # Équilibre de tangage (Trim)
    
    # Stabilité (Le tunnel Hard)
    static_margin >= cfg["sm_range"][0],
    static_margin <= cfg["sm_range"][1],
    
    # Régulation du stabilisateur
    l_stab[1] <= cfg["l_stab_range"][1],
    l_stab[1] >= cfg["l_stab_range"][0],
    v_h >= cfg["vh_range"][0],
    v_h <= cfg["vh_range"][1],

    # Géométrie & Structure
    wing.area() >= cfg["area_target"] - cfg["area_margin"],
    wing.area() <= cfg["area_target"] + cfg["area_margin"],
    wing.aspect_ratio() >= wing_ar_bounds[0],
    wing.aspect_ratio() <= wing_ar_bounds[1],
    *[x.airfoil.max_thickness() >= wing_min_thickness_ratio for x in wing.xsecs],
    *[x.airfoil.max_thickness() >= stab_min_thickness_ratio for x in stab.xsecs],
    wing.xsecs[0].airfoil.max_thickness() * wing.xsecs[0].chord >= min_thickness_root_abs,
    (l_wing[0] / (0.5 * atmosphere.density() * cfg["v_takeoff"]**2 * wing.area())) <= CL_max_takeoff
])


# FONCTION OBJECTIF
objective = (analysis["D"][1] / 0.035) + 1000.0 * total_smoothness_penalty
opti.minimize(objective)

# Configuration du solveur pour éviter les boucles infinies
opti.solver('ipopt', {'ipopt.max_iter': 500, 'ipopt.tol': 1e-3, 'ipopt.mu_strategy': 'adaptive'})
sol = opti.solve(behavior_on_failure="return_last")

# ========================================================================================
# 5. EXPORTS ET BILAN
# ========================================================================================

airplane_sol = sol(airplane)
wing_sol = airplane_sol.wings[0]
stab_sol = airplane_sol.wings[1]

now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("outputs", f"{CASE}_{now}_{N_sections_wing}sections")
airfoils_dir = os.path.join(output_dir, "airfoils")
os.makedirs(airfoils_dir, exist_ok=True)

def safe_export_xflr5_dat(af, path, name, max_points=200):
    """
    Export certifié XFLR5 : normalisé, trié (Selig), et sans doublons.
    """
    # 1. Extraction et Normalisation (Corde de 0 à 1)
    coords = af.coordinates
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    coords[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min)
    coords[:, 1] = coords[:, 1] / (x_max - x_min)

    # 2. Séparation Extrados/Intrados pour trier proprement
    # On trouve le bord d'attaque (x minimum)
    idx_le = np.argmin(coords[:, 0])
    
    # Upper : du BF (x=1) vers le BA (x=0) -> tri décroissant
    upper = coords[:idx_le + 1]
    upper = upper[upper[:, 0].argsort()[::-1]]
    
    # Lower : du BA (x=0) vers le BF (x=1) -> tri croissant
    lower = coords[idx_le + 1:]
    lower = lower[lower[:, 0].argsort()]

    # 3. Forcer la fermeture du bord de fuite à 1.0
    upper[0, 0] = 1.0
    if len(lower) > 0: lower[-1, 0] = 1.0

    # 4. Fusion et limitation du nombre de points
    final_coords = np.vstack((upper, lower))
    if len(final_coords) > max_points:
        indices = np.linspace(0, len(final_coords) - 1, max_points).astype(int)
        final_coords = final_coords[indices]

    # 5. Écriture format Selig (Nom unique en ligne 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        for x, y in final_coords:
            f.write(f" {x:10.8f} {y:10.8f}\n")

# Export Wing
min_TE_thickness = 0.0004

new_wing_xsecs = []
for i, xsec in enumerate(wing_sol.xsecs):
    af = xsec.airfoil
    if af.TE_thickness * xsec.chord < min_TE_thickness:
        af = af.set_TE_thickness(thickness=min_TE_thickness / xsec.chord)
    af_name = f"wing_sec_{i}"
    af.name = af_name
    safe_export_xflr5_dat(af, os.path.join(airfoils_dir, f"{af_name}.dat"), af_name)
    new_wing_xsecs.append(asb.WingXSec(xyz_le=xsec.xyz_le, chord=xsec.chord, twist=xsec.twist, airfoil=af))

# Export Stab
new_stab_xsecs = []
for i, xsec in enumerate(stab_sol.xsecs):
    af = xsec.airfoil
    af_name = f"stab_sec_{i}"
    af.name = af_name
    safe_export_xflr5_dat(af, os.path.join(airfoils_dir, f"{af_name}.dat"), af_name)
    new_stab_xsecs.append(asb.WingXSec(xyz_le=xsec.xyz_le, chord=xsec.chord, twist=xsec.twist, airfoil=af))

airplane_export = asb.Airplane(wings=[asb.Wing(symmetric=True, name="Main_Wing", xsecs=new_wing_xsecs), asb.Wing(symmetric=True, name="Stabilizer", xsecs=new_stab_xsecs)], xyz_ref=airplane_sol.xyz_ref)
airplane_export.export_XFLR5_xml(os.path.join(output_dir, f"{CASE}_plane.xml"))

# --- Bilan Final ---
print("\n" + "="*50 + f"\nBILAN DESIGN FINAL : {CASE}\n" + "="*50)
print(f"Surface Aile : {sol(wing.area())*10000:.1f} cm² | AR : {sol(wing.aspect_ratio()):.2f}")
print(f"Volume de Queue (Vh) : {sol(v_h):.3f}")
print(f"Position CG : {sol(X_cg)*100:.1f} cm ({sol(X_cg/mean_chord)*100:.1f}% corde)")
print(f"Marge Statique : {sol(static_margin)*100:.2f} %")
print(f"Portance Aile (Cruise) : {sol(l_wing[1]):.1f} N")
print(f"Déportance Stab (Cruise) : {sol(l_stab[1]):.1f} N")
print(f"Envergure Aile : {sol(wing.span()):.3f} m")
print(f"Longueur Fuselage : {sol(s_xpos):.3f} m")
print(f"Finesse Croisière : {sol(analysis['L'][1] / analysis['D'][1]):.2f}")
print(f"Vitesse de Décollage : {cfg['v_takeoff']:.1f} m/s | CL max Décollage : {sol(analysis['CL'][0]):.2f}")
print(f"Dossier de sortie : {output_dir}")
print("="*50)