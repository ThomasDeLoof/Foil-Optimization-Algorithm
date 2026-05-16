# === Hydrofoil Optimization Script - Recherche Paramétrique NACA ===
import os
import datetime as dt
import aerosandbox as asb
import aerosandbox.numpy as np
import sys
from pathlib import Path
import time

# --- Configuration Chemins & Atmosphère ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from water_atmosphere import Water as Atmosphere

# MODÈLE DE RÉFÉRENCE wingfoil : "AFS Enduro 1100"
#Surface	1100 cm2
#Envergure	1100 mm
#Aspect Ratio	11
#Corde	136 mm
#Epaisseur	16,2 mm
#Finesse	15
#Surface stab : 230 cm² 
#Envergure stab : 42 cm.
# fuselage 72-75 cm
# Position CG : 35% à 42% de la corde.
# Marge Statique (SM) : 50% à 70%.
# finesse entre 12 et 17 pour un foil de freeride.

# ========================================================================================
# 1. PARAMÈTRES GÉNÉRAUX ET BORNES (DONNÉES D'ENTRÉE)
# ========================================================================================
CASE = "windsurf"  # Options: "wingfoil", "windsurf", "downwind", "pumping"
mass = 80
weight = mass * 9.81
# En général, on navigue avec 45-50 cm de mât dans l'eau.
profondeur_immersion = 0.45  # En mètres

# Espaces de recherche
cambrures = [0,1,2]        # 0=Symétrique ... 3=Très cambré
epaisseurs_root = [15]   # Épaisseur emplanture
angle_reflex = [0.0,1.0,2.0,3.0] # Angle de reflex : Standard (0°), Léger (1°), Moyen (2°), Fort (3°)
alpha_bounds = (-1.0, 10.0)      # Angle d'attaque de croisière
alpha_init = 3.0

# --- Bornes Géométriques Aile (Variables d'optimisation) ---
# On verrouille ici les dimensions "idéales" pour un foil de performance
wing_span_bounds = (0.90, 1.3)      # Envergure autour d'1m
wing_span_init = 1                   
root_chord_bounds = (0.12, 0.28)     # Corde moyenne performante
root_chord_init = 0.18
tip_chord_bounds = (0.02, 0.12)      # Saumon plus fin
tip_chord_init = 0.03
twist_bounds = (-4.0, -1.3)           # Vrillage autorisé
twist_init = -1.6
sweep_deg = 18.0  # 18 degrés standard esthétique et efficace (anti-algues)
wing_anhedral_deg = 5.0              # Esthétique et stabilité spirale
N_sections_wing = 8  # Nombre de sections de l'aile pour la modélisation

# Paramètres géométriques du fuselage (Tube carbone)
fuselage_bounds = (0.80,0.115)  # Longueur du fuselage entre 65 et 90 cm
fuselage_init = 0.70
#fuselage_length = 0.70      # Fuselage fixe à 70 cm du BA de l'aile
fuselage_diameter = 0.035    # 3 cm de diamètre moyen
x_fuselage_start = 0.10              # Le nez est 10cm derrière le bord d'attaque de l'aile
N_sections_fuselage = 6  # Nombre de sections du fuselage pour la modélisation

# Mât + Pilote (Surface frontale approx)
mast_length = 0.82     # 82 cm
mast_chord_top = 0.135   
mast_chord_bot = 0.110   
ratio_immersion = profondeur_immersion / mast_length
chord_at_water_line = mast_chord_bot * (1 - ratio_immersion) + mast_chord_top * ratio_immersion
mast_profile = "naca0015" # Profil épais et symétrique pour la rigidité
x_mast = 0.25      # Décalage en X du mât par rapport au BA de l'aile
N_sections_mast = 4

# --- Stabilisateur (Configuration pour fuselage long) ---
stab_span_bounds = (0.32, 0.56)
stab_span_init = 0.42
stab_root_chord_bounds = (0.06, 0.20)
stab_root_chord_init = 0.10
stab_tip_chord_bounds = (0.015, 0.07)
stab_tip_chord_init = 0.02
s_twist_bounds = (-5.0, 0.5)         # Angle de calage
s_twist_init = -3
s_sweep_deg = 8.0  # degré, sweep du stab
stab_dihedral_deg = 4.0  # degré, angle de dièdre du stab
N_sections_stab = 5

# --- Fabrication & Sécurité ---
# Limites physiques pour guider le solveur
CL_max_takeoff_approx = 1.1          # Estimation moyenne pour un profil cambré

# Chargement de la configuration active
import yaml
with open("scenarios.yaml") as f:
    SCENARIOS = yaml.safe_load(f)
cfg = SCENARIOS[CASE]

# --- HELPER: Chargement Profil ---
def get_airfoil(name):
    try:
        return asb.Airfoil(name).repanel(50)
    except:
        return asb.Airfoil("naca0012").repanel(50) # Fallback

def interpolate_airfoil(af1, af2, r):
    coords = (1 - r) * af1.coordinates + r * af2.coordinates
    return asb.Airfoil("blend", coordinates=coords).repanel(50)

def apply_reflex_to_naca(naca_name, reflex_angle_deg):
    """
    Charge un profil NACA et applique une déformation 'Reflex' (S-Shape)
    sur les 30 derniers % de la corde.
    """
    # 1. Charger le profil de base
    base_af = asb.Airfoil(naca_name)
    x = base_af.coordinates[:, 0]
    y = base_af.coordinates[:, 1]
    # 2. Définir la zone de pliage (à partir de 70% de la corde)
    x_start_bend = 0.7
    mask = x > x_start_bend
    # 3. Appliquer la formule quadratique (courbe douce vers le haut)
    # L'angle est converti en radians
    if reflex_angle_deg > 0:
        bend_strength = np.radians(reflex_angle_deg) * 1.5
        y[mask] += (x[mask] - x_start_bend)**2 * bend_strength
    # 4. Retourner le nouvel objet Airfoil
    new_name = f"{naca_name}_Reflex_{reflex_angle_deg}deg"
    return asb.Airfoil(name=new_name, coordinates=np.stack((x, y), axis=1)).repanel(50)

# ========================================================================================
# 3. MOTEUR D'OPTIMISATION (FONCTION)
# ========================================================================================

def evaluate_design(root_naca_profile, tip_naca_profile, reflex_angle_deg, perform_export=False):
    """
    Lance l'optimisation en utilisant les paramètres globaux ci-dessus.
    Construit une aile morphing entre root_naca_profile et tip_naca_profile.
    """
    global weight
    
    # Initialisation de l'environnement d'optimisation (Reset à chaque appel)
    opti = asb.Opti()
    atmosphere = Atmosphere()
    
    # --- DÉFINITION DES VARIABLES (Basées sur les Bornes Globales) ---
    # === Aile principale ===
    span = opti.variable(init_guess=wing_span_init, lower_bound=wing_span_bounds[0], upper_bound=wing_span_bounds[1])

    # Emplanture
    root_chord = opti.variable(init_guess=root_chord_init, lower_bound=root_chord_bounds[0], upper_bound=root_chord_bounds[1])
    tip_chord = opti.variable(init_guess=tip_chord_init, lower_bound=tip_chord_bounds[0], upper_bound=tip_chord_bounds[1])
    twist = opti.variable(init_guess=twist_init, lower_bound=twist_bounds[0], upper_bound=twist_bounds[1])

    # === Stabilisateur horizontal ===
    s_span = opti.variable(init_guess=stab_span_init, lower_bound=stab_span_bounds[0], upper_bound=stab_span_bounds[1])
    s_root_chord = opti.variable(init_guess=stab_root_chord_init, lower_bound=stab_root_chord_bounds[0], upper_bound=stab_root_chord_bounds[1])
    s_tip_chord = opti.variable(init_guess=stab_tip_chord_init, lower_bound=stab_tip_chord_bounds[0], upper_bound=stab_tip_chord_bounds[1])
    s_twist = opti.variable(init_guess=s_twist_init, lower_bound=s_twist_bounds[0], upper_bound=s_twist_bounds[1])
    
    # Centrage (CG)
    fuselage_length = opti.variable(init_guess=fuselage_init, lower_bound=fuselage_bounds[0], upper_bound=fuselage_bounds[1])
    cg_ratio = opti.variable(init_guess=cfg["cg_ratio_init"], lower_bound=cfg["cg_range"][0], upper_bound=cfg["cg_range"][1])

    # --- CONSTRUCTION GÉOMÉTRIQUE ---
    # Chargement des profils d'entrée
    af_root = get_airfoil(root_naca_profile)
    af_tip = get_airfoil(tip_naca_profile)
    if reflex_angle_deg > 0:
        af_root = apply_reflex_to_naca(root_naca_profile, reflex_angle_deg)
        af_tip = apply_reflex_to_naca(tip_naca_profile, reflex_angle_deg)
    
    # Aile avec Morphing
    wing_xsecs = []
    for i in range(N_sections_wing):
        r = i / (N_sections_wing - 1)
        # Interpolation Profil
        af_blend = interpolate_airfoil(af_root, af_tip, r)
        # Forme elliptique de la corde
        elliptic_dist = np.sqrt(1 - r**2)
        linear_dist = 1 - r
        blend_factor = 0.6 
        combined_factor = (blend_factor * elliptic_dist) + ((1 - blend_factor) * linear_dist)
        c_dist = tip_chord + (root_chord - tip_chord) * combined_factor
        # Position finale Z
        r_start_tip = 0.90 
        rt = (r - r_start_tip) / (1 - r_start_tip)
        if r > r_start_tip:
            closure_factor = np.sqrt(1 - rt**2) 
            c_dist = (c_dist - 0.01) * closure_factor + 0.01  # Corde minimale de 10mm au bout
        # Position finale Z avec winglet    
        z_base = - ((r**2) * span/2) * np.tan(np.radians(wing_anhedral_deg))
        winglet_height = 0.035 # Hauteur du winglet (2.5 cm)
        z_winglet = -winglet_height * (r**6) # Winglet très progressif
        z_pos = z_base + z_winglet
        # On augmente le sweep à la toute fin pour accentuer l'effet "Pointe arrière"
        x_sweep_base = ((r**2.5) * span / 2) * np.tan(np.radians(sweep_deg))
        if r > r_start_tip:
             x_sweep_tip = root_chord - c_dist
        else:
             x_sweep_tip = 0
        x_total = x_sweep_base + x_sweep_tip
        wing_xsecs.append(asb.WingXSec(
            xyz_le=[x_total, r * (span / 2), z_pos],
            chord=c_dist,
            twist=twist * r , 
            airfoil=af_blend
        ))
    wing = asb.Wing(symmetric=True, name="MainWing", xsecs=wing_xsecs)

    mean_chord = wing.mean_geometric_chord()
    X_cg = cg_ratio * mean_chord

    # fuselage (forme de cigare simplifié)
    fuse_xsecs = []
    for i in range(N_sections_fuselage):
        xi_rel = i / (N_sections_fuselage - 1) # 0 à 1
        # Position X absolue
        xi = x_fuselage_start + xi_rel * fuselage_length
        # Rayon variable (Forme de goutte d'eau inversée)
        if i == 0:
            # DÉBUT : On commence "pointu" pour simuler la sortie de l'aile proprement
            width = 0.001 
        elif i < 3:
            # TRANSITION : On grossit vite pour atteindre le diamètre du tube
            width = fuselage_diameter * (i / 3)
        else:
            # Petit effilement à la fin
            width = fuselage_diameter * (1 - 0.5 * xi_rel**3)
            
        fuse_xsecs.append(asb.FuselageXSec(
            xyz_c=[xi, 0, 0], 
            radius=width
        ))
    fuselage_obj = asb.Fuselage(
        name="Fuselage",
        xsecs=fuse_xsecs
    )

    # Mat
    mast_xsecs = []
    for i in range(N_sections_mast):
        r = i / (N_sections_mast - 1)
        # Interpolation entre bas du mât et ligne de flottaison
        c_local = mast_chord_bot * (1 - r) + chord_at_water_line * r
        # Z part de 0 (fuselage) et monte vers le négatif ou positif selon ton repère
        # ATTENTION : Dans ASB standard, Z est "Down". Donc le mât va vers le HAUT = Z Négatif.
        z_pos = - r * profondeur_immersion
        mast_xsecs.append(asb.WingXSec(
            xyz_le=[x_mast, 0, z_pos], # Y=0 (centré), Z monte
            chord=c_local,
            twist=0,
            airfoil=asb.Airfoil(mast_profile)
        ))

    mast_obj = asb.Wing(
        name="Mast_Immersed",
        symmetric=False, # Important : ce n'est pas une aile symétrique gauche/droite, c'est un objet unique
        xsecs=mast_xsecs
    )
    # Surface mouillée estimée (Corde moyenne * Profondeur)
    c_mast_mean = (mast_chord_bot + chord_at_water_line) / 2
    S_mast_immersed = c_mast_mean * profondeur_immersion
    # Cd du mât (Profil épais naca0015 + surface rugueuse + ventilation), Valeur conservatrice : 0.010 - 0.012
    Cd_mast = 0.011 
    q = 0.5 * atmosphere.density() * cfg["v_cruise"]**2
    D_mast = q * S_mast_immersed * Cd_mast
    # Le centre de poussée du mât est environ au milieu de la partie immergée
    z_center_pressure_mast = - (profondeur_immersion / 2)
    # Dans le repère avion standard : Drag est positif vers l'arrière.
    M_mast = - D_mast * abs(z_center_pressure_mast)
    
    # Stab
    stab_xsecs = []
    for i in range(N_sections_stab):
        r = i / (N_sections_stab - 1)
        # Forme de corde mixte elliptique/linéaire
        elliptic_dist = np.sqrt(1 - r**2) # Cercle parfait
        linear_dist = 1 - r               # Triangle parfait
        c_dist = s_tip_chord + (s_root_chord - s_tip_chord) * (0.8 * linear_dist + 0.2 * elliptic_dist)
        
        r_start_tip = 0.80 
        rt = (r - r_start_tip) / (1 - r_start_tip)
        if r > r_start_tip:
            closure_factor = np.sqrt(1 - rt**2) 
            c_dist = (c_dist - 0.01) * closure_factor + 0.01  # Corde minimale de 10mm au bout

        z_dihedral = (r**(3/2) * s_span/2) * np.tan(np.radians(stab_dihedral_deg)) # Dihedral stab
        x_sweep = 0.9*(s_root_chord-c_dist)+ ((r**2.5) * s_span / 2) * np.tan(np.radians(s_sweep_deg))

        # Le stab est souvent posé 5-10cm avant la fin du fuselage
        x_stab = x_fuselage_start + fuselage_length - 0.10
        stab_xsecs.append(asb.WingXSec(
            xyz_le=[x_stab + x_sweep, r*(s_span/2), z_dihedral],
            chord=c_dist, twist=s_twist, airfoil=asb.Airfoil("naca0012")
        ))
    stab = asb.Wing(symmetric=True, name="Stab", xsecs=stab_xsecs)
    
    # Avion
    mean_chord = wing.mean_geometric_chord()
    airplane = asb.Airplane(
        wings=[wing, stab],
        fuselages=[fuselage_obj],
        xyz_ref=np.array([cg_ratio * mean_chord, 0, 0]),
        s_ref=wing.area(), c_ref=mean_chord, b_ref=wing.span()
    )

    # --- ANALYSE AÉRODYNAMIQUE ---
    # Point de vol (Croisière)
    op_point = asb.OperatingPoint(
        velocity=cfg["v_cruise"],
        alpha=opti.variable(init_guess=alpha_init, lower_bound=alpha_bounds[0], upper_bound=alpha_bounds[1]),
        atmosphere=atmosphere
    )
    
    aero = asb.AeroBuildup(airplane, op_point).run() # Analyse Aéro complète sans le mat (à part pour éviter les collisions)
    
    # Forces & Moments
    L = aero["L"]
    D = aero["D"]
    # Normalisation du moment
    M = aero["Cm"] * (0.5 * atmosphere.density() * cfg["v_cruise"]**2 * wing.area() * mean_chord)
    if CASE == "windsurf":
        # On ajoute un moment piqueur supplémentaire pour simuler le poids non négligeale du gréement
        m_greement = 10.0  # kg, estimation grossière
    else: m_greement = 0.0  # gréement négligeable ou inexistant
    weight += m_greement * 9.81
    M_greement = m_greement * 9.81 * (-0.9)  # Moment dû au gréement (bras de levier d'environ 90 cm)
    L_total = L  # Le mât ne porte pas (symétrique)
    D_total = D + D_mast
    M_total = M + M_mast + M_greement # On ajoute le moment piqueur du mât et du gréement
    
    lever_arm = fuselage_length 
    v_h = (stab.area() * lever_arm) / (wing.area() * mean_chord)
    q_takeoff = 0.5 * atmosphere.density() * cfg["v_takeoff"]**2
    
    # --- CONTRAINTES ---
    opti.subject_to([
        # 1. Portance = Poids
        L_total == weight,
        
        # 2. Équilibre Tangage (Moment nul)
        M_total / (weight * mean_chord) >= -0.01,
        M_total / (weight * mean_chord) <= 0.01,
        
        # 3. Surface (Cible Scénario)
        wing.area() >= cfg["area_target_range"][0],
        wing.area() <= cfg["area_target_range"][1],
        stab.area() >= cfg["stab_area_range"][0],
        stab.area() <= cfg["stab_area_range"][1],
        tip_chord <= root_chord * 0.2, # Force le saumon à faire max 20% du root
        s_tip_chord <= s_root_chord * 0.3, # Force le saumon stab à faire max 30% du root
        
        # 4. Charge Stabilisateur (Déportance contrôlée)
        aero["wing_aero_components"][1].L <= cfg["stab_load_range"][1],
        aero["wing_aero_components"][1].L >= cfg["stab_load_range"][0],
        
        # 5. Décollage (Portance max estimée)
        aero["wing_aero_components"][0].L <= q_takeoff * wing.area() * CL_max_takeoff_approx,

        # 6. Volume de queue
        v_h >= cfg["vh_range"][0],
        v_h <= cfg["vh_range"][1],
    ])
    
    # Objectif : Minimiser la Traînée
    opti.minimize(D_total+100*v_h-100*cg_ratio)
    
    # --- RÉSOLUTION ---
    try:
        sol = opti.solve(verbose=False)
        
        # Extraction Résultats
        res = {
            "success": True,
            "finesse": sol(L/D),
            "surface": sol(wing.area()) * 10000, # cm2
            "stab_force": sol(aero["wing_aero_components"][1].L),
            "moment": sol(M),
            "root_chord": sol(root_chord),
            "span": sol(span)
        }

        # --- Calcul de la Marge statique ---
        static_margin = 0.0 # Valeur par défaut si échec
        try:
            # On recrée l'avion figé
            airplane_sol = sol(airplane)
            alpha_val = sol(op_point.alpha)
            # Point 1
            op_1 = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_val, atmosphere=atmosphere)
            aero_1 = asb.AeroBuildup(airplane_sol, op_1).run()
            # Point 2 (+0.1 deg)
            op_2 = asb.OperatingPoint(velocity=cfg["v_cruise"], alpha=alpha_val + 0.1, atmosphere=atmosphere)
            aero_2 = asb.AeroBuildup(airplane_sol, op_2).run()
            dCL = aero_2["CL"] - aero_1["CL"]
            dCm = aero_2["Cm"] - aero_1["Cm"]
            static_margin = - (dCm / (dCL + 1e-9))
        except Exception as e_sm:
            # Si le calcul SM plante, on ne tue pas le programme, on affiche juste l'erreur
            print(f"Warning SM: {e_sm}")
            static_margin = -9.99 # Code d'erreur visible
        
        # --- EXPORT ET BILAN DÉTAILLÉ (Uniquement pour le meilleur profil) ---
        if perform_export:
            # 1. Calculs complémentaires
            airplane_sol = sol(airplane)
            # Physique des fluides
            rho = atmosphere.density()
            mu = atmosphere.dynamic_viscosity() # Viscosité dynamique eau (Pa.s)
            
            # Reynolds à l'emplanture
            Re_root = (rho * cfg["v_cruise"] * sol(root_chord)) / mu
            # Reynolds au saumon
            val_tip_chord_m = float(sol(tip_chord))
            Re_tip = (rho * cfg["v_cruise"] * (val_tip_chord_m)) / mu
            
            # Coefficients Aéro
            CL_cruise = sol(L) / (0.5 * rho * cfg["v_cruise"]**2 * sol(wing.area()))
            CD_cruise = sol(D) / (0.5 * rho * cfg["v_cruise"]**2 * sol(wing.area()))
            
            # Stab : Coefficient de portance local (Est-ce qu'il force trop ?)
            CL_stab = sol(aero["wing_aero_components"][1].L) / (0.5 * rho * cfg["v_cruise"]**2 * sol(stab.area()))
            
            # Charge Alaire (Wing Loading)
            wing_loading = weight / sol(wing.area()) # N/m2

            # --- EXPORT ET BILAN DÉTAILLÉ  ---
        if perform_export:
            # 1. Extraction des valeurs (On sort tout du 'sol' avant d'afficher)
            try:
                # 1. Extraction et Conversion en FLOAT (C'est le secret pour éviter l'erreur)
                # On utilise float() partout pour "nettoyer" les tableaux NumPy
                
                val_surface = float(sol(wing.area())) * 10000
                val_span    = float(sol(span)) * 100
                val_ar      = float(sol(wing.aspect_ratio()))
                val_root_c  = float(sol(root_chord)) * 1000
                val_s_surface = float(sol(stab.area())) * 10000
                val_s_span    = float(sol(s_span)) * 100
                val_s_ar      = float(sol(stab.aspect_ratio()))
                val_s_root_c  = float(sol(s_root_chord)) * 1000
                
                # Pour le saumon, on sécurise l'accès à la dernière section
                chord_tip_raw = sol(wing_xsecs[-1].chord)
                val_tip_c   = float(chord_tip_raw) * 1000
                
                val_twist   = float(sol(twist))
                val_s_twist   = float(sol(s_twist))
                
                val_finesse = float(sol(L/D))
                val_trainee = float(sol(D))
                val_alpha   = float(sol(op_point.alpha))
                
                # Stabilité
                val_sm      = float(static_margin) * 100
                val_cg      = float(sol(cg_ratio)) * 100
                val_xcg     = float(sol(X_cg)) * 100
                val_force_stab = float(sol(aero["wing_aero_components"][1].L))
                val_moment  = float(sol(M)+M_mast)
                val_vh = float(sol(v_h))
                val_fuselage_length = float(sol(fuselage_length)) * 100

                # 2. AFFICHAGE 
                print("\n" + "="*80)
                print(f"FICHE TECHNIQUE FINALE : {CASE.upper()} - {root_naca_profile.upper()}")
                print("="*80)
                
                print(f"\n--- 1. GÉOMÉTRIE AILE ---")
                print(f"• Surface       : {val_surface:.1f} cm²")
                print(f"• Envergure     : {val_span:.1f} cm")
                print(f"• Allongement   : {val_ar:.2f}")
                print(f"• Corde Root    : {val_root_c:.1f} mm")
                print(f"• Corde Tip     : {val_tip_c:.1f} mm")
                print(f"• Vrillage      : {val_twist:.2f}°")
                print(f"• Charge alaire : {wing_loading:.2f} N/m²")
                
                print(f"\n--- 2. GÉOMÉTRIE STAB ---")
                print(f"• Surface       : {val_s_surface:.1f} cm²")
                print(f"• Envergure     : {val_s_span:.1f} cm")
                print(f"• Allongement   : {val_s_ar:.2f}")
                print(f"• Corde Root    : {val_s_root_c:.1f} mm")
                print(f"• Vrillage      : {val_s_twist:.2f}°")
                print(f"• Longueur Fuselage : {val_fuselage_length:.1f} cm")
                

                print(f"\n--- 3. PERFORMANCES ---")
                print(f"• Finesse (L/D) : {val_finesse:.2f}")
                print(f"• Traînée       : {val_trainee:.2f} N")
                print(f"• Incidence     : {val_alpha:.2f}°")
                print(f"• Re Root      : {Re_root:.1e}")
                print(f"• Re Tip       : {Re_tip:.1e}")
                print(f"• CL Cruising    : {CL_cruise:.3f}")
                print(f"• CD Cruising    : {CD_cruise:.4f}")
                print(f"• CL Stab       : {CL_stab:.3f}")
            
                
                print(f"\n--- 4. STABILITÉ & ÉQUILIBRE ---")
                print(f"• Marge Statique: {val_sm:.2f} %") 
                print(f"• Position CG   : {val_cg:.1f}% ({val_xcg:.1f} cm du BA)")
                print(f"• Force Stab    : {val_force_stab:.2f} N")
                print(f"• Moment Résid. : {val_moment:.4f} N.m")
                print(f"• Volume de Queue : {val_vh:.4f}")

                # =========================================================
                # EXPORTATION (Format Selig pour XFLR5/CAD)
                # =========================================================
                print(f"\n--- 5. EXPORTATION ---")
                now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = os.path.join("outputs", f"{CASE}_{root_naca_profile}_{now_str}")
                airfoils_dir = os.path.join(out_dir, "airfoils")
                os.makedirs(airfoils_dir, exist_ok=True)

                def export_profile_dat(af_obj, filename, name_internal):
                    """ Nettoie, trie et exporte un profil au format .dat standard """
                    af_obj = af_obj.repanel(n_points_per_side=50)
                    coords = af_obj.coordinates
                    # Normalisation (0 à 1)
                    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
                    coords[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min)
                    coords[:, 1] = coords[:, 1] / (x_max - x_min)

                    # Séparation Extrados/Intrados pour le tri (Selig Format)
                    # On cherche le point le plus proche de (0,0) -> Bord d'attaque
                    idx_le = np.argmin(coords[:, 0])
                    upper = coords[:idx_le + 1] # Du bord de fuite au bord d'attaque
                    lower = coords[idx_le:]     # Du bord d'attaque au bord de fuite
                    
                    # On s'assure que 'upper' va de 1 vers 0 (X)
                    if upper[0, 0] < upper[-1, 0]: upper = upper[::-1]
                    # On s'assure que 'lower' va de 0 vers 1 (X)
                    if lower[0, 0] > lower[-1, 0]: lower = lower[::-1]
                    
                    # Fusion
                    final_coords = np.concatenate([upper, lower[1:]])
                    
                    # Écriture
                    filepath = os.path.join(airfoils_dir, filename)
                    with open(filepath, "w") as f:
                        f.write(f"{name_internal}\n")
                        for x, y in final_coords:
                            f.write(f" {x:.6f} {y:.6f}\n")

                def export_body(fuselage_obj, filename="fuselage.txt", n_points_per_section=21):
                    """
                    Exporte le fuselage au format natif complet 'BODY' pour XFLR5.
                    Il génère les points Y, Z par trigonométrie (Cercle parfait).
                    Format :
                    BODY
                    Name
                    nSections nPointsPerSection
                    x y z
                    ...
                    """
                    # 1. Préparation
                    xsecs = fuselage_obj.xsecs
                    n_sections = len(xsecs)
                    # On décale les X pour que le nez soit à 0.00 (Standard XFLR5)
                    x_start = xsecs[0].xyz_c[0]

                    filepath = os.path.join(out_dir, filename)
                    with open(filepath, "w") as f:
                        # En-têtes obligatoires
                        f.write("BODY\n")
                        f.write(f"{fuselage_obj.name}\n")
                        # Format standard : "nSections  nPoints"
                        f.write(f" {n_sections}  {n_points_per_section}\n")
                        # 3. Boucle sur les sections (Longitudinal)
                        for i, xsec in enumerate(xsecs):
                            x_local = xsec.xyz_c[0] - x_start
                            # Récupération rayon
                            if hasattr(xsec, 'width'): r = xsec.width/2
                            else: r = 0.0
                            # Génération des points du cercle (Transversal)
                            angles = np.linspace(0, 2*np.pi, n_points_per_section)
                            for theta in angles:
                                y = r * np.cos(theta)
                                z = r * np.sin(theta)
                                f.write(f" {x_local:.6f}  {y:.6f}  {z:.6f}\n")
                    print(f"\nFichier coordonnées du fuselage généré : {filepath}")

                # --- MainWing Sections ---
                wing_optim = sol(wing)
                for i, xsec in enumerate(wing_optim.xsecs):
                    # Nom du fichier : wing_sec_0.dat, wing_sec_1.dat...
                    af_name = f"wing_sec_{i}"
                    fname = f"{af_name}.dat"
                    # L'airfoil dans la section optimisée
                    af = xsec.airfoil
                    af.name = af_name
                    export_profile_dat(af, fname, af_name)

                # --- Stab sections ---
                stab_optim = sol(stab)
                for i, xsec in enumerate(stab_optim.xsecs):
                    af_name = f"stab_sec_{i}"
                    fname = f"{af_name}.dat"
                    af = xsec.airfoil
                    af.name = af_name
                    export_profile_dat(af, fname, af_name)
                
                # --- Export XML Avion complet ---
                airplane_export = asb.Airplane(wings=[asb.Wing(symmetric=True, name="mainwing", xsecs=wing_optim.xsecs), 
                                                      asb.Wing(symmetric=True, name="elevator", xsecs=stab_optim.xsecs),
                                                      mast_obj], 
                                               fuselages=[fuselage_obj],                                             
                                               xyz_ref=sol(airplane).xyz_ref)
                xml_path = os.path.join(out_dir, f"{CASE}_{root_naca_profile}_{now_str}_plane.xml")
                airplane_export.export_XFLR5_xml(xml_path)

                print(f"\nFichier 3D XML généré dans {out_dir}")
                print(f"Profils .dat générés dans : {airfoils_dir}")
                #export_body(fuselage_obj, filename=f"{CASE}_fuselage_{fuselage_length}m.txt")
                print("="*80 + "\n")

            except Exception as e_print:
                print(f"\nErreur DEBUG Affichage : {e_print}")
                import traceback
                traceback.print_exc()
            
        return {
            "success": True,
            "finesse": sol(L/D),
            "surface": sol(wing.area()) * 10000,
            "stab_force": sol(aero["wing_aero_components"][1].L),
            "moment": sol(M),
            "root_chord": sol(root_chord),
            "span": sol(span)
        }

    except Exception:
        return {"success": False}
    

# ========================================================================================
# 4. BOUCLE D'ÉVALUATION PRINCIPALE
# ========================================================================================
print("\n" + "="*60)
print(f"     OPTIMISATION PARAMÉTRIQUE - CAS : {CASE.upper()}")
print("="*60 + "\n")

best_run = None
best_config = ""

for c in cambrures:
    for t_root in epaisseurs_root:
        for reflex in angle_reflex:
            # Le saumon est toujours plus fin que l'emplanture pour réduire la traînée
            t_tip = t_root - 3  # ex: Root 15% -> Tip 12%
            
             # Construction des profils NACA
            root_name = f"naca{c}4{t_root:02d}"
            tip_name = f"naca{c}4{t_tip:02d}" 
            
            print(f"Test: Root {root_name} / Tip {tip_name} ... / reflex : {reflex}° | ", end="")
            
            # Appel du moteur
            res = evaluate_design(root_name, tip_name, reflex, perform_export=False)
            
            if res["success"]:
                print(f"OK | Finesse: {res['finesse']:.2f} | Surf: {res['surface']:.0f} | Stab: {res['stab_force']:.1f}N")
                
                # Conservation du meilleur résultat
                if best_run is None or res["finesse"] > best_run["finesse"]:
                    best_run = res
                    best_config = (root_name, tip_name, reflex)
            else:
                print("Infeasible")

# ========================================================================================
# 5. RÉSULTAT FINAL
# ========================================================================================
print("\n" + "="*60)
if best_run:
    r_name, t_name, reflex = best_config
    
    print(f"CONFIGURATION RETENUE :")
    print(f"   Emplanture : {r_name.upper()}")
    print(f"   Saumon     : {t_name.upper()}")
    print(f"   Angle Reflex : {reflex}°")
    print("-" * 30)
     
    print("\n[Génération des fichiers finaux...]")
    evaluate_design(r_name, t_name, reflex, perform_export=True)
    
else:
    print("Aucun profil n'a permis de satisfaire toutes les contraintes.")
print("="*60)
