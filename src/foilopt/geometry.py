import casadi as ca
import aerosandbox as asb
import aerosandbox.numpy as anp


def create_profile_variables(opti, config):
    wing_cfg = config["wing"]
    baseline = asb.Airfoil(wing_cfg["airfoil_baseline"]).to_kulfan_airfoil(
        n_weights_per_side=wing_cfg["kulfan_weights_per_side"]
    )
    delta = float(wing_cfg["kulfan_weight_delta"])
    lower = ca.vertcat(*[
        opti.variable(
            init_guess=float(weight),
            scale=0.2,
            lower_bound=float(weight) - delta,
            upper_bound=float(weight) + delta,
        )
        for weight in baseline.lower_weights
    ])
    upper = ca.vertcat(*[
        opti.variable(
            init_guess=float(weight),
            scale=0.2,
            lower_bound=float(weight) - delta,
            upper_bound=float(weight) + delta,
        )
        for weight in baseline.upper_weights
    ])
    profile = asb.KulfanAirfoil(
        name=f"Optimized {wing_cfg['airfoil_baseline']} Kulfan",
        lower_weights=lower,
        upper_weights=upper,
        leading_edge_weight=baseline.leading_edge_weight,
        TE_thickness=baseline.TE_thickness,
    )
    return profile, baseline, {"lower": lower, "upper": upper}


def wing_geometry(design):
    span, root, taper = design["span"], design["root_chord"], design["taper"]
    area = span * root * (1 + taper) / 2
    mac = (2 / 3) * root * (1 + taper + taper**2) / (1 + taper)
    return area, mac, root * taper


def build_airplane(design, config, profile):
    wing_cfg = config["wing"]
    stab_cfg = config["stabilizer"]
    fuse_cfg = config["fuselage"]
    depth = float(config["fluid"]["immersion_depth_m"])
    wing_sections = []
    count = int(config["mesh"]["wing_sections"])
    for index in range(count):
        r = index / (count - 1)
        chord = design["root_chord"] * (1 - (1 - design["taper"]) * r)
        sweep = design["sweep_deg"]
        x = design["span"] * r * anp.tan(anp.radians(sweep)) / 2
        y = design["span"] * r / 2
        z = -depth - design["span"] * r**2 * anp.tan(anp.radians(wing_cfg["anhedral_deg"])) / 2
        wing_sections.append(asb.WingXSec(
            xyz_le=[x, y, z],
            chord=chord,
            twist=design["tip_washout"] * r,
            airfoil=profile,
        ))
    main_wing = asb.Wing(symmetric=True, name="OptimizedMainWing", xsecs=wing_sections)

    tail_span = float(config["references"]["stabilizer"]["span_m"])
    tail_area = float(config["references"]["stabilizer"]["area_m2"])
    tail_taper = float(stab_cfg["taper_ratio"])
    tail_root = 2 * tail_area / (tail_span * (1 + tail_taper))
    tail_x = (
        float(fuse_cfg["nose_x_m"]) + float(fuse_cfg["length_m"])
        - float(stab_cfg["fuselage_offset_m"])
    )
    tail_sections = []
    count = int(config["mesh"]["stabilizer_sections"])
    for index in range(count):
        r = index / (count - 1)
        chord = tail_root * (1 - (1 - tail_taper) * r)
        x = tail_x + tail_span * r * anp.tan(anp.radians(stab_cfg["sweep_deg"])) / 2
        y = tail_span * r / 2
        z = -depth + tail_span * r**2 * anp.tan(anp.radians(stab_cfg["dihedral_deg"])) / 2
        tail_sections.append(asb.WingXSec(
            xyz_le=[x, y, z],
            chord=chord,
            twist=float(stab_cfg["incidence_deg"]),
            airfoil=asb.Airfoil(stab_cfg["airfoil"]),
        ))
    stabilizer = asb.Wing(
        symmetric=True, name="FixedReferenceStabilizer", xsecs=tail_sections
    )

    fuselage = asb.Fuselage(
        name="FixedFuselage",
        xsecs=[
            asb.FuselageXSec(
                xyz_c=[float(fuse_cfg["nose_x_m"]) + r * float(fuse_cfg["length_m"]), 0, -depth],
                radius=float(fuse_cfg["diameter_m"]) * scale / 2,
            )
            for r, scale in ((0.0, 0.05), (0.12, 0.75), (0.82, 1.0), (1.0, 0.12))
        ],
    )
    area, mac, tip_chord = wing_geometry(design)
    cg_x = design["cg_fraction"] * mac
    airplane = asb.Airplane(
        wings=[main_wing, stabilizer],
        fuselages=[fuselage],
        xyz_ref=[cg_x, 0, 0],
        s_ref=area,
        c_ref=mac,
        b_ref=design["span"],
    )
    return airplane, {
        "main_wing": main_wing,
        "stabilizer": stabilizer,
        "fuselage": fuselage,
        "area_m2": area,
        "mac_m": mac,
        "tip_chord_m": tip_chord,
        "cg_x_m": cg_x,
        "stabilizer_area_m2": tail_area,
        "stabilizer_span_m": tail_span,
        "stabilizer_root_chord_m": tail_root,
    }


def profile_thickness(profile):
    return profile.local_thickness(x_over_c=0.30)
