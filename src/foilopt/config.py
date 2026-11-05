from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
SCENARIO_DIR = CONFIG_DIR / "scenarios"
DEFAULT_SCENARIO = SCENARIO_DIR / "wingfoil-freeride.yaml"


def resolve_scenario(name=None):
    if name is None:
        return DEFAULT_SCENARIO

    path = Path(name)
    if path.is_file() or path.is_absolute():
        return path
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    candidate = SCENARIO_DIR / path.name
    if candidate.is_file():
        return candidate

    available = ", ".join(sorted(file.stem for file in SCENARIO_DIR.glob("*.yaml")))
    raise FileNotFoundError(
        f"Scenario {name!r} not found. Available scenarios: {available}"
    )


def load_config(path=None):
    path = resolve_scenario(path)
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Scenario configuration must be a mapping: {path}")
    validate_config(config)
    return config


def validate_config(config):
    for key in ("name", "use", "implemented_model", "operating_points"):
        if key not in config:
            raise ValueError(f"Scenario configuration is missing {key!r}.")

    maneuvers = config.get("maneuvers", {})
    if float(maneuvers.get("maximum_positive_g", 0)) <= 0:
        raise ValueError("maneuvers.maximum_positive_g must be positive.")
    _validate_vpp(config.get("vpp"))
    if not config["implemented_model"]:
        if not config.get("not_implemented_reason"):
            raise ValueError("Unimplemented scenarios must explain the missing model.")
        if not config["operating_points"]:
            raise ValueError("Conditions-only scenarios need at least one operating point.")
        return

    turn = maneuvers.get("turn")
    if not turn or float(turn["speed_m_s"]) <= 0 or float(turn["load_factor_g"]) <= 0:
        raise ValueError("Implemented scenarios need a valid turn speed and load factor.")
    for section, key in (
        ("wing", "span_m"),
        ("wing", "root_chord_m"),
        ("wing", "taper_ratio"),
        ("wing", "tip_washout_deg"),
    ):
        bounds = config[section][key]["bounds"]
        if len(bounds) != 2 or not bounds[0] < bounds[1]:
            raise ValueError(f"Invalid bounds for {section}.{key}: {bounds}")

    for section, key in (
        ("wing", "area_m2"),
        ("wing", "aspect_ratio"),
        ("wing", "thickness_ratio_bounds"),
        ("rider", "cg_fraction_bounds"),
    ):
        bounds = config[section][key]
        if len(bounds) != 2 or not bounds[0] < bounds[1]:
            raise ValueError(f"Invalid bounds for {section}.{key}: {bounds}")

    if config["mesh"]["wing_sections"] < 3 or config["mesh"]["stabilizer_sections"] < 3:
        raise ValueError("Each lifting surface needs at least three spanwise sections.")
    if config["optimization"]["starts"] < 1:
        raise ValueError("Optimization must use at least one initial start.")
    for point in ("takeoff", "cruise", "high_speed"):
        if point not in config["operating_points"]:
            raise ValueError(f"Missing operating point: {point}")
        operating_point = config["operating_points"][point]
        if float(operating_point["speed_m_s"]) <= 0:
            raise ValueError(f"{point}.speed_m_s must be positive.")
        bounds = operating_point["alpha_bounds_deg"]
        if len(bounds) != 2 or not bounds[0] < bounds[1]:
            raise ValueError(f"Invalid alpha bounds for operating point {point}: {bounds}")

    if config["hydrodynamics"]["solver"] not in (
        "lifting_line",
        "vortex_lattice",
        "aero_buildup",
    ):
        raise ValueError(
            "hydrodynamics.solver must be lifting_line, vortex_lattice, or aero_buildup."
        )
    if config["hydrodynamics"]["refinement_solver"] != "lifting_line":
        raise ValueError("hydrodynamics.refinement_solver must be lifting_line.")
    for key in (
        "lifting_line_spanwise_resolution",
        "vortex_lattice_spanwise_resolution",
        "vortex_lattice_chordwise_resolution",
        "refinement_spanwise_resolution",
    ):
        if int(config["hydrodynamics"][key]) < 1:
            raise ValueError(f"{key} must be positive.")
    structure = config["structure"]
    if float(structure["material_safety_factor"]) < 1:
        raise ValueError("Structural material safety factor must be at least one.")
    if float(structure["front_wing_load_share_factor"]) < 1:
        raise ValueError("Front-wing load-share factor must be at least one.")
    if int(structure["spanwise_stations"]) < 5:
        raise ValueError("Structural model needs at least five spanwise stations.")
    if int(structure["profile_samples"]) < 11:
        raise ValueError("Structural section integration needs at least eleven samples.")
    if float(structure["smooth_max_beta"]) <= 0:
        raise ValueError("Structural smooth-maximum beta must be positive.")
    for key in ("maximum_strength_utilization", "maximum_deflection_utilization"):
        if not 0 < float(structure[key]) <= 1:
            raise ValueError(f"{key} must be in (0, 1].")
    laminate = structure["laminate"]
    if not 0 < float(laminate["fiber_volume_fraction"]) < 1:
        raise ValueError("Laminate fiber volume fraction must be between zero and one.")
    if not laminate["ply_angles_deg"]:
        raise ValueError("Laminate ply-angle sequence cannot be empty.")
    skin_thickness = (
        float(laminate["fiber_areal_weight_kg_m2"])
        / (
            float(laminate["fiber_density_kg_m3"])
            * float(laminate["fiber_volume_fraction"])
        )
        * len(laminate["ply_angles_deg"])
    )
    if skin_thickness > float(structure["maximum_skin_thickness_m"]):
        raise ValueError(
            "Configured hand-layup exceeds structure.maximum_skin_thickness_m."
        )
    if (
        float(structure["minimum_core_thickness_m"]) <= 0
        or float(structure["maximum_skin_thickness_m"]) <= 0
    ):
        raise ValueError("Manufacturing thickness limits must be positive.")


def _validate_vpp(vpp):
    if not vpp or not vpp.get("enabled"):
        return
    bounds = vpp.get("speed_bounds_m_s", [])
    if len(bounds) != 2 or not 0 < float(bounds[0]) < float(bounds[1]):
        raise ValueError("VPP speed bounds must be positive and increasing.")
    for key in ("max_iterations", "max_leeway_deg", "max_heel_deg", "tolerance"):
        if float(vpp[key]) <= 0:
            raise ValueError(f"VPP {key} must be positive.")
    if not vpp.get("operating_points"):
        raise ValueError("An enabled VPP needs at least one operating point.")
    for point, values in vpp["operating_points"].items():
        if float(values["true_wind_speed_m_s"]) <= 0:
            raise ValueError(f"VPP point {point!r} needs positive true wind speed.")
        if float(values["flight_height_m"]) < 0:
            raise ValueError(f"VPP point {point!r} flight height cannot be negative.")
        if not -180 <= float(values["true_wind_angle_deg"]) <= 180:
            raise ValueError(f"VPP point {point!r} has an invalid true-wind angle.")

    for name in ("sail", "foil"):
        surface = vpp[name]
        required = [
            "area_m2", "span_m", "oswald_efficiency",
            "cl_alpha_per_rad", "cl_max", "reynolds_chord_m", "max_alpha_deg",
        ]
        for key in required:
            if float(surface[key]) <= 0:
                raise ValueError(f"VPP {name}.{key} must be positive.")
        if float(surface["oswald_efficiency"]) > 1:
            raise ValueError(f"VPP {name}.oswald_efficiency cannot exceed one.")
        if float(surface["cd0"]) < 0:
            raise ValueError(f"VPP {name}.cd0 cannot be negative.")
    rider = vpp["rider"]
    if float(rider["mass_kg"]) <= 0 or float(rider["max_lateral_cg_shift_m"]) <= 0:
        raise ValueError("VPP rider mass and lateral CG travel must be positive.")
    fractions = rider["segment_mass_fractions"]
    segment_names = set(fractions)
    if abs(sum(map(float, fractions.values())) - 1) > 1e-6:
        raise ValueError("VPP rider segment mass fractions must sum to one.")
    for key in (
        "segment_cg_positions_m",
        "segment_lean_factors",
        "segment_fore_aft_factors",
    ):
        if set(rider[key]) != segment_names:
            raise ValueError(f"VPP rider {key} must define every mass segment.")
    if "upper_arms" not in segment_names or "forearms_hands" not in segment_names:
        raise ValueError("VPP rider segments need upper_arms and forearms_hands.")
    if len(rider["footstrap_positions_m"]) != 2 or len(rider["handhold_positions_m"]) != 2:
        raise ValueError("VPP rider needs two footstrap and two handhold positions.")
    if float(vpp["board"]["mass_kg"]) <= 0:
        raise ValueError("VPP board mass must be positive.")
    if float(vpp["board"]["underwater_drag_area_m2"]) < 0:
        raise ValueError("VPP underwater board drag area cannot be negative.")
    if (
        float(vpp["minimum_yaw_stability_Nm_per_rad"]) <= 0
        or float(vpp["minimum_roll_stability_Nm_per_rad"]) <= 0
    ):
        raise ValueError("VPP minimum static stability derivatives must be positive.")
    if (
        float(vpp["rider"]["yaw_moment_authority_Nm"]) <= 0
        or float(vpp["rider"]["pitch_moment_authority_Nm"]) <= 0
        or float(vpp["rider"]["max_fore_aft_cg_shift_m"]) <= 0
    ):
        raise ValueError("VPP rider yaw and fore-aft control bounds must be positive.")
    air_mast = vpp["air_mast"]
    if air_mast["enabled"] and (
        float(air_mast["exposed_length_m"]) <= 0
        or float(air_mast["projected_width_m"]) <= 0
        or float(air_mast["drag_coefficient"]) <= 0
        or int(air_mast["segments"]) < 1
    ):
        raise ValueError("Enabled VPP air mast needs positive dimensions and drag.")
    water_mast = vpp["water_mast"]
    ride_height_bounds = water_mast.get("ride_height_bounds_m", [])
    if (
        float(water_mast["length_m"]) <= 0
        or float(water_mast["ride_height_m"]) < 0
        or float(water_mast["ride_height_m"]) >= float(water_mast["length_m"])
        or float(water_mast["projected_width_m"]) <= 0
        or len(ride_height_bounds) != 2
        or not 0 <= float(ride_height_bounds[0]) < float(ride_height_bounds[1])
        or float(ride_height_bounds[1]) >= float(water_mast["length_m"])
        or int(water_mast["sections"]) < 3
    ):
        raise ValueError(
            "VPP water mast needs valid LLT geometry and controllable immersion bounds."
        )
