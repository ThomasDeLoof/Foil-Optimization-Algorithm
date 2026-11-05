import math

import aerosandbox as asb
import aerosandbox.numpy as anp
import casadi as ca
import numpy as np
from scipy.optimize import brentq
from scipy.stats import qmc

from foilopt.geometry import (
    build_airplane,
    create_profile_variables,
    profile_thickness,
    wing_geometry,
)
from foilopt.hydrodynamics import evaluate_aero, mast_drag, water_from_config
from foilopt.structure import sandwich_screen
from foilopt.vpp import build_vpp_point


DESIGN_NAMES = ("span", "root_chord", "taper", "sweep_deg", "tip_washout", "cg_fraction")
VARIABLE_SCALES = {
    "span": 1.0,
    "root_chord": 0.2,
    "taper": 0.5,
    "sweep_deg": 10.0,
    "tip_washout": 1.0,
    "cg_fraction": 0.4,
}


def _bounded_variable(opti, initial, bounds, scale):
    return opti.variable(
        init_guess=float(initial),
        scale=scale,
        lower_bound=float(bounds[0]),
        upper_bound=float(bounds[1]),
    )


def build_problem(config):
    opt_cfg = config["optimization"]
    wing_cfg = config["wing"]
    rider = config["rider"]
    fluid = config["fluid"]
    opti = asb.Opti()

    design_specs = {
        "span": (wing_cfg["span_m"], VARIABLE_SCALES["span"]),
        "root_chord": (wing_cfg["root_chord_m"], VARIABLE_SCALES["root_chord"]),
        "taper": (wing_cfg["taper_ratio"], VARIABLE_SCALES["taper"]),
        "sweep_deg": (wing_cfg["sweep_deg"], VARIABLE_SCALES["sweep_deg"]),
        "tip_washout": (wing_cfg["tip_washout_deg"], VARIABLE_SCALES["tip_washout"]),
    }
    design = {
        name: _bounded_variable(
            opti, spec["initial"], spec["bounds"], scale
        )
        for name, (spec, scale) in design_specs.items()
    }
    design["cg_fraction"] = _bounded_variable(
        opti,
        rider["cg_fraction_initial"],
        rider["cg_fraction_bounds"],
        VARIABLE_SCALES["cg_fraction"],
    )

    profile, baseline, profile_variables = create_profile_variables(opti, config)
    airplane, geometry = build_airplane(design, config, profile)
    water = water_from_config(config)
    operating_points = config["operating_points"]
    hydro = config["hydrodynamics"]
    speed = {
        "cruise": float(operating_points["cruise"]["speed_m_s"]),
        "fast": float(operating_points["high_speed"]["speed_m_s"]),
        "takeoff": float(operating_points["takeoff"]["speed_m_s"]),
    }
    alpha = {
        "cruise": opti.variable(
            init_guess=1, scale=1,
            lower_bound=float(operating_points["cruise"]["alpha_bounds_deg"][0]),
            upper_bound=float(operating_points["cruise"]["alpha_bounds_deg"][1]),
        ),
        "fast": opti.variable(
            init_guess=0, scale=1,
            lower_bound=float(operating_points["high_speed"]["alpha_bounds_deg"][0]),
            upper_bound=float(operating_points["high_speed"]["alpha_bounds_deg"][1]),
        ),
        "takeoff": opti.variable(
            init_guess=5, scale=1,
            lower_bound=float(operating_points["takeoff"]["alpha_bounds_deg"][0]),
            upper_bound=float(operating_points["takeoff"]["alpha_bounds_deg"][1]),
        ),
    }
    aero = {
        name: evaluate_aero(
            airplane,
            speed[name],
            alpha[name],
            water,
            opt_cfg["model_size"],
            solver=hydro["solver"],
            spanwise_resolution=int(
                hydro["vortex_lattice_spanwise_resolution"]
                if hydro["solver"] == "vortex_lattice"
                else hydro["lifting_line_spanwise_resolution"]
            ),
            chordwise_resolution=int(hydro["vortex_lattice_chordwise_resolution"]),
        )
        for name in speed
    }

    cla = ca.jacobian(aero["cruise"]["CL"], alpha["cruise"]) * (180 / math.pi)
    cma = ca.jacobian(aero["cruise"]["Cm"], alpha["cruise"]) * (180 / math.pi)
    static_margin = -cma / cla
    thickness = profile_thickness(profile)
    structure = sandwich_screen(
        design["span"],
        design["root_chord"],
        design["taper"],
        profile,
        thickness,
        config,
    )
    structural_limits = config["structure"]

    rho = float(fluid["density_kg_m3"])
    nominal_weight = (
        float(rider["nominal_mass_kg"]) + float(rider["board_mass_kg"])
    ) * float(fluid["gravity_m_s2"])
    design_weight = (
        float(rider["design_mass_kg"]) + float(rider["board_mass_kg"])
    ) * float(fluid["gravity_m_s2"])
    stance_limit = (
        float(rider["nominal_mass_kg"]) * float(fluid["gravity_m_s2"])
        * float(rider["stance_travel_m"])
    )
    moment_limit = stance_limit * float(opt_cfg["stance_authority_fraction"])
    moments = {
        name: aero[name]["Cm"] * (0.5 * rho * speed[name] ** 2)
        * geometry["area_m2"] * geometry["mac_m"]
        for name in speed
    }

    for name in ("cruise", "fast"):
        opti.subject_to(aero[name]["L"] / nominal_weight == 1)
    opti.subject_to(aero["takeoff"]["L"] / design_weight >= 1)
    opti.subject_to(
        aero["takeoff"]["CL"] <= float(operating_points["takeoff"]["maximum_cl"])
    )
    opti.subject_to(cla >= 1.0)
    opti.subject_to(cma <= -0.05)
    for moment in moments.values():
        opti.subject_to(anp.fabs(moment) / moment_limit <= 1)

    area = geometry["area_m2"]
    opti.subject_to(area >= float(wing_cfg["area_m2"][0]))
    opti.subject_to(area <= float(wing_cfg["area_m2"][1]))
    opti.subject_to(geometry["tip_chord_m"] >= float(wing_cfg["tip_chord_min_m"]))
    opti.subject_to(design["span"] ** 2 / area >= float(wing_cfg["aspect_ratio"][0]))
    opti.subject_to(design["span"] ** 2 / area <= float(wing_cfg["aspect_ratio"][1]))
    thickness_bounds = wing_cfg["thickness_ratio_bounds"]
    opti.subject_to(thickness >= float(thickness_bounds[0]))
    opti.subject_to(thickness <= float(thickness_bounds[1]))
    opti.subject_to(structure["core_depth_utilization"] >= 1)
    opti.subject_to(
        structure["minimum_core_depth_m"]
        >= float(structural_limits["minimum_core_thickness_m"])
    )
    opti.subject_to(
        structure["carbon_utilization"]
        <= float(structural_limits["maximum_strength_utilization"])
    )
    opti.subject_to(
        structure["foam_shear_utilization"]
        <= float(structural_limits["maximum_strength_utilization"])
    )
    opti.subject_to(
        structure["deflection_utilization"]
        <= float(structural_limits["maximum_deflection_utilization"])
    )

    drag = {
        name: aero[name]["D"] + mast_drag(speed[name], config, opt_cfg["model_size"])
        for name in speed
    }
    vpp_config = config.get("vpp", {})
    vpp_points = {}
    if vpp_config.get("enabled"):
        foil_geometry = {
            "area_m2": geometry["area_m2"],
            "span_m": design["span"],
            "reynolds_chord_m": geometry["mac_m"],
        }
        vpp_points = {
            name: build_vpp_point(
                config,
                name,
                opti=opti,
                foil_geometry=foil_geometry,
                foil_airplane=airplane,
                foil_reference_position=[geometry["cg_x_m"], 0, 0],
            )
            for name in vpp_config["operating_points"]
        }

    profile_penalty = (
        anp.sum((profile_variables["lower"] - baseline.lower_weights) ** 2)
        + anp.sum((profile_variables["upper"] - baseline.upper_weights) ** 2)
    ) / float(wing_cfg["kulfan_weight_delta"]) ** 2
    weighted_drag_objective = (
        drag["cruise"]
        + float(opt_cfg["high_speed_drag_weight"]) * drag["fast"]
        + float(opt_cfg["takeoff_drag_weight"]) * drag["takeoff"]
    )
    if vpp_points:
        mean_vpp_finesse = ca.sum1(
            ca.vertcat(
                *[
                    point["hydrodynamic_finesse"]
                    for point in vpp_points.values()
                ]
            )
        ) / len(vpp_points)
        objective = (
            -mean_vpp_finesse
            + float(opt_cfg["profile_regularization_weight"]) * profile_penalty
        )
        objective_name = "maximize_mean_vpp_hydrodynamic_finesse"
    else:
        objective = (
            weighted_drag_objective
            + float(opt_cfg["profile_regularization_weight"]) * profile_penalty
        )
        objective_name = "minimize_weighted_drag"
    opti.minimize(objective)

    return opti, {
        "design": design,
        "profile": profile,
        "baseline_profile": baseline,
        "profile_variables": profile_variables,
        "airplane": airplane,
        "geometry": geometry,
        "aero": aero,
        "alpha": alpha,
        "speed": speed,
        "drag": drag,
        "cla": cla,
        "cma": cma,
        "static_margin": static_margin,
        "thickness_ratio": thickness,
        "structure": structure,
        "moments": moments,
        "objective": objective,
        "objective_name": objective_name,
        "weighted_drag_objective": weighted_drag_objective,
        "profile_penalty": profile_penalty,
        "vpp_points": vpp_points,
        "nominal_weight_N": nominal_weight,
        "design_weight_N": design_weight,
        "stance_limit_Nm": stance_limit,
        "moment_limit_Nm": moment_limit,
    }


def evaluate_solution(solution, model, config, attempts):
    def value(expression):
        if isinstance(expression, (ca.MX, ca.SX, ca.DM)):
            return float(solution.value(expression))
        return float(expression)

    design = {name: value(variable) for name, variable in model["design"].items()}
    geometry = {
        name: value(model["geometry"][name])
        for name in (
            "area_m2", "mac_m", "tip_chord_m", "cg_x_m",
            "stabilizer_area_m2", "stabilizer_span_m", "stabilizer_root_chord_m",
        )
    }
    points = {}
    for name in model["speed"]:
        point = model["aero"][name]
        points[name] = {
            key: value(point[key]) for key in ("L", "D", "CL", "CD", "Cm")
        }
        points[name].update({
            "speed_m_s": model["speed"][name],
            "alpha_deg": value(model["alpha"][name]),
            "drag_including_mast_N": value(model["drag"][name]),
        })

    profile = {
        side: np.asarray(solution.value(model["profile_variables"][side])).reshape(-1).tolist()
        for side in ("lower", "upper")
    }
    structure = {
        name: value(metric) for name, metric in model["structure"].items()
    }
    vpp_points = {}
    vpp_state_bounds_passed = True
    vpp_force_balance_passed = True
    vpp_roll_stability_passed = True
    vpp_yaw_stability_passed = True
    for name, point in model["vpp_points"].items():
        point_result = {
            key: value(point[key])
            for key in (
                "true_wind_speed_m_s",
                "true_wind_angle_deg",
                "flight_height_input_m",
                "ride_height_m",
                "boat_speed_m_s",
                "leeway_deg",
                "heel_deg",
                "sail_alpha_deg",
                "foil_alpha_deg",
                "rider_lateral_cg_shift_m",
                "rider_fore_aft_cg_shift_m",
                "rider_yaw_control_moment_Nm",
                "rider_pitch_control_moment_Nm",
                "apparent_wind_speed_m_s",
                "foil_cl",
                "foil_cd",
                "foil_reynolds",
                "foil_lift_N",
                "foil_drag_N",
                "foil_finesse",
                "board_drag_N",
                "hydrodynamic_drag_N",
                "hydrodynamic_finesse",
                "air_mast_drag_N",
                "water_mast_drag_N",
                "water_mast_sideforce_N",
                "immersed_mast_length_m",
                "water_mast_cd",
                "water_mast_cy",
                "yaw_stability_derivative_Nm_per_rad",
                "roll_stability_derivative_Nm_per_rad",
                "minimum_yaw_stability_Nm_per_rad",
                "minimum_roll_stability_Nm_per_rad",
            )
        }
        state_bounds = point["state_bounds"]
        state_margins = {}
        active_limits = []
        for state_name, bounds in state_bounds.items():
            actual = point_result[state_name]
            lower, upper = bounds
            state_margins[state_name] = {
                "lower": actual - lower,
                "upper": upper - actual,
            }
            tolerance = max(1e-3, 1e-3 * (upper - lower))
            if min(actual - lower, upper - actual) < -1e-5:
                vpp_state_bounds_passed = False
            if min(actual - lower, upper - actual) <= tolerance:
                active_limits.append(state_name)

        force_residual = np.asarray(
            solution.value(point["force_residual_N"])
        ).reshape(-1)
        moment_residual = np.asarray(
            solution.value(point["moment_residual_Nm"])
        ).reshape(-1)
        rider_cg = np.asarray(solution.value(point["rider_cg_m"])).reshape(-1)
        yaw_stable = (
            point_result["yaw_stability_derivative_Nm_per_rad"]
            <= -point_result["minimum_yaw_stability_Nm_per_rad"] + 1e-3
        )
        roll_stable = (
            point_result["roll_stability_derivative_Nm_per_rad"]
            <= -point_result["minimum_roll_stability_Nm_per_rad"] + 1e-3
        )
        force_closed = bool(np.max(np.abs(force_residual)) <= 0.1)
        moment_closed = bool(np.max(np.abs(moment_residual)) <= 0.1)
        vpp_force_balance_passed &= force_closed and moment_closed
        vpp_yaw_stability_passed &= yaw_stable
        vpp_roll_stability_passed &= roll_stable
        point_result.update({
            "name": name,
            "force_residual_N": force_residual.tolist(),
            "moment_residual_Nm": moment_residual.tolist(),
            "rider_cg_m": rider_cg.tolist(),
            "state_margins": state_margins,
            "active_limits": active_limits,
            "boundary_limited": bool(active_limits),
            "roll_stability_passed": roll_stable,
            "yaw_stability_passed": yaw_stable,
            "trim_valid": (
                not active_limits and force_closed and moment_closed
                and roll_stable and yaw_stable
            ),
        })
        vpp_points[name] = point_result
    cla, cma = value(model["cla"]), value(model["cma"])
    area_bounds = config["wing"]["area_m2"]
    ar_bounds = config["wing"]["aspect_ratio"]
    aspect_ratio = design["span"] ** 2 / geometry["area_m2"]
    bounded_values = {
        "span_m": (design["span"], config["wing"]["span_m"]["bounds"]),
        "root_chord_m": (
            design["root_chord"], config["wing"]["root_chord_m"]["bounds"]
        ),
        "taper_ratio": (design["taper"], config["wing"]["taper_ratio"]["bounds"]),
        "sweep_deg": (design["sweep_deg"], config["wing"]["sweep_deg"]["bounds"]),
        "tip_washout_deg": (
            design["tip_washout"], config["wing"]["tip_washout_deg"]["bounds"]
        ),
        "cg_fraction": (
            design["cg_fraction"], config["rider"]["cg_fraction_bounds"]
        ),
        "area_m2": (geometry["area_m2"], config["wing"]["area_m2"]),
        "aspect_ratio": (aspect_ratio, config["wing"]["aspect_ratio"]),
    }
    active_bounds = []
    for name, (value_, bounds) in bounded_values.items():
        tolerance = max(1e-5, 1e-3 * (bounds[1] - bounds[0]))
        if abs(value_ - bounds[0]) <= tolerance:
            active_bounds.append(f"{name}: lower")
        if abs(value_ - bounds[1]) <= tolerance:
            active_bounds.append(f"{name}: upper")
    takeoff_cl_limit = float(config["operating_points"]["takeoff"]["maximum_cl"])
    checks = {
        "cruise_lift_trim": abs(points["cruise"]["L"] - model["nominal_weight_N"]) <= 0.005 * model["nominal_weight_N"],
        "high_speed_lift_trim": abs(points["fast"]["L"] - model["nominal_weight_N"]) <= 0.005 * model["nominal_weight_N"],
        "takeoff_lift": points["takeoff"]["L"] >= 0.995 * model["design_weight_N"],
        "takeoff_cl": points["takeoff"]["CL"] <= takeoff_cl_limit + 0.005,
        "positive_CLa": cla >= 1.0,
        "negative_Cma": cma <= -0.05,
        "pitch_moment_authority": all(
            abs(value_) <= 1.005 * model["moment_limit_Nm"]
            for value_ in (value(model["moments"][name]) for name in model["speed"])
        ),
        "reference_sized_geometry": (
            area_bounds[0] - 1e-5
            <= geometry["area_m2"]
            <= area_bounds[1] + 1e-5
        ),
        "moderate_aspect_ratio": ar_bounds[0] - 1e-3 <= aspect_ratio <= ar_bounds[1] + 1e-3,
        "minimum_tip_chord": geometry["tip_chord_m"] >= float(config["wing"]["tip_chord_min_m"]) - 1e-5,
        "carbon_skin_stress": structure["carbon_utilization"]
        <= config["structure"]["maximum_strength_utilization"] + 0.001,
        "foam_core_shear": structure["foam_shear_utilization"]
        <= config["structure"]["maximum_strength_utilization"] + 0.001,
        "minimum_root_core": structure["core_depth_utilization"] >= 0.999,
        "minimum_spanwise_core": structure["minimum_core_depth_m"]
        >= float(config["structure"]["minimum_core_thickness_m"]) - 1e-5,
        "tip_deflection_screen": structure["deflection_utilization"]
        <= config["structure"]["maximum_deflection_utilization"] + 0.001,
    }
    if model["vpp_points"]:
        checks.update({
            "vpp_state_bounds": vpp_state_bounds_passed,
            "vpp_force_moment_balance": vpp_force_balance_passed,
            "vpp_yaw_stability": vpp_yaw_stability_passed,
            "vpp_roll_stability": vpp_roll_stability_passed,
        })
    return {
        "scenario": config["name"],
        "design": design,
        "geometry": geometry,
        "profile": profile,
        "profile_thickness_ratio": value(model["thickness_ratio"]),
        "points": points,
        "vpp_points": vpp_points,
        "stability": {
            "CLa_per_rad": cla,
            "Cma_per_rad": cma,
            "derived_static_margin": value(model["static_margin"]),
            "pitch_moments_Nm": {
                name: value(model["moments"][name]) for name in model["speed"]
            },
            "stance_limit_Nm": model["stance_limit_Nm"],
            "moment_limit_Nm": model["moment_limit_Nm"],
        },
        "structure": structure,
        "aspect_ratio": aspect_ratio,
        "active_bounds": active_bounds,
        "objective_N": value(model["weighted_drag_objective"]),
        "objective_value": value(model["objective"]),
        "objective_name": model["objective_name"],
        "profile_penalty": value(model["profile_penalty"]),
        "checks": checks,
        "feasible": all(checks.values()),
        "attempts": attempts,
        "nominal_weight_N": model["nominal_weight_N"],
        "design_weight_N": model["design_weight_N"],
    }


def _initial_design(config):
    wing, rider = config["wing"], config["rider"]
    return {
        "span": float(wing["span_m"]["initial"]),
        "root_chord": float(wing["root_chord_m"]["initial"]),
        "taper": float(wing["taper_ratio"]["initial"]),
        "sweep_deg": float(wing["sweep_deg"]["initial"]),
        "tip_washout": float(wing["tip_washout_deg"]["initial"]),
        "cg_fraction": float(rider["cg_fraction_initial"]),
    }


def optimize(config, starts=None, max_iterations=None):
    solver_config = dict(config["optimization"])
    solver_config["starts"] = max(1, int(starts or solver_config["starts"]))
    configured_iterations = solver_config["max_iterations"]
    if config.get("vpp", {}).get("enabled"):
        configured_iterations = max(
            configured_iterations, int(config["vpp"]["max_iterations"])
        )
    solver_config["max_iterations"] = int(
        max_iterations if max_iterations is not None else configured_iterations
    )
    opti, model = build_problem(config)
    opti.solver("ipopt", {
        "ipopt.max_iter": solver_config["max_iterations"],
        "ipopt.tol": float(solver_config["tolerance"]),
        "ipopt.print_level": 0,
        "print_time": False,
    })

    names = list(DESIGN_NAMES)
    low = np.array([
        config["wing"]["span_m"]["bounds"][0],
        config["wing"]["root_chord_m"]["bounds"][0],
        config["wing"]["taper_ratio"]["bounds"][0],
        config["wing"]["sweep_deg"]["bounds"][0],
        config["wing"]["tip_washout_deg"]["bounds"][0],
        config["rider"]["cg_fraction_bounds"][0],
    ])
    high = np.array([
        config["wing"]["span_m"]["bounds"][1],
        config["wing"]["root_chord_m"]["bounds"][1],
        config["wing"]["taper_ratio"]["bounds"][1],
        config["wing"]["sweep_deg"]["bounds"][1],
        config["wing"]["tip_washout_deg"]["bounds"][1],
        config["rider"]["cg_fraction_bounds"][1],
    ])
    random_guesses = qmc.scale(
        qmc.Sobol(d=len(names), scramble=True, seed=42).random(solver_config["starts"] - 1),
        low,
        high,
    ) if solver_config["starts"] > 1 else np.empty((0, len(names)))
    guesses = [_initial_design(config)] + [
        dict(zip(names, map(float, row))) for row in random_guesses
    ]

    attempts, candidates = [], []
    for index, guess in enumerate(guesses, start=1):
        for name in names:
            opti.set_initial(model["design"][name], guess[name])
        for name, alpha in (("cruise", 1), ("fast", 0), ("takeoff", 5)):
            opti.set_initial(model["alpha"][name], alpha)
        opti.set_initial(
            model["profile_variables"]["lower"],
            model["baseline_profile"].lower_weights,
        )
        opti.set_initial(
            model["profile_variables"]["upper"],
            model["baseline_profile"].upper_weights,
        )
        try:
            solution = opti.solve(verbose=False)
        except RuntimeError as error:
            attempts.append({
                "start": index,
                "status": opti.stats().get("return_status", "failed"),
                "message": str(error).splitlines()[0],
            })
            continue
        result = evaluate_solution(solution, model, config, attempts)
        status = solution.stats()
        attempts.append({
            "start": index,
            "status": status.get("return_status", "unknown"),
            "feasible": result["feasible"],
            "objective_value": result["objective_value"],
            "failed_checks": [
                name for name, passed in result["checks"].items() if not passed
            ],
        })
        if status.get("success") and result["feasible"]:
            candidates.append(result)

    if not candidates:
        raise RuntimeError(f"No feasible design found. IPOPT attempts: {attempts}")
    best = min(candidates, key=lambda result: result["objective_value"])
    best["attempts"] = attempts
    best["starts"] = solver_config["starts"]
    best["max_iterations"] = solver_config["max_iterations"]
    best["model_size"] = solver_config["model_size"]
    return best


def lifting_line_check(result, config):
    design = dict(result["design"])
    lower = np.asarray(result["profile"]["lower"])
    upper = np.asarray(result["profile"]["upper"])
    baseline = asb.Airfoil(config["wing"]["airfoil_baseline"]).to_kulfan_airfoil(
        n_weights_per_side=config["wing"]["kulfan_weights_per_side"]
    )
    profile = asb.KulfanAirfoil(
        lower_weights=lower,
        upper_weights=upper,
        leading_edge_weight=baseline.leading_edge_weight,
        TE_thickness=baseline.TE_thickness,
    )
    airplane, geometry = build_airplane(design, config, profile)
    plane = asb.Airplane(
        wings=airplane.wings,
        xyz_ref=airplane.xyz_ref,
        s_ref=geometry["area_m2"],
        c_ref=geometry["mac_m"],
        b_ref=design["span"],
    )
    target = float(result["nominal_weight_N"])
    speed = float(config["operating_points"]["cruise"]["speed_m_s"])
    water = water_from_config(config)

    def run(alpha_deg):
        return asb.LiftingLine(
            plane,
            asb.OperatingPoint(velocity=speed, alpha=alpha_deg, atmosphere=water),
            model_size=result["model_size"],
            spanwise_resolution=int(
                config["hydrodynamics"]["refinement_spanwise_resolution"]
            ),
        ).run()

    design_alpha = float(result["points"]["cruise"]["alpha_deg"])
    at_design_alpha = run(design_alpha)
    design_model_lift = float(result["points"]["cruise"]["L"])
    design_model_drag = float(result["points"]["cruise"]["D"])
    lifting_line_lift = float(at_design_alpha["L"])
    lifting_line_drag = float(at_design_alpha["D"])

    lo, hi = -5.0, 18.0
    if (float(run(lo)["L"]) - target) * (float(run(hi)["L"]) - target) > 0:
        return {"trimmed": False, "message": "No LiftingLine trim root in [-5, 18] degrees."}
    alpha = brentq(lambda value: float(run(value)["L"]) - target, lo, hi)
    output = run(alpha)
    return {
        "trimmed": True,
        "alpha_deg": alpha,
        "lift_N": float(output["L"]),
        "drag_N": float(output["D"]),
        "L_over_D": float(output["L"]) / float(output["D"]),
        "at_design_alpha": {
            "alpha_deg": design_alpha,
            "lift_N": lifting_line_lift,
            "drag_N": lifting_line_drag,
            "lift_difference_percent": 100
            * (lifting_line_lift - design_model_lift)
            / design_model_lift,
            "drag_difference_percent": 100
            * (lifting_line_drag - design_model_drag)
            / design_model_drag,
        },
        "scope": "Independent wing-and-stabilizer check; excludes mast and fuselage.",
    }
