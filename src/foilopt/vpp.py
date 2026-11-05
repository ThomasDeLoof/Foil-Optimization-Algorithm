import json
import math

import aerosandbox as asb
import casadi as ca
import numpy as np

from foilopt.hydrodynamics import (
    evaluate_aero,
    immersed_mast_lifting_line,
    water_from_config,
)
from foilopt.report import new_output_directory
from foilopt.geometry import build_airplane
from foilopt.vpp_disciplines import (
    distributed_air_mast_forces,
    foil_force,
    rider_cg,
    sail_force,
    vector_norm,
)


def _cross_moment(position, force):
    return ca.vertcat(
        position[1] * force[2] - position[2] * force[1],
        position[2] * force[0] - position[0] * force[2],
        position[0] * force[1] - position[1] * force[0],
    )


def _state_derivative(expression, state):
    symbols = ca.symvar(state)
    if len(symbols) != 1:
        raise ValueError("A VPP stability state must map to one solver variable.")
    symbol = symbols[0]
    return ca.jacobian(expression, symbol) / ca.jacobian(state, symbol)


def _initial_foil_model(config):
    wing = config["wing"]
    design = {
        "span": float(wing["span_m"]["initial"]),
        "root_chord": float(wing["root_chord_m"]["initial"]),
        "taper": float(wing["taper_ratio"]["initial"]),
        "sweep_deg": float(wing["sweep_deg"]["initial"]),
        "tip_washout": float(wing["tip_washout_deg"]["initial"]),
        "cg_fraction": float(config["rider"]["cg_fraction_initial"]),
    }
    airplane, geometry = build_airplane(
        design, config, asb.Airfoil(wing["airfoil_baseline"])
    )
    foil_geometry = {
        "area_m2": geometry["area_m2"],
        "span_m": design["span"],
        "reynolds_chord_m": geometry["mac_m"],
    }
    return airplane, foil_geometry, [geometry["cg_x_m"], 0, 0]


def build_vpp_point(
    config,
    point,
    *,
    opti=None,
    foil_geometry=None,
    foil_airplane=None,
    foil_reference_position=None,
):
    """Build a VPP point, optionally using a caller-owned simultaneous NLP."""
    vpp = config["vpp"]
    point_data = vpp["operating_points"][point]
    sail = vpp["sail"]
    if (
        foil_airplane is None
        and opti is None
        and "wing" in config
        and "hydrodynamics" in config
    ):
        foil_airplane, foil_geometry, foil_reference_position = _initial_foil_model(
            config
        )
    foil = dict(vpp["foil"])
    if foil_geometry is not None:
        foil.update(foil_geometry)
    rider = vpp["rider"]
    fluid = config["fluid"]
    gravity = float(fluid["gravity_m_s2"])
    rider_mass = float(rider["mass_kg"])
    board_mass = float(vpp["board"]["mass_kg"])
    total_mass = rider_mass + board_mass
    force_scale = total_mass * gravity
    standalone = opti is None
    if foil_airplane is not None and foil_reference_position is None:
        raise ValueError("A VLM/LLT VPP model requires its force-reference position.")
    if standalone:
        opti = ca.Opti()

    speed_bounds = tuple(map(float, vpp["speed_bounds_m_s"]))
    initial_speed = min(
        max(float(point_data.get("initial_speed_m_s", 6.0)), speed_bounds[0]),
        speed_bounds[1],
    )
    initial_lift_coefficient = (
        force_scale
        / (
            0.5
            * float(fluid["density_kg_m3"])
            * initial_speed**2
            * float(vpp["foil"]["area_m2"])
        )
    )
    initial_lift_coefficient = min(
        max(initial_lift_coefficient, -0.8 * float(vpp["foil"]["cl_max"])),
        0.8 * float(vpp["foil"]["cl_max"]),
    )
    initial_foil_alpha = (
        float(vpp["foil"]["cl_max"])
        * math.atanh(
            initial_lift_coefficient / float(vpp["foil"]["cl_max"])
        )
        / float(vpp["foil"]["cl_alpha_per_rad"])
    )
    initial_foil_alpha = min(
        max(
            initial_foil_alpha,
            -math.radians(float(vpp["foil"]["max_alpha_deg"])) * 0.8,
        ),
        math.radians(float(vpp["foil"]["max_alpha_deg"])) * 0.8,
    )
    initial_states = (
        initial_speed,
        math.radians(1),
        0,
        math.radians(-8),
        initial_foil_alpha,
        0,
        0,
        0,
        0,
    )

    def state_variable(initial):
        if standalone:
            return opti.variable()
        return opti.variable(init_guess=initial)

    (
        speed,
        leeway,
        heel,
        sail_alpha,
        foil_alpha,
        rider_lean,
        rider_fore_aft_shift,
        rider_yaw_moment,
        rider_pitch_moment,
    ) = [state_variable(initial) for initial in initial_states]
    water_mast = vpp["water_mast"]
    ride_height_bounds = tuple(map(float, water_mast["ride_height_bounds_m"]))
    ride_height_initial = float(
        point_data.get("flight_height_m", water_mast["ride_height_m"])
    )
    ride_height_initial = min(
        max(ride_height_initial, ride_height_bounds[0]), ride_height_bounds[1]
    )
    ride_height = state_variable(ride_height_initial)

    sail_alpha_bound = math.radians(float(sail["max_alpha_deg"]))
    foil_alpha_bound = math.radians(float(foil["max_alpha_deg"]))
    max_leeway = math.radians(float(vpp["max_leeway_deg"]))
    max_heel = math.radians(float(vpp["max_heel_deg"]))
    opti.subject_to(opti.bounded(speed_bounds[0], speed, speed_bounds[1]))
    opti.subject_to(opti.bounded(-max_leeway, leeway, max_leeway))
    opti.subject_to(opti.bounded(-max_heel, heel, max_heel))
    opti.subject_to(opti.bounded(-sail_alpha_bound, sail_alpha, sail_alpha_bound))
    opti.subject_to(opti.bounded(-foil_alpha_bound, foil_alpha, foil_alpha_bound))
    opti.subject_to(
        opti.bounded(
            -float(rider["max_lateral_cg_shift_m"]),
            rider_lean,
            float(rider["max_lateral_cg_shift_m"]),
        )
    )
    fore_aft_authority = float(rider["max_fore_aft_cg_shift_m"])
    opti.subject_to(
        opti.bounded(
            -fore_aft_authority,
            rider_fore_aft_shift,
            fore_aft_authority,
        )
    )
    yaw_authority = float(rider["yaw_moment_authority_Nm"])
    pitch_authority = float(rider["pitch_moment_authority_Nm"])
    opti.subject_to(opti.bounded(-yaw_authority, rider_yaw_moment, yaw_authority))
    opti.subject_to(
        opti.bounded(-pitch_authority, rider_pitch_moment, pitch_authority)
    )
    opti.subject_to(
        opti.bounded(ride_height_bounds[0], ride_height, ride_height_bounds[1])
    )
    opti.set_initial(speed, initial_speed)
    opti.set_initial(leeway, math.radians(1))
    opti.set_initial(heel, 0)
    opti.set_initial(sail_alpha, math.radians(-8))
    opti.set_initial(foil_alpha, initial_foil_alpha)
    opti.set_initial(rider_lean, 0)
    opti.set_initial(rider_fore_aft_shift, 0)
    opti.set_initial(rider_yaw_moment, 0)
    opti.set_initial(rider_pitch_moment, 0)
    opti.set_initial(ride_height, ride_height_initial)

    course_velocity = ca.vertcat(speed * ca.cos(leeway), speed * ca.sin(leeway))
    true_wind_angle = math.radians(float(point_data["true_wind_angle_deg"]))
    wind_velocity = float(point_data["true_wind_speed_m_s"]) * ca.vertcat(
        math.cos(true_wind_angle), math.sin(true_wind_angle)
    )

    # All discipline equations share one symbolic graph and one NLP; no
    # aerodynamic or hydrodynamic subproblem is nested inside the optimizer.
    apparent_wind_state = wind_velocity - course_velocity
    water_velocity_state = course_velocity
    apparent_speed = vector_norm(apparent_wind_state)
    water_speed = vector_norm(water_velocity_state)
    foil_position = ca.vertcat(*map(float, foil["center_of_pressure_m"]))
    sail_force_calculated, sail_cl, sail_cd, sail_re = sail_force(
        apparent_wind_state, apparent_speed, sail_alpha, sail
    )

    air_mast = vpp["air_mast"]
    if air_mast["enabled"]:
        air_mast_forces, air_mast_positions = distributed_air_mast_forces(
            apparent_wind_state, apparent_speed, air_mast
        )
    else:
        air_mast_forces, air_mast_positions = [], []

    foil_aero = None
    if foil_airplane is not None:
        hydro = config["hydrodynamics"]
        aero_settings = config["optimization"]
        foil_aero = evaluate_aero(
            foil_airplane,
            water_speed,
            foil_alpha * (180 / math.pi),
            water_from_config(config),
            aero_settings["model_size"],
            solver=hydro["solver"],
            spanwise_resolution=int(
                hydro["vortex_lattice_spanwise_resolution"]
                if hydro["solver"] == "vortex_lattice"
                else hydro["lifting_line_spanwise_resolution"]
            ),
            chordwise_resolution=int(hydro["vortex_lattice_chordwise_resolution"]),
            beta_deg=leeway * (180 / math.pi),
        )
        foil_lift = foil_aero["L"]
        foil_drag = foil_aero["D"]
        foil_cl = foil_aero["CL"]
        foil_cd = foil_aero["CD"]
        foil_re = (
            float(fluid["density_kg_m3"])
            * water_speed
            * foil["reynolds_chord_m"]
            / float(fluid["dynamic_viscosity_pa_s"])
        )
        force_body = ca.vertcat(
            -foil_drag * ca.cos(leeway),
            -foil_drag * ca.sin(leeway) + foil_aero["Y"],
            foil_lift,
        )
        cos_heel, sin_heel = ca.cos(heel), ca.sin(heel)
        foil_force_calculated = ca.vertcat(
            force_body[0],
            force_body[1] * cos_heel - force_body[2] * sin_heel,
            force_body[1] * sin_heel + force_body[2] * cos_heel,
        )
        reference_position = ca.vertcat(*foil_reference_position)
        aero_moment = foil_aero["M_g"]
        foil_moment_calculated = ca.vertcat(
            aero_moment[0],
            aero_moment[1] * cos_heel - aero_moment[2] * sin_heel,
            aero_moment[1] * sin_heel + aero_moment[2] * cos_heel,
        ) + _cross_moment(reference_position, foil_force_calculated)
    else:
        foil_force_calculated, foil_cl, foil_cd, foil_re, foil_lift, foil_drag = foil_force(
            water_velocity_state,
            water_speed,
            leeway,
            heel,
            foil_alpha,
            foil,
            fluid,
        )
        foil_moment_calculated = _cross_moment(foil_position, foil_force_calculated)
    board_drag_area = float(vpp["board"]["underwater_drag_area_m2"])
    board_drag_calculated = (
        -0.5
        * float(fluid["density_kg_m3"])
        * water_speed**2
        * board_drag_area
        * water_velocity_state
        / water_speed
    )
    board_force_calculated = ca.vertcat(
        board_drag_calculated[0], board_drag_calculated[1], 0
    )

    mast_water = immersed_mast_lifting_line(
        water_speed,
        leeway,
        ride_height,
        water_mast,
        config,
        spanwise_resolution=int(water_mast.get("spanwise_resolution", 2)),
    )
    mast_water_calculated = ca.vertcat(
        -mast_water["drag_N"],
        mast_water["sideforce_N"] * ca.cos(heel),
        mast_water["sideforce_N"] * ca.sin(heel),
    )
    immersed_mast_length = mast_water["immersed_length_m"]
    mast_water_position = ca.vertcat(
        float(water_mast["x_m"]),
        float(water_mast["y_m"]),
        -immersed_mast_length / 2,
    )

    rider_cg_calculated = rider_cg(rider, rider_lean, rider_fore_aft_shift)
    rider_cg_state = rider_cg_calculated
    board_cg = ca.vertcat(*map(float, vpp["board"]["cg_position_m"]))
    board_drag_position = ca.vertcat(
        *map(float, vpp["board"]["drag_center_m"])
    )
    sail_position = ca.vertcat(*map(float, sail["center_of_effort_m"]))
    rider_weight = ca.vertcat(0, 0, -rider_mass * gravity)
    board_weight = ca.vertcat(0, 0, -board_mass * gravity)

    force_terms = [
        sail_force_calculated,
        *air_mast_forces,
        foil_force_calculated,
        board_force_calculated,
        mast_water_calculated,
        rider_weight,
        board_weight,
    ]
    force_residual = sum(force_terms[1:], force_terms[0])

    moment_terms = [
        _cross_moment(sail_position, sail_force_calculated),
        *[
            _cross_moment(position, force)
            for position, force in zip(air_mast_positions, air_mast_forces)
        ],
        foil_moment_calculated,
        _cross_moment(board_drag_position, board_force_calculated),
        _cross_moment(mast_water_position, mast_water_calculated),
        _cross_moment(rider_cg_state, rider_weight),
        _cross_moment(board_cg, board_weight),
    ]
    natural_moment = sum(moment_terms[1:], moment_terms[0])
    moment_residual = natural_moment + ca.vertcat(
        0, rider_pitch_moment, rider_yaw_moment
    )
    moment_scale = force_scale * max(
        float(sail["center_of_effort_m"][2]),
        float(vpp["foil"]["reynolds_chord_m"]),
        0.5,
    )
    opti.subject_to(force_residual / force_scale == 0)
    opti.subject_to(moment_residual / moment_scale == 0)

    # These are frozen-control, local static derivatives, not dynamic modes.
    yaw_stability_derivative = _state_derivative(natural_moment[2], leeway)
    roll_stability_derivative = _state_derivative(natural_moment[0], heel)
    minimum_yaw_stability = float(vpp["minimum_yaw_stability_Nm_per_rad"])
    minimum_roll_stability = float(vpp["minimum_roll_stability_Nm_per_rad"])
    opti.subject_to(yaw_stability_derivative <= -minimum_yaw_stability)
    opti.subject_to(roll_stability_derivative <= -minimum_roll_stability)

    objective = (
        -speed / speed_bounds[1]
        + 0.03 * (rider_lean / float(rider["max_lateral_cg_shift_m"])) ** 2
        + 0.03 * (rider_fore_aft_shift / fore_aft_authority) ** 2
        + 0.002 * (sail_alpha / sail_alpha_bound) ** 2
        + 0.002 * (foil_alpha / foil_alpha_bound) ** 2
        + 0.01 * (rider_yaw_moment / yaw_authority) ** 2
        + 0.01 * (rider_pitch_moment / pitch_authority) ** 2
    )
    board_drag_magnitude = vector_norm(board_drag_calculated)
    hydrodynamic_drag = (
        foil_drag + mast_water["drag_N"] + board_drag_magnitude
    )
    foil_finesse = foil_lift / (foil_drag + 1e-8)
    hydrodynamic_finesse = foil_lift / (hydrodynamic_drag + 1e-8)
    if not standalone:
        return {
            "name": point,
            "true_wind_speed_m_s": float(point_data["true_wind_speed_m_s"]),
            "true_wind_angle_deg": float(point_data["true_wind_angle_deg"]),
            "flight_height_input_m": float(point_data["flight_height_m"]),
            "ride_height_m": ride_height,
            "boat_speed_m_s": speed,
            "leeway_deg": leeway * (180 / math.pi),
            "heel_deg": heel * (180 / math.pi),
            "sail_alpha_deg": sail_alpha * (180 / math.pi),
            "foil_alpha_deg": foil_alpha * (180 / math.pi),
            "rider_lateral_cg_shift_m": rider_lean,
            "rider_fore_aft_cg_shift_m": rider_fore_aft_shift,
            "rider_yaw_control_moment_Nm": rider_yaw_moment,
            "rider_pitch_control_moment_Nm": rider_pitch_moment,
            "apparent_wind_speed_m_s": apparent_speed,
            "sail_cl": sail_cl,
            "sail_cd": sail_cd,
            "sail_reynolds": sail_re,
            "foil_cl": foil_cl,
            "foil_cd": foil_cd,
            "foil_reynolds": foil_re,
            "foil_lift_N": foil_lift,
            "foil_drag_N": foil_drag,
            "foil_finesse": foil_finesse,
            "board_drag_N": board_drag_magnitude,
            "hydrodynamic_drag_N": hydrodynamic_drag,
            "hydrodynamic_finesse": hydrodynamic_finesse,
            "air_mast_drag_N": (
                vector_norm(sum(air_mast_forces[1:], air_mast_forces[0]))
                if air_mast_forces
                else 0.0
            ),
            "water_mast_drag_N": mast_water["drag_N"],
            "water_mast_sideforce_N": mast_water["sideforce_N"],
            "immersed_mast_length_m": immersed_mast_length,
            "water_mast_cd": mast_water["cd"],
            "water_mast_cy": mast_water["cy"],
            "force_residual_N": force_residual,
            "moment_residual_Nm": moment_residual,
            "yaw_stability_derivative_Nm_per_rad": yaw_stability_derivative,
            "roll_stability_derivative_Nm_per_rad": roll_stability_derivative,
            "minimum_yaw_stability_Nm_per_rad": minimum_yaw_stability,
            "minimum_roll_stability_Nm_per_rad": minimum_roll_stability,
            "rider_cg_m": rider_cg_state,
            "state_bounds": {
                "boat_speed_m_s": speed_bounds,
                "ride_height_m": ride_height_bounds,
                "leeway_deg": (
                    -float(vpp["max_leeway_deg"]),
                    float(vpp["max_leeway_deg"]),
                ),
                "heel_deg": (
                    -float(vpp["max_heel_deg"]),
                    float(vpp["max_heel_deg"]),
                ),
                "sail_alpha_deg": (
                    -float(sail["max_alpha_deg"]),
                    float(sail["max_alpha_deg"]),
                ),
                "foil_alpha_deg": (
                    -float(foil["max_alpha_deg"]),
                    float(foil["max_alpha_deg"]),
                ),
                "rider_lateral_cg_shift_m": (
                    -float(rider["max_lateral_cg_shift_m"]),
                    float(rider["max_lateral_cg_shift_m"]),
                ),
                "rider_fore_aft_cg_shift_m": (
                    -fore_aft_authority,
                    fore_aft_authority,
                ),
                "rider_yaw_control_moment_Nm": (-yaw_authority, yaw_authority),
                "rider_pitch_control_moment_Nm": (
                    -pitch_authority,
                    pitch_authority,
                ),
            },
        }
    opti.minimize(objective)

    opti.solver(
        "ipopt",
        {
            "ipopt.max_iter": int(vpp["max_iterations"]),
            "ipopt.tol": float(vpp["tolerance"]),
            "ipopt.print_level": 0,
            "print_time": False,
        },
    )
    solution = opti.solve()
    value = lambda expression: float(solution.value(expression))

    limit_values = {
        "speed": (value(speed), speed_bounds),
        "leeway_deg": (
            math.degrees(value(leeway)),
            (-float(vpp["max_leeway_deg"]), float(vpp["max_leeway_deg"])),
        ),
        "heel_deg": (
            math.degrees(value(heel)),
            (-float(vpp["max_heel_deg"]), float(vpp["max_heel_deg"])),
        ),
        "sail_alpha_deg": (
            math.degrees(value(sail_alpha)),
            (-float(sail["max_alpha_deg"]), float(sail["max_alpha_deg"])),
        ),
        "foil_alpha_deg": (
            math.degrees(value(foil_alpha)),
            (-float(foil["max_alpha_deg"]), float(foil["max_alpha_deg"])),
        ),
        "rider_lateral_cg_shift_m": (
            value(rider_lean),
            (
                -float(rider["max_lateral_cg_shift_m"]),
                float(rider["max_lateral_cg_shift_m"]),
            ),
        ),
        "rider_fore_aft_cg_shift_m": (
            value(rider_fore_aft_shift),
            (-fore_aft_authority, fore_aft_authority),
        ),
        "rider_yaw_moment_Nm": (
            value(rider_yaw_moment),
            (-yaw_authority, yaw_authority),
        ),
        "rider_pitch_moment_Nm": (
            value(rider_pitch_moment),
            (-pitch_authority, pitch_authority),
        ),
        "ride_height_m": (value(ride_height), ride_height_bounds),
    }
    active_limits = [
        name
        for name, (actual, bounds) in limit_values.items()
        if min(abs(actual - bounds[0]), abs(actual - bounds[1]))
        < max(1e-3, 1e-3 * (bounds[1] - bounds[0]))
    ]
    force_residual_value = np.asarray(solution.value(force_residual)).reshape(-1)
    moment_residual_value = np.asarray(solution.value(moment_residual)).reshape(-1)

    def resultant_drag(forces):
        if not forces:
            return 0.0
        resultant = sum(forces[1:], forces[0])
        return value(vector_norm(resultant))

    yaw_stable = value(yaw_stability_derivative) <= -minimum_yaw_stability
    roll_stable = value(roll_stability_derivative) <= -minimum_roll_stability

    return {
        "name": point,
        "true_wind_speed_m_s": float(point_data["true_wind_speed_m_s"]),
        "true_wind_angle_deg": float(point_data["true_wind_angle_deg"]),
        "flight_height_input_m": float(point_data["flight_height_m"]),
        "boat_speed_m_s": value(speed),
        "leeway_deg": math.degrees(value(leeway)),
        "heel_deg": math.degrees(value(heel)),
        "sail_alpha_deg": math.degrees(value(sail_alpha)),
        "foil_alpha_deg": math.degrees(value(foil_alpha)),
        "rider_lateral_cg_shift_m": value(rider_lean),
        "rider_fore_aft_cg_shift_m": value(rider_fore_aft_shift),
        "rider_yaw_control_moment_Nm": value(rider_yaw_moment),
        "rider_pitch_control_moment_Nm": value(rider_pitch_moment),
        "apparent_wind_speed_m_s": value(apparent_speed),
        "sail_cl": value(sail_cl),
        "sail_cd": value(sail_cd),
        "sail_reynolds": value(sail_re),
        "foil_cl": value(foil_cl),
        "foil_cd": value(foil_cd),
        "foil_reynolds": value(foil_re),
        "foil_lift_N": value(foil_lift),
        "foil_drag_N": value(foil_drag),
        "foil_finesse": value(foil_finesse),
        "board_drag_N": value(board_drag_magnitude),
        "hydrodynamic_drag_N": value(hydrodynamic_drag),
        "hydrodynamic_finesse": value(hydrodynamic_finesse),
        "air_mast_drag_N": resultant_drag(air_mast_forces),
        "water_mast_drag_N": value(mast_water["drag_N"]),
        "water_mast_sideforce_N": value(mast_water["sideforce_N"]),
        "immersed_mast_length_m": value(immersed_mast_length),
        "ride_height_m": value(ride_height),
        "water_mast_cd": value(mast_water["cd"]),
        "water_mast_cy": value(mast_water["cy"]),
        "force_residual_N": force_residual_value.tolist(),
        "moment_residual_Nm": moment_residual_value.tolist(),
        "yaw_stability_derivative_Nm_per_rad": value(yaw_stability_derivative),
        "roll_stability_derivative_Nm_per_rad": value(roll_stability_derivative),
        "minimum_yaw_stability_Nm_per_rad": minimum_yaw_stability,
        "minimum_roll_stability_Nm_per_rad": minimum_roll_stability,
        "roll_stability_passed": roll_stable,
        "yaw_stability_passed": yaw_stable,
        "trim_valid": not active_limits and yaw_stable and roll_stable,
        "rider_cg_m": np.asarray(solution.value(rider_cg_state)).reshape(-1).tolist(),
        "active_limits": active_limits,
        "boundary_limited": bool(active_limits),
        "status": opti.stats().get("return_status", "unknown"),
    }


def run_vpp(config):
    vpp = config.get("vpp")
    if not vpp or not vpp.get("enabled"):
        raise ValueError(f"Scenario {config['name']!r} has no enabled VPP model.")
    points = [build_vpp_point(config, name) for name in vpp["operating_points"]]
    foil_aero_model = (
        config["hydrodynamics"]["solver"]
        if "hydrodynamics" in config
        else "analytic_finite_wing"
    )
    return {
        "scenario": config["name"],
        "architecture": "SAND: simultaneous discipline states and coupling residuals",
        "model": (
            f"quasi-static SAND trim with {foil_aero_model} foil loads, "
            "NeuralFoil profile drag when a 3D foil model is available, and "
            "lifting-line immersed-mast loads"
        ),
        "validation_status": "exploratory; not validated against on-water measurements",
        "design_usable": all(point["trim_valid"] for point in points),
        "assumptions": vpp["assumptions"],
        "points": points,
    }


def write_vpp_report(result):
    folder = new_output_directory(f"{result['scenario']}_vpp")
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "vpp.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Preliminary VPP — {result['scenario']}",
        "",
        f"Status: **{result['validation_status']}**.",
        "",
        f"Architecture: **{result['architecture']}**.",
        "",
        result["model"],
        "",
        "## Assumptions and limitations",
        "",
        *[f"- {assumption}" for assumption in result["assumptions"]],
        "",
        "## Equilibrium points",
        "",
        "| Point | Wind (m/s, TWA) | Speed | Height input | Ride height | Leeway | Heel | Sail α | Foil α | Pitch ctrl | Yaw ctrl |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for point in result["points"]:
        lines.append(
            f"| {point['name']} | {point['true_wind_speed_m_s']:.1f}, "
            f"{point['true_wind_angle_deg']:.0f}° | {point['boat_speed_m_s']:.2f} m/s | "
            f"{point['flight_height_input_m']:.2f} m | "
            f"{point['ride_height_m']:.2f} m | "
            f"{point['leeway_deg']:.1f}° | {point['heel_deg']:.1f}° | "
            f"{point['sail_alpha_deg']:.1f}° | {point['foil_alpha_deg']:.1f}° | "
            f"{point['rider_pitch_control_moment_Nm']:.1f} Nm | "
            f"{point['rider_yaw_control_moment_Nm']:.1f} Nm |"
        )
        if not point["trim_valid"]:
            lines.append(
                f"\nWarning: `{point['name']}` is not an interior, unconstrained trim. "
                f"Active limits: {', '.join(point['active_limits']) or 'none'}; "
                f"roll stable: {point['roll_stability_passed']}; "
                f"yaw stable: {point['yaw_stability_passed']}."
            )
        if point["boundary_limited"]:
            lines.append(
                f"\nWarning: `{point['name']}` is boundary-limited at "
                f"{', '.join(point['active_limits'])}; treat it as an infeasible "
                "or under-specified operating point, not a performance prediction."
            )
    lines.extend(
        [
            "",
            "## Yaw, mast drag, and static derivatives",
            "",
            "| Point | Air mast drag | Water mast drag | Water mast sideforce | Immersed mast | Hydro L/D | dMroll/dheel | dNyaw/dleeway | static constraints |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|:--|",
        ]
    )
    for point in result["points"]:
        lines.append(
            f"| {point['name']} | {point['air_mast_drag_N']:.1f} N | "
            f"{point['water_mast_drag_N']:.1f} N | "
            f"{point['water_mast_sideforce_N']:.1f} N | "
            f"{point['immersed_mast_length_m']:.2f} m | "
            f"{point['hydrodynamic_finesse']:.2f} | "
            f"{point['roll_stability_derivative_Nm_per_rad']:.1f} Nm/rad | "
            f"{point['yaw_stability_derivative_Nm_per_rad']:.1f} Nm/rad | "
            f"roll {point['roll_stability_passed']}, yaw {point['yaw_stability_passed']} |"
        )
    lines.extend(
        [
            "",
            f"\nOverall trim usable as a design point: **{result['design_usable']}**. "
            "The roll and yaw derivatives are frozen-control static derivatives, not dynamic stability modes. "
            "The input flight height seeds an optimizer-selected, bounded ride height and mast immersion. "
            "This is not a physical ride-height prediction: free-surface lift, board buoyancy, and "
            "height-control dynamics are not calibrated.",
            "",
            "Force/moment closure and discipline residuals are in `vpp.json`. These calculations do not "
            "establish prototype safety or validated performance.",
            "",
        ]
    )
    (folder / "vpp.md").write_text("\n".join(lines), encoding="utf-8")
    return folder
