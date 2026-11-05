import math

import casadi as ca
import numpy as np


AIR_DENSITY_KG_M3 = 1.225
AIR_DYNAMIC_VISCOSITY_PA_S = 1.81e-5


def vector_norm(vector):
    return ca.sqrt(ca.dot(vector, vector) + 1e-8)


def finite_wing_polar(alpha_rad, speed_m_s, surface, reynolds_chord_m, density, viscosity):
    reynolds = (
        density
        * ca.sqrt(speed_m_s**2 + 1e-8)
        * reynolds_chord_m
        / viscosity
    )
    cl_max = float(surface["cl_max"])
    cl = cl_max * ca.tanh(float(surface["cl_alpha_per_rad"]) * alpha_rad / cl_max)
    aspect_ratio = surface["span_m"] ** 2 / surface["area_m2"]
    induced_drag = cl**2 / (
        math.pi * float(surface["oswald_efficiency"]) * aspect_ratio
    )
    # Flat-plate skin friction is only a low-order viscous term, not a sail polar.
    skin_friction = 1.328 / ca.sqrt(reynolds + 1)
    cd = float(surface["cd0"]) + induced_drag + skin_friction
    return cl, cd, reynolds


def sail_force(apparent_wind, apparent_speed, alpha_rad, sail):
    cl, cd, reynolds = finite_wing_polar(
        alpha_rad,
        apparent_speed,
        sail,
        float(sail["reynolds_chord_m"]),
        AIR_DENSITY_KG_M3,
        AIR_DYNAMIC_VISCOSITY_PA_S,
    )
    direction = apparent_wind / apparent_speed
    lift_direction = ca.vertcat(-direction[1], direction[0])
    dynamic_pressure = 0.5 * AIR_DENSITY_KG_M3 * apparent_speed**2
    force_xy = dynamic_pressure * float(sail["area_m2"]) * (
        cd * direction + cl * lift_direction
    )
    return ca.vertcat(force_xy[0], force_xy[1], 0), cl, cd, reynolds


def distributed_air_mast_forces(apparent_wind, apparent_speed, mast):
    count = int(mast["segments"])
    dz = float(mast["exposed_length_m"]) / count
    width = float(mast["projected_width_m"])
    cd = float(mast["drag_coefficient"])
    dynamic_pressure = 0.5 * AIR_DENSITY_KG_M3 * apparent_speed**2
    direction = apparent_wind / apparent_speed
    force = dynamic_pressure * cd * width * dz * direction
    forces = [ca.vertcat(force[0], force[1], 0) for _ in range(count)]
    positions = [
        ca.vertcat(
            float(mast["x_m"]),
            float(mast["y_m"]),
            float(mast["base_height_m"]) + (index + 0.5) * dz,
        )
        for index in range(count)
    ]
    return forces, positions


def foil_force(
    water_velocity,
    water_speed,
    leeway,
    heel,
    alpha_rad,
    foil,
    fluid,
):
    cl, cd, reynolds = finite_wing_polar(
        alpha_rad,
        water_speed,
        foil,
        foil["reynolds_chord_m"],
        float(fluid["density_kg_m3"]),
        float(fluid["dynamic_viscosity_pa_s"]),
    )
    dynamic_pressure = 0.5 * float(fluid["density_kg_m3"]) * water_speed**2
    lift = dynamic_pressure * foil["area_m2"] * cl
    drag = dynamic_pressure * foil["area_m2"] * cd
    force_body = ca.vertcat(
        -drag * ca.cos(leeway),
        -drag * ca.sin(leeway),
        lift,
    )
    cos_heel, sin_heel = ca.cos(heel), ca.sin(heel)
    force = ca.vertcat(
        force_body[0],
        force_body[1] * cos_heel - force_body[2] * sin_heel,
        force_body[1] * sin_heel + force_body[2] * cos_heel,
    )
    return force, cl, cd, reynolds, lift, drag


def rider_cg(rider, lean, fore_aft_shift):
    fractions = rider["segment_mass_fractions"]
    positions = rider["segment_cg_positions_m"]
    lean_factors = rider["segment_lean_factors"]
    fore_aft_factors = rider["segment_fore_aft_factors"]
    footstraps = np.asarray(rider["footstrap_positions_m"], dtype=float)
    handholds = np.asarray(rider["handhold_positions_m"], dtype=float)
    foot_center = footstraps.mean(axis=0)
    hand_center = handholds.mean(axis=0)
    shoulder = np.asarray(positions["upper_arms"], dtype=float)
    cg = ca.vertcat(0, 0, 0)
    for segment, fraction in fractions.items():
        x, y, z = map(float, positions[segment])
        if segment == "shanks_feet":
            x, y = map(float, foot_center[:2])
        elif segment == "forearms_hands":
            x, y, z = map(float, (shoulder + hand_center) / 2)
        cg += float(fraction) * ca.vertcat(
            x + float(fore_aft_factors[segment]) * fore_aft_shift,
            y + float(lean_factors[segment]) * lean,
            z,
        )
    return cg
