import casadi as ca
import aerosandbox.numpy as anp


def _laminate_modulus(laminate):
    e1 = float(laminate["fiber_modulus_pa"])
    e2 = float(laminate["transverse_modulus_pa"])
    g12 = float(laminate["shear_modulus_pa"])
    nu12 = float(laminate["poisson_ratio"])
    nu21 = nu12 * e2 / e1
    denominator = 1 - nu12 * nu21
    q11, q22 = e1 / denominator, e2 / denominator
    q12, q66 = nu12 * e2 / denominator, g12

    qbar = []
    for angle in laminate["ply_angles_deg"]:
        angle_rad = float(angle) * anp.pi / 180
        cosine, sine = anp.cos(angle_rad), anp.sin(angle_rad)
        qbar.append(
            q11 * cosine**4
            + 2 * (q12 + 2 * q66) * sine**2 * cosine**2
            + q22 * sine**4
        )
    return sum(qbar) / len(qbar)


def _smooth_max(values, beta):
    return ca.logsumexp(beta * ca.densify(values)) / beta


def _section_properties(
    chord,
    thickness_ratio,
    upper,
    lower,
    dx,
    skin_thickness,
    carbon_modulus,
    foam_modulus,
    smooth_max_beta,
):
    depth = chord * thickness_ratio
    skin_area = skin_thickness * chord * dx
    upper_skin_z = upper * depth - skin_thickness / 2
    lower_skin_z = lower * depth + skin_thickness / 2
    foam_area = chord**2 * (upper - lower) * dx
    foam_z = (upper + lower) * depth / 2

    axial_rigidity = (
        2 * carbon_modulus * skin_area
        + foam_modulus * anp.sum(foam_area)
    )
    neutral_axis = (
        carbon_modulus * skin_area * anp.sum(upper_skin_z + lower_skin_z)
        + foam_modulus * anp.sum(foam_area * foam_z)
    ) / axial_rigidity
    skin_inertia = skin_area * anp.sum(
        (upper_skin_z - neutral_axis) ** 2
        + (lower_skin_z - neutral_axis) ** 2
    )
    core_inertia = chord**2 * dx * anp.sum(
        (
            (upper * depth - neutral_axis) ** 3
            - (lower * depth - neutral_axis) ** 3
        )
        / 3
    )
    bending_stiffness = (
        carbon_modulus * skin_inertia + foam_modulus * core_inertia
    )
    core_area = anp.sum(foam_area) - 2 * skin_thickness * chord
    relative_skin_distances = anp.concatenate(
        (
            anp.abs(upper * depth - neutral_axis) / depth,
            anp.abs(lower * depth - neutral_axis) / depth,
        )
    )
    max_skin_distance = _smooth_max(
        relative_skin_distances, smooth_max_beta
    ) * depth
    return {
        "bending_stiffness_Nm2": bending_stiffness,
        "neutral_axis_m": neutral_axis,
        "core_area_m2": core_area,
        "skin_thickness_m": skin_thickness,
        "carbon_modulus_pa": carbon_modulus,
        "max_skin_distance_m": max_skin_distance,
    }


def sandwich_screen(span_m, root_chord_m, taper, profile, thickness_ratio, config):
    structure = config["structure"]
    laminate = structure["laminate"]
    core = structure["foam_core"]
    rider = config["rider"]
    gravity = float(config["fluid"]["gravity_m_s2"])
    max_g = float(config["maneuvers"]["maximum_positive_g"])
    safety_factor = float(structure["material_safety_factor"])
    wing_load_share = float(structure["front_wing_load_share_factor"])
    design_load = (
        float(rider["design_mass_kg"]) + float(rider["board_mass_kg"])
    ) * gravity * max_g * safety_factor * wing_load_share

    station_count = int(structure["spanwise_stations"])
    spanwise_fraction = anp.linspace(0, 1, station_count)
    profile_samples = int(structure["profile_samples"])
    x = anp.linspace(0.01, 0.99, profile_samples)
    dx = 0.98 / (profile_samples - 1)
    upper = profile.upper_coordinates(x_over_c=x)[:, 1]
    lower = profile.lower_coordinates(x_over_c=x)[:, 1]
    skin_thickness = (
        float(laminate["fiber_areal_weight_kg_m2"])
        / (
            float(laminate["fiber_density_kg_m3"])
            * float(laminate["fiber_volume_fraction"])
        )
        * len(laminate["ply_angles_deg"])
    )
    carbon_modulus = _laminate_modulus(laminate)
    foam_modulus = float(core["elastic_modulus_pa"])
    semispan = span_m / 2
    stations = spanwise_fraction * semispan
    chord = root_chord_m * (1 - (1 - taper) * spanwise_fraction)
    sections = [
        _section_properties(
            chord[index],
            thickness_ratio,
            upper,
            lower,
            dx,
            skin_thickness,
            carbon_modulus,
            foam_modulus,
            float(structure["smooth_max_beta"]),
        )
        for index in range(station_count)
    ]
    stiffness = anp.array(
        [section["bending_stiffness_Nm2"] for section in sections]
    )
    core_area = anp.array([section["core_area_m2"] for section in sections])
    skin_distance = anp.array(
        [section["max_skin_distance_m"] for section in sections]
    )

    # Approximate half-wing loading by an elliptical distribution.
    midpoints = (stations[:-1] + stations[1:]) / 2
    shape = anp.sqrt(1 - spanwise_fraction[:-1] ** 2)
    cell_lift = (design_load / 2) * shape / anp.sum(shape)
    shear, bending_moment = [], []
    for index in range(station_count):
        station = stations[index]
        outboard_midpoints = midpoints[index:]
        outboard_lift = cell_lift[index:]
        shear.append(anp.sum(outboard_lift))
        bending_moment.append(
            anp.sum(outboard_lift * (outboard_midpoints - station))
        )

    shear = anp.array(shear)
    bending_moment = anp.array(bending_moment)
    curvature = bending_moment / stiffness
    carbon_allowable = min(
        float(laminate["screening_allowables_pa"]["tension_pa"]),
        float(laminate["screening_allowables_pa"]["compression_pa"]),
    )
    carbon_utilization = _smooth_max(
        anp.abs(curvature) * skin_distance * carbon_modulus / carbon_allowable,
        float(structure["smooth_max_beta"]),
    )
    core_shear_allowable = float(core["screening_shear_allowable_pa"])
    foam_shear_utilization = _smooth_max(
        1.5 * shear / core_area / core_shear_allowable,
        float(structure["smooth_max_beta"]),
    )
    carbon_stress = carbon_utilization * carbon_allowable
    foam_shear = foam_shear_utilization * core_shear_allowable
    tip_deflection = anp.sum(
        (curvature[:-1] + curvature[1:])
        / 2
        * (stations[1:] - stations[:-1])
        * (semispan - midpoints)
    )
    shear_modulus = float(core["shear_modulus_pa"])
    shear_correction = float(structure["shear_correction_factor"])
    tip_shear_deflection = anp.sum(
        (shear[:-1] / core_area[:-1] + shear[1:] / core_area[1:])
        / (2 * shear_correction * shear_modulus)
        * (stations[1:] - stations[:-1])
    )
    total_tip_deflection = tip_deflection + tip_shear_deflection
    root = sections[0]
    root_core_depth = (
        root_chord_m * thickness_ratio - 2 * root["skin_thickness_m"]
    )
    section_core_depths = [
        root_chord_m
        * (1 - (1 - taper) * spanwise_fraction[index])
        * thickness_ratio
        - 2 * skin_thickness
        for index in range(station_count)
    ]
    return {
        "root_moment_Nm": bending_moment[0],
        "root_shear_N": shear[0],
        "design_load_N": design_load,
        "maximum_positive_g": max_g,
        "material_safety_factor": safety_factor,
        "front_wing_load_share_factor": wing_load_share,
        "total_depth_m": root_chord_m * thickness_ratio,
        "core_depth_m": root_core_depth,
        "minimum_core_depth_m": ca.mmin(ca.vertcat(*section_core_depths)),
        "core_area_m2": root["core_area_m2"],
        "skin_thickness_m": root["skin_thickness_m"],
        "carbon_effective_modulus_Pa": root["carbon_modulus_pa"],
        "carbon_stress_Pa": carbon_stress,
        "carbon_strain": carbon_stress / root["carbon_modulus_pa"],
        "foam_shear_Pa": foam_shear,
        "tip_bending_deflection_m": tip_deflection,
        "tip_shear_deflection_m": tip_shear_deflection,
        "tip_deflection_m": total_tip_deflection,
        "tip_deflection_ratio": total_tip_deflection / semispan,
        "carbon_utilization": carbon_utilization,
        "foam_shear_utilization": foam_shear_utilization,
        "core_depth_utilization": root_core_depth
        / float(structure["minimum_root_core_thickness_m"]),
        "deflection_utilization": (total_tip_deflection / semispan)
        / float(structure["maximum_tip_deflection_ratio"]),
    }
