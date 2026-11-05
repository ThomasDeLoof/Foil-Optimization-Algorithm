import math

import aerosandbox as asb
import aerosandbox.numpy as anp


class Water(asb.AeroSandboxObject):
    def __init__(
        self,
        depth_m,
        density_kg_m3,
        dynamic_viscosity_pa_s,
        temperature_k,
        gravity_m_s2,
    ):
        self.depth_m = depth_m
        self._density = density_kg_m3
        self._dynamic_viscosity = dynamic_viscosity_pa_s
        self._temperature = temperature_k
        self._gravity = gravity_m_s2

    def __repr__(self):
        return f"Water(depth={self.depth_m} m)"

    def __len__(self):
        return 1

    def __getitem__(self, index):
        depth = self.depth_m[index] if hasattr(self.depth_m, "__getitem__") else self.depth_m
        return Water(
            depth,
            self._density,
            self._dynamic_viscosity,
            self._temperature,
            self._gravity,
        )

    def __array__(self, dtype=None):
        return anp.fromiter([self], dtype=object).reshape(())

    def density(self):
        return self._density

    def pressure(self):
        return 101325.0 + self._density * self._gravity * self.depth_m

    def temperature(self):
        return self._temperature

    def dynamic_viscosity(self):
        return self._dynamic_viscosity

    def kinematic_viscosity(self):
        return self._dynamic_viscosity / self._density

    def speed_of_sound(self):
        return 1500.0

    def vapor_pressure(self):
        temperature_c = self._temperature - 273.15
        pressure_mmhg = 10 ** (8.07131 - 1730.63 / (233.426 + temperature_c))
        return pressure_mmhg * 133.3224


def water_from_config(config):
    fluid = config["fluid"]
    return Water(
        depth_m=float(fluid["immersion_depth_m"]),
        density_kg_m3=float(fluid["density_kg_m3"]),
        dynamic_viscosity_pa_s=float(fluid["dynamic_viscosity_pa_s"]),
        temperature_k=float(fluid["temperature_k"]),
        gravity_m_s2=float(fluid["gravity_m_s2"]),
    )


def evaluate_aero(
    airplane,
    speed_m_s,
    alpha_deg,
    water,
    model_size,
    solver="lifting_line",
    spanwise_resolution=4,
    chordwise_resolution=2,
    beta_deg=0,
):
    operating_point = asb.OperatingPoint(
        velocity=speed_m_s,
        alpha=alpha_deg,
        beta=beta_deg,
        atmosphere=water,
    )
    if solver == "lifting_line":
        analysis = asb.LiftingLine(
            airplane=airplane,
            op_point=operating_point,
            model_size=model_size,
            spanwise_resolution=spanwise_resolution,
        )
    elif solver == "vortex_lattice":
        analysis = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=operating_point,
            spanwise_resolution=spanwise_resolution,
            chordwise_resolution=chordwise_resolution,
            align_trailing_vortices_with_wind=True,
        )
        result = analysis.run()
        main_wing_airfoil = airplane.wings[0].xsecs[0].airfoil
        reynolds = (
            float(water.density())
            * speed_m_s
            * airplane.c_ref
            / float(water.dynamic_viscosity())
        )
        profile_polar = main_wing_airfoil.get_aero_from_neuralfoil(
            alpha=alpha_deg,
            Re=reynolds,
            model_size=model_size,
        )
        profile_drag = profile_polar["CD"]
        dynamic_pressure = 0.5 * float(water.density()) * speed_m_s**2
        profile_drag_force = dynamic_pressure * airplane.s_ref * profile_drag
        result["D"] += profile_drag_force
        result["CD"] += profile_drag
        return result
    elif solver == "aero_buildup":
        analysis = asb.AeroBuildup(
            airplane=airplane,
            op_point=operating_point,
            model_size=model_size,
            include_wave_drag=False,
        )
    else:
        raise ValueError(f"Unsupported aerodynamic solver: {solver}")
    return analysis.run()


def immersed_mast_lifting_line(
    speed_m_s,
    leeway_rad,
    ride_height_m,
    mast,
    config,
    spanwise_resolution=2,
):
    immersed_length = float(mast["length_m"]) - ride_height_m
    section_count = int(mast["sections"])
    sections = [
        asb.WingXSec(
            xyz_le=[
                float(mast["x_m"]),
                float(mast["y_m"]),
                -immersed_length * index / (section_count - 1),
            ],
            chord=float(mast["projected_width_m"]),
            airfoil=asb.Airfoil(mast["airfoil"]),
        )
        for index in range(section_count)
    ]
    mast_wing = asb.Wing(name="ImmersedMast", symmetric=False, xsecs=sections)
    mast_airplane = asb.Airplane(
        wings=[mast_wing],
        xyz_ref=[float(mast["x_m"]), float(mast["y_m"]), 0],
    )
    water = water_from_config(config)
    result = asb.LiftingLine(
        airplane=mast_airplane,
        op_point=asb.OperatingPoint(
            velocity=speed_m_s,
            alpha=0,
            beta=leeway_rad * (180 / math.pi),
            atmosphere=water,
        ),
        model_size="small",
        spanwise_resolution=spanwise_resolution,
        align_trailing_vortices_with_wind=True,
    ).run()
    return {
        "drag_N": result["D"],
        "sideforce_N": result["Y"],
        "cd": result["CD"],
        "cy": result["CY"],
        "immersed_length_m": immersed_length,
    }


def mast_drag(speed_m_s, config, model_size):
    mast = config["mast"]
    fluid = config["fluid"]
    chord = float(mast["chord_m"])
    reynolds = (
        float(fluid["density_kg_m3"]) * speed_m_s * chord
        / float(fluid["dynamic_viscosity_pa_s"])
    )
    thickness = int(round(float(mast["thickness_ratio"]) * 100))
    cd = asb.Airfoil(f"naca00{thickness:02d}").get_aero_from_neuralfoil(
        alpha=0,
        Re=reynolds,
        model_size=model_size,
    )["CD"]
    return (
        0.5 * float(fluid["density_kg_m3"]) * speed_m_s**2 * chord
        * float(fluid["immersion_depth_m"]) * cd
    )
