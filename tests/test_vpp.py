import sys
import tempfile
import unittest
from pathlib import Path

import casadi as ca
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foilopt.config import load_config
from foilopt.hydrodynamics import immersed_mast_lifting_line
from foilopt.problem import build_problem, optimize
from foilopt.report import write_report
from foilopt.vpp import build_vpp_point
from foilopt.vpp_disciplines import finite_wing_polar


ROOT = Path(__file__).resolve().parents[1]


class VPPTests(unittest.TestCase):
    def test_polar_derivative_matches_finite_difference(self):
        alpha = ca.MX.sym("alpha")
        speed = ca.MX.sym("speed")
        surface = {
            "area_m2": 4.0,
            "span_m": 3.464101615,
            "cl_max": 1.1,
            "cl_alpha_per_rad": 4.8,
            "oswald_efficiency": 0.82,
            "cd0": 0.03,
        }
        cl, _, _ = finite_wing_polar(
            alpha, speed, surface, 0.8, 1.225, 1.81e-5
        )
        polar = ca.Function(
            "polar", [alpha, speed], [cl, ca.jacobian(cl, alpha)]
        )
        alpha_value, speed_value, step = 0.12, 9.0, 1e-6
        cl_value, derivative = polar(alpha_value, speed_value)
        plus = polar(alpha_value + step, speed_value)[0]
        minus = polar(alpha_value - step, speed_value)[0]
        finite_difference = float((plus - minus) / (2 * step))
        self.assertAlmostEqual(float(derivative), finite_difference, places=5)
        self.assertTrue(np.isfinite(float(cl_value)))

    def test_both_vpp_scenarios_solve_static_equilibrium(self):
        scenarios = (
            ("wingfoil-freeride.yaml", "crosswind"),
            ("windfoil-freeride.yaml", "cruise"),
        )
        for scenario, point in scenarios:
            with self.subTest(scenario=scenario):
                config = load_config(ROOT / "config" / "scenarios" / scenario)
                result = build_vpp_point(config, point)
                self.assertEqual(result["status"], "Solve_Succeeded")
                self.assertLess(max(map(abs, result["force_residual_N"])), 0.1)
                self.assertEqual(
                    len(result["moment_residual_Nm"]), 3
                )
                self.assertLess(max(map(abs, result["moment_residual_Nm"])), 0.1)
                self.assertLess(
                    result["yaw_stability_derivative_Nm_per_rad"],
                    -result["minimum_yaw_stability_Nm_per_rad"],
                )
                self.assertTrue(result["yaw_stability_passed"])
                self.assertTrue(result["roll_stability_passed"])
                self.assertGreaterEqual(result["water_mast_drag_N"], 0)
                self.assertNotEqual(result["water_mast_sideforce_N"], 0)
                self.assertGreater(result["immersed_mast_length_m"], 0)

    def test_design_problem_couples_vpp_to_optimized_foil_geometry(self):
        config = load_config(ROOT / "config" / "scenarios" / "wingfoil-freeride.yaml")
        _, model = build_problem(config)
        crosswind_finesse = model["vpp_points"]["crosswind"]["hydrodynamic_finesse"]
        self.assertTrue(ca.depends_on(crosswind_finesse, model["design"]["span"]))
        sweep = ca.symvar(model["design"]["sweep_deg"])[0]
        self.assertTrue(ca.depends_on(crosswind_finesse, sweep))
        root_chord = ca.symvar(model["design"]["root_chord"])[0]
        self.assertTrue(ca.depends_on(crosswind_finesse, root_chord))
        profile_weight = ca.symvar(model["profile_variables"]["upper"])[0]
        self.assertTrue(ca.depends_on(crosswind_finesse, profile_weight))

    def test_design_optimizer_returns_stable_accessible_vpp_trims(self):
        config = load_config(ROOT / "config" / "scenarios" / "wingfoil-freeride.yaml")
        result = optimize(config, starts=1, max_iterations=800)

        self.assertEqual(
            result["objective_name"], "maximize_mean_vpp_hydrodynamic_finesse"
        )
        self.assertEqual(set(result["vpp_points"]), {"crosswind", "upwind_reach"})
        self.assertTrue(result["checks"]["vpp_state_bounds"])
        self.assertTrue(result["checks"]["vpp_force_moment_balance"])
        self.assertTrue(result["checks"]["vpp_roll_stability"])
        self.assertTrue(result["checks"]["vpp_yaw_stability"])
        self.assertTrue(result["checks"]["minimum_spanwise_core"])
        self.assertGreaterEqual(
            result["structure"]["minimum_core_depth_m"],
            config["structure"]["minimum_core_thickness_m"] - 1e-5,
        )
        for point in result["vpp_points"].values():
            with self.subTest(point=point["name"]):
                self.assertGreater(point["boat_speed_m_s"], 0)
                self.assertGreater(point["hydrodynamic_finesse"], 0)
                self.assertLess(max(map(abs, point["force_residual_N"])), 0.1)
                self.assertLess(max(map(abs, point["moment_residual_Nm"])), 0.1)
                self.assertTrue(point["roll_stability_passed"])
                self.assertTrue(point["yaw_stability_passed"])
                self.assertTrue(point["trim_valid"])
                self.assertTrue(
                    all(
                        margin >= -1e-5
                        for state in point["state_margins"].values()
                        for margin in state.values()
                    )
                )
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_folder = Path(temporary_directory) / "design"
            write_report(
                result,
                config,
                {"trimmed": False, "message": "No independent trim in test."},
                report_folder,
            )
            report = (report_folder / "datasheet.md").read_text(encoding="utf-8")
            self.assertIn("Coupled VPP trims", report)
            self.assertIn("maximize_mean_vpp_hydrodynamic_finesse", report)

    def test_mast_lifting_line_couples_sideforce_and_drag_to_immersion(self):
        config = load_config(ROOT / "config" / "scenarios" / "wingfoil-freeride.yaml")
        beta = ca.MX.sym("beta")
        ride_height = ca.MX.sym("ride_height")
        mast = immersed_mast_lifting_line(
            8.0,
            beta,
            ride_height,
            config["vpp"]["water_mast"],
            config,
        )
        self.assertTrue(
            ca.depends_on(mast["drag_N"], ride_height)
        )
        self.assertTrue(
            ca.depends_on(mast["sideforce_N"], ca.symvar(beta)[0])
        )
        loads = ca.Function(
            "mast_loads",
            [ca.symvar(ride_height)[0], ca.symvar(beta)[0]],
            [mast["drag_N"], mast["sideforce_N"]],
        )
        shallow_drag, shallow_sideforce = loads(0.08, 5.0)
        high_drag, high_sideforce = loads(0.30, 5.0)
        self.assertGreater(float(shallow_drag), float(high_drag))
        self.assertGreater(abs(float(shallow_sideforce)), abs(float(high_sideforce)))


if __name__ == "__main__":
    unittest.main()
