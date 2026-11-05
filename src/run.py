import argparse

from foilopt.config import SCENARIO_DIR, load_config
from foilopt.problem import lifting_line_check, optimize
from foilopt.report import new_output_directory, write_report
from foilopt.vpp import run_vpp, write_vpp_report


def main():
    parser = argparse.ArgumentParser(description="Optimize a configured hydrofoil use scenario.")
    parser.add_argument(
        "--scenario",
        help="Scenario name (from config/scenarios/) or YAML file path.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List configured scenarios and model implementation status.",
    )
    parser.add_argument("--starts", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument(
        "--mode",
        choices=("design", "vpp"),
        default="design",
        help=(
            "Run the foil optimizer (coupled VPP when enabled) or the standalone "
            "exploratory quasi-static VPP."
        ),
    )
    args = parser.parse_args()
    if args.list_scenarios:
        for path in sorted(SCENARIO_DIR.glob("*.yaml")):
            config = load_config(path)
            status = "implemented" if config["implemented_model"] else "conditions only"
            if config.get("vpp", {}).get("enabled"):
                status += ", exploratory VPP available"
            print(f"{path.stem}: {config['name']} ({status})")
        return
    if args.starts is not None and args.starts < 1:
        parser.error("--starts must be positive")
    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error("--max-iterations must be positive")

    config = load_config(args.scenario)
    if args.mode == "vpp":
        if args.starts is not None or args.max_iterations is not None:
            parser.error("--starts and --max-iterations apply only to --mode design.")
        result = run_vpp(config)
        folder = write_vpp_report(result)
        for point in result["points"]:
            print(
                f"{point['name']}: {point['boat_speed_m_s']:.2f} m/s · "
                f"heel {point['heel_deg']:.1f}° · "
                f"leeway {point['leeway_deg']:.1f}°"
            )
        print(f"Exploratory VPP report: {folder / 'vpp.md'}")
        return
    if not config["implemented_model"]:
        parser.error(
            f"Scenario {config['name']!r} is conditions-only: "
            f"{config['not_implemented_reason']}"
        )
    result = optimize(config, args.starts, args.max_iterations)
    lifting_line = lifting_line_check(result, config)
    folder = new_output_directory(config["name"])
    write_report(result, config, lifting_line, folder)
    design = result["design"]
    print(
        f"Model-feasible: {result['feasible']} · span {design['span']:.3f} m · "
        f"area {result['geometry']['area_m2']*1e4:.0f} cm² · "
        f"AR {result['aspect_ratio']:.2f}"
    )
    for point in result["vpp_points"].values():
        print(
            f"VPP {point['name']}: {point['boat_speed_m_s']:.2f} m/s · "
            f"hydro L/D {point['hydrodynamic_finesse']:.2f} · "
            f"mast immersion {point['immersed_mast_length_m']:.2f} m · "
            f"trim {'valid' if point['trim_valid'] else 'boundary-limited'}"
        )
    print(f"LiftingLine trimmed: {lifting_line['trimmed']}")
    print(f"Report: {folder / 'datasheet.md'}")


if __name__ == "__main__":
    main()
