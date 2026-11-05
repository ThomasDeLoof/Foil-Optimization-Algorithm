import datetime as dt
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs"


def new_output_directory(scenario_name):
    prefix = f"{scenario_name}_{dt.datetime.now():%Y%m%d}_design"
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    numbers = [
        int(match.group(1))
        for path in OUTPUT_ROOT.iterdir()
        if (match := pattern.match(path.name))
    ] if OUTPUT_ROOT.exists() else []
    return OUTPUT_ROOT / f"{prefix}_{max(numbers, default=0) + 1:02d}"


def write_report(result, config, lifting_line, folder):
    folder.mkdir(parents=True, exist_ok=False)
    data = dict(result)
    data["lifting_line_check"] = lifting_line
    data["inputs"] = config
    (folder / "design.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    design, geometry = result["design"], result["geometry"]
    structure, stability = result["structure"], result["stability"]
    references = config["references"]
    board = references.get("board")
    board_lines = (
        [
            f"- Board image reference (not modeled): "
            f"{board['length_m']*100:.1f} × {board['width_m']*100:.1f} × "
            f"{board['thickness_m']*100:.1f} cm, {board['volume_m3']*1000:.2f} L; "
            "no 3D geometry or mass properties.",
            f"- Candidate foil-track stations from the tail: "
            f"{board['candidate_foil_track_stations_in']} in; strap symbols and CG require confirmation.",
        ]
        if board else []
    )
    wing_sail = references.get("wing_sail")
    wing_sail_lines = (
        [
            f"- Wing-sail 3MF reference: {', '.join(wing_sail['files'])}; "
            f"full-model extents {', '.join(f'{axis*1000:.0f}' for axis in wing_sail['full_model_bounds_m'])} mm; "
            f"full-size scale verified: {wing_sail['verified_full_size_scale']}.",
            f"- Wing-sail limitation: {wing_sail['note']}",
        ]
        if wing_sail else []
    )
    lines = [
        f"# Hydrofoil sizing — {result['scenario']}", "",
        f"*Generated {dt.datetime.now().astimezone().isoformat(timespec='seconds')}*",
        "", "## Scope and references", "",
        f"- Differentiable AeroSandbox `{config['hydrodynamics']['solver']}` / NeuralFoil sizing model with a fixed tail, mast, and fuselage.",
        "- Reynolds-dependent NeuralFoil section polars use water kinematic viscosity; an independent, finer lifting-line mesh checks the optimized cruise point.",
        f"- Use scenario: {config.get('use', 'unspecified')}.",
        f"- Main-wing starting reference: `{config['references']['wing']['file']}` "
        f"({config['references']['wing']['span_m']*100:.1f} cm span, "
        f"{config['references']['wing']['projected_area_m2']*1e4:.0f} cm² mesh projection).",
        f"- Reference note: {config['references']['wing']['note']}",
        f"- Fixed tail: {config['references']['stabilizer']['name']} "
        f"({geometry['stabilizer_area_m2']*1e4:.0f} cm², "
        f"{geometry['stabilizer_span_m']*100:.1f} cm span).",
        *board_lines,
        *wing_sail_lines,
        "- Board insert locations are not yet used for rider balance; board mass properties and actual foot centers are unconfirmed.",
        "- Wing-sail aerodynamic force and center of effort are not included in the stability model.",
        "- Mesh dimensions seed the design; the aerodynamic wing is a simplified trapezoidal planform, not a direct mesh simulation.",
        "- The listed LEV 1150 3MF files are mold/toolpath pieces, and Fusion F3D assets are not imported as aerodynamic surfaces.",
        "- Steady longitudinal model only. Cavitation is omitted; ventilation, free-surface, yaw/roll, unsteady flow, joints, fatigue, and manufacturing are not validated.",
        "- Structural outputs are screening calculations, not a laminate design, proof test, or safety certification.",
        "", "## Optimized geometry", "",
        "| Quantity | Result |", "|:--|--:|",
        f"| Span | {design['span']*100:.1f} cm |",
        f"| Root / tip chord | {design['root_chord']*1000:.0f} / {geometry['tip_chord_m']*1000:.0f} mm |",
        f"| Taper ratio | {design['taper']:.3f} |",
        f"| Wing area | {geometry['area_m2']*1e4:.0f} cm² |",
        f"| Aspect ratio | {result['aspect_ratio']:.2f} |",
        f"| Sweep / tip washout | {design['sweep_deg']:.1f}° / {design['tip_washout']:.2f}° |",
        f"| Section | Kulfan deformation of {config['wing']['airfoil_baseline']}; "
        f"t/c at x/c=0.30: {result['profile_thickness_ratio']*100:.1f}% |",
        f"| CG / MAC | {design['cg_fraction']*100:.1f}% |",
        f"| Stabilizer | {config['references']['stabilizer']['name']}; "
        f"{geometry['stabilizer_root_chord_m']*1000:.0f} mm root chord |",
        f"| Fixed fuselage / mast | {config['fuselage']['length_m']*100:.0f} cm / "
        f"{config['mast']['length_m']*100:.0f} cm |",
        *(
            [
                "",
                f"- Active design bounds: {', '.join(result['active_bounds'])}; "
                "this is a boundary-driven candidate.",
            ]
            if result["active_bounds"] else []
        ),
        "", "### Kulfan weights", "",
        "Each weight remains within the configured offset of the NACA baseline.",
        "", "```json", json.dumps(result["profile"], indent=2), "```",
        "", "## Trim and drag", "",
        "| Point | Speed (m/s / kt) | α | Lift | Drag incl. mast | CL | L/D |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name in ("cruise", "fast", "takeoff"):
        point = result["points"][name]
        lines.append(
            f"| {name} | {point['speed_m_s']:.2f} / {point['speed_m_s']*1.94384:.1f} "
            f"| {point['alpha_deg']:.2f}° | {point['L']:.1f} N "
            f"| {point['drag_including_mast_N']:.2f} N | {point['CL']:.3f} "
            f"| {point['L']/point['drag_including_mast_N']:.2f} |"
        )
    lines.extend([
        "",
        f"Design objective: **{result['objective_name']}** "
        f"(value {result['objective_value']:.4f}; "
        f"profile regularization {result['profile_penalty']:.3f}).",
        f"Weighted drag at the fixed-speed design points (diagnostic, not minimized): "
        f"**{result['objective_N']:.2f} N**.",
    ])
    if result["vpp_points"]:
        lines.extend([
            "",
            "## Coupled VPP trims",
            "",
            "The wind-driven trim states and foil geometry were solved in the same NLP. "
            "Hydrodynamic finesse is foil lift divided by foil, immersed-mast, and board drag; "
            "the mast is modeled with a submerged lifting-line model. State margins are in `design.json`.",
            "",
            "| Point | Wind (m/s, TWA) | Trim speed | Hydro L/D | Foil α | Leeway | Heel | Ride height | Mast drag | Mast sideforce | Roll stability margin | Yaw stability margin | State limits |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|",
        ])
        for point in result["vpp_points"].values():
            roll_margin = (
                -point["roll_stability_derivative_Nm_per_rad"]
                - point["minimum_roll_stability_Nm_per_rad"]
            )
            yaw_margin = (
                -point["yaw_stability_derivative_Nm_per_rad"]
                - point["minimum_yaw_stability_Nm_per_rad"]
            )
            limits = ", ".join(point["active_limits"]) or "none"
            lines.append(
                f"| {point['name']} | {point['true_wind_speed_m_s']:.1f}, "
                f"{point['true_wind_angle_deg']:.0f}° | "
                f"{point['boat_speed_m_s']:.2f} m/s | "
                f"{point['hydrodynamic_finesse']:.2f} | "
                f"{point['foil_alpha_deg']:.1f}° | {point['leeway_deg']:.1f}° | "
                f"{point['heel_deg']:.1f}° | {point['ride_height_m']:.2f} m | "
                f"{point['water_mast_drag_N']:.1f} N | "
                f"{point['water_mast_sideforce_N']:.1f} N | "
                f"{roll_margin:.1f} Nm/rad | {yaw_margin:.1f} Nm/rad | {limits} |"
            )
            if not point["trim_valid"]:
                lines.append(
                    f"\nWarning: VPP point `{point['name']}` has a boundary-limited "
                    "state or failed trim check; inspect `design.json` state margins."
                )
    lines.extend([
        "", "## Stability", "",
        f"- CLa: {stability['CLa_per_rad']:.3f} rad⁻¹; Cma: {stability['Cma_per_rad']:.3f} rad⁻¹.",
        f"- Derived static margin (diagnostic only): {stability['derived_static_margin']*100:.1f}% MAC.",
        f"- Pitch moments at cruise / fast / takeoff: "
        f"{stability['pitch_moments_Nm']['cruise']:+.1f} / "
        f"{stability['pitch_moments_Nm']['fast']:+.1f} / "
        f"{stability['pitch_moments_Nm']['takeoff']:+.1f} N·m; "
        f"stance screen ±{stability['moment_limit_Nm']:.1f} N·m "
        f"({stability['stance_limit_Nm']:.1f} N·m before reserve).",
        "", "## Carbon skin and foam core screening", "",
        f"- Assumed laminate: {config['structure']['laminate']['description']} "
        f"({structure['skin_thickness_m']*1000:.2f} mm per face, "
        f"configured maximum {config['structure']['maximum_skin_thickness_m']*1000:.2f} mm; "
        f"effective spanwise modulus {structure['carbon_effective_modulus_Pa']/1e9:.1f} GPa).",
        f"- Structural envelope: {structure['maximum_positive_g']:.1f} g × "
        f"front-wing load share {structure['front_wing_load_share_factor']:.2f} × "
        f"safety factor {structure['material_safety_factor']:.2f}; "
        f"design lift {structure['design_load_N']:.0f} N.",
        f"- Maneuver reference: {config['maneuvers']['turn']['speed_m_s']:.1f} m/s at "
        f"{config['maneuvers']['turn']['load_factor_g']:.1f} g.",
        f"- Foam core: {config['structure']['foam_core']['description']}.",
        f"- Root section depth {structure['total_depth_m']*1000:.1f} mm; "
        f"core {structure['core_depth_m']*1000:.1f} mm "
        f"(minimum spanwise core {structure['minimum_core_depth_m']*1000:.1f} mm; "
        f"required minimum {config['structure']['minimum_core_thickness_m']*1000:.1f} mm; "
        f"root section area {structure['core_area_m2']*1e4:.1f} cm²).",
        f"- Peak spanwise carbon stress / conservative screening allowable: "
        f"{structure['carbon_stress_Pa']/1e6:.1f} / "
        f"{min(config['structure']['laminate']['screening_allowables_pa']['tension_pa'], config['structure']['laminate']['screening_allowables_pa']['compression_pa'])/1e6:.0f} MPa "
        f"({structure['carbon_strain']*100:.3f}% strain).",
        f"- Peak core shear / screening allowable: "
        f"{structure['foam_shear_Pa']/1e6:.3f} / "
        f"{config['structure']['foam_core']['screening_shear_allowable_pa']/1e6:.2f} MPa.",
        f"- Root shear {structure['root_shear_N']:.0f} N; tip deflection "
        f"{structure['tip_deflection_m']*1000:.1f} mm "
        f"{structure['tip_deflection_ratio']*100:.1f}% of semi-span "
        f"(bending {structure['tip_bending_deflection_m']*1000:.1f} mm, "
        f"shear {structure['tip_shear_deflection_m']*1000:.1f} mm).",
        "", "## Constraint checks", "",
    ])
    lines.extend(
        f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}"
        for name, passed in result["checks"].items()
    )
    lines.extend([
        "",
        f"**Model-feasible:** {'yes' if result['feasible'] else 'no'}",
        "", "## Independent LiftingLine check", "",
    ])
    lines.append(
        f"Trim at {lifting_line['alpha_deg']:.2f}°: "
        f"L={lifting_line['lift_N']:.1f} N, D={lifting_line['drag_N']:.2f} N, "
        f"L/D={lifting_line['L_over_D']:.2f}. {lifting_line['scope']}"
        if lifting_line["trimmed"] else lifting_line["message"]
    )
    if lifting_line["trimmed"]:
        comparison = lifting_line["at_design_alpha"]
        lines.append(
            f"- At the optimizer's {comparison['alpha_deg']:.2f}° cruise trim, "
            f"LiftingLine predicts {comparison['lift_N']:.1f} N lift "
            f"({comparison['lift_difference_percent']:+.1f}% vs. the design mesh)."
        )
        lines.append(
            f"- The refined lifting-line mesh needs a "
            f"{lifting_line['alpha_deg'] - comparison['alpha_deg']:+.2f}° "
            "angle change to recover the cruise target."
        )
        lines.append(
            "- Drag comparison is not fully like-for-like: the refined check excludes "
            "fuselage/mast, while the design solver includes fuselage and adds mast drag separately."
        )
    lines.extend([
        "", "## Reproducibility", "",
        f"- {result['starts']} starts; {result['max_iterations']} IPOPT maximum iterations; "
        f"NeuralFoil `{result['model_size']}`; aero solver `{config['hydrodynamics']['solver']}`.",
        f"- Attempt statuses: `{json.dumps(result['attempts'], ensure_ascii=False)}`",
        "- See `design.json` for the full input configuration, optimized coefficients, and numeric checks.",
        "",
    ])
    (folder / "datasheet.md").write_text("\n".join(lines), encoding="utf-8")
