# Hydrofoil sizing model

This repository contains a differentiable, reference-based sizing model for a freeride wingfoil. It varies the front-wing planform, a bounded Kulfan airfoil deformation, and rider CG. The stabilizer, mast, and fuselage remain fixed.

## Run

Install the dependencies from `requirements.txt`, then run:

```sh
python src/run.py
```

Select an operating-use scenario by its filename (without `.yaml`) or by a YAML path:

```sh
python src/run.py --scenario wingfoil-freeride
python src/run.py --list-scenarios
```

Optional solver controls are `--starts N` and `--max-iterations N`. Every run writes `outputs/<scenario>_<date>_design_NN/datasheet.md` and `design.json`.

Run the exploratory quasi-static VPP for either sail type with:

```sh
python src/run.py --mode vpp --scenario wingfoil-freeride
python src/run.py --mode vpp --scenario windfoil-freeride
```

VPP results are written to `outputs/<scenario>_vpp_<date>_NN/vpp.md` and `vpp.json`. These are model demonstrations, not predicted or validated on-water performance.
For the implemented wingfoil design, the default `--mode design` instead couples
the VPP points directly into the geometry optimization; `--mode vpp` remains a
standalone trim analysis.

## Geometry references

* `reference/925-wing/925.stl` is a binary mesh with 69,848 facets. Interpreted in millimeters, it spans about 922 mm; its projected planform estimate is about 1,262 cm². The filename therefore likely refers to span, not a 925 cm² area. STL does not encode units, and the projected area remains an estimate rather than a manufacturer specification. It seeds the initial geometry; the optimizer does not solve aerodynamics on the raw mesh.
* `reference/zeeko-bullet-catalog/` contains the Bullet charts. They list a 1,200 cm² / 100 cm front wing and compatible 250-300 cm² stabilizers; the fixed Bullet 270 reference is 270 cm² / 43 cm span.
* `reference/zeeko-lev-1150/LEV 1150/` contains four `.gcode.3mf` print projects for intrados mold sections, not a foil mesh. Its Fusion files, including `Aile moule 3D/Stab180 v7.f3d`, are retained as source references but are not parsed into the aerodynamic model.
* The supplied `Stab180 v7.f3d` is a useful Zeeko tail reference, but its exact dimensions and compatibility are not recoverable from the Fusion archive here; the catalog-dimensioned Bullet 270 is used instead.
* `reference/easy-rider-31/` groups a full-size SoarFoils mesh with its separate-part archive; the latter is only part 1 of 2.
* `reference/mixed-foils/all-foils (1).3mf` contains several unlabeled assemblies, so its parts are not used to assign a specific component. Other files are grouped by their named foil family.
* `reference/foil-1100-mk1-series/` has a Fusion assembly manifest naming the 1100 Mk1 wing, rear wing, mast mold, and fuselage; its BREP geometry is not directly parsed by this model. It is a promising complete-system reference once converted to a neutral CAD or mesh format.
* The Phantom Whale USDZ loads as a scene, but its unit metadata and root scale imply roughly 3 m overall dimensions, inconsistent with its 1,225 cm² / AR 8.2 filename if interpreted as a normal foil. Do not use its absolute scale until verified.
* The Pumping foil USDZ loads as seven mesh parts and has multiple nested scale transforms. It is useful for visual/shape comparison, but its transforms should be baked and checked before using measured dimensions.
* The Delfi STEP is a faceted assembly rather than a clearly isolated front wing. The Easy Rider and Delfi files can inform assembly proportions, but are not direct reference surfaces for this MDO.
* `reference/wing-foil-board-1.snapshot.9/` contains board drawings/screenshots, not a board CAD model. The dimensioned drawing reports 5'4.8" × 27.8" × 5", 96.04 L and 10.33 ft². Annotated stringer stations and a candidate foil-track range are recorded in the wingfoil scenario, but footstrap symbols and the board CG need confirmation/measurement before balance calculations use them.
* `reference/wingfoil wing/` now contains six 3MF parts for a wing-sail model. The full model's encoded extents are about 305 × 36 × 203 mm; the separate parts span a different assembled-coordinate envelope. No full-size scale or loaded canopy profile is encoded, so use it as a shape/handle-layout reference only, not for sail force prediction yet.
* `reference/wingsurf-hydrofoil/wingsurf hydrofoil.stl` is a hydrofoil reference, not a wing-sail. No windsurf sail CAD, board mass properties, or measured center-of-effort data were found.

## Model outline

* `config/scenarios/` contains YAML files for individual use scenarios. Wingfoil takeoff/cruise/high-speed points and a 2.5 g, 8 m/s turn are the implemented design envelope. Pumping records glide/cycle speed, frequency and g ranges; windfoil records water speed plus true-wind conditions. Those latter scenarios are condition sheets only and cannot yet be optimized.
* `src/foilopt/geometry.py` builds the meshed main wing, fixed stabilizer, and fuselage.
* `src/foilopt/hydrodynamics.py` defines water properties, ASB VortexLatticeMethod/LiftingLine/AeroBuildup options, NeuralFoil profile-drag correction, and the immersed-mast lifting-line model. The wingfoil scenario uses VLM for 3D wing and wake effects; water dynamic viscosity enters through Reynolds number. A finer independent LiftingLine mesh checks differences at optimized cruise.
* `src/foilopt/structure.py` screens a spanwise beam with a hand-layup carbon/epoxy laminate, PVC foam core, elliptical load distribution, maneuver g-load and a material safety factor. Minimum core thickness is constrained across the span; configured skin thickness is checked against its manufacturing limit.
* `src/foilopt/problem.py` assembles the differentiable constraints, objective, solver, checks, and independent LiftingLine comparison.
* `src/foilopt/vpp.py` supplies a differentiable, quasi-static force/moment-balance VPP for both windsurf and wingfoil scenario inputs. The geometry-optimized wingfoil VPP uses the configured 3D foil solver and NeuralFoil profile drag, plus a lifting-line model of the immersed mast; the windsurf reference case retains its analytic foil polar because it has no scaled foil geometry.
* `src/foilopt/report.py` writes the JSON results and technical datasheet.
* [`docs/AEROSANDBOX_API_NOTES.md`](docs/AEROSANDBOX_API_NOTES.md) records the AeroSandbox syntax and version-specific API checks used by this project.

For implemented wingfoil scenarios with an enabled VPP, the design optimizer and
all configured VPP points are assembled in one simultaneous NLP. The VPP uses
the candidate wing's area, span, and mean aerodynamic chord; it solves each
point's boat speed, leeway, heel, sail/foil incidence, and rider controls subject
to force/moment balance and static roll/yaw stability margins. The design
objective is mean VPP hydrodynamic finesse: VLM/LLT foil lift divided by foil,
immersed-mast, and board drag. The submerged mast LLT's immersion is a bounded
VPP state, so its drag and antidrift sideforce change with ride height. Existing
fixed-speed aerodynamic and structural checks remain constraints. VPP speed is
the speed of that optimized steady trim, not a top-speed prediction.

The eight Kulfan weights per side are kept close to the NACA 2412 baseline. Geometry bounds are intentionally broad; lift/trim, longitudinal stability derivatives, skin/core screens, tip chord, area, and aspect ratio bound nonphysical solutions.

## Limitations

The structural scenario assumes a four-ply 200 g/m² carbon/epoxy face laminate ([0/0/+45/-45], mirrored across the core; about 1.27 mm per face at 35% fiber volume) over nominal 80 kg/m³ PVC foam. This is a conservative hand-wet-layup assumption without vacuum, not a guaranteed cured thickness or a layup recipe. The optimizer enforces the configured minimum foam-core depth at every spanwise section and the root minimum, and config validation limits the chosen skin stack to 1.5 mm. Its 2.5 g maneuver envelope is multiplied by a 1.5 material safety factor and a 1.15 front-wing load-share allowance for tail downforce; the optimizer is further limited to 85% strength-screen utilization and 90% of the deflection limit. These are DIY-oriented screening assumptions, not tested material allowables: actual fiber/resin system, core grade, local reinforcements, inserts, joints, defects, wet-out and cure quality must be characterized before fabrication. The screening model is a beam calculation, not laminate failure analysis or FEM.

Only the wingfoil freeride scenario is supported by the foil geometry optimizer. `pumping.yaml` remains conditions-only because it needs a transient cycle, rider/board inertia and validated stroke kinematics. The windsurf scenario also remains unavailable to the foil geometry optimizer, but now has a separately selectable, exploratory VPP.

The VPP balances surge, sway and heave forces plus roll and pitch moments at steady state; heading is prescribed and yaw-moment equilibrium is not solved. In the wingfoil optimizer, foil forces and moments come from the configured 3D VLM/LLT solver with NeuralFoil profile-drag correction; foil geometry and profile are shared with the optimizer. The submerged mast's lift/drag and antidrift sideforce come from LLT. Ride height is bounded and optimized, but is not physically predicted: free-surface lift, board buoyancy, and a height-control law are absent. The windsurf reference case retains its analytic foil polar because no reliable foil geometry is configured. The board pitch is identified with foil angle of attack (zero fixed foil-to-board incidence), not solved as an independent trim variable. The segmented rider computes mass-weighted CG from configured torso/head/limb masses; the two footstrap and handhold anchors locate the feet and arm segment. It does not solve joint reactions, articulated posture, mast-foot loads or dynamic human control.

The windsurf sail, windsurf foil, wingfoil rig, rider posture and board drag values are illustrative placeholders because the supplied assets do not establish full-size sail geometry, loaded profiles, force centers, board mass properties or windsurf foil scale. In particular, the wing 3MF is not a full-size, loaded canopy: the scenario records an assumed 2.4 m leading-edge dimension but marks its scale unverified. Replace those YAML inputs with measured values before interpreting a result. The coupled wingfoil optimizer reports VPP outputs as exploratory; they are not validated on-water predictions.

For model context, low-order VPP force balance is consistent with the steady force/moment approach used in early-design sailing-vessel models, but those sources do not validate this windsurf/wingfoil implementation: Oettle & Krüger, [4-DOF early-design VPP (2023)](https://doi.org/10.1115/OMAE2023-101675), and [finite-rig potential-flow method (2024)](https://doi.org/10.1115/OMAE2024-125658). Segment-mass fractions and center locations are configurable; de Leva's [anthropometric adjustments](https://doi.org/10.1016/0021-9290(95)00178-6) are a suitable reference for replacing the current generic values. Independent sail-specific and on-water validation remains necessary.

The existing foil-sizing model is useful for comparative early sizing, but is not a finished predictive model: its VLM mesh is coarse and its NeuralFoil profile-drag correction is a 2D section estimate rather than a spanwise viscous integration. It omits free-surface/ventilation and full mast/fuselage interference, and its carbon/foam calculation is a beam screening model rather than laminate failure analysis or FEM. The reported `CLa`/`Cma` describe the isolated foil aircraft model, not coupled board–rig–rider stability. The new VPP reports local static moment derivatives, but has no inertia, damping, human-control law, or dynamic stability eigenvalues. Those remain later validation stages.

LiftingLine, AeroBuildup and NeuralFoil are steady, low-order models, not CFD or a manufacturing design. They do not resolve free-surface effects, ventilation, cavitation, foil-body interference accurately, stall transients, yaw/roll dynamics, fatigue or impacts. Structural and aerodynamic feasibility are not safety certification; validate any prototype with actual material data, detailed structural analysis and controlled testing.

## License

Distributed under the MIT License. See `LICENSE`.
