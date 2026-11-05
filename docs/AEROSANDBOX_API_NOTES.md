# AeroSandbox API notes for this project

This is a short, project-specific reference for the AeroSandbox interfaces used
in `src/foilopt/`. It is not a replacement for the full documentation.

## Which version to trust

The project pins `aerosandbox==4.2.6` in `requirements.txt`; the local
documentation checkout beside this repository is version 4.2.9. The signatures
below were checked against the project's `.venv-v5` AeroSandbox 4.2.6 source and
compared with the local tutorials. When they differ, prefer the installed
project version and its source over an example from the newer documentation.
Re-check after changing the dependency pin.

## Syntax used here

```python
import aerosandbox as asb
import aerosandbox.numpy as anp

airplane = asb.Airplane(
    xyz_ref=[x_ref, 0, z_ref],
    wings=[
        asb.Wing(
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[x_le, y, z_le],
                    chord=chord_m,
                    twist=twist_deg,
                    airfoil=asb.Airfoil("naca2412"),
                ),
            ],
        ),
    ],
)

op_point = asb.OperatingPoint(
    velocity=speed_m_s,
    alpha=alpha_deg,
    beta=beta_deg,
    atmosphere=atmosphere,
)

result = asb.VortexLatticeMethod(
    airplane=airplane,
    op_point=op_point,
    spanwise_resolution=8,
    chordwise_resolution=4,
    align_trailing_vortices_with_wind=True,
).run()
```

The snippet is illustrative; use enough wing sections for the geometry being
modeled. AeroSandbox geometry lengths are in meters and geometry/operating-point
angles are in degrees. `xyz_ref` sets the reference point for reported moments.
The result from `.run()` is a mapping with dimensional and coefficient outputs,
including `L`, `D`, `Y`, `M_g`, `CL`, `CD`, and `Cm`.

The project's other aerodynamic interfaces are:

```python
ll_result = asb.LiftingLine(
    airplane=airplane,
    op_point=op_point,
    model_size="small",
    spanwise_resolution=4,
).run()

section = asb.Airfoil("naca2412").get_aero_from_neuralfoil(
    alpha=alpha_deg,
    Re=reynolds_number,
    model_size="small",
)
```

`LiftingLine` is the exported nonlinear lifting-line analysis in AeroSandbox
4.2.6; its constructor has `model_size` and `spanwise_resolution`, but not the
VLM's `chordwise_resolution`. NeuralFoil's `alpha` is in degrees, and the
returned mapping has keys such as `CL`, `CD`, and `CM`. VLM does not provide the
same viscous profile-drag treatment as NeuralFoil; the current code adds its
2D section drag estimate separately. Keep track of reference area and axes
when combining these outputs.

For optimization, the project uses AeroSandbox's `Opti` wrapper:

```python
opti = asb.Opti()
x = opti.variable(
    init_guess=1.0,
    scale=1.0,
    lower_bound=0.1,
    upper_bound=2.0,
)
opti.subject_to(x**2 >= 0.25)
opti.minimize((x - 1.2) ** 2)
opti.solver("ipopt", {"ipopt.max_iter": 200})
solution = opti.solve()
x_value = solution.value(x)
```

Use `aerosandbox.numpy` (`anp`) for NumPy-like expressions that depend on
optimization variables. Ordinary NumPy/SciPy functions generally do not preserve
the CasADi derivative graph; keep non-symbolic preprocessing outside the
optimization expressions or use a differentiable AeroSandbox/CasADi equivalent.
For an ASB analysis to participate in the design optimization, its inputs must
remain symbolic and its internal solver must support the optimization graph;
an API accepting ordinary numeric inputs alone does not guarantee this.

## Local tutorials worth consulting

The full local documentation is in the neighboring `AeroSandbox documentation`
folder. These tutorials cover the APIs used or likely to be extended here:

- [Basic Aircraft Geometry](<../../AeroSandbox documentation/tutorial/04 - Geometry and Data Structures/01 - AeroSandbox Data Structures/01 - Basic Aircraft Geometry.ipynb>)
- [Operating Points](<../../AeroSandbox documentation/tutorial/04 - Geometry and Data Structures/01 - AeroSandbox Data Structures/02 - Operating Points.ipynb>)
- [Vortex Lattice Method](<../../AeroSandbox documentation/tutorial/06 - Aerodynamics/01 - AeroSandbox 3D Aerodynamics Tools/01 - Vortex Lattice Method/01 - Vortex Lattice Method.ipynb>)
- [Nonlinear Lifting-Line](<../../AeroSandbox documentation/tutorial/06 - Aerodynamics/01 - AeroSandbox 3D Aerodynamics Tools/02 - Nonlinear Lifting-Line/02 - Nonlinear Lifting-Line.ipynb>)
- [NeuralFoil](<../../AeroSandbox documentation/tutorial/06 - Aerodynamics/02 - AeroSandbox 2D Aerodynamics Tools/01 - NeuralFoil.ipynb>)
- [Solver Options](<../../AeroSandbox documentation/tutorial/01 - Optimization and Math/10 - Solver Options.ipynb>)

## Before changing ASB calls

1. Check the installed version and inspect that version's signature/docstring.
2. Compare with a local tutorial, noting the documentation checkout's version.
3. Keep units, reference axes, and force/moment sign conventions explicit.
4. Validate symbolic differentiation if the call is inside the optimizer.
5. Run the relevant tests and compare at least one result with a more refined
   mesh or an independent method. Solver convergence alone does not validate
   the physical model.
