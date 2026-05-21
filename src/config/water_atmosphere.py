# water_atmosphere.py

import aerosandbox as asb
import aerosandbox.numpy as np
import yaml
from pathlib import Path

_HERE = Path(__file__).resolve().parent
with open(_HERE / "parameters.yaml") as f:
    phy = yaml.safe_load(f)

depth = float(phy["mast"]["immersion_depth"])

class Water(asb.AeroSandboxObject):
    """
    Simulates a water 'atmosphere' for AeroSandbox:
    constant density and viscosity, hydrostatic pressure.
    """

    def __init__(self, depth=depth):
        self.depth = depth

    def __repr__(self):
        return f"Water(depth={self.depth} m)"

    def __len__(self):
        try:
            return int(np.length(self.depth))
        except Exception:
            return 1

    def __getitem__(self, idx):
        d = self.depth
        new_depth = d[idx] if hasattr(d, "__getitem__") else d
        return self.__class__(depth=new_depth)

    def __array__(self, dtype=None):
        return np.fromiter([self], dtype=object).reshape(())

    # --- Thermodynamic and mechanical properties ---

    def density(self):
        # Salt water density (kg/m^3)
        return 1020.0

    def pressure(self) -> float:
        # Hydrostatic pressure [Pa]
        p0 = 101325.0
        rho_w = self.density()
        g    = 9.81
        return p0 + rho_w * g * depth

    def temperature(self):
        # Constant water temperature [K]
        return 293.15

    def dynamic_viscosity(self):
        # Dynamic viscosity (Pa·s)
        return 1e-3

    def kinematic_viscosity(self):
        return self.dynamic_viscosity() / self.density()

    def speed_of_sound(self):
        # Speed of sound in water ~ 1500 m/s at 20°C
        return 1500.0

    def vapor_pressure(self) -> float:
        # Saturation vapor pressure of water via Antoine equation (temp in °C)
        T_C = self.temperature() - 273.25

        # Antoine coefficients for water (pressure in mmHg)
        A = 8.07131
        B = 1730.63
        C = 233.426

        # Pressure in mmHg
        p_mmHg = 10 ** (A - B / (C + T_C))

        # mmHg to Pascals conversion (1 mmHg = 133.322 Pa)
        return p_mmHg * 133.3224