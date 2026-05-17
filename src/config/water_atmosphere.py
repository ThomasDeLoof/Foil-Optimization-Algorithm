# water_atmosphere.py

import aerosandbox as asb
import aerosandbox.numpy as np
import yaml

with open("/Users/thomas/Documents/Dossier Supaero/clubs/Foil/Foil-Optimization-Algorithm/src/config/parameters.yaml") as f:
    phy = yaml.safe_load(f)

depth = float(phy["mast"]["immersion_depth"])

class Water(asb.AeroSandboxObject):
    """
    Simule une 'atmosphère' d'eau pour AeroSandbox :
    densité et viscosité constantes, pression hydrostatique.
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

    # --- Propriétés thermodynamiques et mécaniques ---

    def density(self):
        # Masse volumique de l'eau salée (kg/m^3)
        return 1020.0 

    def pressure(self) -> float:
        # Pression hydrostatique [Pa]
        p0 = 101325.0
        rho_w = self.density()
        g    = 9.81
        return p0 + rho_w * g * depth

    def temperature(self):
        # Température constante de l'eau [K]
        return 293.15

    def dynamic_viscosity(self):
        # Viscosité dynamique (Pa·s)
        return 1e-3

    def kinematic_viscosity(self):
        return self.dynamic_viscosity() / self.density()

    def speed_of_sound(self):
        # vitesse du son dans l'eau ~ 1500 m/s à 20°C
        return 1500.0

    def vapor_pressure(self) -> float:
        # Calcule la pression de vapeur saturante de l'eau via la formule d'Antoine (temp en °C)
        T_C = self.temperature() - 273.25
        
        # Coefficients d'Antoine pour l'eau (pression en mmHg)
        A = 8.07131
        B = 1730.63
        C = 233.426
        
        # Calcul de la pression en mmHg
        p_mmHg = 10 ** (A - B / (C + T_C))
        
        # Conversion mmHg en Pascals (1 mmHg = 133.322 Pa)
        return p_mmHg * 133.3224