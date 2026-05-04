# water_atmosphere.py

import aerosandbox as asb
import aerosandbox.numpy as np

class Water(asb.AeroSandboxObject):
    """
    Simule une 'atmosphère' d'eau pour AeroSandbox :
    densité et viscosité constantes, pression hydrostatique.
    """

    def __init__(self, depth=0.0):
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

    def pressure(self):
        # Pression hydrostatique [Pa]
        p0 = 101325.0
        rho_w = 1000.0
        g    = 9.81
        return p0 + rho_w * g * self.depth

    def temperature(self):
        # Température constante de l'eau [K]
        return 293.15

    def density(self):
        return 1000.0

    def dynamic_viscosity(self):
        # Viscosité dynamique (Pa·s)
        return 1e-3

    def kinematic_viscosity(self):
        return self.dynamic_viscosity() / self.density()

    def speed_of_sound(self):
        # vitesse du son dans l'eau douce ~ 1480 m/s à 20°C
        return 1480.0

    # --- Interfaces secondaires (pour compatibilité ISA) ---
    def total_pressure(self):
        # hypothétique (incompressible), on renvoie static
        return self.pressure()

    def total_temperature(self):
        return self.temperature()

    def ratio_of_specific_heats(self):
        # non utilisé pour liquide, stub
        return 1.0
