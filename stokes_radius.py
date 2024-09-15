import numpy as np

class StokesRadiusCalculator:
    @staticmethod
    def calculate_stokes_radius(temp, viscosity, diffusivity):
        kb = 1.38E-23
        sol_viscosity = viscosity * 1E-3
        diffusivity_m2_s = diffusivity / (100**2)
        return (kb * temp) / (6 * np.pi * sol_viscosity * diffusivity_m2_s)
