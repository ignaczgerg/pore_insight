import numpy as np

class StokesRadiusCalculator:
    @staticmethod
    def stokes_einstein_radius(D):
        """
        Calculate the radius of a particle from the diffusion coefficient using the Stokes-Einstein equation.

        Parameters:
        D (float): Diffusion coefficient (cm²/s)
        T (float): Absolute temperature (K)
        eta (float): Dynamic viscosity of the fluid (Pa·s)

        Returns:
        float: Radius of the particle (m)
        """
        # Boltzmann constant in J/K
        T = 298.15  # Absolute temperature in Kelvin
        eta = 0.334*0.001  # Dynamic viscosity of acetonitrile at 25°C in Pa·s (or kg/(m·s))
        k_B = 1.380649e-23
        
        # Convert diffusion coefficient from cm²/s to m²/s
        D = D * 1e-4
        
        # Calculate the radius
        r = k_B * T / (6 * np.pi * eta * D)
        
        return r


class DiffusivityCalculator:
    @staticmethod
    def wilke_chang_diffusion_coefficient(molar_volume, molecular_weight, temp, viscosity, alpha):
        return (7.4E-8) * temp * np.sqrt(alpha * molecular_weight) / (viscosity * (alpha * molar_volume)**0.6)
