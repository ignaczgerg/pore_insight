import numpy as np

class StokesRadiusCalculator:
    @staticmethod
    def stokes_einstein_radius(D,T,eta):
        """
        Calculate the radius of a particle from the diffusion coefficient using the Stokes-Einstein equation.

        Parameters:
        D (float): Diffusion coefficient (cm²/s)
        T (float): Absolute temperature (K)
        eta (float): Dynamic viscosity of the fluid (cP)

        Returns:
        float: Radius of the particle (m)
        """
        # Boltzmann constant in J/K
        k_B = 1.380649e-23
        # Convert dynamic viscosity from cP to kg/(m·s) (or Pa·s)
        eta = eta*0.001
        
        # Convert diffusion coefficient from cm²/s to m²/s
        D = D * 1e-4
        
        # Calculate the radius
        r = k_B * T / (6 * np.pi * eta * D)
        
        return r


class DiffusivityCalculator:
    @staticmethod
    def wilke_chang_diffusion_coefficient(molar_volume, molecular_weight, temp, viscosity, alpha):
        d = (7.4E-8) * temp * np.sqrt(alpha * molecular_weight) / (viscosity * (alpha * molar_volume)**0.6)
        d = float(d) # The value was being printed like: [np.float64(5.497635472812708e-06), np.float64(4.708457138648548e-06), np.float64(4.194534214640366e-06), np.float64(3.7831496747983576e-06)]
        return d



class Solvent:
    def __init__(self, sol_type, temperature, molecular_weight, viscosity, alpha):
        self.sol_type = sol_type
        self.temperature = temperature
        self.molecular_weight = molecular_weight
        self.viscosity = viscosity
        self.alpha = alpha

    @staticmethod
    def from_selection(selection, temperature, viscosity):
        if selection == 1:
            return Solvent("Water", temperature, 18.01528, viscosity, 2.6)
        elif selection == 2:
            return Solvent("Methanol", temperature, 32.042, viscosity, 1.9)
        elif selection == 3:
            return Solvent("Ethanol", temperature, 46.069, viscosity, 1.5)
        elif selection == 4:
            return Solvent("Other", temperature, None, viscosity, 1.0)
        else:
            raise ValueError("Invalid solvent selection")
