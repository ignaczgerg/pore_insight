import numpy as np

class DiffusivityCalculator:
    @staticmethod
    def bulk_diffusivity(molar_volume, molecular_weight, temp, viscosity, alpha):
        return (7.4E-8) * temp * np.sqrt(alpha * molecular_weight) / (viscosity * (alpha * molar_volume)**0.6)
