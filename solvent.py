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
