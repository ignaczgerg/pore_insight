from solvent import Solvent
from experimental_data import ExperimentalData
from mwco_fitting import MWCOFitting
from molar_volume import MolarVolumeRelation
from diffusivity import DiffusivityCalculator
from stokes_radius import StokesRadiusCalculator
from pore_size_distribution import PoreSizeDistribution

def main():
    # solvent = Solvent.from_selection(1, 25, 0.89) 

    # exp_data = ExperimentalData("data/prueba.xlsx")
    # exp_data.plot_data()

    # fitting = MWCOFitting(exp_data.mw, exp_data.rejection, exp_data.error)
    # mw_range, fitted_curve = fitting.fit_curve()
    smiles = "CCCC"
    molar_volume = MolarVolumeRelation.joback(smiles)
    print(molar_volume)
    
if __name__ == "__main__":
    main()
