from solvent import Solvent
from experimental_data import ExperimentalData
from mwco_fitting import MWCOFitting
from molar_volume import MolarVolumeRelation
from diffusivity import DiffusivityCalculator
from stokes_radius import StokesRadiusCalculator
from pore_size_distribution import PoreSizeDistribution

def main():
    # Operational parameters
    solvent = Solvent.from_selection(1, 25, 0.89)  # Example parameters

    # Experimental data
    exp_data = ExperimentalData("data/prueba.xlsx")
    exp_data.plot_data()

    # MWCO fitting
    fitting = MWCOFitting(exp_data.mw, exp_data.rejection, exp_data.error)
    mw_range, fitted_curve = fitting.fit_curve()

    # Add calls for other modules (e.g., Molar Volume, Diffusivity, Stokes Radius)
    # Plot final results
    
if __name__ == "__main__":
    main()
