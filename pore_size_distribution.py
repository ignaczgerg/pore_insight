import numpy as np
import argparse
from scipy.optimize import curve_fit
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

from pore_insight.utils import rej_bounds, intp90, rmvstr, read_molecules
from pore_insight.models import CurveModels, DiffusivityCalculator, PSDModels, DistributionModels, MolarVolume, StokesRadiusCalculator
from pore_insight.models import Solvents


class PSDInputHandler:
    def __init__(self):
        self.args = self.parse_args()
        self.rejection_values = None
        self.errors = None
        self.molecule_weights = None
        self.molecules_structure = None
        self._process_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Process PSD parameters.')
        parser.add_argument('--rejection_values', type=float, nargs='+', required=True, help='List of rejection values')
        parser.add_argument('--errors', type=float, nargs='+', required=True, help='List of errors')
        parser.add_argument('--molecule_weights', type=float, nargs='+', help='List of molecule weights')
        parser.add_argument('--molecules_structure', type=str, help='Molecule structure')
        return parser.parse_args()

    def validate_inputs(self):
        if len(self.rejection_values) != len(self.errors):
            raise ValueError("rejection_values and errors must have the same length")
        if self.molecule_weights is not None and len(self.rejection_values) != len(self.molecule_weights):
            raise ValueError("rejection_values and molecule_weights must have the same length")
        if self.molecules_structure is not None and len(self.rejection_values) != len(self.molecules_structure.split(',')):
            raise ValueError("rejection_values and molecules_structure must have the same length")

    def _process_args(self):
        self.rejection_values = np.array(self.args.rejection_values)
        self.errors = np.array(self.args.errors)
        self.molecule_weights = np.array(self.args.molecule_weights) if self.args.molecule_weights else None
        self.molecules_structure = self.args.molecules_structure
        self.validate_inputs()

    def get_inputs(self):
        return self.rejection_values, self.errors, self.molecule_weights, self.molecules_structure


class PSD:
    def __init__(self, rejection_values: np.ndarray = None, errors: np.ndarray = None,
                 molecule_weights: np.ndarray = None, molecules_structure: str = None,
                 solvent='water', temperature=20.0, viscosity=None, alpha=None, molar_volume=None):
        if rejection_values is None or errors is None:
            # Automatically invoke PSDInputHandler if required inputs are not provided
            input_handler = PSDInputHandler()
            rejection_values, errors, molecule_weights, molecules_structure = input_handler.get_inputs()
        

        self.rejection_values = rejection_values
        self.errors = errors
        self.molecule_weights = molecule_weights
        self.molecules_structure = molecules_structure
        self.mols = None
        self.x_values = None
        self.x_range = None
        self.fitted_curve = None
        self.high_fit = None
        self.low_fit = None
        self.optimal_parameters = None
        self._model_function = None
        self._model_functions = None
        self._initial_params = None
        self._bounds = None
        self._model_derivative_functions = None
        self.solvent = solvent
        self.temperature = temperature
        self.viscosity = viscosity
        self.alpha = alpha
        self.molar_volume = molar_volume

        if self.solvent is not None:
            self.viscosity = Solvents.get(self.solvent).viscosity
            self.alpha = Solvents.get(self.solvent).alpha

        if self.molecule_weights is None and self.molecules_structure is None:
            raise ValueError("Either 'molecule_weights' or 'molecules_structure' must be provided.")

    def _get_volume(self, method='schotte'):
            """
            pass
            """
            volume_functions = {
                'schotte': MolarVolume.relation_Schotte,
                'wu': MolarVolume.relation_Wu,
                'joback': MolarVolume.joback,
                }
            if method not in volume_functions:
                raise ValueError(f"Method '{method}' is not recognized. Choose from: {list(volume_functions.keys())}")
            
            if self.molecules_structure is not None:
                try:
                    self._mols = read_molecules(self.molecules_structure)
                except Exception as e:
                    print(e)
                    raise ValueError("Invalid molecule structure provided. Check documentation for valid input format.")
                self.x_values = np.array([volume_functions[method](mol) for mol in self._mols])

            elif self.molecule_weights is not None and self.molar_volume is not None:
                self.diffusivity = np.array([DiffusivityCalculator.wilke_chang_diffusion_coefficient(mol_volume, 
                                                                                                     mol_weight, 
                                                                                                     self.temperature, 
                                                                                                     self.viscosity, 
                                                                                                     self.alpha) for (mol_volume, mol_weight) in zip(self.molar_volume, self.molecule_weights)])                 
                # here self.diffusivity is an array of D-s, which are the diffusivity coefficients.
                self.x_values = np.array([StokesRadiusCalculator.stokes_einstein_radius(D, self.temperature, self.viscosity) for D in self.diffusivity])
            
            elif self.molecule_weights is None:
                raise ValueError("Molecule weights must be provided to calculate volume if molecules_structure is not provided.")
    
    def _get_diffusivity(self):
        raise NotImplementedError

    def _get_radius(self):
        raise NotImplementedError

    
    def fit_sigmoid(self, model_name='boltzmann'):
        """
        Fit the curve using the specified model function.

        Parameters
        ----------
        model_name : str, optional
            The name of the model function to use for fitting. Options are: 
            'boltzmann', 'sigmoid', 'generalized_logistic', 'gompertz', 'double_sigmoid'.
            Default is 'boltzmann'.

        Returns
        -------
        x_range : array-like
            x values range for plotting.
        fitted_curve : array-like
            Fitted curve values.
        """
        self._model_functions = {
            'boltzmann': CurveModels.boltzmann,
            'sigmoid': CurveModels.sigmoid,
            'generalized_logistic': CurveModels.generalized_logistic,
            'gompertz': CurveModels.gompertz,
            'double_sigmoid': CurveModels.double_sigmoid
        }

        if model_name not in self._model_functions:
            raise ValueError(f"Model '{model_name}' is not recognized. Choose from: {list(self._model_functions.keys())}")

        self._model_function = self._model_functions[model_name]

        self._initial_params = {
            'boltzmann': [-10, 1, np.median(self.x_values), 1],  
            'sigmoid': [0.1, np.median(self.x_values), 1],
            'generalized_logistic': [1, 1, 1, 1, 1, 1],
            'gompertz': [1, 1, 1],
            'double_sigmoid': [1, 1, np.median(self.x_values), 1, 1, np.median(self.x_values)]
        }

        self._bounds = {
            'boltzmann': ([-np.inf, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
            'sigmoid': ([0, 0, 0], [np.inf, np.inf, np.inf]),
            'generalized_logistic': ([0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            'gompertz': ([0, 0, 0], [np.inf, np.inf, np.inf]),
            'double_sigmoid': ([0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        }

        self.x_range = np.linspace(0,max(self.x_values)*2,100)

        if self.errors is not None:
            low_bound, high_bound = rej_bounds(self.rejection_values,self.errors)
            opt, _ = curve_fit(self._model_function, self.x_values, low_bound, p0=self._initial_params[model_name], bounds=self._bounds[model_name], maxfev=10000)
            self.low_fit = self._model_function(self.x_range, *opt)
            opt, _ = curve_fit(self._model_function, self.x_values, high_bound, p0=self._initial_params[model_name], bounds=self._bounds[model_name], maxfev=10000)
            self.high_fit = self._model_function(self.x_range, *opt)
            self.optimal_parameters, _ = curve_fit(self._model_function, self.x_values, self.rejection_values, p0=self._initial_params[model_name], bounds=self._bounds[model_name], maxfev=10000)  

        else:
            self.optimal_parameters, _ = curve_fit(self._model_function, self.x_values, self.rejection_values, p0=self._initial_params[model_name], bounds=self._bounds[model_name], maxfev=10000)  
            self.fitted_curve = self._model_function(self.x_range, *self.optimal_parameters)

        # return self.x_range, self.fitted_curve #, self.optimal_parameters, self.low_fit, self.high_fit # we don't need to return self.optimal_parameters, self.low_fit, self.high_fit because they could be non-existent is there is no error.

    def fit_psd(self, model_name='boltzmann'):
        """
        Fit the PSD curve using the specified model function.

        Parameters
        ----------
        model_name : str, optional
            The name of the model function to use for fitting the PSD curve.

        Returns
        -------
        None
        """
        self._model_derivative_functions = {
            'boltzmann': PSDModels.derivative_boltzmann,
            'log_normal': PSDModels.log_normal, 
            'sigmoid': PSDModels.derivative_sigmoid,
            'generalized_logistic': PSDModels.derivative_generalized_logistic,
            'gompertz': PSDModels.derivative_gompertz,
            'double_sigmoid': PSDModels.derivative_double_sigmoid
        }

        if model_name not in self._model_derivative_functions:
            raise ValueError(f"Model '{model_name}' is not recognized. Choose from: {list(self._model_derivative_functions.keys())}")

        # Use the derivative function corresponding to the specified model
        derivative_function = self._model_derivative_functions[model_name]

        # Ensure the model has been fitted first
        if self.optimal_parameters is None or len(self.optimal_parameters) == 0:
            raise ValueError("Optimal parameters not available. Fit the model first using fit_sigmoid().")

        # Calculate the derivative (e.g., PSD curve)
        self.fitted_derivative = derivative_function(self.x_range, *self.optimal_parameters)

        # Calculate PDF parameters
        if self.errors is not None:
            self.pdf_parameters = DistributionModels.PDF(self.x_range, self.fitted_derivative, self.low_fit, self.high_fit)
        else:
            self.pdf_parameters = DistributionModels.derivative_sigmoidal(self.x_range, self.fitted_derivative)