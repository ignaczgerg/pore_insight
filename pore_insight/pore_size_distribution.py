import numpy as np
from numpy import log, exp
import argparse
from scipy.optimize import curve_fit, root
from scipy.integrate import quad
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

from pore_insight.utils import rej_bounds, read_molecules
from pore_insight.models import CurveModels, DiffusivityCalculator, PSDModels, DistributionModels, MolarVolume, StokesRadiusCalculator, AimarMethods
from pore_insight.models import Solvents

VOLUME_FUNCTIONS = {
                'schotte': MolarVolume.relation_Schotte,
                'wu': MolarVolume.relation_Wu,
                'joback': MolarVolume.joback,
                }

class PSD:
    def __init__(self, rejection_values: np.ndarray = None, errors: np.ndarray = None,
                 solute_mol_weights: np.ndarray = None, molecules_structure: str = None,
                 solvent='water', temperature=20.0, solvent_mol_weight: float = None,
                 viscosity=None, alpha=None, molar_volume=None):
        # Input validations moved into __init__
        if rejection_values is None:
            raise ValueError("rejection_values must be provided")
        if errors is not None and len(rejection_values) != len(errors):
            raise ValueError("rejection_values and errors must have the same length")
        if solute_mol_weights is not None and len(rejection_values) != len(solute_mol_weights):
            raise ValueError("rejection_values and solute_mol_weights must have the same length")
        if molecules_structure is not None and len(rejection_values) != len(molecules_structure):
            raise ValueError("rejection_values and molecules_structure must have the same length")
        if molecules_structure is not None and solute_mol_weights is not None:
            raise ValueError("Only one of solute_mol_weights or molecules_structure should be provided")
        if molecules_structure is None and solute_mol_weights is None:
            raise ValueError("solute_mol_weights must be provided if molecules_structure is not provided")
        
        self.rejection_values = np.array(rejection_values)
        self.errors = np.array(errors) if errors is not None else None
        self.solute_mol_weights = np.array(solute_mol_weights) if solute_mol_weights is not None else None
        self.molecules_structure = molecules_structure

        # The remaining attributes are unchanged.
        self.mols = None
        self.x_radii = None
        self.radii_range = None
        self.fitted_curve = None
        self.high_fit = None
        self.low_fit = None
        self.optimal_parameters = None
        self.optimal_parameters_low = None
        self.optimal_parameters_high = None
        self._model_function = None
        self._model_functions = None
        self._initial_params = None
        self._bounds = None
        self._model_derivative_functions = None
        self.solvent = solvent
        self.temperature = temperature
        self.alpha = alpha
        self.molar_volume = molar_volume

        if self.solvent is not None:  # Avoid overwriting user-specified values for 'other' solvents
            self.alpha = Solvents.get(self.solvent).alpha
            self.solvent_mol_weight = (solvent_mol_weight if solvent_mol_weight is not None
                                     else Solvents.get(self.solvent).solvent_mol_weight)
            print(f"Solvent: {self.solvent}, alpha: {self.alpha}, molecular weight: {self.solvent_mol_weight}")
            self.viscosity = viscosity if viscosity is not None else Solvents.get(self.solvent).viscosity

        if self.solute_mol_weights is None and self.molecules_structure is None:
            raise ValueError("Either 'solute_mol_weights' or 'molecules_structure' must be provided.")

    def calculate_radius(self, method='schotte'):
            """
            Function to calculate molecular volumes and diffusivities.
            """
            if method not in VOLUME_FUNCTIONS:
                raise ValueError(f"Method '{method}' is not recognized. Choose from: {list(VOLUME_FUNCTIONS.keys())}")
            
            # We prefer the joback method above anything, except when the user provides volumes directly.
            # This is true even if molecular weights are provided.
            if self.molecules_structure is not None:
                self._get_volume_from_structure(method='joback')
                self.diffusivity = np.array([DiffusivityCalculator.wilke_chang_diffusion_coefficient(mol_volume, 
                                                                                                     self.solvent_mol_weight, # This is the solvent, single value.
                                                                                                     self.temperature, 
                                                                                                     self.viscosity, 
                                                                                                     self.alpha) for mol_volume in self.molar_volume])                 
                # here self.diffusivity is an array of D-s, which are the diffusivity coefficients.
                self.x_radii = np.array([StokesRadiusCalculator.stokes_einstein_radius(D, self.temperature, self.viscosity) for D in self.diffusivity])
                return

            if self.solute_mol_weights is not None:
                self.mws = self.solute_mol_weights
                self.molar_volume = np.array([VOLUME_FUNCTIONS[method](mol) for mol in self.mws])
                self.diffusivity = np.array([DiffusivityCalculator.wilke_chang_diffusion_coefficient(mol_volume, 
                                                                                                     self.solvent_mol_weight, # This is the solvent, single value.
                                                                                                     self.temperature, 
                                                                                                     self.viscosity, 
                                                                                                     self.alpha) for mol_volume in self.molar_volume])                 
                # here self.diffusivity is an array of D-s, which are the diffusivity coefficients.
                self.x_radii = np.array([StokesRadiusCalculator.stokes_einstein_radius(D, self.temperature, self.viscosity) for D in self.diffusivity])
            
            
            if self.molar_volume is not None:
                self.diffusivity = np.array([DiffusivityCalculator.wilke_chang_diffusion_coefficient(mol_volume, 
                                                                                                     self.solvent_mol_weight, # This is the solvent, single value.
                                                                                                     self.temperature, 
                                                                                                     self.viscosity, 
                                                                                                     self.alpha) for mol_volume in self.molar_volume])                 
                # here self.diffusivity is an array of D-s, which are the diffusivity coefficients.
                self.x_radii = np.array([StokesRadiusCalculator.stokes_einstein_radius(D, self.temperature, self.viscosity) for D in self.diffusivity])
            
            elif self.solute_mol_weights is None:
                raise ValueError("Molecule weights must be provided to calculate volume if molecules_structure is not provided.")
    
    def _get_diffusivity(self):
        raise NotImplementedError

    def _get_volume_from_mw(self):
        raise NotImplementedError

    def _get_volume_from_structure(self, method='joback'):
        try:
            self.mws = read_molecules(self.molecules_structure)
        except Exception as e:
            print(e)
            raise ValueError("Invalid molecule structure provided. Check documentation for valid input format.")
        self.molar_volume = np.array([VOLUME_FUNCTIONS[method](mol) for mol in self.mws])
    
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
        radii_range : array-like
            x values range for plotting.
        fitted_curve : array-like
            Fitted curve values.
        """
        self._model_functions = {
            'lognormal_CDF': CurveModels.lognormal_CDF,
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
            'lognormal_CDF': [np.median(self.x_radii),0.1],
            'boltzmann': [-10, 1, np.median(self.x_radii), 1],  
            'sigmoid': [0.1, np.median(self.x_radii), 1],
            'generalized_logistic': [1, 1, 1, 1, 1, 1],
            'gompertz': [1, 1, 1],
            'double_sigmoid': [1, 1, np.median(self.x_radii), 1, 1, np.median(self.x_radii)]
        }

        self._bounds = {
            'lognormal_CDF': ([0,0], [np.inf, np.inf]),
            'boltzmann': ([-np.inf, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
            'sigmoid': ([0, 0, 0], [np.inf, np.inf, np.inf]),
            'generalized_logistic': ([0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            'gompertz': ([0, 0, 0], [np.inf, np.inf, np.inf]),
            'double_sigmoid': ([0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        }

        self.radii_range = np.linspace(0,max(self.x_radii)*2,100)

        if self.errors is not None:
            low_bound, high_bound = rej_bounds(self.rejection_values,self.errors)
            self.optimal_parameters_low, _ = curve_fit(self._model_function, self.x_radii, low_bound, p0=self._initial_params[model_name], bounds=self._bounds[model_name], maxfev=10000)
            self.low_fit = self._model_function(self.radii_range, *self.optimal_parameters_low)
            self.optimal_parameters_high, _ = curve_fit(self._model_function, self.x_radii, high_bound, p0=self._initial_params[model_name], bounds=self._bounds[model_name], maxfev=10000)
            self.high_fit = self._model_function(self.radii_range, *self.optimal_parameters_high)
            self.optimal_parameters, _ = curve_fit(self._model_function, self.x_radii, self.rejection_values, p0=self._initial_params[model_name], bounds=self._bounds[model_name], maxfev=10000)  

        else:
            self.optimal_parameters, _ = curve_fit(self._model_function, self.x_radii, self.rejection_values, p0=self._initial_params[model_name], bounds=self._bounds[model_name], maxfev=10000)  
            self.fitted_curve = self._model_function(self.radii_range, *self.optimal_parameters)

        # return self.radii_range, self.fitted_curve #, self.optimal_parameters, self.low_fit, self.high_fit # we don't need to return self.optimal_parameters, self.low_fit, self.high_fit because they could be non-existent is there is no error.

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

        derivative_function = self._model_derivative_functions[model_name]

        if self.optimal_parameters is None or len(self.optimal_parameters) == 0:
            raise ValueError("Optimal parameters not available. Fit the model first using fit_sigmoid().")

        self.fitted_derivative = derivative_function(self.radii_range, *self.optimal_parameters)

        if self.errors is not None:
            self.pdf_parameters = {
                'radius_mode' : DistributionModels.rp_mode(self.radii_range,self.fitted_derivative),
                'lognormal_STD' : DistributionModels.sigma_FWHM(self.radii_range,self.fitted_derivative),
                'average_radius' : DistributionModels.rp_average(DistributionModels.rp_mode(self.radii_range,self.fitted_derivative),DistributionModels.sigma_FWHM(self.radii_range,self.fitted_derivative))
            }
            
            if self.optimal_parameters_low is None or len(self.optimal_parameters_low) == 0:
                raise ValueError("Optimal parameters of the low bound are not available.")
            self.fitted_derivative_low = derivative_function(self.radii_range,*self.optimal_parameters_low)
            self.pdf_parameters_low = {
                'radius_mode' : DistributionModels.rp_mode(self.radii_range,self.fitted_derivative_low),
                'lognormal_STD' : DistributionModels.sigma_FWHM(self.radii_range,self.fitted_derivative_low),
                'average_radius' : DistributionModels.rp_average(DistributionModels.rp_mode(self.radii_range,self.fitted_derivative_low),DistributionModels.sigma_FWHM(self.radii_range,self.fitted_derivative_low))
            }

            if self.optimal_parameters_high is None or len(self.optimal_parameters_high) == 0:
                raise ValueError("Optimal parameters of the high bound are not available.")
            self.fitted_derivative_high = derivative_function(self.radii_range,*self.optimal_parameters_high)
            self.pdf_parameters_high = {
                'radius_mode' : DistributionModels.rp_mode(self.radii_range,self.fitted_derivative_high),
                'lognormal_STD' : DistributionModels.sigma_FWHM(self.radii_range,self.fitted_derivative_high),
                'average_radius' : DistributionModels.rp_average(DistributionModels.rp_mode(self.radii_range,self.fitted_derivative_high),DistributionModels.sigma_FWHM(self.radii_range,self.fitted_derivative_high))
            }
        else:
            self.pdf_parameters = {
                'radius_mode' : DistributionModels.rp_mode(self.radii_range,self.fitted_derivative),
                'lognormal_STD' : DistributionModels.sigma_FWHM(self.radii_range,self.fitted_derivative),
                'average_radius' : DistributionModels.rp_average(DistributionModels.rp_mode(self.radii_range,self.fitted_derivative),DistributionModels.sigma_FWHM(self.radii_range,self.fitted_derivative))
            }


class TwoPointPSD:
    def __init__(self, solute_radius_zero: float = None, rejection_zero: float = None,
                 solute_radius_one: float = None, rejection_one: float = None):
        if solute_radius_zero is None:
            raise ValueError("solute_radius_zero must be provided")
        if rejection_zero is None:
            raise ValueError("rejection_zero must be provided")
        if solute_radius_one is None:
            raise ValueError("solute_radius_one must be provided")
        if rejection_one is None:
            raise ValueError("rejection_one must be provided")
        
        self.solute_radius_zero = solute_radius_zero
        self.rejection_zero = rejection_zero
        self.solute_radius_one = solute_radius_one
        self.rejection_one = rejection_one
        self.sigma = None
        self.r_star = None
        self.prediction = None

    def numerator(self,lambda_star, sigma):
        """
        Numerator of the overall rejection integral:
            ∫ [r^4 * local_rej(a, r) * lognormal_pdf(r, sigma)] dr
        except here r is dimensionless => r^4 => r_dimless^4
        """
        def integrand(r_dimless):
            return (r_dimless**4) * AimarMethods.local_rejection(lambda_star, r_dimless) * AimarMethods.lognormal_pdf(r_dimless, sigma)
        
        lower, upper = 1e-6, 1e2
        val, _ = quad(integrand, lower, upper, limit=200)
        return val

    def denominator(self,sigma):
        """
        Denominator of the overall rejection integral:
            ∫ [r^4 * lognormal_pdf(r, sigma)] dr
        i.e. the integral with no local rejection factor.
        """
        def integrand(r_dimless):
            return (r_dimless**4) * AimarMethods.lognormal_pdf(r_dimless, sigma)
        
        lower, upper = 1e-6, 1e2
        val, _ = quad(integrand, lower, upper, limit=200)
        return val

    def overall_rejection(self,lambda_star, sigma):
        """
        Overall (global) rejection R(a) for dimensionless solute radius lambda_star = a / r*,
        given a log-normal distribution (sigma) and Poiseuille flow weighting (r^4).
        """
        return self.numerator(lambda_star, sigma) / self.denominator(sigma)
    
    def system_equations(self,vars_):
        """
        We define two equations:
        1) overall_rejection(lambda0_star, sigma) = R0
        2) overall_rejection((a1/a0)*lambda0_star, sigma) = R1
        
        where lambda0_star = a0 / r_star, but we solve for [lambda0_star, sigma] directly.
        """
        lambda0_star, sigma = vars_
        a0 = self.solute_radius_zero
        R0 = self.rejection_zero
        a1 = self.solute_radius_one
        R1 = self.rejection_one

        f1 = self.overall_rejection(lambda0_star, sigma) - R0
        
        lambda1_star = (a1 / a0) * lambda0_star
        f2 = self.overall_rejection(lambda1_star, sigma) - R1
        
        return [f1, f2]
    
    def find_pore_distribution_params(self, lambda0_star_guess=1.0, sigma_guess=1.5):
        """
        Solve for the dimensionless radius lambda0_star = a0 / r_star and sigma
        given two experimental data points:
            (a0, R0) and (a1, R1).
        Returns:
            r_star (mean pore radius, in same units as a0, a1)
            sigma  (geometric standard deviation)
        """
        sol = root(self.system_equations, [lambda0_star_guess, sigma_guess])
    
        if not sol.success:
            raise RuntimeError(f"Solver did not converge: {sol.message}")
        
        lambda0_star, self.sigma = sol.x
        # Then r_star = a0 / lambda0_star
        self.r_star = self.solute_radius_zero / lambda0_star

        self.lognormal_parameters = {
            'average_radius' : self.r_star,
            'lognormal_STD' : self.sigma
        }

    def predict_rejection_curve(self,a):
        """
        Given a solute radius, a, return the predicted overall rejection,
        using the found parameters r_star, sigma.
        """

        if self.r_star and self.sigma is not None:
            lambda_star = a / self.r_star
            self.prediction = {
                'solute_radius' : a,
                'predicted_rejection' : self.overall_rejection(lambda_star,self.sigma)
            }
        else:
            raise ValueError("Distribution parameters not available. A r_star and a sigma value must be provided. Use find_pore_distribution_params().")