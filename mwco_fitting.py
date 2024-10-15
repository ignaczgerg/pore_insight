import numpy as np
from scipy.optimize import curve_fit

class MWCOFitting:
    def __init__(self, mw, rejection, error):
        self.mw = mw
        self.rejection = rejection
        self.error = error

    @staticmethod
    def model_f(x, a, b, c, d):
        return b + (a - b) / (1 + np.exp((x - c) / d))
    
    @staticmethod
    def sigmoid(x, a, b, c):
        return c / (1 + np.exp(-a * (x - b)))

    @staticmethod
    def generalized_logistic(x, A, K, B, Q, C, nu):
        return A + (K - A) / (C + Q * np.exp(-B * x))**(1 / nu)

    @staticmethod
    def gompertz(x, a, b, c):
        return a * np.exp(-b * np.exp(-c * x))

    @staticmethod
    def double_sigmoid(x, K1, B1, M1, K2, B2, M2):
        return (K1 / (1 + np.exp(-B1 * (x - M1)))) + (K2 / (1 + np.exp(-B2 * (x - M2))))

    def fit_curve(self, model_name='model_f'):
        """
        Fit the curve using the specified model function.

        Parameters
        ----------
        model_name : str, optional
            The name of the model function to use for fitting. Options are: 
            'model_f', 'sigmoid', 'generalized_logistic', 'gompertz', 'double_sigmoid'.
            Default is 'model_f'.

        Returns
        -------
        mw_range : array-like
            Molecular weight range for plotting.
        fitted_curve : array-like
            Fitted curve values.
        """
        # Define the model function mapping
        model_functions = {
            'model_f': self.model_f,
            'sigmoid': self.sigmoid,
            'generalized_logistic': self.generalized_logistic,
            'gompertz': self.gompertz,
            'double_sigmoid': self.double_sigmoid
        }

        if model_name not in model_functions:
            raise ValueError(f"Model '{model_name}' is not recognized. Choose from: {list(model_functions.keys())}")

        # Select the model function
        model_function = model_functions[model_name]

        # Define initial parameters (p0) for curve fitting for each model
        initial_params = {
            'model_f': [-50, 100, 50, 110],
            'sigmoid': [1, 50, 100],
            'generalized_logistic': [1, 100, 1, 1, 1, 1],
            'gompertz': [100, 1, 1],
            'double_sigmoid': [100, 1, 50, 100, 1, 150]
        }

        # Get the appropriate initial parameters
        p0 = initial_params[model_name]

        # Fit the curve using the selected model
        mw_range = np.linspace(min(self.mw), max(self.mw), 100)
        popt, _ = curve_fit(model_function, self.mw, self.rejection, p0=p0)
        fitted_curve = model_function(mw_range, *popt)
        
        return mw_range, fitted_curve
