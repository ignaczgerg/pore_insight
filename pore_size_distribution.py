import numpy as np
from mwco_fitting import MWCOFitting

class PoreSizeDistribution:
    @staticmethod
    def calculate_psd(x, avg_r, std_dev):
        """
        Calculates the pore size distribution using the log-normal distribution.
        
        Parameters
        ----------
        x : array-like
            Pore sizes.
        avg_r : float
            Average pore size radius.
        std_dev : float
            Standard deviation of the pore size distribution.

        Returns
        -------
        psd : array-like
            Calculated pore size distribution.
        """
        b = np.log(1 + (std_dev / avg_r)**2)
        c1 = 1 / np.sqrt(2 * np.pi * b)
        return c1 * (1 / x) * np.exp(-((np.log(x / avg_r) + b / 2)**2) / (2 * b)) 

    @staticmethod
    def derivative_sigmoid(x, a, b, c):
        """
        Derivative of the Sigmoid function.
        
        Parameters
        ----------
        x : array-like
            Input values.
        a : float
            Steepness of the curve.
        b : float
            Midpoint of the curve.
        c : float
            Maximum value of the sigmoid.

        Returns
        -------
        array-like
            Derivative of the sigmoid at each point in x.
        """
        sigmoid_value = MWCOFitting.sigmoid(x, a, b, c)
        return a * sigmoid_value * (1 - sigmoid_value / c)

    @staticmethod
    def derivative_generalized_logistic(x, A, K, B, Q, C, nu):
        """
        Derivative of the Generalized Logistic function.

        Parameters
        ----------
        x : array-like
            Input values.
        A, K, B, Q, C, nu : float
            Parameters of the generalized logistic function.

        Returns
        -------
        array-like
            Derivative of the generalized logistic function at each point in x.
        """
        numerator = (K - A) * (-B) * Q * np.exp(-B * x)
        denominator = nu * (C + Q * np.exp(-B * x))**(1 / nu + 1)
        return numerator / denominator

    @staticmethod
    def derivative_gompertz(x, a, b, c):
        """
        Derivative of the Gompertz function.

        Parameters
        ----------
        x : array-like
            Input values.
        a, b, c : float
            Parameters of the Gompertz function.

        Returns
        -------
        array-like
            Derivative of the Gompertz function at each point in x.
        """
        return a * b * np.exp(-b * np.exp(-c * x)) * np.exp(-c * x)
    
    @staticmethod
    def derivative_double_sigmoid(x, K1, B1, M1, K2, B2, M2):
        """
        Derivative of the Double Sigmoid function.

        Parameters
        ----------
        x : array-like
            Input values.
        K1, B1, M1 : float
            Parameters for the first sigmoid function.
        K2, B2, M2 : float
            Parameters for the second sigmoid function.

        Returns
        -------
        array-like
            Derivative of the double sigmoid function at each point in x.
        """
        term1 = (K1 * B1 * np.exp(-B1 * (x - M1))) / ((1 + np.exp(-B1 * (x - M1)))**2)
        term2 = (K2 * B2 * np.exp(-B2 * (x - M2))) / ((1 + np.exp(-B2 * (x - M2)))**2)
        return term1 + term2