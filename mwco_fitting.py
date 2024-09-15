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

    def fit_curve(self):
        mw_range = np.linspace(min(self.mw), max(self.mw), 100)
        popt, _ = curve_fit(self.model_f, self.mw, self.rejection, p0=[-50, 100, 50, 110])
        fitted_curve = self.model_f(mw_range, *popt)
        return mw_range, fitted_curve
