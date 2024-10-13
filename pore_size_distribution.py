import numpy as np

class PoreSizeDistribution:
    @staticmethod
    def calculate_psd(x, avg_r, std_dev):
        b = np.log(1 + (std_dev / avg_r)**2)
        c1 = 1 / np.sqrt(2 * np.pi * b)
        return c1 * (1 / x) * np.exp(-((np.log(x / avg_r) + b / 2)**2) / (2 * b)) 
