import unittest
import numpy as np
from mwco_fitting import MWCOFitting

class TestMWCOFitting(unittest.TestCase):
    def test_fit_curve(self):

        mw = [100, 200, 300, 400]
        rejection = [40, 60, 80, 90]
        error = [5, 5, 5, 5]
        
        fitting = MWCOFitting(mw, rejection, error)
        mw_range, fitted_curve = fitting.fit_curve()
        
        self.assertEqual(len(mw_range), 100)
        self.assertTrue(np.all(fitted_curve < 100)) 
        self.assertAlmostEqual(fitted_curve[0], 40, delta=1)  

if __name__ == '__main__':
    unittest.main()
