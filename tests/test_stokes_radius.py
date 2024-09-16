import unittest
from stokes_radius import StokesRadiusCalculator

class TestStokesRadiusCalculator(unittest.TestCase):
    def test_stokes_radius(self):
        radius = StokesRadiusCalculator.calculate_stokes_radius(298, 0.89, 1.87e-5)
        self.assertAlmostEqual(radius, 2.82e-9, delta=1e-7) 

if __name__ == '__main__':
    unittest.main()
