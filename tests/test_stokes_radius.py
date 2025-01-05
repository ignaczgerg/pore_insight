import unittest
from _utils import StokesRadiusCalculator

class TestStokesRadiusCalculator(unittest.TestCase):
    def test_stokes_radius(self):
        radius = StokesRadiusCalculator.stokes_einstein_radius(298, 298.15, 0.543)
        self.assertAlmostEqual(radius, 2.82e-9, delta=1e-7) 

if __name__ == '__main__':
    unittest.main()
