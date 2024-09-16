import unittest
from diffusivity import DiffusivityCalculator

class TestDiffusivityCalculator(unittest.TestCase):
    def test_bulk_diffusivity(self):
        diffusivity = DiffusivityCalculator.bulk_diffusivity(50, 18.015, 298, 0.89, 2.6)
        self.assertAlmostEqual(diffusivity, 1.87e-5, delta=1e-3)  

if __name__ == '__main__':
    unittest.main()
