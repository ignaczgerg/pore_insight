import unittest
import numpy as np
from pore_size_distribution import PoreSizeDistribution

class TestPoreSizeDistribution(unittest.TestCase):
    def test_psd(self):
        f = PoreSizeDistribution.calculate_psd(1, 2, 0.5)
        self.assertGreater(f, 0)

if __name__ == '__main__':
    unittest.main()
