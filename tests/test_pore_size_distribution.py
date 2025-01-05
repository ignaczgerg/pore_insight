import unittest
import numpy as np
from models import PSD

class TestPoreSizeDistribution(unittest.TestCase):
    def test_psd(self):
        f = PSD.PDF(1, 2, 0.5)
        self.assertGreater(f, 0)

if __name__ == '__main__':
    unittest.main()
