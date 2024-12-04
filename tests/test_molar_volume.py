import unittest
from fitting_models import MolarVolume

class TestMolarVolumeRelation(unittest.TestCase):
    def test_relation_vs_a(self):
        result = MolarVolume.relation_Schotte(100)
        self.assertAlmostEqual(result, 125.948, delta=1e-1)  

    def test_relation_vs_b(self):
        result = MolarVolume.relation_Wu(100)
        self.assertAlmostEqual(result, 128.158, delta=0.3)  

if __name__ == '__main__':
    unittest.main()
