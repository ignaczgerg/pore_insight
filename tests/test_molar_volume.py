import unittest
from molar_volume import MolarVolumeRelation

class TestMolarVolumeRelation(unittest.TestCase):
    def test_relation_vs_a(self):
        result = MolarVolumeRelation.relation_vs_a(100)
        self.assertAlmostEqual(result, 125.948, delta=1e-1)  # Adjust delta for tolerance

    def test_relation_vs_b(self):
        result = MolarVolumeRelation.relation_vs_b(100)
        self.assertAlmostEqual(result, 128.158, delta=0.3)  # Adjust delta for tolerance

if __name__ == '__main__':
    unittest.main()
