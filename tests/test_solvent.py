import unittest
from solvent import Solvent

class TestSolvent(unittest.TestCase):
    def test_from_selection_water(self):
        solvent = Solvent.from_selection(1, 25, 0.89)
        self.assertEqual(solvent.sol_type, "Water")
        self.assertEqual(solvent.molecular_weight, 18.01528)
        self.assertEqual(solvent.alpha, 2.6)
    
    def test_from_selection_invalid(self):
        with self.assertRaises(ValueError):
            Solvent.from_selection(5, 25, 0.89)

if __name__ == '__main__':
    unittest.main()
