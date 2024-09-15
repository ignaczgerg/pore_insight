import unittest
from experimental_data import ExperimentalData
import pandas as pd
from unittest.mock import patch

class TestExperimentalData(unittest.TestCase):
    @patch('pandas.read_excel')
    def test_experimental_data_loading(self, mock_read_excel):
        # Mock data
        mock_read_excel.return_value = pd.DataFrame({
            'Molecular Weight (g mol-1)': [100, 200],
            'Rejection (%)': [50, 70],
            'Error (+/- %)': [5, 7]
        })

        data = ExperimentalData("mock_file.xlsx")
        self.assertEqual(data.mw, [100, 200])
        self.assertEqual(data.rejection, [50, 70])
        self.assertEqual(data.error, [5, 7])

if __name__ == '__main__':
    unittest.main()
