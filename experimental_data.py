import pandas as pd
import matplotlib.pyplot as plt

class ExperimentalData:
    def __init__(self, filename):
        self.data = pd.read_csv(filename) # changed to read_csv from excel. Csv is more common and more efficient/stable.
        self.mw = self.data['molecular_weight'].tolist()
        self.rejection = self.data['rejection'].tolist()
        self.error = self.data['error'].tolist()

    # def plot_data(self):
    #     plt.errorbar(self.mw, self.rejection, self.error, ls='none', marker='o', capsize=5, ecolor='black')
    #     plt.xlabel('Molecular Weight (g mol-1)')
    #     plt.ylabel('Rejection (%)')
    #     plt.title('Experimental rejection data')
    #     plt.show()
