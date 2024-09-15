import pandas as pd
import matplotlib.pyplot as plt

class ExperimentalData:
    def __init__(self, filename):
        self.data = pd.read_excel(filename)
        self.mw = self.data['Molecular Weight (g mol-1)'].tolist()
        self.rejection = self.data['Rejection (%)'].tolist()
        self.error = self.data['Error (+/- %)'].tolist()

    def plot_data(self):
        plt.errorbar(self.mw, self.rejection, self.error, ls='none', marker='o', capsize=5, ecolor='black')
        plt.xlabel('Molecular Weight (g mol-1)')
        plt.ylabel('Rejection (%)')
        plt.title('Experimental rejection data')
        plt.show()
