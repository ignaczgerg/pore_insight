import numpy as np
import pore_insight
from pore_insight import pore_size_distribution as psd

# Define input data
rejection_values = np.array([15, 50, 98, 99, 99.9])
errors = np.array([1, 2, 3, 0, 0])
molecules_structure = ["CCO","CCCO","CCCCO", "CCCCCCO", "CCCCCCCCCCO"]

# Initialize PSD object
membrane_2_psd = psd.PSD(
    rejection_values=rejection_values, 
    errors=errors, 
    molecules_structure=molecules_structure,
    solvent='ethanol',
    temperature=25.0
)

# Calculate molar volumes using Joback method
membrane_2_psd.calculate_radius(method='joback')

# Access calculated x_values (e.g., pore radii)
print("Pore Radii:", membrane_2_psd.x_radii)

# Fit the curve using the Sigmoid model
membrane_2_psd.fit_sigmoid(model_name='gompertz')

# Fit the PSD curve
membrane_2_psd.fit_psd(model_name='gompertz')

# Access PDF parameters
print("PDF Parameters:", membrane_2_psd.pdf_parameters)