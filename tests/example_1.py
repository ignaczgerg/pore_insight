import numpy as np
import pore_insight
from pore_insight import pore_size_distribution as psd

# Define input data
rejection_values = np.array([22.69,28.4,48.9,62.67,78.7,85.94,98.37,100,100])
errors = np.array([1.9,1.8,2.1,2.2,2.8,2.8,1.6,0,0])
molecule_weights = np.array([92.14,104.15,134.22,162.27,236.35,272.38,327.33,354.4,422.92])

# rejection_values = np.array([90, 95, 99])
# errors = np.array([1, 2, 3])
# molecule_weights = np.array([100, 200, 300])
# molecule_volumes = np.array([120, 190, 330])

# Initialize PSD object
membrane_psd = psd.PSD(
    rejection_values=rejection_values, 
    errors=errors, 
    solute_mol_weights=molecule_weights, 
    solvent='water',
    # molar_volume=molecule_volumes
)

# Calculate pore volumes
membrane_psd.calculate_radius(method='schotte')  # You can choose 'schotte', 'wu', or 'joback'

# Access calculated x_values (e.g., pore radii)
print("Pore Radii:", membrane_psd.x_radii)

# Fit the curve using the Boltzmann model
membrane_psd.fit_sigmoid(model_name='sigmoid')
print("radii_range:", membrane_psd.radii_range)
print("x_radii:", membrane_psd.x_radii)
# Fit the PSD curve
membrane_psd.fit_psd(model_name='sigmoid')

# Access PDF parameters
print("PDF Parameters:", membrane_psd.pdf_parameters)