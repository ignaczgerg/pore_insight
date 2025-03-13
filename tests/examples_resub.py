import numpy as np
from pore_size_distribution import PSD, TwoPointPSD

# Define input data
rejection_values = np.array([22.69,28.4,48.9,62.67,78.7,85.94,98.37,100,100])
errors = np.array([1.9,1.8,2.1,2.2,2.8,2.8,1.6,0,0])
molecule_weights = np.array([92.14,104.15,134.22,162.27,236.35,272.38,327.33,354.4,422.92])

# Initialize PSD object, Acetone 0.36 cP 20C
psd = PSD(
    rejection_values=rejection_values, 
    errors=errors, 
    molecule_weights=molecule_weights,
    solvent='other',
    temperature=20.0,
    viscosity=0.00036,
    molecular_weight=58.08
)

print("\nExample 1 - Basic usage")

# Calculate pore volumes
psd._get_volume(method='wu')  # You can choose 'schotte', 'wu', or 'joback'

# Access calculated x_values (e.g., pore radii)
print("Pore Radii:", psd.x_values)

# Fit the curve using the Boltzmann model
psd.fit_sigmoid(model_name='gompertz')

# Fit the PSD curve
psd.fit_psd(model_name='gompertz')

# Access PDF parameters
print("PDF Parameters:", psd.pdf_parameters)
print("PDF Parameters:", psd.pdf_parameters_low)
print("PDF Parameters:", psd.pdf_parameters_high)

# Example 2: Using Molecular Structures
# Define input data - Ruo-yu Fu (2023). Water 25C 0.890 cP
rejection_values = np.array([37.96,47.96,54.84,62.96]) #%
errors = np.zeros_like(rejection_values) # I do not have errors in this example... But there's an error if I don't define it.
molecules_structure = ['C(C(CO)O)O', 'C([C@H]([C@H](CO)O)O)O', 'C1[C@H]([C@@H]([C@H](C(O1)O)O)O)O', 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O']

# Initialize PSD object
psd = PSD(
    rejection_values=rejection_values, 
    errors=errors, # errors should be optional...
    molecules_structure=molecules_structure,
    solvent='water',
    temperature=25.0
)

print("\nExample 2 - Using Molecular Structures")

# Calculate molar volumes using Joback method
psd._get_volume(method='joback')

# Access calculated x_values (e.g., pore radii)
print("Pore Radii:", psd.x_values)

# Fit the curve using the Sigmoid model
psd.fit_sigmoid(model_name='sigmoid')

# Fit the PSD curve
psd.fit_psd(model_name='sigmoid')

# Access PDF parameters
print("PDF Parameters:", psd.pdf_parameters)

# Two point Aimar methodology
rs = np.array([0.264,0.409,0.400,0.383,0.424,0.530,0.804,0.570]) # nm
rejections = np.array([22.67,60.00,67.40,66.90,72.09,74.16,100,100])/100 #%

r0, r1 = 4,3

a0_example = rs[r0] # nm
R0_example = rejections[r0]
a1_example = rs[r1] # nm
R1_example = rejections[r1]

twopoints = TwoPointPSD(
    solute_radius_zero=a0_example,
    rejection_zero=R0_example,
    solute_radius_one=a1_example,
    rejection_one=R1_example
)

print("\nExample 3 - Two Points Method")

twopoints.find_pore_distribution_params()
print("log-normal Parameters",twopoints.lognormal_parameters)

twopoints.predict_rejection_curve(a = 0.264)
print("Retention prediction:",twopoints.prediction)