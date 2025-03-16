# PoreInsight

## Table of Contents
- Overview
- Features
- Installation
- Usage
    - Example 1: Basic Usage
    - Example 2: Using Molecular Structures
- Contributing
- License
- Acknowledgements

## Overview
PoreInsight is a Python-based application designed to analyze and model pore size distributions (PSD) of membrane materials. The application is buildt on scientific libraries such as RDKit, NumPy, and SciPy, this tool provides a comprehensive suite of functionalities including curve fitting, diffusivity calculations, and molar volume estimations based on molecular structures.

## Features
- Curve Fitting Models: Supports various curve fitting models like Boltzmann, Sigmoid, Generalized Logistic, Gompertz, and Double Sigmoid.
- Pore Size Distribution Models: Includes log-normal distribution and derivatives of sigmoid-based models.
- Molar Volume Estimation: Utilizes methods from Schotte, Wu, and Joback to estimate molar volumes from molecular structures.
- Diffusivity Calculations: Implements the Wilke-Chang equation and Stokes-Einstein relation for diffusivity and radius calculations.
- Solvent Handling: Predefined solvent properties with the ability to extend to custom solvents.
- Command-Line Interface: Provides an easy-to-use CLI for processing PSD parameters.

## Installation
### Prerequisites
- Python 3.13
- RDKit
- NumPy
- SciPy
- optional: Matplotlib

### Clone Repository
```bash
git clone git@github.com:ignaczgerg/pore_insight.git
cd pore_insight
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Install PoreInsight
```bash
pip install -e .
```
## Usage
### Example 1: Basic Usage
Here's a simple example demonstrating how to use the PSD class to fit a pore size distribution curve based on rejection values, errors, molecule weights, and molar volumes.
```python
import numpy as np
import pore_insight
from pore_insight import pore_size_distribution as psd

# Define input data
rejection_values = np.array([22.69,28.4,48.9,62.67,78.7,85.94,98.37,100,100])
errors = np.array([1.9,1.8,2.1,2.2,2.8,2.8,1.6,0,0])
molecule_weights = np.array([92.14,104.15,134.22,162.27,236.35,272.38,327.33,354.4,422.92])

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
```

### Example 2: Using Molecular Structures
This example shows how to process molecular structures (SMILES strings) to estimate molar volumes and perform PSD analysis.
```python
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
membrane_2_psd.fit_sigmoid(model_name='sigmoid')

# Fit the PSD curve
membrane_2_psd.fit_psd(model_name='sigmoid')

# Access PDF parameters
print("PDF Parameters:", membrane_2_psd.pdf_parameters)
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any bugs, features, or enhancements.

- Fork the repository.
- Create your feature branch: git checkout -b feature/YourFeature
- Commit your changes: git commit -m 'Add some feature'
- Push to the branch: git push origin feature/YourFeature
- Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- RDKit: https://www.rdkit.org/
- NumPy: https://numpy.org/
- SciPy: https://www.scipy.org/

For any questions or support, please open an issue on the GitHub repository.
