# Pore Insight

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
- Python 3.8 or higher
- RDKit
- NumPy
- SciPy
- optional: Matplotlib

### Clone Repository
```bash
git clone git@github.com:ignaczgerg/pore_size_distribution_estimation.git
cd pore_size_distribution_estimation
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
from models import PSD

# Define input data
rejection_values = np.array([90, 95, 99])
errors = np.array([1, 2, 3])
molecule_weights = np.array([100, 200, 300])
molecule_volumes = np.array([120, 190, 330])

# Initialize PSD object
psd = PSD(
    rejection_values=rejection_values, 
    errors=errors, 
    molecule_weights=molecule_weights, 
    molar_volume=molecule_volumes
)

# Calculate pore volumes
psd._get_volume(method='schotte')  # You can choose 'schotte', 'wu', or 'joback'

# Access calculated x_values (e.g., pore radii)
print("Pore Radii:", psd.x_values)

# Fit the curve using the Boltzmann model
psd.fit_sigmoid(model_name='boltzmann')

# Fit the PSD curve
psd.fit_psd(model_name='boltzmann')

# Access PDF parameters
print("PDF Parameters:", psd.pdf_parameters)
```

### Example 2: Using Molecular Structures
This example shows how to process molecular structures (SMILES strings) to estimate molar volumes and perform PSD analysis.
```python
import numpy as np
from models import PSD

# Define input data
rejection_values = np.array([90, 95, 99])
errors = np.array([1, 2, 3])
molecules_structure = "CCO,CCC,CCCC"

# Initialize PSD object
psd = PSD(
    rejection_values=rejection_values, 
    errors=errors, 
    molecules_structure=molecules_structure,
    solvent='ethanol',
    temperature=25.0
)

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
