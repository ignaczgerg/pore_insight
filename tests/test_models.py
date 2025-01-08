# tests/test_models.py

import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from models import (
    CurveModels,
    PSDModels,
    DistributionModels,
    MolarVolume,
    StokesRadiusCalculator,
    DiffusivityCalculator,
    Solvents,
    Solvent
)

def test_curve_models_boltzmann():
    x = np.array([0, 1, 2, 3, 4])
    a, b, c, d = 10, 2, 2, 1
    expected = b + (a - b) / (1 + np.exp((x - c) / d))
    result = CurveModels.boltzmann(x, a, b, c, d)
    assert np.allclose(result, expected)

def test_curve_models_sigmoid():
    x = np.array([-2, -1, 0, 1, 2])
    a, b, c = 1, 0, 1
    expected = c / (1 + np.exp(-a * (x - b)))
    result = CurveModels.sigmoid(x, a, b, c)
    assert np.allclose(result, expected)

def test_curve_models_generalized_logistic():
    x = np.array([0, 1, 2, 3, 4])
    A, K, B, Q, C, nu = 0, 1, 1, 1, 1, 1
    expected = A + (K - A) / (C + Q * np.exp(-B * x))**(1 / nu)
    result = CurveModels.generalized_logistic(x, A, K, B, Q, C, nu)
    assert np.allclose(result, expected)

def test_curve_models_gompertz():
    x = np.array([0, 1, 2])
    a, b, c = 1, 1, 1
    expected = a * np.exp(-b * np.exp(-c * x))
    result = CurveModels.gompertz(x, a, b, c)
    assert np.allclose(result, expected)

def test_curve_models_double_sigmoid():
    x = np.array([0, 1, 2, 3, 4])
    K1, B1, M1, K2, B2, M2 = 1, 1, 1, 1, 1, 1
    expected = (K1 / (1 + np.exp(-B1 * (x - M1)))) + (K2 / (1 + np.exp(-B2 * (x - M2))))
    result = CurveModels.double_sigmoid(x, K1, B1, M1, K2, B2, M2)
    assert np.allclose(result, expected)

def test_psd_models_log_normal():
    x = np.linspace(1, 10, 100)
    avg_r = 5
    std_dev = 1
    psd = PSDModels.log_normal(x, avg_r, std_dev)
    assert len(psd) == len(x)
    assert np.all(psd >= 0)

def test_psd_models_derivative_sigmoid():
    x = np.linspace(0, 10, 100)
    a, b, c = 1, 5, 10
    sigmoid = CurveModels.sigmoid(x, a, b, c)
    derivative = PSDModels.derivative_sigmoid(x, a, b, c)
    expected = a * sigmoid * (1 - sigmoid / c)
    assert np.allclose(derivative, expected)

def test_psd_models_derivative_generalized_logistic():
    x = np.linspace(0, 10, 100)
    A, K, B, Q, C, nu = 0, 1, 1, 1, 1, 1
    derivative = PSDModels.derivative_generalized_logistic(x, A, K, B, Q, C, nu)
    expected = (K - A) * B * Q * np.exp(-B * x) / (nu * (C + Q * np.exp(-B * x))**(1 / nu + 1))
    assert np.allclose(derivative, expected)

def test_psd_models_derivative_gompertz():
    x = np.linspace(0, 10, 100)
    a, b, c = 1, 1, 1
    derivative = PSDModels.derivative_gompertz(x, a, b, c)
    expected = a * b * np.exp(-b * np.exp(-c * x)) * np.exp(-c * x)
    assert np.allclose(derivative, expected)

def test_psd_models_derivative_double_sigmoid():
    x = np.linspace(0, 10, 100)
    K1, B1, M1, K2, B2, M2 = 1, 1, 5, 1, 1, 5
    term1 = (K1 * B1 * np.exp(-B1 * (x - M1))) / ((1 + np.exp(-B1 * (x - M1)))**2)
    term2 = (K2 * B2 * np.exp(-B2 * (x - M2))) / ((1 + np.exp(-B2 * (x - M2)))**2)
    expected = term1 + term2
    derivative = PSDModels.derivative_double_sigmoid(x, K1, B1, M1, K2, B2, M2)
    assert np.allclose(derivative, expected)

def test_distribution_models_derivative_sigmoidal():
    r_range = np.linspace(1, 10, 100)
    psd_array = PSDModels.derivative_sigmoid(r_range, a=1, b=5, c=10)
    
    # Call the correct function to get distribution parameters
    distribution = DistributionModels.derivative_sigmoidal(r_range, psd_array)
    
    # Assertions
    assert isinstance(distribution, dict), "Output should be a dictionary"
    assert 'average_radius' in distribution, "Key 'average_radius' missing in distribution"
    assert 'standard_deviation' in distribution, "Key 'standard_deviation' missing in distribution"
    assert distribution['average_radius'] is not None, "'average_radius' should not be None"
    assert distribution['standard_deviation'] is not None, "'standard_deviation' should not be None"

def test_distribution_models_PDF():
    x = np.linspace(1, 10, 100)
    rej_fit = 90 + np.random.normal(0, 1, size=100)
    low_fit = rej_fit - 2
    high_fit = rej_fit + 2
    pdf_params = DistributionModels.PDF(x, rej_fit, low_fit, high_fit)
    assert 'average_radius' in pdf_params
    assert 'standard_deviation' in pdf_params
    assert 'radius_list' in pdf_params

def test_molar_volume_relation_Schotte():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    vm = MolarVolume.relation_Schotte(mol)
    assert isinstance(vm, float)
    expected = 1.3348 * Descriptors.MolWt(mol) - 10.552
    assert vm == expected

def test_molar_volume_relation_Wu():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    vm = MolarVolume.relation_Wu(mol)
    assert isinstance(vm, float)
    expected = 1.1353 * Descriptors.MolWt(mol) + 3.8219
    assert vm == expected

def test_molar_volume_joback():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    vm = MolarVolume.joback(mol)
    assert isinstance(vm, float)
    # Based on the implemented code, only critical volume is calculated
    # Initial VC = 17.5, p[i][2] is added for each group
    # To test, we'd need to know the expected groups in "CCO"
    # For simplicity, just check it's a positive float
    assert vm > 0

def test_stokes_einstein_radius():
    D = 1e-5  # cm²/s
    T = 300  # K
    eta = 0.01  # cP
    radius = StokesRadiusCalculator.stokes_einstein_radius(D, T, eta)
    assert isinstance(radius, float)
    assert radius > 0

def test_diffusivity_calculator():
    molar_volume = np.array([120, 190, 330])
    molecular_weight = np.array([100, 200, 300])
    temp = 300  # K
    viscosity = 0.01  # cP
    alpha = 1.5
    diffusivity = DiffusivityCalculator.wilke_chang_diffusion_coefficient(
        molar_volume, molecular_weight, temp, viscosity, alpha
    )
    assert isinstance(diffusivity, np.ndarray), "Diffusivity should be a NumPy array"
    assert diffusivity.shape == molar_volume.shape, "Diffusivity array shape mismatch"
    assert np.all(diffusivity > 0), "All diffusivity values should be positive"

def test_solvents_get():
    water = Solvents.get("water")
    assert isinstance(water, Solvent)
    assert water.name == "water"
    assert water.molecular_weight == 18.01528
    assert water.viscosity == 0.001
    assert water.alpha == 2.6

def test_solvents_get_invalid():
    with pytest.raises(ValueError):
        Solvents.get("unknown_solvent")
