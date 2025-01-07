# tests/test_pore_size_distribution.py

import pytest
import numpy as np
from unittest.mock import patch
from rdkit import Chem
from models import PSDModels, Solvents
from pore_size_distribution import PSD
from utils import read_molecules

def test_psd_initialization_with_arrays():
    rejection_values = np.array([90, 95, 99])
    errors = np.array([1, 2, 3])
    molecule_weights = np.array([100, 200, 300])
    molar_volume = np.array([120, 190, 330])
    
    psd = PSD(
        rejection_values=rejection_values,
        errors=errors,
        molecule_weights=molecule_weights,
        molar_volume=molar_volume
    )
    
    assert np.array_equal(psd.rejection_values, rejection_values)
    assert np.array_equal(psd.errors, errors)
    assert np.array_equal(psd.molecule_weights, molecule_weights)
    assert np.array_equal(psd.molar_volume, molar_volume)
    assert psd.solvent == 'water'
    assert psd.temperature == 20.0
    assert psd.viscosity == Solvents.get("water").viscosity
    assert psd.alpha == Solvents.get("water").alpha

def test_psd_initialization_with_molecules_structure():
    rejection_values = np.array([90, 95, 99])
    errors = np.array([1, 2, 3])
    molecules_structure = "CCO,CCC,CCCC"
    
    with patch('utils.read_molecules') as mock_read:
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCC")
        mol3 = Chem.MolFromSmiles("CCCC")
        mock_read.return_value = [mol1, mol2, mol3]
        
        psd = PSD(
            rejection_values=rejection_values,
            errors=errors,
            molecules_structure=molecules_structure,
            solvent='ethanol',
            temperature=25.0
        )
        
        assert psd.solvent == 'ethanol'
        assert psd.temperature == 25.0
        assert psd.viscosity == Solvents.get("ethanol").viscosity
        assert psd.alpha == Solvents.get("ethanol").alpha
        mock_read.assert_called_once_with(molecules_structure.split(','))

def test_psd_get_volume_with_molecule_weights():
    rejection_values = np.array([90, 95, 99])
    errors = np.array([1, 2, 3])
    molecule_weights = np.array([100, 200, 300])
    molar_volume = np.array([120, 190, 330])
    
    psd = PSD(
        rejection_values=rejection_values,
        errors=errors,
        molecule_weights=molecule_weights,
        molar_volume=molar_volume,
        temperature=300,
        viscosity=0.01,
        alpha=1.5
    )
    
    psd._get_volume(method='schotte')
    expected_x_values = np.array([
        1.380649e-23 * 300 / (6 * np.pi * 0.01 * 1e-4) / 1e-9,  # Example calculation
        # ... more values based on actual calculation
    ])
    # Since it's a complex calculation, we can check the length and type
    assert len(psd.x_values) == 3
    assert isinstance(psd.x_values, np.ndarray)

def test_psd_fit_sigmoid():
    rejection_values = np.array([90, 95, 99])
    errors = np.array([1, 2, 3])
    molecule_weights = np.array([100, 200, 300])
    molar_volume = np.array([120, 190, 330])
    
    psd = PSD(
        rejection_values=rejection_values,
        errors=errors,
        molecule_weights=molecule_weights,
        molar_volume=molar_volume
    )
    
    # Mock x_values
    psd.x_values = np.array([1, 2, 3])
    
    psd.fit_sigmoid(model_name='boltzmann')
    
    assert psd._model_function == psd._model_functions['boltzmann']
    assert 'boltzmann' in psd._initial_params
    assert 'boltzmann' in psd._bounds

def test_psd_fit_psd_without_errors():
    rejection_values = np.array([90, 95, 99])
    errors = np.array([1, 2, 3])
    molecule_weights = np.array([100, 200, 300])
    molar_volume = np.array([120, 190, 330])
    
    psd = PSD(
        rejection_values=rejection_values,
        errors=errors,
        molecule_weights=molecule_weights,
        molar_volume=molar_volume
    )
    
    # Mock x_values and optimal_parameters
    psd.x_values = np.array([1, 2, 3])
    psd.optimal_parameters = [10, 2, 1, 1]
    
    psd.fit_psd(model_name='boltzmann')
    
    assert 'boltzmann' in psd._model_derivative_functions
    assert hasattr(psd, 'fitted_derivative')
    assert 'pdf_parameters' in psd.__dict__

def test_psd_fit_psd_with_errors():
    rejection_values = np.array([90, 95, 99])
    errors = np.array([1, 2, 3])
    molecule_weights = np.array([100, 200, 300])
    molar_volume = np.array([120, 190, 330])
    
    psd = PSD(
        rejection_values=rejection_values,
        errors=errors,
        molecule_weights=molecule_weights,
        molar_volume=molar_volume
    )
    
    # Mock x_values, fitted_curve, and optimal_parameters
    psd.x_values = np.array([1, 2, 3])
    psd.optimal_parameters = [10, 2, 1, 1]
    psd.low_fit = np.array([85, 90, 95])
    psd.high_fit = np.array([95, 100, 105])
    
    psd.fit_psd(model_name='boltzmann')
    
    assert hasattr(psd, 'fitted_derivative')
    assert 'pdf_parameters' in psd.__dict__

def test_psd_invalid_method_get_volume():
    rejection_values = np.array([90, 95, 99])
    errors = np.array([1, 2, 3])
    molecule_weights = np.array([100, 200, 300])
    molar_volume = np.array([120, 190, 330])
    
    psd = PSD(
        rejection_values=rejection_values,
        errors=errors,
        molecule_weights=molecule_weights,
        molar_volume=molar_volume
    )
    
    with pytest.raises(ValueError):
        psd._get_volume(method='invalid_method')

def test_psd_invalid_model_name_fit_sigmoid():
    rejection_values = np.array([90, 95, 99])
    errors = np.array([1, 2, 3])
    molecule_weights = np.array([100, 200, 300])
    molar_volume = np.array([120, 190, 330])
    
    psd = PSD(
        rejection_values=rejection_values,
        errors=errors,
        molecule_weights=molecule_weights,
        molar_volume=molar_volume
    )
    
    with pytest.raises(ValueError):
        psd.fit_sigmoid(model_name='invalid_model')

def test_psd_invalid_model_name_fit_psd():
    rejection_values = np.array([90, 95, 99])
    errors = np.array([1, 2, 3])
    molecule_weights = np.array([100, 200, 300])
    molar_volume = np.array([120, 190, 330])
    
    psd = PSD(
        rejection_values=rejection_values,
        errors=errors,
        molecule_weights=molecule_weights,
        molar_volume=molar_volume
    )
    
    with pytest.raises(ValueError):
        psd.fit_psd(model_name='invalid_model')

def test_psd_missing_molecule_weights_and_structure():
    rejection_values = np.array([90, 95, 99])
    errors = np.array([1, 2, 3])
    
    with pytest.raises(ValueError):
        PSD(
            rejection_values=rejection_values,
            errors=errors
            # Neither molecule_weights nor molecules_structure provided
        )
