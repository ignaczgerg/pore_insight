# tests/test_utils.py

import pytest
import numpy as np
from rdkit import Chem
from models import read_molecules
from utils import rej_bounds, intp90, rmvstr

def test_rej_bounds():
    rejection = np.array([90, 95, 99])
    error = np.array([1, 2, 3])
    rej_low, rej_high = rej_bounds(rejection, error)
    assert rej_low == [91, 97, 102]
    assert rej_high == [89, 93, 96]

def test_rej_bounds_mismatched_lengths():
    rejection = np.array([90, 95])
    error = np.array([1, 2, 3])
    with pytest.raises(IndexError):
        rej_bounds(rejection, error)

def test_intp90_exact():
    r_values = np.array([1, 2, 3, 4, 5])
    rej_lst = np.array([80, 85, 90, 95, 100])
    x_90 = intp90(r_values, rej_lst)
    assert x_90 == 3

def test_intp90_interpolation():
    r_values = np.array([1, 2, 3, 4, 5])
    rej_lst = np.array([80, 85, 92, 95, 100])
    x_90 = intp90(r_values, rej_lst)
    # Between r=2 and r=3: y=85 to y=92
    # 90 = 85 + (92-85)*(x - 2)/(3-2)
    # x = 2 + (90-85)/7 = 2 + 5/7 ≈ 2.7143
    assert np.isclose(x_90, 2.7143, atol=1e-4)

def test_intp90_not_reached():
    r_values = np.array([1, 2, 3])
    rej_lst = np.array([70, 80, 85])
    x_90 = intp90(r_values, rej_lst)
    assert x_90 == "N/A"

def test_rmvstr():
    lst = [1, "N/A", 2, "Error", 3]
    new_lst = rmvstr(lst)
    assert new_lst == [1, 2, 3]

def test_read_molecules_smiles():
    smiles = ["CCO", "CCC", "CCCC"]
    mols = read_molecules(smiles)
    assert all(mol is not None for mol in mols)
    assert len(mols) == 3
    assert Chem.MolToSmiles(mols[0]) == "CCO"

def test_read_molecules_inchi():
    inchi = ["InChI=1S/CH4/h1H4", "InChI=1S/C2H6/c1-2/h1-2H3"]
    mols = read_molecules(inchi)
    assert all(mol is not None for mol in mols)
    assert len(mols) == 2
    assert Chem.MolToInchi(mols[0]) == "InChI=1S/CH4/h1H4"

def test_read_molecules_invalid():
    smiles = ["CCO", "invalid_smiles", "CCCC"]
    mols = read_molecules(smiles)
    assert mols[0] is not None
    assert mols[1] is None
    assert mols[2] is not None
