import numpy as np
from rdkit import Chem
from typing import List, Union



    

def rej_bounds(rejection, error):
    """
    Estimates the lower and higher bounds based on rejection errors. Array-like objects should be of the same size.

    Parameters
    ----------
    rejection : array-like
        Experimental rejection points.
    error : array-like
        Experimental rejection error points (+/-).

    Returns
    -------
    rej_low : list
        Rejection points in the low bound of error (+).
    rej_high : list
        Rejection points in the high bound of error (-).
    """
    if len(rejection) != len(error):
        raise IndexError("rejection and error arrays must have the same length")

    rej_low = rejection + error
    rej_high = rejection - error
    return rej_low.tolist(), rej_high.tolist()

def intp90(r_values,rej_lst):
    """
    Estimates the radius or molecular weight (x value) value at 90% rejection (y value).

    Parameters
    ----------
    rej_lst : array-like
        List of fitted rejections.
    x : array-like
        Radius range obtained from the curve fitting.

    Returns
    ----------
    x_90 : float or str
        Radius or molecular weight float value at 90% rejection.
        If the curve does not reach 90%, str value "N/A" will be returned and discarded from calculations.
    """

    if rej_lst[-1] > 90:
        for i in range(0,len(rej_lst)):
            if rej_lst[i] > 90:
                y1 = rej_lst[i-1]
                y2 = rej_lst[i]
                x1 = r_values[i-1]
                x2 = r_values[i]
                
                x_90 = x1 + ((90-y1)/(y2-y1))*(x2-x1)
                break
    elif rej_lst[-1] == 90:
        x_90 = rej_lst[-1]
    else:
        x_90 = str("N/A")
    
    return x_90

def rmvstr(lst):
    """
    Removes any str value from a list.

    Parameters
    ----------
    lst : array-like
        List of values

    Returns
    ----------
    lst : array-lie
        Same list without str value(s)
    """
    new_lst = []
    for i in lst:
        if type(i) != str:
            new_lst.append(i)

    return new_lst


def read_molecules(molecule_strings: List[str]) -> List[Union[Chem.Mol, None]]:
    mols = []
    for mol_str in molecule_strings:
        try:
            if mol_str.startswith("InChI="):
                mol = Chem.MolFromInchi(mol_str)
                if mol is None:
                    raise ValueError(f"Invalid InChI key: {mol_str}")
            else:
                mol = Chem.MolFromSmiles(mol_str)
                if mol is None:
                    raise ValueError(f"Invalid SMILES string: {mol_str}")
            mols.append(mol)
        except Exception as e:
            print(f"Error processing molecule string {mol_str}: {e}")
            mols.append(None)
    return mols
