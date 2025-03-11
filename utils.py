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
