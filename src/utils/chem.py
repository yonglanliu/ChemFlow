# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

def safe_mol_wt(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Descriptors.MolWt(mol)