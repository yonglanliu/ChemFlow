from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def smiles_to_morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(n_bits)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
    )

    arr = np.zeros((n_bits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def featurize_smiles(
    smiles_list,
    radius: int = 2,
    n_bits: int = 2048,
):
    features = [
        smiles_to_morgan_fp(
            smiles=smi,
            radius=radius,
            n_bits=n_bits,
        )
        for smi in smiles_list
    ]

    return np.array(features)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
import numpy as np
import pandas as pd


def featurize_smiles_2057(smiles_list, n_bits=2048):
    rows = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            fp_arr = np.zeros(n_bits)
            desc = [np.nan] * 9
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            fp_arr = np.zeros((n_bits,), dtype=int)
            AllChem.DataStructs.ConvertToNumpyArray(fp, fp_arr)

            desc = [
                Descriptors.MolWt(mol),
                Crippen.MolLogP(mol),
                Descriptors.TPSA(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol),
                Lipinski.RingCount(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.FractionCSP3(mol),
            ]

        rows.append(list(fp_arr) + desc)

    columns = [f"fp_{i}" for i in range(n_bits)] + [
        "MolWt",
        "MolLogP",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "RingCount",
        "HeavyAtomCount",
        "FractionCSP3",
    ]

    return pd.DataFrame(rows, columns=columns)