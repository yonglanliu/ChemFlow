# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdFingerprintGenerator


def smiles_to_morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False,
):
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)

    if use_features:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )
    else:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
        )

    fp = generator.GetFingerprint(mol)

    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def featurize_smiles(
    smiles_list,
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False,
):
    features = [
        smiles_to_morgan_fp(
            smiles=smi,
            radius=radius,
            n_bits=n_bits,
            use_features=use_features,
        )
        for smi in smiles_list
    ]

    return np.asarray(features, dtype=np.float32)


def featurize_smiles_2057(
    smiles_list,
    n_bits: int = 2048,
    radius: int = 2,
    use_features: bool = False,
):
    rows = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))

        if mol is None:
            fp_arr = np.zeros((n_bits,), dtype=np.float32)
            desc = [np.nan] * 9

        else:
            fp_arr = smiles_to_morgan_fp(
                smiles=smi,
                radius=radius,
                n_bits=n_bits,
                use_features=use_features,
            )

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