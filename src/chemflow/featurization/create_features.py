# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator


# ============================================================
# Featurization constants
# ============================================================

DESC_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
    "FractionCSP3",
]

MOL_REP_NAMES = [
    "ECFP4",
    "ECFP6",
    "FCFP4",
    "FCFP6",
    "MACCS",
    "Descriptors",
]

FP_TYPES = ["ecfp4", "ecfp6", "fcfp4", "fcfp6"]
MACCS_TYPES = ["maccs", "macc"]
DESC_TYPES = ["descriptor", "descriptors"]

FP_BITS = {
    "ecfp": 2048,
    "ecfp4": 2048,
    "ecfp6": 2048,
    "fcfp4": 2048,
    "fcfp6": 2048,
    "maccs": 167,
}


# ============================================================
# Helpers
# ============================================================

def _mol_from_smiles(smiles):
    if smiles is None:
        return None

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    return mol


def _bitvect_to_array(fp, n_bits=None):
    if n_bits is None:
        n_bits = fp.GetNumBits()

    arr = np.zeros((int(n_bits),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def get_morgan_generator(
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False,
):
    """
    New RDKit Morgan fingerprint generator.

    use_features=False -> ECFP-like Morgan fingerprint
    use_features=True  -> FCFP-like feature Morgan fingerprint
    """
    if use_features:
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=int(radius),
            fpSize=int(n_bits),
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
    )


# ============================================================
# Fingerprints
# ============================================================

def smiles_to_fp(
    smiles,
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False,
):
    """
    Convert SMILES to Morgan fingerprint array.

    ECFP4: radius=2, use_features=False
    ECFP6: radius=3, use_features=False
    FCFP4: radius=2, use_features=True
    FCFP6: radius=3, use_features=True
    """
    mol = _mol_from_smiles(smiles)

    if mol is None:
        return None

    generator = get_morgan_generator(
        radius=radius,
        n_bits=n_bits,
        use_features=use_features,
    )

    fp = generator.GetFingerprint(mol)

    return _bitvect_to_array(fp, n_bits=n_bits)


def smiles_to_maccs(smiles):
    """
    Convert SMILES to MACCS keys.

    RDKit MACCS has 167 bits.
    """
    mol = _mol_from_smiles(smiles)

    if mol is None:
        return None

    fp = MACCSkeys.GenMACCSKeys(mol)

    return _bitvect_to_array(fp)


def smiles_to_descriptors(smiles):
    mol = _mol_from_smiles(smiles)

    if mol is None:
        return None

    values = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.FractionCSP3(mol),
    ]

    return np.asarray(values, dtype=np.float32)


# ============================================================
# Representation dispatcher
# ============================================================

def smiles_to_representation(smiles, rep_type: str):
    rep_type = str(rep_type).lower().strip()

    if rep_type == "ecfp4":
        return smiles_to_fp(
            smiles,
            radius=2,
            n_bits=FP_BITS["ecfp4"],
            use_features=False,
        )

    if rep_type == "ecfp6":
        return smiles_to_fp(
            smiles,
            radius=3,
            n_bits=FP_BITS["ecfp6"],
            use_features=False,
        )

    if rep_type == "fcfp4":
        return smiles_to_fp(
            smiles,
            radius=2,
            n_bits=FP_BITS["fcfp4"],
            use_features=True,
        )

    if rep_type == "fcfp6":
        return smiles_to_fp(
            smiles,
            radius=3,
            n_bits=FP_BITS["fcfp6"],
            use_features=True,
        )

    if rep_type in MACCS_TYPES:
        return smiles_to_maccs(smiles)

    if rep_type in DESC_TYPES:
        return smiles_to_descriptors(smiles)

    raise ValueError(f"Unsupported molecular representation: {rep_type}")