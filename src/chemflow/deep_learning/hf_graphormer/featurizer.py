"""RDKit graph conversion for the Hugging Face Graphormer preprocessor."""

from __future__ import annotations

import numpy as np
from rdkit import Chem


ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)) + ["misc"],
    "chirality": [
        "CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER", "misc",
    ],
    "degree": list(range(11)) + ["misc"],
    "formal_charge": list(range(-5, 6)) + ["misc"],
    "num_h": list(range(9)) + ["misc"],
    "radical_electrons": list(range(5)) + ["misc"],
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "aromatic": [False, True],
    "in_ring": [False, True],
}

BOND_FEATURES = {
    "type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "stereo": [
        "STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS",
        "STEREOTRANS", "STEREOANY",
    ],
    "conjugated": [False, True],
}


def _safe_index(values: list, value) -> int:
    try:
        return values.index(value)
    except ValueError:
        return len(values) - 1


def _atom_features(atom: Chem.Atom) -> list[int]:
    return [
        _safe_index(ATOM_FEATURES["atomic_num"], atom.GetAtomicNum()),
        _safe_index(ATOM_FEATURES["chirality"], str(atom.GetChiralTag())),
        _safe_index(ATOM_FEATURES["degree"], atom.GetTotalDegree()),
        _safe_index(ATOM_FEATURES["formal_charge"], atom.GetFormalCharge()),
        _safe_index(ATOM_FEATURES["num_h"], atom.GetTotalNumHs()),
        _safe_index(ATOM_FEATURES["radical_electrons"], atom.GetNumRadicalElectrons()),
        _safe_index(ATOM_FEATURES["hybridization"], str(atom.GetHybridization())),
        ATOM_FEATURES["aromatic"].index(atom.GetIsAromatic()),
        ATOM_FEATURES["in_ring"].index(atom.IsInRing()),
    ]


def _bond_features(bond: Chem.Bond) -> list[int]:
    return [
        _safe_index(BOND_FEATURES["type"], str(bond.GetBondType())),
        _safe_index(BOND_FEATURES["stereo"], str(bond.GetStereo())),
        BOND_FEATURES["conjugated"].index(bond.GetIsConjugated()),
    ]


class GraphormerFeaturizer:
    """Create the raw graph dictionary expected by HF ``preprocess_item``."""

    def __init__(self, remove_hs: bool = True, reorder_atoms: bool = False) -> None:
        self.remove_hs = bool(remove_hs)
        self.reorder_atoms = bool(reorder_atoms)

    def smiles2graph(self, smiles: str) -> dict[str, np.ndarray | int]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        if not self.remove_hs:
            mol = Chem.AddHs(mol)
        if self.reorder_atoms:
            ranks = list(Chem.CanonicalRankAtoms(mol))
            order = [index for _, index in sorted((rank, index) for index, rank in enumerate(ranks))]
            mol = Chem.RenumberAtoms(mol, order)

        node_features = np.asarray(
            [_atom_features(atom) for atom in mol.GetAtoms()], dtype=np.int64
        )
        edges: list[tuple[int, int]] = []
        edge_features: list[list[int]] = []
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            features = _bond_features(bond)
            edges.extend(((begin, end), (end, begin)))
            edge_features.extend((features, features))

        if edges:
            edge_index = np.asarray(edges, dtype=np.int64).T
            edge_attr = np.asarray(edge_features, dtype=np.int64)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, 3), dtype=np.int64)

        return {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "node_feat": node_features,
            "x": node_features,
            "num_nodes": int(node_features.shape[0]),
        }
