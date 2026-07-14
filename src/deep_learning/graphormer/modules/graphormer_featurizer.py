# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data

from src.deep_learning.graphormer.utils.chemistry import (
    allowable_features,
    atom_to_feature_vector,
    bond_to_feature_vector,
    get_bond_feature_dims,
    get_atom_feature_dims,
)


def convert_to_single_emb(x: torch.Tensor, offset: int = 512) -> torch.Tensor:
    feature_num = 1 if x.dim() == 1 else x.size(-1)  # get feature number
    feature_offset = 1 + torch.arange(
        0,
        feature_num * offset,
        offset,
        dtype=torch.long,
        device=x.device,
    )
    return x + feature_offset  # broadcast to add offset to each feature


def reorder_canonical_rank_atoms(mol: Chem.Mol):
    ranks = list(Chem.CanonicalRankAtoms(mol))
    order = [idx for _, idx in sorted((rank, idx) for idx, rank in enumerate(ranks))]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order


@dataclass
class FeaturizerConfig:
    remove_hs: bool = True
    reorder_atoms: bool = False
    multi_hop_max_dist: int = 5


class GraphormerFeaturizer:
    def __init__(
        self,
        remove_hs: bool = True,
        reorder_atoms: bool = False,
        multi_hop_max_dist: int = 5,
    ) -> None:
        self.config = FeaturizerConfig(
            remove_hs=remove_hs,
            reorder_atoms=reorder_atoms,
            multi_hop_max_dist=multi_hop_max_dist,
        )

    # ============================================================
    # Atom vocabulary for diffusion target
    # ============================================================

    @property
    def atom_pad_token(self) -> int:
        return 0

    @property
    def atom_misc_token(self) -> int:
        return 119

    @property
    def atom_mask_token(self) -> int:
        return 120

    @property
    def num_atom_types(self) -> int:
        """
        Clean atom prediction classes:
            0 = PAD
            1~118 = atomic numbers
            119 = misc

        MASK is not a clean target.
        """
        return 120

    @property
    def num_atom_input_tokens(self) -> int:
        """
        Input atom tokens including MASK.
        """
        return 121

    # ============================================================
    # Bond vocabulary for diffusion target
    # ============================================================

    @property
    def bond_pad_token(self) -> int:
        return 0

    @property
    def bond_no_bond_token(self) -> int:
        return 0

    @property
    def bond_mask_token(self) -> int:
        return 6

    @property
    def num_bond_types(self) -> int:
        """
        Clean bond prediction classes:
            0 = NO_BOND
            1 = SINGLE
            2 = DOUBLE
            3 = TRIPLE
            4 = AROMATIC
            5 = misc

        MASK is not a clean target.
        """
        return 6

    @property
    def num_bond_input_tokens(self) -> int:
        """
        Input bond tokens including MASK.
        """
        return 7

    # ============================================================
    # Main API
    # ============================================================

    def smiles_to_mol(self, smiles: str) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        if not self.config.remove_hs:
            mol = Chem.AddHs(mol)

        if self.config.reorder_atoms:
            mol, _ = reorder_canonical_rank_atoms(mol)

        return mol

    def smiles2graph(self, smiles: str) -> dict:
        """
        num_atom_features = 9 : range(0, 8) + 1 for misc
        num_bond_features = 3 : range(0, 2) + 1 for misc
        """
        mol = self.smiles_to_mol(smiles)

        atom_features = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]  
        x = np.asarray(atom_features, dtype=np.int64)  # shape: (num_atoms, num_atom_features)
        num_nodes = x.shape[0]

        edge_index, edge_attr, _ = self._get_edges(mol, num_nodes)  # shape: (2, num_edges), (num_edges, num_bond_features)
        spatial_pos = self._get_spatial_pos(mol)  # shape: (num_nodes, num_nodes)

        edge_input = self._get_edge_input(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_feat=edge_attr,
            spatial_pos=spatial_pos,
        )

        # ============================================================
        # Atom and bond types for diffusion target
        # ============================================================
        # Atom types: 1~118 = atomic numbers, 119 = misc
        atom_types = np.asarray(
            [
                atom.GetAtomicNum()
                if 1 <= atom.GetAtomicNum() <= 118
                else self.atom_misc_token
                for atom in mol.GetAtoms()
            ],
            dtype=np.int64,
        )

        # Bond types: 1 = SINGLE, 2 = DOUBLE, 3 = TRIPLE, 4 = AROMATIC, 5 = misc
        bond_types = np.zeros((num_nodes, num_nodes), dtype=np.int64)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_feat = bond_to_feature_vector(bond)
            bond_type = int(bond_feat[0]) + 1  # 0 is possible_bond_type_list

            bond_types[i, j] = bond_type
            bond_types[j, i] = bond_type

        return {
            "x": torch.from_numpy(x).long(),
            "node_feat": torch.from_numpy(x).long(),
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_attr": torch.from_numpy(edge_attr).long(),
            "num_nodes": num_nodes,
            "spatial_pos": torch.from_numpy(spatial_pos).long(),
            "edge_input": torch.from_numpy(edge_input).long(),
            "atom_types": torch.from_numpy(atom_types).long(),
            "bond_types": torch.from_numpy(bond_types).long(),
        }

    def __call__(self, smiles: str) -> Data:
        """
        return:
            Data object with attributes:
                x: node features (num_nodes, num_atom_features)
                edge_index: edge indices (2, num_edges)
                edge_attr: edge features (num_edges, num_bond_features)
                num_nodes: number of nodes
                spatial_pos: spatial positions (num_nodes, num_nodes)
                edge_input: edge input for Graphormer (num_nodes, num_nodes, multi_hop_max_dist, num_bond_features)
                atom_types: atomic numbers or misc token (num_nodes,)
                bond_types: bond types (num_nodes, num_nodes)
                smiles: original SMILES string
                attn_bias: attention bias for Graphormer (num_nodes + 1, num_nodes + 1)
                attn_edge_type: attention edge types for Graphormer (num_nodes, num_nodes, num_bond_features)
                in_degree: in-degree of nodes (num_nodes,)
                out_degree: out-degree of nodes (num_nodes,)
        """
        graph = self.smiles2graph(smiles)
        data = Data(**graph)
        return self.preprocess_item(data)

    # ============================================================
    # Graph construction
    # ============================================================

    def _get_edges(
        self,
        mol: Chem.Mol,
        num_nodes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_bond_features = 3

        edges = []
        edge_features = []
        adj = np.zeros((num_nodes, num_nodes), dtype=np.int64)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            feat = bond_to_feature_vector(bond)

            edges.append((i, j))
            edge_features.append(feat)

            edges.append((j, i))
            edge_features.append(feat)

            adj[i, j] = 1
            adj[j, i] = 1

        if len(edges) == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
        else:
            edge_index = np.asarray(edges, dtype=np.int64).T
            edge_attr = np.asarray(edge_features, dtype=np.int64)

        return edge_index, edge_attr, adj

    def _get_spatial_pos(self, mol: Chem.Mol) -> np.ndarray:
        return rdmolops.GetDistanceMatrix(mol).astype(np.int64)

    def _get_edge_input(
        self,
        num_nodes: int,
        edge_index: np.ndarray,  # shape (2, num_edges)
        edge_feat: np.ndarray,
        spatial_pos: np.ndarray,
    ) -> np.ndarray:
        """
        num_nodes: number of nodes in the graph
        edge_index: shape (2, num_edges) example: [[0, 1, 2], [1, 2, 0]] for edges (0,1), (1,2), (2,0)
        edge_feat: shape (num_edges, num_bond_features) example: [[0, 1, 0], [1, 0, 0], [2, 0, 1]] for bond features of the edges
        spatial_pos: shape (num_nodes, num_nodes) example: [[0, 1, 2], [1, 0, 1], [2, 1, 0]] for shortest distances between nodes

        return:
            edge_input: shape (num_nodes, num_nodes, multi_hop_max_dist, num_edge_features) 
            example: edge_input[i, j, d, :] = edge features of the edge

        """
        num_edge_features = edge_feat.shape[-1] if edge_feat.shape[0] > 0 else len(get_bond_feature_dims())

        edge_input = np.zeros(
            (
                num_nodes,
                num_nodes,
                self.config.multi_hop_max_dist,
                num_edge_features,
            ),
            dtype=np.int64,
        )  # shape: (num_nodes, num_nodes, multi_hop_max_dist, num_edge_features)

        edge_dict = {}

        for k in range(edge_index.shape[1]):
            i = int(edge_index[0, k])
            j = int(edge_index[1, k])
            edge_dict[(i, j)] = edge_feat[k]

        for i in range(num_nodes):
            for j in range(num_nodes):
                dist = int(spatial_pos[i, j])

                if dist <= 0:
                    continue

                if dist > self.config.multi_hop_max_dist:
                    continue

                if (i, j) in edge_dict:
                    edge_input[i, j, 0, :] = edge_dict[(i, j)]  # assign edge freaturs for direct edges (1-hop). If no direct edge, it will be all zeros (no bond/pad)

        return edge_input 

    # ============================================================
    # Graphormer preprocessing
    # ============================================================

    def preprocess_item(self, item: Data) -> Data:
        x = item.x
        edge_index = item.edge_index
        edge_attr = item.edge_attr

        if x is None:
            raise ValueError("Node features (x) cannot be None.")
        if edge_index is None:
            raise ValueError("Edge index cannot be None.")
        if edge_attr is None:
            raise ValueError("Edge attributes (edge_attr) cannot be None.")
        N = x.size(0)

        x = convert_to_single_emb(x)

        adj = torch.zeros((N, N), dtype=torch.bool)

        # Set adjacency matrix based on edge_index
        if edge_index.numel() > 0: 
            adj[edge_index[0], edge_index[1]] = True

        # Format edge_attr to have shape (num_edges, num_edge_features)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr[:, None]

        # Create attention edge type tensor with shape (N, N, num_edge_features)
        attn_edge_type = torch.zeros(
            (N, N, edge_attr.size(-1)),
            dtype=torch.long,
        )

        # Set attention edge types based on edge_index and edge_attr
        if edge_index.numel() > 0:
            attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
                convert_to_single_emb(edge_attr) + 1
            )

        # Add a virtual node (index N) to the attention bias and edge type tensors
        item.x = x
        item.node_feat = x
        item.attn_bias = torch.zeros((N + 1, N + 1), dtype=torch.float)
        item.attn_edge_type = attn_edge_type
        item.spatial_pos = item.spatial_pos.long()

        item.in_degree = adj.long().sum(dim=1).view(-1)
        item.out_degree = item.in_degree.clone()

        item.edge_input = item.edge_input.long()

        item.atom_types = item.atom_types.long()
        item.bond_types = item.bond_types.long()

        return item