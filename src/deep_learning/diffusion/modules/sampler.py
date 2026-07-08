# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from collections import deque
import pandas as pd

import torch
from torch.nn import functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, rdMolDescriptors

from src.deep_learning.graphormer import (
    GraphormerGraphEncoder,
    GraphormerFeaturizer,
)
from src.deep_learning.diffusion.modules.denoiser import GraphormerDenoiser
from src.deep_learning.diffusion.modules.diffuser import GraphormerDiffuser
from src.deep_learning.diffusion.modules.noise.mask_noise import (
    rebuild_graphormer_inputs_from_types,
)
from src.deep_learning.utils import namespace_to_dict


ALLOWED_ATOMIC_NUMS = [6, 7, 8, 9, 17]  # C, N, O, F, Cl
ALLOWED_BOND_TYPE_CLASSES = [0, 1]  # +1 => SINGLE, DOUBLE

MAX_VALENCE = {
    1: 1.0,
    5: 3.0,
    6: 4.0,
    7: 3.0,
    8: 2.0,
    9: 1.0,
    15: 5.0,
    16: 6.0,
    17: 1.0,
    35: 1.0,
    53: 1.0,
}

BOND_MAP = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}

BOND_ORDER = {
    0: 0.0,
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 1.5,
}

BAD_SMARTS = [
    "[N]-[O]",
    "[O]-[O]",
    "[N]-[N]",
    "[O]-[N]",
    "[O]-[Cl]",
    "[O]-[Br]",
    "[N]-[C]-[O]",
    "[O]-[C]-[O]",
]

BAD_PATTERNS = [Chem.MolFromSmarts(s) for s in BAD_SMARTS]


def passes_basic_filters(mol: Chem.Mol) -> bool:
    if mol is None:
        return False

    smiles = Chem.MolToSmiles(mol)

    if "." in smiles:
        return False

    if mol.GetNumAtoms() < 5:
        return False

    for patt in BAD_PATTERNS:
        if patt is not None and mol.HasSubstructMatch(patt):
            return False

    num_hetero = sum(
        atom.GetAtomicNum() in {7, 8, 9, 17, 35}
        for atom in mol.GetAtoms()
    )

    if num_hetero > 5:
        return False

    return True


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, "r") as f:
        return json.load(f)


def get_sample_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_model_from_config(
    full_config: dict,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[GraphormerDiffuser, GraphormerFeaturizer]:

    encoder_config = SimpleNamespace(**full_config["GraphormerEncoderConfig"])
    denoiser_config = SimpleNamespace(**full_config["GraphormerDenoiserConfig"])
    diffuser_config = SimpleNamespace(**full_config["GraphormerDiffusionConfig"])
    featurizer_config = SimpleNamespace(**full_config["FeaturizerConfig"])

    featurizer = GraphormerFeaturizer(**namespace_to_dict(featurizer_config))
    encoder = GraphormerGraphEncoder(**namespace_to_dict(encoder_config))

    denoiser = GraphormerDenoiser(
        encoder=encoder,
        hidden_dim=denoiser_config.hidden_dim,
        num_atom_types=featurizer.num_atom_types,
        num_bond_types=featurizer.num_bond_types,
        num_timesteps=diffuser_config.num_timesteps,
        dropout=denoiser_config.dropout,
        bond_pair_mode=getattr(denoiser_config, "bond_pair_mode", "sum_mul"),
    )

    model = GraphormerDiffuser(
        denoiser=denoiser,
        num_timesteps=diffuser_config.num_timesteps,
        atom_mask_token=featurizer.atom_mask_token,
        bond_mask_token=featurizer.bond_mask_token,
        atom_pad_token=featurizer.atom_pad_token,
        bond_pad_token=featurizer.bond_pad_token,
        atom_loss_weight=getattr(diffuser_config, "atom_loss_weight", 1.0),
        bond_loss_weight=getattr(diffuser_config, "bond_loss_weight", 1.0),
        bond_binary_loss_weight=getattr(diffuser_config, "bond_binary_loss_weight", 1.0),
        bond_type_loss_weight=getattr(diffuser_config, "bond_type_loss_weight", 1.0),
    )

    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, featurizer


def keep_largest_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return mol

    if not frags:
        return None

    return max(frags, key=lambda m: m.GetNumAtoms())


def graph_to_mol(
    atom_types: torch.Tensor,
    bond_types: torch.Tensor,
    atom_pad_token: int = 0,
    atom_misc_token: int = 119,
    atom_mask_token: int = 120,
    bond_no_bond_token: int = 0,
    bond_misc_token: int = 5,
    bond_mask_token: int = 6,
) -> Chem.Mol | None:

    atom_types_list = atom_types.detach().cpu().tolist()
    bond_types_list = bond_types.detach().cpu().tolist()

    rw_mol = Chem.RWMol()
    atom_idx_map = {}
    current_valence = {}

    for i, atomic_num in enumerate(atom_types_list):
        atomic_num = int(atomic_num)

        if atomic_num in {atom_pad_token, atom_misc_token, atom_mask_token}:
            continue

        if atomic_num <= 0 or atomic_num > 118:
            continue

        idx = rw_mol.AddAtom(Chem.Atom(atomic_num))
        atom_idx_map[i] = idx
        current_valence[i] = 0.0

    if len(atom_idx_map) == 0:
        return None

    num_nodes = len(atom_types_list)
    candidate_bonds = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i not in atom_idx_map or j not in atom_idx_map:
                continue

            b = int(bond_types_list[i][j])

            if b in {bond_no_bond_token, bond_misc_token, bond_mask_token}:
                continue

            if b not in BOND_MAP:
                continue

            candidate_bonds.append((i, j, b))

    candidate_bonds.sort(
        key=lambda x: BOND_ORDER.get(x[2], 1.0),
        reverse=True,
    )

    for i, j, b in candidate_bonds:
        ai = int(atom_types_list[i])
        aj = int(atom_types_list[j])

        order = BOND_ORDER.get(b, 1.0)

        if current_valence[i] + order > MAX_VALENCE.get(ai, 4.0):
            continue

        if current_valence[j] + order > MAX_VALENCE.get(aj, 4.0):
            continue

        try:
            rw_mol.AddBond(atom_idx_map[i], atom_idx_map[j], BOND_MAP[b])
            current_valence[i] += order
            current_valence[j] += order
        except Exception:
            continue

    mol = rw_mol.GetMol()

    if mol.GetNumAtoms() == 0:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return mol


def shortest_path_length(
    adj: torch.Tensor,
    start: int,
    end: int,
) -> int | None:

    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        node, dist = queue.popleft()

        if node == end:
            return dist

        neighbors = torch.where(adj[node] > 0)[0].detach().cpu().tolist()

        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + 1))

    return None

def mol_properties(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)

    return {
        "smiles": smiles,
        "mol_wt": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "qed": QED.qed(mol),
    }

class MaskGraphormerDiffusionSampler:
    def __init__(
        self,
        model,
        featurizer,
        device,
        spatial_pos_max: int = 20,
        multi_hop_max_dist: int = 5,
        allowed_atomic_nums: list[int] | None = None,
        allowed_bond_type_classes: list[int] | None = None,
        forbid_cycles: bool = False,
        min_cycle_size: int = 5,
        max_cycle_edges: int = 1,
    ):
        self.model = model
        self.featurizer = featurizer
        self.device = device

        self.num_atom_types = featurizer.num_atom_types
        self.num_bond_types = featurizer.num_bond_types

        self.atom_pad_token = featurizer.atom_pad_token
        self.atom_misc_token = featurizer.atom_misc_token
        self.atom_mask_token = featurizer.atom_mask_token

        self.bond_pad_token = featurizer.bond_pad_token
        self.bond_no_bond_token = featurizer.bond_no_bond_token
        self.bond_mask_token = featurizer.bond_mask_token
        self.bond_misc_token = 5

        self.spatial_pos_max = spatial_pos_max
        self.multi_hop_max_dist = multi_hop_max_dist

        self.allowed_atomic_nums = allowed_atomic_nums or ALLOWED_ATOMIC_NUMS
        self.allowed_bond_type_classes = (
            allowed_bond_type_classes or ALLOWED_BOND_TYPE_CLASSES
        )

        self.forbid_cycles = forbid_cycles
        self.min_cycle_size = min_cycle_size
        self.max_cycle_edges = max_cycle_edges

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_nodes: int,
        num_steps: int,
        temperature: float = 0.40,
        atom_top_k: int | None = 3,
        bond_type_top_k: int | None = 2,
        exist_threshold: float = 0.27,
        return_trajectory: bool = False,
    ):

        self.model.eval()

        atom_types = torch.full(
            (batch_size, num_nodes),
            self.atom_mask_token,
            dtype=torch.long,
            device=self.device,
        )

        bond_types = torch.full(
            (batch_size, num_nodes, num_nodes),
            self.bond_mask_token,
            dtype=torch.long,
            device=self.device,
        )

        node_mask = torch.ones(
            batch_size,
            num_nodes,
            dtype=torch.bool,
            device=self.device,
        )

        eye = torch.eye(num_nodes, dtype=torch.bool, device=self.device)

        upper = torch.triu(
            torch.ones(num_nodes, num_nodes, dtype=torch.bool, device=self.device),
            diagonal=1,
        ).unsqueeze(0)

        bond_types[:, eye] = self.bond_no_bond_token

        current_valence = torch.zeros(
            batch_size,
            num_nodes,
            dtype=torch.float,
            device=self.device,
        )

        adj_for_cycle = torch.zeros(
            batch_size,
            num_nodes,
            num_nodes,
            dtype=torch.long,
            device=self.device,
        )

        cycle_edge_count = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=self.device,
        )

        trajectory = []

        def get_max_valence(atom_tensor: torch.Tensor) -> torch.Tensor:
            out = torch.full_like(atom_tensor.float(), 4.0)

            for atomic_num, val in MAX_VALENCE.items():
                out = torch.where(
                    atom_tensor == int(atomic_num),
                    torch.full_like(out, float(val)),
                    out,
                )

            out = torch.where(
                atom_tensor.eq(self.atom_mask_token),
                torch.full_like(out, 4.0),
                out,
            )

            out = torch.where(
                atom_tensor.eq(self.atom_pad_token),
                torch.zeros_like(out),
                out,
            )

            return out

        for step in reversed(range(num_steps)):
            batch = self._build_batch(
                atom_types=atom_types,
                bond_types=bond_types,
                node_mask=node_mask,
            )

            batch["t"] = torch.full(
                (batch_size,),
                step,
                dtype=torch.long,
                device=self.device,
            )

            atom_logits, bond_exist_logits, bond_type_logits = self.model.denoiser(batch)

            atom_mask = torch.full_like(atom_logits, -1e9)

            for atomic_num in self.allowed_atomic_nums:
                if 0 <= atomic_num < atom_logits.size(-1):
                    atom_mask[..., atomic_num] = atom_logits[..., atomic_num]

            atom_logits = atom_mask

            if 6 < atom_logits.size(-1):
                atom_logits[..., 6] += 0.8

            if 7 < atom_logits.size(-1):
                atom_logits[..., 7] -= 0.4

            if 8 < atom_logits.size(-1):
                atom_logits[..., 8] -= 0.4

            if 9 < atom_logits.size(-1):
                atom_logits[..., 9] -= 0.8

            if 17 < atom_logits.size(-1):
                atom_logits[..., 17] -= 0.8
            atom_sample = self._sample_logits(
                atom_logits,
                temperature=temperature,
                top_k=atom_top_k,
            )

            exist_prob = F.softmax(
                bond_exist_logits / max(temperature, 1e-8),
                dim=-1,
            )[..., 1]

            progress = 1.0 - float(step) / max(float(num_steps - 1), 1.0)
            dynamic_threshold = exist_threshold + 0.05 * progress

            exist_sample = (exist_prob > dynamic_threshold).long()

            bond_type_mask = torch.full_like(bond_type_logits, -1e9)

            for cls in self.allowed_bond_type_classes:
                if 0 <= cls < bond_type_logits.size(-1):
                    bond_type_mask[..., cls] = bond_type_logits[..., cls]

            bond_type_logits = bond_type_mask

            type_sample = self._sample_logits(
                bond_type_logits,
                temperature=temperature,
                top_k=bond_type_top_k,
            ) + 1

            bond_sample = torch.full_like(exist_sample, self.bond_no_bond_token)
            bond_sample[exist_sample == 1] = type_sample[exist_sample == 1]

            ratio = 1.0 / float(step + 1)

            atom_update_mask = (
                atom_types.eq(self.atom_mask_token)
                & (torch.rand_like(atom_types.float()) < ratio)
                & node_mask
            )

            atom_types[atom_update_mask] = atom_sample[atom_update_mask]

            edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

            bond_update_mask_upper = (
                bond_types.eq(self.bond_mask_token)
                & (torch.rand_like(bond_types.float()) < ratio)
                & edge_mask
                & upper
            )

            max_v = get_max_valence(atom_types)

            candidate_indices = bond_update_mask_upper.nonzero(as_tuple=False)

            if candidate_indices.numel() > 0:
                perm = torch.randperm(candidate_indices.size(0), device=self.device)
                candidate_indices = candidate_indices[perm]

            for b, i, j in candidate_indices:
                b = int(b)
                i = int(i)
                j = int(j)

                proposed_bond = int(bond_sample[b, i, j].item())

                if proposed_bond == self.bond_no_bond_token:
                    bond_types[b, i, j] = self.bond_no_bond_token
                    bond_types[b, j, i] = self.bond_no_bond_token
                    continue

                if proposed_bond not in BOND_ORDER:
                    bond_types[b, i, j] = self.bond_no_bond_token
                    bond_types[b, j, i] = self.bond_no_bond_token
                    continue

                path_len = shortest_path_length(adj_for_cycle[b], i, j)

                if path_len is not None:
                    cycle_size = path_len + 1

                    if self.forbid_cycles:
                        bond_types[b, i, j] = self.bond_no_bond_token
                        bond_types[b, j, i] = self.bond_no_bond_token
                        continue

                    if cycle_size < self.min_cycle_size:
                        bond_types[b, i, j] = self.bond_no_bond_token
                        bond_types[b, j, i] = self.bond_no_bond_token
                        continue

                    if cycle_edge_count[b] >= self.max_cycle_edges:
                        bond_types[b, i, j] = self.bond_no_bond_token
                        bond_types[b, j, i] = self.bond_no_bond_token
                        continue

                order = float(BOND_ORDER[proposed_bond])

                if current_valence[b, i] + order > max_v[b, i]:
                    bond_types[b, i, j] = self.bond_no_bond_token
                    bond_types[b, j, i] = self.bond_no_bond_token
                    continue

                if current_valence[b, j] + order > max_v[b, j]:
                    bond_types[b, i, j] = self.bond_no_bond_token
                    bond_types[b, j, i] = self.bond_no_bond_token
                    continue

                bond_types[b, i, j] = proposed_bond
                bond_types[b, j, i] = proposed_bond

                current_valence[b, i] += order
                current_valence[b, j] += order

                adj_for_cycle[b, i, j] = 1
                adj_for_cycle[b, j, i] = 1

                if path_len is not None:
                    cycle_edge_count[b] += 1

            bond_types[:, eye] = self.bond_no_bond_token

            if return_trajectory:
                trajectory.append(
                    {
                        "t": step,
                        "atom_types": atom_types.detach().cpu().clone(),
                        "bond_types": bond_types.detach().cpu().clone(),
                    }
                )

        atom_types = self._finalize_remaining_atoms(atom_types, node_mask)
        bond_types = self._finalize_remaining_bonds(bond_types, node_mask)
        bond_types[:, eye] = self.bond_no_bond_token

        if return_trajectory:
            return atom_types, bond_types, trajectory

        return atom_types, bond_types

    def sample_smiles(
        self,
        batch_size: int = 32,
        num_nodes: int = 12,
        num_steps: int = 100,
        temperature: float = 0.40,
        atom_top_k: int | None = 3,
        bond_type_top_k: int | None = 2,
        exist_threshold: float = 0.27,
        min_atoms: int = 5,
        max_tries: int = 30,
    ) -> list[str | None]:

        final_smiles = []
        seen = set()

        for _ in range(max_tries):
            if len(final_smiles) >= batch_size:
                break

            atom_types, bond_types = self.sample(
                batch_size=batch_size,
                num_nodes=num_nodes,
                num_steps=num_steps,
                temperature=temperature,
                atom_top_k=atom_top_k,
                bond_type_top_k=bond_type_top_k,
                exist_threshold=exist_threshold,
            )

            for i in range(batch_size):
                mol = graph_to_mol(
                    atom_types=atom_types[i],
                    bond_types=bond_types[i],
                    atom_pad_token=self.atom_pad_token,
                    atom_misc_token=self.atom_misc_token,
                    atom_mask_token=self.atom_mask_token,
                    bond_no_bond_token=self.bond_no_bond_token,
                    bond_misc_token=self.bond_misc_token,
                    bond_mask_token=self.bond_mask_token,
                )
                if not passes_basic_filters(mol):
                    continue
                if mol is not None:
                    mol = keep_largest_fragment(mol)

                if mol is None:
                    continue

                if mol.GetNumAtoms() < min_atoms:
                    continue

                smiles = Chem.MolToSmiles(mol)

                if not smiles:
                    continue

                if "." in smiles:
                    continue

                if smiles in seen:
                    continue

                seen.add(smiles)
                final_smiles.append(smiles)

                if len(final_smiles) >= batch_size:
                    break

        while len(final_smiles) < batch_size:
            final_smiles.append(None)

        return final_smiles

    def _sample_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:

        logits = logits / max(temperature, 1e-8)

        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k=top_k, dim=-1)
            threshold = values[..., -1, None]
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, -1e9),
                logits,
            )

        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        row_sum = probs.sum(dim=-1, keepdim=True)

        probs = torch.where(
            row_sum > 0,
            probs / row_sum.clamp_min(1e-12),
            torch.full_like(probs, 1.0 / probs.size(-1)),
        )

        flat_probs = probs.reshape(-1, probs.size(-1))
        sampled = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)

        return sampled.reshape(logits.shape[:-1])

    def _finalize_remaining_atoms(
        self,
        atom_types: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:

        atom_types = atom_types.clone()

        unresolved = atom_types.eq(self.atom_mask_token) & node_mask
        atom_types[unresolved] = 6
        atom_types[~node_mask] = self.atom_pad_token

        return atom_types

    def _finalize_remaining_bonds(
        self,
        bond_types: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:

        bond_types = bond_types.clone()

        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        unresolved = bond_types.eq(self.bond_mask_token) & edge_mask

        bond_types[unresolved] = self.bond_no_bond_token
        bond_types[~edge_mask] = self.bond_pad_token

        return bond_types

    def _build_batch(
        self,
        atom_types: torch.Tensor,
        bond_types: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        return rebuild_graphormer_inputs_from_types(
            atom_types=atom_types,
            bond_types=bond_types,
            node_mask=node_mask,
            atom_mask_token=self.atom_mask_token,
            bond_mask_token=self.bond_mask_token,
            bond_no_bond_token=self.bond_no_bond_token,
            spatial_pos_max=self.spatial_pos_max,
            multi_hop_max_dist=self.multi_hop_max_dist,
        )


if __name__ == "__main__":
    config_path = "/Users/yonglanliu/Desktop/ChemFlow/diffusion_training/config.json"
    checkpoint_path = "/Users/yonglanliu/Desktop/ChemFlow/diffusion_training/checkpoints/best_model.pt"

    device = get_sample_device("auto")
    config = load_config(config_path)

    model, featurizer = build_model_from_config(
        full_config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    sampler = MaskGraphormerDiffusionSampler(
        model=model,
        featurizer=featurizer,
        device=device,
        allowed_atomic_nums=[6, 7, 8, 9, 17],
        allowed_bond_type_classes=[0, 1],
        forbid_cycles=False,
        min_cycle_size=5,
        max_cycle_edges=1,
    )

    smiles = sampler.sample_smiles(
        batch_size=16,
        num_nodes=30,
        num_steps=200,
        temperature=0.40,
        atom_top_k=3,
        bond_type_top_k=4,
        exist_threshold=0.27,
        min_atoms=5,
        max_tries=30,
    )

    # for s in smiles:
    #     print(s)
    valid_smiles = [s for s in smiles if s is not None]

    df = pd.DataFrame([mol_properties(s) for s in valid_smiles])
    df.to_csv("generated_molecules.csv", index=False)

    print(df)