import torch 
from rdkit import Chem
from src.deep_learning.graphormer import GraphormerGraphEncoder, GraphormerFeaturizer
from src.deep_learning.diffusion.modules.denoiser import GraphormerDenoiser
from src.deep_learning.diffusion.modules.diffuser import GraphormerDiffuser

from types import SimpleNamespace
from src.deep_learning.utils import namespace_to_dict

from src.deep_learning.diffusion.modules.noise.mask_noise import (
    rebuild_graphormer_inputs_from_types,
)
from pathlib import Path
import json
from torch.nn import functional as F
# ============================================================
# Utilities
# ============================================================

ATOM_LABELS = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

BOND_LABELS = {
    1: "-",
    2: "=",
    3: "#",
    4: ":",
}


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

    featurizer = GraphormerFeaturizer(
        **namespace_to_dict(featurizer_config)
    )

    encoder = GraphormerGraphEncoder(
        **namespace_to_dict(encoder_config)
    )

    denoiser = GraphormerDenoiser(
        encoder=encoder,
        hidden_dim=denoiser_config.hidden_dim,
        num_atom_types=featurizer.num_atom_types,
        num_bond_types=featurizer.num_bond_types,
        num_timesteps=diffuser_config.num_timesteps,
        dropout=denoiser_config.dropout,
        bond_pair_mode=getattr(denoiser_config, "bond_pair_mode", "sum"),
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
    )

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, featurizer


def keep_largest_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    return max(frags, key=lambda m: m.GetNumAtoms())


def graph_to_mol(
    atom_types: torch.Tensor,
    bond_types: torch.Tensor,
) -> Chem.Mol | None:

    atom_types = atom_types.detach().cpu().tolist()
    bond_types = bond_types.detach().cpu().tolist()

    rw_mol = Chem.RWMol()
    atom_idx_map = {}

    max_valence = {
        6: 4,
        7: 3,
        8: 2,
        9: 1,
        15: 5,
        16: 6,
        17: 1,
        35: 1,
        53: 1,
    }

    bond_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    bond_order = {
        1: 1,
        2: 2,
        3: 3,
        4: 1.5,
    }

    current_valence = {}

    for i, atomic_num in enumerate(atom_types):
        atomic_num = int(atomic_num)

        if atomic_num <= 0 or atomic_num > 118:
            continue

        atom = Chem.Atom(atomic_num)
        idx = rw_mol.AddAtom(atom)

        atom_idx_map[i] = idx
        current_valence[i] = 0.0

    N = len(atom_types)
    candidate_bonds = []

    for i in range(N):
        for j in range(i + 1, N):
            b = int(bond_types[i][j])

            if b == 0:
                continue

            if b not in bond_map:
                continue

            if i not in atom_idx_map or j not in atom_idx_map:
                continue

            candidate_bonds.append((i, j, b))

    candidate_bonds.sort(key=lambda x: bond_order[x[2]])

    for i, j, b in candidate_bonds:
        ai = int(atom_types[i])
        aj = int(atom_types[j])

        vi_max = max_valence.get(ai, 4)
        vj_max = max_valence.get(aj, 4)

        bo = bond_order[b]

        if current_valence[i] + bo > vi_max:
            continue

        if current_valence[j] + bo > vj_max:
            continue

        try:
            rw_mol.AddBond(
                atom_idx_map[i],
                atom_idx_map[j],
                bond_map[b],
            )
            current_valence[i] += bo
            current_valence[j] += bo
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

# ============================================================
# Sampler with trajectory
# ============================================================

class MaskGraphormerDiffusionSampler:
    def __init__(
        self,
        model,
        featurizer,
        device,
        spatial_pos_max: int = 20,
        multi_hop_max_dist: int = 5,
    ):
        self.model = model
        self.featurizer = featurizer
        self.device = device

        self.num_atom_types = featurizer.num_atom_types
        self.num_bond_types = featurizer.num_bond_types

        self.atom_mask_token = featurizer.atom_mask_token
        self.bond_mask_token = featurizer.bond_mask_token
        self.atom_pad_token = featurizer.atom_pad_token
        self.bond_pad_token = featurizer.bond_pad_token

        self.bond_no_bond_token = featurizer.bond_pad_token

        self.spatial_pos_max = spatial_pos_max
        self.multi_hop_max_dist = multi_hop_max_dist

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_nodes: int,
        num_steps: int,
        temperature: float = 1.0,
        top_k: int | None = None,
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
        bond_types[:, eye] = self.bond_no_bond_token

        trajectory = []

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

            atom_logits, bond_logits = self.model.denoiser(batch)

            atom_logits[..., self.atom_pad_token] = -1e9
            if self.atom_mask_token < atom_logits.size(-1):
                atom_logits[..., self.atom_mask_token] = -1e9

            if self.bond_mask_token < bond_logits.size(-1):
                bond_logits[..., self.bond_mask_token] = -1e9

            # Conservative sampling: only no-bond and single bond.
            # Remove this line later if you want double/triple/aromatic bonds.
            bond_logits[..., 3:] = -1e9

            atom_sample = self._sample_logits(
                atom_logits,
                temperature=temperature,
                top_k=top_k,
            )

            bond_sample = self._sample_logits(
                bond_logits,
                temperature=temperature,
                top_k=top_k,
            )

            ratio = 1.0 / float(step + 1)

            atom_update_mask = (
                atom_types.eq(self.atom_mask_token)
                & (torch.rand_like(atom_types.float()) < ratio)
                & node_mask
            )

            edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

            bond_update_mask = (
                bond_types.eq(self.bond_mask_token)
                & (torch.rand_like(bond_types.float()) < ratio)
                & edge_mask
            )

            atom_types[atom_update_mask] = atom_sample[atom_update_mask]
            bond_types[bond_update_mask] = bond_sample[bond_update_mask]

            bond_types = torch.maximum(bond_types, bond_types.transpose(1, 2))
            bond_types[:, eye] = self.bond_no_bond_token

            if return_trajectory:
                trajectory.append(
                    {
                        "t": step,
                        "atom_types": atom_types.detach().cpu().clone(),
                        "bond_types": bond_types.detach().cpu().clone(),
                    }
                )

        atom_types = atom_types.clamp(0, self.num_atom_types - 1)
        bond_types = bond_types.clamp(0, self.num_bond_types - 1)

        if return_trajectory:
            return atom_types, bond_types, trajectory

        return atom_types, bond_types

    def sample_smiles(
        self,
        batch_size: int = 16,
        num_nodes: int = 32,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> list[str | None]:

        atom_types, bond_types = self.sample(
            batch_size=batch_size,
            num_nodes=num_nodes,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
        )

        smiles_list = []

        for i in range(batch_size):
            mol = graph_to_mol(
                atom_types=atom_types[i],
                bond_types=bond_types[i],
            )

            if mol is not None:
                mol = keep_largest_fragment(mol)

            if mol is None:
                smiles_list.append(None)
            else:
                smiles_list.append(Chem.MolToSmiles(mol))

        return smiles_list

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

        flat_probs = probs.reshape(-1, probs.size(-1))
        sampled = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)

        return sampled.reshape(logits.shape[:-1])

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
