import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED
from typing import Dict, Optional, List, Tuple, Any


def compute_reward(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return 0.0

    return float(QED.qed(mol))


def compute_supervised_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


def compute_reward_loss(
    sequence_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    normalize_rewards: bool = True,
) -> torch.Tensor:
    rewards = rewards.to(sequence_log_probs.device).detach()

    if normalize_rewards and rewards.numel() > 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    return -(rewards * sequence_log_probs).mean()


def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    generated_smiles: Optional[List[str]] = None,
    sequence_log_probs: Optional[torch.Tensor] = None,
    rl_weight: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce_loss = compute_supervised_loss(logits, labels)

    metrics = {
        "ce_loss": float(ce_loss.detach().cpu()),
        "rl_loss": 0.0,
        "reward_mean": 0.0,
        "total_loss": float(ce_loss.detach().cpu()),
    }

    if generated_smiles is None or sequence_log_probs is None:
        return ce_loss, metrics

    rewards = torch.tensor(
        [compute_reward(smi) for smi in generated_smiles],
        dtype=torch.float32,
        device=logits.device,
    )

    rl_loss = compute_reward_loss(
        sequence_log_probs=sequence_log_probs,
        rewards=rewards,
    )

    total_loss = ce_loss + rl_weight * rl_loss

    metrics = {
        "ce_loss": float(ce_loss.detach().cpu()),
        "rl_loss": float(rl_loss.detach().cpu()),
        "reward_mean": float(rewards.mean().detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }

    return total_loss, metrics