import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED


@torch.no_grad()
def evaluate_ce_loss(model, val_loader, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(input_ids)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
    }


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def evaluate_generated_smiles(smiles_list, train_smiles_set=None):
    valid_smiles = []

    for smi in smiles_list:
        can = canonicalize_smiles(smi)

        if can is not None:
            valid_smiles.append(can)

    num_total = len(smiles_list)
    num_valid = len(valid_smiles)
    unique_smiles = set(valid_smiles)

    validity = num_valid / num_total if num_total > 0 else 0.0
    uniqueness = len(unique_smiles) / num_valid if num_valid > 0 else 0.0

    if train_smiles_set is not None:
        novelty = len(unique_smiles - train_smiles_set) / len(unique_smiles) if unique_smiles else 0.0
    else:
        novelty = None

    qed_scores = []
    for smi in valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        qed_scores.append(QED.qed(mol))

    avg_qed = sum(qed_scores) / len(qed_scores) if qed_scores else 0.0

    return {
        "num_total": num_total,
        "num_valid": num_valid,
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "avg_qed": avg_qed,
    }


@torch.no_grad()
def evaluate_generation(
    model,
    tokenizer,
    device,
    num_samples=1000,
    batch_size=64,
    max_generation_length=128,
    temperature=0.8,
    top_k=50,
    condition_ids=None,
    train_smiles_set=None,
):
    model.eval()

    all_smiles = []

    remaining = num_samples

    while remaining > 0:
        current_batch_size = min(batch_size, remaining)

        if condition_ids is not None:
            batch_condition_ids = condition_ids[:current_batch_size].to(device)
        else:
            batch_condition_ids = None

        generated_ids = model.generate(
            batch_size=current_batch_size,
            max_length=max_generation_length,
            condition_ids=batch_condition_ids,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )

        smiles_list = model.decode(
            tokenizer=tokenizer,
            input_ids=generated_ids,
        )

        all_smiles.extend(smiles_list)
        remaining -= current_batch_size

    metrics = evaluate_generated_smiles(
        smiles_list=all_smiles,
        train_smiles_set=train_smiles_set,
    )

    return metrics, all_smiles