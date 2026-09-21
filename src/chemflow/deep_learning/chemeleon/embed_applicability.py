"""Backfill applicability-domain references into a trained CheMeleon checkpoint."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemprop import data
from rdkit import Chem

from chemflow.deep_learning.chemeleon.predictor import CheMeleonPredictor
from chemflow.deep_learning.chemeleon.trainer import (
    _build_applicability_payload,
    _complete_batch_size,
    _datapoints,
    _validate_task_coverage,
)


def _manifest_records(
    frame: pd.DataFrame,
    split: str,
    target_names: list[str],
) -> list[dict]:
    selected = frame.loc[frame["split"].astype(str).str.lower() == split].copy()
    records: list[dict] = []
    for row_index, row in selected.iterrows():
        smiles = str(row["SMILES"]).strip()
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        targets = pd.to_numeric(row[target_names], errors="coerce").to_numpy(
            dtype=np.float32
        )
        if molecule is None:
            raise ValueError(f"Invalid SMILES at manifest row {row_index}: {smiles!r}")
        if not np.isfinite(targets).any():
            raise ValueError(f"No finite target at manifest row {row_index}.")
        records.append(
            {
                "original_index": row.get("original_index", row_index),
                "smiles": smiles,
                "targets": targets,
                "mol": molecule,
                "split": split,
            }
        )
    if not records:
        raise ValueError(f"The manifest has no {split!r} records.")
    _validate_task_coverage(records, target_names, split)
    return records


def embed_applicability(
    checkpoint_path: str | Path,
    split_manifest_path: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    backup: bool = True,
) -> Path:
    """Calculate references/calibration and atomically update a checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    split_manifest_path = Path(split_manifest_path).expanduser().resolve()
    predictor = CheMeleonPredictor(
        checkpoint_path, device="cpu", applicability_domain=False
    )
    frame = pd.read_csv(split_manifest_path)
    # New manifests consistently use the conventional uppercase SMILES name.
    # Accept and normalize older ChemFlow manifests for backward compatibility.
    if "SMILES" not in frame.columns and "smiles" in frame.columns:
        frame = frame.rename(columns={"smiles": "SMILES"})
    required = {"SMILES", "split", *predictor.target_names}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Split manifest is missing columns: {missing}")

    train_records = _manifest_records(frame, "train", predictor.target_names)
    val_records = _manifest_records(frame, "validation", predictor.target_names)

    def loader(records: list[dict]):
        dataset = data.MoleculeDataset(_datapoints(records), predictor.featurizer)
        return data.build_dataloader(
            dataset,
            batch_size=_complete_batch_size(len(dataset), batch_size),
            num_workers=int(num_workers),
            shuffle=False,
        )

    seed = int(predictor.config.get("BaseConfig", {}).get("seed", 42))
    payload = _build_applicability_payload(
        model=predictor.model,
        train_loader=loader(train_records),
        val_loader=loader(val_records),
        train_records=train_records,
        val_records=val_records,
        target_names=predictor.target_names,
        task=predictor.task,
        seed=seed,
    )

    backup_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.pre_applicability{checkpoint_path.suffix}"
    )
    if backup and not backup_path.exists():
        shutil.copy2(checkpoint_path, backup_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["chemflow_applicability"] = payload
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        shutil.copystat(checkpoint_path, temporary)
        os.replace(temporary, checkpoint_path)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("split_manifest")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    output = embed_applicability(
        args.checkpoint,
        args.split_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        backup=not args.no_backup,
    )
    print(f"Updated {output}")


if __name__ == "__main__":
    main()
