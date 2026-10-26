"""Build the compact similarity/OOD reference shipped with the desktop.

Run from the repository root after the physicochemical training splits change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_name("models") / "physchem_reference.csv.gz"
SOURCES = {
    "LOGD": (
        ROOT / "dataset/CHEMBL/logd/splits/train.csv",
        ROOT / "dataset/CHEMBL/logd/splits/val.csv",
        "LogD_pooled",
    ),
    "SOLUBILITY": (
        ROOT / "results/sol/train.csv",
        ROOT / "results/sol/val.csv",
        "LogS_K_pooled",
    ),
    "THERMODYNAMIC_SOLUBILITY": (
        ROOT / "results/sol/train.csv",
        ROOT / "results/sol/val.csv",
        "LogS_T_ChEMBL",
    ),
}


def _load(path: Path, target: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["Ligand ID", "SMILES", target])
    frame = frame.dropna(subset=["SMILES", target]).drop_duplicates("SMILES")
    valid = frame["SMILES"].map(lambda value: Chem.MolFromSmiles(str(value)) is not None)
    return frame.loc[valid, ["Ligand ID", "SMILES"]].reset_index(drop=True)


def build(output: Path = OUTPUT) -> Path:
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    rows: list[pd.DataFrame] = []
    for property_name, (train_path, validation_path, target) in SOURCES.items():
        train = _load(train_path, target)
        validation = _load(validation_path, target)
        train_fingerprints = [
            generator.GetFingerprint(Chem.MolFromSmiles(value))
            for value in train["SMILES"]
        ]
        validation_similarity = []
        for value in validation["SMILES"]:
            fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(value))
            validation_similarity.append(
                max(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fingerprints))
            )
        train = train.rename(
            columns={"Ligand ID": "ligand_id", "SMILES": "smiles"}
        )
        validation = validation.rename(
            columns={"Ligand ID": "ligand_id", "SMILES": "smiles"}
        )
        train.insert(0, "property", property_name)
        train.insert(1, "split", "train")
        train["max_train_tanimoto"] = np.nan
        validation.insert(0, "property", property_name)
        validation.insert(1, "split", "validation")
        validation["max_train_tanimoto"] = validation_similarity
        rows.extend((train, validation))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(
        output, index=False, compression="gzip", float_format="%.7g"
    )
    return output


if __name__ == "__main__":
    print(build())
