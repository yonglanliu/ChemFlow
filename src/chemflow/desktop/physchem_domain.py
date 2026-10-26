"""Morgan-fingerprint similarity and validation-calibrated OOD analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


DEFAULT_OOD_CONFIDENCE = 0.95


def bundled_physchem_reference() -> Path:
    return Path(__file__).with_name("models") / "physchem_reference.csv.gz"


class PhysChemDomainAnalyzer:
    """Compare queries with endpoint-specific training chemistry."""

    def __init__(
        self,
        properties: Sequence[str],
        *,
        reference_path: str | Path | None = None,
        confidence: float = DEFAULT_OOD_CONFIDENCE,
    ) -> None:
        if not 0.5 < float(confidence) < 1.0:
            raise ValueError("OOD confidence must be between 0.5 and 1.0.")
        self.confidence = float(confidence)
        self.reference_path = Path(
            reference_path or bundled_physchem_reference()
        ).resolve()
        if not self.reference_path.is_file():
            raise FileNotFoundError(
                f"Physicochemical domain reference is missing: {self.reference_path}"
            )
        reference = pd.read_csv(self.reference_path)
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=1024
        )
        self.references: dict[str, dict[str, object]] = {}
        for property_name in dict.fromkeys(str(value).upper() for value in properties):
            subset = reference.loc[reference["property"] == property_name]
            train = subset.loc[subset["split"] == "train"].reset_index(drop=True)
            validation = subset.loc[subset["split"] == "validation"]
            if train.empty or validation.empty:
                raise ValueError(
                    f"Domain reference contains no train/validation chemistry for "
                    f"{property_name}."
                )
            fingerprints = [
                self.generator.GetFingerprint(Chem.MolFromSmiles(value))
                for value in train["smiles"].astype(str)
            ]
            calibration = pd.to_numeric(
                validation["max_train_tanimoto"], errors="coerce"
            ).dropna().to_numpy()
            self.references[property_name] = {
                "frame": train,
                "fingerprints": fingerprints,
                "novelty": 1.0 - calibration,
                "similarity_threshold": float(
                    np.quantile(
                        calibration, 1.0 - self.confidence, method="lower"
                    )
                ),
            }

    def analyze(
        self, smiles: Sequence[str], property_name: str
    ) -> pd.DataFrame:
        normalized = str(property_name).upper()
        payload = self.references[normalized]
        references = payload["frame"]
        fingerprints = payload["fingerprints"]
        calibration_novelty = np.asarray(payload["novelty"], dtype=float)
        count = len(smiles)
        similarity = np.full(count, np.nan, dtype=float)
        nearest_id = np.full(count, "", dtype=object)
        nearest_smiles = np.full(count, "", dtype=object)
        ood_score = np.full(count, np.nan, dtype=float)

        for index, value in enumerate(smiles):
            molecule = Chem.MolFromSmiles(str(value).strip())
            if molecule is None:
                continue
            query = self.generator.GetFingerprint(molecule)
            scores = np.asarray(
                DataStructs.BulkTanimotoSimilarity(query, fingerprints), dtype=float
            )
            nearest_index = int(np.argmax(scores))
            similarity[index] = scores[nearest_index]
            nearest_id[index] = references.iloc[nearest_index]["ligand_id"]
            nearest_smiles[index] = references.iloc[nearest_index]["smiles"]
            novelty = 1.0 - similarity[index]
            p_value = (
                1.0 + np.count_nonzero(calibration_novelty >= novelty)
            ) / (calibration_novelty.size + 1.0)
            ood_score[index] = 1.0 - p_value

        assessed = np.isfinite(ood_score)
        ood = assessed & (ood_score >= self.confidence)
        domain = np.full(count, "NOT ASSESSED", dtype=object)
        domain[assessed] = "IN DOMAIN"
        domain[ood] = "OOD"
        return pd.DataFrame(
            {
                "train_max_tanimoto": similarity,
                "nearest_train_id": nearest_id,
                "nearest_train_smiles": nearest_smiles,
                "ood_score": ood_score,
                "ood_flag": ood,
                "domain": domain,
                "similarity_threshold_95": float(
                    payload["similarity_threshold"]
                ),
            }
        )
