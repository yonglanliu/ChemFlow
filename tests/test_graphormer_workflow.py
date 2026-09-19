from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from chemflow.deep_learning.graphormer.modules.dataset import (
    _partition_records,
    _validated_records,
)
from chemflow.deep_learning.graphormer.pretrained import ensure_pretrained_weights
from chemflow.deep_learning.graphormer.trainer import resolve_pretrained_checkpoint


class GraphormerPretrainedTest(unittest.TestCase):
    def test_downloads_and_validates_custom_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            destination = root / "cache" / "weights.pt"
            content = b"test graphormer checkpoint"
            source.write_bytes(content)

            resolved = ensure_pretrained_weights(
                destination,
                url=source.as_uri(),
                sha256=hashlib.sha256(content).hexdigest(),
            )

            self.assertEqual(resolved, destination.resolve())
            self.assertEqual(resolved.read_bytes(), content)

    def test_pretrained_loading_can_be_disabled(self):
        config = SimpleNamespace(use_pretrained=False, pretrained_path="unused.pt")
        self.assertIsNone(resolve_pretrained_checkpoint(config))
        self.assertIsNone(config.pretrained_path)


class GraphormerDatasetWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            smiles_column="SMILES",
            target_column="target",
            split_column=None,
            split_type="random_with_repeated_smiles",
            val_fraction=0.25,
            test_fraction=0.25,
            seed=7,
        )

    def test_validation_reports_bad_rows(self):
        frame = pd.DataFrame(
            {
                "SMILES": ["CC", "not-a-smiles", "CCC"],
                "target": [1.0, 2.0, None],
            }
        )
        records, rejected = _validated_records(
            frame,
            self.config,
            source="training",
            split_column=None,
        )

        self.assertEqual([record["smiles"] for record in records], ["CC"])
        self.assertEqual(
            {row["reason"] for row in rejected},
            {"invalid_smiles", "missing_or_non_numeric_target"},
        )

    def test_repeated_smiles_stay_in_one_split(self):
        smiles = [
            "CC", "CC", "CCC", "CCC", "CCCC", "CCO", "CCN", "CCCl",
            "c1ccccc1", "C1CCCCC1", "CC(=O)O", "COC",
        ]
        frame = pd.DataFrame({"SMILES": smiles, "target": range(len(smiles))})
        records, rejected = _validated_records(
            frame,
            self.config,
            source="training",
            split_column=None,
        )
        self.assertFalse(rejected)

        splits = _partition_records(records, self.config)
        locations: dict[str, set[str]] = {}
        for split_name, split_records in splits.items():
            for record in split_records:
                locations.setdefault(record["smiles"], set()).add(split_name)

        self.assertTrue(all(len(values) == 1 for values in locations.values()))
        self.assertTrue(splits["train"])
        self.assertTrue(splits["val"])
        self.assertTrue(splits["test"])


if __name__ == "__main__":
    unittest.main()
