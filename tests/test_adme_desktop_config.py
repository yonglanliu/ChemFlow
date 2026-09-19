from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chemflow.desktop.adme_config import (
    find_default_adme_config,
    load_adme_deployment_config,
)


class ADMEDesktopConfigTest(unittest.TestCase):
    def test_environment_config_is_discovered_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "automatic.toml"
            path.write_text("[checkpoints]\n", encoding="utf-8")
            with patch.dict(os.environ, {"CHEMFLOW_ADMET_CONFIG": str(path)}):
                self.assertEqual(find_default_adme_config(), path.resolve())

    def test_checkpoint_paths_are_relative_to_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "deployment" / "config.toml"
            config_path.parent.mkdir()
            config_path.write_text(
                """
[checkpoints]
multitask = "../models/multitask/best.ckpt"
mlm = "${MODEL_ROOT}/mlm/best.ckpt"

[inference]
device = "cpu"
embedding_dimensions = 64
calibration_confidence = 0.95

[quality_control]
min_tanimoto_similarity = 0.42
max_embedding_distance = 0.28
max_log_interval_width = 0.75
ood_score_threshold = 0.97
""".strip(),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"MODEL_ROOT": str(root / "models")}):
                config = load_adme_deployment_config(config_path)

        self.assertEqual(
            config.multitask_checkpoint,
            str((root / "models" / "multitask" / "best.ckpt").resolve()),
        )
        self.assertEqual(
            config.mlm_checkpoint,
            str((root / "models" / "mlm" / "best.ckpt").resolve()),
        )
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.embedding_dimensions, 64)
        self.assertEqual(config.calibration_confidence, 0.95)
        self.assertEqual(config.min_tanimoto_similarity, 0.42)
        self.assertEqual(config.max_embedding_distance, 0.28)
        self.assertEqual(config.max_log_interval_width, 0.75)
        self.assertEqual(config.ood_score_threshold, 0.97)


if __name__ == "__main__":
    unittest.main()
