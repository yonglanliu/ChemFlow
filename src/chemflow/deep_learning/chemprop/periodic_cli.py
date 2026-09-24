"""Run Chemprop's official CLI with retained periodic Lightning checkpoints."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    # Imports stay inside main so ChemFlow can be imported without the optional
    # Chemprop/Lightning stack installed.
    from chemflow.deep_learning.chemprop import losses as _losses  # noqa: F401
    from lightning.pytorch.callbacks import ModelCheckpoint
    from chemprop.cli import train as train_module

    interval = int(os.environ.get("CHEMFLOW_CHECKPOINT_EVERY_N_EPOCHS", "0"))
    if interval < 1:
        raise ValueError(
            "CHEMFLOW_CHECKPOINT_EVERY_N_EPOCHS must be positive in the "
            "periodic Chemprop launcher."
        )

    class PeriodicModelCheckpoint(ModelCheckpoint):
        """Keep Chemprop's best/last policy and add full-state snapshots."""

        def on_validation_end(self, trainer, pl_module) -> None:
            super().on_validation_end(trainer, pl_module)
            if trainer.sanity_checking:
                return
            completed_epoch = int(trainer.current_epoch) + 1
            if completed_epoch % interval != 0:
                return
            periodic_path = (
                Path(self.dirpath) / f"epoch_{completed_epoch:04d}.ckpt"
            )
            # Lightning coordinates this call across distributed ranks and
            # writes only from the appropriate global process.
            trainer.save_checkpoint(periodic_path)

    train_module.ModelCheckpoint = PeriodicModelCheckpoint

    from chemprop.cli.main import main as chemprop_main

    chemprop_main()


if __name__ == "__main__":
    main()
