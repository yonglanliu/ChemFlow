"""Run DiffDock's ``inference`` with the receptor embedded ONCE, not once per ligand.

Why: DiffDock's ``InferenceDataset`` computes an ESM2 embedding for every row of the input CSV. In a
library screen every row has the SAME receptor, so a 10,000-ligand run does ~10,000 identical ESM2
passes (910 batches in one real log) and keeps ~10,000 copies of the embedding in RAM
(10,000 x residues x 1280 floats ~ 19 GB for a 367-residue protein), before docking even starts.

This launcher patches two functions in ``utils.inference_utils`` and then runs ``inference`` as usual:

* ``compute_ESM_embeddings`` -> embeds each UNIQUE sequence once and shares the result across rows;
* ``get_sequences_from_pdbfile`` -> parses each distinct receptor PDB once (memoised).

It is fail-safe: if anything about the patch fails (e.g. a DiffDock version with different internals)
it prints a note and runs the unmodified DiffDock. It is started by AlchemFold as
``<diffdock env python> _diffdock_shim.py <inference args…>`` from the DiffDock repo folder; it does
not import AlchemFold, so it works in the separate diffdock env.
"""

from __future__ import annotations

import functools
import os
import runpy
import sys


def _patch() -> None:
    import utils.inference_utils as iu  # DiffDock's own module (cwd = the DiffDock repo)

    original = iu.compute_ESM_embeddings

    def deduplicated(model, alphabet, labels, sequences):
        first_label: dict[str, str] = {}
        for label, seq in zip(labels, sequences):
            first_label.setdefault(seq, label)
        if len(first_label) == len(sequences):
            return original(model, alphabet, labels, sequences)      # nothing to share
        print(f"[alchemfold] ESM2: {len(sequences)} sequences, {len(first_label)} unique -> "
              f"embedding {len(first_label)} (not {len(sequences)})", flush=True)
        embedded = original(model, alphabet, list(first_label.values()), list(first_label.keys()))
        by_sequence = {seq: embedded[label] for seq, label in first_label.items()}
        return {label: by_sequence[seq] for label, seq in zip(labels, sequences)}   # shared tensors

    iu.compute_ESM_embeddings = deduplicated
    iu.get_sequences_from_pdbfile = functools.lru_cache(maxsize=None)(iu.get_sequences_from_pdbfile)


def main() -> None:
    sys.path.insert(0, os.getcwd())          # DiffDock modules (utils, datasets, inference) live in the repo
    try:
        _patch()
    except Exception as exc:                  # fail-safe: stock DiffDock
        print(f"[alchemfold] ESM de-duplication not applied ({type(exc).__name__}: {exc}); "
              "running stock DiffDock", flush=True)
    runpy.run_module("inference", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
