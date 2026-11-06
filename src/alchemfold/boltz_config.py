"""Configuration layer for Boltz-2 co-folding (no Boltz import).

Boltz-2 jointly predicts the 3D structure of one or more protein chains together
with ligand(s) (and optionally their binding affinity). This module builds and
validates a config that the runner (:mod:`alchemfold.boltz_run`)
translates into a Boltz input YAML and a ``boltz predict`` command.

The config/validation/staging layer is import-light so the CLI, desktop, and
tests work without Boltz installed; only ``run`` needs the ``boltz`` package.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_namespace(value: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace (lists handled elementwise)."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Co-folding config must be a YAML mapping: {path}")
    return data


def _resolve(base_dir: Path, value: Any) -> Any:
    if not value:
        return value
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return str(candidate)


SUPPORTED_OUTPUT_FORMATS = {"pdb", "mmcif"}
SUPPORTED_ACCELERATORS = {"gpu", "cpu", "tpu"}

DEFAULTS: dict[str, Any] = {
    "engine": "boltz",            # reserved for future engines
    "boltz_command": "boltz",     # the Boltz-2 CLI entry point
    "workdir": ".",
    "out_dir": "cofold_output",
    "output_format": "pdb",       # pdb | mmcif
    "use_msa_server": True,       # let Boltz compute MSAs via its server (needs internet)
    "diffusion_samples": 1,       # number of predicted poses
    "recycling_steps": 3,
    "sampling_steps": 200,
    "seed": 1,
    "accelerator": "gpu",         # gpu | cpu | tpu
    "devices": 1,
    "override": False,            # overwrite an existing Boltz output dir
    # Binding-affinity prediction (Boltz-2). Set a ligand id as the binder.
    "predict_affinity": False,
    "affinity_binder": None,
    # Inputs:
    #   proteins: list of {id, sequence, [msa]} chains
    #   ligands:  list of {id, smiles} or {id, ccd}
    "proteins": [],
    "ligands": [],
}


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a Boltz co-folding YAML, merged over the defaults; resolve paths."""
    config_path = Path(path).expanduser().resolve()
    conf = _to_namespace(_deep_merge(DEFAULTS, _load_yaml(config_path)))

    base_dir = config_path.parent
    workdir = getattr(conf, "workdir", ".")
    if not workdir or str(workdir) == ".":
        conf.workdir = str(base_dir)
    else:
        conf.workdir = _resolve(base_dir, workdir)

    # Resolve any per-chain MSA file paths relative to the config directory.
    for protein in getattr(conf, "proteins", []) or []:
        msa = getattr(protein, "msa", None)
        if msa and str(msa) not in {"empty", "server"}:
            protein.msa = _resolve(base_dir, msa)
    return conf


def validate_config(conf: SimpleNamespace) -> None:
    """Validate a loaded Boltz co-folding configuration (no Boltz needed)."""
    proteins = getattr(conf, "proteins", []) or []
    ligands = getattr(conf, "ligands", []) or []

    if not proteins:
        raise ValueError("Boltz co-folding requires at least one protein chain")

    ids: set[str] = set()
    for protein in proteins:
        pid = str(getattr(protein, "id", "")).strip()
        seq = str(getattr(protein, "sequence", "")).strip()
        if not pid:
            raise ValueError("each protein needs an 'id'")
        if not seq:
            raise ValueError(f"protein '{pid}' needs a 'sequence'")
        if pid in ids:
            raise ValueError(f"duplicate chain id: {pid}")
        ids.add(pid)

    for ligand in ligands:
        lid = str(getattr(ligand, "id", "")).strip()
        if not lid:
            raise ValueError("each ligand needs an 'id'")
        if lid in ids:
            raise ValueError(f"duplicate chain id: {lid}")
        ids.add(lid)
        smiles = getattr(ligand, "smiles", None)
        ccd = getattr(ligand, "ccd", None)
        if not smiles and not ccd:
            raise ValueError(f"ligand '{lid}' needs a 'smiles' or a 'ccd' code")

    # Co-folding implies either a ligand or a multi-chain assembly.
    if not ligands and len(proteins) < 2:
        raise ValueError(
            "co-folding needs a ligand or >= 2 protein chains; a single apo chain "
            "is plain structure prediction, not co-folding"
        )

    if str(getattr(conf, "output_format", "pdb")) not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"output_format must be one of {sorted(SUPPORTED_OUTPUT_FORMATS)}"
        )
    if str(getattr(conf, "accelerator", "gpu")) not in SUPPORTED_ACCELERATORS:
        raise ValueError(
            f"accelerator must be one of {sorted(SUPPORTED_ACCELERATORS)}"
        )
    for key in ("diffusion_samples", "recycling_steps", "sampling_steps", "devices"):
        if int(getattr(conf, key, 0)) < 1:
            raise ValueError(f"{key} must be >= 1")

    if getattr(conf, "predict_affinity", False):
        binder = getattr(conf, "affinity_binder", None)
        if not binder:
            raise ValueError(
                "predict_affinity is true but 'affinity_binder' (a ligand id) is unset"
            )
        ligand_ids = {str(getattr(l, "id", "")).strip() for l in ligands}
        if str(binder) not in ligand_ids:
            raise ValueError(
                f"affinity_binder '{binder}' is not one of the ligand ids {sorted(ligand_ids)}"
            )


def stage(config_path: str | Path, task_directory: str | Path) -> tuple[Path, Path]:
    """Write a runner script that runs Boltz co-folding for ``config_path``."""
    config_path = Path(config_path).expanduser().resolve()
    task = Path(task_directory).expanduser().resolve()
    task.mkdir(parents=True, exist_ok=True)
    runner = task / "run_cofold_boltz.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'trap \'echo "[FAILED] line $LINENO: $BASH_COMMAND (exit $?)" >&2\' ERR\n'
        'CONFIG="${1:?Usage: $0 cofold_boltz.yaml}"\n'
        f"exec {shlex.quote(sys.executable)} -m alchemfold.boltz_run "
        '--config "$CONFIG"\n',
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, config_path


def default_yaml_text(
    *,
    protein_id: str = "A",
    ligand_id: str = "L",
    sequence: str = "PUT_YOUR_PROTEIN_SEQUENCE_HERE",
    smiles: str = "CCO",
    predict_affinity: bool = True,
) -> str:
    """Return a commented example Boltz-2 co-folding config."""
    affinity_block = (
        "predict_affinity: true          # Boltz-2 binding-affinity prediction\n"
        f'affinity_binder: "{ligand_id}"           # ligand id to score\n'
        if predict_affinity
        else (
            "predict_affinity: false\n"
            '# affinity_binder: "L"          # ligand id to score (Boltz-2)\n'
        )
    )
    return (
        "# ===================================================================\n"
        "# Boltz-2 co-folding: jointly predict a protein+ligand complex (and,\n"
        "# optionally, its binding affinity). Output is a complex PDB/mmCIF that\n"
        "# feeds OpenMM modeling (build-soluble / build-membrane) -> FEP.\n"
        "# Requires the `boltz` package (Boltz-2) installed with GPU.\n"
        "# ===================================================================\n"
        "\n"
        'boltz_command: "boltz"          # Boltz-2 CLI entry point\n'
        'out_dir: "cofold_output"\n'
        'output_format: "pdb"            # pdb | mmcif\n'
        "use_msa_server: true            # compute MSAs via the Boltz server (needs internet)\n"
        "                                # set false and give each protein an 'msa' path for offline runs\n"
        "diffusion_samples: 1            # number of predicted poses\n"
        "recycling_steps: 3\n"
        "sampling_steps: 200\n"
        "seed: 1\n"
        'accelerator: "gpu"              # gpu | cpu | tpu\n'
        "devices: 1\n"
        "override: false                 # overwrite an existing Boltz output dir\n"
        f"{affinity_block}"
        "\n"
        "proteins:\n"
        f'  - id: "{protein_id}"\n'
        f'    sequence: "{sequence}"\n'
        '    # msa: "empty"                # or a precomputed .a3m path when use_msa_server is false\n'
        "\n"
        "ligands:\n"
        f'  - id: "{ligand_id}"\n'
        f'    smiles: "{smiles}"          # or use `ccd: "ATP"` for a PDB chemical-component code\n'
    )
