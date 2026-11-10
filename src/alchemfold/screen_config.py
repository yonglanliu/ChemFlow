"""Configuration for AlchemFold virtual screening (sharded, HPC, ranked).

Screening runs the same docking engines (Boltz-2 / Chai-1 / DiffDock) over a
large ligand library: combine one or more library files, split them into shards,
dock each shard as an independent SLURM array task (across GPUs/CPUs/nodes), then
rank all results and keep the top hits.

This config is a superset of the single-run docking config plus:
  * ``ligand_files``   — one or more library files (CSV / .smi / .cxsmiles,
                          plain or .bz2/.gz). ``ligand_file`` (singular) also works.
  * sharding           — ``n_shards`` or ``shard_size``.
  * ``[slurm]``        — resources for the docking array + ranking job.
  * ``[ranking]``      — metric, top_k, order, output.

Import-light (no engine/model needed): used for generate/stage/rank wiring.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from alchemfold.boltz_config import _deep_merge, _load_yaml, _resolve, _to_namespace
from alchemfold.dock_config import DEFAULTS as _DOCK_DEFAULTS

# Source ligand catalogs the screen can target (DB build is handled elsewhere).
SUPPORTED_LIBRARIES = ("zinc22", "enamine_real")

# Screening defaults = docking defaults + screening-specific sections.
DEFAULTS: dict[str, Any] = copy.deepcopy(_DOCK_DEFAULTS)
DEFAULTS.update(
    {
        "screen_out": "screen_output",
        # Source library catalog for the screen. Supported: "zinc22" | "enamine_real".
        "library": "zinc22",
        "ligand_files": [],        # list of library files; ligand_file also accepted
        # Sharding: set exactly one. n_shards splits into that many subsets;
        # shard_size caps ligands per shard. If neither is set, one shard.
        "n_shards": 0,
        "shard_size": 0,
        "slurm": {
            "partition": "gpu",
            "account": "",
            "nodes": 1,
            "gpus_per_task": 1,     # 0 => CPU-only tasks
            "gpu_type": "",         # e.g. a100 -> --gres=gpu:a100:N
            "cpus_per_task": 8,
            "mem": "32G",
            "time": "24:00:00",
            "conda_env": "alchemfold",
        },
        "ranking": {
            "metric": "auto",       # auto | confidence | affinity | ptm | iptm | plddt
            "top_k": 100,
            "order": "auto",        # auto | asc | desc (auto picks per metric)
            "output": "ranking.csv",
        },
    }
)


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a screening YAML, merged over the defaults; resolve library paths.

    ``workdir: "."`` (default) means the current working directory, so shards,
    the SLURM array/ranking jobs and results are written where the command is
    run rather than in the config/code directory. Library/input files still
    resolve relative to the config file.
    """
    import os

    config_path = Path(path).expanduser().resolve()
    conf = _to_namespace(_deep_merge(DEFAULTS, _load_yaml(config_path)))

    base_dir = config_path.parent
    workdir = getattr(conf, "workdir", ".")
    conf.workdir = os.getcwd() if (not workdir or str(workdir) == ".") else _resolve(base_dir, workdir)

    for key in ("receptor_pdb", "diffdock_model_dir", "diffdock_config"):
        value = getattr(conf, key, None)
        if value:
            setattr(conf, key, _resolve(base_dir, value))

    # Normalize library inputs to an absolute list.
    files: list[str] = []
    single = getattr(conf, "ligand_file", None)
    if single:
        files.append(_resolve(base_dir, single))
    for item in getattr(conf, "ligand_files", []) or []:
        files.append(_resolve(base_dir, item))
    conf.ligand_files = files
    conf.ligand_file = None

    for protein in getattr(conf, "proteins", []) or []:
        msa = getattr(protein, "msa", None)
        if msa and str(msa) not in {"empty", "server"}:
            protein.msa = _resolve(base_dir, msa)
    return conf


def validate_config(conf: SimpleNamespace) -> None:
    """Validate a screening configuration (no engine/model needed)."""
    from alchemfold.dock_config import SUPPORTED_ENGINES

    engine = str(getattr(conf, "engine", "boltz"))
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(f"engine must be one of {sorted(SUPPORTED_ENGINES)}")

    library = str(getattr(conf, "library", "zinc22"))
    if library not in SUPPORTED_LIBRARIES:
        raise ValueError(f"library must be one of {sorted(SUPPORTED_LIBRARIES)}")

    if not (getattr(conf, "proteins", []) or []):
        raise ValueError("screening requires at least one receptor (protein) chain")

    files = getattr(conf, "ligand_files", []) or []
    if not files:
        raise ValueError("screening requires 'ligand_files' (or 'ligand_file')")
    for path in files:
        if not Path(path).is_file():
            raise ValueError(f"ligand library file not found: {path}")

    if int(getattr(conf, "n_shards", 0)) < 0 or int(getattr(conf, "shard_size", 0)) < 0:
        raise ValueError("n_shards and shard_size must be >= 0")

    ranking = getattr(conf, "ranking", None)
    if ranking is None or int(getattr(ranking, "top_k", 0)) < 1:
        raise ValueError("ranking.top_k must be >= 1")
    if str(getattr(ranking, "order", "auto")) not in {"auto", "asc", "desc"}:
        raise ValueError("ranking.order must be auto | asc | desc")

    slurm = getattr(conf, "slurm", None)
    if slurm is None:
        raise ValueError("screening requires a [slurm] section")
    if int(getattr(slurm, "gpus_per_task", 1)) < 0 or int(getattr(slurm, "cpus_per_task", 1)) < 1:
        raise ValueError("slurm.gpus_per_task >= 0 and slurm.cpus_per_task >= 1")


def write_stage_runner(config_path: str | Path, task_directory: str | Path) -> tuple[Path, Path]:
    """Write a runner that stages the screen (shards + SLURM array + ranking)."""
    import shlex
    import sys

    config_path = Path(config_path).expanduser().resolve()
    task = Path(task_directory).expanduser().resolve()
    task.mkdir(parents=True, exist_ok=True)
    runner = task / "run_screen_stage.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'CONFIG="${1:?Usage: $0 screen.yaml}"\n'
        f"exec {shlex.quote(sys.executable)} -m alchemfold.screen --config \"$CONFIG\"\n",
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, config_path


def default_yaml_text(
    *,
    engine: str = "boltz",
    library: str = "zinc22",
    protein_id: str = "A",
    sequence: str = "PUT_YOUR_PROTEIN_SEQUENCE_HERE",
    proteins: list | None = None,
) -> str:
    """Return a commented example screening config."""
    from alchemfold.dock_config import _format_proteins_block
    proteins_block = _format_proteins_block(proteins or [(protein_id, sequence)])
    return (
        "# ===================================================================\n"
        "# AlchemFold virtual screening: dock a ligand library in parallel on HPC,\n"
        "# then rank and keep the top hits.\n"
        "#   combine ligand_files -> split into shards -> SLURM array (1 task/shard)\n"
        "#   -> each task docks its shard with the chosen engine -> rank -> top_k.\n"
        "# ===================================================================\n"
        "\n"
        f'engine: "{engine}"              # boltz | chai | diffdock\n'
        f'library: "{library}"           # source catalog: zinc22 | enamine_real\n'
        'screen_out: "screen_output"\n'
        "use_msa_server: true\n"
        'accelerator: "gpu"\n'
        "diffusion_samples: 1\n"
        "seed: 1\n"
        "\n"
        + proteins_block
        + "# receptor_pdb: receptor.pdb     # for engine: diffdock\n"
        "\n"
        "ligand_files:                   # one or more libraries (CSV/.smi/.cxsmiles, .bz2/.gz)\n"
        '  - "library_part1.cxsmiles.bz2"\n'
        '  - "library_part2.cxsmiles.bz2"\n'
        "\n"
        "# Split the combined library. Set n_shards OR shard_size (not both).\n"
        "n_shards: 0                     # e.g. 100 array tasks\n"
        "shard_size: 1000                # ligands per shard (used when n_shards: 0)\n"
        "\n"
        "slurm:\n"
        '  partition: "gpu"\n'
        '  account: ""\n'
        "  nodes: 1\n"
        "  gpus_per_task: 1              # 0 => CPU-only tasks\n"
        '  gpu_type: ""                  # e.g. a100 -> --gres=gpu:a100:1\n'
        "  cpus_per_task: 8\n"
        '  mem: "32G"\n'
        '  time: "24:00:00"\n'
        '  conda_env: "alchemfold"\n'
        "\n"
        "ranking:\n"
        '  metric: "auto"                # auto | confidence | affinity | ptm | iptm | plddt\n'
        "  top_k: 100\n"
        '  order: "auto"                 # auto | asc | desc\n'
        '  output: "ranking.csv"\n'
    )
