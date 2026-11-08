"""Configuration + batching for single-trajectory MM-GBSA (no OpenMM import).

MM-GBSA estimates a binding free energy from a single MD trajectory of the
complex (from ``alchemforge openmm modeling md``):

    dG_bind ~= <E_complex> - <E_receptor> - <E_ligand>

evaluated per frame in an implicit (Generalized Born + surface-area) solvent,
with explicit water and ions stripped first. It is an *endpoint* estimate — fast
and embarrassingly parallel, meant to triage many candidates before the far more
expensive RBFE. The heavy per-frame energy evaluation lives in
:mod:`alchemforge.mmgbsa_run` (that one needs OpenMM + mdtraj); everything here
(validate / stage / batch / aggregate) is pure Python so it is testable and runs
on a login node.

Units: energies kcal/mol, concentration molar, temperature K.
"""

from __future__ import annotations

import csv
import math
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from chemflow.fe.openmm_modeling.membrane_relax_config import (
    DEFAULT_FORCEFIELDS,
    _deep_merge,
    _load_yaml,
    _resolve,
    _to_namespace,
)

SUPPORTED_GB_MODELS = {"HCT", "OBC1", "OBC2", "GBn", "GBn2"}

DEFAULTS: dict[str, Any] = {
    "trajectory": None,                 # DCD/XTC trajectory of the complex
    "topology": None,                   # PDB/PRMTOP matching the trajectory
    "ligand_selection": None,           # mdtraj selection for the ligand atoms
    # receptor = all solute atoms that are not the ligand (after stripping solvent)
    "strip_selection": "water or resname NA CL K MG ZN CA SOD CLA POT",
    "forcefields": list(DEFAULT_FORCEFIELDS),
    "ligand_sdf": None,                 # if the ligand needs openff/gaff parameters
    "small_molecule_forcefield": "gaff-2.11",
    "gb_model": "OBC2",                 # HCT | OBC1 | OBC2 | GBn | GBn2
    "salt_concentration": 0.15,         # molar (GB Debye screening)
    "solute_dielectric": 1.0,
    "solvent_dielectric": 78.5,
    "surface_area": True,               # add the non-polar SASA term
    "surface_tension": 0.0072,          # kcal/mol/A^2 (standard MM-GBSA value)
    "temperature": 300.0,               # K
    "platform": "CPU",                  # CPU is plenty for single-point GB energies
    "start_frame": 0,
    "stride": 1,                        # use every Nth frame
    "max_frames": 200,                  # cap frames for speed
    "output_prefix": "mmgbsa",
    "workdir": ".",
}

# Columns written per run and aggregated across candidates.
SUMMARY_FIELDS = [
    "candidate", "dG_mean", "dG_sem", "n_frames",
    "complex_mean", "receptor_mean", "ligand_mean",
]


# ---------------------------------------------------------------------------
# Single-run config
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> SimpleNamespace:
    """Load an MM-GBSA YAML merged over the defaults; resolve input paths."""
    config_path = Path(path).expanduser().resolve()
    conf = _to_namespace(_deep_merge(DEFAULTS, _load_yaml(config_path)))

    base = config_path.parent
    conf.trajectory = _resolve(base, getattr(conf, "trajectory", None))
    conf.topology = _resolve(base, getattr(conf, "topology", None))
    if getattr(conf, "ligand_sdf", None):
        conf.ligand_sdf = _resolve(base, conf.ligand_sdf)
    workdir = getattr(conf, "workdir", ".")
    conf.workdir = str(base) if (not workdir or str(workdir) == ".") else _resolve(base, workdir)
    return conf


def validate_config(conf: SimpleNamespace) -> None:
    """Validate a loaded MM-GBSA configuration (no OpenMM needed)."""
    traj = getattr(conf, "trajectory", None)
    if not traj or not Path(traj).is_file():
        raise ValueError(f"trajectory must be an existing file: {traj}")
    top = getattr(conf, "topology", None)
    if not top or not Path(top).is_file():
        raise ValueError(f"topology must be an existing file: {top}")

    if not str(getattr(conf, "ligand_selection", "") or "").strip():
        raise ValueError("ligand_selection is required (an mdtraj atom selection for the ligand)")

    forcefields = getattr(conf, "forcefields", None)
    if not forcefields or any(not str(f).strip() for f in forcefields):
        raise ValueError("MM-GBSA config requires at least one force field")

    gb = str(getattr(conf, "gb_model", "OBC2"))
    if gb not in SUPPORTED_GB_MODELS:
        raise ValueError(f"gb_model must be one of {sorted(SUPPORTED_GB_MODELS)} (got {gb!r})")

    ligand_sdf = getattr(conf, "ligand_sdf", None)
    if ligand_sdf:
        p = Path(ligand_sdf)
        if not p.is_file() or p.suffix.lower() != ".sdf":
            raise ValueError(f"ligand_sdf must be an existing .sdf file: {ligand_sdf}")

    if float(getattr(conf, "temperature", 0.0)) <= 0.0:
        raise ValueError("temperature must be > 0 K")
    if float(getattr(conf, "salt_concentration", -1.0)) < 0.0:
        raise ValueError("salt_concentration must be >= 0 M")
    if int(getattr(conf, "stride", 1)) < 1:
        raise ValueError("stride must be >= 1")
    if int(getattr(conf, "start_frame", 0)) < 0:
        raise ValueError("start_frame must be >= 0")
    if int(getattr(conf, "max_frames", 1)) < 1:
        raise ValueError("max_frames must be >= 1")


def stage(config_path: str | Path, task_directory: str | Path) -> tuple[Path, Path]:
    """Write a runner script that computes MM-GBSA for one ``config_path``."""
    config_path = Path(config_path).expanduser().resolve()
    task = Path(task_directory).expanduser().resolve()
    task.mkdir(parents=True, exist_ok=True)
    runner = task / "run_mmgbsa.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'trap \'echo "[FAILED] line $LINENO: $BASH_COMMAND (exit $?)" >&2\' ERR\n'
        'CONFIG="${1:?Usage: $0 mmgbsa.yaml}"\n'
        f"exec {shlex.quote(sys.executable)} -m alchemforge.mmgbsa_run "
        '--config "$CONFIG"\n',
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, config_path


# ---------------------------------------------------------------------------
# Batch over many candidates (the "at scale" part)
# ---------------------------------------------------------------------------

def discover_runs(parent: str | Path, *, trajectory_glob: str = "*production.dcd") -> list[dict[str, str]]:
    """Find MD runs under ``parent``: each subdirectory with a trajectory + a PDB.

    Returns ``[{"name", "trajectory", "topology"}, ...]`` sorted by name. The
    topology is the first ``*.pdb`` that sits next to the trajectory (prefers one
    sharing the trajectory's stem, e.g. ``md_production.pdb``).
    """
    parent = Path(parent).expanduser().resolve()
    runs: list[dict[str, str]] = []
    for traj in sorted(parent.glob(f"*/{trajectory_glob}")):
        folder = traj.parent
        stem = traj.stem
        preferred = folder / f"{stem}.pdb"
        if preferred.is_file():
            topology = preferred
        else:
            pdbs = sorted(folder.glob("*.pdb"))
            if not pdbs:
                continue
            topology = pdbs[0]
        runs.append({"name": folder.name, "trajectory": str(traj), "topology": str(topology)})
    return runs


def write_batch(
    runs: list[dict[str, str]],
    out_dir: str | Path,
    *,
    base_config: dict[str, Any] | None = None,
    partition: str = "norm",
    cpus_per_task: int = 4,
    memory: str = "8G",
    walltime: str = "08:00:00",
    modules: str = "",
    conda_init: str = "",
    conda_environment: str = "",
) -> dict[str, Path]:
    """Write one MM-GBSA YAML per run plus a plain SLURM array over them all.

    No per-array concurrency cap (``--array=0-N``), matching the project's SLURM
    convention. ``modules`` (space/comma-separated), ``conda_init`` (path to
    conda.sh) and ``conda_environment`` make the array job self-contained on a
    fresh HPC node. Returns the key paths written.
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = {**(base_config or {})}

    config_paths: list[Path] = []
    for run in runs:
        name = run["name"]
        conf = {
            **base,
            "trajectory": run["trajectory"],
            "topology": run["topology"],
            "output_prefix": "mmgbsa",
            "workdir": str(out / name),
        }
        (out / name).mkdir(parents=True, exist_ok=True)
        cfg_path = out / name / "mmgbsa.yaml"
        cfg_path.write_text(_yaml_dump(conf), encoding="utf-8")
        config_paths.append(cfg_path)

    manifest = out / "configs.txt"
    manifest.write_text("".join(f"{p}\n" for p in config_paths), encoding="utf-8")

    n = len(config_paths)
    array = out / "mmgbsa_array.slurm"
    activation = ""
    for mod in str(modules).replace(",", " ").split():
        activation += f"module load {mod}\n"
    if conda_init:
        activation += f"source {conda_init}\n"
    if conda_environment:
        activation += f"conda activate {conda_environment}\n"
    array.write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH --job-name=mmgbsa\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --cpus-per-task={cpus_per_task}\n"
        f"#SBATCH --mem={memory}\n"
        f"#SBATCH --time={walltime}\n"
        f"#SBATCH --array=0-{max(n - 1, 0)}\n"
        f"#SBATCH --output={out}/logs/%A_%a.out\n"
        "set -euo pipefail\n"
        f"{activation}"
        f'mkdir -p "{out}/logs"\n'
        f'CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{manifest}")\n'
        'echo "Task $SLURM_ARRAY_TASK_ID -> $CONFIG"\n'
        f"{shlex.quote(sys.executable)} -m alchemforge.mmgbsa_run --config \"$CONFIG\"\n",
        encoding="utf-8",
    )
    array.chmod(array.stat().st_mode | 0o100)
    return {"array": array, "manifest": manifest, "out_dir": out}


# ---------------------------------------------------------------------------
# Aggregation / ranking (reads each run's summary CSV)
# ---------------------------------------------------------------------------

def _read_summary(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    return rows[0] if rows else None


def aggregate(batch_dir: str | Path, out_csv: str | Path | None = None) -> list[dict[str, Any]]:
    """Collect every ``*/mmgbsa_summary.csv`` under ``batch_dir`` and rank by dG.

    Returns the rows sorted by ``dG_mean`` ascending (most favourable first) and,
    unless ``out_csv`` is None, writes them to ``batch_dir/mmgbsa_ranking.csv``.
    """
    batch_dir = Path(batch_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for summary in sorted(batch_dir.glob("*/mmgbsa_summary.csv")):
        data = _read_summary(summary)
        if not data:
            continue
        data.setdefault("candidate", summary.parent.name)
        rows.append(data)

    def _key(row):
        try:
            return float(row.get("dG_mean", "nan"))
        except (TypeError, ValueError):
            return math.inf
    rows.sort(key=_key)

    if out_csv is not False and rows:
        target = Path(out_csv).expanduser() if out_csv else batch_dir / "mmgbsa_ranking.csv"
        fields = list(dict.fromkeys(SUMMARY_FIELDS + [k for r in rows for k in r]))
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows


# ---------------------------------------------------------------------------
# Tiny YAML emitter (avoid a PyYAML dependency for writing simple configs)
# ---------------------------------------------------------------------------

def _yaml_dump(conf: dict[str, Any]) -> str:
    lines = []
    for key, value in conf.items():
        if value is None:
            lines.append(f"{key}: null")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key}: {value}")
        elif isinstance(value, (list, tuple)):
            lines.append(f"{key}:")
            lines.extend(f'  - "{item}"' for item in value)
        else:
            lines.append(f'{key}: "{value}"')
    return "\n".join(lines) + "\n"


def default_yaml_text(trajectory: str = "md_production.dcd",
                      topology: str = "md_production.pdb",
                      ligand_selection: str = "resname LIG") -> str:
    """Commented example YAML for a single MM-GBSA run."""
    forcefields = "\n".join(f'  - "{item}"' for item in DEFAULT_FORCEFIELDS)
    return (
        "# ===================================================================\n"
        "# Single-trajectory MM-GBSA endpoint (binding free-energy estimate).\n"
        "# Strips water/ions, then per frame: dG = E_complex - E_receptor -\n"
        "# E_ligand in an implicit GB+SA solvent. Fast triage before RBFE.\n"
        "# ===================================================================\n"
        "\n"
        f'trajectory: "{trajectory}"        # complex MD trajectory (DCD/XTC)\n'
        f'topology: "{topology}"          # matching topology (PDB)\n'
        f'ligand_selection: "{ligand_selection}"   # mdtraj selection of the ligand atoms\n'
        'strip_selection: "water or resname NA CL K MG ZN CA SOD CLA POT"\n'
        "\n"
        '# ligand_sdf: "ligand.sdf"               # if the ligand needs GAFF/OpenFF params\n'
        '# small_molecule_forcefield: "gaff-2.11"\n'
        "forcefields:\n"
        f"{forcefields}\n"
        "\n"
        'gb_model: "OBC2"                 # HCT | OBC1 | OBC2 | GBn | GBn2\n'
        "salt_concentration: 0.15         # molar\n"
        "solute_dielectric: 1.0\n"
        "solvent_dielectric: 78.5\n"
        "surface_area: true               # add non-polar SASA term\n"
        "surface_tension: 0.0072          # kcal/mol/A^2\n"
        "temperature: 300.0               # K\n"
        'platform: "CPU"                  # CPU | CUDA | OpenCL\n'
        "start_frame: 0\n"
        "stride: 1                        # use every Nth frame\n"
        "max_frames: 200                  # cap for speed\n"
        'output_prefix: "mmgbsa"\n'
    )
