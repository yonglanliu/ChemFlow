"""Configuration layer for the OpenMM MD runner (no OpenMM import).

Plain, unrestrained molecular dynamics for a prepared/relaxed system, soluble or
membrane: energy minimization -> NVT equilibration -> NPT equilibration ->
production. The heavy runner lives in :mod:`chemflow.fe.openmm_modeling.md_run`.

Units: timestep / simulation_time in ps, temperature in K, pressure in bar,
surface_tension in bar*nm (membrane only).
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from chemflow.fe.openmm_modeling.membrane_relax_config import (
    DEFAULT_FORCEFIELDS,
    SUPPORTED_PLATFORMS,
    _deep_merge,
    _load_yaml,
    _resolve,
    _to_namespace,
)

SUPPORTED_SYSTEM_TYPES = {"soluble", "membrane"}

DEFAULTS: dict[str, Any] = {
    "system_type": "soluble",   # soluble (MonteCarloBarostat) | membrane (NPgT)
    "platform": "CUDA",
    "friction": 1.0,            # Langevin collision rate (1/ps)
    "seed": 1,
    "forcefields": list(DEFAULT_FORCEFIELDS),
    "temperature": 300.0,       # K
    "pressure": 1.0,            # bar
    "surface_tension": 0.0,     # bar*nm (membrane only; 0 => tensionless)
    "report_interval": 5000,    # steps between trajectory/log frames
    "checkpoint_interval": 50000,
    "output_prefix": "md",
    "workdir": ".",
    # Optional: "ligand_sdf" + "small_molecule_forcefield".
    "minimization": {"steps": 10000},
    "nvt": {"timestep": 0.002, "simulation_time": 250.0},    # equilibration
    "npt": {"timestep": 0.002, "simulation_time": 500.0},    # equilibration
    "production": {"timestep": 0.004, "simulation_time": 10000.0},
}

_STAGE_KEYS = ("minimization", "nvt", "npt", "production")


def load_config(path: str | Path) -> SimpleNamespace:
    """Load an MD YAML, merged over the defaults; resolve paths."""
    config_path = Path(path).expanduser().resolve()
    conf = _to_namespace(_deep_merge(DEFAULTS, _load_yaml(config_path)))

    base_dir = config_path.parent
    conf.input_pdb = _resolve(base_dir, getattr(conf, "input_pdb", None))
    if getattr(conf, "ligand_sdf", None):
        conf.ligand_sdf = _resolve(base_dir, conf.ligand_sdf)
    workdir = getattr(conf, "workdir", ".")
    if not workdir or str(workdir) == ".":
        conf.workdir = str(base_dir)
    else:
        conf.workdir = _resolve(base_dir, workdir)
    return conf


def validate_config(conf: SimpleNamespace) -> None:
    """Validate a loaded MD configuration (no OpenMM needed)."""
    input_pdb = getattr(conf, "input_pdb", None)
    if not input_pdb:
        raise ValueError("MD config requires 'input_pdb'")
    pdb_path = Path(input_pdb)
    if not pdb_path.is_file() or pdb_path.suffix.lower() != ".pdb":
        raise ValueError(f"input_pdb must be an existing .pdb file: {input_pdb}")

    system_type = str(getattr(conf, "system_type", "soluble"))
    if system_type not in SUPPORTED_SYSTEM_TYPES:
        raise ValueError(
            f"system_type must be one of {sorted(SUPPORTED_SYSTEM_TYPES)} "
            f"(got {system_type!r})"
        )

    forcefields = getattr(conf, "forcefields", None)
    if not forcefields or any(not str(item).strip() for item in forcefields):
        raise ValueError("MD config requires at least one force field")

    ligand_sdf = getattr(conf, "ligand_sdf", None)
    if ligand_sdf:
        sdf_path = Path(ligand_sdf)
        if not sdf_path.is_file() or sdf_path.suffix.lower() != ".sdf":
            raise ValueError(f"ligand_sdf must be an existing .sdf file: {ligand_sdf}")

    if str(getattr(conf, "platform", "CUDA")) not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Unsupported OpenMM platform: {conf.platform}; choose one of "
            f"{', '.join(sorted(SUPPORTED_PLATFORMS))}"
        )

    if float(getattr(conf, "friction", 0.0)) <= 0.0:
        raise ValueError("friction must be > 0 (1/ps)")
    if float(getattr(conf, "temperature", 0.0)) <= 0.0:
        raise ValueError("temperature must be > 0 K")
    if float(getattr(conf, "pressure", 0.0)) <= 0.0:
        raise ValueError("pressure must be > 0 bar")

    for key in _STAGE_KEYS:
        if not hasattr(conf, key):
            raise ValueError(f"MD config is missing the '{key}' section")
    if int(getattr(conf.minimization, "steps", 0)) < 1:
        raise ValueError("minimization.steps must be >= 1")
    for key in ("nvt", "npt", "production"):
        stage = getattr(conf, key)
        if float(getattr(stage, "timestep", 0.0)) <= 0.0:
            raise ValueError(f"{key}.timestep must be > 0 ps")
        if float(getattr(stage, "simulation_time", 0.0)) <= 0.0:
            raise ValueError(f"{key}.simulation_time must be > 0 ps")


def stage(config_path: str | Path, task_directory: str | Path) -> tuple[Path, Path]:
    """Write a runner script that runs MD for ``config_path``."""
    config_path = Path(config_path).expanduser().resolve()
    task = Path(task_directory).expanduser().resolve()
    task.mkdir(parents=True, exist_ok=True)
    runner = task / "run_md.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'trap \'echo "[FAILED] line $LINENO: $BASH_COMMAND (exit $?)" >&2\' ERR\n'
        'CONFIG="${1:?Usage: $0 md.yaml}"\n'
        f"exec {shlex.quote(sys.executable)} -m chemflow.fe.openmm_modeling.md_run "
        '--config "$CONFIG"\n',
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, config_path


def default_yaml_text(
    input_pdb: str = "relaxed_complex.pdb",
    *,
    system_type: str = "soluble",
    ligand_sdf: str | None = None,
    platform: str = "CUDA",
) -> str:
    """Return a commented example YAML for the MD runner."""
    ligand_block = (
        f'ligand_sdf: "{ligand_sdf}"\n'
        'small_molecule_forcefield: "gaff-2.11"   # or an openff-x.y.z SMIRNOFF line\n'
        if ligand_sdf
        else (
            '# ligand_sdf: "ligand.sdf"            # if a ligand is in the complex PDB\n'
            '# small_molecule_forcefield: "gaff-2.11"\n'
        )
    )
    forcefields = "\n".join(f'  - "{item}"' for item in DEFAULT_FORCEFIELDS)
    return (
        "# ===================================================================\n"
        "# OpenMM MD runner - unrestrained production MD.\n"
        "# min -> NVT equil -> NPT equil -> production. system_type selects the\n"
        "# barostat: soluble = isotropic MonteCarloBarostat; membrane = NPgT\n"
        "# MonteCarloMembraneBarostat (XY-isotropic, Z-free). Input must carry a\n"
        "# periodic box (CRYST1). Units: ps, K, bar, bar*nm.\n"
        "# ===================================================================\n"
        "\n"
        f'input_pdb: "{input_pdb}"      # solvated/relaxed complex with a box\n'
        f"{ligand_block}"
        f'system_type: "{system_type}"      # soluble | membrane\n'
        f'platform: "{platform}"           # CUDA | OpenCL | CPU\n'
        "friction: 1.0                   # Langevin collision rate (1/ps)\n"
        "seed: 1\n"
        "temperature: 300.0              # K\n"
        "pressure: 1.0                   # bar\n"
        "surface_tension: 0.0            # bar*nm (membrane only)\n"
        f'output_prefix: "md"\n'
        "report_interval: 5000           # steps between frames\n"
        "checkpoint_interval: 50000\n"
        "forcefields:\n"
        f"{forcefields}\n"
        "\n"
        "minimization:\n"
        "  steps: 10000\n"
        "nvt:                            # NVT equilibration\n"
        "  timestep: 0.002\n"
        "  simulation_time: 250.0        # ps\n"
        "npt:                            # NPT equilibration\n"
        "  timestep: 0.002\n"
        "  simulation_time: 500.0        # ps\n"
        "production:                     # unrestrained production\n"
        "  timestep: 0.004               # 4 fs with HMR\n"
        "  simulation_time: 10000.0      # ps (10 ns; raise as needed)\n"
    )
