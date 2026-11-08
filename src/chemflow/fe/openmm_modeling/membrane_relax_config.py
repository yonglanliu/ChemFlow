"""Configuration layer for the OpenMM membrane relaxation protocol.

This module is intentionally free of any OpenMM / NumPy import so that the
``alchemforge`` CLI, the desktop dialogs, and the test-suite can load, validate,
and *stage* a membrane-relaxation job without a compute stack installed. The
heavy protocol lives in :mod:`chemflow.fe.openmm_modeling.membrane_relax`.

The relaxation operates on a *pre-trimmed* membrane+protein(+ligand) complex and
preserves the periodic box carried by the input PDB (no extra solvent/membrane
padding is added), so a minimal membrane patch stays minimal.

Units used throughout the configuration:
  * timestep / simulation_time / duration : picoseconds (ps)
  * temperature                           : kelvin (K)
  * force constants                        : kJ/mol/nm^2
  * pressure                               : bar
  * surface_tension                        : bar*nm   (0.0 => tensionless)
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Force-field XML files used to parameterize the membrane+protein system. These
# mirror the OpenFE workflow defaults (AMBER ff14SB + TIP3P + Lipid17). Small
# molecules (ligands) are parameterized separately from an SDF when provided.
DEFAULT_FORCEFIELDS: tuple[str, ...] = (
    "amber/ff14SB.xml",
    "amber/tip3p_standard.xml",
    "amber/tip3p_HFE_multivalent.xml",
    "amber/phosaa10.xml",
    "amber/lipid17_merged.xml",
)

SUPPORTED_PLATFORMS = {"CPU", "CUDA", "OpenCL", "Reference"}

# kcal/mol/A^2 -> kJ/mol/nm^2 is a factor of 418.4. The CHARMM-GUI six-step
# restraint ladder (10, 5, 2.5, 1, 0.5, 0.1 kcal/mol/A^2) therefore becomes the
# values below, decreasing as the system relaxes.
_K_10 = 4184.0   # 10.0 kcal/mol/A^2  (strongest)
_K_5 = 2092.0    #  5.0
_K_2_5 = 1046.0  #  2.5
_K_1 = 418.4     #  1.0
_K_0_5 = 209.2   #  0.5
_K_0_1 = 41.84   #  0.1

_PRESSURE = 1.0          # bar
_SURFACE_TENSION = 0.0   # bar*nm; 0 => tensionless bilayer (free area in XY)

# Optimal multi-stage membrane relaxation, modeled on the Desmond membrane
# relaxation and CHARMM-GUI six-step bilayer equilibration protocols:
#   * start cold and heat gradually (100 -> 150 -> 200 -> 250/275/300 K),
#   * hold strong positional restraints on protein + ligand and a Z-only
#     restraint on lipid head groups, then release them step by step,
#   * switch from NVT to a membrane barostat (NPgT, tensionless) once the box
#     may relax, keeping the XY area free so a trimmed patch finds its density,
#   * grow the timestep (1 -> 2 -> 4 fs) as HMR + constraints allow,
#   * finish with an unrestrained NVT settle and an unrestrained NPgT production.
# A user YAML is merged on top of this, so a minimal config (just ``input_pdb``
# + ``forcefields``) runs this full protocol unchanged.
DEFAULTS: dict[str, Any] = {
    "platform": "CUDA",
    "friction": 1.0,        # Langevin collision rate (1/ps)
    "seed": 1,
    "workdir": ".",
    "forcefields": list(DEFAULT_FORCEFIELDS),
    # Optional: "ligand_sdf" and "small_molecule_forcefield" (e.g. "gaff-2.11"
    # or "openff-2.1.0") to parameterize ligand atoms present in the complex.
    "minimization": {
        "title": "Unrestrained energy minimization",
        "steps": 10000,
        "out_pdb": "00_minimized.pdb",
    },
    "stage1": {
        "name": "01_nvt_100K",
        "title": "Stage 1 - cold restrained NVT (100 K, strongest restraints)",
        "ensemble": "NVT",
        "timestep": 0.001,
        "temperature": 100.0,
        "simulation_time": 50.0,
        "out_pdb": "01_nvt_100K.pdb",
        "restraints": {
            "backbone": {"force_constants": _K_10},
            "sidechain": {"force_constants": _K_5},
            "membrane": {"force_constants": [0.0, 0.0, _K_2_5]},
            "ligand": {"force_constants": _K_10},
        },
    },
    "stage2": {
        "name": "02_npt_150K",
        "title": "Stage 2 - restrained NPgT (150 K, barostat on)",
        "ensemble": "NPGT",
        "timestep": 0.001,
        "temperature": 150.0,
        "simulation_time": 50.0,
        "pressure": _PRESSURE,
        "surface_tension": _SURFACE_TENSION,
        "out_pdb": "02_npt_150K.pdb",
        "restraints": {
            "backbone": {"force_constants": _K_5},
            "sidechain": {"force_constants": _K_2_5},
            "membrane": {"force_constants": [0.0, 0.0, _K_2_5]},
            "ligand": {"force_constants": _K_5},
        },
    },
    "stage3": {
        "name": "03_npt_200K",
        "title": "Stage 3 - restrained NPgT (200 K, short restrained min first)",
        "ensemble": "NPGT",
        "timestep": 0.002,
        "temperature": 200.0,
        "simulation_time": 100.0,
        "pressure": _PRESSURE,
        "surface_tension": _SURFACE_TENSION,
        "out_pdb": "03_npt_200K.pdb",
        "restraints": {
            "backbone": {"force_constants": _K_2_5},
            "sidechain": {"force_constants": _K_1},
            "membrane": {"force_constants": [0.0, 0.0, _K_1]},
            "ligand": {"force_constants": _K_2_5},
        },
    },
    "stage4": {
        "name": "04_heat",
        "title": "Stage 4 - restrained NPgT heating ramp to 300 K",
        "ensemble": "NPGT",
        "timestep": 0.002,
        "temperature": 300.0,
        "pressure": _PRESSURE,
        "surface_tension": _SURFACE_TENSION,
        "out_pdb": "04_heat.pdb",
        # Continue heating from 200 K up to the production temperature.
        "temperature_schedule": [
            {"temperature": 250.0, "duration": 50.0},
            {"temperature": 275.0, "duration": 50.0},
            {"temperature": 300.0, "duration": 100.0},
        ],
        "restraints": {
            "backbone": {"force_constants": _K_1},
            "sidechain": {"force_constants": _K_0_5},
            "membrane": {"force_constants": [0.0, 0.0, _K_0_5]},
            "ligand": {"force_constants": _K_1},
        },
    },
    "stage5": {
        "name": "05_nvt_300K",
        "title": "Stage 5 - lightly restrained NVT (300 K, backbone + ligand only)",
        "ensemble": "NVT",
        "timestep": 0.002,
        "temperature": 300.0,
        "simulation_time": 100.0,
        "out_pdb": "05_nvt_300K.pdb",
        # Sidechain and lipid restraints fully released; weak backbone+ligand
        # anchor remains to keep the pose while packing settles.
        "restraints": {
            "backbone": {"force_constants": _K_0_5},
            "ligand": {"force_constants": _K_0_5},
        },
    },
    "stage6": {
        "name": "06_nvt_unrestrained",
        "title": "Stage 6 - unrestrained NVT settle (300 K, 4 fs)",
        "ensemble": "NVT",
        "timestep": 0.004,
        "temperature": 300.0,
        "simulation_time": 100.0,
        "out_pdb": "06_nvt_unrestrained.pdb",
        "restraints": False,
    },
    "simulation": {
        "name": "production",
        "title": "Production - unrestrained NPgT (300 K, 4 fs)",
        "ensemble": "NPGT",
        "timestep": 0.004,
        "temperature": 300.0,
        "simulation_time": 5000.0,
        "pressure": _PRESSURE,
        "surface_tension": _SURFACE_TENSION,
        "out_pdb": "relaxed_production.pdb",
        "out_state": "relaxed_production.xml",
        "chk": "relaxed_production.chk",
        "restraints": False,
    },
}

_STAGE_KEYS = (
    "minimization",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "simulation",
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_namespace(value: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace (lists handled elementwise)."""
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _to_namespace(item) for key, item in value.items()}
        )
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
        raise ValueError(f"Membrane-relax config must be a YAML mapping: {path}")
    return data


def _resolve(base_dir: Path, value: Any) -> Any:
    if not value:
        return value
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return str(candidate)


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a membrane-relax YAML, merged over the CHARMM-GUI defaults.

    Relative ``input_pdb`` / ``ligand_sdf`` / ``workdir`` paths resolve against
    the config file's directory.
    """
    config_path = Path(path).expanduser().resolve()
    user = _load_yaml(config_path)
    merged = _deep_merge(DEFAULTS, user)
    conf = _to_namespace(merged)

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
    """Validate a loaded membrane-relax configuration (no OpenMM needed)."""
    input_pdb = getattr(conf, "input_pdb", None)
    if not input_pdb:
        raise ValueError("Membrane-relax config requires 'input_pdb'")
    pdb_path = Path(input_pdb)
    if not pdb_path.is_file() or pdb_path.suffix.lower() != ".pdb":
        raise ValueError(f"input_pdb must be an existing .pdb file: {input_pdb}")

    forcefields = getattr(conf, "forcefields", None)
    if not forcefields or any(not str(item).strip() for item in forcefields):
        raise ValueError("Membrane-relax config requires at least one force field")

    ligand_sdf = getattr(conf, "ligand_sdf", None)
    if ligand_sdf:
        sdf_path = Path(ligand_sdf)
        if not sdf_path.is_file() or sdf_path.suffix.lower() != ".sdf":
            raise ValueError(f"ligand_sdf must be an existing .sdf file: {ligand_sdf}")

    platform = str(getattr(conf, "platform", "CUDA"))
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Unsupported OpenMM platform: {platform}; choose one of "
            f"{', '.join(sorted(SUPPORTED_PLATFORMS))}"
        )

    if float(getattr(conf, "friction", 0.0)) <= 0.0:
        raise ValueError("friction must be > 0 (1/ps)")

    for key in _STAGE_KEYS:
        if not hasattr(conf, key):
            raise ValueError(f"Membrane-relax config is missing the '{key}' section")

    minimization = conf.minimization
    if int(getattr(minimization, "steps", 0)) < 1:
        raise ValueError("minimization.steps must be >= 1")

    for key in ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "simulation"):
        stage = getattr(conf, key)
        if float(getattr(stage, "timestep", 0.0)) <= 0.0:
            raise ValueError(f"{key}.timestep must be > 0 ps")
        if key == "stage4":
            schedule = getattr(stage, "temperature_schedule", None)
            if not schedule:
                raise ValueError("stage4.temperature_schedule must list >= 1 segment")
            for segment in schedule:
                if float(getattr(segment, "duration", 0.0)) <= 0.0:
                    raise ValueError(
                        "each stage4 segment needs a positive 'duration' (ps)"
                    )
        else:
            if float(getattr(stage, "simulation_time", 0.0)) <= 0.0:
                raise ValueError(f"{key}.simulation_time must be > 0 ps")


def stage(config_path: str | Path, task_directory: str | Path) -> tuple[Path, Path]:
    """Write a runner script that executes the relaxation for ``config_path``.

    Returns ``(runner_path, config_path)``; the runner takes the YAML config as
    its first argument (matching the desktop job runner convention).
    """
    config_path = Path(config_path).expanduser().resolve()
    task = Path(task_directory).expanduser().resolve()
    task.mkdir(parents=True, exist_ok=True)
    runner = task / "run_membrane_relax.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'trap \'echo "[FAILED] line $LINENO: $BASH_COMMAND (exit $?)" >&2\' ERR\n'
        'CONFIG="${1:?Usage: $0 membrane_relax.yaml}"\n'
        f"exec {shlex.quote(sys.executable)} -m chemflow.fe.openmm_modeling.membrane_relax "
        '--config "$CONFIG"\n',
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, config_path


def default_yaml_text(
    input_pdb: str = "complex_membrane.pdb",
    *,
    ligand_sdf: str | None = None,
    platform: str = "CUDA",
) -> str:
    """Return a commented example YAML with the full optimal multi-stage protocol."""
    ligand_block = (
        f'ligand_sdf: "{ligand_sdf}"\n'
        'small_molecule_forcefield: "gaff-2.11"   # or an openff-x.y.z SMIRNOFF line\n'
        if ligand_sdf
        else (
            '# ligand_sdf: "ligand.sdf"            # uncomment to parameterize a ligand\n'
            '# small_molecule_forcefield: "gaff-2.11"\n'
        )
    )
    forcefields = "\n".join(f'  - "{item}"' for item in DEFAULT_FORCEFIELDS)
    return (
        "# ===================================================================\n"
        "# OpenMM membrane relaxation - optimal multi-stage protocol\n"
        "# Modeled on the Desmond membrane relaxation + CHARMM-GUI six-step\n"
        "# bilayer equilibration: cold start, gradual heating, progressive\n"
        "# restraint release, NVT -> membrane-barostat (NPgT) hand-off.\n"
        "# Operates on a PRE-TRIMMED complex and keeps the input PDB periodic\n"
        "# box (no padding), so the system stays small.\n"
        "# Units: ps, K, kJ/mol/nm^2, bar, bar*nm. k given as kcal/mol/A^2 x 418.4.\n"
        "#\n"
        "#  stage  ensemble   T(K)      dt(fs)  time(ps)  bb    sc    lipidZ  lig\n"
        "#  min    -          -         -       10000 it  unrestrained\n"
        "#  1      NVT        100       1       50        4184  2092  1046    4184\n"
        "#  2      NPgT       150       1       50        2092  1046  1046    2092\n"
        "#  3      NPgT       200       2       100       1046  418   418     1046\n"
        "#  4      NPgT       250->300  2       50/50/100 418   209   209     418\n"
        "#  5      NVT        300       2       100       209   -     -       209\n"
        "#  6      NVT        300       4       100       unrestrained\n"
        "#  prod   NPgT       300       4       5000      unrestrained\n"
        "# ===================================================================\n"
        "\n"
        f'input_pdb: "{input_pdb}"      # prepared complex with a CRYST1 box line\n'
        f"{ligand_block}"
        f'platform: "{platform}"           # CUDA | OpenCL | CPU\n'
        "friction: 1.0                   # Langevin collision rate (1/ps)\n"
        "seed: 1\n"
        "forcefields:\n"
        f"{forcefields}\n"
        "\n"
        "# -------------------------------------------------------------------\n"
        "# Every key below is shown with its default value - edit to tune. The\n"
        "# loader merges this over the built-in protocol, so you may also delete\n"
        "# any stage/key you do not want to change.\n"
        "# -------------------------------------------------------------------\n"
        "minimization:\n"
        "  steps: 10000                  # unrestrained steepest-descent iterations\n"
        "\n"
        "stage1:                         # cold restrained NVT, strongest restraints\n"
        "  temperature: 100.0\n"
        "  timestep: 0.001\n"
        "  simulation_time: 50.0\n"
        "  restraints:\n"
        "    backbone:  {force_constants: 4184.0}   # 10.0 kcal/mol/A^2\n"
        "    sidechain: {force_constants: 2092.0}   #  5.0\n"
        "    membrane:  {force_constants: [0.0, 0.0, 1046.0]}   # lipid head Z, 2.5\n"
        "    ligand:    {force_constants: 4184.0}\n"
        "\n"
        "stage2:                         # restrained NPgT, barostat on\n"
        "  temperature: 150.0\n"
        "  timestep: 0.001\n"
        "  simulation_time: 50.0\n"
        "  pressure: 1.0\n"
        "  surface_tension: 0.0          # bar*nm; 0 => tensionless bilayer\n"
        "  restraints:\n"
        "    backbone:  {force_constants: 2092.0}\n"
        "    sidechain: {force_constants: 1046.0}\n"
        "    membrane:  {force_constants: [0.0, 0.0, 1046.0]}\n"
        "    ligand:    {force_constants: 2092.0}\n"
        "\n"
        "stage3:                         # restrained NPgT, short restrained min first\n"
        "  temperature: 200.0\n"
        "  timestep: 0.002\n"
        "  simulation_time: 100.0\n"
        "  pressure: 1.0\n"
        "  surface_tension: 0.0\n"
        "  restraints:\n"
        "    backbone:  {force_constants: 1046.0}\n"
        "    sidechain: {force_constants: 418.4}\n"
        "    membrane:  {force_constants: [0.0, 0.0, 418.4]}\n"
        "    ligand:    {force_constants: 1046.0}\n"
        "\n"
        "stage4:                         # restrained NPgT heating ramp to 300 K\n"
        "  timestep: 0.002\n"
        "  pressure: 1.0\n"
        "  surface_tension: 0.0\n"
        "  temperature_schedule:\n"
        "    - {temperature: 250.0, duration: 50.0}\n"
        "    - {temperature: 275.0, duration: 50.0}\n"
        "    - {temperature: 300.0, duration: 100.0}\n"
        "  restraints:\n"
        "    backbone:  {force_constants: 418.4}\n"
        "    sidechain: {force_constants: 209.2}\n"
        "    membrane:  {force_constants: [0.0, 0.0, 209.2]}\n"
        "    ligand:    {force_constants: 418.4}\n"
        "\n"
        "stage5:                         # lightly restrained NVT (backbone + ligand)\n"
        "  temperature: 300.0\n"
        "  timestep: 0.002\n"
        "  simulation_time: 100.0\n"
        "  restraints:\n"
        "    backbone: {force_constants: 209.2}\n"
        "    ligand:   {force_constants: 209.2}\n"
        "\n"
        "stage6:                         # unrestrained NVT settle (4 fs, HMR)\n"
        "  temperature: 300.0\n"
        "  timestep: 0.004\n"
        "  simulation_time: 100.0\n"
        "  restraints: false\n"
        "\n"
        "simulation:                     # unrestrained NPgT production\n"
        "  temperature: 300.0\n"
        "  timestep: 0.004\n"
        "  simulation_time: 5000.0\n"
        "  pressure: 1.0\n"
        "  surface_tension: 0.0\n"
        "  restraints: false\n"
    )
