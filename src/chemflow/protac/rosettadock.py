"""Global protein-protein docking with RosettaDock, for the linker-free search.

Given the two binary complexes placed in one PDB as separate chains (target chain
vs. E3 chain, each with its warhead), RosettaDock samples how the two partners
come together. Use *global* mode (randomize the starting orientation) to search
arrangements; rank the outputs by interface score and then by linkability
(:mod:`chemflow.protac.linkability`).

Config-driven subprocess wrapper around an externally-installed Rosetta
(``docking_protocol`` app or ``rosetta_scripts``). Never imports Rosetta.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from alchemfold.boltz_config import _deep_merge, _load_yaml, _resolve, _to_namespace

DEFAULTS: dict[str, Any] = {
    "rosetta_command": "docking_protocol.default.linuxgccrelease",
    "workdir": ".",
    "out_dir": "rosettadock_output",
    "input_pdb": None,          # ONE pdb containing both partners as separate chains
    "partners": None,           # e.g. "A_B" : chain A docks against chain B
    # Rosetta needs a .params file per non-standard residue (the warheads),
    # generated with Rosetta's molfile_to_params.py. Passed as -extra_res_fa.
    "ligand_params": [],
    "mode": "global",           # global (randomize) | local (perturb a known pose)
    "nstruct": 1000,            # models to generate (global search needs many)
    "dock_pert": [3.0, 8.0],    # local-mode perturbation (translation A, rotation deg)
    "extra_flags": [],          # raw Rosetta flags appended verbatim
}

SUPPORTED_MODES = {"global", "local"}


def load_config(path: str | Path) -> SimpleNamespace:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"RosettaDock config not found: {config_path}\n"
            "Create one with: chemflow protac rosettadock generate -o rosettadock.yaml"
        )
    conf = _to_namespace(_deep_merge(DEFAULTS, _load_yaml(config_path)))
    base = config_path.parent
    workdir = getattr(conf, "workdir", ".")
    conf.workdir = str(base) if (not workdir or str(workdir) == ".") else _resolve(base, workdir)
    if getattr(conf, "input_pdb", None):
        conf.input_pdb = _resolve(base, conf.input_pdb)
    conf.ligand_params = [_resolve(base, p) for p in (getattr(conf, "ligand_params", None) or [])]
    conf._config_path = str(config_path)
    return conf


def validate_config(conf: SimpleNamespace) -> None:
    pdb = getattr(conf, "input_pdb", None)
    if not pdb:
        raise ValueError("input_pdb is required (one PDB with both partners as separate chains)")
    if not Path(str(pdb)).is_file():
        raise ValueError(f"input_pdb not found: {pdb}")
    partners = getattr(conf, "partners", None)
    if not partners or "_" not in str(partners):
        raise ValueError("partners is required, e.g. 'A_B' (chains to dock against each other)")
    mode = str(getattr(conf, "mode", "global"))
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"mode must be one of {sorted(SUPPORTED_MODES)} (got {mode!r})")
    if int(getattr(conf, "nstruct", 0)) < 1:
        raise ValueError("nstruct must be >= 1")
    for params in getattr(conf, "ligand_params", None) or []:
        if not Path(str(params)).is_file():
            raise ValueError(f"ligand_params file not found: {params}")


def build_command(conf: SimpleNamespace, out_dir: Path) -> list[str]:
    command = shlex.split(str(conf.rosetta_command))
    command += [
        "-in:file:s", str(conf.input_pdb),
        "-docking:partners", str(conf.partners),
        "-nstruct", str(int(conf.nstruct)),
        "-out:path:all", str(out_dir),
        "-out:file:scorefile", str(out_dir / "score.sc"),
    ]
    params = getattr(conf, "ligand_params", None) or []
    if params:
        command += ["-extra_res_fa", *[str(p) for p in params]]
    if str(conf.mode) == "global":
        # Randomize both partners + spin: a global search of orientations.
        command += ["-docking:randomize1", "-docking:randomize2", "-docking:spin"]
    else:
        pert = [str(x) for x in (getattr(conf, "dock_pert", None) or [3.0, 8.0])]
        command += ["-docking:dock_pert", *pert]
    extra = getattr(conf, "extra_flags", None) or []
    if isinstance(extra, str):
        extra = shlex.split(extra)
    command += [str(x) for x in extra]
    return command


def stage(conf: SimpleNamespace) -> tuple[Path, Path]:
    """Write the output dir + a runner script; return (runner, out_dir)."""
    out_dir = Path(conf.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else Path(conf.workdir) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(conf, out_dir)
    runner = out_dir / "run_rosettadock.sh"
    runner.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'trap \'echo "[FAILED] line $LINENO: $BASH_COMMAND" >&2\' ERR\n'
        + " ".join(shlex.quote(p) for p in command) + "\n",
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, out_dir


def default_yaml_text() -> str:
    return """# ChemFlow -> RosettaDock: linker-free protein-protein docking for the
# PROTAC ternary search. Needs Rosetta installed (its own environment).
#   chemflow protac rosettadock validate rosettadock.yaml
#   chemflow protac rosettadock run      rosettadock.yaml [--dry-run]

# Path to the Rosetta docking app (build suffix depends on your install).
rosetta_command: "docking_protocol.default.linuxgccrelease"
out_dir: rosettadock_output

# ONE pdb holding both partners as separate chains, each with its warhead:
#   e.g. target on chain A (+ warhead), E3 ligase on chain B (+ warhead).
input_pdb: ternary_partners.pdb
partners: "A_B"          # chain A docks against chain B

# Rosetta needs a .params file per warhead (non-standard residue), made with
# Rosetta's molfile_to_params.py. List them here (passed as -extra_res_fa):
# ligand_params:
#   - target_warhead.params
#   - e3_warhead.params

mode: global             # global = randomize+spin (search); local = refine a pose
nstruct: 1000            # global search needs many models
# dock_pert: [3.0, 8.0]  # local mode only: translation (A), rotation (deg)
# extra_flags: ["-ex1", "-ex2aro"]

# After it runs: rank models by interface score (score.sc), then feed the best
# complexes to: chemflow protac linkability  (to pick a linkable arrangement)
# then design the linker with: chemflow generate reinvent linker
"""
