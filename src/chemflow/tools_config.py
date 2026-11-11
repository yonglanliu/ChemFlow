"""Single config file that declares how external tools link to ChemFlow.

Instead of exporting an env var per tool every session, declare them once in a
``chemflow_tools.yaml`` and ChemFlow picks them up automatically. For each tool
you choose *install* (give a conda/Python path or executable) or *load* (list the
HPC modules and let ``PATH``/``PYTHONPATH`` provide it) — the file records that
choice.

Lookup order for the file:
1. ``$CHEMFLOW_TOOLS_CONFIG``
2. ``./chemflow_tools.yaml`` (current working dir / project)
3. ``~/.config/chemflow/tools.yaml``

Schema (every key optional — omit a tool to fall back to auto-detection)::

    fe:        {python: /env/bin/python}     # OpenMM / OpenFE interpreter (imported)
    rosetta:   {python: /env/bin/python}     # PyRosetta interpreter (imported)
    reinvent:  {command: /path/to/reinvent}  # REINVENT4 executable
    gromacs:   {command: gmx}                 # or a full path
    boltz:     {command: boltz}
    diffdock:  {python: /env/bin/python, dir: /path/to/DiffDock}   # DiffDock-L env + repo
    rosetta_apps: {rosetta3: /rosetta/main/source}   # classic Rosetta root
    hpc:       {modules: [gromacs, openmm, openfe, pyrosetta, boltz2],
                conda_init: <conda-prefix>/etc/profile.d/conda.sh,
                conda_environment: openfe}

``apply_env()`` maps these onto the env vars the resolvers already honour, without
overriding anything already set in the shell. This module is pure stdlib (YAML is
imported lazily only when a file is read), so it is import-safe everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = [
    "config_path",
    "load_config",
    "apply_env",
    "hpc_defaults",
    "ENV_MAP",
    "TEMPLATE",
]

# config dotted-key -> environment variable the resolvers read.
ENV_MAP: dict[str, str] = {
    "fe.python": "CHEMFLOW_FE_PYTHON",
    "gromacs.python": "CHEMFLOW_GMX_PYTHON",
    "rosetta.python": "CHEMFLOW_ROSETTA_PYTHON",
    "reinvent.command": "REINVENT_COMMAND",
    "gromacs.command": "CHEMFLOW_GROMACS_COMMAND",
    "boltz.command": "CHEMFLOW_BOLTZ_COMMAND",
    "diffdock.python": "CHEMFLOW_DIFFDOCK_PYTHON",
    "diffdock.dir": "CHEMFLOW_DIFFDOCK_DIR",
    "obabel.command": "CHEMFLOW_OBABEL_COMMAND",
    "acpype.command": "CHEMFLOW_ACPYPE_COMMAND",
    "rosetta_apps.rosetta3": "ROSETTA3",
}

TEMPLATE = """\
# chemflow_tools.yaml — how external tools link to ChemFlow.
# For each tool pick ONE: give a path (you installed it) OR leave it null and
# `module load` it on HPC (PATH/PYTHONPATH provides it). Omit a tool entirely to
# let ChemFlow auto-detect a conda env / PATH.

fe:                       # OpenMM / OpenFE (Python-imported)
  python: null            # e.g. <conda-prefix>/envs/alchemforge/bin/python

gromacs:
  python: null            # GROMACS env interpreter, e.g. <conda-prefix>/envs/alchemforge/bin/python
  command: gmx            # gmx executable (full path, or `gmx` if a module provides it)

rosetta:                  # PyRosetta (Python-imported)
  python: null            # e.g. <conda-prefix>/envs/pyrosetta/bin/python

reinvent:
  command: null           # e.g. <conda-prefix>/envs/reinvent/bin/reinvent

boltz:
  command: boltz

diffdock:                 # DiffDock-L (own conda env + cloned repo; `chemflow diffdock-env install`)
  python: null            # e.g. <conda-prefix>/envs/diffdock/bin/python  (null = auto-detect)
  dir: null               # e.g. <ChemFlow>/DiffDock                       (null = auto-detect)

obabel:
  command: obabel         # Open Babel (GROMACS ligand prep); null to use PATH

acpype:
  command: acpype         # ACPYPE (wraps AmberTools antechamber); null to use PATH

rosetta_apps:             # classic Rosetta C++ apps (molfile_to_params, docking)
  rosetta3: null          # e.g. /sw/rosetta/main/source

hpc:                      # defaults for generated SLURM scripts
  modules: []             # e.g. [gromacs, openmm, openfe, pyrosetta, boltz2]
  conda_init: null        # e.g. <conda-prefix>/etc/profile.d/conda.sh
  conda_environment: null # e.g. alchemforge
"""


def config_path() -> Path | None:
    """Return the first existing tools-config file, or None."""
    explicit = os.environ.get("CHEMFLOW_TOOLS_CONFIG")
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        Path.cwd() / "chemflow_tools.yaml",
        Path.home() / ".config" / "chemflow" / "tools.yaml",
    ]
    for cand in candidates:
        if cand and cand.is_file():
            return cand
    return None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the tools config as a dict ({} if none found)."""
    source = Path(path).expanduser() if path else config_path()
    if source is None or not source.is_file():
        return {}
    import yaml

    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _get(config: dict, dotted: str):
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _recorded_diffdock() -> dict[str, str]:
    """``dir``/``python`` written by ``chemflow diffdock-env install`` (stdlib-only parse)."""
    import re

    out: dict[str, str] = {}
    try:
        text = (Path.home() / ".config" / "chemflow" / "diffdock.yaml").read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        m = re.match(r'\s*(dir|python)\s*:\s*"?([^"#]+?)"?\s*(#.*)?$', line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def apply_env(config: dict | None = None, environ: dict | None = None) -> dict[str, str]:
    """Export the config's tool links as env vars (shell values take precedence).

    Returns the mapping of env vars that were newly set. ``config`` / ``environ``
    can be injected for testing; by default it reads the discovered file and
    mutates ``os.environ``.
    """
    config = load_config() if config is None else config
    env = os.environ if environ is None else environ
    applied: dict[str, str] = {}
    for dotted, var in ENV_MAP.items():
        value = _get(config, dotted)
        if value in (None, "", []):
            continue
        if env.get(var):           # explicit shell value wins
            continue
        text = str(value)
        expand = dotted.endswith(("python", "rosetta3", "diffdock.dir")) or text.startswith("~")
        env[var] = str(Path(text).expanduser()) if expand else text
        applied[var] = env[var]
    # DiffDock: also honour the location `chemflow diffdock-env install` recorded
    # (~/.config/chemflow/diffdock.yaml), unless the shell / tools file already set it.
    for key, var in (("python", "CHEMFLOW_DIFFDOCK_PYTHON"), ("dir", "CHEMFLOW_DIFFDOCK_DIR")):
        if not env.get(var):
            value = _recorded_diffdock().get(key)
            if value:
                env[var] = value
                applied[var] = value
    return applied


def hpc_defaults(config: dict | None = None) -> dict[str, Any]:
    """HPC SLURM defaults from the config (modules / conda_init / conda_environment)."""
    config = load_config() if config is None else config
    hpc = config.get("hpc") if isinstance(config.get("hpc"), dict) else {}
    modules = hpc.get("modules") or []
    if isinstance(modules, str):
        modules = modules.split()
    return {
        "modules": " ".join(str(m) for m in modules),
        "conda_init": str(hpc.get("conda_init") or ""),
        "conda_environment": str(hpc.get("conda_environment") or ""),
    }
