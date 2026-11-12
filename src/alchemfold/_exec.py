"""Resolve engine executables when the env isn't activated.

The AlchemFold CLI is often launched by an absolute interpreter path (e.g.
``/data/<user>/conda/envs/alchemfold/bin/python -m alchemfold ...``) or by the
desktop, *without* ``conda activate``. In that case the env's ``bin/`` is not on
``PATH``, so a bare console-script name like ``boltz`` / ``chai-lab`` fails with
``FileNotFoundError``. :func:`resolve_command` rewrites the leading token to the
matching executable in the running interpreter's own ``bin/`` (the same env), or
anything found on ``PATH``, before we hand the command to ``subprocess``.
"""

from __future__ import annotations

import os
import shutil
import sys


def _sibling_env_executables(name: str, prefer: tuple[str, ...] = ("alchemfold",)):
    """Yield ``bin/<name>`` paths in sibling conda envs of the running interpreter.

    The desktop runs in the ``chemflow`` env, but the docking engines (``boltz`` /
    ``chai-lab``) live in the ``alchemfold`` env. When ``sys.executable`` is inside
    a conda ``envs/<x>/bin`` tree, search every sibling env's ``bin/`` for ``name``,
    trying the envs in ``prefer`` first (in order).
    """
    exe = os.path.abspath(sys.executable)
    parts = exe.split(os.sep)
    if "envs" not in parts:
        return
    envs_root = os.sep.join(parts[: len(parts) - 1 - parts[::-1].index("envs") + 1])
    try:
        env_names = sorted(os.listdir(envs_root))
    except OSError:
        return

    def rank(env_name: str) -> tuple[int, str]:
        try:
            return (prefer.index(env_name), env_name)
        except ValueError:
            return (len(prefer), env_name)

    for env_name in sorted(env_names, key=rank):
        candidate = os.path.join(envs_root, env_name, "bin", name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            yield candidate


def resolve_command(command: list[str],
                    prefer_envs: tuple[str, ...] = ("alchemfold",)) -> list[str]:
    """Return ``command`` with its first token resolved to a real executable.

    If the first token already contains a path separator it is left untouched.
    Otherwise we look for it (1) next to ``sys.executable`` (the active env's
    ``bin/``), then (2) on ``PATH``, then (3) in a sibling conda env (preferring
    ``prefer_envs``, by default ``alchemfold``) — so a desktop run from the
    ``chemflow`` env still finds ``boltz`` / ``chai-lab`` in the ``alchemfold``
    env. If nothing matches the token is returned unchanged so the original
    ``FileNotFoundError`` still surfaces clearly.

    ``prefer_envs`` matters when the SAME executable exists in more than one env:
    Boltz with the GPU kernels lives in its own ``boltz`` env (the kernels need
    torch>=2.6, which Chai-1 forbids), so a kernels run should prefer that one.
    """
    if not command:
        return command
    first = command[0]
    if os.sep in first or (os.altsep and os.altsep in first):
        return command

    bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    candidate = os.path.join(bin_dir, first)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return [candidate, *command[1:]]

    found = shutil.which(first)
    if found:
        return [found, *command[1:]]

    # A sibling conda env (e.g. alchemfold / boltz) may hold the engine
    # executable even though the running interpreter (e.g. chemflow) does not.
    for sibling in _sibling_env_executables(first, prefer_envs):
        return [sibling, *command[1:]]

    # Also try the env bin without the executable bit check (rare), else keep as-is.
    if os.path.exists(candidate):
        return [candidate, *command[1:]]
    return command


def _conda_env_roots():
    """Candidate conda ``envs/`` directories, most likely first."""
    roots: list[str] = []

    def add(path: str | None) -> None:
        if path and path not in roots and os.path.isdir(path):
            roots.append(path)

    # 1. The envs/ dir holding the running interpreter (or CONDA_PREFIX).
    for base in (os.path.abspath(sys.executable), os.environ.get("CONDA_PREFIX", "")):
        parts = base.split(os.sep)
        if "envs" in parts:
            add(os.sep.join(parts[: len(parts) - 1 - parts[::-1].index("envs") + 1]))
    # 2. A base installation (conda run from base has no 'envs' in its prefix).
    for var in ("CONDA_PREFIX", "MAMBA_ROOT_PREFIX", "CONDA_ROOT"):
        base = os.environ.get(var)
        if base:
            add(os.path.join(base, "envs"))
    # 3. Common user installs.
    home = os.path.expanduser("~")
    for name in ("miniforge3", "mambaforge", "miniconda3", "anaconda3", "conda"):
        add(os.path.join(home, name, "envs"))
    return roots


def conda_env_executable(env_name: str, exe: str) -> str | None:
    """Absolute path to ``<conda envs>/<env_name>/bin/<exe>``, or None.

    Used to turn an env NAME the user typed (e.g. ``boltz``) into the concrete
    ``boltz_command`` written into a config, so the run does not depend on which
    env happens to be active.
    """
    if not env_name:
        return None
    for root in _conda_env_roots():
        candidate = os.path.join(root, env_name, "bin", exe)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def boltz_kernels_enabled(conf) -> bool:
    """True when this config asks Boltz to use its optimized triangle kernels.

    Mirrors the condition in :func:`alchemfold.boltz.run.build_command` — the
    default is OFF (pure PyTorch), because ``cuequivariance_torch`` / ``trifast``
    are not installed in the shared AlchemFold env.
    """
    if os.environ.get("ALCHEMFOLD_BOLTZ_NO_KERNELS"):
        return False
    return not bool(getattr(conf, "no_kernels", True))


def prefer_envs_for(conf) -> tuple[str, ...]:
    """Which sibling conda envs to search first for this run's engine binary.

    ONE ENV PER ENGINE: ``boltz`` (``chemflow boltz-env install``), ``chai``
    (``chemflow chai-env install``) and ``diffdock`` (``chemflow diffdock-env
    install``). The old shared ``alchemfold`` env is retired — the AlchemFold
    CLI itself runs from the ``chemflow`` env — but it is kept as the last
    fallback so existing installs keep working.
    """
    engine = str(getattr(conf, "engine", "boltz") or "boltz").lower()
    if engine.startswith("boltz"):
        return ("boltz", "alchemfold")
    if engine.startswith("chai"):
        return ("chai", "alchemfold")
    if engine == "diffdock":
        return ("diffdock", "alchemfold")
    return ("alchemfold",)


def wrap_srun(command: list[str], devices: int = 1) -> list[str]:
    """Launch ``command`` through ``srun`` (one task per GPU) when Lightning needs it.

    Under SLURM, PyTorch Lightning expects one process per GPU started by ``srun``
    (``--ntasks-per-node == devices``). So for multi-GPU runs — or any run inside a
    multi-task allocation (e.g. a 1-GPU MSA step in a 4-GPU job, where Lightning would
    otherwise see SLURM_NTASKS=4 but devices=1) — wrap the engine command in
    ``srun --ntasks=<devices>``. Single-GPU runs in a 1-task job, runs outside SLURM,
    and hosts without ``srun`` are returned unchanged.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        return command
    srun = shutil.which("srun")
    if not srun:
        return command
    devices = max(1, int(devices or 1))
    try:
        ntasks = int(os.environ.get("SLURM_NTASKS", "1") or 1)
    except ValueError:
        ntasks = 1
    if devices == 1 and ntasks <= 1:
        return command
    wrapped = [srun, "--nodes=1", f"--ntasks={devices}", f"--ntasks-per-node={devices}",
               "--kill-on-bad-exit=1"]
    cpt = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpt:  # newer SLURM no longer lets srun inherit --cpus-per-task from sbatch
        wrapped.append(f"--cpus-per-task={cpt}")
    return wrapped + list(command)
