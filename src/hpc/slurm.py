"""Render the ``#SBATCH`` fragments every module shares.

Modules keep their own script templates — an OpenFE array job and a GROMACS
HREX job have little else in common — but the directives themselves, the GPU
request and the conda activation block are identical boilerplate that had been
copy-pasted into nine renderers.

Project rule enforced here: **SLURM arrays are plain ``0-N`` with no ``%``
concurrency cap.** :func:`array_spec` refuses to emit one.
"""

from __future__ import annotations

from typing import Iterable

from hpc.resources import Resources


class ArrayCapError(ValueError):
    """Raised when something tries to put a ``%concurrency`` cap on an array."""


def array_spec(count: int | None = None, *, indices: Iterable[int] | None = None) -> str:
    """``"0-(count-1)"``, or a discrete ``"3,7,11"`` list for a restart.

    Never includes a ``%cap``: capping an array throttles the whole campaign to
    a few concurrent tasks and is forbidden in this project.
    """
    if indices is not None:
        values = sorted({int(i) for i in indices})
        if not values:
            return "0-0"
        return ",".join(str(i) for i in values)
    n = max(int(count or 0), 1)
    return f"0-{n - 1}"


def check_no_cap(spec: str) -> str:
    """Pass a user-supplied array spec through, rejecting a ``%cap``."""
    text = str(spec or "").strip()
    if "%" in text:
        raise ArrayCapError(
            f"array spec {text!r} has a %concurrency cap; this project requires "
            "plain --array=0-N so the scheduler decides how many run at once")
    return text


def directives(res: Resources, *, job_name: str = "", output: str = "",
               error: str = "", array: str = "", gpu_style: str = "gres",
               with_constraint: bool = False, ntasks: int | None = None,
               cpus_per_task: int | None = None) -> str:
    """Return the ``#SBATCH`` block for this allocation.

    ``gpu_style`` picks between ``--gres=gpu:[type:]N`` (most of the repo) and a
    bare ``--gpus=N`` (septop / abfe / resFEP). ``ntasks`` / ``cpus_per_task``
    override the preset for renderers that compute them per job (HREX uses one
    task per lambda window).
    """
    lines: list[str] = []
    add = lines.append
    if job_name:
        add(f"#SBATCH --job-name={job_name}")
    if array:
        add(f"#SBATCH --array={check_no_cap(array)}")
    if res.partition:
        add(f"#SBATCH --partition={res.partition}")
    if res.account:
        add(f"#SBATCH --account={res.account}")
    if res.nodes > 0:
        add(f"#SBATCH --nodes={res.nodes}")
    tasks = res.ntasks if ntasks is None else int(ntasks)
    if tasks > 0:
        add(f"#SBATCH --ntasks={tasks}")
    cpus = res.cpus_per_task if cpus_per_task is None else int(cpus_per_task)
    if cpus > 0:
        add(f"#SBATCH --cpus-per-task={cpus}")
    if res.gpus > 0:
        add(f"#SBATCH --gres={res.gres()}" if gpu_style == "gres"
            else f"#SBATCH --gpus={res.gpus}")
    if with_constraint and res.constraint:
        add(f"#SBATCH --constraint={res.constraint}")
    if res.mem:
        add(f"#SBATCH --mem={res.mem}")
    if res.time:
        add(f"#SBATCH --time={res.time}")
    if output:
        add(f"#SBATCH --output={output}")
    if error:
        add(f"#SBATCH --error={error}")
    return "\n".join(lines) + "\n"


def environment(res: Resources) -> str:
    """``module load`` lines plus the conda activation, or ``""``."""
    lines: list[str] = []
    modules = [m for m in str(res.modules or "").replace(",", " ").split() if m]
    if modules:
        lines.append("module purge 2>/dev/null || true")
        lines += [f"module load {m}" for m in modules]
    if res.conda_env:
        init = str(res.conda_init or "").strip()
        lines.append(f'source "{init}"' if init else
                     'source "$(conda info --base)/etc/profile.d/conda.sh"')
        lines.append(f"conda activate {res.conda_env}")
    return ("\n".join(lines) + "\n") if lines else ""


def sbatch_args(res: Resources, *, gpu_style: str = "gres",
                array: str = "") -> list[str]:
    """The same allocation as ``sbatch`` command-line arguments.

    GROMACS RBFE passes resources at submit time instead of writing them into
    the script, so it needs the argv form.
    """
    args: list[str] = []
    if array:
        args.append(f"--array={check_no_cap(array)}")
    if res.partition:
        args.append(f"--partition={res.partition}")
    if res.account:
        args.append(f"--account={res.account}")
    if res.nodes > 0:
        args.append(f"--nodes={res.nodes}")
    if res.ntasks > 0:
        args.append(f"--ntasks={res.ntasks}")
    if res.cpus_per_task > 0:
        args.append(f"--cpus-per-task={res.cpus_per_task}")
    if res.gpus > 0:
        args.append(f"--gres={res.gres()}" if gpu_style == "gres"
                    else f"--gpus={res.gpus}")
    if res.constraint:
        args.append(f"--constraint={res.constraint}")
    if res.mem:
        args.append(f"--mem={res.mem}")
    if res.time:
        args.append(f"--time={res.time}")
    return args
