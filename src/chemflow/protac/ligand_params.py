"""Generate Rosetta ligand ``.params`` files for the PROTAC workflow.

Rosetta needs a ``.params`` file for each non-standard residue (the warheads)
before RosettaDock can use them. This wraps Rosetta's
``molfile_to_params.py`` so params generation is a first-class pipeline step
between ``prepare`` (extract the warhead) and ``rosettadock``:

    extract warhead SDF  ->  chemflow protac ligand-params  ->  .params + renamed PDB

It never imports Rosetta; it calls the script via subprocess. The script ships
with Rosetta at
``Rosetta/main/source/scripts/python/public/molfile_to_params.py``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_RELPATHS = (
    "scripts/python/public/molfile_to_params.py",            # $ROSETTA3 = main/source
    "main/source/scripts/python/public/molfile_to_params.py",  # $ROSETTA = install root
    "source/scripts/python/public/molfile_to_params.py",     # $ROSETTA = main
)


def default_molfile_to_params() -> str | None:
    """Locate ``molfile_to_params.py`` without the user naming it.

    Checks, in order: ``$MOLFILE_TO_PARAMS`` (full path), the Rosetta roots
    ``$ROSETTA3`` / ``$ROSETTA`` / ``$ROSETTA_HOME`` joined with the usual
    sub-path, the installed **PyRosetta** package (which bundles it), then
    ``molfile_to_params.py`` on ``PATH``. Returns None if absent.
    """
    explicit = os.environ.get("MOLFILE_TO_PARAMS")
    if explicit and Path(explicit).expanduser().is_file():
        return str(Path(explicit).expanduser())
    for var in ("ROSETTA3", "ROSETTA", "ROSETTA_HOME"):
        root = os.environ.get(var)
        if not root:
            continue
        for rel in _SCRIPT_RELPATHS:
            candidate = Path(root).expanduser() / rel
            if candidate.is_file():
                return str(candidate)
    found_in_pyrosetta = _molfile_in_pyrosetta()
    if found_in_pyrosetta:
        return found_in_pyrosetta
    found = shutil.which("molfile_to_params.py")
    return found or None


def _molfile_in_pyrosetta() -> str | None:
    """Find molfile_to_params.py inside an installed PyRosetta package, if any."""
    try:
        import pyrosetta  # noqa: PLC0415 (optional, lazy)
    except Exception:
        return None
    root = Path(pyrosetta.__file__).parent
    for hit in root.rglob("molfile_to_params.py"):
        return str(hit)
    return None



def build_command(
    input_mol: str | Path,
    name: str,
    out_dir: str | Path,
    *,
    molfile_to_params: str,
    python: str | None = None,
    conformers_in_one_file: bool = True,
    keep_names: bool = True,
    extra_args: list[str] | None = None,
) -> list[str]:
    """The ``molfile_to_params.py`` command (``-p`` prefix is inside ``out_dir``)."""
    name = str(name).strip().upper()
    prefix = str(Path(out_dir) / name)
    command = shlex.split(python or sys.executable) + [str(molfile_to_params)]
    command += ["-n", name, "-p", prefix]
    if keep_names:
        command.append("--keep-names")
    if conformers_in_one_file:
        command.append("--conformers-in-one-file")
    command += list(extra_args or [])
    command.append(str(input_mol))
    return command


def generate_params(
    input_mol: str | Path,
    name: str,
    out_dir: str | Path,
    *,
    molfile_to_params: str | None = None,
    python: str | None = None,
    **kwargs,
) -> dict:
    """Run molfile_to_params.py; return ``{params, pdb, name, returncode}``.

    ``molfile_to_params`` defaults to :func:`default_molfile_to_params` (resolved
    from ``$ROSETTA3``/``$ROSETTA``/``$MOLFILE_TO_PARAMS`` or ``PATH``).

    The ligand you place in the docking PDB must use the atom names in the
    returned ``pdb`` (that is why ``--keep-names`` is on by default and the
    generated PDB is returned).
    """
    input_mol = Path(input_mol).expanduser()
    if not input_mol.is_file():
        raise FileNotFoundError(f"ligand structure not found: {input_mol}")
    molfile_to_params = molfile_to_params or default_molfile_to_params()
    if not molfile_to_params:
        raise FileNotFoundError(
            "molfile_to_params.py not found automatically. Set $ROSETTA (Rosetta install "
            "root) or $MOLFILE_TO_PARAMS, or pass --molfile-to-params. It ships with Rosetta "
            "at main/source/scripts/python/public/molfile_to_params.py")
    script = Path(molfile_to_params).expanduser()
    if not script.is_file():
        raise FileNotFoundError(f"molfile_to_params.py not found: {script}")
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = str(name).strip().upper()

    command = build_command(input_mol, name, out_dir, molfile_to_params=str(script),
                            python=python, **kwargs)
    try:
        proc = subprocess.run(command, cwd=str(out_dir), capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"could not launch: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            "molfile_to_params.py failed:\n" + (proc.stderr or proc.stdout or "").strip())

    params = out_dir / f"{name}.params"
    pdbs = sorted(out_dir.glob(f"{name}_*.pdb")) or sorted(out_dir.glob(f"{name}.pdb"))
    if not params.is_file():
        raise RuntimeError(
            f"molfile_to_params.py ran but produced no {params.name}; output:\n"
            + (proc.stdout or proc.stderr or "").strip())
    return {
        "name": name,
        "params": params,
        "pdb": pdbs[0] if pdbs else None,
        "returncode": proc.returncode,
    }
