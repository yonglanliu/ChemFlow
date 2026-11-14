"""Preflight checks on the solvated input structure, before an OpenFE run.

Two failure modes cost days of GPU time in November 2026, both silent:

* ``complex_box_vectors`` in the config disagreed with the real periodic box of
  the input PDB. OpenFE then maps molecules near a boundary into the wrong
  cell, dropping water onto water and lipid onto lipid.
* The solvated input itself contained overlapping atoms (retained waters
  against added solvent, say). OpenFE's hybrid system inherits them, its
  minimisation budget is spent relieving them, and the first MD step NaNs at
  a lambda window — tens of minutes later, with no hint of the cause.

Both are cheap to detect here and essentially impossible to diagnose from the
sampler's error. :func:`preflight` is called from ``create_json`` and reports
in structural terms (residue names and ids), not atom indices.

The geometry itself lives in :mod:`chemflow.utils.structure`; this module only
decides what is acceptable and what the message should say.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chemflow.utils.structure import (
    box_from_topology,
    describe_atom as _describe,
    find_clashes,
    load_structure,
    pseudo_box,
)

# Fatal: atoms this close that are neither bonded nor in the same residue are
# interpenetrating, not merely clashing. Minimisation cannot reliably recover —
# the overlaps that produced 5e14 kJ/mol in Nov 2026 were 0.02-0.03 nm.
DEFAULT_CLASH_NM = 0.05
# Reported but not fatal. Badly placed hydrogens (water H against a polar H,
# say) land around 0.06-0.08 nm and minimisation does clear those.
WARN_CLASH_NM = 0.15
# Box vectors must agree with the file to within this (nm).
BOX_TOLERANCE_NM = 0.01


def preflight(
    pdb_path: str | Path,
    *,
    configured_box=None,
    mode: str = "error",
    clash_nm: float = DEFAULT_CLASH_NM,
    require_box: bool = False,
) -> np.ndarray | None:
    """Validate the solvated input and return the box to use (nm diagonal).

    ``configured_box`` is ``complex_box_vectors`` from the config, if set. A
    disagreement with the box recorded in the file is fatal when ``mode`` is
    ``"error"``, because silently preferring either one is how molecules end up
    in the wrong periodic cell.

    ``require_box`` should be true for membrane systems, where the input is
    already solvated and periodic. For a soluble system OpenFE solvates later,
    so a missing box is normal and only the clash check runs.

    ``mode`` is ``"error"`` (default), ``"warn"`` or ``"off"``.
    """
    mode = str(mode or "error").lower()
    if mode == "off":
        return np.asarray(configured_box, dtype=float) if configured_box is not None else None

    path = Path(pdb_path)
    print(f"Input preflight: {path.name}", flush=True)
    topology, positions = load_structure(path)
    file_box = box_from_topology(topology)
    n_atoms = positions.shape[0]

    if file_box is not None:
        print(f"  periodic box in file : "
              f"{file_box[0]:.4f} x {file_box[1]:.4f} x {file_box[2]:.4f} nm")
    else:
        print("  periodic box in file : (none — no CRYST1 record)")

    box = file_box
    if configured_box is not None:
        wanted = np.asarray(configured_box, dtype=float).reshape(-1)[:3]
        print(f"  complex_box_vectors  : "
              f"{wanted[0]:.4f} x {wanted[1]:.4f} x {wanted[2]:.4f} nm")
        if file_box is not None and not np.allclose(file_box, wanted, atol=BOX_TOLERANCE_NM):
            message = (
                "complex_box_vectors disagrees with the periodic box in "
                f"{path.name}:\n"
                f"    config : {wanted[0]:.4f} x {wanted[1]:.4f} x {wanted[2]:.4f} nm\n"
                f"    file   : {file_box[0]:.4f} x {file_box[1]:.4f} x {file_box[2]:.4f} nm\n"
                "  OpenFE applies the config value, so molecules near a boundary "
                "are mapped into the wrong cell and overlap. Remove "
                "complex_box_vectors to use the box from the file, or rebuild "
                "the system with the box you intended."
            )
            if mode == "error":
                raise ValueError(message)
            print("  WARNING: " + message, flush=True)
        box = wanted

    if box is None:
        if require_box:
            raise ValueError(
                f"{path.name} has no CRYST1 record and complex_box_vectors is not set — "
                "one of them must give the periodic box for a membrane system"
            )
        # Soluble input: not periodic yet, so compare plain distances.
        box = pseudo_box(positions)

    pairs, total = find_clashes(topology, positions, box, cutoff=WARN_CLASH_NM)
    severe = [(d, a, b) for d, a, b in pairs if d < clash_nm]
    poor = [(d, a, b) for d, a, b in pairs if clash_nm <= d < 0.10]
    print(f"  atoms {n_atoms}   close contacts < {WARN_CLASH_NM * 10:.1f} A: {total}"
          f"   (< 1.0 A: {len(severe) + len(poor)}"
          f"   interpenetrating < {clash_nm * 10:.1f} A: {len(severe)})")

    if poor and not severe:
        # Minimisation handles these, but they are still a sign the solvation
        # step ignored hydrogens — worth knowing, not worth stopping for.
        print(f"  note: {len(poor)} pair(s) between {clash_nm * 10:.1f} and 1.0 A, "
              "e.g. " + "; ".join(f"{d * 10:.2f} A {_describe(a).strip()} / "
                                  f"{_describe(b).strip()}" for d, a, b in poor[:3]),
              flush=True)

    if severe:
        lines = [f"    {d * 10:5.2f} A   {_describe(a)}   <->   {_describe(b)}"
                 for d, a, b in severe[:15]]
        message = (
            f"{len(severe)} interpenetrating atom pair(s) in {path.name} "
            f"(closer than {clash_nm * 10:.1f} A, neither bonded nor in the same residue):\n"
            + "\n".join(lines)
            + "\n  These survive into the alchemical hybrid system. OpenFE's "
              "minimisation is spent relieving them and the first MD step of a "
              "lambda window then fails with a NaN. Fix the input: delete the "
              "overlapping waters, or rebuild the solvated system. Set "
              "input_check: warn to proceed anyway."
        )
        if mode == "error":
            raise ValueError(message)
        print("  WARNING: " + message, flush=True)
    elif not poor:
        print("  no overlapping atoms found", flush=True)

    return box
