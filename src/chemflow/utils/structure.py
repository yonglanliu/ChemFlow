"""Geometry helpers for solvated structures: read a file, get its box, find clashes.

These are deliberately free of any workflow policy — no config keys, no
opinions about what counts as fatal. The OpenFE preflight
(:mod:`chemflow.fe.openfe.utils.system_preflight`) layers that on top, and anything
else that needs to sanity-check a built system can reuse these directly.

Only OpenMM and NumPy are needed; nothing here imports RDKit or OpenFE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_structure(path: str | Path):
    """Return ``(topology, positions_nm)`` for a PDB or mmCIF file."""
    from openmm import unit
    from openmm.app import PDBFile, PDBxFile

    path = Path(path)
    reader = PDBxFile if path.suffix.lower() in {".cif", ".mmcif", ".pdbx"} else PDBFile
    structure = reader(str(path))
    positions = np.asarray(
        structure.positions.value_in_unit(unit.nanometer), dtype=float
    )
    return structure.topology, positions


def box_from_topology(topology) -> np.ndarray | None:
    """Diagonal of the periodic box recorded in the file (nm), or None.

    For a PDB this is the ``CRYST1`` record; for an mmCIF, the cell block.
    """
    from openmm import unit

    vectors = topology.getPeriodicBoxVectors()
    if vectors is None:
        return None
    box = np.asarray(vectors.value_in_unit(unit.nanometer), dtype=float)
    return np.array([box[0][0], box[1][1], box[2][2]], dtype=float)


def pseudo_box(positions) -> np.ndarray:
    """A box large enough that minimum-image wrapping is a no-op.

    For a structure that is not periodic yet (a bare protein, say), this lets
    the same clash search run on plain distances.
    """
    positions = np.asarray(positions, dtype=float)
    return (positions.max(axis=0) - positions.min(axis=0)) + 10.0


def find_clashes(topology, positions, box, cutoff: float = 0.15, limit: int = 25):
    """Non-bonded, inter-residue atom pairs closer than ``cutoff`` nm.

    Minimum-image convention against the orthorhombic ``box`` diagonal, a cell
    list at the cutoff spacing, and pairs that are bonded or share a residue
    are skipped (that covers 1-2 and 1-3 contacts within a residue).

    Returns ``(pairs, total)`` where ``pairs`` is up to ``limit``
    ``(distance_nm, atom_a, atom_b)`` tuples, closest first, and ``total`` is
    how many were found altogether.
    """
    atoms = list(topology.atoms())
    residue_of = np.fromiter((a.residue.index for a in atoms), dtype=np.int64,
                             count=len(atoms))
    bonded: set[tuple[int, int]] = set()
    for a, b in topology.bonds():
        i, j = a.index, b.index
        bonded.add((i, j) if i < j else (j, i))

    lengths = np.asarray(box, dtype=float)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(map(tuple, np.floor(positions / cutoff).astype(np.int64))):
        grid.setdefault(cell, []).append(index)

    # Full 27-cell stencil; ``j <= i`` below keeps each pair once.
    stencil = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    found: list[tuple[float, int, int]] = []
    for cell, members in grid.items():
        neighbours: list[int] = []
        for dx, dy, dz in stencil:
            neighbours.extend(grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()))
        for i in members:
            for j in neighbours:
                if j <= i:
                    continue
                if residue_of[i] == residue_of[j] or (i, j) in bonded:
                    continue
                delta = positions[i] - positions[j]
                delta -= lengths * np.round(delta / lengths)
                distance = float(np.sqrt(delta @ delta))
                if distance < cutoff:
                    found.append((distance, i, j))
    found.sort()
    return [(d, atoms[i], atoms[j]) for d, i, j in found[:limit]], len(found)


def describe_atom(atom) -> str:
    """``'  HG1  THR     70 chain D'`` — for messages a human has to act on."""
    residue = atom.residue
    return (f"{atom.name:>4s} {residue.name:>4s} {residue.id:>6s} "
            f"chain {residue.chain.id}")


def bilayer_midplane(topology, positions) -> dict:
    """Locate the bilayer midplane along z from the lipid phosphate planes.

    The midplane is the **midpoint between the two headgroup planes**, not the
    mean of all phosphates and not the centre of the protein. Both of those are
    wrong whenever the system is asymmetric — unequal leaflet populations bias
    the phosphate mean, and a membrane protein with an extramembrane domain
    sits well off the bilayer centre.

    Leaflets are separated at the largest phosphate-free gap in sorted z, which
    is the acyl-chain core, so no assumption about equal populations is made.
    Phosphorus is found by element, since lipid residue and atom naming varies
    (Amber Lipid17 splits POPC into ``POP``/``PC``/``PA``/``OL`` with ``P1``).

    Returns a dict with ``midplane``, ``lower``, ``upper``, ``n_lower``,
    ``n_upper``, ``separation``, ``core_gap`` and ``phosphate_mean`` (the
    biased figure, for comparison). Raises if there are too few phosphates or
    no clear core.
    """
    phosphorus = [
        a.index for a in topology.atoms()
        if (a.element is not None and a.element.symbol == "P")
        or (a.element is None and a.name.upper().startswith("P"))
    ]
    if len(phosphorus) < 8:
        raise ValueError(
            f"found {len(phosphorus)} phosphorus atoms — not enough to locate a "
            "bilayer. Is this structure actually a membrane system?"
        )

    z = np.sort(np.asarray(positions)[phosphorus, 2])
    gaps = np.diff(z)
    core = int(gaps.argmax())
    cut = z[core] + gaps[core] / 2.0
    lower, upper = z[z < cut], z[z >= cut]
    if len(lower) == 0 or len(upper) == 0:
        raise ValueError("the phosphate z-distribution has only one leaflet")
    if gaps[core] < 1.0:
        raise ValueError(
            f"the largest phosphate-free gap is only {gaps[core]:.2f} nm; a bilayer "
            "core should be ~3 nm. The membrane normal is probably not along z."
        )

    return {
        "midplane": float((lower.mean() + upper.mean()) / 2.0),
        "lower": float(lower.mean()),
        "upper": float(upper.mean()),
        "lower_spread": float(lower.std()),
        "upper_spread": float(upper.std()),
        "n_lower": int(len(lower)),
        "n_upper": int(len(upper)),
        "separation": float(upper.mean() - lower.mean()),
        "core_gap": float(gaps[core]),
        "phosphate_mean": float(z.mean()),
    }
