"""Rank candidate ternary complexes by whether the two warheads can be linked.

After the linker-free search (co-folding with Boltz/Chai, or RosettaDock), each
candidate complex places the two warheads somewhere relative to each other. Before
designing any linker you want to know: are the two attachment points close enough
and pointing toward each other to be bridged? This module answers that from the
two warhead exit vectors (attachment atom + the direction a linker would leave),
which :mod:`chemflow.generative.reinvent.linker.heads` already derives from a 3D
structure.

The core :func:`assess_linkability` is pure geometry (numpy only); the
``*_from_*`` helpers build the exit vectors with RDKit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Through-space distance (A) between attachment atoms a linker can plausibly span.
DEFAULT_MIN_DIST = 3.0
DEFAULT_MAX_DIST = 20.0


@dataclass
class Linkability:
    distance: float            # A, between the two attachment atoms
    toward1: float             # cos angle: head1 exit vector vs. direction to head2
    toward2: float             # cos angle: head2 exit vector vs. direction to head1
    verdict: str               # "linkable" | "too_close" | "too_far" | "wrong_orientation"
    reason: str = ""

    @property
    def linkable(self) -> bool:
        return self.verdict == "linkable"


def assess_linkability(
    origin1, direction1, origin2, direction2,
    *, min_dist: float = DEFAULT_MIN_DIST, max_dist: float = DEFAULT_MAX_DIST,
    toward_tol: float = -0.1,
) -> Linkability:
    """Verdict from the two exit vectors (origins + unit directions).

    ``toward_tol`` is how negative the "points toward the partner" cosine may be
    before the orientation is judged wrong (a small negative allows exit vectors
    that are roughly sideways rather than strictly facing).
    """
    import numpy as np

    o1 = np.asarray(origin1, dtype=float)
    o2 = np.asarray(origin2, dtype=float)
    sep = o2 - o1
    distance = float(np.linalg.norm(sep))
    if distance < 1e-6:
        return Linkability(distance, 1.0, 1.0, "too_close", "attachment atoms coincide")
    unit = sep / distance
    toward1 = float(np.dot(_unit(direction1), unit))
    toward2 = float(np.dot(_unit(direction2), -unit))

    if distance < min_dist:
        verdict, reason = "too_close", f"attachment atoms only {distance:.1f} A apart (< {min_dist})"
    elif distance > max_dist:
        verdict, reason = "too_far", f"attachment atoms {distance:.1f} A apart (> {max_dist}); needs a very long linker"
    elif toward1 < toward_tol or toward2 < toward_tol:
        verdict, reason = "wrong_orientation", "one or both exit vectors point away from the partner"
    else:
        verdict, reason = "linkable", ""
    return Linkability(distance, toward1, toward2, verdict, reason)


def _unit(vector):
    import numpy as np

    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else v


def assess_from_heads(head1, head2, **kwargs) -> Linkability:
    """Linkability of two :class:`HeadSpec` that carry 3D exit vectors."""
    if not (head1.has_3d and head2.has_3d):
        raise ValueError("both heads need 3D coordinates (load posed structures, not SMILES)")
    return assess_linkability(
        head1.exit_origin, head1.exit_direction,
        head2.exit_origin, head2.exit_direction, **kwargs,
    )


def distance_verdict(distance: float, *, min_dist: float = DEFAULT_MIN_DIST,
                     max_dist: float = DEFAULT_MAX_DIST) -> str:
    """Verdict from the attachment-atom distance alone (no orientation)."""
    if distance < min_dist:
        return "too_close"
    if distance > max_dist:
        return "too_far"
    return "linkable"


def attachment_coord(pdb_text: str, resname: str, atom_name: str):
    """(x, y, z) of ``atom_name`` in residue ``resname`` from PDB text, or None."""
    resname = resname.strip().upper()
    atom_name = atom_name.strip().upper()
    for line in pdb_text.splitlines():
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        if line[17:20].strip().upper() != resname:
            continue
        if line[12:16].strip().upper() != atom_name:
            continue
        try:
            return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            return None
    return None


def assess_from_files(
    head1_file, atom1: int, head2_file, atom2: int, *,
    protein=None, **kwargs,
) -> Linkability:
    """Load two posed warheads, pick an attachment atom on each, assess linkability.

    The two files must share one coordinate frame (both taken from the same
    candidate complex), or the distance is meaningless.
    """
    from chemflow.generative.reinvent.linker.heads import load_molecule, make_head

    protein_xyz = None
    if protein:
        from chemflow.generative.reinvent.linker.fit3d import read_pdb_heavy_atoms

        protein_xyz = read_pdb_heavy_atoms(protein)
    h1 = make_head(load_molecule(head1_file), atom1, name="head1", protein_xyz=protein_xyz)
    h2 = make_head(load_molecule(head2_file), atom2, name="head2", protein_xyz=protein_xyz)
    return assess_from_heads(h1, h2, **kwargs)
