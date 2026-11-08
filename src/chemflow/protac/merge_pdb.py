"""Merge two binary complexes into one PDB for protein-protein docking.

RosettaDock want the two partners in a single PDB as distinct
chains. This takes the target complex (target protein + its warhead) and the E3
complex (E3 + its warhead) and writes one PDB with each put on its own chain, so
``partners: A_B`` works and each warhead travels with its protein.

Pure text processing (no RDKit): every ATOM/HETATM of file 1 goes to ``chain1``,
of file 2 to ``chain2``; waters are dropped; atom serials are renumbered.
"""

from __future__ import annotations

from pathlib import Path

_WATERS = {"HOH", "WAT", "H2O", "DOD", "TIP", "TIP3", "SPC"}


def _retag_atoms(lines, chain: str, start_serial: int, *, drop_water: bool) -> tuple[list[str], int]:
    out, serial = [], start_serial
    for line in lines:
        record = line[:6].strip()
        if record not in ("ATOM", "HETATM"):
            continue
        if drop_water and line[17:20].strip().upper() in _WATERS:
            continue
        serial += 1
        # Columns are fixed-width PDB: serial 7-11, chain id 22.
        padded = line.rstrip("\n").ljust(80)
        new = f"{record:<6}{serial:>5}" + padded[11:21] + chain[0] + padded[22:]
        out.append(new.rstrip() + "\n")
    return out, serial


def merge_pdbs(
    pdb1: str | Path,
    pdb2: str | Path,
    out: str | Path,
    *,
    chain1: str = "A",
    chain2: str = "B",
    drop_water: bool = True,
) -> dict:
    """Write ``out`` = ``pdb1`` (on ``chain1``) + ``pdb2`` (on ``chain2``).

    Returns ``{out, chain1, chain2, partners, n_atoms1, n_atoms2}``. Each input
    should be a single partner (one protein + its warhead); every atom in it is
    reassigned to that partner's chain so it moves as one rigid body.
    """
    pdb1, pdb2, out = Path(pdb1), Path(pdb2), Path(out)
    for p in (pdb1, pdb2):
        if not p.is_file():
            raise FileNotFoundError(f"PDB not found: {p}")
    if chain1 == chain2:
        raise ValueError("chain1 and chain2 must differ")

    lines1 = pdb1.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    lines2 = pdb2.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    atoms1, serial = _retag_atoms(lines1, chain1, 0, drop_water=drop_water)
    atoms2, _ = _retag_atoms(lines2, chain2, serial, drop_water=drop_water)
    if not atoms1:
        raise ValueError(f"no usable atoms in {pdb1}")
    if not atoms2:
        raise ValueError(f"no usable atoms in {pdb2}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.writelines(atoms1)
        handle.write("TER\n")
        handle.writelines(atoms2)
        handle.write("TER\nEND\n")
    return {
        "out": out, "chain1": chain1, "chain2": chain2,
        "partners": f"{chain1}_{chain2}",
        "n_atoms1": len(atoms1), "n_atoms2": len(atoms2),
    }
