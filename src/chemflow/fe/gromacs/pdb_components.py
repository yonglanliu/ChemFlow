"""Classify the residues in an assembled PDB against the force-field libraries.

The GROMACS system-preparation route loads leaprc force fields that recognize the
standard components by residue name (protein, DNA, RNA, lipids, water, standard
ions). Everything else — ligands, organic cofactors, non-standard / metal ions —
is a "gap" that needs explicit parameters (GAFF2 from an SDF, or a prepared ITP).

This module is a pure parser (no RDKit / GROMACS / Qt imports) so the desktop can
offer a one-click component inventory and the headless CLI can reuse the same
classification. It reuses the residue-name sets already defined for preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chemflow.fe.gromacs.membrane import (
    DEFAULT_LIPID_RESNAMES,
    _AMBER_SPLIT_POPC_COMPONENTS,
)
from chemflow.fe.gromacs.system_prepare import _DNA_RESIDUES, _RNA_RESIDUES

__all__ = ["Component", "classify_pdb_components", "ChargeEstimate", "estimate_formal_charge"]

# Standard amino acids plus common protonation / terminal / cap variants that
# the Amber protein force fields parameterize by name.
_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # Amber protonation / tautomer / disulfide variants
    "HID", "HIE", "HIP", "ASH", "GLH", "LYN", "CYX", "CYM", "ARN", "TYM",
    # CHARMM histidine protonation names
    "HSD", "HSE", "HSP",
    # terminal caps
    "ACE", "NME", "NMA", "NHE",
}

_WATER = {
    "HOH", "WAT", "SOL", "TIP", "TIP3", "TIP4", "TIP5", "T3P", "TP3", "TP4",
    "T4P", "H2O", "OH2", "SPC", "SPCE", "OPC", "OPC3",
}

# Monovalent / common ions covered by the standard Amber/CHARMM ion sets.
# Includes CHARMM resnames (SOD, CLA, POT, CAL, CES, LIT, ZN2, MG).
_STANDARD_IONS = {
    "NA", "NA+", "SOD", "K", "K+", "POT", "LI", "LI+", "LIT", "RB", "RB+",
    "CS", "CS+", "CES", "CL", "CL-", "CLA", "F", "F-", "BR", "BR-", "I", "I-",
    "MG", "MG2+", "CA", "CA2+", "CAL", "ZN", "ZN2+", "ZN2", "IB", "IOD",
}

# Transition / heavy metals that typically need bespoke ion parameters (12-6-4
# or literature sets) rather than the default monovalent library.
_SPECIAL_METALS = {
    "FE", "FE2", "FE3", "MN", "MN2", "CU", "CU1", "CU2", "NI", "CO", "CD",
    "HG", "MO", "W", "PT", "PD", "AU", "AG", "CR", "V", "SE",
}

_LIPIDS = {name.upper() for name in DEFAULT_LIPID_RESNAMES}


@dataclass(frozen=True)
class Component:
    """One distinct residue name found in the PDB and how it should be treated."""

    resname: str
    kind: str          # protein | dna | rna | water | ion | lipid | metal | unknown
    count: int         # number of residues with this name
    atom_count: int    # atoms in the largest residue of this name (element hint)
    covered: bool      # True if the loaded force fields parameterize it by name
    suggestion: str    # how to fill the gap (empty when covered)


def _classify(resname: str, atom_count: int) -> tuple[str, bool, str]:
    name = resname.upper()
    if name in _AMINO_ACIDS:
        return "protein", True, ""
    if name in _DNA_RESIDUES:
        return "dna", True, ""
    if name in _RNA_RESIDUES:
        return "rna", True, ""
    if name in _WATER:
        return "water", True, ""
    if name in _LIPIDS or name in _AMBER_SPLIT_POPC_COMPONENTS:
        return "lipid", True, ""
    if name in _STANDARD_IONS:
        return "ion", True, ""
    if name in _SPECIAL_METALS or atom_count == 1:
        return (
            "metal" if name in _SPECIAL_METALS else "ion",
            False,
            "Attach ion parameters (ITP) — antechamber cannot parameterize metals",
        )
    return (
        "unknown",
        False,
        "Parameterize with GAFF2 from an SDF/MOL2, or attach a prepared ITP",
    )


def _resolve_histidine(resname: str, atoms: set[str]) -> str:
    """Resolve a generic ``HIS`` to HID / HIE / HIP from the protons present.

    A residue named HIS carries no charge information in its name — the state is
    defined by which ring hydrogens exist: δ-proton HD1 only -> HID (neutral),
    ε-proton HE2 only -> HIE (neutral), both -> HIP (+1, charged). Names that are
    already explicit (HID/HIE/HIP, CHARMM HSD/HSE/HSP) are returned unchanged. If
    the structure has no hydrogens, HIS is kept (protonation undeterminable).
    """
    if resname not in {"HIS", "HISA", "HISB", "HISE", "HISD", "HISH"}:
        return resname
    has_delta = "HD1" in atoms or "DD1" in atoms
    has_epsilon = "HE2" in atoms or "DE2" in atoms
    if has_delta and has_epsilon:
        return "HIP"
    if has_epsilon:
        return "HIE"
    if has_delta:
        return "HID"
    return resname


def classify_pdb_components(pdb_path: str | Path) -> list[Component]:
    """Return the distinct residues in a PDB with coverage classification.

    Residues are counted by (chain, residue sequence number, insertion code) so
    repeated copies of the same molecule collapse to one row with a count.
    """
    path = Path(pdb_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    # Count residues by walking atoms in order and detecting residue boundaries,
    # rather than collecting unique (chain, resSeq) keys. PDB resSeq is only 4
    # columns (max 9999) and wraps in large solvated/membrane systems, so a set
    # of keys silently merges many ions/waters. A boundary starts a new residue
    # when the (chain, resSeq, iCode, resName) tuple changes OR when an atom name
    # repeats within the current residue (which also makes each monatomic ion its
    # own residue, fixing ion counts).
    counts: dict[str, int] = {}
    max_atoms: dict[str, int] = {}
    cur_key = None
    cur_resname: str | None = None
    cur_atoms: set[str] = set()

    def _flush() -> None:
        nonlocal cur_resname
        if cur_resname is None:
            return
        name = _resolve_histidine(cur_resname, cur_atoms)
        counts[name] = counts.get(name, 0) + 1
        if len(cur_atoms) > max_atoms.get(name, 0):
            max_atoms[name] = len(cur_atoms)

    for line in text.splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        if len(line) < 27:
            continue
        # resName spans cols 18-20 (3 chars) but MD tools extend to col 21 for
        # 4-char names (lipids/ligands); stripping cols 18-21 covers both.
        resname = line[17:21].strip().upper()
        if not resname:
            continue
        chain = line[21:22]
        resseq = line[22:26].strip()
        icode = line[26:27]
        atom = line[12:16].strip().upper()
        key = (chain, resseq, icode, resname)
        if key != cur_key or atom in cur_atoms:
            _flush()
            cur_key = key
            cur_resname = resname
            cur_atoms = set()
        cur_atoms.add(atom)
    _flush()

    components: list[Component] = []
    for resname, count in counts.items():
        atoms = max_atoms.get(resname, 0)
        kind, covered, suggestion = _classify(resname, atoms)
        components.append(
            Component(
                resname=resname,
                kind=kind,
                count=count,
                atom_count=atoms,
                covered=covered,
                suggestion=suggestion,
            )
        )
    # Covered components first, then gaps; alphabetical within each group.
    components.sort(key=lambda c: (not c.covered, c.kind, c.resname))
    return components


# Formal charge per residue at physiological pH for the components whose charge
# is fixed by residue name. Protonation-specific Amber names are explicit; plain
# HIS is treated as neutral (HIP is the charged form).
_RESIDUE_CHARGE = {
    # charged amino acids (Amber + CHARMM HSP)
    "ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1, "HIP": 1, "HSP": 1,
    "CYM": -1, "TYM": -1,
    # neutral protonation variants (explicit so they are not guessed)
    "ASH": 0, "GLH": 0, "LYN": 0, "HID": 0, "HIE": 0, "HIS": 0,
    "HSD": 0, "HSE": 0, "ARN": 0, "CYX": 0,
    # ions (Amber + CHARMM resnames)
    "NA": 1, "NA+": 1, "SOD": 1, "K": 1, "K+": 1, "POT": 1,
    "LI": 1, "LI+": 1, "LIT": 1, "RB": 1, "RB+": 1, "CS": 1, "CS+": 1, "CES": 1,
    "MG": 2, "MG2+": 2, "CA": 2, "CA2+": 2, "CAL": 2, "ZN": 2, "ZN2+": 2, "ZN2": 2,
    "CL": -1, "CL-": -1, "CLA": -1, "F": -1, "F-": -1,
    "BR": -1, "BR-": -1, "I": -1, "I-": -1, "IOD": -1,
}


@dataclass(frozen=True)
class ChargeEstimate:
    """Estimated formal charge of the components whose charge is known by name."""

    total: int                 # summed formal charge of standard/known components
    unknown: tuple[str, ...]   # gap resnames whose charge is not known from the PDB
    nucleic_residues: int      # nucleotides counted at -1 each (approximate)


def estimate_formal_charge(components: list[Component]) -> ChargeEstimate:
    """Estimate the net formal charge of a classified PDB.

    Standard protein ions/residues and monatomic ions use fixed formal charges;
    each nucleic-acid residue contributes -1 (one backbone phosphate, an estimate
    that ignores a 5'-OH terminus). Ligands, cofactors and non-standard metals
    carry no charge in a PDB, so they are reported separately — add their net
    charge (from the SDF / attached ITP) to get the true system charge.
    """
    total = 0
    unknown: list[str] = []
    nucleic = 0
    for comp in components:
        if comp.kind in ("dna", "rna"):
            nucleic += comp.count
            total += -1 * comp.count
        elif comp.resname in _RESIDUE_CHARGE:
            total += _RESIDUE_CHARGE[comp.resname] * comp.count
        elif comp.kind in ("protein", "water", "lipid"):
            continue  # neutral by name
        elif not comp.covered:
            unknown.append(comp.resname)
    return ChargeEstimate(total=total, unknown=tuple(unknown), nucleic_residues=nucleic)

