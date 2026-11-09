"""Assemble a Maestro-exported membrane system into an OpenFE-ready complex PDB.

A Maestro / Desmond (Schrodinger) membrane system exported straight to a single PDB
usually fails when handed to OpenFE (``ProteinMembraneComponent.from_pdb_file``):
water hydrogens are missing or inconsistent, ion/solvent bookkeeping differs, and
the merged topology confuses the parser. The reliable route is to export the
components separately, repair the solvent hydrogens, and merge them with OpenMM:

  1. In Maestro, split and export the system into separate PDBs:
       - ``protein.pdb``  (protein only)
       - ``membrane.pdb`` (lipids only)
       - ``sol.pdb``      (water + ions). Delete the water hydrogens and export it
                          as ``sol_noh.pdb``.
     Cofactors are exported to ``cofactors.sdf`` and kept separate -- OpenFE treats
     them as ``SmallMoleculeComponent``s, so they are NOT merged into the complex PDB.

  2. Re-add solvent hydrogens with PDBFixer -> ``fixed_sol.pdb``
     (equivalent to: ``pdbfixer sol_noh.pdb --output=fixed_sol.pdb --add-atoms=all``).

  3. Merge protein + membrane + fixed solvent into ``complex.pdb`` (OpenMM Modeller).

``complex.pdb`` is then the ``protein_pdb`` input for the OpenFE membrane workflow
(set ``membrane_system = true`` and ``complex_box_vectors`` / ``complex_box_size``
in the RBFE/ABFE/SepTop config).

See ``MEMBRANE_PREP.md`` in this directory for the full tutorial.

CLI::

    python -m chemflow.fe.openfe.membrane_assembly \
        --protein protein.pdb --membrane membrane.pdb --solvent sol_noh.pdb \
        --output complex.pdb [--fixed-solvent fixed_sol.pdb] [--ph 7.0] \
        [--solvent-has-hydrogens]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _imports():
    """Import OpenMM/PDBFixer lazily with a helpful error in the wrong env."""
    try:
        from openmm.app import Modeller, PDBFile  # noqa: PLC0415
        from pdbfixer import PDBFixer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "OpenMM/PDBFixer are required and were not importable. Run this inside "
            "the 'alchemforge' environment, e.g.:\n"
            "  conda run -n alchemforge python -m chemflow.fe.openfe.membrane_assembly ...\n"
            "or install them with:  conda install -c conda-forge openmm pdbfixer\n"
            f"(original import error: {exc})"
        ) from exc
    return PDBFile, Modeller, PDBFixer


def fix_solvent_hydrogens(
    sol_pdb: str | Path,
    out_pdb: str | Path,
    *,
    ph: float = 7.0,
) -> Path:
    """Re-add hydrogens (and any missing heavy atoms) to a water+ions PDB.

    Mirrors ``pdbfixer sol_noh.pdb --output=fixed_sol.pdb --add-atoms=all`` but via
    the PDBFixer Python API. Missing *residues* are deliberately not added -- only
    atoms within existing residues -- so the solvent/ion set is unchanged apart from
    the hydrogens.
    """
    PDBFile, _Modeller, PDBFixer = _imports()
    sol_pdb, out_pdb = Path(sol_pdb), Path(out_pdb)
    if not sol_pdb.is_file():
        raise FileNotFoundError(f"Solvent PDB not found: {sol_pdb}")

    fixer = PDBFixer(filename=str(sol_pdb))
    # Do not invent missing residues for solvent; only complete atoms + hydrogens.
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with out_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle)
    return out_pdb


def _load(pdb: str | Path):
    PDBFile, _Modeller, _PDBFixer = _imports()
    pdb = Path(pdb)
    if not pdb.is_file():
        raise FileNotFoundError(f"PDB not found: {pdb}")
    return PDBFile(str(pdb))


def assemble_complex(
    protein_pdb: str | Path,
    membrane_pdb: str | Path,
    solvent_pdb: str | Path,
    out_pdb: str | Path,
) -> Path:
    """Merge protein + membrane + solvent PDBs into one complex PDB with Modeller.

    The periodic box vectors are taken from the first input that carries them
    (membrane, then solvent, then protein), so the written ``complex.pdb`` keeps a
    CRYST1 record when the inputs have one.
    """
    PDBFile, Modeller, _PDBFixer = _imports()
    out_pdb = Path(out_pdb)

    protein = _load(protein_pdb)
    membrane = _load(membrane_pdb)
    solvent = _load(solvent_pdb)

    modeller = Modeller(protein.topology, protein.positions)
    modeller.add(membrane.topology, membrane.positions)
    modeller.add(solvent.topology, solvent.positions)

    # Preserve the periodic box from whichever input defines it.
    for source in (membrane, solvent, protein):
        box = source.topology.getPeriodicBoxVectors()
        if box is not None:
            modeller.topology.setPeriodicBoxVectors(box)
            break

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with out_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle)
    return out_pdb


def build_membrane_complex(
    protein_pdb: str | Path,
    membrane_pdb: str | Path,
    sol_noh_pdb: str | Path,
    out_pdb: str | Path,
    *,
    fixed_solvent: str | Path | None = None,
    ph: float = 7.0,
    solvent_has_hydrogens: bool = False,
) -> Path:
    """End-to-end: fix solvent hydrogens then assemble the complex PDB.

    If ``solvent_has_hydrogens`` is True the PDBFixer step is skipped and the
    solvent PDB is merged as-is.
    """
    out_pdb = Path(out_pdb)
    if solvent_has_hydrogens:
        solvent = Path(sol_noh_pdb)
    else:
        solvent = Path(fixed_solvent) if fixed_solvent else out_pdb.with_name("fixed_sol.pdb")
        solvent = fix_solvent_hydrogens(sol_noh_pdb, solvent, ph=ph)
        print(f"PDBFixer wrote solvent with hydrogens: {solvent}")

    complex_pdb = assemble_complex(protein_pdb, membrane_pdb, solvent, out_pdb)
    print(f"Assembled membrane complex: {complex_pdb}")
    return complex_pdb


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a Maestro-exported membrane system (protein + membrane + "
            "water/ions) into an OpenFE-ready complex PDB. Cofactors stay as a "
            "separate SDF."
        )
    )
    parser.add_argument("--protein", type=Path, required=True, help="protein-only PDB")
    parser.add_argument("--membrane", type=Path, required=True, help="lipids-only PDB")
    parser.add_argument(
        "--solvent", type=Path, required=True,
        help="water + ions PDB (sol_noh.pdb; hydrogens re-added unless --solvent-has-hydrogens)",
    )
    parser.add_argument("--output", type=Path, default=Path("complex.pdb"), help="merged complex PDB")
    parser.add_argument(
        "--fixed-solvent", type=Path, default=None,
        help="where to write the hydrogen-fixed solvent (default: fixed_sol.pdb next to --output)",
    )
    parser.add_argument("--ph", type=float, default=7.0, help="pH for addMissingHydrogens (default 7.0)")
    parser.add_argument(
        "--solvent-has-hydrogens", action="store_true",
        help="skip PDBFixer; the solvent PDB already has hydrogens",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_membrane_complex(
            args.protein,
            args.membrane,
            args.solvent,
            args.output,
            fixed_solvent=args.fixed_solvent,
            ph=args.ph,
            solvent_has_hydrogens=args.solvent_has_hydrogens,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
