"""Membrane relaxation protocol for OpenMM.

CHARMM-GUI-style staged equilibration of a PRE-TRIMMED membrane+protein(+ligand)
complex, run before OpenFE free-energy calculations. The input PDB's periodic box
is preserved as-is (no extra solvent/membrane padding is added), so a minimal
membrane patch stays minimal and the FEP system stays small.

  Protein restraint:
      k * periodicdistance(x, y, z, x0, y0, z0)^2
  Bilayer lipid head-group Z restraint:
      k * periodicdistance(0, 0, z, 0, 0, z0)^2

Initial energy minimization is unrestrained; restraints are added only for the
subsequent staged equilibration.

Configuration units:
  timestep / simulation_time : ps     temperature    : K
  force constants            : kJ/mol/nm^2            pressure : bar
  surface_tension            : bar*nm

Run headlessly with::

    python -m chemflow.fe.openmm_modeling.membrane_relax --config membrane_relax.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from sys import stdout
from types import SimpleNamespace

import numpy as np

from openmm import (
    CustomExternalForce,
    LangevinMiddleIntegrator,
    MonteCarloMembraneBarostat,
    Platform,
    VerletIntegrator,
)
from openmm.app import (
    CheckpointReporter,
    DCDReporter,
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
    PDBxFile,
    PME,
    Simulation,
    StateDataReporter,
)
from openmm.unit import (
    amu,
    bar,
    kelvin,
    kilojoules_per_mole,
    nanometer,
    nanometers,
    picosecond,
    picoseconds,
)

from chemflow.fe.openmm_modeling.membrane_relax_config import load_config, validate_config

logger = logging.getLogger("membrane_relaxation")


# ============================================================
# Logging
# ============================================================


def setup_logger(log_file=None, level=logging.INFO):
    """Configure console and optional file logging without duplicate handlers."""
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # The file is append-mode so successive runs share it; mark where each
        # one begins, otherwise repeated headers look like duplicated output.
        logger.info(
            "########## RUN START (pid %d) — appending to %s ##########",
            os.getpid(), log_file,
        )


# ============================================================
# Residue names
# ============================================================

PROTEIN_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "ASH", "GLH", "LYN", "CYX", "ACE", "NME", "NMA",
}

# Three-character lipid residue names used by the trimmed membrane topology.
LIPID_RESNAMES = {
    "POP",  # POPC / POPE represented as POP
    "DLP",  # DLPC / DLPE
    "DMP",  # DMPC
    "DOP",  # DOPC
    "DPP",  # DPPC
}

WATER_ION_RESNAMES = {
    "HOH", "WAT", "NA", "CL", "K", "CA", "MG",
}


# ============================================================
# Generic helpers
# ============================================================


# mmCIF has no 99,999-atom serial limit. A solvated membrane exported as PDB
# overflows it, and Maestro then writes hybrid-36 serials ("A000G") that OpenMM's
# PDB reader cannot parse (ValueError: invalid literal for int()). Residue numbers
# wrap past 9999 the same way, which is what produces the "duplicate atom HOH"
# warnings. Reading/writing mmCIF avoids both.
_CIF_SUFFIXES = {".cif", ".pdbx", ".mmcif"}
def fix_structure(modeller, add_hydrogens=True, ph=7.0):
    """Add missing hydrogens, including the extra N-terminal ones.

    A chain-terminal residue without them matches only the internal template and
    fails with "the set of externally bonded atoms is missing 1 N atom".

    Residue NAMES are left untouched: histidine protonation naming
    (HID/HIE/HIP) is the caller's business, handled upstream in Maestro/PDBFixer.

    Mutates ``modeller`` in place. Returns a short report dict.
    """
    added = 0
    if add_hydrogens:
        before = modeller.topology.getNumAtoms()
        try:
            # No forcefield argument: Modeller then uses its standard hydrogen
            # definitions and simply skips residues it has none for (lipids,
            # ligands), instead of failing on them.
            modeller.addHydrogens(pH=ph)
            added = modeller.topology.getNumAtoms() - before
        except Exception as exc:  # noqa: BLE001 - never block the build
            logger.warning("addHydrogens skipped (%s)", exc)

    if added:
        logger.warning("fix_structure: added %d missing hydrogen(s).", added)
    else:
        logger.info("fix_structure: no hydrogens needed adding.")
    return {"hydrogens_added": added}


def preflight_structure(conf):
    """Check a structure against the force field and report EVERY problem at once.

    ``createSystem`` aborts on the first unmatched residue, which turns preparing a
    system into one error per run. OpenMM's ``getUnmatchedResidues`` reports them
    all, so this surfaces missing hydrogens, split residues, broken peptide bonds
    and untemplated residues in a single pass.

    Returns a list of problem strings (empty == looks runnable).
    """
    from openmm.app import ForceField

    problems = []
    structure = read_structure(conf.input_pdb)
    topology = structure.topology

    # Apply the same auto-repair `run` will apply, so validate predicts the real
    # outcome instead of flagging things the build would have fixed anyway.
    if bool(getattr(conf, "fix_structure", True)):
        from openmm.app import Modeller

        modeller = Modeller(topology, structure.positions)
        fix_structure(
            modeller,
            add_hydrogens=bool(getattr(conf, "add_missing_hydrogens", True)),
            ph=float(getattr(conf, "ph", 7.0)),
        )
        topology = modeller.topology

    no_h, duplicates = _check_topology_hydrogens(topology)
    if duplicates:
        problems.append(
            f"{len(duplicates)} residue id(s) occur twice (atoms not contiguous -> "
            "residue split while reading)"
        )
    if no_h:
        problems.append(f"{len(no_h)} protein residue(s) have no hydrogens")

    if topology.getPeriodicBoxVectors() is None:
        problems.append(
            "no periodic box in the input (it will be guessed; pass box_vectors "
            "for an exact cell)"
        )

    ligand_sdf = getattr(conf, "ligand_sdf", None)
    try:
        forcefield = ForceField(*list(conf.forcefields))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"force fields could not be loaded: {exc}")
        return problems

    # Residues with no template. Ligand/cofactor residues are expected here when a
    # ligand_sdf is configured (their templates are generated at build time).
    unmatched = forcefield.getUnmatchedResidues(topology)
    if unmatched:
        counts = {}
        for res in unmatched:
            counts[res.name] = counts.get(res.name, 0) + 1
        listed = ", ".join(f"{n} x{c}" for n, c in sorted(counts.items()))
        problems.append(f"{len(unmatched)} residue(s) have no force-field template: {listed}")

        # Per-residue reason, using the same helper OpenMM uses in its error text.
        # Without this you only ever learn why the FIRST one failed.
        try:
            from openmm.app.forcefield import _findMatchErrors

            seen_reason = set()
            for res in unmatched[:12]:
                reason = _findMatchErrors(forcefield, res)
                key = (res.name, reason)
                if key in seen_reason:
                    continue
                seen_reason.add(key)
                problems.append(
                    f"  {res.name}{res.id}/{res.chain.id}: {reason}"
                )
        except Exception:  # noqa: BLE001 - private API, best effort only
            pass

        hints = []
        # A PDB round-trip is the usual cause of "no bonds between its atoms":
        # PDB carries bonds only via CONECT, and OpenMM's writer truncates residue
        # names to 3 characters (POPC -> POP), so the lipid no longer matches.
        reasons = " ".join(problems[-6:])
        if "no bonds between its atoms" in reasons or any(
            n in counts for n in ("POP", "DLP", "DMP", "DOP", "DPP")
        ):
            hints.append(
                "lipids have no bonds / are named POP instead of POPC — this is a PDB "
                "round-trip artefact. PDB stores bonds only in CONECT records and "
                "OpenMM truncates residue names to 3 characters. Use the .cif that "
                "build-membrane writes alongside the .pdb as the input here; mmCIF "
                "keeps both the 4-character names and the bonds."
            )
        if any(n in ("HIS",) for n in counts):
            hints.append(
                "HIS: Amber ff14SB has no 'HIS' template — histidines must be named "
                "HID / HIE / HIP by protonation state"
            )
        standard = {"HOH", "WAT", "NA", "CL", "K", "MG", "CA", "ZN"}
        unknown = [
            n for n in counts
            if n not in standard and not _looks_like_protein_or_lipid(n)
        ]
        if unknown:
            # Say definitively whether the configured SDF covers them, instead of
            # leaving the user to guess (preflight does not register ligand
            # templates, because that would trigger a slow charge calculation).
            covered, missing = set(), list(unknown)
            if ligand_sdf and Path(ligand_sdf).is_file():
                try:
                    from openff.toolkit import Molecule

                    mols = Molecule.from_file(
                        str(ligand_sdf), allow_undefined_stereo=True
                    )
                    if not isinstance(mols, list):
                        mols = [mols]
                    names = {(m.name or "").strip().upper() for m in mols}
                    covered = {n for n in unknown if n.upper() in names}
                    missing = [n for n in unknown if n not in covered]
                    problems.append(
                        f"  ligand_sdf '{Path(ligand_sdf).name}' contains "
                        f"{len(mols)} molecule(s): {', '.join(sorted(names)) or '(unnamed)'}"
                    )
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"  could not read ligand_sdf: {exc}")
            if covered:
                hints.append(
                    f"{', '.join(sorted(covered))}: covered by ligand_sdf — OK, these "
                    "get templates at build time"
                )
            if missing:
                hints.append(
                    f"{', '.join(missing)}: no SDF record with a MATCHING NAME. Note "
                    "the template generator matches by molecular GRAPH, not by name, "
                    "so an SDF molecule under a different name (e.g. 'LIG') can still "
                    "cover it — but if no record has the same atoms/bonds the build "
                    "will fail. Add one SDF record per molecule (correct bond orders, "
                    "formal charges, explicit H)."
                    + ("" if ligand_sdf else " ligand_sdf is NOT set.")
                )
        problems.extend(f"  hint: {h}" for h in hints)
    return problems


def _looks_like_protein_or_lipid(name):
    protein = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH",
        "LYN", "ACE", "NME", "NMA", "NH2",
    }
    lipid = {
        "POPC", "POPE", "POPG", "POPS", "POPI", "POPA", "DMPC", "DOPC", "DPPC",
        "DLPC", "DLPE", "DPPE", "DOPE", "DSPC", "POP", "DLP", "DMP", "DOP", "DPP",
    }
    return name in protein or name in lipid


def _check_topology_hydrogens(topology):
    """Warn about residues that will fail template matching, and say why.

    Two distinct causes produce "No template found ... missing N H atoms":
      * the source structure genuinely has no hydrogens on that residue, or
      * the residue got SPLIT while reading, so its H ended up in a second
        residue object with the same id (this happens when a residue's atoms are
        not contiguous in file order).
    Counting both tells them apart immediately.
    """
    protein = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH",
        "LYN", "ACE", "NME", "NMA", "NH2",
    }
    seen, duplicates, no_h = {}, [], []
    for res in topology.residues():
        key = (res.chain.id, str(res.id), res.name)
        if key in seen:
            # A PDB's resSeq column wraps at 9999, so a system with tens of
            # thousands of waters legitimately reuses ids. Only duplicates among
            # NON-water residues indicate a real split.
            if res.name not in ("HOH", "WAT"):
                duplicates.append(key)
        else:
            seen[key] = True
        if res.name in protein:
            atoms = list(res.atoms())
            if atoms and not any(
                a.element is not None and a.element.symbol == "H" for a in atoms
            ):
                no_h.append((res.index, res.chain.id, res.id, res.name, len(atoms)))

    if duplicates:
        logger.warning(
            "%d NON-WATER residue id(s) appear more than once, e.g. %s. Their atoms "
            "are probably not contiguous in the file, so the residue was split while "
            "reading (heavy atoms in one, hydrogens in another). Water ids repeating "
            "is normal and is not counted here — a PDB's resSeq column wraps at 9999.",
            len(duplicates), duplicates[:5],
        )
    if no_h:
        sample = ", ".join(
            f"{n}{i} (chain {c}, {k} atoms, residue index {idx})"
            for idx, c, i, n, k in no_h[:5]
        )
        logger.warning(
            "%d protein residue(s) have NO hydrogens: %s%s. OpenMM cannot match a "
            "template for these. If the duplicate-id warning above also fired this "
            "is a read/split problem; otherwise the input genuinely lacks hydrogens "
            "— add them in Maestro (Protein Preparation Wizard) before exporting.",
            len(no_h), sample, " ..." if len(no_h) > 5 else "",
        )

    # Peptide connectivity: a mid-chain residue must be bonded to BOTH neighbours.
    # Too few external bonds -> "the set of externally bonded atoms is missing
    # 1 N atom. Is the chain missing a terminal capping group?"
    external = {}
    for bond in topology.bonds():
        r1, r2 = bond[0].residue, bond[1].residue
        if r1.index != r2.index:
            external[r1.index] = external.get(r1.index, 0) + 1
            external[r2.index] = external.get(r2.index, 0) + 1
    prot_res = [r for r in topology.residues() if r.name in protein]
    loose = [r for r in prot_res if external.get(r.index, 0) == 0]
    ends = [r for r in prot_res if external.get(r.index, 0) == 1]
    if prot_res and (loose or len(ends) > 2 * _n_protein_chains(topology, protein)):
        logger.warning(
            "Peptide connectivity: %d protein residues; %d isolated, %d chain-end. "
            "Expected 2 chain-end residues per protein chain (%d chains -> %d); more "
            "than that means peptide bonds are missing. Examples: %s",
            len(prot_res), len(loose), len(ends),
            _n_protein_chains(topology, protein),
            2 * _n_protein_chains(topology, protein),
            ", ".join(f"{r.name}{r.id}/{r.chain.id}" for r in (loose + ends)[:6]) or "none",
        )
    return no_h, duplicates


def _n_protein_chains(topology, protein):
    return len({
        r.chain.id for r in topology.residues() if r.name in protein
    })


def _sanitize_cif(path, out_path):
    """Rewrite an mmCIF keeping only what OpenMM's PDBx reader needs.

    OpenMM ships a minimal parser that raises on any category it does not know
    (Maestro emits core-CIF items such as ``_chemical_name_common`` ->
    "Unrecogized syntax element"). It only ever consumes the unit cell and the
    ``_atom_site`` loop, so emit exactly those and drop the rest. Returns True if
    an ``_atom_site`` loop was found and written.
    """
    cell, headers, rows = [], [], []
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith(("_cell.", "_symmetry.")):
            cell.append(stripped)
            i += 1
            continue
        if stripped == "loop_":
            # Collect this loop's item names, then decide whether to keep it.
            j = i + 1
            names = []
            while j < n and lines[j].strip().startswith("_"):
                names.append(lines[j].strip())
                j += 1
            if names and names[0].startswith("_atom_site."):
                headers = names
                while j < n:
                    row = lines[j].strip()
                    # The loop ends at the next block/loop/comment marker.
                    if not row or row.startswith(("#", "loop_", "_", "data_")):
                        break
                    rows.append(row)
                    j += 1
            i = j
            continue
        i += 1

    if not headers or not rows:
        return False

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("data_chemflow\n#\n")
        for item in cell:
            handle.write(item + "\n")
        handle.write("#\nloop_\n")
        for name in headers:
            handle.write(name + "\n")
        for row in rows:
            handle.write(row + "\n")
        handle.write("#\n")
    return True


def _cif_atom_site_items(path, limit=40):
    """Return the ``_atom_site*`` loop item names declared in a CIF (either flavour)."""
    items = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                token = line.strip()
                if token.startswith("_atom_site"):
                    items.append(token.split()[0])
                    if len(items) >= limit:
                        break
                elif items and not token.startswith("_"):
                    break  # the item block ended
    except OSError:
        pass
    return items


def _read_cif_tolerant(path):
    """Read an mmCIF that OpenMM's bundled PDBx parser rejects.

    OpenMM ships a minimal PDBx reader that raises on any category it does not
    know (e.g. Maestro writes core-CIF items such as ``_chemical_name_common``,
    giving "Unrecogized syntax element"). Try a dependency-free sanitize first,
    then gemmi/ParmEd if they happen to be installed.
    """
    errors = []

    # 1. Strip unsupported categories and re-read with OpenMM (no extra deps).
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="chemflow-cif-"))
    tmp = scratch / "clean.cif"
    try:
        if _sanitize_cif(path, tmp):
            structure = PDBxFile(str(tmp))
            logger.info(
                "Read %s after stripping categories OpenMM's PDBx parser "
                "does not support.", path,
            )
            return structure
        errors.append("sanitize: no _atom_site loop found")
    except Exception as exc:  # noqa: BLE001 - fall through to the next reader
        errors.append(f"sanitize: {exc}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    try:  # gemmi: the most complete mmCIF reader, if the env has it
        import gemmi

        doc = gemmi.cif.read(str(path))
        st = gemmi.make_structure_from_block(doc.sole_block())
        st.setup_entities()
        tmp_pdb = Path(path).with_suffix(".openmm-tmp.pdb")
        st.write_pdb(str(tmp_pdb))
        try:
            return PDBFile(str(tmp_pdb))
        finally:
            tmp_pdb.unlink(missing_ok=True)
    except ImportError:
        errors.append("gemmi: not installed")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"gemmi: {exc}")

    try:  # ParmEd is already a dependency of the alchemforge environment
        import parmed

        structure = parmed.load_file(str(path))
        topology = structure.topology
        positions = structure.positions
        box_vectors = getattr(structure, "box_vectors", None)
        if box_vectors is not None:
            topology.setPeriodicBoxVectors(box_vectors)
        return SimpleNamespace(
            topology=topology,
            positions=positions,
            getPositions=lambda asNumpy=False: positions,  # noqa: N803 (OpenMM signature)
        )
    except ImportError:
        errors.append("parmed: not installed")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"parmed: {exc}")

    head = ""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            head = "".join(next(handle, "") for _ in range(8))
    except OSError:
        pass
    items = _cif_atom_site_items(path)
    core_cif = any(
        name.startswith("_atom_site_") and not name.startswith("_atom_site.")
        for name in items
    )
    if core_cif:
        raise ValueError(
            f"{path} is a CORE CIF (small-molecule/crystallographic flavour), not "
            "PDBx/mmCIF.\n"
            f"Its atom items are underscore-separated: {', '.join(items[:8])}"
            + (" ..." if len(items) > 8 else "")
            + "\nPDBx/mmCIF uses dots instead (_atom_site.label_comp_id, "
            "_atom_site.label_asym_id, ...).\n\n"
            "Core CIF carries no chain / residue / compound fields, so the protein, "
            "POPC and water identities needed to build and parameterize a system are "
            "simply not in this file — it cannot be used here.\n\n"
            "Re-export it as PDBx/mmCIF. In Maestro's File > Export Structures dialog "
            "pick the 'PDBx/mmCIF' file type explicitly (a plain '.cif' extension can "
            "give you this core-CIF flavour). From a script:\n"
            "    $SCHRODINGER/run python3 -c \"from schrodinger import structure; \\\n"
            "        st=structure.StructureReader.read('in.mae'); \\\n"
            "        structure.StructureWriter('out-mmcif.cif', format='mmcif').append(st)\"\n"
            "Your earlier complex.cif was valid PDBx/mmCIF, so the same export route "
            "will work for this one."
        )
    raise ValueError(
        f"Could not read the mmCIF {path}.\n"
        "OpenMM's PDBx parser rejected it and every fallback failed:\n  "
        + "\n  ".join(errors)
        + (f"\n\nAtom-site items found: {', '.join(items)}" if items else "")
        + "\n\nFirst lines of the file:\n"
        + head
        + "\nIf this is a small-molecule/chemical CIF rather than a system structure, "
        "point input_pdb at the structure instead. Otherwise re-export it from "
        "Maestro as PDBx/mmCIF, or install gemmi (conda install -c conda-forge gemmi)."
    )


_COVALENT_RADII_NM = {
    "H": 0.031, "B": 0.084, "C": 0.076, "N": 0.071, "O": 0.066, "F": 0.057,
    "SI": 0.111, "P": 0.107, "S": 0.105, "CL": 0.102, "SE": 0.120, "BR": 0.120,
    "I": 0.139,
}


def add_missing_intra_residue_bonds(topology, positions):
    """Give bonds to residues that have none, perceived by distance.

    Neither PDB (without CONECT) nor mmCIF (without ``_struct_conn``) carries
    connectivity for non-standard residues, and OpenMM only auto-bonds residues
    it has templates for. A lipid therefore arrives as a pile of unbonded atoms,
    which breaks every downstream consumer in a different way:

    * ``forcefield.createSystem`` -> "the set of atoms matches POPC, but the
      residue has no bonds between its atoms";
    * gufe's ProteinComponent/ProteinMembraneComponent -> "Unknown ion: C11 in
      residue POPC" (a zero-valence heavy atom looks like a monatomic ion).

    Bonds are perceived WITHIN each affected residue only, so no intermolecular
    contact can be mistaken for a bond. Returns the number of bonds added.
    """
    import numpy as np

    pos = np.asarray(positions.value_in_unit(nanometer))
    bonded = set()
    for bond in topology.bonds():
        bonded.add(bond[0].index)
        bonded.add(bond[1].index)

    added = 0
    touched = []
    for residue in topology.residues():
        atoms = list(residue.atoms())
        if len(atoms) < 2 or any(a.index in bonded for a in atoms):
            continue
        radii = [
            _COVALENT_RADII_NM.get(
                (a.element.symbol if a.element is not None else "").upper(), 0.0)
            for a in atoms
        ]
        for i in range(len(atoms)):
            if radii[i] <= 0:
                continue
            for j in range(i + 1, len(atoms)):
                if radii[j] <= 0:
                    continue
                cutoff = 1.3 * (radii[i] + radii[j])
                if float(np.linalg.norm(pos[atoms[i].index] - pos[atoms[j].index])) < cutoff:
                    topology.addBond(atoms[i], atoms[j])
                    added += 1
        touched.append(residue.name)

    if added:
        names = sorted(set(touched))
        logger.warning(
            "Added %d bond(s) by distance to %d residue(s) that had none (%s). "
            "PDB/mmCIF do not store connectivity for non-standard residues, and "
            "without it force-field template matching and gufe's component "
            "loaders both fail.",
            added, len(touched), ", ".join(names[:6]) + (" ..." if len(names) > 6 else ""),
        )
    return added


def read_structure(path, add_bonds=True):
    """Read a structure file: mmCIF (PDBx) or PDB, dispatched on the extension.

    ``add_bonds`` perceives intra-residue bonds for residues that arrived with
    none (lipids, ligands), which neither format stores for non-standard
    residues. Without it, template matching and gufe's loaders both fail.
    """
    path = Path(path)
    if path.suffix.lower() in _CIF_SUFFIXES:
        try:
            structure = PDBxFile(str(path))
        except Exception as exc:  # noqa: BLE001 - OpenMM raises a bare SyntaxError subclass
            logger.warning(
                "OpenMM's PDBx reader could not parse %s (%s); retrying with a "
                "tolerant mmCIF reader.", path, exc,
            )
            structure = _read_cif_tolerant(path)
    else:
        structure = PDBFile(str(path))

    if add_bonds:
        add_missing_intra_residue_bonds(structure.topology, structure.positions)
    return structure


def write_pdb(topology, positions, filename, keep_ids=True):
    """Write a structure, dispatching on extension (mmCIF -> PDBx).

    ``keep_ids=True`` preserves the input's chain/residue numbering; without it
    OpenMM renumbers every residue sequentially from 1.
    """
    path = Path(filename)
    writer = PDBxFile if path.suffix.lower() in _CIF_SUFFIXES else PDBFile
    with open(path, "w") as handle:
        writer.writeFile(topology, positions, handle, keepIds=keep_ids)


def normalize_namespace(value):
    """Return a plain dictionary for dict/SimpleNamespace-like objects."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, SimpleNamespace):
        return vars(value).copy()
    if hasattr(value, "__dict__"):
        return vars(value).copy()
    return {}


def as_int_steps(simulation_time_ps, timestep_ps):
    """Convert ps / ps into an integer number of integration steps."""
    if float(timestep_ps) <= 0.0:
        raise ValueError("timestep must be > 0 ps")
    return int(round(float(simulation_time_ps) / float(timestep_ps)))


def make_integrator(temperature_K, timestep_ps, friction_ps_inv, random_seed=None):
    integrator = LangevinMiddleIntegrator(
        float(temperature_K) * kelvin,
        float(friction_ps_inv) / picosecond,
        float(timestep_ps) * picoseconds,
    )
    if random_seed is not None:
        integrator.setRandomNumberSeed(int(random_seed))
    return integrator


def make_minimization_integrator(timestep_ps):
    """Integrator required by Simulation; unused for dynamics during minimization."""
    return VerletIntegrator(float(timestep_ps) * picoseconds)


def nested_force_constant(restraint_spec, key, default=0.0):
    raw = normalize_namespace(restraint_spec)
    value = raw.get(key, None)
    if value is None:
        return float(default)
    if isinstance(value, dict):
        value = value.get("force_constants", default)
    elif hasattr(value, "force_constants"):
        value = value.force_constants
    return float(value)


def nested_vector_force_constant(restraint_spec, key, component=2, default=0.0):
    raw = normalize_namespace(restraint_spec)
    value = raw.get(key, None)
    if value is None:
        return float(default)
    if isinstance(value, dict):
        value = value.get("force_constants", default)
    elif hasattr(value, "force_constants"):
        value = value.force_constants
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(
                f"{key}.force_constants must contain exactly three values [kx, ky, kz]"
            )
        return float(value[component])
    return float(value)


def stage_has_restraints(stage):
    restraints = getattr(stage, "restraints", None)
    return restraints is not None and restraints is not False


def get_platform(conf):
    platform_name = str(conf.platform)
    platform = Platform.getPlatformByName(platform_name)
    properties = {}
    if platform.getName().upper() == "CUDA":
        properties["Precision"] = "mixed"
        device_index = getattr(conf, "device_index", None)
        if device_index is not None:
            properties["DeviceIndex"] = str(device_index)
    logger.info("Using OpenMM platform: %s", platform.getName())
    logger.info("Platform properties: %s", properties)
    return platform, properties


# ============================================================
# CHARMM-GUI-style restraint forces
# ============================================================


def make_protein_restraint_force():
    """Protein XYZ positional restraint, matching CHARMM-GUI."""
    force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2;")
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    return force


def make_membrane_z_restraint_force(force_constant):
    """Bilayer Z-only positional restraint, matching CHARMM-GUI.

    ``k`` is per-particle, not global. OpenMM requires a global parameter of a
    given name to carry the same default in every Force, so two restraint
    forces each declaring a global ``k`` at different strengths abort context
    creation with "Two Forces define different default values for the
    parameter 'k'". The protein restraints already do it this way.
    """
    force = CustomExternalForce("k*periodicdistance(0, 0, z, 0, 0, z0)^2;")
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("z0")
    return force


def is_protein_backbone_atom(atom):
    return atom.name.strip().upper() in {"N", "CA", "C", "O"}


def is_membrane_head_atom(atom):
    """Approximate CHARMM-GUI lipid head-group selection using P/N atoms."""
    atom_name = atom.name.strip().upper()
    return atom_name.startswith("P") or atom_name.startswith("N")


def add_protein_restraints(system, topology, reference_positions, restraint_spec):
    """Add separate protein backbone and sidechain XYZ restraints."""
    fc_bb = nested_force_constant(restraint_spec, "backbone")
    fc_sc = nested_force_constant(restraint_spec, "sidechain")

    legacy = nested_force_constant(restraint_spec, "solute_heavy_atom")
    if fc_bb <= 0.0 and fc_sc <= 0.0 and legacy > 0.0:
        fc_bb = legacy
        fc_sc = legacy

    if fc_bb <= 0.0 and fc_sc <= 0.0:
        return system

    force = make_protein_restraint_force()
    n_bb = 0
    n_sc = 0

    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol == "H":
            continue
        resname = atom.residue.name.strip().upper()
        if resname not in PROTEIN_RESNAMES:
            continue
        if is_protein_backbone_atom(atom):
            k = fc_bb
            if k <= 0.0:
                continue
            n_bb += 1
        else:
            k = fc_sc
            if k <= 0.0:
                continue
            n_sc += 1
        pos = reference_positions[atom.index]
        force.addParticle(atom.index, [k, pos.x, pos.y, pos.z])

    if force.getNumParticles() > 0:
        system.addForce(force)

    logger.info(
        'Restraint "protein_backbone": %d atoms, k = %.1f kJ/mol/nm^2, XYZ',
        n_bb, fc_bb,
    )
    logger.info(
        'Restraint "protein_sidechain": %d atoms, k = %.1f kJ/mol/nm^2, XYZ',
        n_sc, fc_sc,
    )
    return system


def add_ligand_restraints(system, topology, reference_positions, restraint_spec):
    """Add XYZ positional restraints to ligand heavy atoms."""
    k = nested_force_constant(restraint_spec, "ligand")
    if k <= 0.0:
        return system

    force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2;")
    # Per-particle k: a global one would clash with the membrane-Z restraint,
    # which declares the same name at a different strength.
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    count = 0
    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol == "H":
            continue
        resname = atom.residue.name.strip().upper()
        if resname in PROTEIN_RESNAMES:
            continue
        if resname in LIPID_RESNAMES:
            continue
        if resname in WATER_ION_RESNAMES:
            continue
        pos = reference_positions[atom.index]
        force.addParticle(atom.index, [k, pos.x, pos.y, pos.z])
        count += 1

    if count > 0:
        system.addForce(force)

    logger.info(
        'Restraint "ligand": %d atoms, k = %.1f kJ/mol/nm^2, XYZ', count, k,
    )
    return system


def add_membrane_z_restraint(system, topology, reference_positions, restraint_spec, key):
    """Add a Z-only CHARMM-GUI-style restraint to lipid head-group atoms."""
    k = nested_vector_force_constant(restraint_spec, key, component=2)
    if k <= 0.0:
        return system

    force = make_membrane_z_restraint_force(k)
    count = 0
    for atom in topology.atoms():
        resname = atom.residue.name.strip().upper()
        if resname not in LIPID_RESNAMES:
            continue
        if not is_membrane_head_atom(atom):
            continue
        pos = reference_positions[atom.index]
        force.addParticle(atom.index, [k, pos.z])
        count += 1

    if count > 0:
        system.addForce(force)

    logger.info(
        'Restraint "%s_Z": %d atoms, k = %.1f kJ/mol/nm^2, Z', key, count, k,
    )
    if count == 0:
        logger.warning('Restraint group "%s" selected 0 lipid head-group atoms.', key)
    return system


def add_stage_restraints(system, topology, reference_positions, restraint_spec):
    """Add exactly the restraint groups present in the stage configuration."""
    if restraint_spec is None or restraint_spec is False:
        logger.info("Restraints: none")
        return system

    raw = normalize_namespace(restraint_spec)
    if not raw:
        logger.info("Restraints: none")
        return system

    if any(key in raw for key in ("backbone", "sidechain", "solute_heavy_atom")):
        system = add_protein_restraints(
            system, topology, reference_positions, restraint_spec
        )
    if "ligand" in raw:
        system = add_ligand_restraints(
            system, topology, reference_positions, restraint_spec
        )
    if "membrane" in raw:
        system = add_membrane_z_restraint(
            system, topology, reference_positions, restraint_spec, "membrane"
        )
    if "membrane_head" in raw:
        system = add_membrane_z_restraint(
            system, topology, reference_positions, restraint_spec, "membrane_head"
        )
    return system


# ============================================================
# Diagnostics and stage logging
# ============================================================


def report_system_forces(system):
    logger.info("System has %d forces", system.getNumForces())
    for i in range(system.getNumForces()):
        logger.info("Force %d: %s", i, type(system.getForce(i)).__name__)


def report_state_diagnostics(simulation, label="state"):
    state = simulation.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, getForces=True,
    )
    positions = state.getPositions(asNumpy=True).value_in_unit(nanometers)
    forces = state.getForces(asNumpy=True).value_in_unit(
        kilojoules_per_mole / nanometer
    )
    magnitudes = np.linalg.norm(forces, axis=1)

    logger.info("%s potential energy: %s", label, state.getPotentialEnergy())
    logger.info("%s maximum force: %.6f kJ/mol/nm", label, magnitudes.max())
    logger.info("%s max-force atom: %d", label, int(magnitudes.argmax()))
    logger.info("%s NaN coordinates: %s", label, np.isnan(positions).any())
    logger.info("%s infinite coordinates: %s", label, np.isinf(positions).any())
    return state


def test_first_steps(simulation, n=1):
    simulation.step(int(n))
    logger.info("%d step(s) completed successfully", int(n))


def log_stage_info(stage, n_steps=None, temperature=None, duration_ps=None):
    """Log ensemble, timestep, temperature, duration, pressure and restraints."""
    if temperature is None:
        temperature = getattr(stage, "temperature", None)
    if duration_ps is None:
        duration_ps = getattr(stage, "simulation_time", None)
    if n_steps is None and duration_ps is not None:
        n_steps = as_int_steps(duration_ps, stage.timestep)

    logger.info("=" * 60)
    logger.info("%s", stage.title)
    logger.info("=" * 60)
    logger.info(
        "Ensemble: %s; timestep: %.4f ps; temperature: %.2f K; "
        "simulation time: %.1f ps; steps: %d",
        getattr(stage, "ensemble", "N/A"),
        float(stage.timestep),
        float(temperature) if temperature is not None else float("nan"),
        float(duration_ps) if duration_ps is not None else float("nan"),
        int(n_steps) if n_steps is not None else 0,
    )

    if str(getattr(stage, "ensemble", "")).upper() == "NPGT":
        logger.info(
            "Pressure: %.3f bar; surface tension: %.3f bar*nm",
            float(stage.pressure),
            float(stage.surface_tension),
        )

    restraints = getattr(stage, "restraints", None)
    if restraints is None or restraints is False:
        logger.info("Restraints: none")
        return
    restraint_dict = normalize_namespace(restraints)
    if not restraint_dict:
        logger.info("Restraints: none")
        return

    for name, config in restraint_dict.items():
        if isinstance(config, dict):
            fc = config.get("force_constants")
        elif hasattr(config, "force_constants"):
            fc = config.force_constants
        else:
            fc = config
        if isinstance(fc, (list, tuple)):
            logger.info(
                "Restraint: %s; k: X=%.1f, Y=%.1f, Z=%.1f kJ/mol/nm^2",
                name, float(fc[0]), float(fc[1]), float(fc[2]),
            )
        else:
            logger.info(
                "Restraint: %s; k: %.1f kJ/mol/nm^2", name, float(fc),
            )


# ============================================================
# System creation
# ============================================================


def create_base_system(forcefield, topology):
    """Create the base PME System without a barostat."""
    if not isinstance(forcefield, ForceField):
        raise TypeError(
            "forcefield must be an openmm.app.ForceField instance, "
            f"not {type(forcefield)!r}."
        )
    return forcefield.createSystem(
        topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
        rigidWater=True,
        hydrogenMass=3.0 * amu,
    )


def add_membrane_barostat(system, stage, random_seed=None):
    pressure = getattr(stage, "pressure", None)
    surface_tension = getattr(stage, "surface_tension", None)
    if pressure is None or surface_tension is None:
        return system
    barostat = MonteCarloMembraneBarostat(
        float(pressure) * bar,
        float(surface_tension) * bar * nanometer,
        float(stage.temperature) * kelvin,
        MonteCarloMembraneBarostat.XYIsotropic,
        MonteCarloMembraneBarostat.ZFree,
    )
    if random_seed is not None:
        barostat.setRandomNumberSeed(int(random_seed))
    system.addForce(barostat)
    return system


def build_simulation(
    modeller,
    forcefield,
    stage,
    conf,
    platform,
    platform_properties,
    reference_positions=None,
    add_restraints=False,
    add_barostat=False,
    velocities=None,
):
    """Build a fresh OpenMM Simulation from a clean System."""
    system = create_base_system(forcefield, modeller.topology)

    if add_restraints:
        system = add_stage_restraints(
            system, modeller.topology, reference_positions,
            getattr(stage, "restraints", None),
        )
    if add_barostat:
        system = add_membrane_barostat(
            system, stage, random_seed=getattr(conf, "seed", None)
        )

    integrator = make_integrator(
        stage.temperature, stage.timestep, conf.friction, random_seed=conf.seed,
    )
    simulation = Simulation(
        modeller.topology, system, integrator, platform, platform_properties,
    )
    if reference_positions is not None:
        simulation.context.setPositions(reference_positions)
    if velocities is not None:
        simulation.context.setVelocities(velocities)

    logger.info(
        "Actual simulation platform: %s",
        simulation.context.getPlatform().getName(),
    )
    report_system_forces(system)
    return simulation


def attach_reporters(
    simulation,
    prefix,
    report_interval=5000,
    checkpoint_interval=50000,
    total_steps=None,
    stdout_log=True,
):
    simulation.reporters.append(DCDReporter(f"{prefix}.dcd", report_interval))
    simulation.reporters.append(
        CheckpointReporter(f"{prefix}.chk", checkpoint_interval)
    )
    if stdout_log:
        simulation.reporters.append(
            StateDataReporter(
                stdout, report_interval, step=True, time=True,
                potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                temperature=True, volume=True, density=True,
                progress=(total_steps is not None),
                remainingTime=(total_steps is not None),
                speed=True,
                totalSteps=total_steps if total_steps is not None else 0,
                separator=",",
            )
        )
    simulation.reporters.append(
        StateDataReporter(
            f"{prefix}.csv", report_interval, step=True, time=True,
            potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
            temperature=True, volume=True, density=True, separator=",",
        )
    )


# ============================================================
# Input assembly (preserve the trimmed membrane box; no padding)
# ============================================================


def _ensure_ambertools_on_path():
    """Make AmberTools' executables visible, returning True if they are.

    OpenFF/openmmforcefields shell out to AmberTools by NAME — ``sqm`` for AM1-BCC
    charges and ``antechamber`` for GAFF atom typing. Both are found via $PATH, not
    by importing anything. Running the interpreter by absolute path (e.g.
    ``/…/envs/alchemforge/bin/python -m …``, which is what the staged runner
    scripts and the desktop do) leaves the env's bin/ off $PATH, so AmberTools
    silently fails and you get "No registered toolkits ..." or
    "antechamber: command not found". Put the interpreter's own bin/ on $PATH.
    """
    import os
    import shutil

    needed = ("sqm", "antechamber")
    if all(shutil.which(tool) for tool in needed):
        return True
    bindir = Path(sys.executable).resolve().parent
    if any((bindir / tool).exists() for tool in needed):
        os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
        logger.info("Added %s to PATH so AmberTools is detectable.", bindir)
    missing = [tool for tool in needed if not shutil.which(tool)]
    if missing:
        logger.warning(
            "AmberTools executable(s) still not on PATH: %s. GAFF needs antechamber "
            "and AM1-BCC needs sqm. Activate the environment (conda activate "
            "alchemforge) or install AmberTools (conda install -c conda-forge "
            "ambertools).", ", ".join(missing),
        )
    return not missing


def _assign_partial_charges(molecules, method="auto"):
    """Give every molecule partial charges before template generation.

    openmmforcefields' GAFF/SMIRNOFF template generators call
    ``assign_partial_charges(partial_charge_method="am1bcc")`` themselves, which
    needs AmberTools (sqm) or OpenEye. If neither is registered the run dies with
    "No registered toolkits can provide the capability assign_partial_charges".
    Charging the molecules up front avoids that: the generators reuse existing
    charges. Order: charges already in the SDF > AM1-BCC > NAGL (a GNN surrogate
    for AM1-BCC, much faster and dependency-light).
    """
    pending = [m for m in molecules if m.partial_charges is None]
    if not pending:
        logger.info("All %d ligand(s) already carry partial charges from the SDF.",
                    len(molecules))
        return

    wanted = str(method or "auto").lower()
    errors = []

    if wanted in ("auto", "am1bcc"):
        have_sqm = _ensure_ambertools_on_path()
        try:
            registry = None
            if have_sqm:
                # Force the AmberTools wrapper: the GLOBAL registry was built at
                # import time, possibly before sqm was on PATH.
                from openff.toolkit.utils.toolkits import AmberToolsToolkitWrapper

                registry = AmberToolsToolkitWrapper()
            for mol in pending:
                if registry is not None:
                    mol.assign_partial_charges(
                        partial_charge_method="am1bcc", toolkit_registry=registry
                    )
                else:
                    mol.assign_partial_charges(partial_charge_method="am1bcc")
            logger.info(
                "Assigned AM1-BCC charges to %d ligand(s) via AmberTools.", len(pending)
            )
            return
        except Exception as exc:  # noqa: BLE001 - toolkit availability varies
            errors.append(f"am1bcc: {exc}")

    if wanted in ("auto", "nagl"):
        try:
            from openff.nagl_models import list_available_nagl_models
            from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

            candidates = [str(p) for p in list_available_nagl_models()]
            am1bcc_models = [p for p in candidates if "am1bcc" in Path(p).name.lower()]
            if not am1bcc_models:
                raise RuntimeError(f"no am1bcc NAGL model among {candidates}")
            model = sorted(am1bcc_models)[-1]
            wrapper = NAGLToolkitWrapper()
            for mol in pending:
                mol.assign_partial_charges(
                    partial_charge_method=model, toolkit_registry=wrapper
                )
            logger.warning(
                "AmberTools/OpenEye unavailable, so AM1-BCC could not be run. "
                "Used the NAGL GNN surrogate (%s) for %d ligand(s) instead. Install "
                "AmberTools (conda install -c conda-forge ambertools) for true "
                "AM1-BCC charges.", Path(model).name, len(pending),
            )
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"nagl: {exc}")

    raise ValueError(
        "Could not assign partial charges to the ligand(s):\n  "
        + "\n  ".join(errors)
        + "\n\nAM1-BCC needs AmberTools (provides sqm) or OpenEye. Fix the "
        "environment:\n"
        "    conda install -c conda-forge ambertools\n"
        "    python -c \"from openff.toolkit.utils.toolkits import "
        'AmberToolsToolkitWrapper as A; print(A.is_available())"\n'
        "Or pre-charge the ligand SDF yourself and the charges in it will be used."
    )


def _output_positions(simulation):
    """Positions to WRITE: wrapped into the box, with the box of THIS frame.

    Two things go wrong otherwise, and both reproduce the OpenFE complex-leg
    step-0 NaN downstream:

    * Raw context positions are UNWRAPPED — molecules diffuse out of the cell
      during MD and OpenMM never wraps them unless asked. ``enforcePeriodicBox``
      images them back, keeping each molecule whole.
    * Under a barostat the box CHANGES EVERY FRAME, but ``PDBFile.writeFile``
      takes CRYST1 from the topology, which still holds the box set at build
      time. Copy the state's current box onto the topology first so the written
      CRYST1 matches the coordinates it describes.
    """
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    simulation.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    return state.getPositions()


def _write_charged_sdf(molecules, ligand_sdf, conf):
    """Write the charged ligands to ``<workdir>/<stem>_charged.sdf``.

    AM1-BCC is slow, so persist the result: point ``ligand_sdf`` at this file on a
    later run and the charges in it are reused verbatim (no recomputation). It is
    also the record of exactly which charges the build used.
    """
    stem = Path(ligand_sdf).stem
    workdir = Path(getattr(conf, "workdir", ".") or ".").expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / f"{stem}_charged.sdf"

    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="chemflow-sdf-"))
    try:
        chunks = []
        for index, mol in enumerate(molecules):
            # One record at a time, then concatenate: an SDF is just records
            # separated by "$$$$", and this avoids relying on append support.
            piece = scratch / f"{index}.sdf"
            mol.to_file(str(piece), file_format="SDF")
            chunks.append(piece.read_text(encoding="utf-8"))
        out.write_text("".join(chunks), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - never fail the build over a side file
        logger.warning("Could not write the charged SDF (%s): %s", out, exc)
        return None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    logger.info(
        "Wrote charged ligands -> %s (reuse it as ligand_sdf to skip recharging)", out
    )
    return out


def _register_small_molecule(forcefield, ligand_sdf, conf):
    """Register a template generator so ligand atoms in the PDB are parameterized."""
    try:
        from openff.toolkit import Molecule
        from openmmforcefields.generators import (
            GAFFTemplateGenerator,
            SMIRNOFFTemplateGenerator,
        )
    except ImportError as error:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "Parameterizing a ligand needs 'openff-toolkit' and "
            "'openmmforcefields'. Install them, or drop 'ligand_sdf' and supply "
            "a ligand force-field XML in 'forcefields' instead."
        ) from error

    molecules = Molecule.from_file(str(ligand_sdf), allow_undefined_stereo=True)
    if not isinstance(molecules, list):
        molecules = [molecules]

    # GAFF needs `antechamber` for ATOM TYPING, regardless of charges, and OpenFF
    # finds AmberTools by looking for its executables on $PATH. Running python by
    # absolute path without activating the env leaves the env's bin/ off $PATH, so
    # repair it here — not only in the charging branch, which is skipped entirely
    # when the SDF already carries partial charges.
    _ensure_ambertools_on_path()

    # Charge up front so the template generator never has to call AM1-BCC itself.
    _assign_partial_charges(
        molecules, getattr(conf, "partial_charge_method", "auto")
    )
    _write_charged_sdf(molecules, ligand_sdf, conf)

    ff_name = str(getattr(conf, "small_molecule_forcefield", "gaff-2.11"))
    lowered = ff_name.lower()
    if lowered.startswith(("openff", "smirnoff", "sage")):
        generator = SMIRNOFFTemplateGenerator(
            molecules=molecules, forcefield=ff_name
        )
    else:
        generator = GAFFTemplateGenerator(molecules=molecules, forcefield=ff_name)

    forcefield.registerTemplateGenerator(generator.generator)
    logger.info(
        "Registered %d ligand template(s) via %s", len(molecules), ff_name
    )


def build_system_inputs(conf):
    """Build a Modeller + ForceField from the config, preserving the input box."""
    input_pdb = Path(conf.input_pdb)
    pdb = read_structure(input_pdb)
    modeller = Modeller(pdb.topology, pdb.positions)

    box = pdb.topology.getPeriodicBoxVectors()
    if box is None:
        raise ValueError(
            f"Input PDB has no periodic box (missing CRYST1): {input_pdb}. "
            "A trimmed membrane system must carry its own box; this protocol "
            "does not add solvent/membrane padding."
        )
    logger.info(
        "Preserving input periodic box (no padding added): %s",
        box,
    )

    forcefield = ForceField(*list(conf.forcefields))
    ligand_sdf = getattr(conf, "ligand_sdf", None)
    if ligand_sdf:
        _register_small_molecule(forcefield, ligand_sdf, conf)

    return modeller, forcefield


# ============================================================
# Main relaxation workflow
# ============================================================


def relax(modeller, forcefield, conf, debug_first_step=False):
    """Run the configured membrane relaxation protocol."""
    if not isinstance(forcefield, ForceField):
        raise TypeError(f"forcefield must be ForceField, got {type(forcefield)!r}")
    if not hasattr(conf, "friction"):
        raise AttributeError("conf must define friction")

    workdir = Path(getattr(conf, "workdir", ".")).expanduser().resolve()
    if workdir.exists() and not workdir.is_dir():
        raise NotADirectoryError(f"Working directory is not a directory: {workdir}")
    # Create the job folder (workdir is <workdir>/<job_name> when a job name is set).
    workdir.mkdir(parents=True, exist_ok=True)

    setup_logger(log_file=workdir / "membrane_relaxation.log")
    logger.info("Working directory: %s", workdir)

    platform, platform_properties = get_platform(conf)

    # ---- protocol: equilibrated -> minimization -> NVT -> NPgT (no cold-start ladder)
    protocol = str(getattr(conf, "protocol", "full")).strip().lower()
    if protocol in ("equilibrated", "equil", "two_stage", "two-stage", "nvt_nprt", "nvt+nprt"):
        _run_equilibrated_two_stage(
            conf, modeller, forcefield, platform, platform_properties, workdir,
            debug_first_step=debug_first_step,
        )
        return

    # -------------------------------------------------- unrestrained minimization
    minimization = conf.minimization
    stage1 = conf.stage1

    logger.info("=" * 60)
    logger.info("%s: %d minimization steps", minimization.title, int(minimization.steps))
    logger.info("Initial energy minimization is UNRESTRAINED")
    logger.info("=" * 60)

    system = create_base_system(forcefield, modeller.topology)
    minimization_integrator = make_minimization_integrator(stage1.timestep)
    simulation = Simulation(
        modeller.topology, system, minimization_integrator,
        platform, platform_properties,
    )
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=int(minimization.steps))

    minimized_state = simulation.context.getState(
        getPositions=True, getEnergy=True, getForces=True,
    )
    report_state_diagnostics(simulation, "After minimization")
    minimized_positions = minimized_state.getPositions()
    minimized_pdb = workdir / minimization.out_pdb
    write_pdb(simulation.topology, _output_positions(simulation), minimized_pdb)
    logger.info("Saved PDB after unrestrained minimization: %s", minimized_pdb)

    # -------------------------------------------------- Stage 1: restrained NVT
    stage = stage1
    stage_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=stage_steps)

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=minimized_positions,
        add_restraints=True, add_barostat=False,
    )
    simulation.context.setVelocitiesToTemperature(float(stage.temperature) * kelvin)

    if debug_first_step:
        report_state_diagnostics(simulation, "Before Stage 1 NVT")
        test_first_steps(simulation, 1)

    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=stage_steps, stdout_log=True,
    )
    logger.info("Running Stage 1 NVT...")
    simulation.step(stage_steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    write_pdb(simulation.topology, _output_positions(simulation), workdir / stage.out_pdb)

    # -------------------------------------------------- Stage 2: restrained NPgT
    stage = conf.stage2
    stage_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=stage_steps)
    previous_velocities = state.getVelocities()

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=state.getPositions(),
        add_restraints=True, add_barostat=True, velocities=previous_velocities,
    )
    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=stage_steps, stdout_log=True,
    )
    logger.info("Running Stage 2 restrained NPgT...")
    simulation.step(stage_steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    write_pdb(simulation.topology, _output_positions(simulation), workdir / stage.out_pdb)

    # -------------------------------------------------- Stage 3: restrained NPgT
    stage = conf.stage3
    stage_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=stage_steps)

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=state.getPositions(),
        add_restraints=True, add_barostat=True, velocities=state.getVelocities(),
    )
    if state.getVelocities() is not None:
        simulation.context.setVelocities(state.getVelocities())

    logger.info("Stage 3: short restrained minimization")
    simulation.minimizeEnergy(maxIterations=500)
    report_state_diagnostics(simulation, "Stage 3 after restrained minimization")

    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=stage_steps, stdout_log=True,
    )
    logger.info("Running Stage 3 restrained NPgT...")
    simulation.step(stage_steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    write_pdb(simulation.topology, _output_positions(simulation), workdir / stage.out_pdb)

    # -------------------------------------------------- Stage 4: NPgT anneal schedule
    stage = conf.stage4
    schedule = stage.temperature_schedule

    logger.info("=" * 60)
    logger.info("%s", stage.title)
    logger.info("=" * 60)
    logger.info("Stage 4 consists of %d temperature segments", len(schedule))

    for schedule_entry in schedule:
        temperature = float(schedule_entry.temperature)
        duration_ps = float(schedule_entry.duration)
        steps = as_int_steps(duration_ps, stage.timestep)
        logger.info(
            "Stage 4 segment: ensemble=%s; timestep=%.4f ps; temperature=%.2f K; "
            "duration=%.1f ps; steps=%d",
            stage.ensemble, stage.timestep, temperature, duration_ps, steps,
        )
        segment_stage = SimpleNamespace(**vars(stage))
        segment_stage.temperature = temperature
        log_stage_info(
            segment_stage, n_steps=steps,
            temperature=temperature, duration_ps=duration_ps,
        )
        simulation = build_simulation(
            modeller, forcefield, segment_stage, conf, platform, platform_properties,
            reference_positions=state.getPositions(),
            add_restraints=True, add_barostat=True, velocities=state.getVelocities(),
        )
        attach_reporters(
            simulation, prefix=str(workdir / f"{stage.name}_{temperature:g}K"),
            total_steps=steps, stdout_log=True,
        )
        logger.info("Running Stage 4 segment at %.2f K...", temperature)
        simulation.step(steps)
        state = simulation.context.getState(getPositions=True, getVelocities=True)

    write_pdb(simulation.topology, _output_positions(simulation), workdir / stage.out_pdb)

    # -------------------------------------------------- Stage 5: restrained NVT
    stage = conf.stage5
    stage_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=stage_steps)

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=state.getPositions(),
        add_restraints=True, add_barostat=False, velocities=state.getVelocities(),
    )
    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=stage_steps, stdout_log=True,
    )
    logger.info("Running Stage 5 restrained NVT...")
    simulation.step(stage_steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    write_pdb(simulation.topology, _output_positions(simulation), workdir / stage.out_pdb)

    # -------------------------------------------------- Stage 6: unrestrained NVT
    stage = conf.stage6
    stage_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=stage_steps)

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=state.getPositions(),
        add_restraints=False, add_barostat=False, velocities=state.getVelocities(),
    )
    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=stage_steps, stdout_log=True,
    )
    logger.info("Running Stage 6 unrestrained NVT...")
    simulation.step(stage_steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    write_pdb(simulation.topology, _output_positions(simulation), workdir / stage.out_pdb)

    # -------------------------------------------------- Production: unrestrained NPgT
    stage = conf.simulation
    prod_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=prod_steps)

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=state.getPositions(),
        add_restraints=False, add_barostat=True, velocities=state.getVelocities(),
    )
    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=prod_steps, stdout_log=True,
    )
    logger.info("Running unrestrained production NPgT...")
    simulation.step(prod_steps)
    final_state = simulation.context.getState(getPositions=True, getVelocities=True)

    production_pdb = workdir / stage.out_pdb
    write_pdb(simulation.topology, _output_positions(simulation), production_pdb)
    # Also write the final relaxed system as mmCIF (OpenFE-ready; no PDB
    # >99,999-atom serial overflow for large membrane systems).
    from openmm.app import PDBxFile

    production_cif = production_pdb.with_suffix(".cif")
    with open(production_cif, "w") as handle:
        PDBxFile.writeFile(simulation.topology, _output_positions(simulation), handle, keepIds=True)
    simulation.saveState(str(workdir / stage.out_state))
    simulation.saveCheckpoint(str(workdir / stage.chk))

    logger.info("Saved final relaxed PDB: %s", production_pdb)
    logger.info("Saved final relaxed mmCIF: %s", production_cif)
    logger.info("Saved final state: %s", workdir / stage.out_state)
    logger.info("Saved final checkpoint: %s", workdir / stage.chk)
    logger.info("Membrane relaxation protocol completed successfully.")
    return production_pdb


def _run_equilibrated_two_stage(
    conf, modeller, forcefield, platform, platform_properties, workdir,
    debug_first_step=False,
):
    """Two-stage protocol for an ALREADY-equilibrated membrane: min -> NVT -> NPgT.

    No cold-start/heating ladder — a short NVT settles velocities/water around the
    (optionally restrained) solute, then a tensionless NPgT lets the box/area
    re-settle. Uses conf.equil_nvt and conf.equil_nprt.
    """
    nvt = conf.equil_nvt
    nprt = conf.equil_nprt

    logger.info("=" * 60)
    logger.info("Equilibrated-system protocol: minimization -> NVT -> NPgT")
    logger.info("=" * 60)

    # -- brief minimization (clears any clashes from re-solvation / wrapping) --
    system = create_base_system(forcefield, modeller.topology)
    min_integrator = make_minimization_integrator(nvt.timestep)
    simulation = Simulation(
        modeller.topology, system, min_integrator, platform, platform_properties,
    )
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=int(conf.minimization.steps))
    report_state_diagnostics(simulation, "After minimization")
    min_positions = simulation.context.getState(getPositions=True).getPositions()
    write_pdb(simulation.topology, _output_positions(simulation), workdir / conf.minimization.out_pdb)

    # -- Stage 1: NVT --
    stage = nvt
    steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=steps)
    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=min_positions,
        add_restraints=bool(getattr(stage, "restraints", None)), add_barostat=False,
    )
    simulation.context.setVelocitiesToTemperature(float(stage.temperature) * kelvin)
    if debug_first_step:
        report_state_diagnostics(simulation, "Before NVT")
        test_first_steps(simulation, 1)
    attach_reporters(
        simulation, prefix=str(workdir / stage.name), total_steps=steps, stdout_log=True,
    )
    logger.info("Running NVT...")
    simulation.step(steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    write_pdb(simulation.topology, _output_positions(simulation), workdir / stage.out_pdb)

    # -- Stage 2: NPgT (membrane barostat, tensionless) --
    stage = nprt
    steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=steps)
    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=state.getPositions(),
        add_restraints=bool(getattr(stage, "restraints", None)), add_barostat=True,
        velocities=state.getVelocities(),
    )
    attach_reporters(
        simulation, prefix=str(workdir / stage.name), total_steps=steps, stdout_log=True,
    )
    logger.info("Running NPgT...")
    simulation.step(steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    out_pdb = workdir / stage.out_pdb
    write_pdb(simulation.topology, _output_positions(simulation), out_pdb)
    if getattr(stage, "out_state", None):
        simulation.saveState(str(workdir / stage.out_state))
    if getattr(stage, "chk", None):
        simulation.saveCheckpoint(str(workdir / stage.chk))
    logger.info("Two-stage equilibration complete: %s", out_pdb)
    return out_pdb


def run_from_config(config_path, debug_first_step=False):
    """Load + validate a YAML config, build inputs, and run the relaxation."""
    conf = load_config(config_path)
    validate_config(conf)
    try:
        modeller, forcefield = build_system_inputs(conf)
        return relax(modeller, forcefield, conf, debug_first_step=debug_first_step)
    except BaseException:
        # The log is append-mode and shared across runs, so a crash that only
        # reached stderr leaves a block that just stops after the banner, with no
        # hint why. Record it in the log file too.
        logger.exception("RUN FAILED (%s)", config_path)
        logger.info("=" * 60)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CHARMM-GUI-style membrane relaxation for OpenMM.",
    )
    parser.add_argument(
        "--config", required=True, help="Membrane-relax YAML configuration file"
    )
    parser.add_argument(
        "--debug-first-step", action="store_true",
        help="Run one diagnostic step before Stage 1",
    )
    args = parser.parse_args(argv)
    run_from_config(Path(args.config), debug_first_step=args.debug_first_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
