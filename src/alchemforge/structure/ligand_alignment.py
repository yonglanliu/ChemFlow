"""Coordinate alignment of ligand files by maximum common substructure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AlignedLigandFile:
    """Aligned molecules serialized as an in-memory SDF."""

    source: Path
    sdf: str
    molecule_count: int
    rmsds: tuple[float, ...]
    aligned_count: int


def _load_molecules(path: Path):
    from rdkit import Chem

    suffix = path.suffix.lower()
    if suffix == ".sdf":
        molecules = [
            molecule
            for molecule in Chem.SDMolSupplier(str(path), removeHs=False)
            if molecule is not None
        ]
    elif suffix == ".mol":
        molecules = [Chem.MolFromMolFile(str(path), removeHs=False)]
    elif suffix == ".mol2":
        molecules = [Chem.MolFromMol2File(str(path), removeHs=False)]
    elif suffix in {".pdb", ".ent"}:
        molecules = [Chem.MolFromPDBFile(str(path), removeHs=False)]
    else:
        raise ValueError(
            f"Ligand alignment does not support {suffix or 'this format'}: {path.name}"
        )
    molecules = [molecule for molecule in molecules if molecule is not None]
    if not molecules:
        raise ValueError(f"RDKit could not read any molecules from {path}")
    if any(molecule.GetNumConformers() == 0 for molecule in molecules):
        raise ValueError(f"Ligand has no 3D coordinates: {path}")
    return molecules


def ligand_record_names(path: str | Path) -> list[str]:
    """Return display names for the ligand records in a supported file."""
    source = Path(path).expanduser().resolve()
    molecules = _load_molecules(source)
    names = []
    for index, molecule in enumerate(molecules, start=1):
        name = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
        names.append(name or f"record {index}")
    return names


def _align_molecule(molecule, reference, timeout_seconds: int) -> float:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS, rdMolAlign

    # Search only heavy atoms. Explicit hydrogens dramatically increase the
    # number of symmetry-equivalent matches and can make an otherwise simple
    # MCS search time out. Preserve the original indices for the final 3D fit.
    reference_copy = Chem.Mol(reference)
    molecule_copy = Chem.Mol(molecule)
    for atom in reference_copy.GetAtoms():
        atom.SetIntProp("_alignmentOriginalIndex", atom.GetIdx())
    for atom in molecule_copy.GetAtoms():
        atom.SetIntProp("_alignmentOriginalIndex", atom.GetIdx())
    reference_for_match = Chem.RemoveHs(reference_copy)
    molecule_for_match = Chem.RemoveHs(molecule_copy)
    result = rdFMCS.FindMCS(
        [reference_for_match, molecule_for_match],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=timeout_seconds,
    )
    if result.canceled:
        raise ValueError("Maximum-common-substructure search timed out")
    if result.numAtoms < 3:
        raise ValueError(
            "Ligands do not share at least three mapped atoms and cannot be aligned"
        )
    query = Chem.MolFromSmarts(result.smartsString)
    reference_matches = reference_for_match.GetSubstructMatches(query, uniquify=False)
    molecule_matches = molecule_for_match.GetSubstructMatches(query, uniquify=False)
    atom_maps = [
        [
            (
                molecule_for_match.GetAtomWithIdx(molecule_index).GetIntProp(
                    "_alignmentOriginalIndex"
                ),
                reference_for_match.GetAtomWithIdx(reference_index).GetIntProp(
                    "_alignmentOriginalIndex"
                ),
            )
            for molecule_index, reference_index in zip(
                molecule_match, reference_match, strict=True
            )
        ]
        for molecule_match in molecule_matches[:100]
        for reference_match in reference_matches[:100]
    ]
    if not atom_maps:
        raise ValueError(
            "The common substructure could not be mapped onto both ligands"
        )
    return float(rdMolAlign.GetBestRMS(molecule, reference, map=atom_maps))


def _to_sdf(molecules) -> str:
    from rdkit import Chem

    records = []
    for molecule in molecules:
        records.append(Chem.MolToMolBlock(molecule).rstrip() + "\n$$$$\n")
    return "".join(records)


def align_ligand_files(
    paths: list[str | Path],
    *,
    reference_file_index: int = 0,
    reference_molecule_index: int = 0,
    selected_molecule_indices: list[set[int]] | None = None,
    timeout_seconds: int = 10,
) -> list[AlignedLigandFile]:
    """Align every molecule to a selected molecule in the reference file.

    Input files are never modified. The aligned coordinates are returned as SDF
    strings suitable for visualization or explicit export by a caller.
    """
    if not paths:
        raise ValueError("Select at least one ligand file")
    if timeout_seconds < 1:
        raise ValueError("MCS timeout must be at least one second")
    if not 0 <= reference_file_index < len(paths):
        raise ValueError("Reference ligand index is outside the selected file list")
    sources = [Path(path).expanduser().resolve() for path in paths]
    for source in sources:
        if not source.is_file():
            raise ValueError(f"Ligand file does not exist: {source}")
    molecule_sets = [_load_molecules(source) for source in sources]
    if sum(map(len, molecule_sets)) < 2:
        raise ValueError("Load at least two ligand molecules to perform alignment")

    if not 0 <= reference_molecule_index < len(molecule_sets[reference_file_index]):
        raise ValueError("Reference molecule index is outside the selected ligand file")
    if selected_molecule_indices is None:
        selected_molecule_indices = [
            set(range(len(molecules))) for molecules in molecule_sets
        ]
    if len(selected_molecule_indices) != len(molecule_sets):
        raise ValueError("Ligand selection does not match the selected files")
    for file_index, (indices, molecules) in enumerate(
        zip(selected_molecule_indices, molecule_sets, strict=True)
    ):
        if any(index < 0 or index >= len(molecules) for index in indices):
            raise ValueError(
                f"Ligand selection is invalid for {sources[file_index].name}"
            )
    if reference_molecule_index not in selected_molecule_indices[reference_file_index]:
        raise ValueError("The reference molecule must be selected for alignment")
    if sum(map(len, selected_molecule_indices)) < 2:
        raise ValueError("Select at least two ligand molecules to perform alignment")
    reference = molecule_sets[reference_file_index][reference_molecule_index]
    outputs = []
    for file_index, (source, molecules) in enumerate(zip(sources, molecule_sets)):
        rmsds = []
        for molecule_index, molecule in enumerate(molecules):
            if molecule_index not in selected_molecule_indices[file_index]:
                rmsds.append(float("nan"))
            elif (
                file_index == reference_file_index
                and molecule_index == reference_molecule_index
            ):
                rmsds.append(0.0)
            else:
                try:
                    rmsds.append(_align_molecule(molecule, reference, timeout_seconds))
                except ValueError as error:
                    name = (
                        molecule.GetProp("_Name").strip()
                        if molecule.HasProp("_Name")
                        else ""
                    )
                    label = name or f"record {molecule_index + 1}"
                    raise ValueError(
                        f"Could not align {label!r} from {source.name}: {error}"
                    ) from error
        outputs.append(
            AlignedLigandFile(
                source=source,
                sdf=_to_sdf(molecules),
                molecule_count=len(molecules),
                rmsds=tuple(rmsds),
                aligned_count=len(selected_molecule_indices[file_index]),
            )
        )
    return outputs
