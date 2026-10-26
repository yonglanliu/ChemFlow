"""Stage and validate complete GROMACS systems for free-energy workflows."""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def installed_amber_lipid_force_fields() -> list[str]:
    """Return Amber lipid leaprc sets available to the active environment."""
    roots: list[Path] = []
    amberhome = os.environ.get("AMBERHOME")
    if amberhome:
        roots.append(Path(amberhome).expanduser() / "dat" / "leap" / "cmd")
    roots.append(Path(sys.prefix) / "dat" / "leap" / "cmd")
    available: list[str] = []
    for label in ("Lipid21", "Lipid17"):
        filename = f"leaprc.{label.casefold()}"
        if any(
            (root / filename).is_file() or (root / "oldff" / filename).is_file()
            for root in roots
        ):
            available.append(label)
    return available


def _amber_lipid_leaprc(lipid_force_field: str) -> str:
    """Return the LEaP source path for an installed Amber lipid set."""
    filename = f"leaprc.{lipid_force_field.casefold()}"
    roots: list[Path] = []
    amberhome = os.environ.get("AMBERHOME")
    if amberhome:
        roots.append(Path(amberhome).expanduser() / "dat" / "leap" / "cmd")
    roots.append(Path(sys.prefix) / "dat" / "leap" / "cmd")
    for relative in (filename, f"oldff/{filename}"):
        if any((root / relative).is_file() for root in roots):
            return relative
    raise ValueError(
        f"Amber {lipid_force_field} is not installed (missing {filename})."
    )


@dataclass(frozen=True)
class GromacsSystemPreparationConfig:
    mode: str
    structure: Path
    output_directory: Path
    topology: Path | None = None
    system_type: str = "membrane"
    gmx_command: str = "gmx"
    force_field: str = "charmm27"
    force_field_directory: Path | None = None
    water_model: str = "tip3p"
    protein_force_field: str = "ff14SB"
    lipid_force_field: str = "Lipid21"
    dna_force_field: str = "OL21"
    rna_force_field: str = "OL3"
    ion_parameters: str = "Joung-Cheatham TIP3P"
    ligand_force_field: str = "GAFF2"
    cofactor_force_field: str = "None"
    cofactor_structure: Path | None = None
    prepared_cofactor_directory: Path | None = None
    cofactor_resname: str = "COF"
    cofactor_charge: int = 0
    ligand_library: Path | None = None
    prepared_ligand_directory: Path | None = None
    reference_ligand: str | None = None
    ligand_resname: str = "LIG"
    ligand_charge: int = 0
    exclude_ligand_from_system: bool = False
    neutralize: bool = False
    cofactor_topology: Path | None = None
    cofactor_count: int = 1
    box_lengths_nm: tuple[float, float, float] | None = None
    box_angles_deg: tuple[float, float, float] = (90.0, 90.0, 90.0)

    def validate(self) -> None:
        if self.mode not in {"existing", "pdb2gmx", "amber_pdb_sdf"}:
            raise ValueError(
                "Preparation mode must be existing, pdb2gmx, or amber_pdb_sdf"
            )
        if self.system_type not in {"soluble", "membrane"}:
            raise ValueError("System type must be soluble or membrane")
        if not self.structure.is_file():
            raise ValueError(f"Structure does not exist: {self.structure}")
        if not self.gmx_command.strip():
            raise ValueError("GROMACS command is required")
        if self.box_lengths_nm is not None:
            if len(self.box_lengths_nm) != 3 or any(
                value <= 0 for value in self.box_lengths_nm
            ):
                raise ValueError("Periodic box lengths must be three positive values")
            if len(self.box_angles_deg) != 3 or any(
                value <= 0 or value >= 180 for value in self.box_angles_deg
            ):
                raise ValueError(
                    "Periodic box angles must be three values between 0 and 180 degrees"
                )
        if self.mode == "existing":
            if self.structure.suffix.lower() != ".gro":
                raise ValueError("Existing-system mode requires a GRO structure")
            if self.topology is None or not self.topology.is_file():
                raise ValueError("Select the topology matching the GRO structure")
        elif self.structure.suffix.lower() != ".pdb":
            raise ValueError("System preparation requires an assembled PDB")
        if self.mode == "pdb2gmx":
            if self.force_field_directory is not None:
                if not self.force_field_directory.is_dir():
                    raise ValueError(
                        "Force-field directory does not exist: "
                        f"{self.force_field_directory}"
                    )
                if self.force_field_directory.suffix.casefold() != ".ff":
                    raise ValueError("Force-field directory name must end in .ff")
            unsupported = unsupported_pdb2gmx_residues(
                self.structure,
                self.force_field,
                self.force_field_directory,
            )
            if unsupported:
                names = ", ".join(unsupported)
                raise ValueError(
                    f"The selected {self.force_field} force field has no pdb2gmx "
                    f"residue templates for: {names}. pdb2gmx cannot parameterize "
                    "an assembled membrane/ligand PDB by itself. Use Existing GRO "
                    "+ topology from the system's original parameterization, or "
                    "choose a preparation route that explicitly supports every "
                    "ligand, lipid, ion, and cofactor."
                )
        if self.mode == "amber_pdb_sdf":
            if self.lipid_force_field not in installed_amber_lipid_force_fields():
                raise ValueError(
                    f"Amber {self.lipid_force_field} is not installed in the active "
                    "AmberTools environment (missing "
                    f"leaprc.{self.lipid_force_field.casefold()})."
                )
            if self.prepared_ligand_directory is not None:
                if not self.prepared_ligand_directory.is_dir():
                    raise ValueError(
                        "Prepared ligand directory does not exist: "
                        f"{self.prepared_ligand_directory}"
                    )
                _prepared_ligand_parameters(
                    self.prepared_ligand_directory, self.reference_ligand
                )
            else:
                if self.ligand_library is None or not self.ligand_library.is_file():
                    raise ValueError("Select the prepared ligand directory")
                if self.ligand_library.suffix.casefold() not in {".sdf", ".mol2"}:
                    raise ValueError("The ligand input must be an SDF or MOL2 file")
            if not self.reference_ligand:
                raise ValueError("Select the reference ligand from the SDF")
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,2}", self.ligand_resname):
                raise ValueError(
                    "Amber ligand residue name must begin with a letter and contain "
                    "at most three letters/numbers"
                )
            supported_components = {
                "protein": {"ff14SB", "ff19SB", "ff99SBildn"},
                "lipid": {"Lipid17", "Lipid21"},
                "DNA": {"None", "OL21", "bsc1"},
                "RNA": {"None", "OL3"},
                "ions": set(_AMBER_WATER_IONS.values()),
                "ligand": {"GAFF2"},
                "cofactor": {"None", "GAFF2"},
            }
            selected_components = {
                "protein": self.protein_force_field,
                "lipid": self.lipid_force_field,
                "DNA": self.dna_force_field,
                "RNA": self.rna_force_field,
                "ions": self.ion_parameters,
                "ligand": self.ligand_force_field,
                "cofactor": self.cofactor_force_field,
            }
            for component, selection in selected_components.items():
                if selection not in supported_components[component]:
                    raise ValueError(
                        f"Unsupported Amber {component} parameter set: {selection}"
                    )
            has_dna, has_rna = _pdb_nucleic_components(self.structure)
            if self.protein_force_field == "ff99SBildn" and (
                (has_dna and self.dna_force_field != "None")
                or (has_rna and self.rna_force_field != "None")
            ):
                raise ValueError(
                    "ff99SBildn loads legacy parm99 nucleic-acid parameters and "
                    "cannot be combined in one LEaP session with OL21, bsc1, or "
                    "OL3 (which load parm10). Choose ff14SB/ff19SB for modern "
                    "nucleic-acid parameters, or select None to use the legacy "
                    "nucleic parameters already supplied by ff99SBildn."
                )
            water_model = self.water_model.casefold()
            if water_model not in _AMBER_WATER_IONS:
                raise ValueError(
                    "Unsupported Amber water model. Choose TIP3P, OPC, OPC3, "
                    "SPC/E, or TIP4P-Ew"
                )
            expected_ions = _AMBER_WATER_IONS[water_model]
            if self.ion_parameters != expected_ions:
                raise ValueError(
                    f"Amber {self.water_model} water requires {expected_ions} "
                    "ion parameters"
                )
            if self.cofactor_force_field == "GAFF2":
                if self.prepared_cofactor_directory is not None:
                    if not self.prepared_cofactor_directory.is_dir():
                        raise ValueError(
                            "Prepared cofactor directory does not exist: "
                            f"{self.prepared_cofactor_directory}"
                        )
                    _prepared_ligand_parameters(
                        self.prepared_cofactor_directory,
                        self.cofactor_resname or None,
                    )
                else:
                    if (
                        self.cofactor_structure is None
                        or not self.cofactor_structure.is_file()
                    ):
                        raise ValueError(
                            "Select a cofactor SDF/MOL2 or a prepared cofactor "
                            "parameter directory"
                        )
                    if self.cofactor_structure.suffix.casefold() not in {
                        ".sdf",
                        ".mol2",
                    }:
                        raise ValueError(
                            "The Amber cofactor input must be an SDF or MOL2 file"
                        )
                if not re.fullmatch(
                    r"[A-Za-z0-9]{1,5}", self.cofactor_resname
                ):
                    raise ValueError(
                        "Cofactor residue name must contain 1-5 letters/numbers"
                    )
        if self.cofactor_topology is not None:
            if self.mode != "existing":
                raise ValueError(
                    "Cofactor ITP import requires Existing GRO + topology mode. "
                    "PDB build modes cannot safely reconstruct membrane/cofactor "
                    "coordinates and topology ordering."
                )
            if not self.cofactor_topology.is_file():
                raise ValueError(
                    f"Cofactor topology does not exist: {self.cofactor_topology}"
                )
            if self.cofactor_topology.suffix.lower() != ".itp":
                raise ValueError("Cofactor topology must be a GROMACS ITP file")
            if self.cofactor_count < 1:
                raise ValueError("Cofactor molecule count must be at least 1")


_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"')
_SINGLE_RESIDUE_LIPIDS = {
    "DLPC",
    "DMPC",
    "DOPC",
    "DPPC",
    "POPC",
    "POPE",
    "POPG",
    "POPS",
}
_RTP_SUBSECTIONS = {
    "atoms",
    "bonds",
    "angles",
    "dihedrals",
    "impropers",
    "exclusions",
    "bondedtypes",
    "cmap",
}
_PDB2GMX_PASSTHROUGH_RESIDUES = {
    "HOH",
    "SOL",
    "WAT",
    "NA",
    "CL",
    "K",
    "CA",
    "MG",
    "ZN",
}

_AMBER_WATER_IONS = {
    "tip3p": "Joung-Cheatham TIP3P",
    "opc": "Li-Merz OPC",
    "opc3": "Li-Merz OPC3",
    "spce": "Joung-Cheatham SPC/E",
    "tip4pew": "Joung-Cheatham TIP4P-Ew",
}

_AMBER_WATER_LEAPRC = {
    model: f"leaprc.water.{model}" for model in _AMBER_WATER_IONS
}


def _load_small_molecules(path: Path):
    """Read one or more RDKit molecules from SDF or concatenated MOL2."""
    from rdkit import Chem

    if path.suffix.casefold() == ".sdf":
        return [
            molecule
            for molecule in Chem.SDMolSupplier(str(path), removeHs=False)
            if molecule is not None
        ]
    blocks = path.read_text(encoding="utf-8").split("@<TRIPOS>MOLECULE")
    return [
        molecule
        for block in blocks[1:]
        if (
            molecule := Chem.MolFromMol2Block(
                "@<TRIPOS>MOLECULE" + block, removeHs=False
            )
        )
        is not None
    ]


def _prepared_ligand_parameters(
    directory: Path, reference: str | None
) -> tuple[Path, Path, str]:
    """Locate an ACPYPE charged MOL2 and matching frcmod in a prepared library."""
    target: Path | None = None
    resname = reference or directory.name.removesuffix(".acpype")
    manifest = directory / "ligand_manifest.tsv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                values = {
                    row.get("ligand_name", ""),
                    row.get("resname", ""),
                    row.get("directory", ""),
                }
                if reference and reference in values:
                    target = directory / row.get("directory", "")
                    resname = row.get("resname", "") or reference
                    break
    if target is None and reference:
        candidate = directory / reference
        target = candidate if candidate.is_dir() else directory
    elif target is None:
        target = directory
    mol2_files = list(target.rglob("*.mol2"))
    charged = sorted(
        mol2_files,
        key=lambda path: (
            ".acpype" not in str(path.parent),
            not any(tag in path.name.casefold() for tag in ("bcc", "user", "ac")),
            len(path.parts),
            path.name.casefold(),
        ),
    )
    frcmods = sorted(target.rglob("*.frcmod"), key=lambda path: path.name.casefold())
    if not charged or not frcmods:
        raise ValueError(
            f"Prepared parameter folder {directory} must contain an assigned-charge MOL2 "
            "and its matching ACPYPE frcmod file"
        )
    return charged[0], frcmods[0], resname.upper()


def _copy_mol2_with_resname(
    source: Path,
    destination: Path,
    resname: str,
    net_charge: int | None = None,
) -> None:
    """Copy typed MOL2 data with normalized residue identity and integer charge."""
    normalized = resname.upper()
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    section = ""
    output: list[str] = []
    charge_fields: list[tuple[int, list[str]]] = []
    for line in lines:
        if line.startswith("@<TRIPOS>"):
            section = line.removeprefix("@<TRIPOS>").strip().upper()
            output.append(line)
            continue
        fields = line.split()
        if section == "ATOM" and len(fields) >= 8:
            fields[7] = normalized
            line = " ".join(fields)
            if len(fields) >= 9:
                charge_fields.append((len(output), fields))
        elif section == "SUBSTRUCTURE" and len(fields) >= 2:
            fields[1] = normalized
            line = " ".join(fields)
        output.append(line)
    if net_charge is not None and charge_fields:
        total = sum(float(fields[8]) for _, fields in charge_fields)
        correction = float(net_charge) - total
        target_index, target_fields = max(
            charge_fields, key=lambda item: abs(float(item[1][8]))
        )
        target_fields[8] = f"{float(target_fields[8]) + correction:.8f}"
        output[target_index] = " ".join(target_fields)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


_PROTEIN_RESIDUES = {
    "ACE", "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYM", "CYX",
    "GLN", "GLU", "GLH", "GLY", "HID", "HIE", "HIP", "HIS", "ILE",
    "LEU", "LYS", "LYN", "MET", "NME", "PHE", "PRO", "SER", "THR",
    "TRP", "TYR", "VAL",
}

_DNA_RESIDUES = {
    "DA", "DC", "DG", "DT", "DA3", "DC3", "DG3", "DT3",
    "DA5", "DC5", "DG5", "DT5",
}
_RNA_RESIDUES = {
    "A", "C", "G", "U", "A3", "C3", "G3", "U3",
    "A5", "C5", "G5", "U5", "RA", "RC", "RG", "RU",
}


def _pdb_nucleic_components(path: Path) -> tuple[bool, bool]:
    """Return whether a PDB contains DNA and RNA polymer residues."""
    residues = {
        line[17:20].strip().upper()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith("ATOM  ") and len(line) >= 20
    }
    return bool(residues & _DNA_RESIDUES), bool(residues & _RNA_RESIDUES)

# OpenMM/PDBFixer and CHARMM-GUI use different names for the same POPC
# headgroup atoms. The AmberTools converter accepts the CHARMM-GUI names.
_OPENMM_POPC_TO_CHARMM = {
    "P1": "P",
    "O1": "O12",
    "O2": "O11",
    "O3": "O13",
    "O4": "O14",
    "C11": "C13",
    "H21": "H13A",
    "H22": "H13B",
    "H23": "H13C",
    "C13": "C14",
    "H31": "H14A",
    "H32": "H14B",
    "H33": "H14C",
    "C14": "C15",
    "H41": "H15A",
    "H42": "H15B",
    "H43": "H15C",
    "C15": "C12",
    "H11": "H12A",
    "H12": "H12B",
    "C12": "C11",
    "H51": "H11A",
    "H52": "H11B",
}


def installed_gromacs_force_fields() -> list[str]:
    """Return force fields visible to the current local GROMACS environment."""

    roots: list[Path] = []
    gmxlibrary = os.environ.get("GMXLIB")
    if gmxlibrary:
        roots.append(Path(gmxlibrary).expanduser())
    roots.extend(
        (
            Path.cwd(),
            Path(sys.prefix) / "share" / "gromacs" / "top",
        )
    )
    names: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        names.update(path.name.removesuffix(".ff") for path in root.glob("*.ff"))
    return sorted(names, key=str.casefold)


def _force_field_directory(name: str) -> Path | None:
    roots: list[Path] = []
    gmxlibrary = os.environ.get("GMXLIB")
    if gmxlibrary:
        roots.append(Path(gmxlibrary).expanduser())
    roots.extend((Path.cwd(), Path(sys.prefix) / "share" / "gromacs" / "top"))
    for root in roots:
        candidate = root / f"{name}.ff"
        if candidate.is_dir():
            return candidate
    return None


def unsupported_pdb2gmx_residues(
    structure: Path,
    force_field: str,
    force_field_directory: Path | None = None,
) -> list[str]:
    """Return PDB residues absent from the selected force field RTP files."""
    directory = force_field_directory or _force_field_directory(force_field)
    if directory is None:
        return []
    supported = set(_PDB2GMX_PASSTHROUGH_RESIDUES)
    section = re.compile(r"^\s*\[\s*([^]]+?)\s*]\s*(?:;.*)?$")
    for rtp in directory.glob("*.rtp"):
        for line in rtp.read_text(encoding="utf-8", errors="replace").splitlines():
            match = section.match(line)
            if not match:
                continue
            residue = match.group(1).strip()
            if residue.casefold() not in _RTP_SUBSECTIONS:
                supported.add(residue.upper())
    observed = set()
    for line in structure.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 21:
            continue
        # CHARMM uses the normally blank column 21 to extend residue names to
        # four characters (for example POPC). Strict PDB readers expose only
        # the standard three-character field and misleadingly report POP.
        extended = line[17:21].strip().upper()
        observed.add(extended if line[20].strip() else line[17:20].strip().upper())
    return sorted(observed - supported)


def _single_residue_lipids(path: Path) -> list[str]:
    names = {
        line[17:21].strip().upper()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 21
    }
    return sorted(names & _SINGLE_RESIDUE_LIPIDS)


def _cofactor_atom_names(template: Path, atoms: list[str]) -> list[str]:
    """Rename a PDB cofactor copy to the SDF/Antechamber atom order."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    from rdkit.Geometry import Point3D

    reference_names: list[str] | None = None
    if template.suffix.casefold() == ".mol2":
        mol2_lines = template.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        section = ""
        atom_rows: list[list[str]] = []
        bond_rows: list[list[str]] = []
        for line in mol2_lines:
            if line.startswith("@<TRIPOS>"):
                section = line.removeprefix("@<TRIPOS>").strip().upper()
                continue
            fields = line.split()
            if section == "ATOM" and len(fields) >= 6:
                atom_rows.append(fields)
            elif section == "BOND" and len(fields) >= 4:
                bond_rows.append(fields)
        editable = Chem.RWMol()
        conformer = Chem.Conformer(len(atom_rows))
        reference_names = []
        for index, fields in enumerate(atom_rows):
            atom_type = fields[5].split(".", 1)[0].casefold()
            if atom_type.startswith("cl"):
                element = "Cl"
            elif atom_type.startswith("br"):
                element = "Br"
            else:
                element = {
                    "c": "C", "n": "N", "o": "O", "s": "S",
                    "p": "P", "f": "F", "i": "I", "h": "H",
                }.get(atom_type[:1])
            if element is None:
                raise ValueError(
                    f"Cannot derive element from MOL2 atom type {fields[5]!r}"
                )
            editable.AddAtom(Chem.Atom(element))
            conformer.SetAtomPosition(
                index, Point3D(float(fields[2]), float(fields[3]), float(fields[4]))
            )
            reference_names.append(fields[1])
        bond_types = {
            "1": Chem.BondType.SINGLE,
            "2": Chem.BondType.DOUBLE,
            "3": Chem.BondType.TRIPLE,
            "ar": Chem.BondType.AROMATIC,
            "am": Chem.BondType.SINGLE,
        }
        for fields in bond_rows:
            editable.AddBond(
                int(fields[1]) - 1,
                int(fields[2]) - 1,
                bond_types.get(fields[3].casefold(), Chem.BondType.SINGLE),
            )
        reference = editable.GetMol()
        reference.AddConformer(conformer)
    else:
        reference = next(
            (mol for mol in Chem.SDMolSupplier(str(template), removeHs=False) if mol),
            None,
        )
    block = "\n".join(atoms) + "\nEND\n"
    target = Chem.MolFromPDBBlock(
        block, removeHs=False, sanitize=False, proximityBonding=True
    )
    if reference is None or target is None or reference.GetNumAtoms() != len(atoms):
        raise ValueError("Cofactor SDF does not match the cofactor residues in the PDB")
    mcs = rdFMCS.FindMCS(
        [reference, target],
        timeout=10,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
    )
    if mcs.canceled or mcs.numAtoms != reference.GetNumAtoms():
        raise ValueError("Could not map the cofactor SDF onto a PDB cofactor copy")
    query = Chem.MolFromSmarts(mcs.smartsString)
    reference_match = reference.GetSubstructMatch(query)
    target_match = target.GetSubstructMatch(query)
    target_to_reference = dict(zip(target_match, reference_match, strict=True))
    if reference_names is None:
        counters: dict[str, int] = {}
        reference_names = []
        for atom in reference.GetAtoms():
            element = atom.GetSymbol().upper()
            counters[element] = counters.get(element, 0) + 1
            reference_names.append(f"{element}{counters[element]}")
    renamed = []
    for target_index, line in enumerate(atoms):
        name = reference_names[target_to_reference[target_index]]
        atom_field = name if len(name) == 4 else f" {name:<3}"
        renamed.append(line[:12] + atom_field + line[16:])
    return renamed


def _stage_charmm_lipids_for_amber(
    source: Path,
    destination: Path,
    cofactor_template: Path | None = None,
    cofactor_resname: str | None = None,
) -> int:
    """Regroup CHARMM lipid atoms for AmberTools lipid conversion.

    Assembled PDB files can interleave ions between lipid heavy atoms and
    hydrogens. ``charmmlipid2amber.py`` requires each lipid to be contiguous,
    so lipid atoms are grouped by their original residue identity first.
    """
    lipid_groups: dict[tuple[str, str, str, str], list[str]] = {}
    headers: list[str] = []
    base_groups: list[tuple[tuple[str, str, str, str], list[str]]] = []
    current_key: tuple[str, str, str, str] | None = None
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        is_atom = line.startswith(("ATOM  ", "HETATM"))
        residue = line[17:21].strip().upper() if is_atom and len(line) >= 21 else ""
        if residue in _SINGLE_RESIDUE_LIPIDS:
            key = (residue, line[21:22], line[22:26], line[26:27])
            lipid_groups.setdefault(key, []).append(line)
            current_key = None
            continue
        if is_atom:
            # PDBFixer commonly names ACE/NME hydrogens differently from the
            # Amber residue templates. Keep cap heavy atoms and let LEaP add
            # template-consistent hydrogens.
            atom_name = line[12:16].strip().lstrip("0123456789").upper()
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if residue in {"ACE", "NME"} and (
                element in {"H", "D"} or atom_name.startswith(("H", "D"))
            ):
                continue
            key = (residue, line[21:22], line[22:26], line[26:27])
            if key != current_key:
                base_groups.append((key, []))
                current_key = key
            base_groups[-1][1].append(line)
            continue
        current_key = None
        if line.startswith("CRYST1"):
            headers.append(line)

    if not lipid_groups:
        return 0
    lipid_lines: list[str] = []
    for (residue, _chain, _number, _insertion), atoms in lipid_groups.items():
        names = {line[12:16].strip() for line in atoms}
        if residue == "POPC" and "P1" in names and "P" not in names:
            translated = []
            for line in atoms:
                atom_name = line[12:16].strip()
                replacement = _OPENMM_POPC_TO_CHARMM.get(atom_name)
                if replacement:
                    atom_field = (
                        replacement if len(replacement) == 4 else f" {replacement:<3}"
                    )
                    line = line[:12] + atom_field + line[16:]
                translated.append(line)
            atoms = translated
        lipid_lines.extend(atoms)
        lipid_lines.append("TER")
    base_lines = list(headers)
    hetero_index = 0
    hetero_chains = "EFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    previous_protein_chain: str | None = None
    for (residue, chain, number, _insertion), atoms in base_groups:
        output_residue = residue
        if residue in {"HIS", "HID", "HIE", "HIP"}:
            atom_names = {line[12:16].strip() for line in atoms}
            has_hd1 = "HD1" in atom_names
            has_he2 = "HE2" in atom_names
            output_residue = (
                "HIP" if has_hd1 and has_he2 else ("HID" if has_hd1 else "HIE")
            )
        if (
            cofactor_template is not None
            and cofactor_resname is not None
            and residue == cofactor_resname.upper()
        ):
            atoms = _cofactor_atom_names(cofactor_template, atoms)
            # LEaP does not reliably register templates whose names start with
            # a digit (for example 4DY). Use a stable internal residue name.
            output_residue = "COF"
        if residue in _PROTEIN_RESIDUES:
            if previous_protein_chain is not None and chain != previous_protein_chain:
                base_lines.append("TER")
            new_chain, new_number = chain or "A", number
            previous_protein_chain = chain
            terminate = False
        else:
            if previous_protein_chain is not None:
                base_lines.append("TER")
                previous_protein_chain = None
            chain_index, residue_index = divmod(hetero_index, 9999)
            if chain_index >= len(hetero_chains):
                raise ValueError("The PDB contains too many non-protein residues")
            new_chain = hetero_chains[chain_index]
            new_number = str(residue_index + 1)
            hetero_index += 1
            terminate = True
        for line in atoms:
            if output_residue != residue:
                line = line[:17] + f"{output_residue:>3}" + line[20:]
            base_lines.append(
                line[:21] + new_chain + f"{int(new_number):4d}" + " " + line[27:]
            )
        if terminate:
            base_lines.append("TER")
    if previous_protein_chain is not None:
        base_lines.append("TER")
    (destination / "amber_base.pdb").write_text(
        "\n".join(base_lines) + "\nEND\n", encoding="utf-8"
    )
    (destination / "charmm_lipids.pdb").write_text(
        "\n".join(lipid_lines) + "\nEND\n", encoding="utf-8"
    )
    return len(lipid_groups)


def _amber_tleap_sources(config: GromacsSystemPreparationConfig) -> list[str]:
    """Return a coherent set of Amber component force-field commands."""
    protein_leaprc = {
        "ff14SB": "leaprc.protein.ff14SB",
        "ff19SB": "leaprc.protein.ff19SB",
        # AmberTools keeps this legacy force field below dat/leap/cmd/oldff;
        # tleap does not recursively search that directory.
        "ff99SBildn": "oldff/leaprc.ff99SBildn",
    }[config.protein_force_field]
    sources = [f"source {protein_leaprc}"]
    has_dna, has_rna = _pdb_nucleic_components(config.structure)
    if has_dna and config.dna_force_field != "None":
        sources.append(f"source leaprc.DNA.{config.dna_force_field}")
    if has_rna and config.rna_force_field != "None":
        sources.append(f"source leaprc.RNA.{config.rna_force_field}")
    sources.extend(
        (
            f"source {_amber_lipid_leaprc(config.lipid_force_field)}",
            "source leaprc.gaff2",
            f"source {_AMBER_WATER_LEAPRC[config.water_model.casefold()]}",
        )
    )
    return sources


def _copy_topology_tree(source_topology: Path, destination: Path) -> None:
    """Copy a topology and all locally resolvable quoted includes."""
    root = source_topology.parent.resolve()
    pending = [source_topology.resolve()]
    copied: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in copied:
            continue
        try:
            relative = source.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"Topology include escapes its directory: {source}. Place all local "
                "TOP/ITP dependencies below the selected topology directory."
            ) from error
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.add(source)
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _INCLUDE.match(line)
            if not match:
                continue
            included = (source.parent / match.group(1)).resolve()
            if included.is_file():
                pending.append(included)


def _molecule_type_name(topology: Path) -> str:
    in_molecule_type = False
    for line in topology.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_molecule_type = stripped[1:-1].strip().casefold() == "moleculetype"
            continue
        if in_molecule_type and stripped and not stripped.startswith((";", "#")):
            return stripped.split()[0]
    raise ValueError(
        f"Cofactor ITP has no readable [ moleculetype ] entry: {topology}"
    )


def _register_cofactor(
    topology: Path, include_path: str, molecule_type: str, count: int
) -> None:
    """Add a cofactor include and molecule count to a staged main topology."""
    lines = topology.read_text(encoding="utf-8", errors="replace").splitlines()
    section = None
    molecule_present = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().casefold()
            continue
        if (
            section == "molecules"
            and stripped
            and not stripped.startswith((";", "#"))
            and stripped.split()[0] == molecule_type
        ):
            molecule_present = True
            break
    if molecule_present:
        return

    include = f'#include "{include_path}"'
    if include not in lines:
        insertion = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip().casefold() == "[ system ]"
            ),
            len(lines),
        )
        lines[insertion:insertion] = [include, ""]

    molecules = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().casefold() == "[ molecules ]"
        ),
        None,
    )
    if molecules is None:
        lines.extend(["", "[ molecules ]", "; Compound        #mols"])
    lines.append(f"{molecule_type:<20} {count}")
    topology.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_gromacs_system(
    config: GromacsSystemPreparationConfig,
) -> tuple[Path, Path]:
    """Stage a reproducible preparation script and return it with its manifest."""
    config.validate()
    task = config.output_directory.expanduser().resolve()
    task.mkdir(parents=True, exist_ok=True)
    if config.mode == "pdb2gmx" and config.force_field_directory is not None:
        staged_force_field = task / f"{config.force_field}.ff"
        if config.force_field_directory.resolve() != staged_force_field.resolve():
            shutil.copytree(
                config.force_field_directory,
                staged_force_field,
                dirs_exist_ok=True,
            )
    structure_name = "system.gro" if config.mode == "existing" else "assembled.pdb"
    staged_structure = task / structure_name
    if config.structure.resolve() != staged_structure.resolve():
        shutil.copy2(config.structure, staged_structure)
    if config.mode == "existing":
        assert config.topology is not None
        _copy_topology_tree(config.topology, task)
        staged_topology = task / config.topology.name
        if staged_topology.name != "topol.top":
            shutil.copy2(staged_topology, task / "topol.top")
        if config.cofactor_topology is not None:
            cofactor_dir = task / "cofactors"
            _copy_topology_tree(config.cofactor_topology, cofactor_dir)
            molecule_type = _molecule_type_name(config.cofactor_topology)
            _register_cofactor(
                task / "topol.top",
                f"cofactors/{config.cofactor_topology.name}",
                molecule_type,
                config.cofactor_count,
            )
    elif config.mode == "amber_pdb_sdf":
        from rdkit import Chem

        prepared_cofactor = (
            _prepared_ligand_parameters(
                config.prepared_cofactor_directory,
                config.cofactor_resname or None,
            )
            if config.prepared_cofactor_directory is not None
            else None
        )
        charmm_lipids = _single_residue_lipids(config.structure)
        lipid_count = (
            _stage_charmm_lipids_for_amber(
                config.structure,
                task,
                (
                    prepared_cofactor[0]
                    if prepared_cofactor is not None
                    else (
                        config.cofactor_structure
                        if config.cofactor_force_field == "GAFF2"
                        else None
                    )
                ),
                config.cofactor_resname,
            )
            if charmm_lipids
            else 0
        )
        if config.prepared_ligand_directory is not None:
            charged_mol2, frcmod, _ = _prepared_ligand_parameters(
                config.prepared_ligand_directory, config.reference_ligand
            )
            _copy_mol2_with_resname(
                charged_mol2,
                task / "ligand.mol2",
                config.ligand_resname,
                config.ligand_charge,
            )
            shutil.copy2(frcmod, task / "ligand.frcmod")
        else:
            assert config.ligand_library is not None
            reference = None
            for index, molecule in enumerate(
                _load_small_molecules(config.ligand_library), start=1
            ):
                molecule_name = (
                    molecule.GetProp("_Name").strip()
                    if molecule.HasProp("_Name")
                    else f"ligand_{index:03d}"
                )
                if molecule_name == config.reference_ligand:
                    reference = molecule
                    break
            if reference is None:
                raise ValueError(
                    f"Reference ligand {config.reference_ligand!r} was not found in "
                    f"{config.ligand_library.name}"
                )
            if reference.GetNumConformers() == 0:
                raise ValueError("The reference ligand has no 3D coordinates")
            writer = Chem.SDWriter(str(task / "reference_ligand.sdf"))
            writer.write(reference)
            writer.close()
        if config.cofactor_force_field == "GAFF2":
            if prepared_cofactor is not None:
                _copy_mol2_with_resname(
                    prepared_cofactor[0],
                    task / "cofactor.mol2",
                    "COF",
                    config.cofactor_charge,
                )
                shutil.copy2(prepared_cofactor[1], task / "cofactor.frcmod")
            else:
                assert config.cofactor_structure is not None
                cofactors = _load_small_molecules(config.cofactor_structure)
                if not cofactors:
                    raise ValueError("The cofactor file contains no readable molecule")
                if len(cofactors) != 1:
                    raise ValueError(
                        "The cofactor file contains multiple molecules. Select or "
                        "export one cofactor chemotype per parameter entry; multiple "
                        "identical copies in the PDB need only one SDF molecule."
                    )
                cofactor = cofactors[0]
                if cofactor.GetNumConformers() == 0:
                    raise ValueError(
                        "The cofactor in the SDF has no 3D coordinates"
                    )
                cofactor_writer = Chem.SDWriter(str(task / "cofactor.sdf"))
                cofactor_writer.write(cofactor)
                cofactor_writer.close()
        source_lines = "\n".join(_amber_tleap_sources(config))
        ligand_unit = config.ligand_resname.upper()
        cofactor_lines = ""
        if config.cofactor_force_field == "GAFF2":
            cofactor_lines = (
                "loadAmberParams cofactor.frcmod\n"
                "COF = loadmol2 cofactor.mol2\n"
            )
        ligand_lines = ""
        if not config.exclude_ligand_from_system:
            ligand_lines = (
                "loadAmberParams ligand.frcmod\n"
                f"{ligand_unit} = loadmol2 ligand.mol2\n"
            )
        (task / "tleap.in").write_text(
            f"""{source_lines}
{ligand_lines}{cofactor_lines}system = loadpdb amber_complex.pdb
check system
saveamberparm system system.prmtop system.inpcrd
quit
""",
            encoding="utf-8",
        )
        if config.neutralize and config.system_type == "membrane":
            # The reusable membrane base excludes the reference ligand.  Its
            # target charge is therefore the opposite of the reference ligand
            # so that the complex A state is neutral.  LEaP must first build
            # the topology before we can know the charge contributed by the
            # protein, caps, cofactors, existing ions, and membrane.
            neutralized_tleap = (
                f"{source_lines}\n"
                f"{cofactor_lines}system = loadpdb amber_complex.pdb\n"
                "__ADD_IONS__\n"
                "check system\n"
                "saveamberparm system system.prmtop system.inpcrd\n"
                "quit\n"
            )
            (task / "neutralize_amber_system.py").write_text(
                f'''from pathlib import Path
import parmed as pmd

system = pmd.load_file("system.prmtop", "system.inpcrd")
Path("tleap_neutralize.in").unlink(missing_ok=True)
current = float(sum(atom.charge for atom in system.atoms))
target = {-config.ligand_charge!r}
delta = target - current
count = round(abs(delta))
if abs(delta - round(delta)) > 1.0e-3:
    raise SystemExit(
        f"Cannot neutralize the membrane base: current charge {{current:.6f}} e "
        f"and target charge {{target:+d}} e differ by a non-integer amount."
    )
if count == 0:
    print(f"Membrane base already has target charge {{target:+d}} e")
else:
    ion = "Na+" if delta > 0 else "Cl-"
    template = {neutralized_tleap!r}
    Path("tleap_neutralize.in").write_text(
        template.replace("__ADD_IONS__", f"addIonsRand system {{ion}} {{count}}")
    )
    print(
        f"Adjusting membrane-base charge from {{current:+.0f}} e to "
        f"{{target:+d}} e with {{count}} {{ion}} counterion(s)"
    )
''',
                encoding="utf-8",
            )
        system_sources = (
            ["amber_base.pdb", "amber_lipids.pdb"]
            if lipid_count
            else ["assembled.pdb"]
        )
        (task / "assemble_amber_complex.py").write_text(
            f'''from pathlib import Path

resname = {config.ligand_resname.upper()!r}
system = []
for source in {system_sources!r}:
    system.extend(Path(source).read_text().splitlines())
ligand = ([] if {config.exclude_ligand_from_system!r}
          else Path("ligand.pdb").read_text().splitlines())
with Path("amber_complex.pdb").open("w") as output:
    for line in system:
        if line.startswith(("ATOM  ", "HETATM")) and line[17:20].strip().upper() == resname:
            continue
        if not line.startswith(("END", "ENDMDL")):
            output.write(line + "\\n")
    if not {config.exclude_ligand_from_system!r}:
        for line in ligand:
            if line.startswith(("ATOM  ", "HETATM")):
                line = line[:17] + f"{{resname:>3}}" + line[20:]
                output.write(line + "\\n")
    output.write("END\\n")
''',
            encoding="utf-8",
        )
        (task / "convert_amber_to_gromacs.py").write_text(
            '''from pathlib import Path
import parmed as pmd

system = pmd.load_file("system.prmtop", "system.inpcrd")
cryst1 = next((line for line in Path("assembled.pdb").read_text().splitlines()
               if line.startswith("CRYST1")), None)
if system.box is None and cryst1:
    system.box = [float(cryst1[6:15]), float(cryst1[15:24]),
                  float(cryst1[24:33]), float(cryst1[33:40]),
                  float(cryst1[40:47]), float(cryst1[47:54])]
system.save("topol.top", overwrite=True)
system.save("system.gro", overwrite=True)
''',
            encoding="utf-8",
        )

    mdp = task / "validate.mdp"
    mdp.write_text(
        """integrator = md
nsteps = 0
continuation = yes
cutoff-scheme = Verlet
nstlist = 20
coulombtype = PME
rcoulomb = 1.0
vdwtype = Cut-off
rvdw = 1.0
pbc = xyz
constraints = none
""",
        encoding="utf-8",
    )
    gmx = shlex.split(config.gmx_command)
    if not gmx:
        raise ValueError("GROMACS command is required")
    quoted_gmx = shlex.join(gmx)
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", 'cd "$(dirname "$0")"']
    if config.mode == "pdb2gmx":
        commands.extend(
            [
                (
                    f"{quoted_gmx} pdb2gmx -f assembled.pdb -o system.gro "
                    f"-p topol.top -i posre.itp -ff "
                    f"{shlex.quote(config.force_field)} -water "
                    f"{shlex.quote(config.water_model)} -ignh"
                ),
            ]
        )
    elif config.mode == "amber_pdb_sdf":
        resname = shlex.quote(config.ligand_resname.upper())
        cofactor_resname = shlex.quote("COF")
        commands.extend(
            [
                "command -v antechamber >/dev/null",
                "command -v tleap >/dev/null",
                (
                    'if [ -z "${AMBERHOME:-}" ]; then export AMBERHOME="$(cd '
                    '"$(dirname "$(command -v tleap)")/.." && pwd)"; fi'
                ),
                "python -c 'import parmed'",
            ]
        )
        if config.prepared_ligand_directory is None:
            commands.extend(
                [
                    "command -v parmchk2 >/dev/null",
                    (
                        "antechamber -i reference_ligand.sdf -fi sdf "
                        "-o ligand.mol2 -fo mol2 -at gaff2 -c bcc "
                        f"-nc {config.ligand_charge} -rn {resname} -s 2"
                    ),
                    "parmchk2 -i ligand.mol2 -f mol2 -o ligand.frcmod -s gaff2",
                ]
            )
        if not config.exclude_ligand_from_system:
            commands.append(
                "antechamber -i ligand.mol2 -fi mol2 -o ligand.pdb "
                f"-fo pdb -rn {resname} -s 0"
            )
        if _single_residue_lipids(config.structure):
            commands.extend(
                [
                    "command -v charmmlipid2amber.py >/dev/null",
                    (
                        "charmmlipid2amber.py -i charmm_lipids.pdb "
                        "-o amber_lipids.pdb"
                    ),
                ]
            )
        if (
            config.cofactor_force_field == "GAFF2"
            and config.prepared_cofactor_directory is None
        ):
            commands.extend(
                [
                    (
                        "antechamber -i cofactor.sdf -fi sdf -o cofactor.mol2 "
                        f"-fo mol2 -at gaff2 -c bcc -nc {config.cofactor_charge} "
                        f"-rn {cofactor_resname} -s 2"
                    ),
                    (
                        "parmchk2 -i cofactor.mol2 -f mol2 "
                        "-o cofactor.frcmod -s gaff2"
                    ),
                ]
            )
        commands.extend(
            [
                "python assemble_amber_complex.py",
                "tleap -f tleap.in",
            ]
        )
        if config.neutralize and config.system_type == "membrane":
            commands.extend(
                [
                    "python neutralize_amber_system.py",
                    'if [ -s tleap_neutralize.in ]; then tleap -f tleap_neutralize.in; fi',
                ]
            )
        commands.append("python convert_amber_to_gromacs.py")
    if config.box_lengths_nm is not None:
        lengths = " ".join(f"{value:g}" for value in config.box_lengths_nm)
        angles = " ".join(f"{value:g}" for value in config.box_angles_deg)
        triclinic = any(abs(value - 90.0) > 1.0e-6 for value in config.box_angles_deg)
        editconf = (
            f"{quoted_gmx} editconf -f system.gro -o system_with_pbc.gro "
            f"-box {lengths} -noc"
        )
        if triclinic:
            editconf += f" -bt triclinic -angles {angles}"
        commands.extend(
            [
                editconf,
                "mv system_with_pbc.gro system.gro",
                'echo "Applied explicit periodic box to system.gro"',
            ]
        )
    if config.system_type == "soluble" and config.exclude_ligand_from_system:
        commands.append(
            'echo "Unsolvated ligand-free receptor base prepared; final GROMACS "'
            '"validation is deferred until the hybrid complex is boxed, solvated, "'
            '"and ionized."'
        )
    else:
        commands.append(
            f"{quoted_gmx} grompp -f validate.mdp -c system.gro "
            "-r system.gro -p topol.top -o validation.tpr"
        )
    commands.extend(
        [
            'echo "Prepared system: $(pwd)/system.gro"',
            'echo "Matching topology: $(pwd)/topol.top"',
        ]
    )
    runner = task / "prepare_gromacs_system.sh"
    runner.write_text("\n".join(commands) + "\n", encoding="utf-8")
    runner.chmod(runner.stat().st_mode | 0o100)
    manifest = task / "system_preparation.json"
    payload = asdict(config)
    payload.update(
        {
            "structure": str(config.structure),
            "topology": str(config.topology) if config.topology else None,
            "force_field_directory": (
                str(config.force_field_directory)
                if config.force_field_directory
                else None
            ),
            "ligand_library": (
                str(config.ligand_library) if config.ligand_library else None
            ),
            "prepared_ligand_directory": (
                str(config.prepared_ligand_directory)
                if config.prepared_ligand_directory
                else None
            ),
            "prepared_cofactor_directory": (
                str(config.prepared_cofactor_directory)
                if config.prepared_cofactor_directory
                else None
            ),
            "cofactor_topology": (
                str(config.cofactor_topology)
                if config.cofactor_topology
                else None
            ),
            "cofactor_structure": (
                str(config.cofactor_structure) if config.cofactor_structure else None
            ),
            "output_directory": str(task),
            "prepared_structure": str(task / "system.gro"),
            "prepared_topology": str(task / "topol.top"),
        }
    )
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return runner, manifest
