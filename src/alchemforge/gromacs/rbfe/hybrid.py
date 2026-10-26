"""Stage and assemble edge-specific PMX hybrid systems for GROMACS RBFE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import LigandRBFEConfig
from .mapping import NetworkPlan

_PROTEIN_RESIDUES = {
    "ACE", "ALA", "ARG", "ASH", "ASN", "ASP", "CYS", "CYX", "GLH",
    "GLN", "GLU", "GLY", "HID", "HIE", "HIP", "HIS", "HSD", "HSE",
    "HSP", "ILE", "LEU", "LYN", "LYS", "MET", "NME", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}


@dataclass(frozen=True)
class PreparedLigand:
    name: str
    resname: str
    charge: int
    directory: Path
    gro: Path
    itp: Path


def _manifest_rows(path: Path) -> dict[str, dict[str, str]]:
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    result: dict[str, dict[str, str]] = {}
    for number, row in enumerate(rows, start=2):
        name = (row.get("ligand_name") or row.get("name") or "").strip()
        if not name:
            raise ValueError(f"Missing ligand_name in {path} line {number}")
        if name in result:
            raise ValueError(f"Duplicate ligand_name in {path}: {name}")
        result[name] = row
    return result


def _find_parameter_file(directory: Path, resname: str, suffix: str) -> Path:
    preferred = [
        directory / f"{resname}.{suffix}",
        directory / f"{resname}_GMX.{suffix}",
        directory / f"{resname}.acpype" / f"{resname}_GMX.{suffix}",
    ]
    for path in preferred:
        if path.is_file():
            return path
    candidates = [
        path
        for path in sorted(directory.rglob(f"*.{suffix}"))
        if "posre" not in path.name.casefold()
        and "opls" not in path.name.casefold()
    ]
    if not candidates:
        raise ValueError(
            f"No usable GROMACS {suffix.upper()} file was found for {resname} "
            f"under {directory}"
        )
    return candidates[0]


def read_prepared_ligands(path: Path) -> dict[str, PreparedLigand]:
    """Resolve each network name to its assigned-charge GRO and ITP files."""
    ligands: dict[str, PreparedLigand] = {}
    for name, row in _manifest_rows(path).items():
        resname = (row.get("resname") or "").strip().upper()
        directory_text = (row.get("directory") or "").strip()
        if not resname or not directory_text:
            raise ValueError(
                f"Prepared ligand {name!r} must define resname and directory"
            )
        directory = Path(directory_text).expanduser()
        if not directory.is_absolute():
            directory = path.parent / directory
        directory = directory.resolve()
        if not directory.is_dir():
            raise ValueError(
                f"Prepared parameter directory for {name!r} does not exist: "
                f"{directory}"
            )
        try:
            charge = round(float((row.get("net_charge") or "0").strip()))
        except ValueError as error:
            raise ValueError(f"Invalid net charge for prepared ligand {name!r}") from error
        ligands[name] = PreparedLigand(
            name=name,
            resname=resname,
            charge=charge,
            directory=directory,
            gro=_find_parameter_file(directory, resname, "gro"),
            itp=_find_parameter_file(directory, resname, "itp"),
        )
    return ligands


def _edge_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Ligand network contains no edges: {path}")
    return rows


def _charge_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["edge_id"].strip(): row
            for row in csv.DictReader(handle)
            if row.get("edge_id")
        }


def _copy_ligand(ligand: PreparedLigand, target: Path, label: str) -> None:
    shutil.copy2(ligand.gro, target / f"ligand_{label}.gro")
    shutil.copy2(ligand.itp, target / f"ligand_{label}.itp")


def stage_hybrid_systems(
    config: LigandRBFEConfig, network: NetworkPlan
) -> Path | None:
    """Create an executable PMX build plan for every network edge and both legs."""
    if config.prepared_complex_gro is None or config.prepared_complex_topology is None:
        return None
    if config.ligand_charge_manifest is None:
        raise ValueError(
            "Hybrid-system construction requires [ligands].charge_manifest from "
            "GROMACS Ligand Preparation"
        )
    prepared = read_prepared_ligands(config.ligand_charge_manifest)
    charge_rows = _charge_rows(config.project_dir / "network" / "charge_changes.csv")
    top_script = config.project_dir / "build_hybrid_systems.sh"
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]

    for row in _edge_rows(network.edges_csv):
        edge_id = row["edge_id"].strip()
        ligand_a_name = row["ligand_a"].strip()
        ligand_b_name = row["ligand_b"].strip()
        missing = [
            name for name in (ligand_a_name, ligand_b_name) if name not in prepared
        ]
        if missing:
            raise ValueError(
                "Prepared ligand manifest has no parameters for network ligand(s): "
                + ", ".join(missing)
            )
        ligand_a, ligand_b = prepared[ligand_a_name], prepared[ligand_b_name]
        edge_dir = config.project_dir / "edges" / edge_id
        input_dir = edge_dir / "input"
        hybrid_dir = edge_dir / "hybrid"
        mapping_dir = edge_dir / "mapping"
        for directory in (
            input_dir,
            hybrid_dir,
            mapping_dir,
            edge_dir / "complex" / "setup",
            edge_dir / "solvent" / "setup",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        _copy_ligand(ligand_a, input_dir, "a")
        _copy_ligand(ligand_b, input_dir, "b")

        charge = charge_rows.get(edge_id, {})
        coion_source = config.project_dir / "charge_correction" / edge_id / "coion.itp"
        if coion_source.is_file():
            shutil.copy2(coion_source, hybrid_dir / "coion.itp")
        spec = {
            "edge_id": edge_id,
            "ligand_a": ligand_a_name,
            "ligand_b": ligand_b_name,
            "base_gro": str(config.prepared_complex_gro.resolve()),
            "base_topology": str(config.prepared_complex_topology.resolve()),
            "edge_directory": str(edge_dir.resolve()),
            "gmx": config.gmx,
            "system_type": config.membrane.system_type,
            "force_field": config.force_field,
            "water_model": config.water_model,
            "box_shape": config.box_shape,
            "box_distance": config.box_distance,
            "neutralize": config.neutralize,
            "ion_concentration": config.ion_concentration,
            "protein_restraint_force": config.protein_restraint_force,
            "ligand_restraint_force": config.ligand_restraint_force,
            "cofactor_restraint_force": config.cofactor_restraint_force,
            "protein_residue_ids": list(config.protein_residue_ids),
            "protein_atom_names": list(config.protein_atom_names),
            "ligand_resnames": list(config.ligand_resnames),
            "ligand_atom_names": list(config.ligand_atom_names),
            "cofactor_resnames": list(config.cofactor_resnames),
            "cofactor_atom_names": list(config.cofactor_atom_names),
            "coion": charge.get("coion", ""),
            "coion_count": int(charge.get("coion_count") or 0),
        }
        spec_path = edge_dir / "hybrid_system.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

        pmx = shlex.split(config.pmx_command)
        build = edge_dir / "build_hybrid.sh"
        script = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(edge_dir))}",
            f"{shlex.quote(config.gmx)} editconf -f input/ligand_a.gro -o input/ligand_a.pdb",
            f"{shlex.quote(config.gmx)} editconf -f input/ligand_b.gro -o input/ligand_b.pdb",
            shlex.join(
                [
                    *pmx,
                    "atomMapping",
                    "-i1", "input/ligand_a.pdb",
                    "-i2", "input/ligand_b.pdb",
                    "-o1", "mapping/pairs1.dat",
                    "-o2", "mapping/pairs2.dat",
                    *(["--H2Hpolar"] if config.pmx_h2h_polar else []),
                    "-opdbm1", "mapping/mapped1.pdb",
                    "-opdbm2", "mapping/mapped2.pdb",
                ]
            ),
            shlex.join(
                [
                    *pmx,
                    "ligandHybrid",
                    "-i1", "input/ligand_a.pdb",
                    "-i2", "input/ligand_b.pdb",
                    "-itp1", "input/ligand_a.itp",
                    "-itp2", "input/ligand_b.itp",
                    "-pairs", "mapping/pairs1.dat",
                    "--fit",
                    "-oA", "hybrid/mergedA.pdb",
                    "-oB", "hybrid/mergedB.pdb",
                    "-oitp", "hybrid/merged.itp",
                    "-offitp", "hybrid/ffmerged.itp",
                ]
            ),
            shlex.quote(sys.executable)
            + " -m alchemforge.gromacs.rbfe.hybrid assemble-edge "
            + shlex.quote(str(spec_path)),
        ]
        build.write_text("\n".join(script) + "\n", encoding="utf-8")
        build.chmod(build.stat().st_mode | 0o100)
        commands.append(f"bash {shlex.quote(str(build))}")

    top_script.write_text("\n".join(commands) + "\n", encoding="utf-8")
    top_script.chmod(top_script.stat().st_mode | 0o100)
    return top_script


def _read_gro(path: Path) -> tuple[str, list[str], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {path}")
    try:
        count = int(lines[1].strip())
    except ValueError as error:
        raise ValueError(f"Invalid atom count in GRO file: {path}") from error
    atoms = lines[2 : 2 + count]
    if len(atoms) != count or len(lines) <= 2 + count:
        raise ValueError(f"Truncated GRO file: {path}")
    return lines[0], atoms, lines[2 + count]


def _renumber_gro_atom(line: str, residue: int, atom: int) -> str:
    if len(line) < 20:
        raise ValueError(f"Invalid GRO atom line: {line!r}")
    return f"{residue % 100000:5d}{line[5:15]}{atom % 100000:5d}{line[20:]}"


def _merge_gro(base: Path, ligand: Path, output: Path) -> None:
    _, base_atoms, box = _read_gro(base)
    _, ligand_atoms, _ = _read_gro(ligand)
    last_residue = int(base_atoms[-1][:5]) if base_atoms else 0
    residue_map: dict[int, int] = {}
    next_residue = last_residue
    merged = list(base_atoms)
    for line in ligand_atoms:
        source_residue = int(line[:5])
        if source_residue not in residue_map:
            next_residue += 1
            residue_map[source_residue] = next_residue
        merged.append(
            _renumber_gro_atom(line, residue_map[source_residue], len(merged) + 1)
        )
    output.write_text(
        "AlchemForge PMX hybrid complex\n"
        + f"{len(merged):5d}\n"
        + "\n".join(merged)
        + "\n"
        + box
        + "\n",
        encoding="utf-8",
    )


def _molecule_type(path: Path) -> str:
    in_section = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_section = line.casefold().replace(" ", "") == "[moleculetype]"
            continue
        if in_section and line and not line.startswith((";", "#")):
            return line.split()[0]
    raise ValueError(f"No [ moleculetype ] name found in {path}")


def _heavy_atom_indices(
    text: str,
    residue_names: set[str] | None = None,
    residue_ids: set[int] | None = None,
    atom_names: set[str] | None = None,
) -> list[int]:
    """Return heavy-atom indices from a GROMACS [ atoms ] section."""
    section = ""
    indices: list[int] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().casefold()
            continue
        if section != "atoms" or not stripped or stripped.startswith((";", "#")):
            continue
        fields = stripped.split()
        if len(fields) < 8 or not fields[0].isdigit():
            continue
        if residue_names and fields[3].upper() not in residue_names:
            continue
        if residue_ids and int(fields[2]) not in residue_ids:
            continue
        atom_name = fields[4].upper()
        if atom_names and atom_name not in atom_names:
            continue
        try:
            mass = float(fields[7])
        except ValueError:
            mass = 0.0
        if atom_name.startswith("H") or (mass and mass <= 2.1):
            continue
        indices.append(int(fields[0]))
    return indices


def _restraint_block(
    indices: list[int],
    macro: str,
    force: float,
    force_symbol: str,
) -> str:
    rows = "\n".join(
        f"{index:6d} 1 {force_symbol} {force_symbol} {force_symbol}"
        for index in indices
    )
    return (
        f"#ifdef {macro}\n"
        f"#ifndef {force_symbol}\n"
        f"#define {force_symbol} {force:g}\n"
        "#endif\n"
        "[ position_restraints ]\n"
        "; atom  type      fx      fy      fz\n"
        f"{rows}\n"
        "#endif\n"
    )


def _write_ligand_position_restraints(
    merged_itp: Path,
    output: Path,
    force: float,
    residue_names: tuple[str, ...] = (),
    atom_names: tuple[str, ...] = (),
) -> None:
    indices = _heavy_atom_indices(
        merged_itp.read_text(encoding="utf-8"),
        set(residue_names) or None,
        atom_names=set(atom_names) or None,
    )
    if not indices:
        raise ValueError(f"Hybrid ligand topology has no heavy atoms: {merged_itp}")
    output.write_text(
        "; Stage-dependent hybrid-ligand heavy-atom restraints.\n"
        "#ifndef POSRES_LIG_FC\n"
        f"#define POSRES_LIG_FC {force:g}\n"
        "#endif\n"
        "[ position_restraints ]\n"
        "; atom  type      fx      fy      fz\n"
        + "\n".join(
            f"{index:6d} 1 POSRES_LIG_FC POSRES_LIG_FC POSRES_LIG_FC"
            for index in indices
        )
        + "\n",
        encoding="utf-8",
    )


def _add_cofactor_restraints(
    text: str,
    source: Path,
    residue_names: tuple[str, ...],
    force: float,
    atom_names: tuple[str, ...] = (),
) -> str:
    """Attach conditional heavy-atom restraints to monolithic or included cofactors."""
    names = {name.upper() for name in residue_names}
    if not names:
        return text

    # A ParmEd/tleap topology often stores protein and cofactors in one
    # moleculetype. Insert the cofactor atom restraints before [ system ].
    indices = _heavy_atom_indices(
        text, names, atom_names=set(atom_names) or None
    )
    if indices:
        system = re.search(r"(?im)^\s*\[\s*system\s*\]", text)
        if system is None:
            raise ValueError("Base topology has no [ system ] section")
        text = (
            text[: system.start()]
            + _restraint_block(
                indices,
                "POSRES_COFACTOR",
                force,
                "POSRES_COFACTOR_FC",
            )
            + "\n"
            + text[system.start() :]
        )

    # A component topology may keep the cofactor in a directly included ITP.
    # Placing the restraint block immediately after that include associates it
    # with the included cofactor moleculetype.
    rewritten: list[str] = []
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"')
    for line in text.splitlines():
        rewritten.append(line)
        match = include_pattern.match(line)
        if not match:
            continue
        included = (source.parent / match.group(1)).resolve()
        if not included.is_file():
            continue
        included_text = included.read_text(encoding="utf-8", errors="replace")
        try:
            molecule = _molecule_type(included)
        except ValueError:
            continue
        included_indices = _heavy_atom_indices(
            included_text,
            None if molecule.upper() in names else names,
            atom_names=set(atom_names) or None,
        )
        if included_indices:
            rewritten.extend(
                _restraint_block(
                    included_indices,
                    "POSRES_COFACTOR",
                    force,
                    "POSRES_COFACTOR_FC",
                ).rstrip().splitlines()
            )
    return "\n".join(rewritten) + "\n"


def _add_protein_restraints(
    text: str,
    force: float,
    residue_ids: tuple[int, ...] = (),
    atom_names: tuple[str, ...] = (),
) -> str:
    """Add protein heavy-atom restraints for monolithic Amber topologies."""
    if re.search(r"(?m)^\s*#\s*ifdef\s+POSRES\b", text):
        return text
    indices = _heavy_atom_indices(
        text,
        _PROTEIN_RESIDUES,
        set(residue_ids) or None,
        set(atom_names) or None,
    )
    if not indices:
        # Component/pdb2gmx topologies normally provide their own POSRES block;
        # the stage MDP enables it with -DPOSRES.
        return text
    system = re.search(r"(?im)^\s*\[\s*system\s*\]", text)
    if system is None:
        raise ValueError("Base topology has no [ system ] section")
    return (
        text[: system.start()]
        + _restraint_block(indices, "POSRES", force, "POSRES_FC")
        + "\n"
        + text[system.start() :]
    )


def _atomtype_definitions(path: Path) -> dict[str, str]:
    """Return atom-type definition lines from every [ atomtypes ] section."""
    definitions: dict[str, str] = {}
    in_section = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        section = re.fullmatch(r"\[\s*([^]]+?)\s*]", stripped)
        if section:
            in_section = section.group(1).casefold() == "atomtypes"
            continue
        content = raw.split(";", 1)[0].strip()
        if not in_section or not content or content.startswith("#"):
            continue
        name = content.split()[0]
        previous = definitions.get(name)
        normalized = " ".join(content.split())
        if previous is not None and " ".join(previous.split()) != normalized:
            raise ValueError(f"Conflicting definitions for atom type {name!r} in {path}")
        definitions[name] = raw
    return definitions


def _hybrid_atomtypes(path: Path) -> set[str]:
    """Return A- and B-state atom types referenced by a hybrid molecule ITP."""
    result: set[str] = set()
    in_atoms = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        section = re.fullmatch(r"\[\s*([^]]+?)\s*]", stripped)
        if section:
            in_atoms = section.group(1).casefold() == "atoms"
            continue
        fields = raw.split(";", 1)[0].split()
        if not in_atoms or not fields or not fields[0].isdigit():
            continue
        if len(fields) < 8:
            raise ValueError(f"Malformed hybrid atom line in {path}: {raw}")
        result.add(fields[1])
        if len(fields) >= 11:
            result.add(fields[8])
    return result


def _complete_hybrid_atomtypes(
    base_topology: Path,
    ligand_itps: tuple[Path, Path],
    merged_itp: Path,
    ffmerged_itp: Path,
) -> None:
    """Supply ligand types PMX assumes are already present in the base force field."""
    base_types = _atomtype_definitions(base_topology)
    hybrid_types = _atomtype_definitions(ffmerged_itp)
    ligand_types: dict[str, str] = {}
    for ligand_itp in ligand_itps:
        for name, line in _atomtype_definitions(ligand_itp).items():
            previous = ligand_types.get(name)
            if previous is not None:
                old = " ".join(previous.split(";", 1)[0].split())
                new = " ".join(line.split(";", 1)[0].split())
                if old != new:
                    raise ValueError(
                        f"Ligand A/B define atom type {name!r} differently; "
                        "the parameter sets are incompatible"
                    )
            ligand_types[name] = line
    missing_lines = [
        line
        for name, line in ligand_types.items()
        if name not in base_types and name not in hybrid_types
    ]
    if missing_lines:
        with ffmerged_itp.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n; AlchemForge: ligand atom types absent from receptor base\n"
                + "\n".join(missing_lines)
                + "\n"
            )
        hybrid_types.update(_atomtype_definitions(ffmerged_itp))
    unresolved = _hybrid_atomtypes(merged_itp) - base_types.keys() - hybrid_types.keys()
    if unresolved:
        raise ValueError(
            "Hybrid topology references undefined atom type(s): "
            + ", ".join(sorted(unresolved))
            + ". Regenerate both ligands with a complete, compatible GROMACS ITP."
        )


def _write_solvent_ligand_atomtypes(
    force_field: str,
    ligand_itps: tuple[Path, Path],
    merged_itp: Path,
    ffmerged_itp: Path,
    output: Path,
) -> None:
    """Write ligand types needed outside the monolithic receptor topology.

    The complex leg inherits GAFF types embedded by ParmEd in the prepared
    receptor topology. The ligand-only solvent leg does not, so it needs an
    explicit copy of every referenced ligand type that is not already supplied
    by PMX's force field or ffmerged.itp.
    """
    force_types: set[str] = set()
    force_include = Path(_force_field_include(force_field))
    if force_include.is_file():
        for itp in force_include.parent.glob("*.itp"):
            force_types.update(_atomtype_definitions(itp))
    supplied = force_types | _atomtype_definitions(ffmerged_itp).keys()

    ligand_types: dict[str, str] = {}
    for ligand_itp in ligand_itps:
        for name, line in _atomtype_definitions(ligand_itp).items():
            previous = ligand_types.get(name)
            if previous is not None:
                old = " ".join(previous.split(";", 1)[0].split())
                new = " ".join(line.split(";", 1)[0].split())
                if old != new:
                    raise ValueError(
                        f"Ligand A/B define atom type {name!r} differently; "
                        "the parameter sets are incompatible"
                    )
            ligand_types[name] = line

    required = _hybrid_atomtypes(merged_itp) - supplied
    unresolved = required - ligand_types.keys()
    if unresolved:
        raise ValueError(
            "Solvent hybrid topology references atom type(s) unavailable from "
            "the force field or ligand ITPs: " + ", ".join(sorted(unresolved))
        )
    lines = ["[ atomtypes ]", "; AlchemForge: ligand types for the solvent leg"]
    lines.extend(ligand_types[name] for name in ligand_types if name in required)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _alchemize_base_ions(text: str, atom_indices: list[int]) -> str:
    """Add B-state zero charges and restraints to selected base-system ions."""
    if not atom_indices:
        return text
    molecule_sections = list(
        re.finditer(r"(?im)^\s*\[\s*moleculetype\s*\]", text)
    )
    if len(molecule_sections) != 1:
        raise ValueError(
            "Complex-leg co-ion conversion currently requires the prepared base "
            "topology to contain one monolithic [ moleculetype ] (the topology "
            "written by the Amber/ParmEd system builder)."
        )
    atoms_match = re.search(r"(?im)^\s*\[\s*atoms\s*\]", text)
    if atoms_match is None:
        raise ValueError("Prepared complex topology has no [ atoms ] section")
    next_section = re.search(r"(?im)^\s*\[", text[atoms_match.end() :])
    end = (
        atoms_match.end() + next_section.start()
        if next_section is not None
        else len(text)
    )
    selected = set(atom_indices)
    found: set[int] = set()
    rewritten: list[str] = []
    for raw in text[atoms_match.end() : end].splitlines():
        stripped = raw.strip()
        fields = stripped.split()
        if stripped and not stripped.startswith((";", "#")) and fields[0].isdigit():
            atom = int(fields[0])
            if atom in selected:
                if len(fields) < 8:
                    raise ValueError(f"Cannot alchemize malformed topology atom line: {raw}")
                # nr type resnr residue atom cgnr charge mass [typeB chargeB massB]
                fields = fields[:8] + [fields[1], "0.000000", fields[7]]
                raw = " ".join(fields)
                found.add(atom)
        rewritten.append(raw)
    if found != selected:
        missing = ", ".join(map(str, sorted(selected - found)))
        raise ValueError(f"Selected complex co-ion atom(s) absent from topology: {missing}")
    replacement = "\n" + "\n".join(rewritten) + "\n"
    text = text[: atoms_match.end()] + replacement + text[end:]
    system_section = re.search(r"(?im)^\s*\[\s*system\s*\]", text)
    assert system_section is not None
    restraints = (
        "; AlchemForge: restrain farthest alchemical co-ion(s)\n"
        "[ position_restraints ]\n"
        "; ai funct fcx fcy fcz\n"
        + "\n".join(f"{atom} 1 1000 1000 1000" for atom in atom_indices)
        + "\n\n"
    )
    return text[: system_section.start()] + restraints + text[system_section.start() :]


def _inject_complex_topology(
    source: Path,
    output: Path,
    molecule: str,
    coion_atom_indices: list[int],
    *,
    solvate: bool = False,
    force_field: str = "",
    water: str = "",
    hybrid_include_prefix: str = "../hybrid",
    protein_restraint_force: float = 1000.0,
    protein_residue_ids: tuple[int, ...] = (),
    protein_atom_names: tuple[str, ...] = (),
    cofactor_resnames: tuple[str, ...] = (),
    cofactor_restraint_force: float = 1000.0,
    cofactor_atom_names: tuple[str, ...] = (),
) -> None:
    text = source.read_text(encoding="utf-8")
    text = _add_protein_restraints(
        text,
        protein_restraint_force,
        protein_residue_ids,
        protein_atom_names,
    )
    text = _add_cofactor_restraints(
        text,
        source,
        cofactor_resnames,
        cofactor_restraint_force,
        cofactor_atom_names,
    )
    text = _alchemize_base_ions(text, coion_atom_indices)
    first_molecule = re.search(r"(?im)^\s*\[\s*moleculetype\s*\]", text)
    system_section = re.search(r"(?im)^\s*\[\s*system\s*\]", text)
    if first_molecule is None or system_section is None:
        raise ValueError(f"Base topology lacks required sections: {source}")
    text = (
        text[: first_molecule.start()]
        + f'#include "{hybrid_include_prefix}/ffmerged.itp"\n\n'
        + text[first_molecule.start() :]
    )
    system_section = re.search(r"(?im)^\s*\[\s*system\s*\]", text)
    assert system_section is not None
    includes = (
        f'#include "{hybrid_include_prefix}/merged.itp"\n'
        "#ifdef POSRES_LIG\n"
        f'#include "{hybrid_include_prefix}/posre_ligand.itp"\n'
        "#endif\n"
    )
    if solvate:
        force_include = _force_field_include(force_field)
        ff_dir = str(Path(force_include).parent)
        includes += (
            f'#include "{ff_dir}/{_water_itp(water)}"\n'
            f'#include "{ff_dir}/ions.itp"\n'
        )
    text = text[: system_section.start()] + includes + "\n" + text[system_section.start() :]
    text = text.rstrip() + f"\n{molecule:<20} 1\n"
    output.write_text(text, encoding="utf-8")


def _water_itp(water: str) -> str:
    value = water.casefold().replace("-", "")
    return {
        "tip3p": "tip3p.itp",
        "spc": "spc.itp",
        "spce": "spce.itp",
        "tip4p": "tip4p.itp",
        "tip4pew": "tip4pew.itp",
    }.get(value, f"{water.casefold()}.itp")


def _force_field_include(force_field: str) -> str:
    path = Path(force_field).expanduser()
    explicit = path if path.name.endswith(".ff") else Path(f"{path}.ff")
    if explicit.is_dir() and (explicit / "forcefield.itp").is_file():
        return str(explicit.resolve() / "forcefield.itp")

    # PMX mutation force fields live in PMX's package data and are not part of
    # GROMACS's normal GMXLIB search path. Resolve them explicitly so grompp
    # can find them when it runs from an edge's complex/ or solvent/ folder.
    try:
        import pmx
    except ImportError:
        pmx = None
    if pmx is not None:
        candidate = (
            Path(pmx.__file__).resolve().parent
            / "data"
            / "mutff"
            / explicit.name
        )
        if candidate.is_dir() and (candidate / "forcefield.itp").is_file():
            return str(candidate / "forcefield.itp")

    if path.name.endswith(".ff"):
        return f"{path}/forcefield.itp"
    return f"{force_field}.ff/forcefield.itp"


def _write_solvent_topology(
    output: Path,
    force_field: str,
    water: str,
    molecule: str,
    hybrid_include_prefix: str = "../hybrid",
) -> None:
    force_include = _force_field_include(force_field)
    ff_dir = str(Path(force_include).parent)
    lines = [
        f'#include "{force_include}"',
        f'#include "{hybrid_include_prefix}/solvent_atomtypes.itp"',
        f'#include "{hybrid_include_prefix}/ffmerged.itp"',
        f'#include "{hybrid_include_prefix}/merged.itp"',
        "#ifdef POSRES_LIG",
        f'#include "{hybrid_include_prefix}/posre_ligand.itp"',
        "#endif",
    ]
    lines.extend(
        [
            f'#include "{ff_dir}/{_water_itp(water)}"',
            f'#include "{ff_dir}/ions.itp"',
            "",
            "[ system ]",
            "PMX hybrid ligand in solvent",
            "",
            "[ molecules ]",
            f"{molecule:<20} 1",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(command: list[str], cwd: Path, *, stdin: str | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, input=stdin, text=True, check=False)
    if result.returncode:
        raise ValueError(
            f"Command failed with exit code {result.returncode}: {shlex.join(command)}"
        )


def _write_validation_mdp(path: Path) -> None:
    path.write_text(
        "integrator = steep\n"
        "nsteps = 50\n"
        "cutoff-scheme = Verlet\n"
        "coulombtype = PME\n"
        "rcoulomb = 1.0\n"
        "vdwtype = Cut-off\n"
        "rvdw = 1.0\n"
        "constraints = none\n"
        "free-energy = yes\n"
        "init-lambda-state = 0\n"
        "fep-lambdas = 0 1\n",
        encoding="utf-8",
    )


def _contains_water(path: Path) -> bool:
    _, atoms, _ = _read_gro(path)
    return any(line[5:10].strip().upper() in _WATER_RESIDUES for line in atoms)


def _solvate_and_ionize(
    *,
    gmx: str,
    directory: Path,
    coordinates: Path,
    topology: Path,
    box_shape: str,
    box_distance: float,
    neutralize: bool,
    ion_concentration: float,
) -> tuple[Path, Path]:
    """Box, solvate, and optionally neutralize/add salt to one RBFE leg."""
    boxed = directory / "boxed.gro"
    solvated = directory / "solvated.gro"
    output = directory / "system.gro"
    mdp = directory / "ions.mdp"
    tpr = directory / "ions.tpr"
    _run(
        [
            gmx,
            "editconf",
            "-f",
            str(coordinates),
            "-o",
            str(boxed),
            "-bt",
            box_shape,
            "-d",
            str(box_distance),
        ],
        directory,
    )
    _run(
        [
            gmx,
            "solvate",
            "-cp",
            str(boxed),
            "-cs",
            "spc216.gro",
            "-p",
            str(topology),
            "-o",
            str(solvated),
        ],
        directory,
    )
    _write_validation_mdp(mdp)
    if not neutralize and ion_concentration == 0:
        shutil.copy2(solvated, output)
        return output, mdp
    _run(
        [
            gmx,
            "grompp",
            "-f",
            str(mdp),
            "-c",
            str(solvated),
            "-r",
            str(solvated),
            "-p",
            str(topology),
            "-o",
            str(tpr),
            "-maxwarn",
            "1",
        ],
        directory,
    )
    command = [
        gmx,
        "genion",
        "-s",
        str(tpr),
        "-o",
        str(output),
        "-p",
        str(topology),
        "-pname",
        "NA",
        "-nname",
        "CL",
    ]
    if neutralize:
        command.append("-neutral")
    if ion_concentration > 0:
        command.extend(["-conc", str(ion_concentration)])
    _run(command, directory, stdin="SOL\n")
    return output, mdp


_WATER_RESIDUES = {"SOL", "HOH", "WAT", "TIP3", "TIP4", "SPC", "SPCE"}
_ION_NAMES = {
    "NA": {"NA", "SOD"},
    "CL": {"CL", "CLA"},
}


def _gro_position(line: str) -> tuple[float, float, float]:
    return float(line[20:28]), float(line[28:36]), float(line[36:44])


def _minimum_image_distance(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    box: np.ndarray,
) -> float:
    delta = np.asarray(left) - np.asarray(right)
    fractional = np.linalg.solve(box, delta)
    fractional -= np.round(fractional)
    wrapped = box @ fractional
    return math.sqrt(float(wrapped @ wrapped))


def _gro_box_matrix(values: list[float]) -> np.ndarray:
    if len(values) == 3:
        return np.diag(values)
    if len(values) == 9:
        # GRO order: v1x v2y v3z v1y v1z v2x v2z v3x v3y.
        return np.asarray(
            [
                [values[0], values[5], values[7]],
                [values[3], values[1], values[8]],
                [values[4], values[6], values[2]],
            ]
        )
    raise ValueError("GRO box must contain either 3 or 9 values")


def _select_farthest_ions(path: Path, ion: str, count: int) -> list[int]:
    """Return one-based atom indices of ions farthest from every solute atom."""
    if not count:
        return []
    _, atoms, box_line = _read_gro(path)
    box_values = [float(value) for value in box_line.split()]
    if len(box_values) < 3:
        raise ValueError(f"GRO box has fewer than three vectors: {path}")
    box = _gro_box_matrix(box_values)
    all_ions = set().union(*_ION_NAMES.values())
    solute = [
        _gro_position(line)
        for line in atoms
        if line[5:10].strip().upper() not in (_WATER_RESIDUES | all_ions)
        and not line[10:15].strip().upper().startswith("H")
    ]
    if not solute:
        raise ValueError(f"No non-water solute atoms found while selecting {ion}: {path}")
    candidates = [
        (index, _gro_position(line))
        for index, line in enumerate(atoms, start=1)
        if line[5:10].strip().upper() in _ION_NAMES[ion]
        or line[10:15].strip().upper() in _ION_NAMES[ion]
    ]
    if len(candidates) < count:
        raise ValueError(
            f"The {path.name} system has {len(candidates)} {ion} ion(s), but "
            f"this charge-changing edge requires {count}. Add sufficient salt "
            "before building hybrid systems."
        )
    ranked = sorted(
        candidates,
        key=lambda item: min(
            _minimum_image_distance(item[1], position, box) for position in solute
        ),
        reverse=True,
    )
    return [index for index, _ in ranked[:count]]


def _convert_solvent_coions(
    topology: Path, coordinates: Path, coion_itp: Path, ion: str, count: int
) -> list[int]:
    selected_indices = _select_farthest_ions(coordinates, ion, count)
    selected = set(selected_indices)
    title, atoms, box = _read_gro(coordinates)
    ordinary = [line for index, line in enumerate(atoms, start=1) if index not in selected]
    alchemical = [line for index, line in enumerate(atoms, start=1) if index in selected]
    reordered = ordinary + alchemical
    reordered = [
        _renumber_gro_atom(line, int(line[:5]), index)
        for index, line in enumerate(reordered, start=1)
    ]
    coordinates.write_text(
        f"{title}\n{len(reordered):5d}\n" + "\n".join(reordered) + f"\n{box}\n",
        encoding="utf-8",
    )
    text = topology.read_text(encoding="utf-8")
    system = re.search(r"(?im)^\s*\[\s*system\s*\]", text)
    if system is None:
        raise ValueError(f"Solvent topology lacks [ system ]: {topology}")
    text = text[: system.start()] + '#include "../hybrid/coion.itp"\n\n' + text[system.start() :]
    molecule_match = re.search(
        rf"(?im)^(\s*{re.escape(ion)}\s+)(\d+)(\s*(?:;.*)?)$", text
    )
    if molecule_match is None:
        aliases = "|".join(re.escape(value) for value in _ION_NAMES[ion])
        molecule_match = re.search(rf"(?im)^(\s*(?:{aliases})\s+)(\d+)(\s*(?:;.*)?)$", text)
    if molecule_match is None or int(molecule_match.group(2)) < count:
        raise ValueError(f"Cannot locate {count} ordinary {ion} molecule(s) in {topology}")
    remaining = int(molecule_match.group(2)) - count
    text = (
        text[: molecule_match.start()]
        + molecule_match.group(1)
        + str(remaining)
        + molecule_match.group(3)
        + text[molecule_match.end() :]
    )
    text = text.rstrip() + f"\n{ion}_COION{'':<9} {count}\n"
    topology.write_text(text, encoding="utf-8")
    if not coion_itp.is_file():
        raise ValueError(f"Missing generated co-ion topology: {coion_itp}")
    return selected_indices


def assemble_edge(spec_path: Path) -> None:
    """Assemble complex and solvated ligand legs after PMX ligandHybrid."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    edge = Path(spec["edge_directory"])
    hybrid = edge / "hybrid"
    complex_dir = edge / "complex" / "setup"
    solvent_dir = edge / "solvent" / "setup"
    for required in (hybrid / "mergedA.pdb", hybrid / "merged.itp", hybrid / "ffmerged.itp"):
        if not required.is_file():
            raise ValueError(f"PMX did not create required hybrid file: {required}")
    _complete_hybrid_atomtypes(
        Path(spec["base_topology"]),
        (edge / "input/ligand_a.itp", edge / "input/ligand_b.itp"),
        hybrid / "merged.itp",
        hybrid / "ffmerged.itp",
    )
    _write_solvent_ligand_atomtypes(
        str(spec["force_field"]),
        (edge / "input/ligand_a.itp", edge / "input/ligand_b.itp"),
        hybrid / "merged.itp",
        hybrid / "ffmerged.itp",
        hybrid / "solvent_atomtypes.itp",
    )
    _write_ligand_position_restraints(
        hybrid / "merged.itp",
        hybrid / "posre_ligand.itp",
        float(spec.get("ligand_restraint_force", 1000.0)),
        tuple(spec.get("ligand_resnames", [])),
        tuple(spec.get("ligand_atom_names", [])),
    )
    molecule = _molecule_type(hybrid / "merged.itp")
    gmx = str(spec["gmx"])
    membrane = str(spec.get("system_type", "soluble")) == "membrane"
    coion = str(spec.get("coion") or "")
    coion_count = int(spec.get("coion_count") or 0)

    merged_gro = hybrid / "mergedA.gro"
    _run([gmx, "editconf", "-f", str(hybrid / "mergedA.pdb"), "-o", str(merged_gro)], edge)

    base_gro = Path(spec["base_gro"])
    complex_initial = complex_dir / "complex_unsolvated.gro"
    _merge_gro(base_gro, merged_gro, complex_initial)
    # Membrane systems retain their prepared membrane/water/ion box. Soluble
    # systems must start from an unsolvated receptor and are boxed and solvated
    # after the edge-specific hybrid ligand has been inserted.
    if not membrane and _contains_water(base_gro):
        raise ValueError(
            "Automatic soluble-complex solvation requires an unsolvated prepared "
            "receptor base. The selected base GRO already contains water. Remove "
            "the old solvent or use a membrane system to preserve its box."
        )
    complex_coions = (
        _select_farthest_ions(base_gro, coion, coion_count)
        if membrane and coion_count
        else []
    )
    _inject_complex_topology(
        Path(spec["base_topology"]),
        complex_dir / "topol.top",
        molecule,
        complex_coions,
        solvate=not membrane,
        force_field=str(spec["force_field"]),
        water=str(spec["water_model"]),
        hybrid_include_prefix="../../hybrid",
        protein_restraint_force=float(
            spec.get("protein_restraint_force", 1000.0)
        ),
        protein_residue_ids=tuple(spec.get("protein_residue_ids", [])),
        protein_atom_names=tuple(spec.get("protein_atom_names", [])),
        cofactor_resnames=tuple(spec.get("cofactor_resnames", [])),
        cofactor_restraint_force=float(
            spec.get("cofactor_restraint_force", 1000.0)
        ),
        cofactor_atom_names=tuple(spec.get("cofactor_atom_names", [])),
    )
    complex_mdp = complex_dir / "validate.mdp"
    _write_validation_mdp(complex_mdp)
    complex_coordinates: Path
    if membrane:
        complex_coordinates = complex_dir / "system.gro"
        shutil.copy2(complex_initial, complex_coordinates)
    else:
        complex_coordinates, complex_mdp = _solvate_and_ionize(
            gmx=gmx,
            directory=complex_dir,
            coordinates=complex_initial,
            topology=complex_dir / "topol.top",
            box_shape=str(spec["box_shape"]),
            box_distance=float(spec["box_distance"]),
            neutralize=bool(spec["neutralize"]),
            ion_concentration=float(spec["ion_concentration"]),
        )
        if coion_count:
            complex_coions = _convert_solvent_coions(
                complex_dir / "topol.top",
                complex_coordinates,
                hybrid / "coion.itp",
                coion,
                coion_count,
            )
    _run(
        [
            gmx,
            "grompp",
            "-f",
            str(complex_mdp),
            "-c",
            str(complex_coordinates),
            "-r",
            str(complex_coordinates),
            "-p",
            str(complex_dir / "topol.top"),
            "-o",
            str(complex_dir / "validated.tpr"),
        ],
        complex_dir,
    )

    _write_solvent_topology(
        solvent_dir / "topol.top",
        str(spec["force_field"]),
        str(spec["water_model"]),
        molecule,
        hybrid_include_prefix="../../hybrid",
    )
    solvent_coordinates, ions_mdp = _solvate_and_ionize(
        gmx=gmx,
        directory=solvent_dir,
        coordinates=merged_gro,
        topology=solvent_dir / "topol.top",
        box_shape=str(spec["box_shape"]),
        box_distance=float(spec["box_distance"]),
        neutralize=bool(spec["neutralize"]),
        ion_concentration=float(spec["ion_concentration"]),
    )
    solvent_coions: list[int] = []
    if coion_count:
        solvent_coions = _convert_solvent_coions(
            solvent_dir / "topol.top",
            solvent_coordinates,
            hybrid / "coion.itp",
            coion,
            coion_count,
        )
        _run(
            [
                gmx, "grompp", "-f", str(ions_mdp),
                "-c", str(solvent_coordinates),
                "-r", str(solvent_coordinates),
                "-p", str(solvent_dir / "topol.top"),
                "-o", str(solvent_dir / "validated.tpr"),
            ],
            solvent_dir,
        )
    elif not (solvent_dir / "validated.tpr").is_file():
        _run(
            [
                gmx,
                "grompp",
                "-f",
                str(ions_mdp),
                "-c",
                str(solvent_coordinates),
                "-r",
                str(solvent_coordinates),
                "-p",
                str(solvent_dir / "topol.top"),
                "-o",
                str(solvent_dir / "validated.tpr"),
            ],
            solvent_dir,
        )
    summary = {
        "edge_id": spec["edge_id"],
        "transformation": f"{spec['ligand_a']} -> {spec['ligand_b']}",
        "hybrid_molecule_type": molecule,
        "solvation": {
            "complex_leg": "preserved" if membrane else "rebuilt",
            "solvent_leg": "rebuilt",
            "box_shape": spec["box_shape"],
            "box_distance_nm": spec["box_distance"],
            "neutralize": spec["neutralize"],
            "ion_concentration_molar": spec["ion_concentration"],
        },
        "legs": {
            "complex": {
                "coordinates": "complex/setup/system.gro",
                "topology": "complex/setup/topol.top",
            },
            "solvent": {
                "coordinates": "solvent/setup/system.gro",
                "topology": "solvent/setup/topol.top",
            },
        },
        "coion": {
            "residue": coion,
            "count": coion_count,
            "selection": "farthest minimum-image distance from non-water solute",
            "restraint_kj_mol_nm2": 1000,
            "complex_base_atom_indices": complex_coions,
            "solvent_pre_reorder_atom_indices": solvent_coions,
        },
    }
    (edge / "hybrid_system_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assemble = commands.add_parser("assemble-edge")
    assemble.add_argument("spec", type=Path)
    args = parser.parse_args(argv)
    if args.command == "assemble-edge":
        assemble_edge(args.spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
