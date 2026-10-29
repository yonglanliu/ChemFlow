"""Remove existing hydrogens and repair PDB files with PDBFixer."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PDBFixTask:
    source: Path
    output: Path
    label: str
    repair: bool = True


def _path_or_none(value: str | None) -> Path | None:
    return Path(value) if value else None


def is_hydrogen_record(line: str) -> bool:
    """Return true for an ATOM/HETATM record whose element is hydrogen."""
    if not line.startswith(("ATOM  ", "HETATM")):
        return False
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element == "H"
    atom_name = line[12:16].strip().upper() if len(line) >= 16 else ""
    inferred = atom_name.lstrip("0123456789")
    return inferred.startswith("H")


def remove_hydrogens(source: Path, destination: Path) -> tuple[int, int]:
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    hydrogen_serials = {
        line[6:11].strip()
        for line in lines
        if is_hydrogen_record(line) and len(line) >= 11
    }
    kept: list[str] = []
    atom_count = 0
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            atom_count += 1
        if is_hydrogen_record(line):
            continue
        if line.startswith("ANISOU") and line[6:11].strip() in hydrogen_serials:
            continue
        if line.startswith("CONECT"):
            serials = line[6:].split()
            if not serials or serials[0] in hydrogen_serials:
                continue
            heavy_serials = [
                serial for serial in serials if serial not in hydrogen_serials
            ]
            if len(heavy_serials) < 2:
                continue
            line = (
                "CONECT"
                + "".join(f"{int(serial):5d}" for serial in heavy_serials)
                + "\n"
            )
        kept.append(line)
    if atom_count == 0:
        raise ValueError(f"PDB contains no ATOM or HETATM records: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(kept), encoding="utf-8")
    return atom_count, len(hydrogen_serials)


def discover_tasks(
    main_pdb: Path | None,
    output_directory: Path,
    *,
    include_components: bool = True,
    protein_pdb: Path | None = None,
    cofactor_pdb: Path | None = None,
    membrane_pdb: Path | None = None,
    solvent_pdb: Path | None = None,
) -> list[PDBFixTask]:
    tasks: list[PDBFixTask] = []
    seen: set[Path] = set()
    if main_pdb is not None:
        resolved_main = main_pdb.expanduser().resolve()
        tasks.append(
            PDBFixTask(
                resolved_main, output_directory / "fixed_structure.pdb", "structure"
            )
        )
        seen.add(resolved_main)
    selected_components = (
        (protein_pdb, "fixed_protein.pdb", "protein", True),
        (cofactor_pdb, "fixed_cofactor.pdb", "cofactor", False),
        (membrane_pdb, "fixed_membrane.pdb", "membrane", False),
        (solvent_pdb, "fixed_solvent.pdb", "solvent", False),
    )
    for selected, output_name, label, repair in selected_components:
        if selected is None:
            continue
        source = selected.expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".pdb":
            raise ValueError(f"Selected {label} must be an existing PDB: {source}")
        if source not in seen:
            tasks.append(
                PDBFixTask(source, output_directory / output_name, label, repair)
            )
            seen.add(source)

    if include_components and main_pdb is not None:
        for filename, output_name, label, repair in (
            ("protein.pdb", "fixed_protein.pdb", "protein", True),
            ("cofactor.pdb", "fixed_cofactor.pdb", "cofactor", False),
            ("membrane.pdb", "fixed_membrane.pdb", "membrane", False),
            ("solvent.pdb", "fixed_solvent.pdb", "solvent", False),
        ):
            source = (main_pdb.parent / filename).resolve()
            if source.is_file() and source not in seen:
                tasks.append(
                    PDBFixTask(source, output_directory / output_name, label, repair)
                )
                seen.add(source)
    return tasks


def stage(
    main_pdb: Path | None,
    output_directory: Path,
    task_directory: Path,
    *,
    include_components: bool = True,
    protein_pdb: Path | None = None,
    cofactor_pdb: Path | None = None,
    membrane_pdb: Path | None = None,
    solvent_pdb: Path | None = None,
    box_source_pdb: Path | None = None,
    assemble_system: bool = False,
    overwrite: bool = False,
    pdbfixer_command: str = "pdbfixer",
) -> tuple[Path, Path, list[PDBFixTask]]:
    main_pdb = main_pdb.expanduser().resolve() if main_pdb is not None else None
    output_directory = output_directory.expanduser().resolve()
    if main_pdb is not None and (
        not main_pdb.is_file() or main_pdb.suffix.lower() != ".pdb"
    ):
        raise ValueError(f"Main structure must be an existing PDB: {main_pdb}")
    if not pdbfixer_command.strip():
        raise ValueError("PDBFixer command cannot be empty")
    if box_source_pdb is not None:
        box_source_pdb = box_source_pdb.expanduser().resolve()
        if not box_source_pdb.is_file() or box_source_pdb.suffix.lower() != ".pdb":
            raise ValueError(f"PBC source must be an existing PDB: {box_source_pdb}")
    tasks = discover_tasks(
        main_pdb,
        output_directory,
        include_components=include_components,
        protein_pdb=protein_pdb,
        cofactor_pdb=cofactor_pdb,
        membrane_pdb=membrane_pdb,
        solvent_pdb=solvent_pdb,
    )
    if not tasks:
        raise ValueError("Select at least one PDB file to repair")
    task_by_label = {task.label: task for task in tasks}
    system: dict[str, str | None] | None = None
    if assemble_system:
        if "protein" not in task_by_label:
            raise ValueError("A protein PDB is required to assemble complex.pdb")
        has_membrane = "membrane" in task_by_label
        has_solvent = "solvent" in task_by_label
        if has_membrane != has_solvent:
            raise ValueError(
                "Membrane-system assembly requires both membrane and solvent PDBs"
            )
        system = {
            "protein": str(task_by_label["protein"].output),
            "cofactor": (
                str(task_by_label["cofactor"].output)
                if "cofactor" in task_by_label
                else None
            ),
            "membrane": (
                str(task_by_label["membrane"].output) if has_membrane else None
            ),
            "solvent": str(task_by_label["solvent"].output) if has_solvent else None,
            "box_source": str(box_source_pdb) if box_source_pdb else None,
            "output": str(output_directory / "complex.pdb"),
            "cif_output": str(output_directory / "complex.cif"),
        }
    if not overwrite:
        existing = [str(task.output) for task in tasks if task.output.exists()]
        if system is not None and Path(str(system["output"])).exists():
            existing.append(str(system["output"]))
        if system is not None and Path(str(system["cif_output"])).exists():
            existing.append(str(system["cif_output"]))
        if existing:
            raise ValueError(
                "Refusing to overwrite existing fixed structure(s): "
                + ", ".join(existing)
            )
    task_directory.mkdir(parents=True, exist_ok=True)
    manifest = task_directory / "pdbfixer_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "pdbfixer_command": pdbfixer_command,
                "overwrite": overwrite,
                "system": system,
                "tasks": [
                    {
                        "source": str(task.source),
                        "output": str(task.output),
                        "label": task.label,
                        "repair": task.repair,
                    }
                    for task in tasks
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    runner = task_directory / "run_pdbfixer.sh"
    python = shlex.quote(sys.executable)
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        'MANIFEST="${1:?Usage: $0 /path/to/pdbfixer_manifest.json}"\n'
        f"{python} -m alchemforge.structure.pdbfix \"$MANIFEST\"\n",
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, manifest, tasks


def assemble_complex(
    protein: Path,
    output: Path,
    *,
    cofactor: Path | None = None,
    membrane: Path | None = None,
    solvent: Path | None = None,
    box_source: Path | None = None,
    cif_output: Path | None = None,
) -> None:
    """Assemble fixed components into a single PDB with OpenMM Modeller."""
    try:
        from openmm import unit
        from openmm.app import Modeller, PDBFile, PDBxFile
    except ImportError as error:
        raise ValueError(
            "OpenMM is required to assemble complex.pdb; activate the "
            "AlchemForge environment"
        ) from error

    def load_pdb(path: Path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pdb = PDBFile(str(path))
            return pdb.topology, pdb.positions
        except (IndexError, TypeError, ValueError):
            try:
                import mdtraj
            except ImportError as error:
                raise ValueError(
                    f"OpenMM cannot read extended PDB identifiers in {path}; "
                    "install MDTraj or convert the component to mmCIF"
                ) from error
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trajectory = mdtraj.load_pdb(str(path))
            return trajectory.topology.to_openmm(), trajectory.openmm_positions(0)

    def box_values(topology) -> tuple[tuple[float, float, float], ...] | None:
        vectors = topology.getPeriodicBoxVectors()
        if vectors is None:
            return None
        return tuple(
            tuple(float(value) for value in vector.value_in_unit(unit.nanometer))
            for vector in vectors
        )

    def boxes_match(
        first: tuple[tuple[float, float, float], ...],
        second: tuple[tuple[float, float, float], ...],
    ) -> bool:
        return all(
            math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-4)
            for first_vector, second_vector in zip(first, second)
            for a, b in zip(first_vector, second_vector)
        )

    protein_topology, protein_positions = load_pdb(protein)
    modeller = Modeller(protein_topology, protein_positions)
    candidates: list[tuple[str, tuple[tuple[float, float, float], ...], object]] = []

    def collect_box(label: str, topology) -> None:
        values = box_values(topology)
        vectors = topology.getPeriodicBoxVectors()
        if values is not None and vectors is not None:
            candidates.append((label, values, vectors))

    collect_box("protein", protein_topology)
    for label, component in (
        ("cofactor", cofactor),
        ("membrane", membrane),
        ("solvent", solvent),
    ):
        if component is None:
            continue
        component_topology, component_positions = load_pdb(component)
        collect_box(label, component_topology)
        modeller.add(component_topology, component_positions)

    if box_source is not None:
        source_topology, _source_positions = load_pdb(box_source)
        source_values = box_values(source_topology)
        source_vectors = source_topology.getPeriodicBoxVectors()
        if source_values is None or source_vectors is None:
            raise ValueError(
                f"PBC source PDB has no CRYST1 periodic box record: {box_source}"
            )
        candidates.insert(0, ("PBC source", source_values, source_vectors))

    if candidates:
        reference_label, reference_values, reference_vectors = candidates[0]
        conflicts = [
            label
            for label, values, _vectors in candidates[1:]
            if not boxes_match(reference_values, values)
        ]
        if conflicts:
            raise ValueError(
                f"Periodic box in {reference_label} conflicts with: "
                + ", ".join(conflicts)
            )
        modeller.topology.setPeriodicBoxVectors(reference_vectors)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.modeller-tmp.pdb")
    with temporary.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    _restore_four_character_residue_names(temporary, modeller.topology)
    temporary.replace(output)
    if cif_output is not None:
        cif_output.parent.mkdir(parents=True, exist_ok=True)
        cif_temporary = cif_output.with_name(f".{cif_output.stem}.modeller-tmp.cif")
        with cif_temporary.open("w", encoding="utf-8") as handle:
            PDBxFile.writeFile(
                modeller.topology, modeller.positions, handle, keepIds=False
            )
        cif_temporary.replace(cif_output)


def _restore_four_character_residue_names(path: Path, topology) -> None:
    """Restore GROMACS' four-character PDB residue-name extension.

    OpenMM reads the fourth residue-name character from PDB column 21 but its
    writer truncates every residue name to the standard three columns.  GROMACS
    membrane force fields commonly require four-character names such as POPC.
    """

    atoms = iter(topology.atoms())
    last_residue_name = ""
    restored: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        line = raw_line
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                atom = next(atoms)
            except StopIteration as error:
                raise ValueError(
                    "OpenMM PDB output contains more atoms than its topology"
                ) from error
            last_residue_name = atom.residue.name
            if len(last_residue_name) == 4:
                line = line[:17] + last_residue_name + line[21:]
        elif line.startswith("TER   ") and len(last_residue_name) == 4:
            line = line[:17] + last_residue_name + line[21:]
        restored.append(line)
    try:
        next(atoms)
    except StopIteration:
        pass
    else:
        raise ValueError("OpenMM PDB output contains fewer atoms than its topology")
    path.write_text("".join(restored), encoding="utf-8")


def _restore_residue_names_from_source(source: Path, output: Path) -> int:
    """Restore four-character names that PDBFixer truncated in a repaired PDB."""

    names: dict[tuple[str, str, str], str] = {}
    conflicts: set[tuple[str, str, str]] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        name = line[17:21].strip()
        if len(name) != 4:
            continue
        key = (line[21], line[22:26].strip(), line[26])
        previous = names.get(key)
        if previous is not None and previous != name:
            conflicts.add(key)
        else:
            names[key] = name
    for key in conflicts:
        names.pop(key, None)
    if not names:
        return 0

    restored_count = 0
    restored: list[str] = []
    for raw_line in output.read_text(encoding="utf-8").splitlines(keepends=True):
        line = raw_line
        if line.startswith(("ATOM  ", "HETATM", "TER   ")) and len(line) >= 27:
            key = (line[21], line[22:26].strip(), line[26])
            name = names.get(key)
            if name is not None and line[17:20].strip() == name[:3]:
                line = line[:17] + name + line[21:]
                restored_count += 1
        restored.append(line)
    output.write_text("".join(restored), encoding="utf-8")
    return restored_count


def run_manifest(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    command = str(payload.get("pdbfixer_command", "pdbfixer"))
    if shutil.which(command) is None:
        raise ValueError(
            f"PDBFixer command is unavailable: {command}. Activate the "
            "AlchemForge environment before launching the desktop."
        )
    overwrite = bool(payload.get("overwrite", False))
    work = path.parent / "dehydrogenated"
    work.mkdir(parents=True, exist_ok=True)
    for item in payload["tasks"]:
        source = Path(item["source"])
        output = Path(item["output"])
        label = str(item["label"])
        if output.exists() and not overwrite:
            raise ValueError(f"Output already exists: {output}")
        if not bool(item.get("repair", True)):
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
            print(
                f"{label}: preserved without PDBFixer to avoid corrupting "
                "nonstandard component topology"
            )
            continue
        no_h = work / f"{label}_noh.pdb"
        atoms, removed = remove_hydrogens(source, no_h)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.pdbfixer-tmp.pdb")
        if temporary.exists():
            temporary.unlink()
        print(f"{label}: removed {removed} hydrogen records from {atoms} atoms")
        result = subprocess.run(
            [
                command,
                str(no_h),
                f"--output={temporary}",
                "--add-atoms=all",
            ],
            check=False,
        )
        if result.returncode != 0:
            temporary.unlink(missing_ok=True)
            return result.returncode
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ValueError(f"PDBFixer did not create a valid output for {source}")
        restored = _restore_residue_names_from_source(no_h, temporary)
        if restored:
            print(
                f"{label}: restored four-character residue names in "
                f"{restored} PDB records"
            )
        temporary.replace(output)
        print(f"{label}: fixed structure written to {output}")
    system = payload.get("system")
    if system is not None:
        complex_output = Path(system["output"])
        cif_output = Path(system["cif_output"])
        if complex_output.exists() and not overwrite:
            raise ValueError(f"Output already exists: {complex_output}")
        if cif_output.exists() and not overwrite:
            raise ValueError(f"Output already exists: {cif_output}")
        cofactor = Path(system["cofactor"]) if system.get("cofactor") else None
        membrane = Path(system["membrane"]) if system.get("membrane") else None
        solvent = Path(system["solvent"]) if system.get("solvent") else None
        box_source = Path(system["box_source"]) if system.get("box_source") else None
        assemble_complex(
            Path(system["protein"]),
            complex_output,
            cofactor=cofactor,
            membrane=membrane,
            solvent=solvent,
            box_source=box_source,
            cif_output=cif_output,
        )
        components = ["protein"]
        if cofactor is not None:
            components.append("cofactor")
        if membrane is not None:
            components.extend(("membrane", "solvent"))
        print(f"system: assembled {' + '.join(components)} into {complex_output}")
        print(f"system: wrote large-system-safe structure to {cif_output}")
    return 0


def run_cli(args: argparse.Namespace) -> int:
    """Run PDB repair and system assembly from the public CLI."""
    output_directory = Path(args.output_directory).expanduser().resolve()
    work_directory = output_directory / ".alchemforge-pdbfix"
    _, manifest, tasks = stage(
        None,
        output_directory,
        work_directory,
        include_components=False,
        protein_pdb=Path(args.protein),
        cofactor_pdb=_path_or_none(args.cofactor),
        membrane_pdb=_path_or_none(args.membrane),
        solvent_pdb=_path_or_none(args.solvent),
        box_source_pdb=_path_or_none(args.pbc_source),
        assemble_system=True,
        overwrite=args.overwrite,
        pdbfixer_command=args.pdbfixer_command,
    )
    result = run_manifest(manifest)
    if result == 0:
        print("Fixed components: " + ", ".join(task.output.name for task in tasks))
        print(f"Final system: {output_directory / 'complex.pdb'}")
        print(f"Visualization-safe system: {output_directory / 'complex.cif'}")
    return result


def add_pdbfix_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``alchemforge fix-pdb`` command."""
    parser = subparsers.add_parser(
        "fix-pdb",
        help="Remove hydrogens, repair PDBs, and assemble complex.pdb",
        description=(
            "Repair and assemble a protein system with an optional cofactor and "
            "optional membrane/solvent PDB files."
        ),
    )
    parser.add_argument("--protein", required=True, metavar="PDB", help="Protein PDB")
    parser.add_argument(
        "--cofactor",
        metavar="PDB",
        help="Cofactor PDB; may contain one or multiple retained cofactors",
    )
    parser.add_argument(
        "--membrane", metavar="PDB", help="Membrane/lipid PDB (requires --solvent)"
    )
    parser.add_argument(
        "--solvent", metavar="PDB", help="Water/ion PDB (requires --membrane)"
    )
    parser.add_argument(
        "--pbc-source",
        metavar="PDB",
        help="PDB whose CRYST1 record supplies the assembled periodic box",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        required=True,
        metavar="DIR",
        help="Directory for fixed PDBs and complex.pdb",
    )
    parser.add_argument(
        "--pdbfixer-command",
        default="pdbfixer",
        metavar="COMMAND",
        help="PDBFixer executable name or path (default: pdbfixer)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output PDBs"
    )
    parser.set_defaults(func=run_cli)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        return run_manifest(args.manifest)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
