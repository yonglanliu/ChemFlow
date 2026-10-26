"""Stage and run standalone GROMACS or OpenMM energy minimization."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_OPENMM_FORCEFIELDS = (
    "amber/ff14SB.xml",
    "amber/tip3p_standard.xml",
    "amber/tip3p_HFE_multivalent.xml",
    "amber/phosaa10.xml",
    "amber/lipid17_merged.xml",
)

DESMOND_POPC_TO_LIPID17 = {
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
    "P1": "P",
    "O1": "O12",
    "O2": "O11",
    "O3": "O13",
    "O4": "O14",
}


@dataclass(frozen=True)
class GromacsMinimizationConfig:
    structure: Path
    topology: Path
    output_directory: Path
    gmx_command: str = "gmx"
    steps: int = 50000
    tolerance: float = 1000.0
    step_size: float = 0.01
    mdrun_options: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.structure.is_file() or self.structure.suffix.lower() not in {
            ".gro",
            ".pdb",
        }:
            raise ValueError(
                f"Structure must be an existing GRO or PDB: {self.structure}"
            )
        if not self.topology.is_file() or self.topology.suffix.lower() != ".top":
            raise ValueError(f"Topology must be an existing TOP file: {self.topology}")
        if not self.gmx_command.strip():
            raise ValueError("GROMACS command cannot be empty")
        if self.steps < 1 or self.tolerance <= 0 or self.step_size <= 0:
            raise ValueError("Steps, tolerance, and step size must be positive")


@dataclass(frozen=True)
class OpenMMMinimizationConfig:
    structure: Path
    output_directory: Path
    forcefields: tuple[str, ...] = DEFAULT_OPENMM_FORCEFIELDS
    platform: str = "CPU"
    steps: int = 50000
    tolerance: float = 10.0
    periodic: bool = False
    box_vectors_nm: tuple[tuple[float, float, float], ...] | None = None
    small_molecule_files: tuple[Path, ...] = ()
    template_generator: str = "smirnoff"
    small_molecule_forcefield: str = "openff-2.2.1"
    normalize_desmond_lipids: bool = True
    export_gromacs: bool = False

    def validate(self) -> None:
        if not self.structure.is_file() or self.structure.suffix.lower() not in {
            ".pdb",
            ".cif",
            ".mmcif",
        }:
            raise ValueError(
                f"Structure must be an existing PDB or mmCIF: {self.structure}"
            )
        if not self.forcefields:
            raise ValueError("At least one OpenMM force-field XML file is required")
        if self.platform not in {"CPU", "CUDA", "OpenCL", "Reference"}:
            raise ValueError(f"Unsupported OpenMM platform: {self.platform}")
        if self.steps < 1 or self.tolerance <= 0:
            raise ValueError("Steps and tolerance must be positive")
        if self.template_generator not in {"smirnoff", "gaff"}:
            raise ValueError(
                "Small-molecule template generator must be smirnoff or gaff"
            )
        if self.small_molecule_files and not self.small_molecule_forcefield.strip():
            raise ValueError("Select a small-molecule force field")
        for path in self.small_molecule_files:
            if not path.is_file() or path.suffix.lower() not in {".sdf", ".mol2"}:
                raise ValueError(
                    "Ligand/cofactor definitions must be existing SDF or MOL2 "
                    f"files: {path}"
                )
        if (
            self.periodic
            and self.box_vectors_nm is not None
            and (
                len(self.box_vectors_nm) != 3
                or any(len(vector) != 3 for vector in self.box_vectors_nm)
            )
        ):
            raise ValueError("Box vectors must be a 3 by 3 array")


def _serialized_config(config: Any) -> dict[str, Any]:
    def serialize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value.expanduser().resolve())
        if isinstance(value, (list, tuple)):
            return [serialize(item) for item in value]
        if isinstance(value, dict):
            return {key: serialize(item) for key, item in value.items()}
        return value

    return serialize(asdict(config))


def stage_gromacs(
    config: GromacsMinimizationConfig, task_directory: Path
) -> tuple[Path, Path]:
    config.validate()
    return _stage("gromacs", _serialized_config(config), task_directory)


def stage_openmm(
    config: OpenMMMinimizationConfig, task_directory: Path
) -> tuple[Path, Path]:
    config.validate()
    return _stage("openmm", _serialized_config(config), task_directory)


def _stage(
    backend: str, payload: dict[str, Any], task_directory: Path
) -> tuple[Path, Path]:
    task_directory.mkdir(parents=True, exist_ok=True)
    payload["backend"] = backend
    manifest = task_directory / f"minimize_{backend}.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    runner = task_directory / "run_minimization.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'MANIFEST="${1:?Usage: $0 /path/to/minimize.json}"\n'
        'python -m alchemforge.structure.minimize "${MANIFEST}"\n',
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)
    return runner, manifest


def _require_empty_outputs(output: Path, filenames: tuple[str, ...]) -> None:
    existing = [
        str(output / filename) for filename in filenames if (output / filename).exists()
    ]
    if existing:
        raise ValueError(
            "Refusing to overwrite minimization output(s): " + ", ".join(existing)
        )


def _run(command: list[str], *, cwd: Path) -> None:
    print("Running:", shlex.join(command))
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise ValueError(
            f"Command failed with exit code {result.returncode}: {shlex.join(command)}"
        )


def normalize_desmond_lipids(topology: Any) -> dict[str, int]:
    """Rename recognized Desmond lipid atoms to OpenMM force-field conventions."""
    normalized: dict[str, int] = {}
    desmond_signature = {"P1", "O1", "O2", "O3", "O4", "H21", "H51"}
    for residue in topology.residues():
        atoms = list(residue.atoms())
        atom_names = {atom.name for atom in atoms}
        if residue.name == "POPC" and desmond_signature.issubset(atom_names):
            for atom in atoms:
                atom.name = DESMOND_POPC_TO_LIPID17.get(atom.name, atom.name)
            normalized["POPC"] = normalized.get("POPC", 0) + 1
    return normalized


def run_gromacs(config: GromacsMinimizationConfig) -> None:
    config.validate()
    if shutil.which(config.gmx_command) is None:
        raise ValueError(f"GROMACS command is unavailable: {config.gmx_command}")
    output = config.output_directory.expanduser().resolve()
    _require_empty_outputs(
        output,
        ("minimization.gro", "minimized.pdb", "minimization.tpr", "minimization.edr"),
    )
    output.mkdir(parents=True, exist_ok=True)
    mdp = output / "minimization.mdp"
    mdp.write_text(
        "integrator               = steep\n"
        f"nsteps                   = {config.steps}\n"
        f"emtol                    = {config.tolerance}\n"
        f"emstep                   = {config.step_size}\n"
        "cutoff-scheme            = Verlet\n"
        "nstlist                  = 10\n"
        "coulombtype              = PME\n"
        "rcoulomb                 = 1.0\n"
        "vdwtype                  = Cut-off\n"
        "rvdw                     = 1.0\n"
        "pbc                      = xyz\n"
        "constraints              = none\n",
        encoding="utf-8",
    )
    tpr = output / "minimization.tpr"
    _run(
        [
            config.gmx_command,
            "grompp",
            "-f",
            str(mdp),
            "-c",
            str(config.structure.resolve()),
            "-p",
            str(config.topology.resolve()),
            "-o",
            str(tpr),
            "-po",
            str(output / "minimization_processed.mdp"),
        ],
        cwd=config.topology.resolve().parent,
    )
    deffnm = output / "minimization"
    _run(
        [
            config.gmx_command,
            "mdrun",
            "-deffnm",
            str(deffnm),
            *config.mdrun_options,
        ],
        cwd=output,
    )
    _run(
        [
            config.gmx_command,
            "editconf",
            "-f",
            str(output / "minimization.gro"),
            "-o",
            str(output / "minimized.pdb"),
        ],
        cwd=output,
    )


def run_openmm(config: OpenMMMinimizationConfig) -> None:
    config.validate()
    from openmm import LangevinMiddleIntegrator, Platform, Vec3, unit
    from openmm.app import (
        PME,
        ForceField,
        HBonds,
        Modeller,
        NoCutoff,
        PDBFile,
        PDBxFile,
        Simulation,
    )

    output = config.output_directory.expanduser().resolve()
    output_names = ["minimized.pdb", "minimized.cif", "minimization_report.json"]
    if config.export_gromacs:
        output_names.extend(("system.gro", "topol.top"))
    _require_empty_outputs(output, tuple(output_names))
    output.mkdir(parents=True, exist_ok=True)
    structure_path = config.structure.resolve()
    if structure_path.suffix.lower() in {".cif", ".mmcif"}:
        structure = PDBxFile(str(structure_path))
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure = PDBFile(str(structure_path))
        except (IndexError, TypeError, ValueError):
            try:
                import mdtraj
            except ImportError as error:
                raise ValueError(
                    "OpenMM cannot read this PDB's extended identifiers; select "
                    "complex.cif or install MDTraj"
                ) from error
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trajectory = mdtraj.load_pdb(str(structure_path))
            modeller = Modeller(
                trajectory.topology.to_openmm(), trajectory.openmm_positions(0)
            )
            structure = None
    if structure is not None:
        modeller = Modeller(structure.topology, structure.positions)
    normalized_lipids = (
        normalize_desmond_lipids(modeller.topology)
        if config.normalize_desmond_lipids
        else {}
    )
    if config.box_vectors_nm is not None:
        vectors = tuple(
            Vec3(*vector) * unit.nanometer for vector in config.box_vectors_nm
        )
        modeller.topology.setPeriodicBoxVectors(vectors)
    if config.periodic and modeller.topology.getPeriodicBoxVectors() is None:
        raise ValueError(
            "Periodic minimization requires CRYST1 box vectors in the PDB or "
            "explicit box vectors"
        )
    forcefield = ForceField(*config.forcefields)
    if config.small_molecule_files:
        from openff.toolkit import Molecule
        from openmmforcefields.generators import (
            GAFFTemplateGenerator,
            SMIRNOFFTemplateGenerator,
        )

        molecules = []
        for path in config.small_molecule_files:
            loaded = Molecule.from_file(
                str(path.resolve()), allow_undefined_stereo=True
            )
            molecules.extend(loaded if isinstance(loaded, list) else [loaded])
        generator_class = (
            SMIRNOFFTemplateGenerator
            if config.template_generator == "smirnoff"
            else GAFFTemplateGenerator
        )
        generator = generator_class(
            molecules=molecules,
            forcefield=config.small_molecule_forcefield,
        )
        forcefield.registerTemplateGenerator(generator.generator)
    try:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME if config.periodic else NoCutoff,
            constraints=None if config.export_gromacs else HBonds,
            rigidWater=not config.export_gromacs,
        )
    except ValueError as error:
        if "No template found for residue" in str(error):
            raise ValueError(
                f"{error} Add a matching charged SDF or MOL2 in the desktop "
                "Ligand/cofactor definitions field, or add the residue's custom "
                "OpenMM force-field XML."
            ) from error
        raise
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    platform = Platform.getPlatformByName(config.platform)
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    simulation.minimizeEnergy(
        tolerance=config.tolerance * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=config.steps,
    )
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    after = state.getPotentialEnergy()
    positions = state.getPositions()
    with (output / "minimized.pdb").open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(modeller.topology, positions, handle, keepIds=True)
    with (output / "minimized.cif").open("w", encoding="utf-8") as handle:
        PDBxFile.writeFile(modeller.topology, positions, handle, keepIds=False)
    if config.export_gromacs:
        try:
            import parmed as pmd
        except ImportError as error:
            raise ValueError(
                "ParmEd is required to export the OpenMM system to GROMACS"
            ) from error
        parameterized = pmd.openmm.load_topology(
            modeller.topology,
            system,
            xyz=positions,
        )
        box_vectors = modeller.topology.getPeriodicBoxVectors()
        if box_vectors is not None:
            parameterized.box_vectors = box_vectors
        parameterized.save(str(output / "system.gro"), overwrite=False)
        parameterized.save(str(output / "topol.top"), overwrite=False)
    report = {
        "backend": "OpenMM (OpenFE environment)",
        "platform": config.platform,
        "forcefields": list(config.forcefields),
        "periodic": config.periodic,
        "small_molecule_files": [
            str(path.resolve()) for path in config.small_molecule_files
        ],
        "template_generator": config.template_generator,
        "small_molecule_forcefield": config.small_molecule_forcefield,
        "normalized_desmond_lipids": normalized_lipids,
        "gromacs_exported": config.export_gromacs,
        "gromacs_structure": (
            str(output / "system.gro") if config.export_gromacs else None
        ),
        "gromacs_topology": (
            str(output / "topol.top") if config.export_gromacs else None
        ),
        "potential_energy_before_kj_mol": before.value_in_unit(unit.kilojoule_per_mole),
        "potential_energy_after_kj_mol": after.value_in_unit(unit.kilojoule_per_mole),
        "maximum_iterations": config.steps,
        "tolerance_kj_mol_nm": config.tolerance,
    }
    (output / "minimization_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def run_manifest(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    backend = payload.pop("backend")
    if backend == "gromacs":
        payload["structure"] = Path(payload["structure"])
        payload["topology"] = Path(payload["topology"])
        payload["output_directory"] = Path(payload["output_directory"])
        payload["mdrun_options"] = tuple(payload.get("mdrun_options", ()))
        run_gromacs(GromacsMinimizationConfig(**payload))
    elif backend == "openmm":
        payload["structure"] = Path(payload["structure"])
        payload["output_directory"] = Path(payload["output_directory"])
        payload["forcefields"] = tuple(payload["forcefields"])
        payload["small_molecule_files"] = tuple(
            Path(item) for item in payload.get("small_molecule_files", ())
        )
        if payload.get("box_vectors_nm") is not None:
            payload["box_vectors_nm"] = tuple(
                tuple(vector) for vector in payload["box_vectors_nm"]
            )
        run_openmm(OpenMMMinimizationConfig(**payload))
    else:
        raise ValueError(f"Unknown minimization backend: {backend}")
    return 0


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
