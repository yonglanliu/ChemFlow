"""Typed configuration for absolute binding free energy with GROMACS."""

from __future__ import annotations

import re
import shlex
import shutil
import tomllib
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from alchemforge.gromacs.membrane import MembraneSettings, validate_membrane_gro


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"[{name}] must be a TOML table")
    return value


def _required(table: dict[str, Any], section: str, key: str) -> Any:
    value = table.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required configuration: [{section}].{key}")
    return value


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a number") from error
    if result <= 0:
        raise ValueError(f"{label} must be greater than zero")
    return result


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _choice(value: Any, label: str, choices: set[str]) -> str:
    result = str(value).lower()
    if result not in choices:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return result


def _options(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError(f"{label} must be a string or an array of strings")


@dataclass(frozen=True)
class ABFEConfig:
    source: Path
    work_dir: Path
    job_name: str
    protein: Path | None
    ligand: Path
    ligand_resname: str
    ligand_charge: int
    gmx: str
    force_field: str
    water_model: str
    box_distance: float
    ion_concentration: float
    temperature_k: float
    pressure_bar: float
    membrane: MembraneSettings
    prepared_setup_dir: Path | None
    topology_backend: str
    charge_method: str
    restraint_method: str
    restraint_seed: int
    distance_force_constant: float
    angle_force_constant: float
    dihedral_force_constant: float
    replicas: int
    lambda_windows: int
    lambda_function: str
    lambda_power: float
    timestep_ps: float
    equilibration_ns: float
    production_ns: float
    decouple_coulomb_first: bool
    softcore_alpha: float
    execution_mode: str
    max_parallel_jobs: int
    mdrun_options: tuple[str, ...]
    worker_script: Path | None
    slurm_partition: str
    slurm_nodes: int
    slurm_tasks: int
    slurm_cpus_per_task: int
    slurm_gpus: int
    slurm_memory: str
    slurm_walltime: str

    @classmethod
    def load(cls, path: str | Path) -> ABFEConfig:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Configuration file does not exist: {source}")
        with source.open("rb") as handle:
            data = tomllib.load(handle)
        project = _table(data, "project")
        system = _table(data, "system")
        ligand = _table(data, "ligand")
        topology = _table(data, "topology")
        restraints = _table(data, "restraints")
        simulation = _table(data, "simulation")
        execution = _table(data, "execution")
        slurm = _table(data, "slurm")
        base = source.parent

        def resolve(value: Any) -> Path:
            candidate = Path(str(value)).expanduser()
            return (
                (base / candidate).resolve()
                if not candidate.is_absolute()
                else candidate.resolve()
            )

        name = str(_required(project, "project", "name"))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("[project].name contains unsupported characters")
        worker = execution.get("worker_script")
        prepared_setup = system.get("prepared_setup_dir")
        membrane = MembraneSettings.from_system(system)
        protein_value = system.get("protein")
        if not protein_value and not prepared_setup:
            raise ValueError(
                "[system].protein is required unless prepared_setup_dir is supplied"
            )
        decouple_first = simulation.get("decouple_coulomb_first", True)
        if not isinstance(decouple_first, bool):
            raise TypeError("[simulation].decouple_coulomb_first must be true or false")
        return cls(
            source=source,
            work_dir=resolve(_required(project, "project", "work_dir")),
            job_name=name,
            protein=resolve(protein_value) if protein_value else None,
            ligand=resolve(_required(ligand, "ligand", "sdf")),
            ligand_resname=str(ligand.get("resname", "LIG")),
            ligand_charge=_integer(ligand.get("charge", 0), "[ligand].charge", -100),
            gmx=str(system.get("gromacs_command", "gmx")),
            force_field=str(_required(system, "system", "force_field")),
            water_model=str(_required(system, "system", "water_model")),
            box_distance=_positive_float(
                system.get("box_distance", 1.2), "[system].box_distance"
            ),
            ion_concentration=_positive_float(
                system.get("ion_concentration", 0.15), "[system].ion_concentration"
            ),
            temperature_k=_positive_float(
                system.get("temperature_k", 300.0), "[system].temperature_k"
            ),
            pressure_bar=_positive_float(
                system.get("pressure_bar", 1.0), "[system].pressure_bar"
            ),
            membrane=membrane,
            prepared_setup_dir=resolve(prepared_setup) if prepared_setup else None,
            topology_backend=_choice(
                topology.get("backend", "pmx"), "[topology].backend", {"pmx"}
            ),
            charge_method=str(topology.get("charge_method", "bcc")).lower(),
            restraint_method=_choice(
                restraints.get("method", "boresch"), "[restraints].method", {"boresch"}
            ),
            restraint_seed=_integer(restraints.get("seed", 2026), "[restraints].seed"),
            distance_force_constant=_positive_float(
                restraints.get("distance_force_constant", 4184.0),
                "[restraints].distance_force_constant",
            ),
            angle_force_constant=_positive_float(
                restraints.get("angle_force_constant", 41.84),
                "[restraints].angle_force_constant",
            ),
            dihedral_force_constant=_positive_float(
                restraints.get("dihedral_force_constant", 41.84),
                "[restraints].dihedral_force_constant",
            ),
            replicas=_integer(
                simulation.get("replicas", 1), "[simulation].replicas", 1
            ),
            lambda_windows=_integer(
                simulation.get("lambda_windows", 21), "[simulation].lambda_windows", 2
            ),
            lambda_function=_choice(
                simulation.get("lambda_function", "cosine"),
                "[simulation].lambda_function",
                {"cosine", "linear", "power"},
            ),
            lambda_power=_positive_float(
                simulation.get("lambda_power", 2.0), "[simulation].lambda_power"
            ),
            timestep_ps=_positive_float(
                simulation.get("timestep_ps", 0.002), "[simulation].timestep_ps"
            ),
            equilibration_ns=_positive_float(
                simulation.get("equilibration_ns", 1.0), "[simulation].equilibration_ns"
            ),
            production_ns=_positive_float(
                simulation.get("production_ns", 5.0), "[simulation].production_ns"
            ),
            decouple_coulomb_first=decouple_first,
            softcore_alpha=_positive_float(
                simulation.get("softcore_alpha", 0.5), "[simulation].softcore_alpha"
            ),
            execution_mode=_choice(
                execution.get("mode", "local"), "[execution].mode", {"local", "slurm"}
            ),
            max_parallel_jobs=_integer(
                execution.get("max_parallel_jobs", 3),
                "[execution].max_parallel_jobs",
                1,
            ),
            mdrun_options=_options(
                execution.get("mdrun_options", ["-ntmpi", "1"]),
                "[execution].mdrun_options",
            ),
            worker_script=resolve(worker) if worker else None,
            slurm_partition=str(slurm.get("partition", "gpu")),
            slurm_nodes=_integer(slurm.get("nodes", 1), "[slurm].nodes", 1),
            slurm_tasks=_integer(slurm.get("tasks", 1), "[slurm].tasks", 1),
            slurm_cpus_per_task=_integer(
                slurm.get("cpus_per_task", 8), "[slurm].cpus_per_task", 1
            ),
            slurm_gpus=_integer(slurm.get("gpus", 1), "[slurm].gpus"),
            slurm_memory=str(slurm.get("memory", "16G")),
            slurm_walltime=str(slurm.get("walltime", "24:00:00")),
        )

    @property
    def project_dir(self) -> Path:
        return self.work_dir / self.job_name

    @property
    def total_tasks(self) -> int:
        return 2 * self.replicas * self.lambda_windows

    def validate_inputs(self, *, require_worker: bool = False) -> None:
        for label, path in (("Protein", self.protein), ("Ligand", self.ligand)):
            if path is not None and not path.is_file():
                raise ValueError(f"{label} file does not exist: {path}")
        if self.protein is not None and self.protein.suffix.lower() != ".pdb":
            raise ValueError("[system].protein must be a PDB file")
        if self.ligand.suffix.lower() != ".sdf":
            raise ValueError("[ligand].sdf must be an SDF file")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,4}", self.ligand_resname):
            raise ValueError(
                "[ligand].resname must contain 1-5 letters/numbers and start with a letter"
            )
        if self.charge_method not in {"bcc", "user"}:
            raise ValueError("[topology].charge_method must be bcc or user")
        if self.worker_script is None and self.prepared_setup_dir is None:
            if self.membrane.enabled and self.prepared_setup_dir is None:
                raise ValueError(
                    "Built-in membrane ABFE requires [system].prepared_setup_dir. "
                    "It must contain already restrained complex and solvent legs."
                )
            if self.charge_method != "bcc":
                raise ValueError(
                    "The built-in SDF/ACPYPE worker requires charge_method = 'bcc'"
                )
            if self.water_model.lower() != "tip3p":
                raise ValueError(
                    "The built-in PMX ABFE setup currently requires water_model = 'tip3p'"
                )
            if self.box_distance != 1.2 or self.ion_concentration != 0.15:
                raise ValueError(
                    "The built-in PMX ABFE setup currently requires box_distance = 1.2 "
                    "and ion_concentration = 0.15"
                )
        if self.prepared_setup_dir is not None:
            required = (
                "complex/genion.gro",
                "complex/complex.top",
                "ligand/genion.gro",
                "ligand/ligand.top",
                "restraints.info",
            )
            missing = [
                item
                for item in required
                if not (self.prepared_setup_dir / item).is_file()
            ]
            if missing:
                raise ValueError(
                    "Prepared ABFE setup is incomplete: " + ", ".join(missing)
                )
            if self.membrane.enabled:
                validate_membrane_gro(
                    self.prepared_setup_dir / "complex" / "genion.gro",
                    self.membrane,
                )
        try:
            from rdkit import Chem
        except ImportError as error:
            raise ValueError("RDKit is required to validate the ligand SDF") from error
        supplier = Chem.SDMolSupplier(str(self.ligand), removeHs=False)
        molecules = list(supplier)
        if len(molecules) != 1 or molecules[0] is None:
            raise ValueError("[ligand].sdf must contain exactly one valid molecule")
        if self.worker_script is not None and not self.worker_script.is_file():
            raise ValueError(f"ABFE worker script does not exist: {self.worker_script}")
        if require_worker and self.worker_script is None:
            commands = (
                (self.gmx,)
                if self.prepared_setup_dir is not None
                else (self.gmx, "obabel", "acpype")
            )
            missing = [command for command in commands if shutil.which(command) is None]
            if self.prepared_setup_dir is None and find_spec("pmx") is None:
                missing.append("PMX Python package")
            if missing:
                raise ValueError(
                    "Built-in ABFE execution dependencies are unavailable: "
                    + ", ".join(missing)
                )

    def shell_variables(self) -> dict[str, str | int | float]:
        values: dict[str, str | int | float] = {
            "WORK_DIR": str(self.work_dir),
            "JOB_NAME": self.job_name,
            "GMX": self.gmx,
            "PROTEIN": str(self.protein or ""),
            "LIGAND_SDF": str(self.ligand),
            "LIGAND_RESNAME": self.ligand_resname,
            "LIGAND_CHARGE": self.ligand_charge,
            "FF": self.force_field,
            "WATER": self.water_model,
            "BOX_DISTANCE": self.box_distance,
            "ION_CONC": self.ion_concentration,
            "TEMPERATURE": self.temperature_k,
            "PRESSURE": self.pressure_bar,
            "TOPOLOGY_BACKEND": self.topology_backend,
            "CHARGE_METHOD": self.charge_method,
            "RESTRAINT_METHOD": self.restraint_method,
            "RESTRAINT_SEED": self.restraint_seed,
            "RESTRAINT_K_DISTANCE": self.distance_force_constant,
            "RESTRAINT_K_ANGLE": self.angle_force_constant,
            "RESTRAINT_K_DIHEDRAL": self.dihedral_force_constant,
            "NREP": self.replicas,
            "NLAMBDA": self.lambda_windows,
            "N_ABFE_TASKS": self.total_tasks,
            "FEP_LAMBDA_FUNCTION": self.lambda_function,
            "FEP_LAMBDA_POWER": self.lambda_power,
            "DT_PS": self.timestep_ps,
            "EQUIL_NS": self.equilibration_ns,
            "PROD_NS": self.production_ns,
            "DECOUPLE_COULOMB_FIRST": str(self.decouple_coulomb_first).lower(),
            "SOFTCORE_ALPHA": self.softcore_alpha,
            "EXECUTION_MODE": self.execution_mode,
            "MAX_ABFE_JOBS": self.max_parallel_jobs,
            "MDRUN_OPTIONS": shlex.join(self.mdrun_options),
            "ABFE_PARTITION": self.slurm_partition,
            "ABFE_NODES": self.slurm_nodes,
            "ABFE_NTASKS": self.slurm_tasks,
            "ABFE_CPUS_PER_TASK": self.slurm_cpus_per_task,
            "ABFE_GPUS": self.slurm_gpus,
            "ABFE_MEMORY": self.slurm_memory,
            "ABFE_WALLTIME": self.slurm_walltime,
        }
        values.update(self.membrane.shell_variables())
        if self.prepared_setup_dir is not None:
            values["PREPARED_ABFE_SETUP"] = str(self.prepared_setup_dir)
        return values

    def render_shell_config(self) -> str:
        lines = [
            "# Generated by AlchemForge for GROMACS ABFE.",
            f"# Source: {self.source}",
        ]
        lines.extend(
            f"{key}={shlex.quote(str(value))}"
            for key, value in self.shell_variables().items()
        )
        return "\n".join(lines) + "\n"
