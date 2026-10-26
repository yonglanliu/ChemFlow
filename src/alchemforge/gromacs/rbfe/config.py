"""Configuration model for multi-ligand GROMACS RBFE projects."""

from __future__ import annotations

import math
import re
import shlex
import tomllib
from dataclasses import dataclass
from itertools import pairwise
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


def _nonnegative_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a number") from error
    if result < 0:
        raise ValueError(f"{label} cannot be negative")
    return result


def _positive_int(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be true or false")
    return value


def _options(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError(f"{label} must be a string or an array of strings")


def _choice(value: Any, label: str, choices: set[str]) -> str:
    result = str(value).lower()
    if result not in choices:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return result


def _lambda_vector(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty TOML array")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only numbers") from error
    if any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError(f"{label} values must be between 0 and 1")
    if any(right < left for left, right in pairwise(result)):
        raise ValueError(f"{label} must be monotonically increasing")
    return result


def _steps(duration_ns: float, timestep_ps: float) -> int:
    exact = duration_ns * 1000.0 / timestep_ps
    result = round(exact)
    if not math.isclose(exact, result, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Simulation duration is not an integer number of timesteps")
    return result


@dataclass(frozen=True)
class LigandRBFEConfig:
    source: Path
    work_dir: Path
    job_name: str
    protein: Path | None
    membrane: MembraneSettings
    prepared_complex_gro: Path | None
    prepared_complex_topology: Path | None
    ligand_library: Path
    ligand_charge_manifest: Path | None
    existing_network_dir: Path | None
    gmx: str
    force_field: str
    water_model: str
    box_shape: str
    box_distance: float
    neutralize: bool
    ion_concentration: float
    temperature_k: float
    pressure_bar: float
    network_method: str
    atom_mapper: str
    pmx_command: str
    pmx_mapping_mode: str
    pmx_h2h_polar: bool
    element_change: bool
    max3d_angstrom: float
    require_cycle_covering: bool
    radial_hub: str | None
    redundant_networks: int
    topology_backend: str
    charge_method: str
    charge_change_method: str
    charge_correction_file: Path | None
    start_replicate: int
    replicates: int
    lambda_windows: int
    lambda_schedule_mode: str
    lambda_function: str
    lambda_power: float
    coul_lambdas: tuple[float, ...] | None
    vdw_lambdas: tuple[float, ...] | None
    bonded_lambdas: tuple[float, ...] | None
    restraint_lambdas: tuple[float, ...] | None
    timestep_ps: float
    equilibration_ns: float
    production_ns: float
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
    mdp_integrator: str
    mdp_constraints: str
    mdp_coulomb_cutoff_nm: float
    mdp_vdw_cutoff_nm: float
    mdp_softcore_alpha: float
    mdp_calc_lambda_neighbors: int
    mdp_nstdhdl: int

    @classmethod
    def load(cls, path: str | Path) -> LigandRBFEConfig:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Configuration file does not exist: {source}")
        with source.open("rb") as handle:
            data = tomllib.load(handle)
        project = _table(data, "project")
        system = _table(data, "system")
        ligands = _table(data, "ligands")
        mapping = _table(data, "mapping")
        topology = _table(data, "topology")
        simulation = _table(data, "simulation")
        mdp = _table(data, "mdp")
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
            raise ValueError(
                "[project].name must contain only letters, numbers, '.', '_' or '-'"
            )
        method = _choice(
            mapping.get("network_method", "lomap"),
            "[mapping].network_method",
            {"lomap", "minimal_spanning", "minimal_redundant", "radial"},
        )
        hub = mapping.get("radial_hub")
        if method == "radial" and not hub:
            raise ValueError("[mapping].radial_hub is required for a radial network")
        worker = execution.get("worker_script")
        membrane = MembraneSettings.from_system(system)
        protein_value = system.get("protein")
        complex_gro = system.get("prepared_complex_gro")
        complex_top = system.get("prepared_complex_topology")
        if not protein_value and not (complex_gro and complex_top):
            raise ValueError(
                "[system].protein or both prepared_complex_gro and "
                "prepared_complex_topology are required"
            )
        replicates = _positive_int(
            simulation.get("replicates", 1), "[simulation].replicates"
        )
        windows = _positive_int(
            simulation.get("lambda_windows", 21), "[simulation].lambda_windows", 2
        )
        schedule_mode = _choice(
            simulation.get("lambda_schedule_mode", "generated"),
            "[simulation].lambda_schedule_mode",
            {"generated", "manual"},
        )
        manual_vectors: dict[str, tuple[float, ...] | None] = {
            name: None
            for name in (
                "coul_lambdas",
                "vdw_lambdas",
                "bonded_lambdas",
                "restraint_lambdas",
            )
        }
        if schedule_mode == "manual":
            for name in manual_vectors:
                manual_vectors[name] = _lambda_vector(
                    _required(simulation, "simulation", name),
                    f"[simulation].{name}",
                )
            lengths = {
                len(vector) for vector in manual_vectors.values() if vector is not None
            }
            if lengths != {windows}:
                raise ValueError(
                    "Manual lambda arrays must all contain exactly "
                    f"lambda_windows ({windows}) values"
                )
            for name in ("coul_lambdas", "vdw_lambdas", "bonded_lambdas"):
                vector = manual_vectors[name]
                assert vector is not None
                if vector[0] != 0.0 or vector[-1] != 1.0:
                    raise ValueError(
                        f"[simulation].{name} must start at 0 and end at 1"
                    )
        atom_mapper = _choice(
            mapping.get("atom_mapper", "lomap"),
            "[mapping].atom_mapper",
            {"kartograf", "lomap", "pmx"},
        )
        pmx_command = str(mapping.get("pmx_command", "pmx")).strip()
        if atom_mapper == "pmx" and not pmx_command:
            raise ValueError("[mapping].pmx_command cannot be empty")
        return cls(
            source=source,
            work_dir=resolve(_required(project, "project", "work_dir")),
            job_name=name,
            protein=resolve(protein_value) if protein_value else None,
            membrane=membrane,
            prepared_complex_gro=resolve(complex_gro) if complex_gro else None,
            prepared_complex_topology=resolve(complex_top) if complex_top else None,
            ligand_library=resolve(_required(ligands, "ligands", "library_sdf")),
            ligand_charge_manifest=(
                resolve(ligands["charge_manifest"])
                if ligands.get("charge_manifest")
                else None
            ),
            existing_network_dir=(
                resolve(mapping["existing_network_dir"])
                if mapping.get("existing_network_dir")
                else None
            ),
            gmx=str(system.get("gromacs_command", "gmx")),
            force_field=str(_required(system, "system", "force_field")),
            water_model=str(_required(system, "system", "water_model")),
            box_shape=_choice(
                system.get("box_shape", "dodecahedron"),
                "[system].box_shape",
                {"cubic", "dodecahedron", "octahedron"},
            ),
            box_distance=_positive_float(
                system.get("box_distance", 1.2), "[system].box_distance"
            ),
            neutralize=_boolean(
                system.get("neutralize", True), "[system].neutralize"
            ),
            ion_concentration=_nonnegative_float(
                system.get("ion_concentration", 0.15), "[system].ion_concentration"
            ),
            temperature_k=_positive_float(
                system.get("temperature_k", 300.0), "[system].temperature_k"
            ),
            pressure_bar=_positive_float(
                system.get("pressure_bar", 1.0), "[system].pressure_bar"
            ),
            network_method=method,
            atom_mapper=atom_mapper,
            pmx_command=pmx_command,
            pmx_mapping_mode=_choice(
                mapping.get("pmx_mapping_mode", "stage"),
                "[mapping].pmx_mapping_mode",
                {"run", "stage"},
            ),
            pmx_h2h_polar=_boolean(
                mapping.get("pmx_h2h_polar", True), "[mapping].pmx_h2h_polar"
            ),
            element_change=_boolean(
                mapping.get("element_change", False), "[mapping].element_change"
            ),
            max3d_angstrom=_positive_float(
                mapping.get("max3d_angstrom", 1.0), "[mapping].max3d_angstrom"
            ),
            require_cycle_covering=_boolean(
                mapping.get("require_cycle_covering", True),
                "[mapping].require_cycle_covering",
            ),
            radial_hub=str(hub) if hub else None,
            redundant_networks=_positive_int(
                mapping.get("redundant_networks", 2), "[mapping].redundant_networks"
            ),
            topology_backend=_choice(
                topology.get("backend", "pmx"), "[topology].backend", {"pmx"}
            ),
            charge_method=str(topology.get("charge_method", "bcc")).lower(),
            charge_change_method=_choice(
                topology.get("charge_change_method", "reject"),
                "[topology].charge_change_method",
                {"reject", "coion", "correction"},
            ),
            charge_correction_file=(
                resolve(topology["charge_correction_file"])
                if topology.get("charge_correction_file")
                else None
            ),
            start_replicate=_positive_int(
                simulation.get("start_replicate", 1), "[simulation].start_replicate"
            ),
            replicates=replicates,
            lambda_windows=windows,
            lambda_schedule_mode=schedule_mode,
            lambda_function=_choice(
                simulation.get("lambda_function", "cosine"),
                "[simulation].lambda_function",
                {"cosine", "linear", "power"},
            ),
            lambda_power=_positive_float(
                simulation.get("lambda_power", 2.0), "[simulation].lambda_power"
            ),
            coul_lambdas=manual_vectors["coul_lambdas"],
            vdw_lambdas=manual_vectors["vdw_lambdas"],
            bonded_lambdas=manual_vectors["bonded_lambdas"],
            restraint_lambdas=manual_vectors["restraint_lambdas"],
            timestep_ps=_positive_float(
                simulation.get("timestep_ps", 0.002), "[simulation].timestep_ps"
            ),
            equilibration_ns=_positive_float(
                simulation.get("equilibration_ns", 1.0), "[simulation].equilibration_ns"
            ),
            production_ns=_positive_float(
                simulation.get("production_ns", 5.0), "[simulation].production_ns"
            ),
            max_parallel_jobs=_positive_int(
                execution.get("max_parallel_jobs", 3),
                "[execution].max_parallel_jobs",
            ),
            mdrun_options=_options(
                execution.get("mdrun_options", ["-ntmpi", "1"]),
                "[execution].mdrun_options",
            ),
            worker_script=resolve(worker) if worker else None,
            slurm_partition=str(slurm.get("partition", "gpu")),
            slurm_nodes=_positive_int(slurm.get("nodes", 1), "[slurm].nodes"),
            slurm_tasks=_positive_int(slurm.get("tasks", 1), "[slurm].tasks"),
            slurm_cpus_per_task=_positive_int(
                slurm.get("cpus_per_task", 8), "[slurm].cpus_per_task"
            ),
            slurm_gpus=_positive_int(slurm.get("gpus", 1), "[slurm].gpus", 0),
            slurm_memory=str(slurm.get("memory", "16G")),
            slurm_walltime=str(slurm.get("walltime", "24:00:00")),
            mdp_integrator=_choice(
                mdp.get("integrator", "sd"), "[mdp].integrator", {"sd", "md"}
            ),
            mdp_constraints=_choice(
                mdp.get("constraints", "h-bonds"),
                "[mdp].constraints",
                {"h-bonds", "all-bonds", "none"},
            ),
            mdp_coulomb_cutoff_nm=_positive_float(
                mdp.get("coulomb_cutoff_nm", 1.0), "[mdp].coulomb_cutoff_nm"
            ),
            mdp_vdw_cutoff_nm=_positive_float(
                mdp.get("vdw_cutoff_nm", 1.0), "[mdp].vdw_cutoff_nm"
            ),
            mdp_softcore_alpha=_positive_float(
                mdp.get("softcore_alpha", 0.5), "[mdp].softcore_alpha"
            ),
            mdp_calc_lambda_neighbors=int(mdp.get("calc_lambda_neighbors", -1)),
            mdp_nstdhdl=_positive_int(mdp.get("nstdhdl", 100), "[mdp].nstdhdl"),
        )

    @property
    def project_dir(self) -> Path:
        return self.work_dir / self.job_name

    @property
    def tasks_per_edge(self) -> int:
        return 2 * self.replicates * self.lambda_windows

    @property
    def equilibration_steps(self) -> int:
        return _steps(self.equilibration_ns, self.timestep_ps)

    @property
    def production_steps(self) -> int:
        return _steps(self.production_ns, self.timestep_ps)

    def validate_inputs(self, *, require_runtime: bool = False) -> None:
        for label, path in (
            ("Protein", self.protein),
            ("Prepared complex GRO", self.prepared_complex_gro),
            ("Prepared complex topology", self.prepared_complex_topology),
            ("Ligand SDF", self.ligand_library),
            ("Ligand charge manifest", self.ligand_charge_manifest),
            ("Charge correction table", self.charge_correction_file),
            ("Worker script", self.worker_script),
        ):
            if path is not None and not path.is_file():
                raise ValueError(f"{label} file does not exist: {path}")
        if self.ligand_library.suffix.lower() != ".sdf":
            raise ValueError("[ligands].library_sdf must be a multi-molecule SDF file")
        if (
            self.charge_change_method == "correction"
            and self.charge_correction_file is None
        ):
            raise ValueError(
                "[topology].charge_correction_file is required when "
                "charge_change_method = 'correction'"
            )
        if self.existing_network_dir is not None:
            for filename in (
                "edges.csv",
                "ligand_network.graphml",
                "network_manifest.json",
            ):
                required = self.existing_network_dir / filename
                if not required.is_file():
                    raise ValueError(
                        f"Existing ligand network bundle is missing {filename}: "
                        f"{self.existing_network_dir}"
                    )
        if self.membrane.enabled:
            if (
                self.prepared_complex_gro is None
                or self.prepared_complex_topology is None
            ):
                raise ValueError(
                    "Membrane ligand RBFE requires [system].prepared_complex_gro "
                    "and [system].prepared_complex_topology"
                )
            validate_membrane_gro(self.prepared_complex_gro, self.membrane)
        if require_runtime and self.worker_script is None:
            raise ValueError("Execution requires [execution].worker_script")

    def shell_variables(self) -> dict[str, str | int | float]:
        project_dir = self.project_dir
        input_dir = project_dir / "input"
        values: dict[str, str | int | float] = {
            # ROOT_DIR is the folder selected in the desktop. WORK_DIR is the
            # self-contained, staged job folder used by the worker.
            "ROOT_DIR": str(self.work_dir),
            "WORK_DIR": str(project_dir),
            "PROJECT_DIR": str(project_dir),
            "JOB_NAME": self.job_name,
            "GMX": self.gmx,
            "PROTEIN": str(input_dir / self.protein.name) if self.protein else "",
            "LIGAND_LIBRARY": str(project_dir / "network" / "ligands.sdf"),
            "NETWORK_DIR": str(project_dir / "network"),
            "EDGES_FILE": str(project_dir / "network" / "edges.csv"),
            "FF": self.force_field,
            "WATER": self.water_model,
            "BOX_SHAPE": self.box_shape,
            "BOX_DISTANCE": self.box_distance,
            "NEUTRALIZE": str(self.neutralize).lower(),
            "ION_CONC": self.ion_concentration,
            "TEMPERATURE": self.temperature_k,
            "PRESSURE": self.pressure_bar,
            "NETWORK_METHOD": self.network_method,
            "ATOM_MAPPER": self.atom_mapper,
            "PMX_MAPPING_SCRIPT": str(
                project_dir / "network" / "run_pmx_mappings.sh"
            ),
            "TOPOLOGY_BACKEND": self.topology_backend,
            "CHARGE_METHOD": self.charge_method,
            "CHARGE_CHANGE_METHOD": self.charge_change_method,
            "CHARGE_CHANGES_FILE": str(project_dir / "network" / "charge_changes.csv"),
            "CHARGE_PLAN": str(project_dir / ".alchemforge" / "charge_strategy.json"),
            "COION_TOPOLOGY_DIR": str(project_dir / "charge_correction"),
            "START_REP": self.start_replicate,
            "NREP": self.replicates,
            "NLAMBDA": self.lambda_windows,
            "LAMBDA_SCHEDULE_MODE": self.lambda_schedule_mode,
            "LAMBDA_FUNCTION": self.lambda_function,
            "LAMBDA_POWER": self.lambda_power,
            "DT_PS": self.timestep_ps,
            "EQUIL_NS": self.equilibration_ns,
            "PROD_NS": self.production_ns,
            "EQUIL_STEPS": self.equilibration_steps,
            "PROD_STEPS": self.production_steps,
            "MAX_FEP_JOBS": self.max_parallel_jobs,
            "TASKS_PER_EDGE": self.tasks_per_edge,
            "MDRUN_OPTIONS": shlex.join(self.mdrun_options),
            "FEP_PARTITION": self.slurm_partition,
            "FEP_NODES": self.slurm_nodes,
            "FEP_NTASKS": self.slurm_tasks,
            "FEP_CPUS_PER_TASK": self.slurm_cpus_per_task,
            "FEP_GPUS": self.slurm_gpus,
            "FEP_MEMORY": self.slurm_memory,
            "FEP_WALLTIME": self.slurm_walltime,
            "RBFE_MDP_DIR": str(project_dir / "mdp"),
            "RBFE_MDP_TEMPLATE": str(project_dir / "mdp" / "prod.mdp"),
            "RBFE_COMPLEX_MDP_DIR": str(project_dir / "mdp" / "complex"),
            "RBFE_SOLVENT_MDP_DIR": str(project_dir / "mdp" / "solvent"),
            "RBFE_MINI_MDP_TEMPLATE": str(project_dir / "mdp" / "mini.mdp"),
            "RBFE_NVT_MDP_TEMPLATE": str(project_dir / "mdp" / "nvt.mdp"),
            "RBFE_NPT_MDP_TEMPLATE": str(project_dir / "mdp" / "npt.mdp"),
            "RBFE_PROD_MDP_TEMPLATE": str(project_dir / "mdp" / "prod.mdp"),
        }
        values.update(self.membrane.shell_variables())
        if self.membrane.enabled:
            values["MEMBRANE_MDP"] = str(
                project_dir / ".alchemforge" / "membrane_pressure.mdp"
            )
        if self.prepared_complex_gro is not None:
            values["PREPARED_COMPLEX_GRO"] = str(
                input_dir / self.prepared_complex_gro.name
            )
        if self.prepared_complex_topology is not None:
            values["PREPARED_COMPLEX_TOPOLOGY"] = str(
                input_dir / self.prepared_complex_topology.name
            )
        if self.ligand_charge_manifest is not None:
            values["LIGAND_CHARGE_MANIFEST"] = str(
                input_dir / self.ligand_charge_manifest.name
            )
        if self.charge_correction_file is not None:
            values["CHARGE_CORRECTION_FILE"] = str(
                input_dir / self.charge_correction_file.name
            )
        return values

    def render_shell_config(self) -> str:
        lines = [
            "# Generated by AlchemForge from ligand-RBFE TOML.",
            f"# Source: {self.source}",
        ]
        lines.extend(
            f"{key}={shlex.quote(str(value))}"
            for key, value in self.shell_variables().items()
        )
        return "\n".join(lines) + "\n"
