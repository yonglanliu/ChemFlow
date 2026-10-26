"""Typed configuration for the legacy residue-FEP shell workflow."""

from __future__ import annotations

import math
import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alchemforge.gromacs.membrane import MembraneSettings, validate_membrane_gro


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"[{name}] must be a TOML table")
    return value


def _required(section: dict[str, Any], section_name: str, key: str) -> Any:
    value = section.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required configuration: [{section_name}].{key}")
    return value


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


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a number") from error
    if result <= 0:
        raise ValueError(f"{label} must be greater than zero")
    return result


def _choice(value: Any, label: str, choices: set[str]) -> str:
    result = str(value).upper()
    if result not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{label} must be one of: {allowed}")
    return result


def _duration_ps(section: dict[str, Any], name: str, default_ps: float) -> float:
    """Read a duration expressed in either picoseconds or nanoseconds."""
    ps_key = f"{name}_ps"
    ns_key = f"{name}_ns"
    if ps_key in section and ns_key in section:
        raise ValueError(f"Specify only one of [simulation].{ps_key} and .{ns_key}")
    if ns_key in section:
        return 1000.0 * _positive_float(section[ns_key], f"[simulation].{ns_key}")
    return _positive_float(section.get(ps_key, default_ps), f"[simulation].{ps_key}")


def _step_count(
    section: dict[str, Any], key: str, duration_ps: float, timestep_ps: float
) -> int:
    exact = duration_ps / timestep_ps
    derived = round(exact)
    if not math.isclose(exact, derived, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"{key.removesuffix('_steps')} duration is not an integer number of timesteps"
        )
    configured = section.get(key)
    if configured is None:
        return derived
    result = _positive_int(configured, f"[simulation].{key}")
    if result != derived:
        raise ValueError(
            f"[simulation].{key}={result} conflicts with the duration and timestep "
            f"(expected {derived})"
        )
    return result


def _command_options(value: Any, label: str) -> tuple[str, ...]:
    """Normalize either a TOML string or string array into command arguments."""
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            return tuple(shlex.split(value))
        except ValueError as error:
            raise ValueError(f"{label} contains invalid shell quoting") from error
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError(f"{label} must be a string or an array of strings")


@dataclass(frozen=True)
class ResFEPConfig:
    source: Path
    work_dir: Path
    job_name: str
    gmx: str
    force_field: str
    water_model: str
    complex_path: Path
    temperature_k: float
    pressure_bar: float
    membrane: MembraneSettings
    prepared_complex_gro: Path | None
    prepared_complex_topology: Path | None
    chain: str
    residue: int
    mutation: str
    box_distance: float
    ion_concentration: float
    ligand_mol2: Path
    ligand_itp: Path | None
    ligand_gro: Path | None
    ligand_top: Path | None
    ligand_resname: str
    ligand_charge: int
    ligand_charge_method: str
    ligand_prep_mode: str
    start_replicate: int
    replicates: int
    system_setup_mode: str
    lambda_windows: int
    max_parallel_jobs: int
    md_timestep_ps: float
    timestep_ps: float
    nvt_ps: float
    npt_ps: float
    production_ns: float
    nvt_steps: int
    npt_steps: int
    production_steps: int
    lambda_function: str
    lambda_mode: str
    lambda_power: float | None
    simulation_mode: str
    worker_script: Path | None
    environment_setup: Path | None
    mdrun_em_options: tuple[str, ...]
    mdrun_md_options: tuple[str, ...]
    mdrun_fep_options: tuple[str, ...]
    mdrun_switch_options: tuple[str, ...]
    slurm_partition: str
    slurm_nodes: int
    slurm_tasks: int
    slurm_cpus_per_task: int
    slurm_gpus: int
    slurm_memory: str
    slurm_walltime: str

    @classmethod
    def load(cls, path: str | Path) -> ResFEPConfig:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Configuration file does not exist: {source}")
        with source.open("rb") as handle:
            data = tomllib.load(handle)

        project = _section(data, "project")
        system = _section(data, "system")
        mutation = _section(data, "mutation")
        ligand = _section(data, "ligand")
        simulation = _section(data, "simulation")
        execution = _section(data, "execution")
        slurm = _section(data, "slurm")

        base = source.parent

        def path_value(value: Any) -> Path:
            candidate = Path(str(value)).expanduser()
            return (
                (base / candidate).resolve()
                if not candidate.is_absolute()
                else candidate.resolve()
            )

        lambda_power = simulation.get("lambda_power")
        worker = execution.get("worker_script")
        environment = execution.get("environment_setup")
        ligand_itp = ligand.get("itp")
        ligand_gro = ligand.get("gro")
        ligand_top = ligand.get("top")
        membrane = MembraneSettings.from_system(system)
        membrane_gro = system.get("prepared_complex_gro")
        membrane_topology = system.get("prepared_complex_topology")
        charge_method = str(ligand.get("charge_method", "bcc")).lower()
        if charge_method not in {"bcc", "user"}:
            raise ValueError("[ligand].charge_method must be 'bcc' or 'user'")

        job_name = str(_required(project, "project", "name"))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job_name):
            raise ValueError(
                "[project].name must contain only letters, numbers, '.', '_' or '-'"
            )

        replicates = _positive_int(
            simulation.get("replicates", 1), "[simulation].replicates"
        )
        lambda_windows = _positive_int(
            simulation.get("lambda_windows", 17), "[simulation].lambda_windows", 2
        )
        timestep_ps = _positive_float(
            simulation.get("timestep_ps", 0.002), "[simulation].timestep_ps"
        )
        nvt_ps = _duration_ps(simulation, "nvt", 100.0)
        npt_ps = _duration_ps(simulation, "npt", 100.0)
        production_ns = _positive_float(
            simulation.get("production_ns", 5.0), "[simulation].production_ns"
        )
        lambda_function = str(simulation.get("lambda_function", "linear")).lower()
        if lambda_function not in {"cosine", "linear", "power"}:
            raise ValueError(
                "[simulation].lambda_function must be one of: cosine, linear, power"
            )
        parsed_lambda_power = (
            _positive_float(lambda_power, "[simulation].lambda_power")
            if lambda_power is not None
            else None
        )
        if lambda_function == "power" and parsed_lambda_power is None:
            raise ValueError(
                "[simulation].lambda_power is required when lambda_function is 'power'"
            )

        return cls(
            source=source,
            work_dir=path_value(_required(project, "project", "work_dir")),
            job_name=job_name,
            gmx=str(system.get("gromacs_command", "gmx")),
            force_field=str(_required(system, "system", "force_field")),
            water_model=str(_required(system, "system", "water_model")),
            complex_path=path_value(_required(system, "system", "complex")),
            temperature_k=_positive_float(
                system.get("temperature_k", 300.0), "[system].temperature_k"
            ),
            pressure_bar=_positive_float(
                system.get("pressure_bar", 1.0), "[system].pressure_bar"
            ),
            membrane=membrane,
            prepared_complex_gro=(path_value(membrane_gro) if membrane_gro else None),
            prepared_complex_topology=(
                path_value(membrane_topology) if membrane_topology else None
            ),
            chain=str(_required(mutation, "mutation", "chain")),
            residue=_positive_int(
                _required(mutation, "mutation", "residue"), "[mutation].residue"
            ),
            mutation=str(_required(mutation, "mutation", "target")),
            box_distance=_positive_float(
                system.get("box_distance", 1.0), "[system].box_distance"
            ),
            ion_concentration=_positive_float(
                system.get("ion_concentration", 0.15), "[system].ion_concentration"
            ),
            ligand_mol2=path_value(_required(ligand, "ligand", "mol2")),
            ligand_itp=path_value(ligand_itp) if ligand_itp else None,
            ligand_gro=path_value(ligand_gro) if ligand_gro else None,
            ligand_top=path_value(ligand_top) if ligand_top else None,
            ligand_resname=str(_required(ligand, "ligand", "resname")),
            ligand_charge=int(ligand.get("charge", 0)),
            ligand_charge_method=charge_method,
            ligand_prep_mode=_choice(
                ligand.get("preparation_mode", "SKIP"),
                "[ligand].preparation_mode",
                {"OVERWRITE", "SKIP"},
            ),
            start_replicate=_positive_int(
                simulation.get("start_replicate", 1), "[simulation].start_replicate"
            ),
            replicates=replicates,
            system_setup_mode=_choice(
                system.get("setup_mode", "SKIP"),
                "[system].setup_mode",
                {"OVERWRITE", "SKIP"},
            ),
            lambda_windows=lambda_windows,
            max_parallel_jobs=_positive_int(
                execution.get("max_parallel_jobs", 3),
                "[execution].max_parallel_jobs",
            ),
            md_timestep_ps=_positive_float(
                simulation.get("md_timestep_ps", timestep_ps),
                "[simulation].md_timestep_ps",
            ),
            timestep_ps=timestep_ps,
            nvt_ps=nvt_ps,
            npt_ps=npt_ps,
            production_ns=production_ns,
            nvt_steps=_step_count(simulation, "nvt_steps", nvt_ps, timestep_ps),
            npt_steps=_step_count(simulation, "npt_steps", npt_ps, timestep_ps),
            production_steps=_step_count(
                simulation, "production_steps", production_ns * 1000.0, timestep_ps
            ),
            lambda_function=lambda_function,
            lambda_mode=str(simulation.get("lambda_mode", "single")),
            lambda_power=parsed_lambda_power,
            simulation_mode=_choice(
                simulation.get("mode", "RESUME"),
                "[simulation].mode",
                {"OVERWRITE", "RESUME", "SKIP"},
            ),
            worker_script=path_value(worker) if worker else None,
            environment_setup=path_value(environment) if environment else None,
            mdrun_em_options=_command_options(
                execution.get("mdrun_em_options", ["-ntmpi", "1"]),
                "[execution].mdrun_em_options",
            ),
            mdrun_md_options=_command_options(
                execution.get("mdrun_md_options", ["-ntmpi", "1"]),
                "[execution].mdrun_md_options",
            ),
            mdrun_fep_options=_command_options(
                execution.get("mdrun_fep_options", ["-ntmpi", "1"]),
                "[execution].mdrun_fep_options",
            ),
            mdrun_switch_options=_command_options(
                execution.get("mdrun_switch_options", ["-ntmpi", "1"]),
                "[execution].mdrun_switch_options",
            ),
            slurm_partition=str(slurm.get("partition", "gpu")),
            slurm_nodes=_positive_int(slurm.get("nodes", 1), "[slurm].nodes"),
            slurm_tasks=_positive_int(slurm.get("tasks", 1), "[slurm].tasks"),
            slurm_cpus_per_task=_positive_int(
                slurm.get("cpus_per_task", 8), "[slurm].cpus_per_task"
            ),
            slurm_gpus=_positive_int(slurm.get("gpus", 1), "[slurm].gpus", 0),
            slurm_memory=str(slurm.get("memory", "16G")),
            slurm_walltime=str(slurm.get("walltime", "08:00:00")),
        )

    @property
    def project_dir(self) -> Path:
        return self.work_dir / self.job_name

    @property
    def total_fep_tasks(self) -> int:
        return 2 * self.replicates * self.lambda_windows

    @property
    def staged_ligand_dir(self) -> Path:
        return self.project_dir / "para_ligand" / f"{self.ligand_resname}.acpype"

    def validate_inputs(self, *, require_worker: bool = False) -> None:
        if not self.complex_path.is_file():
            raise ValueError(f"Complex structure does not exist: {self.complex_path}")
        if not self.ligand_mol2.is_file():
            raise ValueError(f"Ligand MOL2 does not exist: {self.ligand_mol2}")
        for label, path in (
            ("Ligand ITP", self.ligand_itp),
            ("Ligand GRO", self.ligand_gro),
            ("Ligand TOP", self.ligand_top),
        ):
            if path is not None and not path.is_file():
                raise ValueError(f"{label} does not exist: {path}")
        if self.membrane.enabled:
            if (
                self.prepared_complex_gro is None
                or self.prepared_complex_topology is None
            ):
                raise ValueError(
                    "Membrane residue FEP requires [system].prepared_complex_gro "
                    "and [system].prepared_complex_topology"
                )
            for label, path in (
                ("Prepared membrane GRO", self.prepared_complex_gro),
                ("Prepared membrane topology", self.prepared_complex_topology),
            ):
                if not path.is_file():
                    raise ValueError(f"{label} does not exist: {path}")
            validate_membrane_gro(self.prepared_complex_gro, self.membrane)
            if self.system_setup_mode != "SKIP":
                raise ValueError(
                    "Membrane residue FEP requires [system].setup_mode = 'SKIP'; "
                    "the soluble setup script must not rebuild the membrane"
                )
        if self.environment_setup is not None and not self.environment_setup.is_file():
            raise ValueError(
                f"Environment setup script does not exist: {self.environment_setup}"
            )
        if require_worker:
            installed_worker = self.project_dir / "ResAlchemFEP.sh"
            if self.worker_script is None and not installed_worker.is_file():
                raise ValueError(
                    "Submission requires [execution].worker_script or an existing "
                    f"{installed_worker}"
                )
            if self.worker_script is not None and not self.worker_script.is_file():
                raise ValueError(
                    f"FEP worker script does not exist: {self.worker_script}"
                )

    def shell_variables(self) -> dict[str, str | int | float]:
        values: dict[str, str | int | float] = {
            "ROOT": str(self.work_dir),
            "WORK_DIR": str(self.work_dir),
            "JOB_NAME": self.job_name,
            "GMX": self.gmx,
            "FF": self.force_field,
            "WATER": self.water_model,
            "COMPLEX": str(self.complex_path),
            "TEMPERATURE": self.temperature_k,
            "PRESSURE": self.pressure_bar,
            "CHAIN": self.chain,
            "RESID": self.residue,
            "MUT": self.mutation,
            "BOX_DISTANCE": self.box_distance,
            "ION_CONC": self.ion_concentration,
            "LIGAND_MOL2": str(self.ligand_mol2),
            "LIGAND_RESNAME": self.ligand_resname,
            "LIGAND_CHARGE": self.ligand_charge,
            "LIGAND_CHARGE_METHOD": self.ligand_charge_method,
            "LIGAND_PREP_MODE": self.ligand_prep_mode,
            "START_REP": self.start_replicate,
            "NREP": self.replicates,
            "SYSTEM_SETUP_MODE": self.system_setup_mode,
            "NLAMBDA": self.lambda_windows,
            "MAX_FEP_JOBS": self.max_parallel_jobs,
            "N_FEP_TASKS": self.total_fep_tasks,
            "DT_PS": self.md_timestep_ps,
            "FEP_DT_PS": self.timestep_ps,
            "FEP_NVT_PS": self.nvt_ps,
            "FEP_NPT_PS": self.npt_ps,
            "FEP_PROD_NS": self.production_ns,
            "FEP_NVT_STEPS": self.nvt_steps,
            "FEP_NPT_STEPS": self.npt_steps,
            "FEP_PROD_STEPS": self.production_steps,
            "FEP_LAMBDA_FUNCTION": self.lambda_function,
            "LAMBDA_MODE": self.lambda_mode,
            "SIMULATION_MODE": self.simulation_mode,
            "MDRUN_EM_OPTIONS": shlex.join(self.mdrun_em_options),
            "MDRUN_MD_OPTIONS": shlex.join(self.mdrun_md_options),
            "MDRUN_FEP_OPTIONS": shlex.join(self.mdrun_fep_options),
            "MDRUN_SWITCH_OPTIONS": shlex.join(self.mdrun_switch_options),
            "FEP_PARTITION": self.slurm_partition,
            "FEP_NODES": self.slurm_nodes,
            "FEP_NTASKS": self.slurm_tasks,
            "FEP_CPUS_PER_TASK": self.slurm_cpus_per_task,
            "FEP_GPUS": self.slurm_gpus,
            "FEP_MEMORY": self.slurm_memory,
            "FEP_WALLTIME": self.slurm_walltime,
        }
        values.update(self.membrane.shell_variables())
        if self.membrane.enabled:
            values["MEMBRANE_MDP"] = str(
                self.project_dir / "mdp" / "membrane_pressure.mdp"
            )
        if self.prepared_complex_gro is not None:
            values["PREPARED_COMPLEX_GRO"] = str(self.prepared_complex_gro)
        if self.prepared_complex_topology is not None:
            values["PREPARED_COMPLEX_TOPOLOGY"] = str(self.prepared_complex_topology)
        if self.ligand_itp is not None:
            values["LIGAND_ITP"] = str(self.ligand_itp)
        if self.ligand_gro is not None:
            values["LIGAND_GRO"] = str(self.ligand_gro)
        if self.ligand_top is not None:
            values["LIGAND_TOP"] = str(self.ligand_top)
        if self.lambda_power is not None:
            values["FEP_LAMBDA_POWER"] = self.lambda_power
        return values

    def render_shell_config(self) -> str:
        lines = [
            "# Generated by AlchemForge. Do not edit; edit the TOML source instead.",
            f"# Source: {self.source}",
        ]
        for key, value in self.shell_variables().items():
            lines.append(f"{key}={shlex.quote(str(value))}")
        return "\n".join(lines) + "\n"
