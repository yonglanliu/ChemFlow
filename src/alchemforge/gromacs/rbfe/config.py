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


def _restraint_forces(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain protein, ligand, and cofactor values")
    result = tuple(_nonnegative_float(item, label) for item in value)
    return result[0], result[1], result[2]


def _name_array(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of names")
    return tuple(item.strip().upper() for item in value if item.strip())


def _positive_integer_array(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array of positive residue numbers")
    return tuple(_positive_int(item, label) for item in value)


def _choice(value: Any, label: str, choices: set[str]) -> str:
    result = str(value).lower()
    if result not in choices:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return result


def _stage_choices(
    value: Any,
    default: tuple[str, ...],
    label: str,
    choices: dict[str, str],
) -> tuple[str, ...]:
    values = list(default) if value is None else value
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError(f"{label} must contain six stage values")
    result: list[str] = []
    for item in values:
        canonical = choices.get(str(item).casefold())
        if canonical is None:
            raise ValueError(
                f"{label} contains unsupported value {item!r}; choose from "
                + ", ".join(choices.values())
            )
        result.append(canonical)
    return tuple(result)


def _stage_timesteps(value: Any) -> tuple[float, ...]:
    values = [0.0, 0.001, 0.001, 0.002, 0.002, 0.002] if value is None else value
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError("[stage_protocol].timesteps_ps must contain six values")
    result = tuple(
        _nonnegative_float(item, "[stage_protocol].timesteps_ps") for item in values
    )
    if result[0] != 0.0 or any(item <= 0 for item in result[1:]):
        raise ValueError(
            "The EM timestep must be 0 and all dynamic-stage timesteps must be positive"
        )
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
    protein_restraint_force: float
    ligand_restraint_force: float
    cofactor_restraint_force: float
    em_restraint_forces: tuple[float, float, float]
    nvt_restraint_forces: tuple[float, float, float]
    npt_restraint_forces: tuple[float, float, float]
    transition_restraint_forces: tuple[float, float, float]
    final_restraint_forces: tuple[float, float, float]
    production_restraint_forces: tuple[float, float, float]
    protein_residue_ids: tuple[int, ...]
    protein_atom_names: tuple[str, ...]
    ligand_resnames: tuple[str, ...]
    ligand_atom_names: tuple[str, ...]
    cofactor_resnames: tuple[str, ...]
    cofactor_atom_names: tuple[str, ...]
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
    sampling_method: str
    exchange_interval_steps: int
    lambda_windows: int
    lambda_schedule_mode: str
    lambda_function: str
    lambda_power: float
    coul_lambdas: tuple[float, ...] | None
    vdw_lambdas: tuple[float, ...] | None
    bonded_lambdas: tuple[float, ...] | None
    restraint_lambdas: tuple[float, ...] | None
    timestep_ps: float
    stage_integrators: tuple[str, ...]
    stage_timesteps_ps: tuple[float, ...]
    stage_temperature_coupling: tuple[str, ...]
    stage_pressure_coupling: tuple[str, ...]
    equilibration_ns: float
    em_steps: int
    em_tolerance: float
    nvt_ns: float
    npt_ns: float
    transition_ns: float
    final_equilibration_ns: float
    production_ns: float
    mdrun_options: tuple[str, ...]
    hrex_gmx_command: str
    worker_script: Path | None
    slurm_partition: str
    slurm_nodes: int
    slurm_tasks: int
    slurm_cpus_per_task: int
    slurm_gpus: int
    slurm_memory: str
    slurm_walltime: str
    slurm_conda_init: str
    slurm_conda_environment: str
    slurm_log_output: str
    slurm_log_error: str
    mdp_integrator: str
    mdp_constraints: str
    mdp_coulomb_cutoff_nm: float
    mdp_vdw_cutoff_nm: float
    mdp_softcore_alpha: float
    mdp_calc_lambda_neighbors: int
    mdp_nstdhdl: int

    def __post_init__(self) -> None:
        minimizers = {"steep", "cg", "l-bfgs"}
        if self.stage_integrators[0] not in minimizers:
            raise ValueError("The EM stage must use steep, cg, or l-bfgs")
        if any(value in minimizers for value in self.stage_integrators[1:]):
            raise ValueError("Dynamic stages cannot use an energy-minimization integrator")
        for index, integrator in enumerate(self.stage_integrators):
            if integrator == "sd" and self.stage_temperature_coupling[index] != "no":
                raise ValueError(
                    "integrator=sd requires temperature_coupling='no'; stochastic "
                    "dynamics controls temperature internally"
                )
        if self.stage_pressure_coupling[0] != "no":
            raise ValueError("Energy minimization cannot use pressure coupling")

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
        stage_protocol = _table(data, "stage_protocol")
        mdp = _table(data, "mdp")
        restraints = _table(data, "restraints")
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
        stage_restraint_forces = {
            stage: _restraint_forces(
                restraints.get(key, default), f"[restraints].{key}"
            )
            for stage, key, default in (
                ("em", "em_forces", [1000, 1000, 1000]),
                ("nvt", "nvt_forces", [1000, 1000, 1000]),
                ("npt", "npt_forces", [500, 500, 500]),
                ("transition", "transition_forces", [0, 100, 100]),
                ("final", "final_forces", [0, 20, 0]),
                ("production", "production_forces", [0, 0, 0]),
            )
        }
        protein_residue_ids = _positive_integer_array(
            restraints.get("protein_residue_ids", []),
            "[restraints].protein_residue_ids",
        )
        protein_atom_names = _name_array(
            restraints.get("protein_atom_names", []),
            "[restraints].protein_atom_names",
        )
        ligand_resnames = _name_array(
            restraints.get("ligand_resnames", []),
            "[restraints].ligand_resnames",
        )
        ligand_atom_names = _name_array(
            restraints.get("ligand_atom_names", []),
            "[restraints].ligand_atom_names",
        )
        cofactor_resnames = _name_array(
            restraints.get("cofactor_resnames", []),
            "[restraints].cofactor_resnames",
        )
        if any(not re.fullmatch(r"[A-Z0-9]{1,5}", name) for name in cofactor_resnames):
            raise ValueError(
                "[restraints].cofactor_resnames entries must contain 1-5 letters/numbers"
            )
        cofactor_atom_names = _name_array(
            restraints.get("cofactor_atom_names", []),
            "[restraints].cofactor_atom_names",
        )
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
        sampling_method = _choice(
            simulation.get("sampling_method", "independent"),
            "[simulation].sampling_method",
            {"independent", "hrex"},
        )
        exchange_interval_steps = _positive_int(
            simulation.get("exchange_interval_steps", 500),
            "[simulation].exchange_interval_steps",
        )
        hrex_gmx_command = str(
            execution.get("hrex_gromacs_command", "gmx_mpi")
        ).strip()
        if sampling_method == "hrex" and not hrex_gmx_command:
            raise ValueError(
                "[execution].hrex_gromacs_command is required for lambda-HREX"
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
            protein_restraint_force=_positive_float(
                max(forces[0] for forces in stage_restraint_forces.values()) or 1000.0,
                "[restraints].protein_force_kj_mol_nm2",
            ),
            ligand_restraint_force=_positive_float(
                max(forces[1] for forces in stage_restraint_forces.values()) or 1000.0,
                "[restraints].ligand_force_kj_mol_nm2",
            ),
            cofactor_restraint_force=_positive_float(
                max(forces[2] for forces in stage_restraint_forces.values()) or 1000.0,
                "[restraints].cofactor_force_kj_mol_nm2",
            ),
            em_restraint_forces=stage_restraint_forces["em"],
            nvt_restraint_forces=stage_restraint_forces["nvt"],
            npt_restraint_forces=stage_restraint_forces["npt"],
            transition_restraint_forces=stage_restraint_forces["transition"],
            final_restraint_forces=stage_restraint_forces["final"],
            production_restraint_forces=stage_restraint_forces["production"],
            protein_residue_ids=protein_residue_ids,
            protein_atom_names=protein_atom_names,
            ligand_resnames=ligand_resnames,
            ligand_atom_names=ligand_atom_names,
            cofactor_resnames=cofactor_resnames,
            cofactor_atom_names=cofactor_atom_names,
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
            sampling_method=sampling_method,
            exchange_interval_steps=exchange_interval_steps,
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
            stage_integrators=_stage_choices(
                stage_protocol.get("integrators"),
                ("steep", "md", "md", "sd", "sd", "sd"),
                "[stage_protocol].integrators",
                {
                    "steep": "steep",
                    "cg": "cg",
                    "l-bfgs": "l-bfgs",
                    "md": "md",
                    "md-vv": "md-vv",
                    "md-vv-avek": "md-vv-avek",
                    "sd": "sd",
                },
            ),
            stage_timesteps_ps=_stage_timesteps(
                stage_protocol.get("timesteps_ps")
            ),
            stage_temperature_coupling=_stage_choices(
                stage_protocol.get("temperature_coupling"),
                ("no", "Berendsen", "Berendsen", "no", "no", "no"),
                "[stage_protocol].temperature_coupling",
                {
                    "no": "no",
                    "berendsen": "Berendsen",
                    "v-rescale": "V-rescale",
                    "nose-hoover": "Nose-Hoover",
                },
            ),
            stage_pressure_coupling=_stage_choices(
                stage_protocol.get("pressure_coupling"),
                ("no", "no", "Berendsen", "C-rescale", "C-rescale", "C-rescale"),
                "[stage_protocol].pressure_coupling",
                {
                    "no": "no",
                    "berendsen": "Berendsen",
                    "c-rescale": "C-rescale",
                    "parrinello-rahman": "Parrinello-Rahman",
                },
            ),
            equilibration_ns=(
                _positive_float(
                    simulation.get("nvt_ns", simulation.get("equilibration_ns", 1.0) / 2),
                    "[simulation].nvt_ns",
                )
                + _positive_float(
                    simulation.get("npt_ns", simulation.get("equilibration_ns", 1.0) / 2),
                    "[simulation].npt_ns",
                )
            ),
            em_steps=_positive_int(
                simulation.get("em_steps", 5000), "[simulation].em_steps"
            ),
            em_tolerance=_positive_float(
                simulation.get("em_tolerance", 200.0),
                "[simulation].em_tolerance",
            ),
            nvt_ns=_positive_float(
                simulation.get("nvt_ns", simulation.get("equilibration_ns", 1.0) / 2),
                "[simulation].nvt_ns",
            ),
            npt_ns=_positive_float(
                simulation.get("npt_ns", simulation.get("equilibration_ns", 1.0) / 2),
                "[simulation].npt_ns",
            ),
            transition_ns=_positive_float(
                simulation.get("transition_ns", 0.5),
                "[simulation].transition_ns",
            ),
            final_equilibration_ns=_positive_float(
                simulation.get("final_equilibration_ns", 0.5),
                "[simulation].final_equilibration_ns",
            ),
            production_ns=_positive_float(
                simulation.get("production_ns", 5.0), "[simulation].production_ns"
            ),
            mdrun_options=_options(
                execution.get("mdrun_options", ["-ntmpi", "1"]),
                "[execution].mdrun_options",
            ),
            hrex_gmx_command=hrex_gmx_command,
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
            slurm_conda_init=str(
                slurm.get(
                    "conda_init",
                    "",
                )
            ),
            slurm_conda_environment=str(
                slurm.get(
                    "conda_environment",
                    "",
                )
            ),
            slurm_log_output=str(slurm.get("log_output", "logs/%A_%a.out")),
            slurm_log_error=str(slurm.get("log_error", "logs/%A_%a.err")),
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
        if self.sampling_method == "hrex":
            return 2 * self.replicates
        return 2 * self.replicates * self.lambda_windows

    @property
    def equilibration_steps(self) -> int:
        return self.nvt_steps + self.npt_steps

    @property
    def nvt_steps(self) -> int:
        return _steps(self.nvt_ns, self.stage_timesteps_ps[1])

    @property
    def npt_steps(self) -> int:
        return _steps(self.npt_ns, self.stage_timesteps_ps[2])

    @property
    def transition_steps(self) -> int:
        return _steps(self.transition_ns, self.stage_timesteps_ps[3])

    @property
    def final_equilibration_steps(self) -> int:
        return _steps(self.final_equilibration_ns, self.stage_timesteps_ps[4])

    @property
    def production_steps(self) -> int:
        return _steps(self.production_ns, self.stage_timesteps_ps[5])

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
            "SAMPLING_METHOD": self.sampling_method,
            "EXCHANGE_INTERVAL_STEPS": self.exchange_interval_steps,
            "NLAMBDA": self.lambda_windows,
            "LAMBDA_SCHEDULE_MODE": self.lambda_schedule_mode,
            "LAMBDA_FUNCTION": self.lambda_function,
            "LAMBDA_POWER": self.lambda_power,
            "DT_PS": self.timestep_ps,
            "EQUIL_NS": self.equilibration_ns,
            "EM_STEPS": self.em_steps,
            "EM_TOLERANCE": self.em_tolerance,
            "NVT_NS": self.nvt_ns,
            "NVT_STEPS": self.nvt_steps,
            "NPT_NS": self.npt_ns,
            "NPT_STEPS": self.npt_steps,
            "TRANSITION_NS": self.transition_ns,
            "TRANSITION_STEPS": self.transition_steps,
            "FINAL_EQUIL_NS": self.final_equilibration_ns,
            "FINAL_EQUIL_STEPS": self.final_equilibration_steps,
            "PROD_NS": self.production_ns,
            "EQUIL_STEPS": self.equilibration_steps,
            "PROD_STEPS": self.production_steps,
            "TASKS_PER_EDGE": self.tasks_per_edge,
            "MDRUN_OPTIONS": shlex.join(self.mdrun_options),
            "FEP_PARTITION": self.slurm_partition,
            "FEP_NODES": self.slurm_nodes,
            "FEP_NTASKS": self.slurm_tasks,
            "FEP_CPUS_PER_TASK": self.slurm_cpus_per_task,
            "FEP_GPUS": self.slurm_gpus,
            "FEP_MEMORY": self.slurm_memory,
            "FEP_WALLTIME": self.slurm_walltime,
            "FEP_CONDA_INIT": self.slurm_conda_init,
            "FEP_CONDA_ENVIRONMENT": self.slurm_conda_environment,
            "FEP_LOG_OUTPUT": self.slurm_log_output,
            "FEP_LOG_ERROR": self.slurm_log_error,
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
