"""Configuration for the GROMACS separated-topologies (SepTop) workflow."""

from __future__ import annotations

import math
import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def _positive_int(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{label} must be greater than zero")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    result = float(value)
    if result < 0:
        raise ValueError(f"{label} must not be negative")
    return result


def _float_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be an array containing two numbers")
    return (
        _positive_float(value[0], f"{label}[0]"),
        _positive_float(value[1], f"{label}[1]"),
    )


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(f"{label} must be a non-empty array of strings")
    return tuple(item.strip().upper() for item in value)


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


@dataclass(frozen=True)
class SepTopGromacsConfig:
    source: Path
    work_dir: Path
    job_name: str
    ligand_library: Path
    existing_network_dir: Path | None
    prepared_root: Path
    septop_source_dir: Path
    system_type: str
    membrane_normal: str
    lipid_resnames: tuple[str, ...]
    minimum_lipids: int
    complexes_pre_aligned: bool
    membrane_barostat: str
    membrane_reference_pressure_bar: tuple[float, float]
    membrane_compressibility_bar: tuple[float, float]
    membrane_refcoord_scaling: str
    network_method: str
    atom_mapper: str
    element_change: bool
    max3d_angstrom: float
    require_cycle_covering: bool
    radial_hub: str | None
    redundant_networks: int
    complex_gamma: float
    solvent_gamma: float
    complex_ligand_resname: str
    solvent_ligand_resname: str
    allow_charge_change: bool
    replicates: int
    start_replicate: int
    timestep_ps: float
    equilibration_ns: float
    pressure_equilibration_ns: float
    production_ns: float
    gmx: str
    mdrun_options: tuple[str, ...]
    max_parallel_jobs: int
    slurm_partition: str | None
    slurm_time: str
    slurm_nodes: int
    slurm_tasks: int
    slurm_gpus: int
    slurm_cpus: int
    slurm_memory: str

    @classmethod
    def load(cls, path: str | Path) -> SepTopGromacsConfig:
        source = Path(path).expanduser().resolve()
        with source.open("rb") as handle:
            data = tomllib.load(handle)
        project = _table(data, "project")
        inputs = _table(data, "inputs")
        system = _table(data, "system")
        network = _table(data, "network")
        protocol = _table(data, "protocol")
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
        method = str(network.get("method", "minimal_spanning")).lower()
        if method not in {"lomap", "minimal_spanning", "minimal_redundant", "radial"}:
            raise ValueError("[network].method is not supported")
        hub = network.get("radial_hub")
        if method == "radial" and not hub:
            raise ValueError("[network].radial_hub is required for a radial network")
        system_type = str(system.get("type", "soluble")).lower()
        if system_type not in {"soluble", "membrane"}:
            raise ValueError("[system].type must be 'soluble' or 'membrane'")
        membrane_normal = str(system.get("membrane_normal", "z")).lower()
        if system_type == "membrane" and membrane_normal != "z":
            raise ValueError(
                "GROMACS semi-isotropic coupling requires the membrane normal to be z"
            )
        refcoord_scaling = str(system.get("refcoord_scaling", "com")).lower()
        if refcoord_scaling not in {"no", "all", "com"}:
            raise ValueError("[system].refcoord_scaling must be no, all, or com")
        return cls(
            source=source,
            work_dir=resolve(_required(project, "project", "work_dir")),
            job_name=name,
            ligand_library=resolve(_required(inputs, "inputs", "ligand_library")),
            existing_network_dir=(
                resolve(network["existing_network_dir"])
                if network.get("existing_network_dir")
                else None
            ),
            prepared_root=resolve(_required(inputs, "inputs", "prepared_root")),
            septop_source_dir=resolve(_required(inputs, "inputs", "septop_source_dir")),
            system_type=system_type,
            membrane_normal=membrane_normal,
            lipid_resnames=_string_tuple(
                system.get(
                    "lipid_resnames",
                    [
                        "POPC",
                        "POPE",
                        "POPS",
                        "POPG",
                        "DOPC",
                        "DOPE",
                        "DOPS",
                        "DPPC",
                        "DMPC",
                        "DLPC",
                        "CHOL",
                        "PSM",
                    ],
                ),
                "[system].lipid_resnames",
            ),
            minimum_lipids=_positive_int(
                system.get("minimum_lipids", 20), "[system].minimum_lipids"
            ),
            complexes_pre_aligned=_boolean(
                system.get("complexes_pre_aligned", system_type == "membrane"),
                "[system].complexes_pre_aligned",
            ),
            membrane_barostat=str(system.get("barostat", "Parrinello-Rahman")),
            membrane_reference_pressure_bar=_float_pair(
                system.get("reference_pressure_bar", [1.0, 1.0]),
                "[system].reference_pressure_bar",
            ),
            membrane_compressibility_bar=_float_pair(
                system.get("compressibility_bar", [4.5e-5, 4.5e-5]),
                "[system].compressibility_bar",
            ),
            membrane_refcoord_scaling=refcoord_scaling,
            network_method=method,
            atom_mapper=str(network.get("atom_mapper", "lomap")).lower(),
            element_change=_boolean(
                network.get("element_change", False), "[network].element_change"
            ),
            max3d_angstrom=_positive_float(
                network.get("max3d_angstrom", 1.0), "[network].max3d_angstrom"
            ),
            require_cycle_covering=_boolean(
                network.get("require_cycle_covering", True),
                "[network].require_cycle_covering",
            ),
            radial_hub=str(hub) if hub else None,
            redundant_networks=_positive_int(
                network.get("redundant_networks", 2), "[network].redundant_networks"
            ),
            complex_gamma=_positive_float(
                protocol.get("complex_gamma", 0.5), "[protocol].complex_gamma"
            ),
            solvent_gamma=_positive_float(
                protocol.get("solvent_gamma", 0.3), "[protocol].solvent_gamma"
            ),
            complex_ligand_resname=str(protocol.get("complex_ligand_resname", "LIG")),
            solvent_ligand_resname=str(protocol.get("solvent_ligand_resname", "UNL")),
            allow_charge_change=_boolean(
                protocol.get("allow_charge_change", False),
                "[protocol].allow_charge_change",
            ),
            replicates=_positive_int(
                protocol.get("replicates", 1), "[protocol].replicates"
            ),
            start_replicate=_positive_int(
                protocol.get("start_replicate", 1), "[protocol].start_replicate"
            ),
            timestep_ps=_positive_float(
                protocol.get("timestep_ps", 0.002), "[protocol].timestep_ps"
            ),
            equilibration_ns=_positive_float(
                protocol.get("equilibration_ns", 1.0), "[protocol].equilibration_ns"
            ),
            pressure_equilibration_ns=_nonnegative_float(
                protocol.get(
                    "pressure_equilibration_ns",
                    1.0 if system_type == "membrane" else 0.0,
                ),
                "[protocol].pressure_equilibration_ns",
            ),
            production_ns=_positive_float(
                protocol.get("production_ns", 5.0), "[protocol].production_ns"
            ),
            gmx=str(execution.get("gromacs_command", "gmx")),
            mdrun_options=_options(
                execution.get("mdrun_options", ["-ntmpi", "1"]),
                "[execution].mdrun_options",
            ),
            max_parallel_jobs=_positive_int(
                execution.get("max_parallel_jobs", 3), "[execution].max_parallel_jobs"
            ),
            slurm_partition=str(slurm["partition"]) if slurm.get("partition") else None,
            slurm_time=str(slurm.get("time", "24:00:00")),
            slurm_nodes=_positive_int(slurm.get("nodes", 1), "[slurm].nodes"),
            slurm_tasks=_positive_int(slurm.get("tasks", 1), "[slurm].tasks"),
            slurm_gpus=_positive_int(slurm.get("gpus", 1), "[slurm].gpus", 0),
            slurm_cpus=_positive_int(
                slurm.get("cpus_per_task", 4), "[slurm].cpus_per_task"
            ),
            slurm_memory=str(slurm.get("memory", "16G")),
        )

    @property
    def project_dir(self) -> Path:
        return self.work_dir / self.job_name

    @property
    def equilibration_steps(self) -> int:
        return self._steps(self.equilibration_ns)

    @property
    def production_steps(self) -> int:
        return self._steps(self.production_ns)

    @property
    def pressure_equilibration_steps(self) -> int:
        return self._steps(self.pressure_equilibration_ns)

    def _steps(self, duration_ns: float) -> int:
        exact = duration_ns * 1000.0 / self.timestep_ps
        rounded = round(exact)
        if not math.isclose(exact, rounded, abs_tol=1e-6):
            raise ValueError(
                "Simulation duration is not an integer number of timesteps"
            )
        return rounded

    def validate_inputs(self, *, runtime: bool = False) -> None:
        if (
            not self.ligand_library.is_file()
            or self.ligand_library.suffix.lower() != ".sdf"
        ):
            raise ValueError(
                f"[inputs].ligand_library must be an existing SDF: {self.ligand_library}"
            )
        if self.existing_network_dir is not None:
            for filename in (
                "edges.csv",
                "ligand_network.graphml",
                "network_manifest.json",
            ):
                if not (self.existing_network_dir / filename).is_file():
                    raise ValueError(
                        f"Ligand network bundle is missing {filename}: "
                        f"{self.existing_network_dir}"
                    )
        if not self.prepared_root.is_dir():
            raise ValueError(
                f"Prepared ligand root does not exist: {self.prepared_root}"
            )
        required_source = (
            "make_septop.py",
            "boresch_restraints.py",
            "combine_coordinates.py",
            "distance_solvent.py",
        )
        missing = [
            name
            for name in required_source
            if not (self.septop_source_dir / name).is_file()
        ]
        if missing:
            raise ValueError(
                "SepTop source directory is missing: " + ", ".join(missing)
            )
        if runtime:
            import importlib.util

            missing_modules = [
                name
                for name in ("openeye", "parmed", "mdtraj")
                if importlib.util.find_spec(name) is None
            ]
            if missing_modules:
                raise ValueError(
                    "SepTop preparation is missing Python module(s): "
                    + ", ".join(missing_modules)
                    + ". The OpenEye module also requires a valid license."
                )

    def validate_prepared_ligands(self, names: set[str]) -> None:
        required = (
            "ligand.mol2",
            "complex.pdb",
            "complex.gro",
            "complex.top",
            "solvent.gro",
            "solvent.top",
        )
        problems = []
        for name in sorted(names):
            directory = self.prepared_root / name
            missing = [item for item in required if not (directory / item).is_file()]
            if missing:
                problems.append(f"{name}: {', '.join(missing)}")
        if problems:
            raise ValueError(
                "Prepared ligand inputs are incomplete (" + "; ".join(problems) + ")"
            )
        if self.system_type == "membrane":
            self._validate_membrane_complexes(names)

    def _validate_membrane_complexes(self, names: set[str]) -> None:
        problems = []
        for name in sorted(names):
            gro = self.prepared_root / name / "complex.gro"
            lines = gro.read_text(encoding="utf-8").splitlines()
            try:
                atom_count = int(lines[1].strip())
                atom_lines = lines[2 : 2 + atom_count]
                box = [float(value) for value in lines[2 + atom_count].split()]
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"Cannot parse membrane coordinate file: {gro}"
                ) from error
            if len(box) not in {3, 9} or any(value <= 0 for value in box[:3]):
                problems.append(f"{name}: missing or invalid periodic box vectors")
                continue
            lipids = {
                (line[:5].strip(), line[5:10].strip().upper())
                for line in atom_lines
                if len(line) >= 10 and line[5:10].strip().upper() in self.lipid_resnames
            }
            if len(lipids) < self.minimum_lipids:
                problems.append(
                    f"{name}: found {len(lipids)} lipid residues, expected at least "
                    f"{self.minimum_lipids}; check [system].lipid_resnames"
                )
        if problems:
            raise ValueError("Membrane validation failed (" + "; ".join(problems) + ")")
