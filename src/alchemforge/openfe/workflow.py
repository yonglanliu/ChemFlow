"""Staging support shared by the OpenFE desktop workflows."""

from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

WORKFLOWS = {"rbfe", "abfe", "septop"}
SUPPORTED_PLATFORMS = {"CPU", "CUDA", "OpenCL"}
SUPPORTED_WATER_MODELS = {"tip3p", "spce", "tip4pew", "tip5p"}
SUPPORTED_BOX_SHAPES = {"cube", "dodecahedron", "octahedron"}
DEFAULT_FORCEFIELDS = (
    "amber/ff14SB.xml",
    "amber/tip3p_standard.xml",
    "amber/tip3p_HFE_multivalent.xml",
    "amber/phosaa10.xml",
    "amber/lipid17_merged.xml",
)


@dataclass(frozen=True)
class OpenFEWorkflowConfig:
    workflow: str
    job_name: str
    protein_pdb: Path
    ligand_sdf: Path
    task_directory: Path
    cofactor_sdf: Path | None = None
    ligand_network_graphml: Path | None = None
    platform: str = "CPU"
    membrane_system: bool = False
    complex_box_vectors: list[list[float]] | None = None
    mapper: str = "LomapAtomMapper"
    planner: str = "minimal_spanning"
    transformation: str = "both"
    protocol_repeats: int = 1
    temperature: float = 298.15
    pressure: float = 1.0
    forcefields: tuple[str, ...] = DEFAULT_FORCEFIELDS
    solvent_model: str = "tip3p"
    box_shape: str = "dodecahedron"
    slurm_partition: str = "gpu"
    slurm_nodes: int = 1
    slurm_tasks: int = 1
    slurm_cpus_per_task: int = 8
    slurm_gpus: int = 1
    slurm_memory: str = "16G"
    slurm_walltime: str = "24:00:00"
    max_parallel_jobs: int = 3
    model_parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.workflow not in WORKFLOWS:
            raise ValueError(f"Unsupported OpenFE workflow: {self.workflow}")
        if not self.job_name.strip():
            raise ValueError("Job name cannot be empty")
        if not self.protein_pdb.is_file() or self.protein_pdb.suffix.lower() != ".pdb":
            raise ValueError(f"Protein must be an existing PDB: {self.protein_pdb}")
        if not self.ligand_sdf.is_file() or self.ligand_sdf.suffix.lower() != ".sdf":
            raise ValueError(f"Ligands must be an existing SDF: {self.ligand_sdf}")
        if self.cofactor_sdf is not None and (
            not self.cofactor_sdf.is_file()
            or self.cofactor_sdf.suffix.lower() != ".sdf"
        ):
            raise ValueError(f"Cofactor must be an existing SDF: {self.cofactor_sdf}")
        sdf_text = self.ligand_sdf.read_text(encoding="utf-8", errors="replace")
        molecule_count = sdf_text.count("$$$$")
        if molecule_count == 0 and sdf_text.strip():
            molecule_count = 1
        if self.workflow == "abfe" and molecule_count != 1:
            raise ValueError(
                f"OpenFE ABFE requires exactly one ligand; found {molecule_count}"
            )
        if self.workflow in {"rbfe", "septop"} and molecule_count < 2:
            raise ValueError(
                f"OpenFE {self.workflow.upper()} requires at least two ligands; "
                f"found {molecule_count}"
            )
        if self.platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Unsupported OpenMM platform: {self.platform}")
        if not self.forcefields or any(not item.strip() for item in self.forcefields):
            raise ValueError("Select at least one system force-field XML file")
        if self.solvent_model not in SUPPORTED_WATER_MODELS:
            raise ValueError(
                "Unsupported OpenFE water model: "
                f"{self.solvent_model}; choose one of "
                f"{', '.join(sorted(SUPPORTED_WATER_MODELS))}"
            )
        if self.box_shape not in SUPPORTED_BOX_SHAPES:
            raise ValueError(
                "Unsupported OpenFE water-box shape: "
                f"{self.box_shape}; choose one of "
                f"{', '.join(sorted(SUPPORTED_BOX_SHAPES))}"
            )
        if self.protocol_repeats < 1:
            raise ValueError("Protocol repeats must be at least 1")
        if self.temperature <= 0 or self.pressure <= 0:
            raise ValueError("Temperature and pressure must be positive")
        if (
            min(
                self.slurm_nodes,
                self.slurm_tasks,
                self.slurm_cpus_per_task,
                self.max_parallel_jobs,
            )
            < 1
        ):
            raise ValueError(
                "Slurm nodes, tasks, CPUs, and parallel jobs must be positive"
            )
        if self.slurm_gpus < 0:
            raise ValueError("Slurm GPUs cannot be negative")
        if not self.slurm_memory.strip() or not self.slurm_walltime.strip():
            raise ValueError("Slurm memory and walltime cannot be empty")
        if self.membrane_system:
            vectors = self.complex_box_vectors
            if (
                vectors is None
                or len(vectors) != 3
                or any(len(row) != 3 for row in vectors)
            ):
                raise ValueError(
                    "A membrane system requires three 3D complex box vectors"
                )

    def payload(self) -> dict[str, Any]:
        config: dict[str, Any] = dict(self.model_parameters)
        config.update(
            {
                "protein_pdb": str(self.protein_pdb.resolve()),
                "ligand_sdf": str(self.ligand_sdf.resolve()),
                "cofactor_sdf": (
                    str(self.cofactor_sdf.resolve()) if self.cofactor_sdf else None
                ),
                "workdir": str(self.task_directory.resolve()),
                "json_dir": str((self.task_directory / "json").resolve()),
                "platform": self.platform,
                "membrane_system": self.membrane_system,
                "complex_box_vectors": self.complex_box_vectors,
                "ligand_network_graphml": (
                    str(self.ligand_network_graphml.resolve())
                    if self.ligand_network_graphml
                    else None
                ),
                "temperature": self.temperature,
                "pressure": self.pressure,
                "forcefields": list(self.forcefields),
                "solvent_model": self.solvent_model,
                "box_shape": self.box_shape,
            }
        )
        if self.workflow == "rbfe":
            config.update(
                {
                    "complex_protocol_repeats": self.protocol_repeats,
                    "solvent_protocol_repeats": self.protocol_repeats,
                    "alchemical_network": "all",
                    "transformation": self.transformation,
                    "mapper": self.mapper,
                    "planner": self.planner,
                }
            )
        elif self.workflow == "abfe":
            config.update(
                {"name": self.job_name, "protocol_repeats": self.protocol_repeats}
            )
        else:
            config.update(
                {
                    "protocol_repeats": self.protocol_repeats,
                    "mapper": self.mapper,
                    "planner": self.planner,
                }
            )
        return config


def stage(config: OpenFEWorkflowConfig) -> tuple[Path, Path]:
    config.validate()
    config.task_directory.mkdir(parents=True, exist_ok=True)
    inputs_dir = config.task_directory / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    staged_inputs = {
        "protein_pdb": (config.protein_pdb, inputs_dir / "protein.pdb"),
        "ligand_sdf": (config.ligand_sdf, inputs_dir / "ligands.sdf"),
    }
    if config.cofactor_sdf is not None:
        staged_inputs["cofactor_sdf"] = (
            config.cofactor_sdf,
            inputs_dir / "cofactor.sdf",
        )
    if config.ligand_network_graphml is not None:
        staged_inputs["ligand_network_graphml"] = (
            config.ligand_network_graphml,
            inputs_dir / "ligand_network.graphml",
        )
    for source, destination in staged_inputs.values():
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    payload = config.payload()
    payload.update(
        {
            key: str(destination.relative_to(config.task_directory))
            for key, (_source, destination) in staged_inputs.items()
        }
    )
    payload.update({"workdir": ".", "json_dir": "json"})
    config_path = config.task_directory / f"{config.workflow}_conf.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    runner = config.task_directory / "run_openfe_workflow.sh"
    module = f"alchemforge.openfe.{config.workflow}.create_json"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'cd "${PROJECT_DIR}"\n'
        'CONFIG="${1:?Usage: $0 /path/to/config.yaml}"\n'
        f'python -m {shlex.quote(module)} --conf_file "${{CONFIG}}"\n',
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | 0o100)

    slurm_dir = config.task_directory / "slurm"
    logs_dir = config.task_directory / "logs"
    slurm_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    execution_config = config.task_directory / "openfe_execution.yaml"
    execution_config.write_text(
        yaml.safe_dump(
            {
                "json_directory": "json",
                "output_directory": "results",
                "max_parallel": config.max_parallel_jobs,
                "openfe_command": "openfe",
                "resume": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    partition = (
        f"#SBATCH --partition={config.slurm_partition}\n"
        if config.slurm_partition.strip()
        else ""
    )
    gpu = f"#SBATCH --gpus={config.slurm_gpus}\n" if config.slurm_gpus else ""
    resources = (
        partition
        + f"#SBATCH --nodes={config.slurm_nodes}\n"
        + f"#SBATCH --ntasks={config.slurm_tasks}\n"
        + f"#SBATCH --cpus-per-task={config.slurm_cpus_per_task}\n"
        + gpu
        + f"#SBATCH --mem={config.slurm_memory}\n"
        + f"#SBATCH --time={config.slurm_walltime}\n"
    )
    generate_job = slurm_dir / "01_generate_transformations.sbatch"
    generate_job.write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH --job-name={config.job_name}_generate\n"
        + resources
        + "#SBATCH --output=logs/generate_%j.out\n"
        + "#SBATCH --error=logs/generate_%j.err\n"
        + "set -euo pipefail\n"
        + 'PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
        + 'cd "${PROJECT_DIR}"\n'
        + 'echo "OpenFE generation started on $(hostname) at $(date -Is)"\n'
        + f"./{runner.name} ./{config_path.name}\n",
        encoding="utf-8",
    )
    generate_job.chmod(generate_job.stat().st_mode | 0o100)

    execute_job = slurm_dir / "02_run_transformations.sbatch"
    execute_job.write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH --job-name={config.job_name}_openfe\n"
        + resources
        + "#SBATCH --output=logs/openfe_%j.out\n"
        + "#SBATCH --error=logs/openfe_%j.err\n"
        + "set -euo pipefail\n"
        + 'PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
        + 'cd "${PROJECT_DIR}"\n'
        + 'echo "OpenFE simulations started on $(hostname) at $(date -Is)"\n'
        + "python -m alchemforge.openfe.execute --config "
        + f"./{execution_config.name}\n",
        encoding="utf-8",
    )
    execute_job.chmod(execute_job.stat().st_mode | 0o100)

    submit = config.task_directory / "submit_openfe_pipeline.sh"
    submit.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'cd "${PROJECT_DIR}"\n'
        "mkdir -p logs\n"
        f"generation_job_id=$(sbatch --parsable slurm/{generate_job.name})\n"
        'echo "Submitted transformation generation: ${generation_job_id}"\n'
        f"simulation_job_id=$(sbatch --parsable --dependency=afterok:${{generation_job_id}} slurm/{execute_job.name})\n"
        'echo "Submitted OpenFE simulations: ${simulation_job_id}"\n',
        encoding="utf-8",
    )
    submit.chmod(submit.stat().st_mode | 0o100)
    return runner, config_path
