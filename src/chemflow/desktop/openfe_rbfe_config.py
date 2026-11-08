"""Qt-free config core for the OpenFE RBFE workflow page.

Mirrors AlchemForge's OpenFE RBFE setup (the single YAML consumed by
``chemflow.fe.openfe.rbfe.create_json``) so the ChemFlow desktop can present the
same five-step workflow — charge, network/mapping, system (soluble or
membrane), FEP setup, job submission — and emit a faithful config plus an HPC
submit script. No OpenMM / OpenFE / Qt imports, so it is fully testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Full default config, mirroring generate_cli._RBFE_OPENFE_YAML. The form overrides
# a curated subset; everything else keeps these defaults so the YAML stays complete.
DEFAULTS: dict[str, Any] = {
    # --- Input files ---
    "ligand_sdf": "ligands.sdf",
    "protein_pdb": "protein.pdb",
    "cofactor_sdf": None,
    # --- Output ---
    "workdir": "work/rbfe",
    "result_dir": "work/rbfe/results",
    "json_dir": "work/rbfe/json",
    "input_json_files": None,
    "restart": False,
    # --- Alchemical network ---
    "alchemical_network": "all",          # "all" or a list of ligand names
    "transformation": "both",             # both | complex | solvent
    # --- Platform ---
    "platform": "CUDA",
    # --- Box shape (solvent + complex legs); membrane complex uses box vectors instead ---
    "solvent_box_shape": "dodecahedron",  # cube | dodecahedron | octahedron
    "complex_box_shape": "dodecahedron",
    "box_shape": "dodecahedron",          # legacy fallback for both legs
    # --- Membrane ---
    "membrane_system": False,
    "complex_box_size": None,             # [x, y, z] nm, required for membrane PBC
    # --- General ---
    "solvent_protocol_repeats": 1,
    "complex_protocol_repeats": 1,
    "pressure": 1.0,
    "temperature": 310.0,
    # --- Force field / nonbonded ---
    "ff_constraints": "hbonds",
    "nonbonded_cutoff": 1.0,
    "nonbonded_method": "PME",
    "rigid_water": True,
    "small_molecule_forcefield": "openff-2.1.0",
    "hydrogen_mass": 3.0,
    # --- Partial charges ---
    "off_toolkit_backend": "ambertools",  # openeye | rdkit | ambertools
    "partial_charge_method": "am1bcc",    # am1bcc | nagl
    # --- Alchemical settings ---
    "endstate_dispersion_correction": False,
    "explicit_charge_correction": True,
    "explicit_charge_correction_cutoff": 0.8,
    "softcore_LJ": "gapsys",              # gapsys | beutler
    "softcore_alpha": 0.85,
    "turn_off_core_unique_exceptions": False,
    "use_dispersion_correction": True,
    "n_lambda_windows_solvent_leg": 15,
    "n_lambda_windows_complex_leg": 15,
    # --- Solvation ---
    "complex_solvation_padding": 1.5,
    "solvent_solvation_padding": 1.5,
    "solvent_model": "tip3p",
    # --- Ions / solvent charge ---
    "ion_concentration": 0.15,   # molar (SolventComponent salt concentration)
    "positive_ion": "Na+",
    "negative_ion": "Cl-",
    "neutralize": True,          # add counter-ions to neutralize net charge
    # --- Simulation ---
    "early_termination_target_error": 0.0,
    "complex_sim_mini_step": 5000,
    "complex_sim_equil_length": 1.0,
    "complex_sim_prod_length": 15.0,
    "solvent_sim_mini_step": 5000,
    "solvent_sim_equil_length": 1.0,
    "solvent_sim_prod_length": 15.0,
    # --- Integrator ---
    "complex_timestep": 4.0,
    "solvent_timestep": 4.0,
    "barostat_frequency": 25.0,
    "surface_tension": 300.0,             # bar*nm (membrane only)
    # --- Mapper / planner ---
    "mapper": "KartografAtomMapper",      # LomapAtomMapper | KartografAtomMapper
    "planner": "minimal_redundant",       # minimal_spanning | minimal_redundant | radial
}

# Default HPC settings for the submit script.
HPC_DEFAULTS: dict[str, Any] = {
    "partition": "gpu",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 16,
    "gpus": 1,
    "gpu_type": "",           # e.g. v100x -> --gres=gpu:v100x:1 (empty -> gpu:1)
    "memory": "32G",
    "walltime": "240:00:00",
    "replicates": 3,          # repeats per transformation (array = N_json * replicates)
    "array": "",              # SBATCH --array range, e.g. 0-95 (blank -> auto 0-(N*rep-1) note)
    "modules": "",            # whitespace/comma-separated `module load` names (HPC)
    "conda_init": "",         # path/line to source (e.g. ~/bin/myconda or conda.sh)
    "conda_environment": "",  # env to activate (e.g. openfe)
}


def build_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge form overrides over the defaults (unknown keys are kept)."""
    conf = dict(DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                conf[key] = value
    return conf


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(v) for v in value) + "]"
    text = str(value)
    if text == "" or any(ch in text for ch in ' :#"\''):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def to_yaml(conf: dict[str, Any]) -> str:
    """Emit a flat YAML mapping for the create_json config."""
    return "".join(f"{key}: {_yaml_scalar(value)}\n" for key, value in conf.items())


def validate_setup(conf: dict[str, Any]) -> list[str]:
    """Return a list of problems with the setup (empty = OK).

    Enforces the membrane PBC rule: a membrane system needs an explicit
    ``complex_box_size`` of three positive numbers.
    """
    issues: list[str] = []
    if not conf.get("ligand_sdf"):
        issues.append("Ligand SDF is required.")
    if not conf.get("protein_pdb"):
        issues.append("Protein PDB is required.")
    if conf.get("membrane_system"):
        box = conf.get("complex_box_size")
        if not (isinstance(box, (list, tuple)) and len(box) == 3
                and all(isinstance(v, (int, float)) and v > 0 for v in box)):
            issues.append("Membrane system requires complex_box_size = [x, y, z] (nm, all > 0) "
                          "for a valid periodic box.")
    for key in ("n_lambda_windows_complex_leg", "n_lambda_windows_solvent_leg"):
        value = conf.get(key)
        if not isinstance(value, int) or value < 2:
            issues.append(f"{key} must be an integer >= 2.")
    if conf.get("transformation") not in ("both", "complex", "solvent"):
        issues.append("transformation must be both | complex | solvent.")
    return issues


def parse_box_size(text: str):
    """Parse 'x,y,z' (or 'x y z') into a float list; '' / None -> None."""
    if not text or not str(text).strip():
        return None
    parts = str(text).replace(",", " ").split()
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"box size must be three numbers, got {text!r}") from exc
    if len(values) != 3:
        raise ValueError(f"box size must be three numbers (x y z), got {text!r}")
    return values


def slurm_script(config_path: str | Path, hpc: dict[str, Any] | None = None,
                 conf: dict[str, Any] | None = None) -> str:
    """A SLURM array script that runs ``openfe quickrun`` per transformation.

    One array task per (transformation JSON × replicate): index = json_index *
    replicates + rep_index. ``json_dir`` / ``result_dir`` are inlined from the
    config (no ``yq`` dependency); ``input_json_files`` (if set) selects a subset.
    """
    hpc = {**HPC_DEFAULTS, **(hpc or {})}
    conf = conf or {}
    config_path = str(config_path)
    json_dir = conf.get("json_dir") or "json"
    result_dir = conf.get("result_dir") or conf.get("output_directory") or "results"
    input_json = conf.get("input_json_files") or ""
    n_rep = int(hpc.get("replicates", 3))
    gpus = int(hpc.get("gpus", 0))
    gpu_type = str(hpc.get("gpu_type", "")).strip()
    gres = (f"gpu:{gpu_type}:{gpus}" if gpu_type else f"gpu:{gpus}") if gpus > 0 else ""
    array = str(hpc.get("array", "")).strip() or "0-0"

    header = [
        "#!/bin/bash",
        "#SBATCH --job-name=rbfe",
        f"#SBATCH --partition={hpc['partition']}",
        f"#SBATCH --time={hpc['walltime']}",
    ]
    if gres:
        header.append(f"#SBATCH --gres={gres}")
    header += [
        f"#SBATCH --cpus-per-task={hpc['cpus_per_task']}",
        f"#SBATCH --mem={hpc['memory']}",
        f"#SBATCH --array={array}    # set to 0-(N_transformations*replicates - 1)",
    ]

    activate = []
    modules = str(hpc.get("modules", "")).replace(",", " ").split()
    if modules:
        activate.append("module purge")
        activate += [f"module load {m}" for m in modules]
    if hpc.get("conda_init"):
        activate.append(f"source {hpc['conda_init']}")
    if hpc.get("conda_environment"):
        activate.append(f"conda activate {hpc['conda_environment']}")

    body = f"""
JSON_DIR="{json_dir}"
OUT_ROOT="{result_dir}"
N_REP={n_rep}
INPUT_JSON="{input_json}"

mkdir -p "${{OUT_ROOT}}"
for ((i=0; i<N_REP; i++)); do mkdir -p "${{OUT_ROOT}}/replicate_${{i}}"; done

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export CUDA_VISIBLE_DEVICES="${{SLURM_LOCALID:-0}}"

# Collect the transformation JSONs: all in JSON_DIR, a .txt list, or one file.
if [[ -z "$INPUT_JSON" ]]; then
    mapfile -t JSONS < <(find "${{JSON_DIR}}" -maxdepth 1 -type f -name "*.json" | sort)
elif [[ "$INPUT_JSON" == *.txt ]]; then
    mapfile -t JSONS < "$INPUT_JSON"
else
    JSONS=("$INPUT_JSON")
fi
N_JSON="${{#JSONS[@]}}"
if [[ "${{N_JSON}}" -eq 0 ]]; then echo "No JSON files in ${{JSON_DIR}}"; exit 1; fi

TASK_ID="${{SLURM_ARRAY_TASK_ID}}"
JSON_INDEX=$(( TASK_ID / N_REP ))
REP_INDEX=$(( TASK_ID % N_REP ))
if [[ "${{JSON_INDEX}}" -ge "${{N_JSON}}" ]]; then
    echo "JSON_INDEX ${{JSON_INDEX}} out of range (N_JSON=${{N_JSON}}); check --array."; exit 1
fi

JSON_FILE="${{JSONS[$JSON_INDEX]}}"
BASE="$(basename "${{JSON_FILE}}" .json)"
WORKDIR="${{OUT_ROOT}}/replicate_${{REP_INDEX}}/${{BASE}}"
RESULT_JSON="${{OUT_ROOT}}/replicate_${{REP_INDEX}}/${{BASE}}.json"
mkdir -p "${{WORKDIR}}"
echo "quickrun ${{JSON_FILE}} -> ${{RESULT_JSON}}"
openfe quickrun "${{JSON_FILE}}" -d "${{WORKDIR}}" -o "${{RESULT_JSON}}"
"""
    return "\n".join(header + ["", *activate]) + "\n" + body
