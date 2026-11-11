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
    "workdir": ".",
    "jobname": "rbfe",
    "result_dir": "rbfe/results",
    "json_dir": "rbfe/json",
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
    "surface_tension": 0.0,               # bar*nm (membrane only; 0 => tensionless)
    # --- Mapper / planner ---
    "mapper": "KartografAtomMapper",      # LomapAtomMapper | KartografAtomMapper
    "planner": "minimal_redundant",       # minimal_spanning | minimal_redundant | radial
}

# Default HPC settings for the submit script.
HPC_DEFAULTS: dict[str, Any] = {
    "partition": "gpu",
    "account": "",            # SBATCH --account (empty -> omitted)
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 16,
    "gpus": 1,
    "gpu_type": "",           # e.g. v100x -> --gres=gpu:v100x:1 (empty -> gpu:1)
    "memory": "32G",
    "walltime": "240:00:00",
    "replicates": 3,          # repeats per transformation (array = N_json * replicates)
    "array": "",              # SBATCH --array range, e.g. 0-95 (blank -> auto 0-(N*rep-1) note)
    "mpi": False,             # multi-GPU: launch quickrun under MPI (auto-on when gpus>1)
    "mpi_launcher": "srun",   # srun (SLURM) or e.g. "mpirun -np N"
    "modules": "",            # whitespace/comma-separated `module load` names (HPC)
    "conda_init": "",         # path/line to source (e.g. ~/bin/myconda or conda.sh)
    "conda_environment": "",  # env to activate (e.g. openfe)
}


def hpc_defaults() -> dict[str, Any]:
    """HPC_DEFAULTS merged with the ``openfe.defaults`` block of env/hpc.yaml.

    Lets a cluster set OpenFE's allocated resources (partition, account, cpus,
    gpus, gpu_type, memory, walltime, replicates, nodes/ntasks, modules/conda) in
    one customizable file instead of editing code. Lookup order mirrors the
    AlchemFold HPC config: ``$CHEMFLOW_HPC_CONFIG`` -> ``<PROJECT_ROOT>/env/hpc.yaml``
    -> ``~/.config/chemflow/hpc.yaml``. Missing file/keys fall back to built-ins.
    """
    import os
    from pathlib import Path as _Path

    merged = dict(HPC_DEFAULTS)
    candidates = []
    env = os.environ.get("CHEMFLOW_HPC_CONFIG", "").strip()
    if env:
        candidates.append(_Path(env).expanduser())
    try:
        from chemflow.config import PROJECT_ROOT  # noqa: PLC0415

        candidates.append(_Path(PROJECT_ROOT) / "env" / "hpc.yaml")
    except Exception:
        pass
    candidates.append(_Path.home() / ".config" / "chemflow" / "hpc.yaml")
    for candidate in candidates:
        try:
            if candidate and candidate.is_file():
                import yaml  # noqa: PLC0415

                data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                block = (data.get("openfe") or {}).get("defaults") or {}
                for key, value in block.items():
                    if value is not None:
                        merged[str(key)] = value
                break
        except Exception:
            continue
    return merged


# Embed the HPC allocation as an inline `slurm:` block in the config (alchemfold
# screen.yaml style), seeded from the file-backed defaults.
DEFAULTS["slurm"] = dict(hpc_defaults())


def _read_hpc_file() -> dict:
    """Return the parsed env/hpc.yaml (lookup order like hpc_defaults), or {}."""
    import os
    from pathlib import Path as _Path

    candidates = []
    env = os.environ.get("CHEMFLOW_HPC_CONFIG", "").strip()
    if env:
        candidates.append(_Path(env).expanduser())
    try:
        from chemflow.config import PROJECT_ROOT  # noqa: PLC0415

        candidates.append(_Path(PROJECT_ROOT) / "env" / "hpc.yaml")
    except Exception:
        pass
    candidates.append(_Path.home() / ".config" / "chemflow" / "hpc.yaml")
    for candidate in candidates:
        try:
            if candidate and candidate.is_file():
                import yaml  # noqa: PLC0415

                return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
    return {}


def slurm_preset_names() -> list[str]:
    """Named presets available in env/hpc.yaml (e.g. singlenode-singlegpu-long)."""
    return list((_read_hpc_file().get("presets") or {}).keys())


def single_gpu_preset_names() -> list[str]:
    """Preset names that request at most one GPU (for OpenFE, which is single-GPU).

    Excludes multi-GPU presets (gpus/gpus_per_node > 1); those are only meaningful
    for GROMACS HREX, not for OpenFE's single-process ``quickrun``.
    """
    presets = _read_hpc_file().get("presets") or {}
    out = []
    for name, value in presets.items():
        gpus = 1
        if isinstance(value, dict):
            gpus = int(value.get("gpus_per_node", value.get("gpus", 1)) or 1)
        if gpus <= 1:
            out.append(name)
    return out


# env/hpc.yaml preset keys -> OpenFE inline `slurm:` block keys.
_PRESET_KEY_MAP = {
    "time": "walltime", "walltime": "walltime",
    "nodes": "nodes",
    "ntasks_per_node": "ntasks", "ntasks": "ntasks",
    "gpus_per_node": "gpus", "gpus": "gpus",
    "gpu_type": "gpu_type",
    "cpus_per_task": "cpus_per_task",
    "mem": "memory", "memory": "memory",
    "mpi": "mpi", "mpi_launcher": "mpi_launcher",
    "partition": "partition", "account": "account",
    "conda_environment": "conda_environment", "conda_env": "conda_environment",
}


def apply_slurm_preset(slurm: dict[str, Any], name: str) -> dict[str, Any]:
    """Overlay a named env/hpc.yaml preset onto an OpenFE `slurm:` block.

    A preset value may be a plain walltime string (single-node single-GPU) or a
    mapping of resources. Returns a new dict; raises ValueError for unknown names.
    """
    presets = _read_hpc_file().get("presets") or {}
    if name not in presets:
        raise ValueError(f"unknown HPC preset {name!r}; choose one of {sorted(presets)}")
    raw = presets[name]
    raw = {"time": raw} if not isinstance(raw, dict) else dict(raw)
    out = dict(slurm)
    for key, value in raw.items():
        mapped = _PRESET_KEY_MAP.get(key)
        if mapped and value is not None:
            out[mapped] = value
    return out


HPC_JOB_HELP_HTML = """
<h3>Job settings (HPC)</h3>
<p>These fields build the SLURM submit script and are also written as an inline
<code>slurm:</code> block in the config, so the script and config stay in sync.</p>

<h4>HPC preset</h4>
<p>Pick a named preset from <code>env/hpc.yaml</code> (e.g.
<b>singlenode-singlegpu-long</b>) to fill the fields below in one click. Presets are
single node / single GPU and differ mainly in walltime (quick=4&nbsp;h, day=1&nbsp;d,
long=4&nbsp;d, tendays=10&nbsp;d). Choose <b>(custom)</b> to keep your own values.</p>

<h4>CPUs per task &amp; HREX</h4>
<p>On the <b>CUDA</b> platform dynamics run on the GPU, so more CPUs do <u>not</u>
speed up the replica-exchange (HREX) lambda windows &mdash; they run on one GPU,
serially per iteration. 4&ndash;8 is plenty. To parallelize windows you need
multiple GPUs + MPI, not more CPUs.</p>

<h4>GPUs / GPU type</h4>
<p><b>GPUs=1</b> with a <b>GPU type</b> (e.g. a100) requests
<code>--gres=gpu:a100:1</code>; blank type = any GPU. GPUs=0 runs on CPU and is
impractically slow for production.</p>

<h4>Multi-GPU</h4>
<p><b>OpenFE's <code>quickrun</code> uses ONE GPU</b> — it runs the whole edge in a
single process and is not MPI-parallel, so one edge's lambda windows cannot be
split across GPUs (wrapping quickrun in srun just launches duplicate copies of the
edge). The script always requests a single GPU per task. Parallelize across
<i>edges and replicates</i> with the SLURM array instead — that's where throughput
comes from. (True within-edge multi-GPU exists only for GROMACS HREX via
<code>-multidir</code>.)</p>

<h4>Replicates &amp; Array</h4>
<p>Each edge runs <b>Replicates</b> times; the SLURM array is auto-sized to
<i>transformations &times; replicates</i> when the script is written after
<b>Setup FEP (create_json)</b>. Leave <b>Array</b> blank to auto-size.</p>

<h4>Walltime</h4>
<p>Use <code>D-HH:MM:SS</code> or <code>HH:MM:SS</code>. RBFE edges often need days.</p>

<h4>Platform = CUDA</h4>
<p>Keep <b>platform: CUDA</b>; it is honored literally and the real GPU is chosen at
run time on the compute node. JSONs generated before this fix carry a baked CPU
platform &mdash; <b>regenerate</b> them (Setup FEP).</p>

<h4>Modules / conda</h4>
<p>Optional pre-job lines: <code>module load</code> names, a conda init script to
<code>source</code>, and the env to activate (usually <code>alchemforge</code>).</p>
"""


def format_slurm_block(slurm: dict[str, Any]) -> str:
    """Render an inline ``slurm:`` YAML block from a resources dict."""
    lines = ["slurm:"]
    for key, value in slurm.items():
        lines.append(f"  {key}: {_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


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
    """Emit a YAML mapping for the create_json config (supports a nested dict block)."""
    lines = []
    for key, value in conf.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {_yaml_scalar(sub_value)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _box_is_valid(value: Any) -> bool:
    """A membrane box is valid if it is 3/6/9 numbers (list or comma/space string),
    or a path to a file (e.g. pbc.txt) whose contents are parsed at run time."""
    if value is None:
        return False
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return False
        if s.lower().endswith(".txt") or Path(s).is_file():
            return True
        parts = [p for p in s.replace(",", " ").split() if p]
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return False
        return len(nums) in (3, 6, 9)
    if isinstance(value, (list, tuple)):
        if len(value) not in (3, 6, 9):
            return False
        try:
            nums = [float(v) for v in value]
        except (TypeError, ValueError):
            return False
        return all(v > 0 for v in nums) if len(nums) == 3 else True
    return False


def validate_setup(conf: dict[str, Any]) -> list[str]:
    """Return a list of problems with the setup (empty = OK).

    Enforces the membrane PBC rule: a membrane system needs a periodic box via
    ``complex_box_size``/``complex_box_vectors`` — 3 ``[x,y,z]``, 6
    ``[a,b,c,alpha,beta,gamma]``, or 9 numbers, or a ``.txt`` file.
    """
    issues: list[str] = []
    if not conf.get("ligand_sdf"):
        issues.append("Ligand SDF is required.")
    if not conf.get("protein_pdb"):
        issues.append("Protein PDB is required.")
    if conf.get("membrane_system"):
        if not (_box_is_valid(conf.get("complex_box_size"))
                or _box_is_valid(conf.get("complex_box_vectors"))):
            issues.append(
                "Membrane system requires a periodic box: complex_box_size / "
                "complex_box_vectors = 3 [x,y,z], 6 [a,b,c,alpha,beta,gamma], or 9 "
                "numbers (nm/deg), or a .txt file (comma-separated)."
            )
    for key in ("n_lambda_windows_complex_leg", "n_lambda_windows_solvent_leg"):
        value = conf.get(key)
        if not isinstance(value, int) or value < 2:
            issues.append(f"{key} must be an integer >= 2.")
    if conf.get("transformation") not in ("both", "complex", "solvent"):
        issues.append("transformation must be both | complex | solvent.")
    return issues


def parse_box_size(text: str):
    """Parse the complex-box entry into an explicit float list (or None).

    '' / None -> None. A path to a ``.txt`` file (or any existing file) is **read
    and parsed** so the config stores the exact numbers rather than a path.
    Accepts 3 ([x,y,z]), 6 ([a,b,c,alpha,beta,gamma]) or 9 (3x3) numbers,
    separated by commas, spaces or newlines.
    """
    if not text or not str(text).strip():
        return None
    s = str(text).strip()
    path = Path(s)
    if s.lower().endswith(".txt") or path.is_file():
        if not path.is_file():
            raise ValueError(f"PBC file not found: {s}")
        s = path.read_text(encoding="utf-8")
    parts = [p for p in s.replace(";", " ").replace(",", " ").split() if p.strip()]
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"box size must be numbers (3/6/9) or a .txt file of them, got {text!r}"
        ) from exc
    if len(values) not in (3, 6, 9):
        raise ValueError(
            f"box size must be 3 [x,y,z], 6 [a,b,c,alpha,beta,gamma] or 9 numbers; "
            f"got {len(values)}"
        )
    return values


def slurm_script(config_path: str | Path, hpc: dict[str, Any] | None = None,
                 conf: dict[str, Any] | None = None) -> str:
    """A SLURM array script that runs ``openfe quickrun`` per transformation.

    One array task per (transformation JSON × replicate): index = json_index *
    replicates + rep_index. ``json_dir`` / ``result_dir`` are inlined from the
    config (no ``yq`` dependency); ``input_json_files`` (if set) selects a subset.
    """
    conf = conf or {}
    # Resource precedence: built-in defaults (+ optional env/hpc.yaml openfe.defaults)
    # < the inline `slurm:` block in the config < an explicit hpc override.
    hpc = {**hpc_defaults(), **(conf.get("slurm") or {}), **(hpc or {})}
    config_path = str(config_path)
    json_dir = conf.get("json_dir") or "json"
    result_dir = conf.get("result_dir") or conf.get("output_directory") or "results"
    input_json = conf.get("input_json_files") or ""
    n_rep = int(hpc.get("replicates", 3))
    gpus = int(hpc.get("gpus", 0))
    gpu_type = str(hpc.get("gpu_type", "")).strip()
    # OpenFE `quickrun` runs the whole DAG in ONE process on ONE GPU — it is NOT
    # MPI-parallel, so we never wrap it in srun/mpirun (that would just launch N
    # redundant copies of the same edge). Parallelism is across edges/replicates
    # via the SLURM array below. Request a single GPU per task regardless of any
    # multi-GPU preset.
    gpus = 1 if gpus > 1 else gpus
    gres = (f"gpu:{gpu_type}:{gpus}" if gpu_type else f"gpu:{gpus}") if gpus > 0 else ""
    array = str(hpc.get("array", "")).strip()
    if not array:
        # Auto-size: one array task per (transformation JSON x replicate). Count the
        # JSONs beside the config if create_json already produced them; otherwise fall
        # back to a single task (0-0) and flag it in the directive comment + runtime
        # guard below, so a script written before create_json can't silently run only
        # one edge.
        base_dir = Path(config_path).expanduser().resolve().parent
        jdir = Path(json_dir)
        if not jdir.is_absolute():
            jdir = base_dir / jdir
        n_json = len(list(jdir.glob("*.json"))) if jdir.is_dir() else 0
        array = f"0-{n_json * n_rep - 1}" if n_json > 0 else "0-0"

    header = [
        "#!/bin/bash",
        "#SBATCH --job-name=rbfe",
        f"#SBATCH --partition={hpc['partition']}",
        f"#SBATCH --time={hpc['walltime']}",
    ]
    if str(hpc.get("account", "")).strip():
        header.append(f"#SBATCH --account={str(hpc['account']).strip()}")
    if int(hpc.get("nodes", 0) or 0) > 0:
        header.append(f"#SBATCH --nodes={int(hpc['nodes'])}")
    header.append("#SBATCH --ntasks=1")
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

echo "=== GPU check (before quickrun) ==="
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}  SLURM_JOB_GPUS=${{SLURM_JOB_GPUS:-unset}}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi: not available on this node"
python -c "from openmm import Platform; ns=[Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]; print('OpenMM platforms:', ns); print('CUDA usable:', 'CUDA' in ns)" 2>/dev/null || echo "OpenMM CUDA probe failed"
echo "==================================="

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

# Guard against an undersized --array (e.g. the script was written before
# create_json, so the directive stayed 0-0). One task per (JSON x replicate).
NEEDED=$(( N_JSON * N_REP ))
if [[ -n "${{SLURM_ARRAY_TASK_COUNT:-}}" && "${{SLURM_ARRAY_TASK_COUNT}}" -lt "${{NEEDED}}" ]]; then
    echo "WARNING: --array covers ${{SLURM_ARRAY_TASK_COUNT}} task(s) but ${{NEEDED}} are needed" \
         "(N_JSON=${{N_JSON}} x N_REP=${{N_REP}}). Re-submit with --array=0-$(( NEEDED - 1 ))." >&2
fi

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
