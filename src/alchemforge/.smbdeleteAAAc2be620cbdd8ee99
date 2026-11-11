"""Scaffold starter configuration + driver scripts for the FEP workflows.

    alchemforge generate rbfe      [name] [--engine gromacs|openfe]
    alchemforge generate abfe      [name] [--engine ...]
    alchemforge generate septop    [name] [--engine ...]
    alchemforge generate res_rbfe  [name] [--engine ...]

Each command writes a commented ``<name>.toml`` configuration and a matching
``<name>.sh`` driver that validates, stages, and points at the submit step for
that workflow. ``name`` defaults to the workflow key and is used as the project
name, the output file basename, and the Slurm job name.

The ``--engine`` flag selects the simulation engine (default ``gromacs``). The
``openfe`` engine is reserved for a future set of templates; requesting a
workflow it does not yet provide raises a clear error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# --------------------------------------------------------------------------- #
# TOML templates (sentinels __PROJECT__ substituted at write time)
# --------------------------------------------------------------------------- #
_RBFE_TOML = """\
# AlchemForge ligand-RBFE configuration
# Edit the paths/values, then:  alchemforge ligand-rbfe validate __PROJECT__.toml

[project]
name = "__PROJECT__"               # project dir = work_dir/name
work_dir = "work"

[system]
protein = "protein.pdb"            # receptor PDB (or use prepared_complex_* below)
force_field = "amber99sb-star-ildn-mut"
water_model = "tip3p"
gromacs_command = "gmx"
# prepared_complex_gro = "complex.gro"        # pre-built GROMACS system (skip pdb prep)
# prepared_complex_topology = "topol.top"
box_shape = "dodecahedron"
box_distance = 1.2
neutralize = true
ion_concentration = 0.15

[ligands]
library_sdf = "ligands.sdf"        # one SDF; every record needs a unique _Name
# charge_manifest = "charges.csv"            # optional per-ligand net charges

[mapping]
network_method = "minimal_spanning"   # lomap | minimal_spanning | minimal_redundant | radial
atom_mapper = "lomap"                  # lomap | kartograf
element_change = false
max3d_angstrom = 10.0
require_cycle_covering = true
# radial_hub = "ligand_name"            # required when network_method = "radial"

[topology]
backend = "pmx"
charge_method = "bcc"                   # bcc | user
charge_change_method = "reject"         # reject | correction
# charge_correction_file = "coions.csv" # required when charge_change_method = "correction"

[simulation]
replicates = 3
lambda_windows = 21
timestep_ps = 0.002
equilibration_ns = 1.0
production_ns = 5.0
sampling_method = "independent"         # independent | hrex

[execution]
mdrun_options = ["-ntmpi", "1"]
# hrex_module = "gromacs/2026.3-gcc13.2"      # only when sampling_method = "hrex"
# hrex_mpi_launcher = "mpirun"
# hrex_gromacs_command = "gmx_mpi"

[slurm]
partition = "gpu"
nodes = 1
cpus_per_task = 8
gpus = 1
memory = "16G"
walltime = "24:00:00"
# conda_init = "/path/to/conda/etc/profile.d/conda.sh"
# conda_environment = "alchemforge"
"""

_ABFE_TOML = """\
# AlchemForge ABFE (absolute binding free energy) configuration
# Edit, then:  alchemforge abfe-gromacs validate __PROJECT__.toml

[project]
name = "__PROJECT__"
work_dir = "work"

[system]
protein = "protein.pdb"
force_field = "amber99sb-ildn"
water_model = "tip3p"

[ligand]
sdf = "ligand.sdf"                 # a single ligand
resname = "LIG"
charge = 0

[topology]
backend = "pmx"
charge_method = "bcc"

[restraints]
method = "boresch"                 # Boresch orientational restraints
seed = 17

[simulation]
replicas = 2
lambda_windows = 7
lambda_function = "cosine"         # linear | cosine | power
lambda_power = 2.0
timestep_ps = 0.002
equilibration_ns = 1.0
production_ns = 5.0

[execution]
mode = "local"                     # local | slurm
max_parallel_jobs = 4
mdrun_options = ["-ntmpi", "1"]
# gpus = 1
# memory = "16G"
"""

_SEPTOP_TOML = """\
# AlchemForge SepTop (separated-topologies RBFE) configuration
# Edit, then:  alchemforge septop-gromacs validate __PROJECT__.toml

[project]
name = "__PROJECT__"
work_dir = "work"

[inputs]
ligand_library = "ligands.sdf"
prepared_root = "prepared"
septop_source_dir = "SeparatedTopologies"

# [system]
# type = "membrane"                # soluble (default) | membrane

[network]
method = "minimal_spanning"
atom_mapper = "lomap"
max3d_angstrom = 10.0
element_change = false
require_cycle_covering = true

[protocol]
replicates = 2
timestep_ps = 0.002
equilibration_ns = 1.0
production_ns = 5.0

[execution]
max_parallel_jobs = 8
mdrun_options = ["-ntmpi", "1"]

[slurm]
gpus = 1
# nodes = 1
# cpus_per_task = 4
# memory = "16G"
"""

_RESFEP_TOML = """\
# AlchemForge resFEP (protein residue-mutation FEP) configuration
# Edit, then:  alchemforge resfep validate __PROJECT__.toml

[project]
name = "__PROJECT__"
work_dir = "work"

[system]
complex = "complex.pdb"
force_field = "amber99sb"
water_model = "tip3p"

# One [[mutation]] block per residue; pmx applies them all together.
[[mutation]]
chain = "A"
residue = 42
target = "VAL"                     # three-letter residue code

# [[mutation]]
# chain = "A"
# residue = 55
# target = "ALA"

[ligand]
mol2 = "ligand.mol2"
resname = "LIG"
charge = 0

[simulation]
replicates = 2
lambda_windows = 17
mode = "RESUME"                    # OVERWRITE | RESUME | SKIP

[execution]
max_parallel_jobs = 8
mdrun_options = ["-ntmpi", "1"]
"""


_DNA_RBFE_TOML = """\
# AlchemForge dnaRBFE (DNA base-mutation RBFE) configuration
# Edit, then:  alchemforge dna-rbfe validate __PROJECT__.toml

[project]
name = "__PROJECT__"
work_dir = "work"

[system]
molecule_type = "dna"              # overridden by the dna-rbfe command anyway
complex = "complex.pdb"
force_field = "amber99sb"          # use a pmx force field that supports DNA mutations
water_model = "tip3p"

# One [[mutation]] block per base; pmx applies them all together.
# DNA targets: DA | DT | DG | DC (DU)
[[mutation]]
chain = "B"
residue = 7
target = "DG"

# [[mutation]]
# chain = "B"
# residue = 12
# target = "DA"

[ligand]
mol2 = "ligand.mol2"
resname = "LIG"
charge = 0

[simulation]
replicates = 2
lambda_windows = 17
mode = "RESUME"                    # OVERWRITE | RESUME | SKIP

[execution]
max_parallel_jobs = 8
mdrun_options = ["-ntmpi", "1"]
"""


_RNA_RBFE_TOML = """\
# AlchemForge rnaRBFE (RNA base-mutation RBFE) configuration
# Edit, then:  alchemforge rna-rbfe validate __PROJECT__.toml

[project]
name = "__PROJECT__"
work_dir = "work"

[system]
molecule_type = "rna"              # overridden by the rna-rbfe command anyway
complex = "complex.pdb"
force_field = "amber99sb"          # use a pmx force field that supports RNA mutations
water_model = "tip3p"

# One [[mutation]] block per base; pmx applies them all together.
# RNA targets: RA | RU | RG | RC
[[mutation]]
chain = "B"
residue = 7
target = "RG"

# [[mutation]]
# chain = "B"
# residue = 12
# target = "RA"

[ligand]
mol2 = "ligand.mol2"
resname = "LIG"
charge = 0

[simulation]
replicates = 2
lambda_windows = 17
mode = "RESUME"                    # OVERWRITE | RESUME | SKIP

[execution]
max_parallel_jobs = 8
mdrun_options = ["-ntmpi", "1"]
"""


# --------------------------------------------------------------------------- #
# OpenFE native YAML templates (consumed directly by create_json)
# --------------------------------------------------------------------------- #
_SEPTOP_OPENFE_YAML = """\
# OpenFE SepTop configuration (consumed by create_json). Fill in the paths.

# --- Input files ---
ligand_sdf: ligands.sdf
protein_pdb: protein.pdb
cofactor_sdf: null
restart_json: null

# Cofactor-from-template (optional; avoids the isomorphic-charge error for repeated
# cofactor copies): define the cofactor ONCE as a charged template SDF (with H) and
# place it on every matching residue in protein_pdb — all copies then share identical
# charges. When set, cofactor_sdf is ignored and the listed residues are stripped
# from protein_pdb before protein parameterization.
# cofactor_template_sdf: cofactor_template.sdf
# cofactor_resnames: [4DY]

# --- Output ---
workdir: .                         # base dir; the project lives at <workdir>/<jobname>/
jobname: __PROJECT__               # project folder name (holds inputs/, json/, results/, this config)
result_dir: __PROJECT__/results
json_dir: __PROJECT__/json

# --- Platform ---
platform: CUDA              # CPU | CUDA | OpenCL

# --- Membrane or not ---
membrane_system: false

# --- General ---
protocol_repeats: 1
pressure: 1.0              # bar
temperature: 298.15        # kelvin

# --- Force field / nonbonded ---
nonbonded_cutoff: 1.0      # nanometer
nonbonded_method: PME
rigid_water: true
small_molecule_forcefield: openff-2.1.0
hydrogen_mass: 3.0

# --- Partial charges ---
off_toolkit_backend: ambertools   # openeye | rdkit | ambertools
partial_charge_method: am1bcc     # am1bcc | nagl
skip_charge_assignment: false     # true = use charges already in the SDFs (skip the slow charge check/gen)

# --- Alchemical settings ---
disable_alchemical_dispersion_correction: false
annihilate_sterics: false
softcore_a: 1.0
softcore_b: 1.0
softcore_c: 6.0
softcore_alpha: 0.5

# --- Restraints ---
anchor_finding_strategy: bonded   # bonded | multi-residue
host_max_distance: 2.0     # nanometer
host_min_distance: 0.3     # nanometer
host_selection: backbone   # backbone | all
rmsf_cutoff: 0.1           # nanometer
solvent_restraint_spring_constant: 1000.0   # kJ/mol/nm^2

# --- Equilibration settings ---
complex_equil_mini_steps: 5000
complex_equil_equil_length: 0.1
complex_equil_equil_length_nvt: 0.1
complex_equil_production_length: 2.0
solvent_equil_mini_steps: 5000
solvent_equil_equil_length: 0.1
solvent_equil_equil_length_nvt: 0.1
solvent_equil_production_length: 2.0

# --- Solvation ---
complex_solvation_padding: 1.5    # nm (non-membrane)
solvent_solvation_padding: 1.5    # nm
complex_box_size: null            # [x,y,z] nm when membrane_system: true
solvent_model: tip3p

# --- Simulation settings ---
early_termination_target_error: 0.0   # kcal/mol
complex_sim_mini_step: 5000
complex_sim_equil_length: 1.0
complex_sim_prod_length: 20.0
solvent_sim_mini_step: 5000
solvent_sim_equil_length: 5.0
solvent_sim_prod_length: 30.0

# --- Lambda schedules (all lists in a leg must share one length = n_replicas) ---
complex_lambda_elec_A: [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
complex_lambda_elec_B: [1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]
complex_lambda_vdw_A: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0]
complex_lambda_vdw_B: [1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
complex_lambda_restraints_A: [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
complex_lambda_restraints_B: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0]
solvent_lambda_elec_A: [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
solvent_lambda_elec_B: [1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0]
solvent_lambda_vdw_A: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.65, 1.0]
solvent_lambda_vdw_B: [1.0, 0.7, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
solvent_lambda_restraints_A: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
solvent_lambda_restraints_B: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# --- Integrator ---
complex_timestep: 4.0   # fs
solvent_timestep: 4.0   # fs
barostat_frequency: 25.0
surface_tension: 0.0

# --- Mapper / planner ---
mapper: KartografAtomMapper   # LomapAtomMapper | KartografAtomMapper
planner: minimal_spanning     # minimal_spanning | minimal_redundant | radial

# --- HPC resources (inline; used to write the SLURM array submit script) ---
slurm:
  partition: gpu
  account: ""                 # SBATCH --account (blank -> omitted)
  nodes: 1
  ntasks: 1
  cpus_per_task: 16
  gpus: 1
  gpu_type: ""                # e.g. a100 -> --gres=gpu:a100:1 (blank -> gpu:1)
  memory: 32G
  walltime: "240:00:00"
  replicates: 3               # repeats per transformation (array = N_json * replicates)
  array: ""                   # blank -> auto 0-(N_json*replicates - 1)
  mpi: false                  # multi-GPU: parallelize lambda windows across GPUs (auto-on if gpus>1)
  mpi_launcher: srun          # srun (SLURM) or e.g. "mpirun -np N"
  modules: ""                 # e.g. "cuda/12.1 gcc/11"
  conda_init: ""              # line to source, e.g. ~/miniconda3/etc/profile.d/conda.sh
  conda_environment: alchemforge
"""


_RBFE_OPENFE_YAML = """\
# OpenFE RBFE configuration (consumed by create_json). Fill in the paths.

# --- Input files ---
ligand_sdf: ligands.sdf
protein_pdb: protein.pdb
cofactor_sdf: null

# Cofactor-from-template (optional; avoids the isomorphic-charge error for repeated
# cofactor copies): define the cofactor ONCE as a charged template SDF (with H) and
# place it on every matching residue in protein_pdb — all copies then share identical
# charges. When set, cofactor_sdf is ignored and the listed residues are stripped
# from protein_pdb before protein parameterization.
# cofactor_template_sdf: cofactor_template.sdf
# cofactor_resnames: [4DY]

# --- Output ---
workdir: .                         # base dir; the project lives at <workdir>/<jobname>/
jobname: __PROJECT__               # project folder name (holds inputs/, json/, results/, this config)
result_dir: __PROJECT__/results
json_dir: __PROJECT__/json

# Which transformation JSONs to execute (used by `run`):
#   null      -> every JSON in json_dir
#   a .txt    -> the JSON list in that file
#   a list    -> only those JSON files
input_json_files: null
restart: false

# --- Alchemical network ---
alchemical_network: all    # "all" or a list of ligand names, e.g. [ligA, ligB]
transformation: both       # both | complex | solvent

# --- Platform ---
platform: CUDA             # CPU | CUDA | OpenCL

# --- Membrane or not ---
membrane_system: false
complex_box_size: null     # [x, y, z] nm when membrane_system: true

# --- General ---
solvent_protocol_repeats: 1
complex_protocol_repeats: 1
pressure: 1.0              # bar
temperature: 310.0         # kelvin

# --- Force field / nonbonded ---
ff_constraints: hbonds
nonbonded_cutoff: 1.0      # nanometer
nonbonded_method: PME
rigid_water: true
small_molecule_forcefield: openff-2.1.0
hydrogen_mass: 3.0

# --- Partial charges ---
off_toolkit_backend: ambertools   # openeye | rdkit | ambertools
partial_charge_method: am1bcc     # am1bcc | nagl
skip_charge_assignment: false     # true = use charges already in the SDFs (skip the slow charge check/gen)

# --- Alchemical settings ---
endstate_dispersion_correction: false
explicit_charge_correction: true
explicit_charge_correction_cutoff: 0.8   # nanometer
softcore_LJ: gapsys        # gapsys | beutler
softcore_alpha: 0.85
turn_off_core_unique_exceptions: false
use_dispersion_correction: true
n_lambda_windows_solvent_leg: 30
n_lambda_windows_complex_leg: 30

# --- Solvation ---
complex_solvation_padding: 1.5    # nanometer (non-membrane)
solvent_solvation_padding: 1.5    # nanometer
solvent_model: tip3p

# --- Simulation settings ---
early_termination_target_error: 0.0   # kcal/mol
complex_sim_mini_step: 5000
complex_sim_equil_length: 5.0         # nanosecond
complex_sim_prod_length: 30.0         # nanosecond
solvent_sim_mini_step: 5000
solvent_sim_equil_length: 2.0         # nanosecond
solvent_sim_prod_length: 30.0         # nanosecond

# --- Integrator ---
complex_timestep: 4.0      # femtosecond
solvent_timestep: 4.0      # femtosecond
barostat_frequency: 25.0
surface_tension: 300.0     # bar*nm (membrane only)

# --- Mapper / planner ---
mapper: KartografAtomMapper   # LomapAtomMapper | KartografAtomMapper
planner: minimal_redundant    # minimal_spanning | minimal_redundant | radial

# --- HPC resources (inline; used to write the SLURM array submit script) ---
# Edit these to change the allocation; `generate`/desktop read this block to build
# the .slurm script (one array task per transformation JSON x replicate).
slurm:
  partition: gpu
  account: ""                 # SBATCH --account (blank -> omitted)
  nodes: 1
  ntasks: 1
  cpus_per_task: 16
  gpus: 1
  gpu_type: ""                # e.g. a100 -> --gres=gpu:a100:1 (blank -> gpu:1)
  memory: 32G
  walltime: "240:00:00"
  replicates: 3               # repeats per transformation (array = N_json * replicates)
  array: ""                   # blank -> auto 0-(N_json*replicates - 1)
  mpi: false                  # multi-GPU: parallelize lambda windows across GPUs (auto-on if gpus>1)
  mpi_launcher: srun          # srun (SLURM) or e.g. "mpirun -np N"
  modules: ""                 # e.g. "cuda/12.1 gcc/11"
  conda_init: ""              # line to source, e.g. ~/miniconda3/etc/profile.d/conda.sh
  conda_environment: alchemforge
"""


_ABFE_OPENFE_YAML = """\
# OpenFE ABFE configuration (consumed by create_json). Fill in the paths.
# ABFE decouples ONE ligand, so ligand_sdf must contain exactly one molecule.

# --- Input files ---
ligand_sdf: ligand.sdf
protein_pdb: protein.pdb
cofactor_sdf: null
restart_json: null
name: abfe_my_ligand        # transformation name (output JSON basename)

# Cofactor-from-template (optional; avoids the isomorphic-charge error for repeated
# cofactor copies): define the cofactor ONCE as a charged template SDF (with H) and
# place it on every matching residue in protein_pdb — all copies then share identical
# charges. When set, cofactor_sdf is ignored and the listed residues are stripped
# from protein_pdb before protein parameterization.
# cofactor_template_sdf: cofactor_template.sdf
# cofactor_resnames: [4DY]

# --- Output ---
workdir: .                         # base dir; the project lives at <workdir>/<jobname>/
jobname: __PROJECT__               # project folder name (holds inputs/, json/, results/, this config)
result_dir: __PROJECT__/results
json_dir: __PROJECT__/json

# --- Platform ---
platform: CUDA             # CPU | CUDA | OpenCL

# --- Membrane or not ---
membrane_system: false
complex_box_size: null     # [x, y, z] nm when membrane_system: true

# --- General ---
protocol_repeats: 1
pressure: 1.0              # bar
temperature: 310.0         # kelvin

# --- Force field / nonbonded ---
nonbonded_cutoff: 1.0      # nanometer
nonbonded_method: PME
rigid_water: true
small_molecule_forcefield: openff-2.1.0
hydrogen_mass: 3.0

# --- Partial charges ---
off_toolkit_backend: ambertools   # openeye | rdkit | ambertools
partial_charge_method: am1bcc     # am1bcc | nagl
skip_charge_assignment: false     # true = use charges already in the SDFs (skip the slow charge check/gen)

# --- Alchemical settings ---
disable_alchemical_dispersion_correction: false
annihilate_sterics: false
softcore_a: 1.0
softcore_b: 1.0
softcore_c: 6.0
softcore_alpha: 0.5

# --- Restraints ---
anchor_finding_strategy: bonded   # bonded | multi-residue
host_max_distance: 2.0     # nanometer
host_min_distance: 0.3     # nanometer
host_selection: backbone   # backbone | all
rmsf_cutoff: 0.1           # nanometer
solvent_restraint_spring_constant: 1000.0   # kJ/mol/nm^2

# --- Equilibration settings ---
complex_equil_mini_steps: 5000
complex_equil_equil_length: 1.0        # nanosecond
complex_equil_equil_length_nvt: 1.0    # nanosecond
complex_equil_production_length: 2.0   # nanosecond (NPT production equilibration)
solvent_equil_mini_steps: 5000
solvent_equil_equil_length: 1.0        # nanosecond
solvent_equil_equil_length_nvt: 1.0    # nanosecond
solvent_equil_production_length: 2.0   # nanosecond (NPT production equilibration)

# --- Solvation ---
complex_solvation_padding: 1.5    # nanometer (non-membrane)
solvent_solvation_padding: 1.5    # nanometer
solvent_model: tip3p

# --- Simulation settings ---
early_termination_target_error: 0.0   # kcal/mol
complex_sim_mini_step: 5000
complex_sim_equil_length: 1.0         # nanosecond
complex_sim_prod_length: 10.0         # nanosecond
solvent_sim_mini_step: 5000
solvent_sim_equil_length: 1.0         # nanosecond
solvent_sim_prod_length: 10.0         # nanosecond

# --- Lambda schedules (each leg's elec/vdw/restraints lists must share one length) ---
complex_lambda_elec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
complex_lambda_restraints: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
complex_lambda_vdw: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
solvent_lambda_elec: [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
solvent_lambda_restraints: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
solvent_lambda_vdw: [0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.24, 0.36, 0.48, 0.6, 0.7, 0.77, 0.85, 1.0]

# --- Integrator ---
complex_timestep: 4.0      # femtosecond
solvent_timestep: 4.0      # femtosecond
barostat_frequency: 25.0
surface_tension: 300.0     # bar*nm (membrane only)

# --- HPC resources (inline; used to write the SLURM array submit script) ---
slurm:
  partition: gpu
  account: ""                 # SBATCH --account (blank -> omitted)
  nodes: 1
  ntasks: 1
  cpus_per_task: 16
  gpus: 1
  gpu_type: ""                # e.g. a100 -> --gres=gpu:a100:1 (blank -> gpu:1)
  memory: 32G
  walltime: "240:00:00"
  replicates: 3               # repeats per transformation (array = N_json * replicates)
  array: ""                   # blank -> auto 0-(N_json*replicates - 1)
  mpi: false                  # multi-GPU: parallelize lambda windows across GPUs (auto-on if gpus>1)
  mpi_launcher: srun          # srun (SLURM) or e.g. "mpirun -np N"
  modules: ""                 # e.g. "cuda/12.1 gcc/11"
  conda_init: ""              # line to source, e.g. ~/miniconda3/etc/profile.d/conda.sh
  conda_environment: alchemforge
"""


# --------------------------------------------------------------------------- #
# Driver (.sh) template
# --------------------------------------------------------------------------- #
def _gromacs_driver(engine: str, command: str, submit_hint: str, ext: str) -> str:
    return f"""\
#!/usr/bin/env bash
# AlchemForge driver for __PROJECT__  (engine: {engine}, command: {command}).
# Generated by `alchemforge generate`. Edit the environment setup for your host.
set -euo pipefail

CONFIG="${{1:-__PROJECT__.{ext}}}"

# --- environment (uncomment/edit as needed) ---
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate alchemforge
# module load gromacs

echo "== validate =="
alchemforge {command} validate "$CONFIG"

echo "== stage =="
alchemforge {command} stage "$CONFIG"

echo
echo "Staging complete. To launch the simulations:"
echo "  {submit_hint}"
"""


def _openfe_driver(engine: str, command: str, submit_hint: str, ext: str) -> str:
    return f"""\
#!/usr/bin/env bash
# AlchemForge driver for __PROJECT__  (engine: {engine}, workflow: {command}).
# Generated by `alchemforge generate`. Edit the environment setup for your host.
set -euo pipefail

CONFIG="${{1:-__PROJECT__.{ext}}}"

# --- environment (uncomment/edit as needed) ---
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate alchemforge

echo "== validate =="
chemflow openfe {command} validate "$CONFIG"

echo "== generate + execute =="
chemflow openfe {command} run "$CONFIG"
"""


def _alanine_driver(engine: str, command: str, submit_hint: str, ext: str) -> str:
    return f"""\
#!/usr/bin/env bash
# AlchemForge driver for __PROJECT__  (engine: {engine}, {command}).
# Generated by `alchemforge generate`. Edit the environment setup for your host.
set -euo pipefail

CONFIG="${{1:-__PROJECT__.{ext}}}"

# --- environment (uncomment/edit as needed) ---
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate alchemforge

echo "== stage (expand selection -> one resFEP run per residue) =="
alchemforge {command} stage "$CONFIG"

echo
echo "Staging complete. To launch every per-residue run:"
echo "  {submit_hint}"
"""


_ALASCAN_TOML = """\
# AlchemForge alanine scanning (per-residue -> ALA resFEP) configuration
# Edit, then:  alchemforge alanine-scan stage __PROJECT__.toml

[project]
name = "__PROJECT__"
work_dir = "work"

[system]
complex = "complex.pdb"
force_field = "amber99sb"
water_model = "tip3p"

[ligand]
mol2 = "ligand.mol2"
resname = "LIG"
charge = 0

[simulation]
replicates = 2
lambda_windows = 17
mode = "RESUME"                    # OVERWRITE | RESUME | SKIP

[execution]
max_parallel_jobs = 8
mdrun_options = ["-ntmpi", "1"]

# Residues to mutate to ALA. Combine any of these; ALA/GLY/PRO are skipped.
[scan]
chain = "A"
residues = [40, 41, 42]           # explicit residue numbers (optional)
residue_range = "45-55"           # inclusive range (optional)
interface_cutoff_nm = 0.5         # also add residues within this cutoff of the ligand (optional)
"""


# engine -> workflow key -> (template text, extension, command, driver fn, submit hint)
_GROMACS: dict[str, tuple[str, str, str, object, str]] = {
    "rbfe": (
        _RBFE_TOML,
        "toml",
        "ligand-rbfe",
        _gromacs_driver,
        "bash work/__PROJECT__/submit_production_array.sh",
    ),
    "abfe": (
        _ABFE_TOML,
        "toml",
        "abfe-gromacs",
        _gromacs_driver,
        "alchemforge abfe-gromacs run __PROJECT__.toml",
    ),
    "septop": (
        _SEPTOP_TOML,
        "toml",
        "septop-gromacs",
        _gromacs_driver,
        "alchemforge septop-gromacs submit __PROJECT__.toml",
    ),
    "res_rbfe": (
        _RESFEP_TOML,
        "toml",
        "resfep",
        _gromacs_driver,
        "alchemforge resfep submit __PROJECT__.toml",
    ),
    "dna_rbfe": (
        _DNA_RBFE_TOML,
        "toml",
        "dna-rbfe",
        _gromacs_driver,
        "alchemforge dna-rbfe submit __PROJECT__.toml",
    ),
    "rna_rbfe": (
        _RNA_RBFE_TOML,
        "toml",
        "rna-rbfe",
        _gromacs_driver,
        "alchemforge rna-rbfe submit __PROJECT__.toml",
    ),
    "alanine_scan": (
        _ALASCAN_TOML,
        "toml",
        "alanine-scan",
        _alanine_driver,
        "alchemforge alanine-scan submit __PROJECT__.toml",
    ),
}

# OpenFE uses its native YAML (consumed by create_json). Add rbfe/abfe here as
# their templates become available.
_OPENFE: dict[str, tuple[str, str, str, object, str]] = {
    "rbfe": (
        _RBFE_OPENFE_YAML,
        "yaml",
        "rbfe",
        _openfe_driver,
        "chemflow openfe rbfe run __PROJECT__.yaml",
    ),
    "abfe": (
        _ABFE_OPENFE_YAML,
        "yaml",
        "abfe",
        _openfe_driver,
        "chemflow openfe abfe run __PROJECT__.yaml",
    ),
    "septop": (
        _SEPTOP_OPENFE_YAML,
        "yaml",
        "septop",
        _openfe_driver,
        "chemflow openfe septop run __PROJECT__.yaml",
    ),
}

_ENGINES: dict[str, dict[str, tuple[str, str, str, object, str]]] = {
    "gromacs": _GROMACS,
    "openfe": _OPENFE,
}
_ENGINE_CHOICES = tuple(_ENGINES)
_ALIASES = {"resfep": "res_rbfe", "res-fep": "res_rbfe", "abfe-gromacs": "abfe"}


def _overlay_toml_slurm(text: str, merged: dict) -> str:
    """Overlay resolved preset values onto a TOML ``[slurm]`` table (active keys only).

    Replaces the value of each active (non-commented) ``key = ...`` assignment for
    the keys present in ``merged``; commented lines and other tables are untouched.
    """
    import re

    str_keys = {"walltime", "memory", "partition", "account", "conda_environment"}
    int_keys = {"nodes", "ntasks", "cpus_per_task", "gpus"}
    for key in ("partition", "nodes", "cpus_per_task", "gpus", "memory", "walltime"):
        if key not in merged or merged[key] in (None, ""):
            continue
        value = merged[key]
        if key in str_keys:
            rendered = f'"{value}"'
        elif key in int_keys:
            rendered = str(int(value))
        else:
            rendered = str(value)
        pattern = re.compile(rf"^(\s*){key}\s*=\s*.*$", re.MULTILINE)
        text = pattern.sub(rf"\g<1>{key} = {rendered}", text, count=1)
    return text


def _generate(args: argparse.Namespace) -> int:
    engine = args.engine
    workflow = _ALIASES.get(args.generate_workflow, args.generate_workflow)
    table = _ENGINES.get(engine, {})
    if workflow not in table:
        available = ", ".join(sorted(table)) or "(none yet)"
        raise ValueError(
            f"The '{engine}' engine has no '{workflow}' template yet. "
            f"Available for '{engine}': {available}."
        )
    text, ext, command, driver_fn, submit_hint = table[workflow]
    project = args.name
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{project}.{ext}"
    sh_path = output_dir / f"{project}.sh"
    # OpenFE workflows also get an example SLURM array script next to the config.
    slurm_path = output_dir / f"{project}.slurm" if engine == "openfe" else None

    check_paths = [config_path, sh_path] + ([slurm_path] if slurm_path else [])
    for path in check_paths:
        if path.exists() and not args.force:
            raise ValueError(f"{path} already exists (use --force to overwrite)")

    header = f"# engine = {engine}\n"
    config_text = text.replace("__PROJECT__", project)
    # OpenFE: an optional named HPC preset (e.g. singlenode-singlegpu-long) is
    # overlaid onto the inline `slurm:` block so the config + .slurm both reflect it.
    if engine == "openfe" and getattr(args, "hpc_preset", None):
        import yaml  # noqa: PLC0415

        from chemflow.desktop.openfe_rbfe_config import (  # noqa: PLC0415
            apply_slurm_preset, format_slurm_block,
        )
        conf_full = yaml.safe_load(config_text) or {}
        slurm = apply_slurm_preset(conf_full.get("slurm") or {}, args.hpc_preset)
        if "\nslurm:" in config_text:
            config_text = config_text.split("\nslurm:")[0].rstrip() + "\n\n"
        config_text += (
            f"# --- HPC resources (preset: {args.hpc_preset}) ---\n"
            + format_slurm_block(slurm)
        )
    # GROMACS: overlay the preset onto the TOML [slurm] table (active keys only).
    elif engine == "gromacs" and getattr(args, "hpc_preset", None):
        from chemflow.desktop.openfe_rbfe_config import apply_slurm_preset  # noqa: PLC0415

        merged = apply_slurm_preset({}, args.hpc_preset)
        config_text = _overlay_toml_slurm(config_text, merged)
    config_path.write_text(header + config_text, encoding="utf-8")
    sh_text = driver_fn(engine, command, submit_hint, ext).replace(
        "__PROJECT__", project
    )
    sh_path.write_text(sh_text, encoding="utf-8")
    sh_path.chmod(sh_path.stat().st_mode | 0o100)

    # Example SLURM array script for the OpenFE workflows (one task per
    # transformation JSON x replicate). The --array starts as a 0-0 placeholder;
    # it auto-sizes when written after create_json, and the script guards at run
    # time, so edit --array (= transformations x replicates) before sbatch.
    if slurm_path is not None:
        import yaml  # noqa: PLC0415

        from chemflow.desktop.openfe_rbfe_config import slurm_script  # noqa: PLC0415

        conf = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        # Resources come from the inline `slurm:` block in the config (edit it there,
        # or pass --hpc-preset to seed it from env/hpc.yaml).
        slurm_path.write_text(slurm_script(config_path, None, conf), encoding="utf-8")

    print(f"Wrote {config_path}  (engine: {engine})")
    print(f"Wrote {sh_path}")
    if slurm_path is not None:
        print(f"Wrote {slurm_path}  (set --array after create_json, then: sbatch {slurm_path.name})")
    print(f"Next: edit {config_path.name}, then run  bash {sh_path.name}")
    return 0


def add_generate_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``alchemforge generate`` scaffolding command."""
    parser = subparsers.add_parser(
        "generate",
        help="Generate a starter config.toml + driver .sh for a workflow",
        description=(
            "Scaffold a commented TOML configuration and a matching driver "
            "script for an rbfe, abfe, septop, or res_rbfe workflow. Choose the "
            "engine with --engine (default: gromacs; openfe reserved for later)."
        ),
    )
    workflows = parser.add_subparsers(dest="generate_workflow", required=True)
    helps = {
        "rbfe": "Relative binding free energy",
        "abfe": "Absolute binding free energy",
        "septop": "Separated-topologies RBFE",
        "res_rbfe": "Protein residue-mutation FEP",
        "dna_rbfe": "DNA base-mutation RBFE",
        "rna_rbfe": "RNA base-mutation RBFE",
        "alanine_scan": "Alanine scanning (per-residue -> ALA)",
    }
    for key in (
        "rbfe",
        "abfe",
        "septop",
        "res_rbfe",
        "dna_rbfe",
        "rna_rbfe",
        "alanine_scan",
    ):
        step = workflows.add_parser(key, help=helps[key])
        step.add_argument(
            "name",
            nargs="?",
            default=key,
            help=f"Project name / output basename (default: {key})",
        )
        step.add_argument(
            "--engine",
            choices=_ENGINE_CHOICES,
            default="gromacs",
            help="Simulation engine for the generated config (default: gromacs)",
        )
        step.add_argument(
            "-o",
            "--output-dir",
            type=Path,
            default=Path("."),
            help="Directory to write the files into (default: current directory)",
        )
        step.add_argument(
            "--force", action="store_true", help="Overwrite existing files"
        )
        step.add_argument(
            "--hpc-preset",
            default=None,
            help="OpenFE only: overlay a named HPC preset from env/hpc.yaml "
                 "(e.g. singlenode-singlegpu-long) onto the config's inline slurm: block.",
        )
        step.set_defaults(func=_generate)
    return parser
