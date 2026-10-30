# GROMACS LigandFEP-RBFE Tutorial

This module builds ligand relative binding free-energy (RBFE) calculations
with GROMACS and PMX. It prepares a ligand transformation network, constructs
one hybrid topology per edge, creates complex and solvent legs, writes the
alchemical MDP files, and generates local or Slurm execution scripts.

The production protocol is:

```text
Energy minimization → restrained NVT → restrained NPT → unrestrained production
```

Production can use independent lambda windows or Hamiltonian replica exchange
(HREX). In HREX mode, every edge, thermodynamic leg, and replicate is an
independent exchange group.

## 1. Requirements

Use the AlchemForge environment described in the repository installation
guide. Confirm the main tools before creating a workflow:

```bash
conda activate alchemforge

alchemforge --help
gmx --version
pmx --help
python -c "import parmed, rdkit; print('Python dependencies: OK')"
```

Independent-window calculations can use a thread-MPI GROMACS executable such
as `gmx`. HREX additionally requires an external-MPI build, normally
`gmx_mpi`, because GROMACS uses `-multidir` with one MPI rank per lambda
replica.

On a module-based cluster, verify the HREX build after loading its module:

```bash
module load gromacs/2026.3-gcc13.2
gmx_mpi --version
```

The version output must not report `MPI version: built in` for the HREX
executable.

## 2. Required inputs

### Ligands

Start from a multi-molecule SDF containing one named, three-dimensional
structure per ligand. Names must be unique and stable because they are used in
the network, edge names, and prepared-ligand manifest.

Run **GROMACS → Ligand Preparation** in the desktop first. The prepared ligand
folder must contain `ligand_manifest.tsv`; each record points to the matching
charged MOL2, force-field parameters, GRO, and ITP files.

Check that:

- the intended formal charge is assigned to every ligand;
- the reported partial-charge sum matches that formal charge;
- all ligands use compatible GAFF/force-field conventions;
- their coordinates represent the intended binding pose or have been aligned;
- ligand names in the manifest match names in the network SDF.

### Soluble receptor

Provide a prepared protein PDB. The workflow can construct and solvate the
complex and ligand-only solvent legs. Nonstandard residues and cofactors need
compatible parameters.

### Membrane receptor

Provide a fully parameterized, ligand-free membrane system:

```text
system.gro
topol.top
```

The GRO and topology must describe exactly the same atom order and molecule
counts. The system should already contain protein, membrane, solvent, ions,
and retained cofactors. The periodic box must be correct.

The membrane base must not retain the original bound ligand. The workflow
inserts the PMX hybrid ligand separately for each network edge.

Before RBFE staging, verify that the prepared base can be minimized without
infinite forces. A successful `grompp` alone checks consistency but does not
prove that coordinates are free of atomic overlaps.

## 3. Desktop workflow

Launch the desktop:

```bash
alchemforge-desktop
```

Open:

```text
Binding Free Energy Workflow
└── GROMACS
    └── LigandFEP-RBFE
```

Complete the tabs in order.

### Ligand Preparation

Select the prepared GROMACS ligand folder. Confirm ligand names, residue
names, assigned charges, and parameter files.

### Network Creation

Create or reuse the ligand network. LoMap is a reasonable default for network
selection. PMX performs the atom mapping used to construct each hybrid
topology.

Review the graph before continuing:

- every ligand should be connected;
- avoid transformations with implausibly small common substructures;
- inspect mappings for changes in ring systems, stereochemistry, and charge;
- use redundant edges when cycle closure is needed for diagnostics.

### System

Select the receptor or prepared membrane system, force-field components,
solvent model, box, salt concentration, and prepared ligand folder.

For charge-changing edges, choose one explicit strategy:

- **Co-ion:** transform a distant counterion to preserve the periodic-cell
  charge. This is the recommended automated option.
- **Correction table:** supply validated complex- and solvent-leg finite-size
  corrections.
- **Reject:** prevent charge-changing edges.

### FEP Simulation

The stage table contains four stages:

| Stage | Purpose | Default restraints |
|---|---|---|
| EM | Remove finite steric strain | None |
| NVT | Establish temperature | Protein, hybrid ligand, and cofactors in the complex leg |
| NPT | Establish pressure and density | Reduced complex-leg restraints |
| Production | Collect free-energy data | None |

The solvent leg uses no positional restraints. NVT and NPT restraint reference
coordinates are taken from `em.gro`, so restraints are anchored to the
minimized structure rather than the original input coordinates.

The default generated lambda schedule introduces bonded/soft-core van der
Waals terms before charges on appearing atoms. Inspect the schedule preview
and increase the number of windows for difficult transformations.

### Job Setting

For one GPU per independent window, a typical setting is:

```text
nodes:             1
tasks:             1
cpus per task:     8
GPUs:              1
thread-MPI ranks:  1
PME:               CPU
```

For HREX, set:

```text
sampling method:       Lambda HREX
HREX module:           cluster external-MPI GROMACS module
MPI launcher:          mpirun (or the cluster-supported launcher)
HREX command:          gmx_mpi
exchange interval:     500 steps
```

## 4. Command-line configuration

Copy `example.toml` and edit its paths and parameters:

```bash
cp src/alchemforge/gromacs/rbfe/example.toml ligand_rbfe.toml
```

Validate and stage it:

```bash
alchemforge ligand-rbfe validate ligand_rbfe.toml
alchemforge ligand-rbfe stage ligand_rbfe.toml
```

When the network already exists and must not be replanned:

```bash
alchemforge ligand-rbfe stage ligand_rbfe.toml --reuse-network
```

For a newly constructed project, run the generated hybrid-system builder once:

```bash
bash <project>/build_hybrid_systems.sh
```

Do not run that builder merely to add replicates to systems that have already
been prepared.

## 5. Generated directory layout

The important output paths are:

```text
<project>/
├── network/
│   ├── edges.csv
│   ├── ligand_network.graphml
│   ├── ligands.sdf
│   └── mappings/
├── edges/
│   └── edge_LIGA_to_LIGB/
│       ├── hybrid/
│       ├── complex/
│       │   ├── setup/
│       │   │   ├── system.gro
│       │   │   └── topol.top
│       │   └── rep_1/lambda_000/
│       └── solvent/
│           ├── setup/
│           │   ├── system.gro
│           │   └── topol.top
│           └── rep_1/lambda_000/
├── equilibration_tasks.txt
├── production_tasks.txt
├── gromacs_equilibration_array.sh
├── gromacs_production_array.sh
└── submit_production_array.sh
```

Readable edge names identify the transformation directly instead of using
anonymous names such as `edge_0001`.

## 6. Independent-window execution

In independent mode, each array task runs one lambda window sequentially:

```text
EM → NVT → NPT → Production
```

Submit with:

```bash
bash <project>/submit_production_array.sh
```

The task runner reuses a completed stage when its output GRO exists. A
checkpoint is passed with `-cpi` and `-append` when available.

## 7. HREX execution

For every edge × leg × replicate group, AlchemForge submits:

```text
one lambda equilibration array
            ↓ afterok
one coordinated HREX production job
```

For 21 lambda windows, the equilibration array has 21 tasks. Its dependent
production job starts as soon as those 21 NPT stages succeed. It does not wait
for other edges, the opposite thermodynamic leg, or other replicates.

Production uses:

```text
gmx_mpi mdrun -multidir <lambda directories> -replex <interval>
```

Every HREX replica must have `npt.gro` before production begins. Complex and
solvent legs are separate HREX groups and are never exchanged with each other.

Inspect submitted jobs with the command printed by the submit script, or use:

```bash
sacct -X -j <job-ids> \
    --format=JobID,State,ExitCode,Reason,Elapsed,NodeList
```

## 8. Adding replicates without rebuilding systems

Suppose `rep_1` is complete and two additional replicates are required. Edit:

```toml
[simulation]
start_replicate = 2
replicates = 2
```

Then stage only the new replicate range:

```bash
alchemforge ligand-rbfe stage ligand_rbfe.toml --reuse-network
bash <project>/submit_production_array.sh
```

This creates `rep_2` and `rep_3`, rewrites the task manifests for that range,
and preserves `rep_1`. Do not run `build_gromacs_rbfe_workflow.sh` or
`build_hybrid_systems.sh` again.

For three replicates starting at 3:

```toml
start_replicate = 3
replicates = 3
```

This creates `rep_3`, `rep_4`, and `rep_5`.

## 9. Restarts and regeneration

Generated shell scripts are snapshots. Updating AlchemForge does not alter
scripts already present in an existing project. Regenerate or restage the
workflow after a code update.

A TPR also stores its restraint reference coordinates and MDP settings. If
those settings change, remove or archive the affected TPR and downstream
stage outputs before resubmission; otherwise the runner correctly assumes the
existing stage is complete.

Do not delete completed replicate directories when merely adding replicates.

## 10. Validation and troubleshooting

### Verify stage completion

For equilibration, the required final file is:

```text
npt.gro
```

For production, check for `prod.gro`, `prod.log`, `prod.edr`, and the expected
free-energy output such as `dhdl.xvg`.

### Infinite force during EM

This indicates a coordinate clash, not a scheduler failure. A valid system
should have finite potential energy and force. Do not hide infinite forces
with `-maxwarn`.

For the atom number reported by GROMACS:

```bash
ATOM=<one-based-atom-number>
sed -n "$((ATOM + 2))p" system.gro
```

Inspect periodic nearest neighbors and remove or rebuild overlapping water,
ion, lipid, ligand, cofactor, or protein components before continuing.

### “Other residues” in `grompp`

This category commonly contains lipids, cofactors, and the hybrid ligand. For
Amber Lipid17/21, one physical lipid can be represented by multiple residue
components such as `PA`, `PC`, and `OL`. The classification itself is not an
error.

### `gmx_mpi` not found

Load the external-MPI GROMACS module and set the HREX executable to `gmx_mpi`.
The Conda GROMACS build may provide only thread-MPI and therefore cannot run
multi-directory HREX.

### OpenMP thread mismatch

`OMP_NUM_THREADS` must equal `-ntomp`. Generated scripts run EM with one CPU
thread and use the configured CPUs per task for NVT/NPT. HREX uses one OpenMP
thread per MPI replica by default.

### Python module missing from a generated job

Regenerate the job after activating AlchemForge. Current generated helpers pin
the active Python interpreter rather than using `/usr/bin/python3`.

### ParmEd missing

Install ParmEd into the active environment with Conda:

```bash
conda install --prefix "$CONDA_PREFIX" -c conda-forge parmed
```

Verify:

```bash
python -c "import sys, parmed; print(sys.executable); print(parmed.__file__)"
```

## 11. Scientific checks before analysis

Before trusting RBFE results, check:

- energy minimization converged with finite forces;
- temperature and pressure stabilized during NVT/NPT;
- no ligand escaped or adopted an unintended pose;
- HREX acceptance and state mixing are adequate;
- neighboring lambda states have sufficient overlap;
- replicate estimates agree within uncertainty;
- thermodynamic cycles close within uncertainty;
- charge-changing transformations used the intended co-ion or correction;
- complex and solvent legs used consistent force-field and lambda settings.

Preserve all input manifests, generated MDP files, topology files, job scripts,
software versions, and random seeds with the final results.
