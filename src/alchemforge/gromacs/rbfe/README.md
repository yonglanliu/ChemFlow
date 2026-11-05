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

## Quick start: end to end from the command line

This walkthrough runs a complete soluble-receptor RBFE project headlessly, with
no desktop application. Every step is driven by one TOML configuration.

```bash
conda activate alchemforge
```

**1. Scaffold a config and driver.** `generate` writes a commented
`<name>.toml` plus a matching `<name>.sh` driver. The engine defaults to
GROMACS (`--engine openfe` is reserved for later):

```bash
alchemforge generate rbfe my_rbfe            # writes my_rbfe.toml and my_rbfe.sh
```

Open `my_rbfe.toml` and fill in the real paths and parameters — at minimum the
protein under `[system]`, the ligand SDF under `[ligands]`, and the Slurm block.
Every path is resolved relative to the config file.

**2. Prepare the inputs (optional headless prep).** If your receptor PDB and
ligand parameters are not ready yet, the `rbfe` subcommand group runs the four
preparation steps from the same config:

```bash
alchemforge rbfe fix-pdb        my_rbfe.toml   # repair/assemble the receptor
alchemforge rbfe assign-charge  my_rbfe.toml   # charge + parameterize ligands (AM1-BCC/GAFF2)
alchemforge rbfe plan           my_rbfe.toml   # build the perturbation network
alchemforge rbfe workflow       my_rbfe.toml   # stage edges + submit scripts
```

or run all four in order:

```bash
alchemforge rbfe all my_rbfe.toml
```

`fix-pdb` and `assign-charge` read two optional config sections,
`[protein_prep]` (cofactor/membrane/solvent/pbc_source, e.g. retained metals)
and `[ligand_prep]` (atom_type, net_charge/net_charges, resnames). `plan`
writes `network/edges.csv`; `workflow` reuses that network if present (pass
`--replan` to force replanning).

**3. Or stage directly.** If the receptor and ligands are already prepared, the
generated driver just validates and stages in one shot:

```bash
bash my_rbfe.sh            # = alchemforge ligand-rbfe validate + stage
# equivalently:
alchemforge ligand-rbfe validate my_rbfe.toml
alchemforge ligand-rbfe stage    my_rbfe.toml
```

**4. Submit.** Staging produces one self-contained pipeline per edge. Launch
them all:

```bash
bash work/my_rbfe/submit_production_array.sh
```

Each edge submits its own model-build job, then its own simulation array that
depends (`afterok`) only on *that edge's* build — so edges never wait on one
another, and no edge sits behind a global barrier. Per-edge arrays are small, so
they stay well under the scheduler's `MaxArraySize` and carry no concurrency cap.

**5. Monitor.** The submit script prints an inspection command; otherwise:

```bash
sacct -X -j <job-ids> --format=JobID,State,ExitCode,Reason,Elapsed,NodeList
```

**6. Restart a leg that failed or needs more wall time.** Resubmit just one
edge's complex or solvent leg. The task runner skips finished stages and resumes
unfinished ones from their `.cpt` checkpoint, so a plain restart continues where
it stopped:

```bash
# continue the complex leg of one edge
alchemforge ligand-rbfe restart my_rbfe.toml --edge edge_LIGA_to_LIGB --leg complex

# only specific replicates / lambda windows (independent sampling)
alchemforge ligand-rbfe restart my_rbfe.toml --edge edge_LIGA_to_LIGB --leg solvent \
    --replicate 2 --window 0 --window 1

# wipe that leg's outputs first for a clean rerun; preview without submitting
alchemforge ligand-rbfe restart my_rbfe.toml --edge edge_LIGA_to_LIGB --leg complex \
    --force --dry-run
```

`--window` is independent-sampling only; under HREX all windows of a replicate
are coupled in one exchange run, so restart operates at the replicate level
(`--replicate`) and rejects `--window`.

The sections below document each stage, the inputs, HREX, and troubleshooting in
depth.

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

Run **GROMACS → Ligand Preparation** in the desktop first, or do it headlessly
with `alchemforge rbfe assign-charge <config>` (see the quick start). Either way
the prepared ligand folder must contain `ligand_manifest.tsv`; each record points
to the matching charged MOL2, force-field parameters, GRO, and ITP files.

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
Simulation module:     cluster external-MPI GROMACS module
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

Do not build every hybrid edge locally. Staging creates one model-build task
per network edge plus the per-edge simulation arrays, all launched by:

```bash
bash <project>/submit_production_array.sh
```

The submission script iterates over edges and, for each one, submits a
single-task model build followed by that edge's simulation array with an
`afterok` dependency on **its own** build. Edges are therefore fully
independent — one edge begins simulating as soon as its model is ready, without
waiting for other edges. Completed edge models are detected and skipped, so
adding replicates does not rebuild existing models. Because each array covers
only one edge, the indices stay small (no `MaxArraySize` failures) and no
`%N` concurrency cap is applied.

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
├── model_build_tasks.txt
├── gromacs_model_build_array.sh
├── production_tasks.txt
├── gromacs_equilibration_array.sh
├── gromacs_production_array.sh
└── submit_production_array.sh
```

Readable edge names identify the transformation directly instead of using
anonymous names such as `edge_0001`.

## 6. Independent-window execution

Each lambda window has its own complete preparation and sampling sequence:

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
one per-lambda EM/NVT/NPT equilibration array
            ↓ afterok
one coordinated HREX production job
```

For 21 lambda windows, the equilibration array has 21 tasks. Its dependent
production job starts when all 21 lambda-specific NPT stages succeed. It does
not wait for other edges, the opposite thermodynamic leg, or other replicates.

Production uses:

```text
gmx_mpi mdrun -multidir <lambda directories> -replex <interval>
```

Every HREX replica must have its own `npt.gro` before production begins.
Complex and solvent legs are separate HREX groups and are never exchanged
with each other.

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

### Restarting one edge's leg

To relaunch a single edge's complex or solvent leg — after a node failure, a
wall-time limit, or a `--force` wipe — use the `restart` command instead of
resubmitting the whole project:

```bash
alchemforge ligand-rbfe restart my_rbfe.toml --edge edge_LIGA_to_LIGB --leg complex
```

It resubmits only that leg's tasks (offset into the staged manifest), so other
edges and the opposite leg are untouched. Options:

- `--replicate N` / `--window W` — restart only specific tasks (repeatable).
  `--window` is independent-sampling only; HREX couples all windows of a
  replicate, so it is rejected there — use `--replicate`.
- `--force` — delete that leg's stage outputs (`em/nvt/npt/prod` `.gro/.cpt/...`)
  first, keeping the `.mdp` inputs, for a clean rerun.
- `--dry-run` — print the `sbatch` command(s) without submitting.

Under HREX, `restart` resubmits the replicate's equilibration array and then its
coordinated production job (`afterok` on the equilibration).

### Regeneration

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
