# AlchemForge residue FEP

AlchemForge exposes the existing residue-based GROMACS FEP workflow through a
validated Python interface while retaining the established shell programs as
the execution backend.

## Installation

AlchemForge uses a separate environment from ChemFlow so OpenFE, OpenMM,
AmberTools, and their NumPy/RDKit constraints cannot change ChemFlow's ML
stack:

```bash
mamba env create -f environment-alchemforge.yaml
mamba activate alchemforge

python -m pip install -e src --no-deps --no-build-isolation
```

For a Linux HPC installation on NVIDIA GPU nodes, use the dedicated CUDA 12.6
environment instead. Create it from the repository root so the relative pip
requirements file can be found:

```bash
conda env create \
  --prefix /data/liuy48/conda/envs/alchemforge \
  --file environment-alchemforge-hpc.yaml
conda activate /data/liuy48/conda/envs/alchemforge

python -m pip install -e src --no-deps --no-build-isolation
```

The cluster supplies the NVIDIA driver. The HPC environment installs the CUDA
12.6 toolkit, including the runtime and NVRTC needed by OpenMM. Test CUDA from
an allocated GPU compute node, not from a login node. GPUs newer than the CUDA
12.6 architecture set (for example, Blackwell) require a newer CUDA environment
rather than this P100-compatible file.

GROMACS ABFE also requires the Python 3 `develop` branch of PMX. It is not
available from Bioconda on macOS ARM, so install it after activating the
AlchemForge environment:

```bash
git clone --branch develop https://github.com/deGrootLab/pmx.git
python -m pip install ./pmx --no-deps --no-build-isolation
```

The dedicated editable install registers the `alchemforge` and
`alchemforge-desktop` commands without registering ChemFlow or installing its
dependencies into this environment.

Launch the desktop application with:

```bash
alchemforge-desktop
```

The `Jobs` page monitors local background processes and submitted SLURM jobs.
Select an active row and click `Terminate Selected Job` to stop it. Local jobs
are launched in an isolated process group so termination also reaches the
Python/OpenMM/GROMACS child process; a process that ignores `SIGTERM` is killed
after three seconds. For submitted SLURM jobs, the same action runs `scancel`
for every recorded job ID.

The desktop includes `Binding Free Energy > Ligand Preparation > OpenFE >
Charge Molecules`. It runs ligand charging as a background job using the
packaged `openfe/charge_settings.yaml`. Its defaults are equivalent to:

```bash
openfe charge-molecules -M input.sdf -o output.sdf -n 4 \
  -s charge_settings.yaml --overwrite-charges
```

The dialog allows the input/output SDF, settings YAML, OpenFE executable,
process count, and overwrite behavior to be changed.

`Binding Free Energy > Ligand Preparation > GROMACS > Prepare Ligands` is a
standalone Open Babel + ACPYPE/GAFF preparation job. It accepts one MOL2 ligand
or every molecule in a multi-record SDF and creates a separate directory named
for each selected residue name. For example, residue `4DY` produces
`4DY/4DY.mol2`, `4DY/4DY.itp`, `4DY/4DY.gro`, and `4DY/4DY.top` alongside the
native ACPYPE output. A `ligand_manifest.tsv` maps
SDF names to the generated three-character residue names. This preparation is
separate from ABFE, ligand RBFE, SepTop RBFE, and residue FEP workflows so the
same parameterized ligands can be reused.

`Binding Free Energy > System Preparation > Fix & Assemble PDB` removes hydrogen
ATOM/HETATM records from a selected PDB and then runs `pdbfixer
--add-atoms=all`. The dialog accepts a protein PDB, an optional cofactor PDB,
and optional membrane and solvent PDB files. A cofactor file may contain one or
multiple cofactors. They are processed in the same background job. Original
files are preserved; outputs are named `fixed_protein.pdb`,
`fixed_cofactor.pdb`, `fixed_membrane.pdb`, and `fixed_solvent.pdb`. The job
then uses OpenMM `Modeller` to write `complex.pdb` in protein + cofactor +
membrane + solvent order, omitting unselected components. Cofactors and
membranes are copied unchanged rather than repaired by PDBFixer, which can
alter nonstandard residues. Solvent is also preserved because large Desmond
PDB solvent files can use extended atom identifiers that PDBFixer does not
support reliably. The assembler automatically falls back to MDTraj for those
extended identifiers. A large-system-safe `complex.cif` is also written
for visualization. Cofactor force-field parameters are still required by the
downstream OpenMM or GROMACS system.

The desktop also accepts an optional `PBC source PDB`. This is intended for a
full-system PDB converted from a Desmond `.cms` file. Only its `CRYST1` box is
read; its atoms are not included a second time. When no PBC source is selected,
the assembler searches the protein, cofactor, membrane, and solvent inputs for
`CRYST1`. It writes the selected periodic box into both `complex.pdb` and
`complex.cif` and rejects conflicting component boxes.

The same preparation is available from the command line. For a soluble
protein:

```bash
alchemforge fix-pdb --protein protein.pdb -o fixed
```

For a membrane system:

```bash
alchemforge fix-pdb --protein protein.pdb --membrane membrane.pdb \
  --solvent solvent.pdb -o fixed
```

Add retained cofactors with `--cofactor cofactors.pdb`. Supply the box from a
CMS-converted PDB with `--pbc-source system_from_cms.pdb`.

Use `--overwrite` to replace existing fixed PDB files and `complex.pdb`.

`Binding Free Energy > System Preparation > Energy Minimization` provides two
standalone minimizers. `OpenFE / OpenMM` loads an assembled PDB, creates an
OpenMM system from selected force-field XML files, and writes minimized PDB and
mmCIF structures plus an energy report. Every protein, lipid, solvent, ion,
ligand, and cofactor residue must have a force-field template. Periodic membrane
systems also require box vectors, either in the PDB `CRYST1` record or entered
in the dialog. Nonstandard ligands and cofactors can be supplied as charged SDF
or MOL2 definitions in the minimization dialog. AlchemForge registers either a
SMIRNOFF/OpenFF or GAFF template generator before creating the OpenMM system;
the definition must describe the same atoms and connectivity as its residue in
the assembled PDB.

For large membrane systems, select the generated `complex.cif` instead of
`complex.pdb`; mmCIF avoids legacy PDB atom/residue identifier overflow. The
desktop OpenMM minimizer can also normalize recognized Desmond lipid naming.
The current converter maps Desmond-style POPC headgroup atom names to the
Amber lipid17/CHARMM POPC convention while preserving coordinates and
connectivity. The minimization report records how many lipid residues were
normalized. This is a coordinate/topology naming conversion, not a transfer of
the OPLS parameters embedded in the original CMS.

`GROMACS` accepts an already parameterized `.gro`/`.pdb` and `.top` pair, runs
`grompp`, steepest-descent `mdrun`, and `editconf`, and writes both
`minimization.gro` and `minimized.pdb`. Its coordinate and topology inputs must
contain the same molecules in the same order. Fix & Assemble PDB is therefore a
structure-building step, not a substitute for GROMACS topology construction;
membranes, ligands, ions, and cofactors must be parameterized and included in
the `.top`/`.itp` files before minimization.

The desktop separates ligand-network planning from molecular-system setup.
`Binding Free Energy > Ligand Network Planning` accepts only a multi-molecule
SDF and graph/mapping settings. It writes a portable network bundle containing
`ligands.sdf`, full OpenFE `ligand_network.graphml`, `edges.csv`, per-edge mapping
JSON files, and `network_manifest.json`. The viewer can inspect structures and
atom mappings or prune edges while keeping the network connected.

The engine-specific entries under `RBFE` are system-setup dialogs. They consume
the saved network bundle and collect protein, membrane, solvent, cofactor,
protocol, lambda, and execution settings. OpenFE loads the saved GraphML rather
than replanning. GROMACS copies the same bundle into the project and uses the
saved edge table and mapping IDs. This permits one reviewed ligand graph to be
reused with multiple molecular systems or simulation protocols.

Start from `gromacs/resFEP/example.toml`, then validate and stage a project:

```bash
alchemforge resfep validate my_resfep.toml
alchemforge resfep stage my_resfep.toml
```

Ligand RBFE with GROMACS uses `gromacs/rbfe/example.toml`. It accepts one
prepared protein PDB and one multi-molecule ligand SDF. Staging uses OpenFE's
LoMap or Kartograf mapper to write a GraphML network, an edge table, and a
JSON atom mapping for every selected edge. Execution additionally requires a
GROMACS/PMX worker script:

```bash
alchemforge ligand-rbfe validate my_ligand_rbfe.toml
alchemforge ligand-rbfe stage my_ligand_rbfe.toml
alchemforge ligand-rbfe validate --for-execution my_ligand_rbfe.toml
alchemforge ligand-rbfe run my_ligand_rbfe.toml
```

GROMACS ABFE uses `gromacs/abfe/example.toml` and accepts one prepared protein
PDB plus one single-molecule ligand SDF. The built-in workflow parameterizes
the ligand with ACPYPE, prepares the protein with `pdb2gmx`, asks PMX to create
the Boresch restraints and complex/solvent systems, and writes EM, NVT, NPT,
and production MDP files for every lambda window. The lambda spacing reuses
the shared `cosine`, `linear`, or `power` generator:

```bash
alchemforge abfe-gromacs validate my_abfe.toml
alchemforge abfe-gromacs stage my_abfe.toml
alchemforge abfe-gromacs run my_abfe.toml
alchemforge abfe-gromacs analyze my_abfe.toml
```

`abfe` and `abfe_gromacs` remain accepted as compatibility aliases. The
desktop lists `OpenFE` and `GROMACS` separately beneath `ABFE`; both open their
dedicated configuration dialogs.

GROMACS SepTop RBFE is a separate backend named `septop-gromacs` (also
`septop_gromacs`). Start with `gromacs/septop/example.toml`. Its ligand SDF is
used to select a similarity network; SepTop itself keeps the two ligand
topologies separate and does not consume the generated atom mapping. Every SDF
`_Name` must have a matching directory below `prepared_root` containing
`ligand.mol2`, `complex.pdb`, `complex.gro`, `complex.top`, `solvent.gro`, and
`solvent.top` (an optional `complex.h5` trajectory improves restraint atom
selection). `septop_source_dir` is a checkout of MobleyLab's
SeparatedTopologies reference code. Its preparation step requires a licensed
OpenEye Toolkit installation.

```bash
alchemforge septop-gromacs validate my_septop.toml
alchemforge septop-gromacs stage my_septop.toml
alchemforge septop-gromacs prepare my_septop.toml
alchemforge septop-gromacs submit my_septop.toml
```

Staging preserves the published four-stage schedules: 12 and 8 complex
windows, then 18 and 10 solvent windows. It writes one auditable task table and
a bounded Slurm array. The desktop exposes the same setup as
`RBFE > SepTop RBFE > GROMACS`.

## Membrane systems

All four GROMACS backends understand `[system].type = "membrane"`. Membrane
inputs must already be built, solvated, ionized, equilibrated, aligned with the
normal along z, and include valid periodic box vectors. AlchemForge counts
recognized lipid residues before staging and exports a shared membrane
contract (`SYSTEM_TYPE`, lipid names, barostat, xy/z pressure and
compressibility, and reference-coordinate scaling). Complex-leg MDPs generated
by ABFE and SepTop use semi-isotropic pressure coupling; their solvent legs
remain isotropic.

- `abfe-gromacs` requires `system.prepared_setup_dir` containing
  `complex/genion.gro`, `complex/complex.top`, `ligand/genion.gro`,
  `ligand/ligand.top`, and `restraints.info`. It bypasses the soluble builder.
- `septop-gromacs` validates the membrane `complex.gro` in every ligand's
  prepared directory, assumes those complexes are pre-aligned by default, and
  adds a semi-isotropic NPT equilibration before complex production.
- `ligand-rbfe` requires `prepared_complex_gro` and
  `prepared_complex_topology`; its configured worker consumes the exported
  membrane variables. Its prepared-ligand charge manifest is checked against
  every network edge. Charge-changing edges can be rejected, staged with an
  alchemical co-ion for both complex and solvent legs, or paired with a
  user-supplied CSV of validated leg corrections. The staged worker contract is
  written to `.alchemforge/charge_strategy.json`, and the edge-level values are
  written to `network/charge_changes.csv`.
- `resfep` requires the same two prepared complex files and `setup_mode =
  "SKIP"`. `prepare-system` stages them instead of invoking the soluble
  pdb2gmx/solvation path, and writes `mdp/membrane_pressure.mdp` for the
  residue-FEP worker.

Common optional controls are `lipid_resnames`, `minimum_lipids`,
`reference_pressure_bar = [xy, z]`, `compressibility_bar = [xy, z]`, and
`refcoord_scaling`. See each backend's `example.toml` for a complete template.

Local mode uses the `gmx`, `acpype`, `obabel`, and `pmx` commands from the
active environment. Slurm mode submits setup followed by a dependent
complex/solvent array, so those commands must also exist on the cluster compute
nodes. A custom `execution.worker_script` is optional.

The same workflow is available through ChemFlow:

```bash
chemflow free-energy resfep validate my_resfep.toml
chemflow free-energy resfep prepare-ligand my_resfep.toml
chemflow free-energy resfep prepare-system my_resfep.toml
chemflow free-energy resfep submit my_resfep.toml
chemflow free-energy abfe-gromacs validate my_abfe.toml
chemflow free-energy septop-gromacs validate my_septop.toml
```

The caller must activate the desired Conda environment before invoking a
command. If a cluster requires modules or other site setup, set
`execution.environment_setup` to a readable shell fragment. This avoids
embedding cluster-specific Conda and module paths in the workflow.

`stage` creates `<work_dir>/<name>/.alchemforge/resfep_config.sh` for the legacy
shell backend and installs default MDP files when they are absent. Existing MDP
files are preserved. Equilibration durations can be written as `nvt_ps` /
`npt_ps` or `nvt_ns` / `npt_ns`; AlchemForge derives and validates all step
counts. Existing ligand `itp`, `gro`, and `top` files can also be declared in
the `[ligand]` table and are staged into the ACPYPE layout expected by ResFEP.

Submission additionally requires the simulation worker named
`ResAlchemFEP.sh`. Set `execution.worker_script` in TOML, or place that file in
the project directory before running `submit`.
