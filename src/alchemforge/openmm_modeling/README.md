# OpenMM System Modeling (`alchemforge.openmm_modeling`)

OpenMM-based tools to turn a bare protein (optionally with a bound ligand) into a
solvated, equilibrated system ready for OpenFE / GROMACS free-energy calculations
— for
both **soluble** and **membrane** proteins.

Everything is driven by a single, fully-commented YAML config per step, exposed
through the `alchemforge openmm modeling` CLI and matching desktop dialogs.

---

## Pipeline

```
                 generate  →  edit YAML  →  run
Soluble:   build-soluble ─────────────────────────────┐
                                                       ├──►  md  ──►  OpenFE FEP
Membrane:  build-membrane ──►  relax-membrane ─────────┘
```

- **Soluble path:** solvate + ionize a water box around the protein, then run MD.
- **Membrane path:** build and *trim* a minimal lipid patch around the protein,
  carefully pre-equilibrate it (staged restrained protocol), then run MD.

Each step reads a YAML config and writes a PDB (carrying its periodic box) that
feeds the next step.

---

## Requirements

- **OpenMM** (compute) — needed only to *run* a step (`... run`), not to
  `generate`/`validate`/`stage`.
- **openff-toolkit** + **openmmforcefields** — only if you parameterize a ligand
  from an SDF (`ligand_sdf` in the config).
- Amber **ff14SB** + **Lipid17** force-field XMLs ship as the defaults.

The config/validation/staging layer is import-light (no OpenMM), so configs can
be generated and checked anywhere.

---

## Membrane orientation — align to OPM (required for membrane systems)

The membrane builder and relaxer assume the input protein is **oriented in the
OPM frame**:

- the **membrane normal is the Z axis**, and
- the **bilayer midplane is at z = 0**.

This matches the [OPM database](https://opm.phar.umich.edu/) (Orientations of
Proteins in Membranes) convention. The builder places both leaflets symmetrically
about `build.membrane_center_z` (default `0.0`), and the relaxer restrains lipid
head groups along Z about that same plane — so if the protein is not OPM-aligned,
the lipids will be built across the wrong slab and the restraints will be
meaningless.

**How to align before building:**

1. **Download a pre-oriented structure from OPM** — https://opm.phar.umich.edu/
   returns the protein already rotated/translated into the membrane frame
   (normal = Z, bilayer center = z = 0). Use that PDB directly as `input_pdb`.
2. **Or orient your own structure with PPM** (the PPM server/standalone, the
   method behind OPM) to produce the same OPM frame, then use its output.

Keep `build.membrane_center_z = 0.0` to match the OPM midplane. If your bilayer
is deliberately centered elsewhere, set `membrane_center_z` to that Z value — but
the protein must still be oriented with its membrane-spanning axis along Z.

> Soluble systems have no such requirement — `build-soluble` just wraps a water
> box around the solute in its current orientation.

---

## Quick start

### Soluble system

```bash
# 1. generate a commented config and edit it
alchemforge openmm modeling generate soluble -o soluble.yaml --input-pdb protein.pdb

# 2. solvate + ionize
alchemforge openmm modeling build-soluble run soluble.yaml       # -> soluble_complex.pdb

# 3. MD (min -> NVT -> NPT -> production)
alchemforge openmm modeling generate md -o md.yaml --input-pdb soluble_complex.pdb
alchemforge openmm modeling md run md.yaml
```

### Membrane system

> **First align the protein to the OPM frame** (membrane normal = Z, bilayer
> center = z = 0) — download it pre-oriented from OPM or orient with PPM. See
> "Membrane orientation" above.

```bash
# 1. build + trim a minimal membrane around the (OPM-oriented) protein
alchemforge openmm modeling generate membrane -o membrane.yaml --input-pdb protein.pdb
alchemforge openmm modeling build-membrane run membrane.yaml     # -> membrane_complex.pdb

# 2. staged restrained pre-equilibration
alchemforge openmm modeling generate relax -o relax.yaml --input-pdb membrane_complex.pdb
alchemforge openmm modeling relax-membrane run relax.yaml        # -> relaxed_production.pdb

# 3. production MD (membrane barostat)
alchemforge openmm modeling generate md -o md.yaml \
    --input-pdb relaxed_production.pdb --system-type membrane
alchemforge openmm modeling md run md.yaml
```

Add `--ligand ligand.sdf` to `generate` for any step whose input PDB contains a
bound ligand; the config will include a ligand block and the ligand atoms are
parameterized with GAFF2 (or an OpenFF SMIRNOFF line) at run time.

---

## CLI reference

All steps live under `alchemforge openmm modeling`:

| Command | Purpose |
|---|---|
| `generate {soluble,membrane,relax,md}` | Write a fully-commented example config YAML |
| `build-soluble {validate,stage,run}` | Solvate + ionize a soluble complex |
| `build-membrane {validate,stage,run}` | Tile + trim a membrane (or solvate an existing one) |
| `relax-membrane {validate,stage,run}` | Staged restrained membrane pre-equilibration |
| `md {validate,stage,run}` | Unrestrained MD (soluble or membrane) |

Every runnable step shares the same three verbs:

- **`validate CONFIG`** — check the config (no OpenMM needed).
- **`stage CONFIG`** — write a `run_*.sh` runner script (used by the desktop job
  runner / SLURM); does not execute.
- **`run CONFIG [--dry-run]`** — execute now (requires OpenMM). `--dry-run`
  prints the command instead.

### `generate`

```bash
alchemforge openmm modeling generate KIND [-o OUT] [--input-pdb P] \
    [--ligand SDF] [--lipid POPC] [--system-type soluble|membrane] [--force]
```

- `KIND` — one of `soluble`, `membrane`, `relax`, `md`.
- `-o/--output` — output path (default `<kind>.yaml` in the current directory).
- `--input-pdb` — pre-fill the `input_pdb` field.
- `--ligand` — include a ligand block pre-filled with this SDF path.
- `--lipid` — lipid type for the membrane builder (default `POPC`).
- `--system-type` — barostat type for the MD config (default `soluble`).
- `--force` — overwrite an existing file (otherwise the command refuses).

The generated file carries a header block plus an inline comment on every field
(units, allowed values, what it feeds).

---

## The steps

### `build-soluble`

Solvates a soluble protein(+ligand) in a water box with ions
(`Modeller.addSolvent`). Key config fields:

- `box.mode`: `padding` (buffer + `box_shape` = cube/dodecahedron/octahedron),
  `boxsize` (`[Lx, Ly, Lz]` nm), or `vectors` (3×3 nm).
- `solvent_model` (`tip3p`/`spce`/`tip4pew`/`tip5p`), `positive_ion`,
  `negative_ion`, `ionic_strength` (molar), `neutralize`.

Output: `soluble_complex.pdb`.

### `build-membrane`

Wraps `MembraneModeller`. **The input protein must be OPM-aligned** (membrane
normal = Z, bilayer center = z = 0; see "Membrane orientation" above). Two modes:

- **`build`** — tiles a bundled OpenMM lipid patch around the protein, balances
  the leaflets, relaxes the membrane against a scaled-in protein, then **trims
  lipids and waters to protein-XY + padding**, giving a *minimal* membrane patch;
  then solvates and adds ions. Lipids are placed symmetrically about
  `build.membrane_center_z` (default `0.0` = the OPM midplane).
- **`existing`** — the input already has a membrane; only solvate the empty space
  (preserving existing waters, deleting hydrophobic-core waters) and add ions.

Key fields: `build.lipid_type` (POPC/POPE/DLPC/DLPE/DMPC/DOPC/DPPC),
`build.scale_factor`, `build.membrane_center_z` (nm, OPM midplane = 0),
`build.{x,y,z}_padding` (nm), ions/strength.

Output: `membrane_complex.pdb`.

> **Note:** build mode uses OpenMM's bundled lipid patch PDBs, so the lipid must
> be one of the supported seven, and its internal water is OpenMM's default
> TIP3P. The membrane normal is **Z** and the bilayer is centered at
> `membrane_center_z` — align the protein to the OPM frame first.

### `relax-membrane`

A Desmond/CHARMM-GUI-style staged pre-equilibration that keeps a trimmed patch
from rupturing: **cold start + gradual heating + progressive restraint release**,
with the periodic box preserved (no padding added).

| stage | ensemble | T (K) | dt (fs) | time (ps) | bb | sc | lipid-Z | ligand |
|---|---|---|---|---|---|---|---|---|
| min | — | — | — | 10000 it | unrestrained |
| 1 | NVT | 100 | 1 | 50 | 4184 | 2092 | 1046 | 4184 |
| 2 | NPγT | 150 | 1 | 50 | 2092 | 1046 | 1046 | 2092 |
| 3 | NPγT | 200 | 2 | 100 (+short min) | 1046 | 418 | 418 | 1046 |
| 4 | NPγT | 250→275→300 | 2 | 50/50/100 | 418 | 209 | 209 | 418 |
| 5 | NVT | 300 | 2 | 100 | 209 | — | — | 209 |
| 6 | NVT | 300 | 4 | 100 | unrestrained |
| prod | NPγT | 300 | 4 | 5000 | unrestrained |

Force constants are in kJ/mol/nm² (the CHARMM-GUI 10→0.5 kcal/mol/Å² ladder).
Every stage is in the generated YAML and fully editable.

Output: `relaxed_production.pdb` (+ `.xml` state, `.chk` checkpoint).

### `md`

Unrestrained production MD for a prepared/relaxed complex:
**min → NVT → NPT → production**. `system_type` selects the barostat:

- `soluble` → isotropic `MonteCarloBarostat`
- `membrane` → `MonteCarloMembraneBarostat` (NPγT, XY-isotropic, Z-free;
  set `surface_tension` for an applied tension, default tensionless)

Key fields: `temperature`, `pressure`, `surface_tension`, `friction`,
per-phase `timestep`/`simulation_time`, `report_interval`. The timestep defaults
ramp to 4 fs for production (hydrogen-mass repartitioning is on). The input PDB
**must** carry a periodic box (CRYST1).

Outputs: `<prefix>_{minimized,nvt,npt,equilibrated,production}.pdb`, a DCD
trajectory + CSV energy log per phase, and a final state + checkpoint.

---

## Desktop

The same steps are available in the desktop navigation under **OpenMM Modeling &
MD**:

- Soluble System Builder
- Membrane Builder
- Membrane Relaxation
- MD Simulation

Each dialog writes the YAML config + a runner script and launches it through the
Job Monitor.

---

## Files

| File | Role |
|---|---|
| `soluble_build_config.py` / `soluble_build.py` | soluble builder (config / compute) |
| `membrane_build_config.py` / `membrane_build.py` | membrane builder (`MembraneModeller`) |
| `membrane_relax_config.py` / `membrane_relax.py` | staged membrane relaxation |
| `md_config.py` / `md_run.py` | unified MD runner |
| `*.example.yaml` | generated example configs |

The `*_config.py` modules are OpenMM-free (used by the CLI, desktop, and tests);
the compute modules import OpenMM and are only loaded when a step runs.

---

## Notes & gotchas

- **Membrane normal is Z and the bilayer center is z = 0 (OPM frame)** — align
  the protein to OPM (download a pre-oriented structure from
  https://opm.phar.umich.edu/ or orient with PPM) before building.
- **Lipid residue names**: the relaxer recognizes 3-character names
  (POP/DLP/DMP/DOP/DPP) and water/ions (HOH/WAT/NA/CL/K/CA/MG). If your topology
  uses other names, extend the sets in the relax/build modules.
- **Ligands** must be present in the input PDB as non-protein/non-lipid/non-water
  residues, with a matching SDF (correct bond orders/protonation) for
  parameterization.
- `generate`/`validate`/`stage` work without OpenMM; only `run` needs it.
