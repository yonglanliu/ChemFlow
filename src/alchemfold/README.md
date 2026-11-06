# AlchemFold — Co-Folding (`alchemfold`)

Co-folding predicts the 3D structure of a protein **together with** its ligand
(and, optionally, their binding affinity) in one shot — rather than docking into a
fixed receptor. AlchemFold is a **standalone** component of the platform, separate
from the AlchemForge free-energy package. It supports multiple models behind one
interface: **Boltz-2** and **Chai-1** (sequence-based co-folding) and **DiffDock**
(diffusion docking into a receptor structure).

Two entry points:

- `alchemfold dock …` — **docking with a selectable model** (Boltz-2 / Chai-1) and
  an optional binding-site **pocket** constraint. One config, `engine:` picks the model.
- `alchemfold boltz …` — the Boltz-2-only path (incl. affinity prediction).

The predicted complex (PDB/mmCIF) drops straight into the OpenMM modeling →
free-energy pipeline:

```
alchemfold dock run  →  complex PDB
      │
      ▼
alchemforge openmm modeling build-soluble | build-membrane  →  relax  →  FEP
```

---

## Docking (multi-model) — `alchemfold dock`

```bash
# pick a model with --engine (boltz | chai); --pocket adds a binding-site block
alchemfold dock generate --engine boltz -o docking.yaml \
    --sequence MKT... --smiles "CC(=O)Oc1ccccc1C(=O)O" --pocket
# edit docking.yaml (sequence, ligand, pocket residues as chain:residue)
alchemfold dock validate docking.yaml
alchemfold dock run docking.yaml            # --dry-run prints the engine command
```

| Model (`engine`) | Package | Notes |
|---|---|---|
| `boltz` (Boltz-2) | `boltz` | MIT, local; supports binding-affinity prediction + pocket constraints |
| `chai` (Chai-1) | `chai-lab` | open weights, local; pocket via a restraints file (best-effort) |
| `diffdock` (DiffDock) | GitHub repo (not PyPI) | diffusion **docking into a receptor PDB** (or ESMFold from sequence); blind — no pocket/affinity |

The **pocket** block steers the ligand into a known site (Boltz/Chai only; DiffDock
is blind docking):

```yaml
pocket:
  binder: L                 # which ligand binds
  max_distance: 6.0         # Angstrom
  contacts:                 # pocket-lining residues
    - {chain: A, residue: 123}
    - {chain: A, residue: 145}
```

`dock` commands: `generate` (`--engine`, `--pocket`, `--sequence`, `--smiles`,
`--protein-id`, `--ligand-id`, `-o`, `--force`), `validate`, `stage`,
`run [--dry-run]`. The desktop equivalent is **AlchemFold → Docking**, whose
**Model** dropdown selects the engine.

You can also dock from a **ligand file** instead of a single SMILES — a CSV
(`smiles` column, optional `name`), a `.smi`, or a `.cxsmiles` file (plain or
`.bz2`/`.gz`). A single-file list is docked one run per entry:

```yaml
ligand_file: "ligands.smi"
ligand_file_format: "auto"     # auto | csv | smiles | cxsmiles
```

---

## Virtual screening (`alchemfold screen`)

Screen a ligand library in parallel on HPC and keep the top hits:

```
combine ligand_files → split into shards → SLURM array (1 task/shard, across
GPUs/nodes) → dock each shard with the chosen engine → rank → top-K
```

```bash
alchemfold screen generate --engine boltz -o screen.yaml   # commented template
# edit screen.yaml: add receptor sequence + ligand_files + sharding + [slurm]
alchemfold screen validate screen.yaml
alchemfold screen stage    screen.yaml                      # writes shards + jobs
bash submit_screen.sh                                       # on the cluster
#   -> sbatch dock_array.sh  (array), then rank.sh (afterok dependency)
alchemfold screen rank     screen.yaml                      # (also run by rank.sh)
```

Config highlights: `ligand_files` (a **list** of libraries), `n_shards` **or**
`shard_size`, a `[slurm]` block (`partition`, `account`, `nodes`,
`gpus_per_task` — `0` for CPU, `gpu_type`, `cpus_per_task`, `mem`, `time`,
`conda_env`), and `[ranking]` (`metric`, `top_k`, `order`, `output`). The array
is emitted as a plain `#SBATCH --array=0-N` (no concurrency cap), so the
scheduler fills every free GPU/node.

Ranking scrapes a score per predicted complex (`auto` picks per engine: Boltz-2
affinity/confidence, Chai aggregate/ptm, DiffDock confidence), writes a sorted
`ranking.csv`, and copies the top-K structures to `top/`. Score parsing is
best-effort across engine versions — check `ranking.csv` and adjust
`ranking.metric` if needed. The desktop equivalent is **AlchemFold → Virtual
Screening**.

---

## Requirements

- **Boltz-2** (`pip install boltz`) with a GPU — needed only to `run`.
- Internet access if `use_msa_server: true` (Boltz computes MSAs on its server);
  otherwise provide a precomputed MSA per chain and set `use_msa_server: false`.

The config / validation / staging layer is import-light (only PyYAML), so you can
`generate`, `validate`, and `stage` without Boltz installed.

---

## Quick start

```bash
# 1. generate a commented config and edit in your sequence + ligand
alchemfold generate -o cofold.yaml --sequence MKT... --smiles "CC(=O)Oc1ccccc1C(=O)O"

# 2. (optional) validate without Boltz
alchemfold boltz validate cofold.yaml

# 3. run Boltz-2 (GPU)
alchemfold boltz run cofold.yaml            # --dry-run prints the boltz command

# 4. feed the predicted complex into system modeling + FEP
alchemforge openmm modeling build-soluble run soluble.yaml
```

`alchemfold generate` flags: `-o/--output`, `--protein-id`, `--ligand-id`,
`--sequence`, `--smiles`, `--no-affinity`, `--force`.

---

## CLI

| Command | Purpose |
|---|---|
| `alchemfold generate` | Write a fully-commented example config YAML |
| `alchemfold boltz validate CONFIG` | Check inputs (no Boltz needed) |
| `alchemfold boltz stage CONFIG` | Write a `run_cofold_boltz.sh` runner |
| `alchemfold boltz run CONFIG [--dry-run]` | Build the Boltz input + run `boltz predict` |

(`python -m alchemfold ...` works too.)

---

## Config

Key fields (see the generated file for inline explanations):

- `proteins`: list of `{id, sequence, [msa]}` chains.
- `ligands`: list of `{id, smiles}` or `{id, ccd}` (PDB chemical-component code).
- `predict_affinity` / `affinity_binder`: enable Boltz-2 affinity for a ligand id.
- `use_msa_server`, `diffusion_samples`, `recycling_steps`, `sampling_steps`,
  `seed`, `accelerator` (gpu/cpu/tpu), `devices`, `output_format` (pdb/mmcif),
  `boltz_command`, `out_dir`, `override`.

Co-folding requires either a **ligand** or **≥ 2 protein chains** — a single apo
chain is plain structure prediction and is rejected.

---

## Files

| File | Role |
|---|---|
| `boltz_config.py` | light config layer (load/validate/stage/generate) |
| `boltz_run.py` | builds the Boltz input YAML + `boltz predict` command, runs it |
| `cli.py` / `__main__.py` | the `alchemfold` CLI |

---

## Note on accuracy

Co-folding recovers **orthosteric** poses well but is weaker at **allosteric**
sites, large **induced-fit** motions, and **virtual-screening ranking** (often no
better than docking on diverse decoys). Treat AlchemFold as a pose/complex
*generator* that feeds physics-based relaxation + FEP, not as a standalone
affinity ranker.
