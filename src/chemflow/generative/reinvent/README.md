# REINVENT4 in ChemFlow — installation and tutorial

[REINVENT4](https://github.com/MolecularAI/REINVENT4) is a generative AI tool for
small-molecule design (de novo design, scaffold decoration, linker design,
molecule optimisation, peptides). ChemFlow wraps it behind one YAML file and adds
two things on top: a bridge that lets **ChemFlow-trained models act as the
reinforcement-learning reward**, and an **export step** that turns the best
generated molecules into files for AlchemFold docking and Alchemforge free-energy
work.

```
 chemflow generate reinvent run   →  molecules (CSV)
        │   (reward can call your ChemFlow models)
        ▼
 chemflow generate reinvent export →  top-N SMI / CSV / 3D SDF
        │
        ▼
 alchemfold dock  /  alchemforge rbfe  /  alchemforge openmm modeling
```

ChemFlow does **not** import REINVENT4. It writes a REINVENT4 settings file and
launches the `reinvent` program as a separate process. That is why the two live in
**separate conda environments** (see below).

---

## 1. Installation

### 1.1 Why two environments

REINVENT4 pins `torch==2.13.0` and requires numpy 2, pandas 3 and a very recent
RDKit. Installing it into the `chemflow` environment upgrades those packages and
breaks others there (we saw pytdc, anndata, tiledb, accelerate and streamlit
conflicts). Keep them apart:

| Environment | Contains | You run here |
|---|---|---|
| `chemflow` | ChemFlow, chemprop, your trained models | every `chemflow ...` command |
| `reinvent` | REINVENT4 and its pinned packages | nothing directly; ChemFlow launches it |

### 1.2 Create the `reinvent` environment

```bash
conda create -n reinvent python=3.11 -y
conda activate reinvent

# Get REINVENT4: clone it, or download the ZIP from GitHub and unzip it
git clone --depth 1 https://github.com/MolecularAI/REINVENT4.git
cd REINVENT4
pip install -e .

# REINVENT4 imports scipy but does not declare it as a dependency
pip install scipy

which reinvent          # note this path; you need it in step 2
reinvent --help
```

Keep the REINVENT4 folder where it is. The `-e` (editable) install points at it.

Upstream's README now recommends `uv sync` instead of `pip install -e .`. Use that
if plain pip does not work for you.

**macOS note.** If `pip install -e .` fails with
`AssertionError: .../distutils/core.py`, something in the environment loads
Python's built-in `distutils` too early. This workaround is safe on Python 3.11:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib pip install -e . --no-build-isolation
```

### 1.3 Download the pretrained models ("priors")

The model files are **not** in the REINVENT4 repository. Download them from
Zenodo: <https://doi.org/10.5281/zenodo.15641296>

Create a `priors` folder inside the REINVENT4 folder and save the files there:

```
REINVENT4/priors/reinvent_pubchem.prior
```

You only need the files for the models you use:

| ChemFlow `model:` | Prior file to download | Task |
|---|---|---|
| `reinvent` | `reinvent_pubchem.prior` (23.5 MB) | de novo design |
| `libinvent` | `libinvent.prior` | add R-groups to a scaffold |
| `linkinvent` | `linkinvent.prior` | link two fragments |
| `mol2mol` | `pubchem_ecfp4_with_count_with_rank_reinvent4_dict_voc.prior` | optimise existing molecules |
| `pepinvent` | `pepinvent.prior` | peptides |

`libinvent_transformer_pubchem.prior` and `linkinvent_transformer_pubchem.prior`
are alternative transformer versions of the LibInvent and LinkInvent models.

If you want to check a download, compare its MD5 with the one Zenodo shows:
`md5 reinvent_pubchem.prior` (macOS) should print `75052d5e93b2470e9c6684d369704f5f`.

### 1.4 Check the ChemFlow side

In the `chemflow` environment:

```bash
conda activate chemflow
chemflow generate reinvent --help
```

You should see `generate`, `validate`, `stage`, `run` and `export`.

---

## 2. Tutorial 1 — sample molecules from the pretrained model

Sampling asks the pretrained model to write new drug-like molecules. It needs no
score and runs on a CPU in seconds. Use it to confirm the installation works.

Create `sample.yaml` (a commented template is also available from
`chemflow generate reinvent generate --model reinvent --run-type sampling -o sample.yaml`):

```yaml
model: reinvent
run_type: sampling
reinvent_command: <conda-prefix>/envs/reinvent/bin/reinvent   # from `which reinvent`
out_dir: reinvent_output
device: cpu
seed: 42

prior_file: <REINVENT4-dir>/priors/reinvent_pubchem.prior

num_smiles: 100
unique_molecules: true
randomize_smiles: true
```

Always use **full paths** for `reinvent_command` and `prior_file`. Relative paths
are resolved against the folder that contains the YAML file.

Run, from the `chemflow` environment:

```bash
chemflow generate reinvent validate sample.yaml      # checks paths and settings
chemflow generate reinvent run sample.yaml --dry-run # writes the TOML, prints the command
chemflow generate reinvent run sample.yaml
```

Results land in `reinvent_output/`:

| File | Content |
|---|---|
| `sampling.csv` | `SMILES`, `SMILES_state`, `NLL` for each sampled molecule |
| `reinvent.log` | REINVENT4's log; read this first if anything looks wrong |
| `reinvent_sampling.toml` | the settings file ChemFlow generated for REINVENT4 |

A typical 100-molecule run returns about 96 valid molecules; the rest fail RDKit
sanitisation and are dropped.

> The log may contain an `ERROR ... has invalid hash` message about the prior
> file. REINVENT4 only logs it and carries on; the sampling is unaffected.

---

## 3. Tutorial 2 — reinforcement learning with a built-in score

Reinforcement learning (RL) trains a copy of the model (the *agent*) to produce
molecules that score well, while a diversity filter and the original prior keep
the molecules varied and realistic. This tutorial uses QED, a 0–1
drug-likeness score built into REINVENT4.

`rl_qed.yaml`:

```yaml
model: reinvent
run_type: staged_learning
reinvent_command: <conda-prefix>/envs/reinvent/bin/reinvent
out_dir: reinvent_rl_output
device: cpu
seed: 42

prior_file: <REINVENT4-dir>/priors/reinvent_pubchem.prior

batch_size: 64                 # molecules generated per step

stages:
  - chkpt_file: stage1.chkpt
    termination: simple
    max_score: 0.95            # stop early if the average score gets this high
    min_steps: 10
    max_steps: 30              # short test; use 200+ for a real run
    scoring:
      type: geometric_mean
      components:
        - {name: QED, weight: 1.0}
```

```bash
chemflow generate reinvent validate rl_qed.yaml
chemflow generate reinvent run rl_qed.yaml
chemflow generate reinvent export reinvent_rl_output/staged_learning_1.csv --top 20 --sdf
```

`staged_learning_1.csv` has one row per generated molecule with its SMILES and
score. Watch the average score rise over the steps (REINVENT4 prints progress to
the terminal and `reinvent.log`); if it does not, run more steps.
RL is much faster on an NVIDIA GPU (`device: cuda:0`). On Apple silicon,
`device: mps` is accepted by REINVENT4 but `cpu` is the safe choice.

Other built-in scoring components include `MolecularWeight`, `SlogP`, `TPSA`,
`custom_alerts` (SMARTS filters) and many more; see `configs/SCORING.md` in the
REINVENT4 folder. Each component entry accepts:

```yaml
- name: MolecularWeight          # REINVENT4 component name
  endpoint_name: MW              # optional label in the output CSV
  weight: 0.5
  params: {smarts: ["[#6;+]"]}   # optional component parameters
  transform: {type: double_sigmoid, low: 200, high: 500,
              coef_div: 500.0, coef_si: 20.0, coef_se: 20.0}
```

A *transform* maps a raw value (such as a molecular weight) onto a 0–1 score.
Available types include `sigmoid`, `reverse_sigmoid`, `double_sigmoid`, `step`,
`exponential_decay` and `value_mapping`.

---

## 4. Tutorial 3 — use a ChemFlow model as the reward

This is the point of having REINVENT4 inside ChemFlow: any model trained with
ChemFlow can score the generated molecules, so RL optimises *your* property
(for example predicted logD or a classifier probability).

Add `chemflow_scorers` to an RL config:

```yaml
chemflow_scorers:
  - name: logD
    model: /path/to/trained_model.pkl      # a ChemFlow model package
    column: prediction                      # prediction | prob_class_1 | ...
    weight: 1.0
    python: <conda-prefix>/envs/chemflow/bin/python
    transform:                              # reward logD between 1 and 3
      type: double_sigmoid
      low: 1.0
      high: 3.0
      coef_div: 3.0
      coef_si: 10.0
      coef_se: 10.0
```

How it works: for each batch, REINVENT4 starts
`python -m chemflow.generative.reinvent.scorer --model ...`, sends it the SMILES
on standard input, and reads back JSON scores. Molecules ChemFlow cannot
featurise get `NaN`, which REINVENT4 treats as invalid.

Things to know:

- `python:` must be an interpreter that can `import chemflow`; point it at the
  `chemflow` environment, because REINVENT4 itself runs in the `reinvent`
  environment. If omitted, the interpreter that runs `chemflow` is used.
- `column` selects which predictor output column becomes the score. Use
  `prediction` for regression; for classifiers use `prob_class_1` (or the class
  you want to maximise).
- Use `transform` to say what "good" means. A raw logD of 8 is not better than 3,
  and an untransformed score would reward it anyway.
- Each ChemFlow scorer becomes its own REINVENT4 component (REINVENT4 allows
  only one executable per component), so several scorers can be combined, each
  with its own weight. With `geometric_mean`, a molecule that scores near 0 on
  any one component scores near 0 overall.
- Test the scorer on its own first:
  `echo "CCO" | python -m chemflow.generative.reinvent.scorer --model model.pkl`
  should print one line of JSON.

> **Status:** the scorer's JSON protocol and the generated TOML are covered by
> unit tests and match REINVENT4's documented `ExternalProcess` format, but a full
> RL run driven by a ChemFlow model has not yet been run end to end. Start with a
> short `max_steps` run and check `reinvent.log`.

---

## 5. Other models: scaffolds, linkers, optimisation, peptides

Generate a template for any model and run type:

```bash
chemflow generate reinvent generate --model libinvent --run-type sampling -o lib.yaml
```

These models need an input file, set with `smiles_file:` (a full path).
Lines starting with `#` are comments.

| `model` | `smiles_file` content | Example line |
|---|---|---|
| `libinvent` | one scaffold per line, attachment points marked `[*:0]`, `[*:1]` | `[*:0]Cc2ccc1cncc(C[*:1])c1c2` |
| `linkinvent` | two warheads per line, separated by `\|` | `Oc1cncc(*)c1\|*c1ccoc1` |
| `mol2mol` | one molecule per line (optional name after a tab) | `O=S(=O)(c1cccc2cnccc12)N1CCCNCC1	CHEMBL38380` |
| `pepinvent` | one masked peptide per line (CHUCKLES notation) | see `REINVENT4/configs/pepinvent.smi` |

Example files for each are in `REINVENT4/configs/` (`scaffolds.smi`,
`warheads.smi`, `mol2mol.smi`, `pepinvent.smi`). For these models `num_smiles` is
the number of molecules **per input line**. `mol2mol` and `pepinvent` also accept
`sample_strategy: multinomial` or `beamsearch`, and `mol2mol` accepts
`temperature`. Mol2Mol can need a lot of memory, so keep the input list short.

> **Status:** only the `reinvent` (de novo) model has been run end to end. The
> others generate valid REINVENT4 settings, but treat your first run as a test.

---

## 6. Export and hand-off to the rest of the platform

```bash
chemflow generate reinvent export reinvent_rl_output/staged_learning_1.csv \
    --top 100 --min-score 0.6 --sdf -o results/top_molecules
```

| Option | Meaning |
|---|---|
| `--top N` | keep the N best unique molecules (by score; input order if there is no score) |
| `--min-score X` | drop molecules scoring below X |
| `--sdf` | also write a 3D SDF (RDKit embedding + MMFF optimisation) |
| `-o PREFIX` | output prefix; default is `<csv>_top` |

Output: `PREFIX.smi`, `PREFIX.csv` (`id,smiles,score`) and, with `--sdf`,
`PREFIX.sdf`. Molecules get IDs `gen0001`, `gen0002`, ... in score order.

The RDKit warnings `UFFTYPER: Unrecognized charge state` mean one molecule has an
unusual atom and RDKit used a simpler force field for it; its 3D geometry is
rougher, so inspect it before using it.

The 3D structures are ligand-only starting points. Typical next steps:

- dock them: `alchemfold dock ...` (see `src/alchemfold/README.md`)
- free-energy work: `alchemforge rbfe ...` or `alchemforge alanine-scan ...`

---

## 7. PROTAC linker designer (pick an atom on each end)

Design a linker between two ligands by choosing **one atom on each** and letting
LinkInvent propose linkers. Every linker comes back with scores.

```bash
# 1. See the atom numbers (RDKit indices) of each ligand
chemflow generate reinvent linker --head1-file target.sdf --head2-file e3.sdf --list-atoms

# 2. Pick one atom per ligand that carries a hydrogen, and generate
chemflow generate reinvent linker \
    --head1-file target_pose.sdf --atom1 14 \
    --head2-file e3_pose.sdf     --atom2 9 \
    --protein ternary_complex.pdb \
    --prior /path/to/priors/linkinvent.prior \
    --reinvent-command <conda-prefix>/envs/reinvent/bin/reinvent \
    --num 200 --fit-top 50 -o linker_design
```

Use `--head1-smiles` / `--head2-smiles` instead of files for a 2D-only design
(no 3D fit). `--dry-run` only writes the LinkInvent input line so you can check the
attachment points.

**Scores per linker** (written to `linkers.csv`):

| Group | Columns |
|---|---|
| Linker | length in bonds (as REINVENT4's `FragmentEffectiveLength`), heavy atoms, rotatable bonds, rings |
| Whole PROTAC | MW, cLogP, TPSA, HBD, HBA, rotatable bonds, QED, SA score |
| ChemFlow models | one column per `--model NAME=PATH[:COLUMN]` (repeatable), run on the complete PROTAC |
| 3D pocket fit | pose match, clashes, closest contact, strain (needs 3D heads in one coordinate frame) |

**How the 3D fit works.** Both ligands stay at their input poses. RDKit builds
conformers of the whole PROTAC around them, the conformers are superimposed back
onto the poses, and the linker is relaxed with the ligands held fixed. Then it
counts linker–protein contacts closer than 3 Å and estimates strain. A linker that
is too short, too rigid, or crashes into the protein is flagged. This removes bad
linkers; it does not predict binding or degradation. Only `ATOM` records of the
`--protein` PDB count as obstacles (ligand `HETATM` records are ignored so a
ligand cannot clash with its own linker).

**Two separate structures instead of one complex?** Add `--separate-frames`. The
3D fit is then skipped (it needs both ligands in one frame, e.g. a ternary-complex
model), but with `--protein` the tool still reports whether each exit vector points
into open space.

Output files in `-o`: `linkers.csv` (all scores), `protacs.smi`, and
`protac_poses.sdf` (best 3D pose of each fitted PROTAC).

### Point-and-click in the desktop apps

The designer is a page in ChemFlow Studio:

* **ChemFlow Studio** (`chemflow-desktop`): the **PROTAC linker** item in the left
  sidebar. Set the REINVENT4 executable and `linkinvent.prior` path in the panel
  (or via the `REINVENT_COMMAND` / `LINKINVENT_PRIOR` environment variables).

Load each ligand, click one atom on each in the 3D view, press
**Generate Linkers**, and the results appear in a table with their scores.
Selecting a row shows that PROTAC's 3D pose.

**Try Example** loads two example warheads (a real CRBN binder, pomalidomide, and
a simple aryl target binder) built with RDKit, with their attachment atoms
pre-selected — so you only need to set the `linkinvent.prior` path and press
Generate. It needs no download and works offline.

### Prepare a real complex from the PDB

To design against a real pocket, start from a downloaded ternary complex and
split it into a protein file and two posed ligand files:

```bash
# 1. See what ligands the structure contains (the PROTAC is usually the biggest)
chemflow generate reinvent prepare --pdb 5T35 --list

# 2. Extract the protein + the two warhead residues you want
chemflow generate reinvent prepare --pdb 5T35 \
    --ligand <RESNAME1>:<CHAIN> --ligand <RESNAME2>:<CHAIN> -o complex_prep
```

This writes `complex_prep/protein.pdb` and one SDF per ligand. Load those in the
designer with **Load Ligand 1 / Ligand 2** (the SDFs) and **Load Protein**
(`protein.pdb`); because they share the crystal frame, the 3D pocket-fit column
becomes meaningful. PDB coordinates carry no bond orders, so the tool transfers
them from each component's ideal SDF when it can — still, check the extracted
ligand structures before relying on them. If your PROTAC is a single connected
ligand (one residue) rather than two, you'll need to split it at the linker in a
tool like PyMOL first; this command extracts whole residues.

```bash
export REINVENT_COMMAND=<conda-prefix>/envs/reinvent/bin/reinvent
export LINKINVENT_PRIOR=<REINVENT4-dir>/priors/linkinvent.prior
chemflow-desktop
```

Clicking needs an internet connection (3Dmol.js loads in the embedded browser),
the same as the existing structure viewers.

> **Status:** the sampling, parsing, export and PDB-reading code is covered by
> tests that run without RDKit. The chemistry code (atom attachment, scores, 3D
> fit) needs RDKit; run `pytest tests/test_chemflow_linker.py -v` in the
> `chemflow` environment to check it. The desktop HTML builders are unit-tested,
> but the full click-to-generate flow in each running app still needs hands-on
> testing (it cannot be automated here) — the most likely thing to verify first
> is that a click in the 3D view highlights the atom.

---

## 8. Command reference

```bash
chemflow generate reinvent generate [--model M] [--run-type T] [-o FILE] [--force]
chemflow generate reinvent validate CONFIG
chemflow generate reinvent stage    CONFIG            # write the TOML only
chemflow generate reinvent run      CONFIG [--dry-run]
chemflow generate reinvent export   CSV [--top N] [--min-score X] [--sdf] [-o PREFIX]
```

`chemflow generate gpt ...` (the GPT generator) is unchanged apart from now being
an explicit subcommand.

### YAML settings

| Key | Default | Meaning |
|---|---|---|
| `model` | `reinvent` | `reinvent`, `libinvent`, `linkinvent`, `mol2mol`, `pepinvent` |
| `run_type` | `sampling` | `sampling` or `staged_learning` (RL) |
| `reinvent_command` | `reinvent` | the `reinvent` executable; use a full path |
| `out_dir` | `reinvent_output` | results folder, relative to the YAML |
| `device` | `cuda:0` | `cpu`, `cuda`, `cuda:0`, `mps` |
| `seed` | none | random seed (passed as `-s`) |
| `prior_file` | required | the model file (full path) |
| `agent_file` | `prior_file` | RL starting model; set to a `.chkpt` to continue a run |
| `smiles_file` | none | seed file for lib/link/mol2mol/pepinvent |
| `num_smiles` | 1000 | sampling: molecules to generate (per input line for seeded models) |
| `batch_size` | 100 | RL: molecules per step |
| `learning_strategy` | `dap`, sigma 128, rate 0.0001 | RL reward shaping |
| `diversity_filter` | Murcko scaffold, bucket 25 | RL: penalise repeated scaffolds |
| `stages` | none | RL stages (see Tutorial 2); at least one is required for RL |
| `chemflow_scorers` | none | ChemFlow models used as rewards (Tutorial 3) |
| `parameters_extra` | none | any extra REINVENT4 `[parameters]` keys, passed through |

REINVENT4 rejects keys it does not know, so check `configs/PARAMS.md` in the
REINVENT4 folder before adding anything under `parameters_extra`.

---

## 9. Troubleshooting

| Message | Cause and fix |
|---|---|
| `bash: chemflow: command not found` | you are in the `reinvent` environment; run `conda activate chemflow` |
| `REINVENT config not found` | the YAML path is wrong; create it with `chemflow generate reinvent generate -o x.yaml` or give the full path |
| `REINVENT4 executable not found: 'reinvent'` | set `reinvent_command` to the full path printed by `which reinvent` in the `reinvent` environment |
| `ModuleNotFoundError: No module named 'scipy'` (from REINVENT4) | `conda activate reinvent && pip install scipy` |
| `prior_file not found` | the prior is not at that path; download it (section 1.3) and use the full path |
| `model 'X' requires smiles_file` | add `smiles_file:` for libinvent, linkinvent, mol2mol and pepinvent |
| `AssertionError ... distutils/core.py` during `pip install` | use the macOS workaround in section 1.2 |
| `... has invalid hash` in the log | harmless; REINVENT4 only logs it |
| "exited without an error code but produced no result CSV" | REINVENT4 logged an error and stopped; read `<out_dir>/reinvent.log` |
| RL score does not improve | run more steps, relax `max_score`, or check that your transform does not flatten the reward |

---

## 10. Files

| Path | Purpose |
|---|---|
| `src/chemflow/generative/reinvent/config.py` | YAML loading, validation, TOML generation |
| `src/chemflow/generative/reinvent/run.py` | stages the TOML and launches `reinvent` |
| `src/chemflow/generative/reinvent/scorer.py` | ChemFlow model as a REINVENT4 scoring component |
| `src/chemflow/generative/reinvent/export.py` | top-N selection and 3D SDF export |
| `src/chemflow/cli/reinvent.py` | the `chemflow generate reinvent` commands |
| `environment-reinvent.yaml` | starting point for the `reinvent` conda environment |
| `tests/test_chemflow_reinvent.py` | unit tests (no REINVENT4 or GPU needed) |
