# Preparing a Maestro membrane system for OpenFE

A membrane system built in **Maestro / Desmond (Schrödinger)** and exported directly
to a single PDB will usually **fail** when passed to OpenFE
(`ProteinMembraneComponent.from_pdb_file`): the water hydrogens are missing or
inconsistent, ions/solvent are bookkept differently, and the combined topology
confuses the parser.

The reliable route is to export the pieces separately, repair the solvent
hydrogens, and merge them with OpenMM. This directory ships a helper that automates
steps 2–3 — [`membrane_assembly.py`](membrane_assembly.py).

## Step 1 — split and export in Maestro

Export the system into separate files:

- `protein.pdb` — protein only
- `membrane.pdb` — lipids only
- `sol.pdb` — water + ions. **Delete the water hydrogens** and export as `sol_noh.pdb`.
- `cofactors.sdf` — any cofactors (optional)

Cofactors stay as an SDF. OpenFE treats them as `SmallMoleculeComponent`s, so they
are **not** merged into the complex PDB — you point the workflow config at
`cofactors.sdf` separately (`cofactor_sdf`).

## Step 2 — re-add solvent hydrogens with PDBFixer

Add the water hydrogens back:

```bash
pdbfixer sol_noh.pdb --output=fixed_sol.pdb --add-atoms=all
```

## Step 3 — assemble protein + membrane + solvent into one PDB

```python
import openmm.app
from openmm.app import Modeller, PDBFile

protein = PDBFile("protein.pdb")
lipids  = PDBFile("membrane.pdb")
water   = PDBFile("fixed_sol.pdb")

# Start from the protein, then add lipids and solvent
modeller = Modeller(protein.topology, protein.positions)
modeller.add(lipids.topology, lipids.positions)
modeller.add(water.topology, water.positions)

with open("complex.pdb", "w") as f:
    PDBFile.writeFile(modeller.topology, modeller.positions, f)
```

> The original note used `openfe.ProteinComponent.from_pdb_file(...).to_openmm_topology()`
> to obtain each topology; loading directly with `openmm.app.PDBFile` is equivalent
> and avoids OpenFE's protein-only sanitisation of the lipid/solvent PDBs.

## Automated: one command

The three-file merge (and the PDBFixer step) is wrapped in `membrane_assembly.py`:

```bash
conda run -n alchemforge python -m chemflow.fe.openfe.membrane_assembly \
    --protein protein.pdb \
    --membrane membrane.pdb \
    --solvent sol_noh.pdb \
    --output complex.pdb
```

This writes `fixed_sol.pdb` (hydrogens re-added) and `complex.pdb` (merged). Useful
flags:

- `--fixed-solvent fixed_sol.pdb` — choose where the hydrogen-fixed solvent is written.
- `--ph 7.0` — pH used for `addMissingHydrogens`.
- `--solvent-has-hydrogens` — skip PDBFixer when the solvent already has hydrogens.

Or call it from Python:

```python
from chemflow.fe.openfe.membrane_assembly import build_membrane_complex

build_membrane_complex(
    "protein.pdb", "membrane.pdb", "sol_noh.pdb", "complex.pdb",
)
```

The merged `complex.pdb` keeps the periodic box (CRYST1) from the inputs when present.

## Step 4 — use it in the OpenFE membrane workflow

Point the workflow config at the assembled `complex.pdb` as the protein input, turn
on the membrane path, and give the box. In the RBFE/ABFE/SepTop config:

```toml
protein_pdb         = "complex.pdb"     # protein + membrane + solvent
ligand_sdf          = "ligands.sdf"
cofactor_sdf        = "cofactors.sdf"   # optional; kept separate
membrane_system     = true
complex_box_vectors = [a, b, c]         # nm; or use complex_box_size
surface_tension     = 0.0               # bar·nm (nonzero uses the membrane barostat)
```

With `membrane_system = true`, the OpenFE `create_json` step builds the complex leg
with a `MonteCarloMembraneBarostat` and reads the box from `complex_box_vectors`
(falling back to `complex_box_size`). Then run as usual, e.g.:

```bash
conda run -n alchemforge python -m chemflow.fe.openfe.rbfe.create_json --conf_file membrane_rbfe.toml
```
