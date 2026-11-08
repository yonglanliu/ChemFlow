# PROTAC workflows in ChemFlow (`chemflow protac`)

PROTAC-specific steps that sit alongside the linker designer
(`chemflow generate reinvent linker`) and the structure-prep tool
(`chemflow generate reinvent prepare`).

## Ternary-complex docking — PRosettaC

Model the target-PROTAC-E3 ternary complex from the two binary structures
(target+warhead and E3+warhead) plus the full PROTAC, using
[PRosettaC](https://doi.org/10.1021/acs.jcim.9b00758). ChemFlow only calls an
externally-installed PRosettaC (Rosetta + PatchDock, in its own environment); it
never imports Rosetta.

```bash
chemflow protac prosettac generate -o prosettac.yaml      # commented example
# edit prosettac.yaml: the two PDBs + warhead residue names, the PROTAC SMILES,
# and prosettac_command (how you launch PRosettaC)
chemflow protac prosettac validate prosettac.yaml
chemflow protac prosettac run prosettac.yaml --dry-run    # stage params, print command
chemflow protac prosettac run prosettac.yaml
```

Inputs (`prosettac.yaml`): `target` and `e3` blocks (each `pdb` + `chain` +
warhead `ligand` residue name), `protac_smiles`, sampling (`n_models`,
`top_models`, `rmsd_cutoff`), and `prosettac_command`.

> **Parameter-file format:** PRosettaC's parameter keys vary by version. The
> generated `prosettac_params.txt` is a **provisional** scaffold — verify it
> against your PRosettaC version. The robust alternative is `settings_template:`
> pointing at your PRosettaC example params file, which the wrapper fills in using
> `{target_pdb} {target_chain} {target_ligand} {e3_pdb} {e3_chain} {e3_ligand}
> {protac_smiles} {n_models} {top_models} {rmsd_cutoff} {out_dir}` placeholders.

> **Status:** config loading, validation, params generation and template
> substitution are unit-tested (`tests/test_chemflow_protac.py`, no Rosetta
> needed). A real PRosettaC run has NOT been exercised — install PRosettaC and
> confirm the parameter format on your machine.

## Where this fits in the PROTAC pipeline

The right order is **search the interaction first, then design the linker to fit
it** — the linker length is a consequence of the complex you pick, not an input.

1. **Search the ternary interaction (no linker).** Bring the two warhead-bound
   proteins together and explore arrangements:
   - **Co-folding** — `alchemfold dock --engine boltz|chai` with the two proteins
     and the two *separate* warheads. Fast; returns the most likely complex.
   - **RosettaDock** — `chemflow protac rosettadock` global docking. Samples many
     arrangements to rank; better for a broad search. Its input is one PDB with
     both partners as chains — build it with
     `chemflow protac merge-pdb --pdb1 target_warhead.pdb --pdb2 e3_warhead.pdb`
     (puts target on chain A, E3 on chain B, each warhead on its protein's chain).
     Optional, only if you dock with the warheads: make a `.params` per warhead
     with `chemflow protac ligand-params` and pass them via `ligand_params`.
2. **Screen the docked models (rank + linkability in one step).**
   `chemflow protac screen` reads the docking `score.csv`, takes the best models by
   interface score (`I_sc`), and for each measures the through-space distance
   between the two warhead attachment atoms. It writes `screen.csv` and copies every
   **linkable** model into a `linkable/` folder — the set worth taking forward.

   ```bash
   chemflow protac screen --dir pyrosetta_dock_output \
       --atom1 LIG:C7 --atom2 VH1:C3 --top 50
   ```
3. **(Optional) Judge linkability for one candidate.**
   `chemflow protac linkability` reports the distance and orientation between the
   two warhead attachment points for a single candidate complex — a closer look at
   one arrangement once you've narrowed the field with `screen`.

   ```bash
   chemflow protac linkability \
       --head1-file target_warhead.sdf --atom1 14 \
       --head2-file e3_warhead.sdf     --atom2 9
   ```
4. **Design the linker for the chosen complex.** Extract the two posed warheads
   (`chemflow generate reinvent prepare --ligand …`) and run
   `chemflow generate reinvent linker` (or the desktop page); the 3D-fit check
   confirms the linker spans this exact arrangement.
5. **Assemble the full PROTAC**, then optionally **refine the ternary complex with
   PRosettaC** using that complete PROTAC (above).
6. **Relax + MD** — `alchemforge openmm modeling build-soluble` → `relax` → `md`.
7. **Endpoint triage** — an MM/PBSA-style ΔG over the trajectory to rank
   candidates before any rigorous FEP.

So PRosettaC is a **refinement** step (it needs the full PROTAC); the front of the
pipeline is co-folding / RosettaDock → screen → linkability → linker design.
