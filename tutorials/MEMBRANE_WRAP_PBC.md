# Wrapping a membrane system for OpenFE with `alchemforge wrap-pbc`

The fastest way to turn an equilibrated Desmond/Maestro membrane system into an
input OpenFE RBFE can actually run: read the box from Schrödinger, then let
`alchemforge wrap-pbc` **rotate, wrap, and center** the structure in one command.

This is the single-command alternative to the all-Schrödinger route in
[OPENFE_MEMBRANE_INPUT.md](OPENFE_MEMBRANE_INPUT.md). Use whichever you prefer;
both fix the same crash:

```
Potential energy is NaN after 0 attempts of integration ...
Propagating replica 0 at state 0 resulted in a NaN!
```

That NaN is the **input structure**, not the alchemy, GPU, temperature, charge, or
surface tension. At state 0 (the physical endpoint) OpenMM evaluates the energy of
the frame as-is. Three things in a raw Desmond/Maestro frame make it blow up, and
`wrap-pbc` fixes all three:

1. **Unwrapped atoms** sitting outside the box get imaged onto their periodic
   partners → ion–ion overlaps → infinite Coulomb energy.
2. **Molecules split across a box face** (a per-residue wrap of a protein does this)
   → a bond stretched ~one box length → enormous `HarmonicBond`/`Angle` forces.
3. **A tilted ("chorus") box** — orthorhombic but rotated in the lab frame — so
   wrapping along x/y/z shears the system into diagonally offset blocks.

A reliable way to confirm the input is the cause: run a force decomposition on the
saved `nan-error-logs` state. A broken frame shows a few atoms with ~10⁸ kJ/mol/nm
forces while the ligand/alchemical term stays small; a healthy one is ~10²–10³.

---

## Step 1 — get the box (and the structure) from Schrödinger

You need two things from the equilibrated system: the periodic box vectors and a
single-frame structure (`complex.pdb`).

**Box vectors.** The Desmond box is stored as the nine `r_chorus_box_*` values
(row-major lattice vectors `a, b, c`, in Ångström). Read them from the `-out.cms`:

```bash
$SCHRODINGER/run python3 -c "
from schrodinger.application.desmond.packages import topo
_, cms = topo.read_cms('jobname-out.cms')
print(cms.box)     # 9 numbers, row-major a,b,c (Angstrom)
"
```

Example output:

```
[122.13, 2.57, -8.64, -2.77, 124.12, -2.21, 9.77, 2.69, 138.76]
```

The non-zero off-diagonal terms (`2.57, -8.64, …`) mean the box is **tilted**. It
is still orthorhombic — the three edges are mutually perpendicular — just rotated in
the lab frame. `wrap-pbc` will un-tilt it for you.

> Sanity check (optional): the edge **lengths** should equal your
> `complex_box_vectors`. Here |a|,|b|,|c| = 12.246, 12.417, 13.913 nm.

**Structure.** If you built the system with `alchemforge fix-pdb`, you already have
`complex.cif` (the large-system-safe mmCIF it writes alongside `complex.pdb`).
Otherwise export the equilibrated frame from Maestro (*File → Export Structure…*)
as **mmCIF** — for a full solvated membrane, not PDB (>99,999-atom overflow). Do
**not** `trj_align` or `trj_unwrap` first — `wrap-pbc` handles wrapping, and
alignment only rotates the slab away from its box.

---

## Step 2 — rotate, wrap, and center in one command

Simplest of all: point it straight at the **`.mae`**. A Maestro `.mae` carries explicit
bonds *and* the chorus box, so no `--box…` and no `--guess-bonds` are needed:

```bash
conda activate alchemforge     # needs openmm; .mae reading also needs $SCHRODINGER

alchemforge wrap-pbc complex.mae complex_wrapped.cif --center protein
```

If you only have a CIF/PDB (no bonds, no box), supply the box and let it guess bonds:

```bash
# easiest: let wrap-pbc read the box straight from the .cms
alchemforge wrap-pbc complex.cif complex_wrapped.cif \
    --box-from-cms jobname-out.cms \
    --center protein --guess-bonds
```

> **Use mmCIF (`.cif`) for a full solvated membrane system.** `wrap-pbc` reads and
> writes PDB or mmCIF, dispatched on the file extension. A full system is usually
> >99,999 atoms, which overflows PDB's serial column (Maestro then writes Hybrid36
> like `A000G` that OpenMM/OpenFE can't read back). mmCIF has no such limit, and
> `create_json` accepts it. PDB in/out is fine for small systems.

What each option does:

- **Where the box comes from** — three ways, pick one:
  - `--box-from-cms jobname-out.cms` — read the box (with Desmond's tilt) straight
    from the `.cms` via `$SCHRODINGER/run`, so you never hand-copy numbers. Easiest.
  - `--box9-ang "…"` — paste the nine chorus-box numbers yourself (Ångström,
    row-major `a,b,c`).
  - *nothing* — if the input PDB/mmCIF carries a `CRYST1`/`_cell` box, it's read
    automatically. **But** a Maestro/Desmond export usually has no box, and even when
    it does, `CRYST1`/`_cell` cannot store Desmond's lab-frame tilt — only the `.cms`
    has that. So for a tilted chorus box, use `--box-from-cms` or `--box9-ang`.
  Either `.cms` route rotates the coordinates into the box frame and writes the output
  axis-aligned, so your `complex_box_vectors` stay the three edge **lengths**. (For an
  already-axis-aligned system, `--box-nm Lx,Ly,Lz` or the auto-read is enough.)
- `--center protein` — centers on the **protein**, so it stays whole and in the
  middle of the box while the bilayer and water tile around it. This is the right
  choice for a membrane protein: centering on the *membrane* (or on `complex`, whose
  center is dominated by the far more numerous lipid atoms) pushes the protein into a
  neighboring periodic image, where it shows up as a blob off to the side. Other
  options: `--center membrane` (lipids only), `--center complex` (everything except
  water/ions), `--center all`, `--center none`.
  **Centering is purely cosmetic** — it does not affect whether the run NaNs. The
  rotation, whole-molecule fix, and wrap below are what prevent the crash, and they
  happen regardless of `--center`.

Under the hood it builds each molecule from the topology bonds, makes it **whole**
(never per-residue, which would split the protein), centers the chosen group, then
images each whole molecule into the box.

### Read the printout

```
molecules: 71234  (6210 multi-atom, 210337/216337 atoms carry bonds)
rotated coordinates into the box frame (box was tilted 3.6 deg off the axes)
centered 'protein' group (6180 atoms) at the box centre
box (nm):          [12.246, 12.417, 13.913]  <- use these as complex_box_vectors
bonds re-joined:   5417  (molecules that were split across a face)
largest molecule extent (nm): [8.21, 7.94, 9.03]
atoms outside box: before 38.0%  ->  after 0.00%
wrote complex_wrapped.cif
```

- **`bonds re-joined:` > 0** confirms molecules were split across a face in the raw
  frame — that was (part of) the NaN.
- **`largest molecule extent` < box on every axis** — if instead it *errors* that a
  molecule is bigger than the box, your PDB is missing CONECT records for the lipids/
  ligand (it will also warn `most atoms have NO bonds`). See Troubleshooting.

> It is **normal** to still see protein loops and a few whole water/lipid molecules
> poking through the box faces in VMD. Those are whole molecules straddling the
> periodic boundary — harmless. The only thing that must never happen is a molecule
> *cut* across a face, which `wrap-pbc` prevents.

---

## Step 3 — regenerate the JSON and run

Point `rbfe.yaml` at the wrapped structure and set the box to the printed lengths:

```yaml
protein_pdb: complex_wrapped.cif
complex_box_vectors: [12.246, 12.417, 13.913]
```

```bash
chemflow openfe rbfe generate rbfe.yaml
# then submit the complex leg as usual
```

---

## Step 4 — verify before spending queue time

The VMD view looking right does **not** prove it will run — judge by the step-0
energy. Re-run the force decomposition on the regenerated system: the top-atom
forces should have dropped from ~10⁸ kJ/mol/nm back to the normal ~10²–10³ range.
If they have, the quickrun will clear step 0.

---

## Troubleshooting

- **`most atoms have NO bonds` / residues come out as fragments** — the structure
  lost its connectivity. A `mae → pdb/cif` conversion often drops bonds, and OpenMM
  can't rebuild them for Desmond/OPLS atom names it doesn't recognize. With no bonds,
  `wrap-pbc` images atoms individually and tears residues apart. Add `--guess-bonds`
  so it perceives bonds from interatomic distance (covalent radii):

  ```bash
  alchemforge wrap-pbc complex.cif complex_wrapped.cif \
      --box9-ang "…" --center protein --guess-bonds
  ```

  This needs the frame's molecules **intact in Cartesian space** (a Desmond
  `-out.cms` last frame is). It cannot reconnect a molecule that is *already* split
  across a face — for that, use the `.prmtop` route below.

- **CONECT/connectivity truly unavailable (or the frame is already split)** — wrap
  from the Amber topology instead, which has every bond explicitly and doesn't need
  distance perception:

  ```bash
  cpptraj -p system.prmtop <<'EOF'
  trajin simulation.nc <frame> <frame>
  autoimage
  trajout complex_wrapped.pdb pdb
  run
  EOF
  ```

- **"genuinely TRICLINIC … > 5 deg from perpendicular"** — your box is sheared, not
  just rotated. `wrap-pbc` only handles orthorhombic (rotated) boxes; use the
  `cpptraj autoimage` route above, which honours the full triclinic cell.

- **Still NaNs at step 0 after wrapping** — confirm the box lengths in `rbfe.yaml`
  match the `box (nm)` line `wrap-pbc` printed, and that `complex_wrapped.pdb` is
  the file `create_json` actually read (check the generated JSON path).
