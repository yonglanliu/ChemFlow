# Preparing a Desmond/Maestro membrane system for OpenFE RBFE

How to turn an equilibrated Desmond/Maestro membrane system into a structure OpenFE
RBFE can actually run — a **wrapped, box‑consistent mmCIF with the protein whole and
embedded in the bilayer**. Getting this wrong is the #1 cause of a complex‑leg crash:

```
Potential energy is NaN after 0 attempts of integration ...
Propagating replica 0 at state 0 resulted in a NaN!
```

That NaN is the *input structure*, not the alchemy, GPU, temperature, charge, or
surface tension. At state 0 (the physical endpoint) OpenMM images every atom into
the periodic box; if the frame isn't wrapped, atoms land on top of their periodic
images (two ions at ~0.5 Å → infinite Coulomb energy) and it NaNs before step 1.
A quick way to confirm this is the cause: count atoms outside the box in the saved
`nan-error-logs` state (see the last section) — a healthy system is ~0%, a broken
one is 12–90%.

---

## The rules

1. **Never `trj_align` or `trj_unwrap` the FEP input.** Both move atoms relative to
   the box and are the usual cause of this NaN:

   - `trj_unwrap` makes molecules continuous across frames → coordinates drift far
     outside the box (we saw 91% of atoms outside).
   - `trj_align` rigidly **rotates** the system while the box vectors stay
     axis‑aligned → the tilted slab no longer fits its cell.
     For RBFE the orientation is irrelevant (OpenFE builds each edge in its own box),
     so there is never a reason to align the FEP input.
2. **Center on the PROTEIN, then wrap whole molecules.** Centering on the *membrane*
   pushes the protein into a neighboring periodic image — it ends up as a separate
   blob beside the box instead of embedded in the bilayer. Centering on the protein
   keeps it whole and in the middle; the bilayer and water tile around it.
3. **Keep every molecule whole.** Wrapping per‑atom (or per‑residue for a
   multi‑residue protein) splits the protein across a face → a stretched peptide
   bond → NaN. Desmond's wrap keeps molecules whole; use it.
4. **Export mmCIF, not PDB.** A solvated membrane is usually >99,999 atoms; PDB's
   serial column overflows and Maestro writes Hybrid36 (`A000G`) that OpenMM can't
   read. mmCIF has no such limit, and `create_json` accepts it.

Also: **don't run `pdbfixer`** on a pre‑equilibrated membrane — unnecessary, and it
chokes on the serial overflow.

> It is **normal** for the bilayer to span the whole X/Y box and touch the faces —
> the membrane is periodic and tiles across the boundary. That is not "out of the
> box." The only thing that must not happen is a single molecule cut across a face.

---

## Step 1 — center on the protein and wrap

Operate on the Desmond `-out.cms` + trajectory (`_trj`). Centering is a
lattice‑preserving shift (unlike alignment), so it's safe:

```bash
$SCHRODINGER/run trj_center.py jobname-out.cms jobname_trj centered -asl "protein"
```

This writes `centered-out.cms` + `centered_trj` with the protein centered and whole
and everything wrapped around it. (Maestro GUI equivalent: Trajectory panel →
settings → Center on ASL `protein`, enable Wrap.)

If the protein happens to straddle the Z boundary after centering, that still means
center on `protein` — not on the membrane.

---

## Step 2 — extract the frame you want

`trj2mae.py` selects frames by a 0‑based slice. **Set `-align-asl ""`** so it does
**not** rotate, and `-extract-asl "all"` to keep the whole system:

```bash
# frame 68 (0-based) of the centered trajectory, no alignment, whole system
$SCHRODINGER/run trj2mae.py centered-out.cms centered_trj frame68 \
    -s 68:69:1 -extract-asl "all" -align-asl ""
#   (GUI player is 1-based: GUI frame 68 == index 67 -> use -s 67:68:1)

# convert to mmCIF for OpenFE
$SCHRODINGER/utilities/structconvert frame68*.mae* frame68.cif
```

`-align-asl ""` is essential — the default aligns each frame onto the input CMS (a
rotation), which reintroduces the tilt/NaN problem.

---

## Step 3 — verify in VMD before you run

Load `frame68.cif`, then in **Extensions → Tk Console**:

```tcl
package require pbctools
pbc set {122.46 124.17 139.13 90 90 90} -all   ;# your box in ANGSTROM (= complex_box_vectors * 10)
pbc box -center bb -width 3 -color blue         ;# draw the box centered on the system
measure minmax [atomselect top all]             ;# system extent (Angstrom)
```

What you should see:

- The protein is **embedded in the cyan bilayer**, inside the box — not a separate
  blob off to one side.
- The extent is ≈ the box (a little larger is fine — whole molecules straddle the
  faces). `pbc box -center bb` matters: the default draws the cell at the origin, so
  a centered system looks offset even when it's correct.

Numeric check (the real validity test is **no clashes**, not span ≤ box):

```bash
# writes nothing (checker mode): look only at the clash count
python scripts/membrane_prep/wrap_cif_for_openfe.py frame68.cif /dev/null --box-nm 12.246,12.417,13.913
#   atoms outside box: ~0-1%  (straddling molecules, fine)
#   minimum-image clashes (<0.9 A): 0   <-- must be 0
```

(`scripts/membrane_prep/` also has `wrap_pdb_for_openfe.py` for PDB inputs, and
`wrap_cif_for_openfe.py` can wrap directly too: `… in.cif out.cif --box-nm …`.)

If clashes > 0, a molecule is still split or the packing is off — re‑do Step 1 with
`-asl "protein"`, or minimize briefly before running.

---

## Step 4 — point OpenFE at it and regenerate the JSON

```yaml
# rbfe.yaml
protein_pdb: frame68.cif        # create_json accepts mmCIF (ProteinComponent.from_pdbx_file)
membrane_system: true
complex_box_vectors: [12.246, 12.417, 13.913]   # nm; matches the cif's cell
```

```bash
chemflow openfe rbfe validate  rbfe.yaml
chemflow openfe rbfe generate  rbfe.yaml   # REBUILDS json/ with the wrapped coordinates
# then resubmit quickrun
```

**Regenerating the JSON is mandatory** — `quickrun` reads the JSON, not the yaml, and
the JSON has the coordinates baked in. Editing `protein_pdb` without re‑running
`generate` changes nothing.

Protocol notes for the membrane complex leg:

- It is **pre‑solvated**, so the complex leg takes **no `SolventComponent`** (adding
  one double‑counts water).
- Use the **`MonteCarloMembraneBarostat`**; `surface_tension = 0` (tensionless) is
  the standard, correct setting — it is *not* a cause of NaNs.
- Matching the integrator temperature to your equilibration (e.g. 300 K) is fine but
  optional; a 300 K → 310 K gap is harmless.

---

## Triage if it still NaNs

Point this at the run's `nan-error-logs/` state — it settles whether the input is
the problem (it almost always is):

```python
import xml.etree.ElementTree as ET, numpy as np
pos=[]; box=[]; cur=None
for ev,el in ET.iterparse("iteration0-replica0-state0-state.xml", events=("start","end")):
    if ev=="start" and el.tag=="PeriodicBoxVectors": cur=el.tag
    if ev=="end":
        if el.tag=="Position": pos.append([float(el.get(k)) for k in "xyz"])
        elif el.tag in "ABC" and cur=="PeriodicBoxVectors": box.append([float(el.get(k)) for k in "xyz"])
        if el.tag=="PeriodicBoxVectors": cur=None
        el.clear()
pos=np.array(pos); L=np.diag(np.array(box))
print("atoms outside box: %.1f%%" % (100*((pos<0)|(pos>L)).any(1).mean()))
```

- **~0%** → the input is fine; look elsewhere (bad ligand parameters, a genuine clash
  — decompose the energy by force group; `NonbondedForce` NaN = overlap).
- **>a few %** → still unwrapped/aligned; an `align`/`unwrap` slipped back in, or you
  edited `protein_pdb` without regenerating the JSON. Redo Steps 1–4.

Ruled out by this single number: GPU/CUDA, temperature, net charge, and surface
tension never produce a step‑0 NaN — a frame that doesn't fit its box does.
