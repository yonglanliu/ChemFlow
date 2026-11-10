# Preparing a membrane system (`sol.cif`) for OpenFE RBFE

How to turn an equilibrated Maestro/Desmond membrane system into a structure that
OpenFE RBFE can actually run — i.e. a **wrapped, box‑consistent mmCIF**. Getting
this wrong is the #1 cause of a complex‑leg crash that looks like:

```
Potential energy is NaN after 0 attempts of integration ...
Propagating replica 0 at state 0 resulted in a NaN!
```

That NaN is almost always the *input structure*, not the alchemy: at state 0 (the
physical endpoint) OpenMM images the atoms into the periodic box, and if the
coordinates don't fit the box, atoms land on top of each other → infinite energy →
NaN before step 1.

---

## The three rules

1. **Use the raw equilibrated MD frame — do NOT `trj_align` or `trj_unwrap` it.**
   Both are analysis operations that move atoms relative to the box:
   - `trj_unwrap` (`-glue -temporal`) makes molecules continuous across frames →
     coordinates drift far outside the box.
   - `trj_align` rigidly **rotates** the system, but the box vectors stay
     axis‑aligned, so the rotated slab no longer fits its (unrotated) cell and
     OpenMM's minimum‑image uses the wrong lattice vectors.
   The raw MD frame is already consistent with its box — that's the state your
   simulation was stable in. For RBFE the orientation is irrelevant (OpenFE builds
   each edge in its own box), so there is no reason to align the FEP input.

2. **Export mmCIF, not PDB.** A solvated membrane is typically >99,999 atoms. PDB's
   atom‑serial column can't hold that, so Maestro writes Hybrid36 (`A000G`), which
   OpenMM/pdbfixer can't read (`ValueError: invalid literal for int() ... 'A000G'`).
   mmCIF has no such limit.

3. **Wrap molecules into the box** (whole molecules, imaged into the cell) and make
   sure the box vectors travel with the file. Verify `span ≤ box` before using it.

Also: **do not run `pdbfixer`** on a pre‑equilibrated membrane — it's meant for
adding missing atoms/H to raw structures, it's unnecessary here, and it chokes on
the serial overflow.

---

## Step 1 — get the raw frame + box out of Desmond

From the `-out.cms` (System Builder output is already wrapped) or the trajectory
frame you equilibrated, **without aligning or unwrapping**:

```bash
# convert the Desmond system to mmCIF directly
$SCHRODINGER/utilities/structconvert sol.cms sol.cif
```

or in Maestro: **File → Export Structures → PDBx/mmCIF (`.cif`)**.

If you need to pull a specific trajectory frame and re‑center on the protein
(centering is fine; it's a lattice‑preserving shift, unlike rotation):

```bash
$SCHRODINGER/run trj_center.py sol.cms sol_trj centered -asl "protein"   # check -h for wrap options
```

The Desmond box lives in the `.cms` (`chorus_box_*`). Confirm it survived the
export — the `.cif` must carry the cell (e.g. 122.46 × 124.17 × 139.13 Å).

---

## Step 2 — wrap into the box + verify

Wrap whole molecules into the cell and assert the system fits, using ParmEd
(tolerant reader; handles large systems):

```python
import parmed as pmd, numpy as np

s   = pmd.load_file("sol.cif")
L   = np.array(s.box[:3])              # box lengths (Angstrom)
xyz = s.coordinates.copy()

for res in s.residues:                 # make each residue whole, then image it in
    idx = [a.idx for a in res.atoms]
    d = xyz[idx] - xyz[idx[0]]
    d -= L * np.round(d / L)           # minimum image -> reassemble molecule
    xyz[idx] = xyz[idx[0]] + d
    xyz[idx] -= np.floor(xyz[idx].mean(0) / L) * L   # wrap COM into [0, L)

s.coordinates = xyz
span = xyz.max(0) - xyz.min(0)
assert (span <= L + 1e-3).all(), f"still not contained: span={span} box={L}"
s.save("sol_wrapped.cif", overwrite=True)
print("OK  span =", span, " box =", L)
```

Sanity‑check in VMD if you like — every atom must be inside the box:

```tcl
pbc get                               ;# the real cell
measure minmax [atomselect top all]   ;# span must be <= box, nothing poking out
```

If `span > box` on any axis, the file is still aligned/unwrapped — go back to the
raw frame.

---

## Step 3 — point OpenFE at it

Feed `sol_wrapped.cif` to `create_json`:

- Read it as PDBx/mmCIF (`ProteinComponent.from_pdbx_file`), not PDB.
- The membrane is **pre‑solvated**, so the complex leg takes **no `SolventComponent`**
  (adding one re‑solvates and double‑counts water).
- Use the **`MonteCarloMembraneBarostat`** for the membrane leg.
- You may set the integrator temperature to match your equilibration (e.g. 300 K),
  but a 300 K → 310 K difference is harmless and does **not** cause NaNs.

---

## Quick triage if it still NaNs

Load the saved state from the run's `nan-error-logs/` and check the geometry:

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
out=((pos<0)|(pos>L)).any(1).sum()
print("atoms outside box: %d/%d (%.1f%%)"%(out,len(pos),100*out/len(pos)))
print("span:", pos.max(0)-pos.min(0), " box:", L)
```

Any significant fraction outside the box (a healthy system is ~0%) means the input
still isn't wrapped/box‑consistent — i.e. an align/unwrap slipped back in.
