#!/usr/bin/env python
"""Wrap DCD trajectories into their periodic box, frame by frame.

    conda activate alchemforge
    python wrap_trajectory.py top.pdb md.dcd -o wrapped.dcd
    python wrap_trajectory.py top.pdb p1.dcd p2.dcd -o wrapped.dcd --drop-duplicate-first
    python wrap_trajectory.py top.pdb p*.dcd --separate

Imaging is done explicitly, not with mdtraj's ``image_molecules``, which images
BONDED molecules only — a multi-chain protein has no bonds between chains, so
each chain is imaged separately and the assembly scatters into the box corners.

Per frame, using that frame's own box:

  1. make every molecule whole by WALKING ITS BONDS (parent -> child). Unwrapping
     relative to a molecule's first atom instead is wrong for anything larger
     than half the box — a protein chain gets torn apart, which VMD then draws
     as long bonds stretching across the cell.
  2. reassemble the anchor group: image every anchor chain onto the LARGEST one,
     so a multimer becomes a single block.
  3. translate so the anchor group's centre is at the box centre.
  4. image all remaining molecules into the box around it.

Connectivity: bonds come from the topology. Residues that have NO bonds at all
(lipids/waters from a PDB with no CONECT records) are treated as single units,
so they are never split atom-by-atom. Use --guess-bonds to perceive bonds by
distance instead when even that is not enough.

Orthorhombic boxes only. Frames are streamed, so the trajectory need not fit in
memory.
"""

import argparse
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- connectivity
def _adjacency(top, n_atoms, guess_bonds, xyz0):
    """Bond adjacency: topology bonds, plus whole-residue links where absent.

    ``xyz0`` (the first frame) is used ONLY to perceive bonds by distance when
    --guess-bonds is set. Connectivity does not change over a trajectory. The
    WRAPPING itself always uses each frame's own box, never this one.
    """
    adj = defaultdict(list)
    bonded = set()
    for a, b in top.bonds:
        adj[a.index].append(b.index)
        adj[b.index].append(a.index)
        bonded.add(a.index)
        bonded.add(b.index)
    n_top = sum(len(v) for v in adj.values()) // 2

    # Residues with no bonds at all -> chain their atoms together so the residue
    # moves as one unit (a PDB carries no CONECT for lipids).
    n_res_linked = 0
    for res in top.residues:
        idx = [a.index for a in res.atoms]
        if len(idx) > 1 and not any(i in bonded for i in idx):
            for i, j in zip(idx, idx[1:]):
                adj[i].append(j)
                adj[j].append(i)
            n_res_linked += 1

    n_guessed = 0
    if guess_bonds:
        from scipy.spatial import cKDTree

        # ONLY for atoms that still have no connectivity. Guessing across the
        # whole system invents bonds between atoms that merely happen to be
        # close in the reference frame; BFS then drags those unrelated atoms
        # within half a box of each other, which shows up as multi-nm "bonds".
        free = np.array([i for i in range(n_atoms) if not adj.get(i)])
        if free.size:
            radii_all = np.array([_COVALENT_RADII.get(
                (a.element.symbol if a.element else "").upper(), 0.0)
                for a in top.atoms])
            sub = xyz0[free]
            radii = radii_all[free]
            ok = radii > 0
            if ok.any():
                pairs = cKDTree(sub).query_pairs(
                    r=2.0 * radii.max() * 1.3, output_type="ndarray")
                if len(pairs):
                    i, j = pairs[:, 0], pairs[:, 1]
                    d = np.linalg.norm(sub[i] - sub[j], axis=1)
                    keep = ok[i] & ok[j] & (d < 1.3 * (radii[i] + radii[j]))
                    for a, b in pairs[keep]:
                        ga, gb = int(free[a]), int(free[b])
                        adj[ga].append(gb)
                        adj[gb].append(ga)
                        n_guessed += 1
        print(f"  (guessing restricted to {free.size} atom(s) that had no bonds)")

    print(f"connectivity: {n_top} topology bonds"
          + (f", {n_res_linked} unbonded residue(s) linked as units" if n_res_linked else "")
          + (f", {n_guessed} guessed by distance" if n_guessed else ""))
    return adj


_COVALENT_RADII = {  # nm
    "H": 0.031, "B": 0.084, "C": 0.076, "N": 0.071, "O": 0.066, "F": 0.057,
    "SI": 0.111, "P": 0.107, "S": 0.105, "CL": 0.102, "SE": 0.120, "BR": 0.120,
    "I": 0.139,
}


def _components_and_edges(adj, n_atoms):
    """Connected components plus parent->child edges in BFS order."""
    seen = np.zeros(n_atoms, dtype=bool)
    comps, edges = [], []
    for start in range(n_atoms):
        if seen[start]:
            continue
        comp, bfs = [start], []
        seen[start] = True
        q = deque([start])
        while q:
            a = q.popleft()
            for b in adj.get(a, ()):
                if not seen[b]:
                    seen[b] = True
                    bfs.append((a, b))
                    comp.append(b)
                    q.append(b)
        comps.append(np.array(comp))
        edges.append(bfs)
    return comps, edges


# ---------------------------------------------------------------------- wrapping
def _wrap_frame(xyz, box, edges_flat, comps, anchor_ids, other_ids):
    # 1. make molecules whole by walking bonds (correct for any molecule size)
    for parent, child in edges_flat:
        xyz[child] -= box * np.round((xyz[child] - xyz[parent]) / box)

    if anchor_ids:
        # 2. reassemble the anchor group onto its largest member
        ref = xyz[anchor_ids[0]].mean(axis=0)
        for idx in anchor_ids:
            xyz[idx] -= box * np.round((xyz[idx].mean(axis=0) - ref) / box)
        # 3. centre it
        xyz += box / 2.0 - xyz[np.concatenate(anchor_ids)].mean(axis=0)

    # 4. image everything else into the box
    for idx in other_ids:
        xyz[idx] -= np.floor(xyz[idx].mean(axis=0) / box) * box
    return xyz


def _clock(seconds):
    m, s = divmod(int(max(seconds, 0)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _save_last_frame(frame, out_path):
    """Write the final wrapped frame.

    A PDB's serial column only holds 99,999 atoms. Past that we write hybrid-36
    serials (the standard large-system encoding) via alchemforge's writer, which
    VMD and Maestro both read. Note OpenMM CANNOT read hybrid-36 back, so for a
    large system use the .cif for the pipeline and the .pdb for viewing.
    """
    out_path = Path(out_path)
    n_atoms = frame.n_atoms
    is_cif = out_path.suffix.lower() in (".cif", ".pdbx", ".mmcif")

    if is_cif:
        from openmm import unit
        from openmm.app import PDBxFile

        topology = frame.topology.to_openmm()
        topology.setPeriodicBoxVectors(frame.unitcell_vectors[0] * unit.nanometer)
        with open(out_path, "w", encoding="utf-8") as handle:
            PDBxFile.writeFile(topology, frame.xyz[0] * unit.nanometer, handle,
                               keepIds=True)
    elif n_atoms >= 100000:
        from alchemforge.structure.wrap_pbc import _write_pdb_hybrid36

        _write_pdb_hybrid36(out_path, frame.topology.to_openmm(),
                            frame.xyz[0], frame.unitcell_lengths[0], np)
        print(f"  ({n_atoms} atoms > 99,999: written with hybrid-36 serials — "
              "readable by VMD/Maestro, but NOT by OpenMM)")
    else:
        frame.save(str(out_path))

    box = frame.unitcell_lengths[0]
    angles = frame.unitcell_angles[0]
    print(f"wrote final wrapped frame -> {out_path}  ({n_atoms} atoms)")
    print(f"  box (nm): [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}]"
          "   <- use as complex_box_vectors")

    # The box is the thing that gets lost when a structure is passed between
    # tools (a core CIF has no _cell, PDBFixer drops CRYST1, ...), so persist it
    # next to the frame in a form that is easy to paste or read back.
    box_txt = out_path.with_name(out_path.stem + "_box.txt")
    with open(box_txt, "w", encoding="utf-8") as handle:
        handle.write(
            f"# periodic box of {out_path.name} (final wrapped frame)\n"
            f"# lengths in nm; angles in degrees\n"
            f"box_nm: {box[0]:.6f},{box[1]:.6f},{box[2]:.6f}\n"
            f"box_angstrom: {box[0] * 10:.4f},{box[1] * 10:.4f},{box[2] * 10:.4f}\n"
            f"angles_deg: {angles[0]:.3f},{angles[1]:.3f},{angles[2]:.3f}\n"
            f"\n# paste into rbfe.yaml / abfe.yaml:\n"
            f"complex_box_vectors: [{box[0]:.6f}, {box[1]:.6f}, {box[2]:.6f}]\n"
            f"\n# or for alchemforge wrap-pbc:\n"
            f"#   --box-nm {box[0]:.6f},{box[1]:.6f},{box[2]:.6f}\n"
        )
    print(f"  box written to -> {box_txt}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("topology")
    ap.add_argument("trajectories", nargs="+")
    ap.add_argument("-o", "--output")
    ap.add_argument("--separate", action="store_true")
    ap.add_argument("--suffix", default="_wrapped")
    ap.add_argument("--center", default="protein",
                    help="MDTraj selection held together and centred "
                         "(default: protein; 'none' = wrap only)")
    ap.add_argument("--guess-bonds", action="store_true",
                    help="also perceive bonds by distance from the first frame")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--drop-duplicate-first", action="store_true")
    ap.add_argument("--last-frame", metavar="FILE",
                    help="where to write the final wrapped frame "
                         "(default: <output>_last.pdb; give a .cif name if you "
                         "need OpenMM to read it back above 99,999 atoms)")
    ap.add_argument("--no-last-frame", action="store_true",
                    help="do not write the final wrapped frame")
    ap.add_argument("--check", action="store_true",
                    help="report the longest bond after wrapping (should be < 0.3 nm)")
    args = ap.parse_args()

    if not args.separate and not args.output:
        sys.exit("give -o/--output, or --separate")

    import mdtraj as md

    top = md.load_topology(args.topology)
    n_atoms = top.n_atoms
    first = next(md.iterload(str(args.trajectories[0]), top=top, chunk=1))
    adj = _adjacency(top, n_atoms, args.guess_bonds, first.xyz[0])
    comps, edges = _components_and_edges(adj, n_atoms)
    edges_flat = [e for group in edges for e in group]
    print(f"{args.topology}: {n_atoms} atoms, {len(comps)} molecules")

    anchor_ids, other_ids = [], comps
    if args.center and args.center.lower() != "none":
        chosen = set(top.select(args.center).tolist())
        if not chosen:
            sys.exit(f"--center '{args.center}' selected no atoms")
        anchor_ids = [c for c in comps if chosen.intersection(c.tolist())]
        other_ids = [c for c in comps if not chosen.intersection(c.tolist())]
        anchor_ids.sort(key=len, reverse=True)
        print(f"anchor '{args.center}': {len(anchor_ids)} molecule(s), "
              f"{sum(len(i) for i in anchor_ids)} atoms "
              f"(largest {len(anchor_ids[0])} atoms = reference)")

    bond_pairs = np.array(edges_flat) if (args.check and edges_flat) else None

    def run(paths, out_path):
        total, last, t0, worst = 0, None, time.time(), 0.0
        worst_pair = None
        expected = 0
        for p in paths:
            try:
                with md.formats.DCDTrajectoryFile(str(p)) as fh:
                    expected += len(fh)
            except Exception:  # noqa: BLE001
                expected = 0
                break
        tty = sys.stdout.isatty()
        with md.formats.DCDTrajectoryFile(str(out_path), "w") as writer:
            for n, path in enumerate(paths):
                first_chunk = True
                for chunk in md.iterload(str(path), top=top, chunk=args.chunk):
                    if chunk.unitcell_lengths is None:
                        sys.exit(f"{path} has no box information.")
                    if np.any(np.abs(chunk.unitcell_angles - 90.0) > 1.0):
                        sys.exit("non-orthorhombic box; this handles orthorhombic only.")
                    if args.drop_duplicate_first and n > 0 and first_chunk:
                        chunk = chunk[1:]
                        if chunk.n_frames == 0:
                            continue
                    first_chunk = False
                    for i in range(chunk.n_frames):
                        _wrap_frame(chunk.xyz[i], chunk.unitcell_lengths[i],
                                    edges_flat, comps, anchor_ids, other_ids)
                        if bond_pairs is not None:
                            d = np.linalg.norm(
                                chunk.xyz[i][bond_pairs[:, 0]]
                                - chunk.xyz[i][bond_pairs[:, 1]], axis=1)
                            k = int(d.argmax())
                            if float(d[k]) > worst:
                                worst = float(d[k])
                                worst_pair = (int(bond_pairs[k, 0]),
                                              int(bond_pairs[k, 1]))
                    writer.write(chunk.xyz * 10.0,
                                 cell_lengths=chunk.unitcell_lengths * 10.0,
                                 cell_angles=chunk.unitcell_angles)
                    total += chunk.n_frames
                    last = chunk[-1]
                    el = time.time() - t0
                    rate = total / el if el else 0
                    if expected:
                        frac = total / expected
                        eta = (expected - total) / rate if rate else 0
                        msg = (f"wrapping [{'#' * int(28 * frac)}"
                               f"{'.' * (28 - int(28 * frac))}] "
                               f"{total}/{expected} ({frac * 100:5.1f}%) "
                               f"{rate:.1f} fr/s ETA {_clock(eta)}")
                    else:
                        msg = f"wrapping {total} frames ({rate:.1f} fr/s)"
                    print(("\r" + msg) if tty else msg, end="" if tty else "\n",
                          flush=True)
        if tty:
            print()
        print(f"wrote {out_path}: {total} frames in {_clock(time.time() - t0)}")
        if bond_pairs is not None:
            verdict = "OK" if worst < 0.3 else "STILL BROKEN"
            print(f"  longest bond after wrapping: {worst:.3f} nm  -> {verdict}")
            if worst_pair and worst >= 0.3:
                a, b = (top.atom(worst_pair[0]), top.atom(worst_pair[1]))
                print(f"    offender: {a} (res {a.residue}) -- {b} (res {b.residue})")
                if a.residue != b.residue:
                    print("    the two atoms are in DIFFERENT residues, so this is "
                          "almost certainly a SPURIOUS bond. Re-run WITHOUT "
                          "--guess-bonds: the topology already has the real bonds.")
        return last

    last = None
    if args.separate:
        for path in args.trajectories:
            last = run([path], Path(path).with_name(
                Path(path).stem + args.suffix + ".dcd"))
    else:
        last = run(args.trajectories, args.output)

    if last is not None and not args.no_last_frame:
        if args.last_frame:
            target = Path(args.last_frame)
        else:
            # Default alongside the wrapped trajectory. PDB regardless of size:
            # hybrid-36 serials cover systems over 99,999 atoms.
            stem = Path(args.output if args.output
                        else args.trajectories[-1]).with_suffix("")
            target = Path(f"{stem}_last.pdb")
        _save_last_frame(last, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
