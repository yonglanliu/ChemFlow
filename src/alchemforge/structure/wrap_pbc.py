"""Wrap a membrane/solvated structure into its periodic box for OpenFE.

``alchemforge wrap-pbc in.pdb out.pdb --box-nm Lx,Ly,Lz``
``alchemforge wrap-pbc in.cif out.cif --box9-ang "ax,ay,az, bx,by,bz, cx,cy,cz"``

Reads and writes PDB or mmCIF (dispatched on the file extension). Use **mmCIF**
(``.cif``) for a full solvated membrane system: it has no 99,999-atom serial limit,
whereas PDB overflows and writes Hybrid36 serials OpenMM/OpenFE can't read back.

Fixes the OpenFE complex-leg step-0 NaN ("Potential energy is NaN after 0
attempts of integration"). Three imaging defects cause it, all handled here:

  1. UNWRAPPED frame  -> atoms sit outside the box and get imaged onto their
     periodic partners -> nonbonded overlaps (ion-ion clashes) -> inf energy.
  2. SPLIT molecules  -> a molecule straddles a box face, so part of it is imaged
     to the far side -> a bond is stretched ~one box length -> huge HarmonicBond/
     Angle forces -> inf energy. (A per-RESIDUE wrap of a protein does exactly
     this.) This wraps per WHOLE MOLECULE (connected component over the bonds).
  3. ROTATED box  -> a Desmond/Maestro "chorus box" is often orthorhombic but
     TILTED in the lab frame (its edge vectors are mutually perpendicular yet not
     aligned to x/y/z). Wrapping straight along x/y/z then shears the system into
     diagonally-offset blocks (a visible "staircase") and leaves bad contacts at
     the seam. With --box9-ang the coordinates are first ROTATED into the box
     frame (edges -> x/y/z) so an orthorhombic wrap tiles correctly.

The membrane box is usually NOT in the PDB (no CRYST1). OpenFE takes the box from
``complex_box_vectors`` in rbfe.yaml — pass the SAME orthorhombic lengths with
--box-nm. If your source box is tilted (Schrodinger chorus_box with non-zero
off-diagonals), pass the full 3x3 with --box9-ang instead; the output is written
axis-aligned so ``complex_box_vectors`` stays the three edge LENGTHS.

After wrapping, point ``protein_pdb`` at the output and regenerate the JSON:

    alchemforge wrap-pbc complex.pdb complex_wrapped.pdb \\
        --box9-ang "122.13,2.57,-8.64, -2.77,124.12,-2.21, 9.77,2.69,138.76"
    # rbfe.yaml:  protein_pdb: complex_wrapped.pdb
    #             complex_box_vectors: [12.246, 12.417, 13.913]   # the edge lengths
    chemflow openfe rbfe generate rbfe.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

# Residue names used to pick a centering group. WATER/IONS are enumerated fully so
# "complex" = NOT(water or ion) robustly captures protein + membrane (+ ligand /
# cofactor) no matter how the lipids are named.
_WATER_RESNAMES = {
    "HOH", "WAT", "SOL", "TIP", "TIP3", "TIP4", "TIP5", "T3P", "T4P", "SPC",
    "SPCE", "OPC", "OPC3", "DOD", "H2O",
}
_ION_RESNAMES = {
    "NA", "NA+", "SOD", "CL", "CL-", "CLA", "K", "K+", "POT", "MG", "MG2", "CAL",
    "CA", "CA2", "ZN", "ZN2", "MN", "FE", "FE2", "FE3", "CU", "LI", "RB", "CS",
    "BR", "IOD", "I", "IB", "F",
}
_AMINO_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH", "LYN",
    "ACE", "NME", "NMA", "NH2",
}
# Common lipid residue names (CHARMM-GUI / AMBER Lipid / Slipids). Not exhaustive —
# use the "complex" group when in doubt.
_LIPID_RESNAMES = {
    "POPC", "POPE", "POPG", "POPS", "POPI", "POPA", "POPT", "DPPC", "DPPE",
    "DPPG", "DPPS", "DOPC", "DOPE", "DOPS", "DOPG", "DMPC", "DMPE", "DLPC",
    "DSPC", "DSPE", "SOPC", "SAPC", "SAPE", "SAPI", "PLPC", "PLPE", "PSM",
    "SSM", "CHL1", "CHOL", "CHL", "CHO", "PA", "PC", "PE", "PG", "PS", "PI",
    "OL", "PX", "SM",
}


# Covalent radii (nm) for distance-based bond perception. Only elements that form
# covalent bonds are listed; metals / monatomic ions are omitted on purpose so they
# stay singleton "molecules" (and never get glued to a coordinating water).
_COVALENT_RADII_NM = {
    "H": 0.031, "B": 0.084, "C": 0.076, "N": 0.071, "O": 0.066, "F": 0.057,
    "SI": 0.111, "P": 0.107, "S": 0.105, "CL": 0.102, "SE": 0.120, "BR": 0.120,
    "I": 0.139,
}


def _guess_bonds(pos_nm, elements, np, scale=1.3):
    """Perceive bonds from interatomic distance using covalent radii (nm).

    Two atoms are bonded if their separation is below ``scale * (r_i + r_j)``. This
    excludes H-bonds and van-der-Waals contacts but catches every real covalent
    bond, as long as molecules are intact in Cartesian space (not split across the
    periodic boundary). Returns a list of (i, j) index pairs. Uses SciPy's KD-tree
    when available, otherwise a pure-NumPy cell list.
    """
    radii = np.array(
        [_COVALENT_RADII_NM.get((e or "").upper(), 0.0) for e in elements]
    )
    bondable = radii > 0
    if not bondable.any():
        return []
    rmax = float(radii.max())
    cutoff = 2.0 * rmax * scale

    def _accept(i, j):
        d = np.linalg.norm(pos_nm[i] - pos_nm[j], axis=1)
        thr = scale * (radii[i] + radii[j])
        keep = bondable[i] & bondable[j] & (d < thr)
        return [(int(a), int(b)) for a, b in zip(i[keep], j[keep])]

    try:
        from scipy.spatial import cKDTree

        pairs = cKDTree(pos_nm).query_pairs(r=cutoff, output_type="ndarray")
        if len(pairs) == 0:
            return []
        return _accept(pairs[:, 0], pairs[:, 1])
    except ImportError:
        pass

    # NumPy cell list: bucket atoms into cutoff-sized cells, check 27-cell neighbourhood.
    from collections import defaultdict

    idx = np.where(bondable)[0]
    cell = np.floor(pos_nm[idx] / cutoff).astype(np.int64)
    grid = defaultdict(list)
    for local, atom in enumerate(idx):
        grid[(int(cell[local, 0]), int(cell[local, 1]), int(cell[local, 2]))].append(atom)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    out = []
    for local, a in enumerate(idx):
        ca = cell[local]
        cand = []
        for dx, dy, dz in offsets:
            cand.extend(grid.get((int(ca[0]) + dx, int(ca[1]) + dy, int(ca[2]) + dz), ()))
        cand = np.array([b for b in cand if b > a], dtype=np.int64)
        if cand.size:
            out.extend(_accept(np.full(cand.shape, a), cand))
    return out



def _hy36(width: int, value: int) -> str:
    """Encode a serial/resSeq as hybrid-36 (PDB's standard >99,999 extension)."""
    if value < 10 ** width:
        return "%*d" % (width, value)
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    value -= 10 ** width
    if value < 26 * 36 ** (width - 1):
        value += 10 * 36 ** (width - 1)
        out = ""
        while value:
            out = digits[value % 36] + out
            value //= 36
        return out.rjust(width, "0")
    return "*" * width  # beyond hybrid-36 range (astronomically large)


def _write_pdb_hybrid36(path, topology, pos_nm, box_nm, np) -> None:
    """Write a PDB with hybrid-36 serials/resSeq + a CRYST1 box.

    OpenMM's own PDB writer garbles systems over 99,999 atoms (serial overflow),
    which both breaks viewers and errors on export. This writer uses hybrid-36 (the
    standard large-system encoding VMD and Maestro both read), so a full membrane
    system round-trips cleanly for visualization.
    """
    pos_a = np.asarray(pos_nm) * 10.0  # nm -> Angstrom
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n"
            % (box_nm[0] * 10, box_nm[1] * 10, box_nm[2] * 10, 90.0, 90.0, 90.0)
        )
        for i, atom in enumerate(topology.atoms()):
            res = atom.residue
            name = atom.name
            aname = (
                name[:4].ljust(4)
                if (len(name) >= 4 or name[:1].isdigit())
                else (" " + name).ljust(4)
            )
            chain = (res.chain.id or "A")[:1] or "A"
            try:
                resseq = _hy36(4, int(res.id))
            except (TypeError, ValueError):
                resseq = "   1"
            elem = (atom.element.symbol if atom.element is not None else "").rjust(2)
            record = (
                "HETATM"
                if res.name in _WATER_RESNAMES or res.name in _ION_RESNAMES
                else "ATOM"
            )
            handle.write(
                "%-6s%5s %4s %-4s%1s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n"
                % (
                    record, _hy36(5, i + 1), aname, res.name[:4].ljust(4), chain,
                    resseq, pos_a[i, 0], pos_a[i, 1], pos_a[i, 2], 0.0, 0.0, elem,
                )
            )
        handle.write("END\n")


def _connected_components(n_atoms, bonds):
    """Group atoms into molecules (connected components) over the bond graph.

    Returns (components, bfs_edges): ``components`` is a list of atom-index lists;
    ``bfs_edges`` is the matching list of parent->child edges in BFS order, so a
    parent is always placed before its children when unwrapping. Atoms with no
    bonds (ions, lone waters) become singleton molecules.
    """
    adjacency = [[] for _ in range(n_atoms)]
    for first, second in bonds:
        adjacency[first].append(second)
        adjacency[second].append(first)
    seen = [False] * n_atoms
    components = []
    bfs_edges = []
    for start in range(n_atoms):
        if seen[start]:
            continue
        component = []
        edges = []
        seen[start] = True
        queue = deque([start])
        while queue:
            atom = queue.popleft()
            component.append(atom)
            for neighbour in adjacency[atom]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    edges.append((atom, neighbour))
                    queue.append(neighbour)
        components.append(component)
        bfs_edges.append(edges)
    return components, bfs_edges


def _lengths_and_rotation(box_vectors, np):
    """From 3 lattice vectors (rows, nm) return (lengths, R, max_off_deg).

    ``R`` has the unit edge vectors as COLUMNS, so ``pos @ R`` expresses the
    coordinates in the box frame (edges -> x/y/z). ``max_off_deg`` is how far the
    three edges are from being mutually perpendicular (0 = orthorhombic).
    """
    B = np.asarray(box_vectors, dtype=float).reshape(3, 3)
    lengths = np.linalg.norm(B, axis=1)
    if (lengths <= 0).any():
        raise ValueError(f"Box edge with non-positive length: {lengths}")
    units = B / lengths[:, None]
    # angles between edges; 90 deg apart = orthorhombic
    offs = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        cos = float(np.clip(np.dot(units[i], units[j]), -1.0, 1.0))
        offs.append(abs(90.0 - np.degrees(np.arccos(cos))))
    R = units.T  # columns = unit edge vectors
    return lengths, R, max(offs)


def _box_from_cms(cms_path):
    """Read the 9-component chorus box (nm, 3x3 rows) from a Desmond ``-out.cms``.

    Shells out to ``$SCHRODINGER/run python3`` because reading a .cms needs the
    Schrodinger toolkit (not the OpenMM env wrap-pbc runs in). Returns a 3x3 list of
    lattice vectors in nm, preserving Desmond's lab-frame tilt.
    """
    import os
    import subprocess

    cms_path = str(Path(cms_path).expanduser())
    if not Path(cms_path).is_file():
        raise ValueError(f"CMS file does not exist: {cms_path}")
    schrodinger = os.environ.get("SCHRODINGER")
    runner = str(Path(schrodinger) / "run") if schrodinger else "run"
    code = (
        "import sys;"
        "from schrodinger.application.desmond.packages import topo;"
        "_,c=topo.read_cms(sys.argv[1]);"
        "print(' '.join('%.6f'%v for v in c.box))"
    )
    try:
        out = subprocess.check_output(
            [runner, "python3", "-c", code, cms_path], text=True, stderr=subprocess.STDOUT
        )
    except FileNotFoundError as error:
        raise ValueError(
            "Could not run Schrodinger's 'run'. Set $SCHRODINGER (so "
            "$SCHRODINGER/run is found) to read the box from a .cms."
        ) from error
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Reading the box from {cms_path} failed:\n{error.output}") from error
    vals = [float(x) for x in out.split()]
    if len(vals) != 9:
        raise ValueError(f"Expected 9 chorus-box numbers from the .cms, got {len(vals)}: {vals}")
    return [[v / 10.0 for v in vals[i : i + 3]] for i in (0, 3, 6)]  # Angstrom -> nm


def _read_mae(path, np):
    """Read a Maestro/Desmond .mae via Schrodinger: atoms + bonds + chorus box.

    Shells out to ``$SCHRODINGER/run python3`` (a .mae needs the Schrodinger toolkit),
    dumps the structure to JSON, and builds an OpenMM Topology from it. A .mae carries
    explicit bonds and the chorus box, so no bond-guessing or separate box is needed.
    Returns (topology, positions_nm, box_vectors_nm_or_None).
    """
    import json
    import os
    import subprocess
    import tempfile

    from openmm.app import Element, Topology

    path = str(Path(path).expanduser())
    schrodinger = os.environ.get("SCHRODINGER")
    runner = str(Path(schrodinger) / "run") if schrodinger else "run"
    code = (
        "import sys, json\n"
        "from schrodinger import structure\n"
        "st = structure.StructureReader.read(sys.argv[1])\n"
        "atoms = []\n"
        "for a in st.atom:\n"
        "    r = a.getResidue()\n"
        "    nm = (a.pdbname or a.element or 'X').strip() or (a.element or 'X')\n"
        "    atoms.append([a.element, nm, a.x, a.y, a.z, (r.pdbres or 'UNK').strip() or 'UNK', (a.chain or 'A'), int(a.resnum)])\n"
        "bonds = [[b.atom1.index - 1, b.atom2.index - 1] for b in st.bond]\n"
        "p = st.property\n"
        "ks = ['r_chorus_box_ax','r_chorus_box_ay','r_chorus_box_az','r_chorus_box_bx','r_chorus_box_by','r_chorus_box_bz','r_chorus_box_cx','r_chorus_box_cy','r_chorus_box_cz']\n"
        "box = [p[k] for k in ks] if all(k in p for k in ks) else None\n"
        "json.dump({'atoms': atoms, 'bonds': bonds, 'box': box}, open(sys.argv[2], 'w'))\n"
    )
    with tempfile.TemporaryDirectory() as td:
        dump = os.path.join(td, "mae_dump.json")
        try:
            subprocess.check_output(
                [runner, "python3", "-c", code, path, dump],
                text=True, stderr=subprocess.STDOUT,
            )
        except FileNotFoundError as error:
            raise ValueError(
                "Could not run Schrodinger's 'run'. Set $SCHRODINGER (so $SCHRODINGER/run "
                "is found) to read a .mae."
            ) from error
        except subprocess.CalledProcessError as error:
            raise ValueError(f"Reading {path} with Schrodinger failed:\n{error.output}") from error
        with open(dump) as fh:
            data = json.load(fh)

    atoms, bonds, boxv = data["atoms"], data["bonds"], data["box"]
    top = Topology()
    atom_objs = []
    chain = None
    residue = None
    prev_chain_id = object()  # sentinel so the first atom always opens a chain
    prev_res_key = None
    for elem, name, _x, _y, _z, resn, chid, resnum in atoms:
        chid = chid or "A"
        # Start a NEW chain whenever the id changes in atom order — Maestro reuses
        # chain ids across non-adjacent blocks, but OpenMM needs contiguous chains.
        if chid != prev_chain_id:
            chain = top.addChain(chid)
            prev_chain_id = chid
            prev_res_key = None
        res_key = (resnum, resn)
        if res_key != prev_res_key:
            residue = top.addResidue(resn, chain, id=str(resnum))
            prev_res_key = res_key
        try:
            element = Element.getBySymbol(elem)
        except Exception:
            element = None
        atom_objs.append(top.addAtom(name, element, residue))
    for i, j in bonds:
        top.addBond(atom_objs[i], atom_objs[j])

    pos = np.array([[a[2], a[3], a[4]] for a in atoms], dtype=float) / 10.0  # A -> nm
    box_vectors_nm = (
        [[v / 10.0 for v in boxv[k : k + 3]] for k in (0, 3, 6)] if boxv else None
    )
    return top, pos, box_vectors_nm


def wrap_pdb(
    source: Path,
    output: Path,
    box_nm: tuple[float, float, float] | None = None,
    box_vectors_nm=None,
    center: str = "none",
    guess_bonds: bool = False,
) -> dict:
    """Make every molecule whole and image it into the orthorhombic box.

    Exactly one of ``box_nm`` (three axis-aligned lengths) or ``box_vectors_nm``
    (a 3x3 of lattice vectors, rows, nm — may be tilted) should be given. A tilted
    box triggers a rigid rotation of the coordinates into the box frame before
    wrapping. ``center`` selects a group to place at the box centre before the final
    wrap (``none`` | ``membrane`` | ``protein`` | ``complex`` | ``all``); ``complex``
    = everything except water and ions (i.e. protein + membrane + ligand/cofactor),
    which is what you want for a membrane system so the bilayer sits in the middle
    with water split symmetrically. Returns a small report dict.
    """
    try:
        import numpy as np
        from openmm import Vec3, unit
        from openmm.app import PDBFile, PDBxFile
    except ImportError as error:  # pragma: no cover - env guard
        raise ValueError(
            "OpenMM and NumPy are required for wrap-pbc; activate the "
            "AlchemForge environment"
        ) from error

    source = Path(source).expanduser()
    output = Path(output).expanduser()
    if not source.is_file():
        raise ValueError(f"Input structure does not exist: {source}")

    # mmCIF (PDBx) has no 99,999-atom serial limit — required for full solvated
    # membrane systems. .mae is read via Schrodinger (carries bonds + chorus box).
    cif_suffixes = {".cif", ".pdbx", ".mmcif"}
    name_lower = source.name.lower()
    is_mae = name_lower.endswith((".mae", ".maegz", ".mae.gz"))
    writer = PDBxFile if output.suffix.lower() in cif_suffixes else PDBFile

    mae_box_vectors_nm = None
    if is_mae:
        topology, pos, mae_box_vectors_nm = _read_mae(source, np)
    else:
        reader = PDBxFile if source.suffix.lower() in cif_suffixes else PDBFile
        pdb = reader(str(source))
        topology = pdb.topology
        pos = np.array(pdb.getPositions().value_in_unit(unit.nanometer))
    n_atoms = len(pos)

    # A .mae's own chorus box is used unless the user overrode it on the command line.
    if box_nm is None and box_vectors_nm is None and mae_box_vectors_nm is not None:
        box_vectors_nm = mae_box_vectors_nm

    rotated_deg = 0.0
    if box_vectors_nm is not None:
        box, R, max_off = _lengths_and_rotation(box_vectors_nm, np)
        if max_off > 5.0:
            raise ValueError(
                f"Box edges are {max_off:.1f} deg from perpendicular — this is a "
                "genuinely TRICLINIC (sheared) box, not a rotated orthorhombic one. "
                "wrap-pbc only handles orthorhombic boxes; wrap from the .prmtop with "
                "cpptraj `autoimage` (it honours the full triclinic cell) instead."
            )
        if max_off > 0.05:  # tilted -> rotate coordinates into the box frame
            pos = pos @ R
            rotated_deg = max_off
    elif box_nm is not None:
        box = np.asarray(box_nm, dtype=float)
    else:
        vectors = topology.getPeriodicBoxVectors()
        if vectors is None:
            raise ValueError(
                "No periodic box found (no CRYST1 / _cell, and the .mae had no chorus "
                "box). Pass --box-nm Lx,Ly,Lz, --box9-ang for a tilted chorus box, or "
                "--box-from-cms to read it from the Desmond -out.cms."
            )
        file_vectors = np.array(vectors.value_in_unit(unit.nanometer))
        box, R, max_off = _lengths_and_rotation(file_vectors, np)
        if max_off > 5.0:
            raise ValueError(
                f"File box edges are {max_off:.1f} deg from perpendicular (triclinic). "
                "wrap-pbc only handles orthorhombic boxes; use cpptraj `autoimage`."
            )
        if max_off > 0.05:
            pos = pos @ R
            rotated_deg = max_off
    if box.shape != (3,) or (box <= 0).any():
        raise ValueError(f"Box must be three positive lengths (nm); got {box}")

    bonds = [(bond[0].index, bond[1].index) for bond in topology.bonds()]
    topology_bonds = len(bonds)
    guessed_bonds = 0
    # If connectivity is missing (converted mae/cif often drops bonds, or Desmond/OPLS
    # atom names don't match OpenMM templates), perceive bonds by distance so molecules
    # don't fragment. Auto-triggers when almost nothing is bonded; forced by guess_bonds.
    bonded_frac = (2 * topology_bonds) / max(n_atoms, 1)
    if guess_bonds or bonded_frac < 0.3:
        elements = [
            (atom.element.symbol if atom.element is not None else None)
            for atom in topology.atoms()
        ]
        extra = _guess_bonds(pos, elements, np)
        if extra:
            existing = {(min(a, b), max(a, b)) for a, b in bonds}
            for a, b in extra:
                key = (min(a, b), max(a, b))
                if key not in existing:
                    existing.add(key)
                    bonds.append((a, b))
                    guessed_bonds += 1

    before = float(((pos < 0) | (pos > box)).any(axis=1).mean() * 100)

    components, bfs_edges = _connected_components(n_atoms, bonds)
    multi = sum(1 for component in components if len(component) > 1)
    bonded_atoms = n_atoms - sum(1 for component in components if len(component) == 1)

    # 1) make each molecule whole: walk bonds parent->child, image child onto parent.
    bonds_rejoined = 0
    for edges in bfs_edges:
        for parent, child in edges:
            shift = box * np.round((pos[child] - pos[parent]) / box)
            if np.any(shift):
                bonds_rejoined += 1
            pos[child] -= shift

    # 1b) optionally translate a chosen group to the box centre (whole molecules
    #     stay whole; the final wrap then fills the box symmetrically around it).
    centered_on = "none"
    centered_atoms = 0
    if center != "none":
        resnames = np.array(
            [atom.residue.name.strip().upper() for atom in topology.atoms()]
        )
        is_water = np.isin(resnames, list(_WATER_RESNAMES))
        is_ion = np.isin(resnames, list(_ION_RESNAMES))
        if center == "membrane":
            sel = np.isin(resnames, list(_LIPID_RESNAMES))
        elif center == "protein":
            sel = np.isin(resnames, list(_AMINO_RESNAMES))
        elif center == "complex":
            sel = ~(is_water | is_ion)
        elif center == "all":
            sel = np.ones(n_atoms, dtype=bool)
        else:
            raise ValueError(f"Unknown --center group: {center}")
        if not sel.any():
            raise ValueError(
                f"--center {center} selected 0 atoms (residue names did not match). "
                "Use --center complex (everything except water/ions) or check resnames."
            )
        group_center = pos[sel].mean(0)
        pos += box / 2.0 - group_center
        centered_on = center
        centered_atoms = int(sel.sum())

    # 2) image each whole molecule so its centroid sits inside the box.
    for component in components:
        idx = np.asarray(component)
        pos[idx] -= np.floor(pos[idx].mean(0) / box) * box

    after = float(((pos < 0) | (pos > box)).any(axis=1).mean() * 100)
    worst = np.zeros(3)
    for component in components:
        idx = np.asarray(component)
        worst = np.maximum(worst, pos[idx].max(0) - pos[idx].min(0))

    report = {
        "n_atoms": n_atoms,
        "molecules": len(components),
        "multi_atom_molecules": multi,
        "bonded_atoms": bonded_atoms,
        "box_nm": box.tolist(),
        "rotated_deg": rotated_deg,
        "centered_on": centered_on,
        "centered_atoms": centered_atoms,
        "topology_bonds": topology_bonds,
        "guessed_bonds": guessed_bonds,
        "bonds_rejoined": bonds_rejoined,
        "largest_molecule_extent_nm": worst.tolist(),
        "outside_before_pct": before,
        "outside_after_pct": after,
        "connectivity_missing": bonded_atoms < 0.3 * n_atoms,
    }

    if (worst > box + 1e-3).any():
        raise ValueError(
            f"A molecule is still larger than the box {box.tolist()} after unwrap. "
            "Either the box is wrong, or connectivity is missing from the PDB so a "
            "molecule could not be made whole (add CONECT records, or wrap from the "
            ".prmtop with cpptraj `autoimage`)."
        )

    topology.setPeriodicBoxVectors(
        (Vec3(box[0], 0, 0), Vec3(0, box[1], 0), Vec3(0, 0, box[2])) * unit.nanometer
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if writer is PDBFile:
        # Our own hybrid-36 writer — OpenMM's overflows past 99,999 atoms and the
        # result won't open in VMD/Maestro. (mmCIF goes through PDBxFile below.)
        _write_pdb_hybrid36(output, topology, pos, box, np)
    else:
        with open(output, "w", encoding="utf-8") as handle:
            writer.writeFile(topology, pos * unit.nanometer, handle, keepIds=True)
    return report


def _parse_floats(text: str, n: int, flag: str):
    parts = [p for p in text.replace(" ", ",").split(",") if p]
    if len(parts) != n:
        raise ValueError(f"{flag} expects {n} comma-separated numbers; got {len(parts)}")
    return [float(p) for p in parts]


def run_cli(args: argparse.Namespace) -> int:
    given = [bool(args.box_nm), bool(args.box9_ang), bool(args.box_from_cms)]
    if sum(given) > 1:
        raise ValueError("Pass only one of --box-nm, --box9-ang, or --box-from-cms")

    box_nm = None
    box_vectors_nm = None
    if args.box_from_cms:
        box_vectors_nm = _box_from_cms(args.box_from_cms)
    elif args.box9_ang:
        vals = _parse_floats(args.box9_ang, 9, "--box9-ang")
        # Schrodinger chorus_box is Angstrom, row-major (a, b, c); convert to nm.
        box_vectors_nm = [[v / 10.0 for v in vals[i : i + 3]] for i in (0, 3, 6)]
    elif args.box_nm:
        box_nm = tuple(_parse_floats(args.box_nm, 3, "--box-nm"))
    # else: wrap_pdb reads the box from the file's CRYST1 / _cell (axis-aligned only).

    report = wrap_pdb(
        Path(args.input),
        Path(args.output),
        box_nm=box_nm,
        box_vectors_nm=box_vectors_nm,
        center=args.center,
        guess_bonds=args.guess_bonds,
    )

    print(
        f"molecules: {report['molecules']}  "
        f"({report['multi_atom_molecules']} multi-atom, "
        f"{report['bonded_atoms']}/{report['n_atoms']} atoms carry bonds)"
    )
    if report["guessed_bonds"]:
        print(
            f"guessed {report['guessed_bonds']} bonds by distance "
            f"(topology had {report['topology_bonds']}) — connectivity was incomplete"
        )
    if report["connectivity_missing"]:
        print(
            "WARNING: most atoms still have NO bonds. Connectivity is missing (no "
            "CONECT / non-standard names) and distance perception could not recover "
            "it — molecules may fragment. Try --guess-bonds, or if the frame itself is "
            "split, wrap from the .prmtop with cpptraj `autoimage` instead.",
            file=sys.stderr,
        )
    if report["rotated_deg"]:
        print(
            f"rotated coordinates into the box frame "
            f"(box was tilted {report['rotated_deg']:.1f} deg off the axes)"
        )
    if report["centered_on"] != "none":
        print(
            f"centered '{report['centered_on']}' group "
            f"({report['centered_atoms']} atoms) at the box centre"
        )
    box_str = ", ".join(f"{v:.3f}" for v in report["box_nm"])
    extent_str = ", ".join(f"{v:.3f}" for v in report["largest_molecule_extent_nm"])
    print(f"box (nm):          [{box_str}]  <- use these as complex_box_vectors")
    print(f"bonds re-joined:   {report['bonds_rejoined']}  (molecules that were split across a face)")
    print(f"largest molecule extent (nm): [{extent_str}]")
    print(
        f"atoms outside box: before {report['outside_before_pct']:.1f}%  ->  "
        f"after {report['outside_after_pct']:.2f}%  (want ~0)"
    )
    print(f"wrote {args.output}")
    print(
        "Next: set protein_pdb to this file in rbfe.yaml, set complex_box_vectors to the "
        "box lengths above, then `chemflow openfe rbfe generate rbfe.yaml` and resubmit."
    )
    return 0


def add_wrap_pbc_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``alchemforge wrap-pbc`` command."""
    parser = subparsers.add_parser(
        "wrap-pbc",
        help="Wrap a membrane/solvated PDB into its box (whole molecules) for OpenFE",
        description=(
            "Make every molecule whole and image it into the orthorhombic periodic "
            "box, fixing the OpenFE complex-leg step-0 NaN caused by unwrapped frames, "
            "molecules split across a box face, or a tilted (rotated) chorus box."
        ),
    )
    parser.add_argument("input", metavar="IN.pdb/.cif/.mae", help="Input structure (PDB, mmCIF, or Maestro .mae; .mae/.cif carry 216k+ atoms)")
    parser.add_argument("output", metavar="OUT.pdb/.cif", help="Wrapped output; mmCIF (.cif) has no 99,999-atom limit")
    parser.add_argument(
        "--box-nm",
        metavar="Lx,Ly,Lz",
        help="Orthorhombic, axis-aligned box lengths in nm (= complex_box_vectors). "
        "Use this when your coordinates are already aligned to x/y/z.",
    )
    parser.add_argument(
        "--box9-ang",
        metavar='"ax,ay,az,bx,by,bz,cx,cy,cz"',
        help="Full 3x3 box in Angstrom (Schrodinger chorus_box, row-major). Use this "
        "when the box is tilted; coordinates are rotated into the box frame first.",
    )
    parser.add_argument(
        "--box-from-cms",
        metavar="system-out.cms",
        help="Read the box (incl. Desmond tilt) straight from a Desmond -out.cms via "
        "$SCHRODINGER/run — so you don't hand-copy the chorus numbers. Needs Schrodinger "
        "installed. If no box flag is given at all, the box is read from the input "
        "file's own CRYST1/_cell (axis-aligned systems only — a .cms/chorus tilt is not "
        "stored there).",
    )
    parser.add_argument(
        "--center",
        choices=["none", "membrane", "protein", "complex", "all"],
        default="none",
        help="Place a group at the box centre before the final wrap. For a membrane "
        "system use 'complex' (protein + membrane + ligand/cofactor, i.e. everything "
        "except water and ions) so the bilayer sits in the middle with water split "
        "symmetrically. Default: none.",
    )
    parser.add_argument(
        "--guess-bonds",
        action="store_true",
        help="Perceive bonds from interatomic distance (covalent radii) when the "
        "input lacks connectivity — e.g. a mae->pdb/cif conversion that dropped bonds, "
        "or Desmond/OPLS atom names OpenMM doesn't template. Prevents residues/molecules "
        "fragmenting. Needs the frame's molecules intact in Cartesian space (auto-enabled "
        "when almost nothing is bonded).",
    )
    parser.set_defaults(func=run_cli)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_wrap_pbc_parser(parser.add_subparsers(dest="_cmd", required=True))
    args = parser.parse_args(["wrap-pbc", *(argv or sys.argv[1:])])
    try:
        return int(run_cli(args) or 0)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
