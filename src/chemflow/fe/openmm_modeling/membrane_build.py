"""Membrane builder for OpenMM.

Wraps the user's :class:`MembraneModeller` (an ``openmm.app.Modeller`` subclass)
with a YAML-config-driven entrypoint. Two modes:

  * ``build``    - tile a bundled lipid patch around the protein and TRIM it to
                   protein-XY + padding (a minimal membrane), then solvate and
                   add ions. The output complex feeds ``relax-membrane``.
  * ``existing`` - the input already contains a membrane; only solvate the empty
                   space (preserving existing waters, deleting core waters) and
                   add ions.

Run headlessly with::

    python -m chemflow.fe.openmm_modeling.membrane_build --config membrane_build.yaml

The membrane normal is Z. The MembraneModeller class below is authored by
Yonglan Liu and incorporated verbatim; only the config/IO wrapper is added.
"""

from __future__ import division, absolute_import

__author__ = "Yonglan Liu"

import argparse
import gc
import os
from math import ceil, floor
from pathlib import Path
from typing import Iterable

import numpy as np
from openmm import Vec3, app
from openmm.app import Modeller, PDBFile, ForceField, Topology
from openmm.unit import nanometer, molar, is_quantity, norm
from copy import deepcopy
from openmm.app.internal import compiled
from collections import defaultdict, namedtuple
from openmm import System, Context, NonbondedForce, AmoebaVdwForce, AmoebaMultipoleForce, CustomNonbondedForce, HarmonicBondForce, HarmonicAngleForce, VerletIntegrator, LangevinIntegrator, LocalEnergyMinimizer
from openmm.app.modeller import elem, _CellList
from openmm.app.forcefield import AllBonds, CutoffNonPeriodic, CutoffPeriodic, DrudeGenerator, _getDataDirectories
import sys
import openmm.unit as unit

from chemflow.fe.openmm_modeling.membrane_build_config import load_config, validate_config


class MembraneModeller(Modeller):
    def __init__(self, topology, positions):
        super().__init__(topology, positions)

    @staticmethod
    def _residue_key(residue):
        chain_id = residue.chain.id if residue.chain is not None else ""
        return (chain_id, residue.id, residue.insertionCode, residue.name)

    @staticmethod
    def _to_nm_array(positions):
        if is_quantity(positions):
            return np.array(positions.value_in_unit(nanometer), dtype=float)
        else:
            # plain list of Vec3
            return np.array([[v.x, v.y, v.z] for v in positions], dtype=float)

    @staticmethod
    def _strip_names(ref_names):
        if ref_names is None:
            return []
        if isinstance(ref_names, str):
            return [n.strip() for n in ref_names.split(",") if n.strip()]
        return [n.strip() for n in ref_names if n.strip()]

    @staticmethod
    def _align_membrane_normal(pos_nm, p_indices):
        """Rotate the system so the bilayer normal points along +Z.

        The lipid phosphorus atoms form two parallel planes, so their thinnest
        principal axis IS the bilayer normal (spread across the membrane is a few
        nm versus ~12 nm in-plane). Rotate that axis onto Z about the membrane
        centroid, which leaves the system in place. Returns (positions, tilt_deg).
        """
        heads = pos_nm[p_indices]
        centre = heads.mean(axis=0)
        # Smallest-variance principal axis of the head-group positions.
        _vals, vecs = np.linalg.eigh(np.cov((heads - centre).T))
        normal = vecs[:, 0]
        if normal[2] < 0:
            normal = -normal

        target = np.array([0.0, 0.0, 1.0])
        axis = np.cross(normal, target)
        sin_a = float(np.linalg.norm(axis))
        cos_a = float(np.clip(np.dot(normal, target), -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(cos_a)))
        if sin_a < 1e-8:
            return pos_nm, tilt_deg
        axis = axis / sin_a
        # Rodrigues' rotation formula.
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        R = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)
        return (pos_nm - centre) @ R.T + centre, tilt_deg

    def addSolventWithExistingMembrane(
        self,
        forcefield: ForceField,
        solvent_model = "tip3p",
        pbc_x_vector = None, # Vec3(Lx, 0, 0) * nanometer
        pbc_y_vector = None, # Vec3(0, Ly, 0) * nanometer
        padding=1.0 * nanometer,
        boxsize=None, # Vec3(Lx, Ly, Lz) * nanometer
        box_vectors=None, # (Vec3(Lx, 0, 0) * nanometer, Vec3(0, Ly, 0) * nanometer, Vec3(0, 0, Lz) * nanometer)
        positiveIon="Na+",
        negativeIon="Cl-",
        ionicStrength=0 * molar,
        neutralize=True,
        write_structure=False,
        residueTemplates=dict(),
        align_membrane=True,
        water_lipid_cutoff=0.28,   # nm (2.8 A)
    ):
        """
        Add solvent to a system with an existing lipid membrane (and optional pre-existing water).

        This method adds water molecules and, if requested, neutralizing counterions and additional salt to the system.

        ---Periodic boundary conditions (PBC)---:

        If the input system already defines PBC, those box vectors are reused. Water will be added to fill that existing box.
        If you do not want to use the original PBC, you should remove them before calling this method.
        If the system does not define PBC, the box will be constructed as follows:
            1. If pbc_x_vector and pbc_y_vector are provided, they are used together with the specified padding (for the z dimension).
            2. Otherwise, the x and y dimensions are inferred from the membrane, and the z dimension is determined from the padding.
        The membrane extent is estimated from the positions of lipid head-group atoms (e.g., “P” atoms), which provides a more robust measure than using all lipid atoms.

        Once the periodic box is defined, the system is solvated within that box.

        ---Water handling---:

        Existing water molecules in the input system are preserved; the method only fills previously empty space with additional water.
        Water molecules that fall inside the membrane (e.g., in the hydrophobic core) are removed after solvation.

        ---Ion placement---:

        Counterions are added to neutralize the system if requested.
        Additional ions are added to achieve the specified ionic strength by replacing randomly selected water molecules.
        """
        # All accepted lipid residue names (3- and 4-letter conventions).
        lipid_names = [
            "POPC", "POPE", "POPG", "POPS", "POPI", "POPA", "DMPC", "DOPC",
            "DPPC", "DLPC", "DLPE", "DPPE", "DOPE", "DSPC", "POP", "DLP",
            "DMP", "DOP", "DPP",
        ]
        protein_names = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
            "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH",
            "LYN", "ACE", "NME", "NMA", "NH2",
        }
        # Save original waters so they are never deleted.
        original_water_keys = {
            self._residue_key(res)
            for res in self.topology.residues()
            if res.name in ["HOH","WAT"]
        }
        print(f'The original system contains {len(original_water_keys)} water molecules')

        # -------------------------------------------------
        # Positions as plain numpy arrays in nm.
        # -------------------------------------------------
        all_pos = self._to_nm_array(self.positions)
        sys_min = np.min(all_pos, axis=0)
        sys_max = np.max(all_pos, axis=0)
        sys_zmin = sys_min[2]
        sys_zmax = sys_max[2]

        # -------------------------------------------------
        # Lipid head-group (phosphate) z-range. Detect P by ELEMENT (robust to
        # atom-name variants) within lipid residues; fail clearly if none match.
        # -------------------------------------------------
        lipid_atom_indices = [
            atom.index
            for atom in self.topology.atoms()
            if atom.residue.name in lipid_names
            and atom.element is not None
            and atom.element.symbol == "P"
        ]
        if not lipid_atom_indices:
            found = sorted({a.residue.name for a in self.topology.atoms()})
            raise ValueError(
                "No lipid phosphate atoms found. Either the lipid residue names "
                f"aren't in the recognized list {lipid_names}, or there is no 'P' "
                f"atom. Residue names present: {found}. Add your lipid name to "
                "`lipid_names`."
            )

        # -------------------------------------------------
        # Align the bilayer normal with Z before anything else.
        # A Desmond frame is usually expressed in a tilted lab frame, so the
        # membrane sits at an angle. Everything downstream assumes normal = Z:
        # the core-water test, the box, and MonteCarloMembraneBarostat (which
        # scales XY and Z separately). Rotating here fixes all three at once.
        # -------------------------------------------------
        if align_membrane:
            all_pos, tilt_deg = self._align_membrane_normal(
                all_pos, lipid_atom_indices
            )
            if tilt_deg > 0.5:
                self.positions = all_pos * nanometer
                sys_min = np.min(all_pos, axis=0)
                sys_max = np.max(all_pos, axis=0)
                sys_zmin, sys_zmax = sys_min[2], sys_max[2]
                print(
                    f"Aligned the bilayer normal to Z (rotated {tilt_deg:.1f} deg)."
                )
            else:
                print(f"Bilayer normal already along Z ({tilt_deg:.1f} deg tilt).")

        # Use a TRIMMED range (1st-99th percentile), not min-max. In an unwrapped
        # frame a single lipid straddling the z face sits a whole box length away,
        # and min-max would then stretch the "membrane core" across nearly the
        # entire box — so every newly added water would be classed as core water
        # and deleted, leaving the bulk empty.
        lipid_pos = all_pos[lipid_atom_indices, :]
        lipid_zmin = float(np.percentile(lipid_pos[:, 2], 1.0))
        lipid_zmax = float(np.percentile(lipid_pos[:, 2], 99.0))
        raw_zmin = float(np.min(lipid_pos[:, 2]))
        raw_zmax = float(np.max(lipid_pos[:, 2]))
        print(
            f"Membrane core z-range: {lipid_zmin:.2f} to {lipid_zmax:.2f} nm "
            f"(trimmed; raw min-max was {raw_zmin:.2f} to {raw_zmax:.2f})"
        )
        if (raw_zmax - raw_zmin) > 1.5 * (lipid_zmax - lipid_zmin):
            print(
                "WARNING: lipid z min-max is far wider than the trimmed range — the "
                "frame is UNWRAPPED in z. The trimmed range is used so bulk water is "
                "not wrongly deleted, but you should wrap the frame or pass a box.",
                file=sys.stderr,
            )

        # -----------------------------------------------------
        #                   Determine PBC
        #
        # Precedence (first available wins per dimension):
        #   explicit box_vectors / boxsize  >  pbc_x/pbc_y (+padding for z)  >
        #   the input topology's own PBC  >  padding (z only; x/y must be known).
        # padding NO LONGER silently overrides an explicit or input-file box z —
        # so an equilibrated box is preserved for FEP consistency.
        # -----------------------------------------------------
        box_x, box_y, box_z = None, None, None

        # (0) Reuse the input system's PBC if it defines one (docstring behaviour).
        in_pbc = self.topology.getPeriodicBoxVectors()
        if in_pbc is not None:
            v = in_pbc.value_in_unit(nanometer)
            box_x, box_y, box_z = float(v[0][0]), float(v[1][1]), float(v[2][2])
            print(f'Reusing input PBC: box_x={box_x}, box_y={box_y}, box_z={box_z}')

        # (1) explicit full box vectors override everything.
        if box_vectors is not None:
            print(f'box_vectors is {box_vectors}')
            vector_x, vector_y, vector_z = box_vectors[0], box_vectors[1], box_vectors[2]
            vector_x = vector_x.value_in_unit(nanometer) if is_quantity(vector_x) else vector_x
            vector_y = vector_y.value_in_unit(nanometer) if is_quantity(vector_y) else vector_y
            vector_z = vector_z.value_in_unit(nanometer) if is_quantity(vector_z) else vector_z
            box_x, box_y, box_z = vector_x[0], vector_y[1], vector_z[2]

        # (2) explicit box lengths override.
        if boxsize is not None:
            print(f'boxsize is {boxsize}')
            boxsize = boxsize.value_in_unit(nanometer) if is_quantity(boxsize) else boxsize
            box_x, box_y, box_z = boxsize

        # (3) explicit in-plane vectors override x/y.
        if (pbc_x_vector is not None) and (pbc_y_vector is not None):
            pbc_x_vector = pbc_x_vector.value_in_unit(nanometer) if is_quantity(pbc_x_vector) else pbc_x_vector
            pbc_y_vector = pbc_y_vector.value_in_unit(nanometer) if is_quantity(pbc_y_vector) else pbc_y_vector
            box_x, box_y = pbc_x_vector[0], pbc_y_vector[1]

        # (4) padding is a FALLBACK for z, used only when SOME box info was given
        # but the z length is still missing.
        explicit_box = (box_x is not None) or (box_y is not None) or (box_z is not None)
        if explicit_box and box_z is None and padding is not None:
            padding = padding.value_in_unit(nanometer) if is_quantity(padding) else padding
            box_z = float(sys_zmax - sys_zmin + 2.0 * padding)
            print(f'padding fallback -> box_z={box_z}')

        # (5) GUESS the box when nothing supplied one:
        #       XY  = min-max of the lipid head groups (phosphorus)
        #       Z   = min-max of the PROTEIN + 2 * padding (default 1.0 nm)
        # Phosphorus is found by ELEMENT, not by name: Maestro writes "P1", so an
        # atom.name == "P" test finds nothing.
        #
        # Caveat this prints about: min-max is only correct for a WRAPPED frame. A
        # raw MD frame is usually unwrapped, and the few lipids straddling a face
        # inflate it — measured on a 216k-atom POPC system whose true box was
        # 12.246 x 12.417 nm: P min-max gave 12.92 x 13.11 (+11% area) while the
        # 1st-99th percentile span gave 12.35 x 12.44 (+1%). So when the two
        # disagree the frame is unwrapped, and we report the trimmed value too.
        if box_x is None or box_y is None or box_z is None:
            head = [
                atom.index
                for atom in self.topology.atoms()
                if atom.residue.name in lipid_names
                and atom.element is not None
                and atom.element.symbol == "P"
            ]
            if head:
                ref_xy, xy_source = all_pos[head, :], "lipid head groups (P)"
            else:
                lipid_all = [
                    atom.index
                    for atom in self.topology.atoms()
                    if atom.residue.name in lipid_names
                ]
                if lipid_all:
                    ref_xy, xy_source = all_pos[lipid_all, :], "all lipid atoms"
                else:
                    ref_xy, xy_source = all_pos, "the whole system"

            protein_idx = [
                atom.index
                for atom in self.topology.atoms()
                if atom.residue.name in protein_names
            ]
            if protein_idx:
                ref_z, z_source = all_pos[protein_idx, 2], "protein"
            else:
                ref_z, z_source = all_pos[:, 2], "the whole system"

            pad = padding if padding is not None else 1.0
            pad = pad.value_in_unit(nanometer) if is_quantity(pad) else float(pad)

            span_xy = ref_xy.max(axis=0) - ref_xy.min(axis=0)
            if box_x is None:
                box_x = float(span_xy[0])
            if box_y is None:
                box_y = float(span_xy[1])
            if box_z is None:
                box_z = float(ref_z.max() - ref_z.min() + 2.0 * pad)
            print(
                f"WARNING: no periodic box supplied — GUESSED it: "
                f"XY from {xy_source} min-max, Z from {z_source} min-max + 2 x {pad} nm "
                f"-> [{box_x:.3f}, {box_y:.3f}, {box_z:.3f}] nm.",
                file=sys.stderr,
            )

            # Unwrapped-frame check: compare min-max against a trimmed span.
            lo_xy, hi_xy = np.percentile(ref_xy, [1.0, 99.0], axis=0)
            trimmed = hi_xy - lo_xy
            if (span_xy[:2] > trimmed[:2] * 1.05).any():
                print(
                    "WARNING: the XY min-max exceeds the 1st-99th percentile span "
                    f"({span_xy[0]:.2f} x {span_xy[1]:.2f} vs "
                    f"{trimmed[0]:.2f} x {trimmed[1]:.2f} nm), which means this frame "
                    "is UNWRAPPED — molecules straddling a face inflate min-max, so "
                    "the guessed box is too large. Wrap the frame first "
                    "(alchemforge wrap-pbc), or pass the real box via box_vectors.",
                    file=sys.stderr,
                )

        print(f'box_x: {box_x}, box_y: {box_y}, box_z: {box_z}')
        if (box_x is not None) and (box_y is not None) and (box_z is not None):
            box_vector =(Vec3(box_x, 0, 0),
                        Vec3(0, box_y, 0),
                        Vec3(0, 0, box_z)) * nanometer
            print(f'New PBC: {box_vector}')
        else:
            raise ValueError(
                "PBC cannot be determined; give the input a CRYST1 box, or set "
                "box_vectors / boxsize / (pbc_x_vector + pbc_y_vector)."
            )

        # -----------------------------------------------------
        #                    Copy system
        # -----------------------------------------------------
        newTopology = Topology()
        newTopology.setUnitCellDimensions(Vec3(box_x, box_y, box_z) * nanometer)
        newTopology.setPeriodicBoxVectors(box_vector)

        newAtoms = {}
        newPositions = []*nanometer
        newResidueTemplates=dict()
        for chain in self.topology.chains():
            newChain = newTopology.addChain(chain.id)
            for residue in chain.residues():
                newResidue = newTopology.addResidue(residue.name, newChain, residue.id, residue.insertionCode)
                if residue in residueTemplates:
                    newResidueTemplates[newResidue] = residueTemplates[residue]
                for atom in residue.atoms():
                    newAtom = newTopology.addAtom(atom.name, atom.element, newResidue, atom.id, atom.formalCharge)
                    newAtoms[atom] = newAtom
                    newPositions.append(deepcopy(self.positions[atom.index]))
        for bond in self.topology.bonds():
            newTopology.addBond(newAtoms[bond[0]], newAtoms[bond[1]], bond.type, bond.order)

        # Sort the solute atoms into cells for fast lookup.
        if len(self.positions) == 0:
            newPositions= []
        else:
            newPositions = deepcopy(self.positions.value_in_unit(nanometer))

        # -----------------------------------------------------
        #       Add solvent around the whole system
        # -----------------------------------------------------

        modeller = Modeller(newTopology, newPositions)
        print(f'PBC: {newTopology.getPeriodicBoxVectors()}')
        modeller.topology.setPeriodicBoxVectors(newTopology.getPeriodicBoxVectors())
        modeller.topology.setUnitCellDimensions(newTopology.getUnitCellDimensions())

        modeller.addSolvent(
            forcefield,
            model=solvent_model,
            neutralize=False,
            residueTemplates=newResidueTemplates,
        )

        # Recompute positions after solvation.
        solv_pos = self._to_nm_array(modeller.positions)

        # Atoms 0..n_original-1 are the ORIGINAL system (copied in the same order
        # above); addSolvent only APPENDS new atoms at higher indices. So a water is
        # "newly added" iff its atoms sit beyond n_original_atoms. We delete ONLY
        # newly added waters that land in the membrane core — every original water
        # is preserved, including any that sit inside the membrane.
        n_original_atoms = len(all_pos)

        to_delete = []
        # Which newly added waters clash with the membrane? Distance to the nearest
        # LIPID atom, not a z-band: a z-band also deletes legitimate water at the
        # head groups and inside protein pores, which empties the bulk. Anything
        # closer than `water_lipid_cutoff` to a lipid atom is overlapping and goes.
        lipid_all_indices = [
            atom.index
            for atom in modeller.topology.atoms()
            if atom.residue.name in lipid_names
        ]
        lipid_xyz = solv_pos[lipid_all_indices, :] if lipid_all_indices else None
        cutoff = float(water_lipid_cutoff)

        def _too_close(points):
            """Boolean mask: point within `cutoff` of any lipid atom."""
            if lipid_xyz is None or len(points) == 0:
                return np.zeros(len(points), dtype=bool)
            try:
                from scipy.spatial import cKDTree

                tree = cKDTree(lipid_xyz)
                return np.array([len(h) > 0 for h in tree.query_ball_point(points, cutoff)])
            except ImportError:
                from collections import defaultdict

                grid = defaultdict(list)
                for i, c in enumerate(np.floor(lipid_xyz / cutoff).astype(np.int64)):
                    grid[tuple(c)].append(i)
                offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
                mask = np.zeros(len(points), dtype=bool)
                for n, p in enumerate(points):
                    cx, cy, cz = np.floor(p / cutoff).astype(np.int64)
                    cand = []
                    for da, db, dc in offs:
                        cand.extend(grid.get((cx + da, cy + db, cz + dc), ()))
                    if cand and (np.linalg.norm(lipid_xyz[cand] - p, axis=1) < cutoff).any():
                        mask[n] = True
                return mask

        candidates, cand_pos = [], []
        for residue in modeller.topology.residues():
            if residue.name not in ["HOH", "WAT"]:
                continue
            atom_indices = [atom.index for atom in residue.atoms()]
            if min(atom_indices) < n_original_atoms:
                continue  # NEVER delete an original water
            oxy_index = next(
                (a.index for a in residue.atoms()
                 if a.element is not None and a.element.symbol == "O"),
                None,
            )
            if oxy_index is None:
                continue
            candidates.append(residue)
            cand_pos.append(solv_pos[oxy_index])

        if candidates:
            clashing = _too_close(np.array(cand_pos))
            to_delete = [r for r, bad in zip(candidates, clashing) if bad]

        print(
            f'Water cleanup: {len(candidates)} newly-added water(s) checked; deleting '
            f'{len(to_delete)} within {cutoff * 10:.1f} A of a lipid atom. Original '
            'waters are never deleted.'
        )
        if len(to_delete) > 0:
            modeller.delete(to_delete)


        # Update this object in-place.
        newResidueTemplates = {}
        for r1, r2 in zip(self.topology.residues(), modeller.topology.residues()):
            if r1 in residueTemplates:
                newResidueTemplates[r2] = residueTemplates[r1]

        self.topology = modeller.topology
        self.positions = modeller.positions

        # Select a subset of water molecules to replace with ions, ignoring
        # those within a certain distance from either leaflet of the membrane.
        waterPos = {}  # redo because modeller.delete changes chain indexes
        for chain in list(modeller.topology.chains()):
            for residue in chain.residues():
                if residue.name in ['HOH', 'WAT']:
                    for atom in residue.atoms():
                        if atom.element.symbol == "O":
                            waterPos[residue] = modeller.positions[atom.index]

        # Total number of water molecules
        # Use this number to avoid underestimating the concentration of ions
        # in _addIons after we exclude waters close to lipids.
        numTotalWaters = len(waterPos)

        # Ignore waters that are within a certain distance of the membrane
        lipidOffset = 0.25
        upperZBoundary = (lipid_zmax + lipidOffset)
        lowerZBoundary = (lipid_zmin - lipidOffset)
        waterResidues = list(waterPos)
        for wRes in waterResidues:
            waterZ = waterPos[wRes][2]
            if lowerZBoundary < waterZ.value_in_unit(nanometer) < upperZBoundary:
                del waterPos[wRes]

        gc.collect()
        if write_structure:
            with open("addSolventWithExistingMembrane_sol.pdb", "w") as f:
                PDBFile.writeFile(self.topology, self.positions, f)

        self._addIons(forcefield, numTotalWaters, waterPos, positiveIon=positiveIon, negativeIon=negativeIon, ionicStrength=ionicStrength, neutralize=neutralize, residueTemplates=newResidueTemplates)
        if write_structure:
            with open("addSolventWithExistingMembrane_sol_ion.pdb", "w") as f:
                PDBFile.writeFile(self.topology, self.positions, f)


    def addMembrane(
        self,
        forcefield,
        lipidType='POPC',
        scale_factor=0.6,
        membraneCenterZ=0 * nanometer,
        x_padding=1.0 * nanometer,
        y_padding=1.0 * nanometer,
        z_padding=1.0 * nanometer,
        positiveIon='Na+',
        negativeIon='Cl-',
        ionicStrength=0 * molar,
        neutralize=True,
        write_structure=False,
        residueTemplates=dict(),
        platform=None
    ):

        # ============================================================
        # Determine membrane boundaries and trim excess water/lipids
        # ============================================================

        # ------------------------------------------------------------
        # 1. Read lipid patch
        # ------------------------------------------------------------
        lipid_types = (
            'POPC', 'POPE', 'DLPC', 'DLPE',
            'DMPC', 'DOPC', 'DPPC'
        )

        lipid_types_3resn = {
            "POPC": "POP",
            'POPE': "POP",
            'DLPC': 'DLP',
            'DLPE': 'DLP',
            'DMPC': 'DMP',
            'DOPC': 'DOP',
            'DPPC': 'DPP',
        }
        if 'topology' in dir(lipidType) and 'positions' in dir(lipidType):
            patch = lipidType
        elif lipidType.upper() in lipid_types:
            openmm_path = os.path.dirname(app.__file__)
            patch = PDBFile(os.path.join(openmm_path,'data', lipidType.upper() + '.pdb'))
        else:
            raise ValueError('Unsupported lipid type: ' + lipidType)

        # ------------------------------------------------------------
        # 2. Convert all input lengths to plain nm floats
        # ------------------------------------------------------------
        if is_quantity(membraneCenterZ):
            membraneCenterZ = membraneCenterZ.value_in_unit(nanometer)

        if is_quantity(x_padding):
            x_padding = x_padding.value_in_unit(nanometer)

        if is_quantity(y_padding):
            y_padding = y_padding.value_in_unit(nanometer)

        if is_quantity(z_padding):
            z_padding = z_padding.value_in_unit(nanometer)

        # ------------------------------------------------------------
        # 3. Protein coordinates and dimensions
        # ------------------------------------------------------------
        proteinPos = self.positions.value_in_unit(nanometer)
        proteinMinPos = Vec3(*[min(p[i] for p in proteinPos) for i in range(3)])
        proteinMaxPos = Vec3(*[max(p[i] for p in proteinPos) for i in range(3)])
        proteinSize = proteinMaxPos - proteinMinPos
        proteinCenterPos = (proteinMinPos + proteinMaxPos) / 2.0

        # Keep XY center from protein, set Z to membraneCenterZ
        proteinCenterPos = Vec3(proteinCenterPos[0], proteinCenterPos[1], membraneCenterZ)

        # ------------------------------------------------------------
        # 4. Membrane patch dimensions
        # ------------------------------------------------------------
        patchPos = patch.positions.value_in_unit(nanometer)
        patchSize = patch.topology.getUnitCellDimensions().value_in_unit(nanometer)
        patchMinPos = Vec3(*[min(p[i] for p in patchPos) for i in range(3)])
        patchMaxPos = Vec3(*[max(p[i] for p in patchPos) for i in range(3)])
        patchCenterPos = (patchMinPos + patchMaxPos) / 2.0

        # ------------------------------------------------------------
        # 5. Desired final XY dimensions
        #
        # IMPORTANT:
        # final dimensions = protein dimensions + 2 * padding
        # ------------------------------------------------------------
        paddingSizeX = proteinSize[0] + 2.0 * x_padding
        paddingSizeY = proteinSize[1] + 2.0 * y_padding

        print(f"Protein XY: "f"({proteinSize[0]}, {proteinSize[1]}) nm")

        print(f"Desired final XY: ({paddingSizeX}, {paddingSizeY}) nm")

        # ------------------------------------------------------------
        # 6. Number of lipid patches needed
        # ------------------------------------------------------------
        nx = int(ceil(paddingSizeX / patchSize[0]))
        ny = int(ceil(paddingSizeY / patchSize[1]))

        boxSizeX = nx * patchSize[0]
        boxSizeY = ny * patchSize[1]

        print(f"Temporary membrane XY: ({boxSizeX}, {boxSizeY}) nm")

        # ------------------------------------------------------------
        # 7. Record lipid residue bonds
        # ------------------------------------------------------------
        resBonds = defaultdict(list)

        for bond in patch.topology.bonds():
            resBonds[bond[0].residue].append(bond)

        # ------------------------------------------------------------
        # 8. Identify membrane leaflets
        # ------------------------------------------------------------
        numLipidAtoms = 0
        resMeanZ = {}
        membraneMeanZ = 0.0

        for res in patch.topology.residues():
            if res.name not in ['HOH', 'WAT']:

                numResAtoms = 0
                sumZ = 0.0

                for atom in res.atoms():
                    numResAtoms += 1
                    sumZ += patchPos[atom.index][2]

                numLipidAtoms += numResAtoms
                membraneMeanZ += sumZ
                resMeanZ[res] = sumZ / numResAtoms

        membraneMeanZ /= numLipidAtoms

        lipidLeaf = {res: 0 if resMeanZ[res] < membraneMeanZ else 1 for res in resMeanZ}

        # ------------------------------------------------------------
        # 9. Scaled protein coordinates
        # ------------------------------------------------------------
        scaledProteinPos = [None] * len(proteinPos)

        for i, p in enumerate(proteinPos):

            p = p - proteinCenterPos

            p = Vec3(scale_factor * p[0], scale_factor * p[1], p[2])

            scaledProteinPos[i] = p + proteinCenterPos

        # ------------------------------------------------------------
        # 10. Create temporary membrane topology
        # ------------------------------------------------------------
        membraneTopology = Topology()
        membranePos = []

        boxSizeZ = patchSize[2]

        if self.topology.getUnitCellDimensions() is not None:
            boxSizeZ = max(boxSizeZ, self.topology.getUnitCellDimensions()[2].value_in_unit(nanometer) + 2.0 * z_padding)
        else:
            boxSizeZ = max(boxSizeZ, proteinSize[2] + 2.0 * z_padding)

        membraneTopology.setUnitCellDimensions((boxSizeX, boxSizeY, boxSizeZ))

        # ------------------------------------------------------------
        # 11. Spatial cell lists
        # ------------------------------------------------------------
        overlapCutoff = 0.22

        addedWater = []
        addedLipids = []
        removedFromLeaf = [0, 0]

        vectors = (membraneTopology.getPeriodicBoxVectors().value_in_unit(nanometer))

        proteinCells = _CellList(proteinPos, overlapCutoff, vectors, False)

        scaledProteinCells = _CellList(scaledProteinPos, overlapCutoff, vectors, False)

        # ------------------------------------------------------------
        # 12. Build temporary membrane
        # ------------------------------------------------------------
        for x in range(nx):
            for y in range(ny):

                offset = (proteinCenterPos - patchCenterPos + Vec3((x - 0.5 * (nx - 1)) * patchSize[0], (y - 0.5 * (ny - 1)) * patchSize[1],0))

                for res in patch.topology.residues():

                    resPos = [patchPos[atom.index] + offset for atom in res.atoms()]

                    if res.name not in ['HOH', 'WAT']:

                        referencePosLists = [proteinPos, scaledProteinPos]
                        cellLists = [proteinCells, scaledProteinCells]

                    else:

                        referencePosLists = [scaledProteinPos]
                        cellLists = [scaledProteinCells]

                    overlap = False
                    nearest = nx * patchSize[0]

                    for cells, referencePos in zip(cellLists,referencePosLists):

                        if overlap:
                            break

                        for index, atom in enumerate(res.atoms()):

                            pos = resPos[index]

                            for atom_index in cells.neighbors(pos):

                                distance = norm(pos - referencePos[atom_index])

                                if distance < overlapCutoff:
                                    overlap = True
                                    break

                                nearest = min(nearest, distance)

                            if overlap:
                                break

                    if res.name in ['HOH', 'WAT']:

                        if not overlap:
                            addedWater.append((res, resPos))

                    else:

                        if overlap:
                            removedFromLeaf[lipidLeaf[res]] += 1

                        else:
                            addedLipids.append((nearest, res, resPos))

        skipFromLeaf = [max(removedFromLeaf) - removedFromLeaf[i] for i in (0, 1)]

        # ------------------------------------------------------------
        # 13. Add lipids
        # ------------------------------------------------------------
        newAtoms = {}

        lipidChain = membraneTopology.addChain()
        lipidResNum = 1

        for nearest, residue, pos in addedLipids:
            leaf = lipidLeaf[residue]
            if skipFromLeaf[leaf] > 0:
                skipFromLeaf[leaf] -= 1
            else:
                newResidue = membraneTopology.addResidue(residue.name, lipidChain, str(lipidResNum), residue.insertionCode)
                lipidResNum += 1

                for atom in residue.atoms():

                    newAtom = membraneTopology.addAtom(atom.name, atom.element, newResidue, atom.id, atom.formalCharge)
                    newAtoms[atom] = newAtom

                membranePos += pos

                for bond in resBonds[residue]:

                    membraneTopology.addBond(
                        newAtoms[bond[0]],
                        newAtoms[bond[1]],
                        bond.type,
                        bond.order
                    )

        del lipidLeaf
        del addedLipids

        # ------------------------------------------------------------
        # 14. Add membrane water
        # ------------------------------------------------------------
        solventChain = membraneTopology.addChain()

        for residue, pos in addedWater:

            newResidue = membraneTopology.addResidue(
                residue.name,
                solventChain,
                residue.id,
                residue.insertionCode
            )

            for atom in residue.atoms():

                newAtom = membraneTopology.addAtom(
                    atom.name,
                    atom.element,
                    newResidue,
                    atom.id,
                    atom.formalCharge
                )

                newAtoms[atom] = newAtom

            membranePos += pos

            for bond in resBonds[residue]:

                membraneTopology.addBond(
                    newAtoms[bond[0]],
                    newAtoms[bond[1]],
                    bond.type,
                    bond.order
                )

        del newAtoms
        del addedWater
        del resBonds

        gc.collect()

        # ------------------------------------------------------------
        # 15. Create membrane system + fixed protein
        # ------------------------------------------------------------
        system = forcefield.createSystem(
            membraneTopology,
            nonbondedMethod=CutoffPeriodic,
            residueTemplates=residueTemplates
        )

        proteinSystem = forcefield.createSystem(
            self.topology,
            nonbondedMethod=CutoffNonPeriodic,
            residueTemplates=residueTemplates
        )

        numMembraneParticles = system.getNumParticles()
        numProteinParticles = proteinSystem.getNumParticles()

        for i in range(numProteinParticles):
            system.addParticle(0.0)

        nonbonded = None

        for f1, f2 in zip(
            system.getForces(),
            proteinSystem.getForces()
        ):

            if isinstance(f1, NonbondedForce):

                nonbonded = f2

                for i in range(numProteinParticles):

                    f1.addParticle(
                        *f2.getParticleParameters(i)
                    )

                    for j in scaledProteinCells.neighbors(
                        scaledProteinPos[i]
                    ):

                        if j < i:

                            f1.addException(
                                i + numMembraneParticles,
                                j + numMembraneParticles,
                                0.0,
                                1.0,
                                0.0
                            )

            elif isinstance(f1, CustomNonbondedForce):

                for i in range(numProteinParticles):

                    f1.addParticle(
                        f2.getParticleParameters(i)
                    )

                    for j in scaledProteinCells.neighbors(
                        scaledProteinPos[i]
                    ):

                        if j < i:

                            f1.addExclusion(
                                i + numMembraneParticles,
                                j + numMembraneParticles
                            )

        if nonbonded is None:
            raise ValueError(
                'The ForceField does not specify a NonbondedForce'
            )

        mergedPositions = membranePos + scaledProteinPos

        del membranePos

        del scaledProteinCells

        gc.collect()

        # ------------------------------------------------------------
        # 16. Relax membrane
        # ------------------------------------------------------------
        steps = int(
            max(proteinSize.x, proteinSize.y) * 10
        ) + 1

        integrator = LangevinIntegrator(
            10.0,
            50.0,
            0.001
        )

        if platform is None:
            context = Context(
                system,
                integrator
            )
        else:
            context = Context(
                system,
                integrator,
                platform
            )

        context.setPositions(mergedPositions)

        LocalEnergyMinimizer.minimize(
            context,
            10.0,
            30
        )

        try:
            import numpy as np

            hasNumpy = True

            proteinPosArray = np.array(
                proteinPos
            )

            scaledProteinPosArray = np.array(
                scaledProteinPos
            )

        except:
            hasNumpy = False

        for i in range(steps):

            weight1 = i / (steps - 1)
            weight2 = 1.0 - weight1

            mergedPositions = (
                context
                .getState(positions=True)
                .getPositions(
                    asNumpy=hasNumpy
                )
                .value_in_unit(nanometer)
            )

            if hasNumpy:

                mergedPositions[
                    numMembraneParticles:
                ] = (
                    weight1 * proteinPosArray
                    + weight2 * scaledProteinPosArray
                )

            else:

                for j in range(len(proteinPos)):

                    mergedPositions[
                        j + numMembraneParticles
                    ] = (
                        weight1 * proteinPos[j]
                        + weight2 * scaledProteinPos[j]
                    )

            context.setPositions(
                mergedPositions
            )

            integrator.step(20)

        # ------------------------------------------------------------
        # 17. Add membrane to protein
        # ------------------------------------------------------------
        modeller = Modeller(self.topology, self.positions)

        modeller.add(membraneTopology, context.getState(positions=True).getPositions()[:numMembraneParticles])

        # Use temporary membrane box first
        modeller.topology.setPeriodicBoxVectors(membraneTopology.getPeriodicBoxVectors())

        del context
        del system
        del integrator

        if write_structure:
            with open("addMembrane_before_sol.pdb", "w") as f:
                PDBFile.writeFile(modeller.topology, modeller.positions, f)

        # ------------------------------------------------------------
        # 18. Add extra water if Z box needs expansion
        # ------------------------------------------------------------
        needExtraWater = (boxSizeZ > patchSize[2])

        if needExtraWater:
            newResidueTemplates = {}

            for r1, r2 in zip(self.topology.residues(), modeller.topology.residues()):

                if r1 in residueTemplates:
                    newResidueTemplates[r2] = (residueTemplates[r1])

            modeller.addSolvent(forcefield, neutralize=False, residueTemplates=newResidueTemplates)

        # ------------------------------------------------------------
        # 19. Find all water residues
        # ------------------------------------------------------------
        waterPos = {}

        for residue in modeller.topology.residues():
            if residue.name in ['HOH', 'WAT']:
                for atom in residue.atoms():
                    if atom.element == elem.oxygen:
                        waterPos[residue] = (modeller.positions[atom.index].value_in_unit(nanometer))
                        break

        # ------------------------------------------------------------
        # 20. Calculate membrane Z boundaries
        # ------------------------------------------------------------
        lipidNames = {res.name for res in patch.topology.residues() if res.name not in {"HOH", "WAT"}}
        print("Lipid residue names:", lipidNames)

        lipidZMax = -float("inf")
        lipidZMin = float("inf")

        for res in modeller.topology.residues():
            if res.name in lipidNames:
                for atom in res.atoms():
                    atomZ = (modeller.positions[atom.index][2].value_in_unit(nanometer))

                    lipidZMax = max(lipidZMax, atomZ)
                    lipidZMin = min(lipidZMin, atomZ)

        # Keep 0.25 nm clearance from membrane
        lipidOffset = 0.25

        upperZBoundary = (lipidZMax + lipidOffset)
        lowerZBoundary = (lipidZMin - lipidOffset)

        # ------------------------------------------------------------
        # 21. Define FINAL XY region
        #
        # Center this region on the protein center.
        # This is equivalent to:
        #
        # center +/- proteinSize/2 +/- padding
        # ------------------------------------------------------------
        offset = 0.25

        leftXboundary = (proteinCenterPos[0] - paddingSizeX / 2.0 - offset)
        rightXboundary = (proteinCenterPos[0] + paddingSizeX / 2.0 + offset)
        leftYboundary = (proteinCenterPos[1] - paddingSizeY / 2.0 - offset)
        rightYboundary = (proteinCenterPos[1] + paddingSizeY / 2.0 + offset)

        print()
        print("FINAL DELETION BOUNDARIES")
        print("X:", leftXboundary, rightXboundary)
        print("Y:", leftYboundary, rightYboundary)
        print("Z:", lowerZBoundary, upperZBoundary)

        # ------------------------------------------------------------
        # 22. DELETE WATER FROM TOPOLOGY
        # ------------------------------------------------------------
        waterToDelete = []

        for residue, pos in waterPos.items():

            waterX = pos[0]
            waterY = pos[1]
            waterZ = pos[2]

            delete_water = (
                lowerZBoundary + 1.1 < waterZ < upperZBoundary - 1.1
                or waterX < leftXboundary
                or waterX > rightXboundary
                or waterY < leftYboundary
                or waterY > rightYboundary
            )

            if delete_water:
                waterToDelete.append(residue)

        print("Water before deletion:", len(waterPos))
        print("Waters to delete:", len(waterToDelete))

        if waterToDelete:
            modeller.delete(waterToDelete)

        # ------------------------------------------------------------
        # 23. Rebuild lipid position dictionary
        #     because topology changed
        # ------------------------------------------------------------
        lipid_types_3resn = {
            "POPC": "POP",
            'POPE': "POP",
            'DLPC': 'DLP',
            'DLPE': 'DLP',
            'DMPC': 'DMP',
            'DOPC': 'DOP',
            'DPPC': 'DPP',
        }
        lipid_to_head = {
            "POP": "P",
            "DLP": "P",
            "DMP": "P",
            "DOP": "P",
            "DPP": "P",
        }

        lipidPos = {}

        head_symbol = lipid_to_head[lipid_types_3resn[lipidType]]

        for residue in modeller.topology.residues():
            if residue.name != lipid_types_3resn[lipidType]:
                continue
            for atom in residue.atoms():
                if atom.element.symbol == head_symbol:
                    lipidPos[residue] = (modeller.positions[atom.index].value_in_unit(nanometer))
                    break

        # ------------------------------------------------------------
        # 24. DELETE LIPIDS FROM TOPOLOGY
        #
        # Use head-group coordinates to decide if a whole lipid
        # lies outside the desired membrane XY region.
        # ------------------------------------------------------------
        lipidToDelete = []

        for residue, pos in lipidPos.items():
            lipidX = pos[0]
            lipidY = pos[1]

            delete_lipid = (
                lipidX < leftXboundary
                or lipidX > rightXboundary
                or lipidY < leftYboundary
                or lipidY > rightYboundary
            )

            if delete_lipid:
                lipidToDelete.append(residue)

        print("Lipids before deletion:", len(lipidPos))
        print("Lipids to delete:", len(lipidToDelete))

        if lipidToDelete:
            modeller.delete(lipidToDelete)

        # ------------------------------------------------------------
        # 25. Rebuild water dictionary AFTER topology deletions
        # ------------------------------------------------------------
        waterPos = {}

        for residue in modeller.topology.residues():
            if residue.name in ['HOH', 'WAT']:
                for atom in residue.atoms():
                    if atom.element == elem.oxygen:
                        waterPos[residue] = (modeller.positions[atom.index])
                        break

        # Actual final number of waters
        numTotalWaters = len(waterPos)

        print("Final water count:", numTotalWaters)

        # ------------------------------------------------------------
        # 26. Set FINAL periodic box
        # ------------------------------------------------------------

        # Get the original Z length in nm
        modeller_pbc = modeller.topology.getPeriodicBoxVectors()
        z_vec = modeller_pbc[2].value_in_unit(unit.nanometer)

        # Make sure z_vec is a plain Vec3
        final_zVec = Vec3(z_vec[0], z_vec[1], z_vec[2])
        final_xVec = Vec3(paddingSizeX, 0, 0)
        final_yVec = Vec3(0, paddingSizeY, 0)

        # Apply nanometer ONCE to the complete tuple
        final_box = (final_xVec, final_yVec, final_zVec) * unit.nanometer

        modeller.topology.setPeriodicBoxVectors(final_box)

        print("FINAL BOX:")
        print(modeller.topology.getPeriodicBoxVectors())

        # ------------------------------------------------------------
        # 27. Update self ONLY NOW
        # ------------------------------------------------------------
        self.topology = modeller.topology
        self.positions = modeller.positions

        # Re-create residue templates for current topology
        newResidueTemplates = {}

        for r1, r2 in zip(self.topology.residues(), modeller.topology.residues()):

            if r1 in residueTemplates:
                newResidueTemplates[r2] = (residueTemplates[r1])

        # ------------------------------------------------------------
        # 28. Write structure before ions
        # ------------------------------------------------------------
        if write_structure:

            with open("addMembrane_sol.pdb", "w") as f:
                PDBFile.writeFile(self.topology, self.positions, f)

        # ------------------------------------------------------------
        # 29. Add ions
        # ------------------------------------------------------------
        self._addIons(
            forcefield,
            numTotalWaters,
            waterPos,
            positiveIon=positiveIon,
            negativeIon=negativeIon,
            ionicStrength=ionicStrength,
            neutralize=neutralize,
            residueTemplates=newResidueTemplates
        )

        # ------------------------------------------------------------
        # 30. Write final structure
        # ------------------------------------------------------------
        if write_structure:

            with open("addMembrane_sol_ion.pdb", "w") as f:
                PDBFile.writeFile(self.topology, self.positions, f)


# ============================================================
# Config-driven entrypoint
# ============================================================


def _resolve_platform(name):
    from openmm import Platform

    try:
        return Platform.getPlatformByName(str(name))
    except Exception:
        return None


def build_from_config(config_path):
    """Load + validate a YAML config, build/trim or solvate a membrane, write PDB."""
    conf = load_config(config_path)
    validate_config(conf)

    # The relax module owns the shared ligand/IO helpers (both are heavy).
    from chemflow.fe.openmm_modeling.membrane_relax import (
        _register_small_molecule,
        fix_structure,
        read_structure,
    )

    workdir = Path(conf.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    pdb = read_structure(conf.input_pdb)
    modeller = MembraneModeller(pdb.topology, pdb.positions)

    # Repair the usual Maestro->Amber mismatches (HIS naming, missing/terminal H)
    # before anything tries to match force-field templates.
    if bool(getattr(conf, "fix_structure", True)):
        fix_structure(
            modeller,
            add_hydrogens=bool(getattr(conf, "add_missing_hydrogens", True)),
            ph=float(getattr(conf, "ph", 7.0)),
        )

    forcefield = ForceField(*list(conf.forcefields))
    if getattr(conf, "ligand_sdf", None):
        _register_small_molecule(forcefield, conf.ligand_sdf, conf)

    platform = _resolve_platform(conf.platform)
    ionic = float(conf.ionic_strength) * molar

    if str(conf.mode) == "build":
        build = conf.build
        print(f"Building a trimmed {build.lipid_type} membrane around the protein...")
        modeller.addMembrane(
            forcefield,
            lipidType=str(build.lipid_type),
            scale_factor=float(build.scale_factor),
            membraneCenterZ=float(build.membrane_center_z) * nanometer,
            x_padding=float(build.x_padding) * nanometer,
            y_padding=float(build.y_padding) * nanometer,
            z_padding=float(build.z_padding) * nanometer,
            positiveIon=str(conf.positive_ion),
            negativeIon=str(conf.negative_ion),
            ionicStrength=ionic,
            neutralize=bool(conf.neutralize),
            write_structure=bool(conf.write_intermediates),
            platform=platform,
        )
    else:  # existing membrane: solvate empty space + ions
        existing = conf.existing
        kwargs = {
            "solvent_model": str(conf.solvent_model),
            "padding": float(getattr(existing, "padding", 1.0)) * nanometer,
            "positiveIon": str(conf.positive_ion),
            "negativeIon": str(conf.negative_ion),
            "ionicStrength": ionic,
            "neutralize": bool(conf.neutralize),
            "write_structure": bool(conf.write_intermediates),
            "align_membrane": bool(getattr(conf, "align_membrane", True)),
            "water_lipid_cutoff": float(getattr(conf, "water_lipid_cutoff", 0.28)),
        }
        boxsize = getattr(existing, "boxsize", None)
        if boxsize:
            kwargs["boxsize"] = Vec3(*[float(v) for v in boxsize]) * nanometer
        box_vectors = getattr(existing, "box_vectors", None)
        if box_vectors:
            kwargs["box_vectors"] = (
                tuple(Vec3(*[float(v) for v in row]) for row in box_vectors) * nanometer
            )
        pbc_x = getattr(existing, "pbc_x", None)
        pbc_y = getattr(existing, "pbc_y", None)
        if pbc_x and pbc_y:
            kwargs["pbc_x_vector"] = Vec3(float(pbc_x), 0, 0) * nanometer
            kwargs["pbc_y_vector"] = Vec3(0, float(pbc_y), 0) * nanometer
        print("Solvating + ionizing an existing-membrane system...")
        modeller.addSolventWithExistingMembrane(forcefield, **kwargs)

    from openmm.app import PDBxFile

    n_atoms = modeller.topology.getNumAtoms()
    out_pdb = workdir / conf.output_pdb
    out_cif = out_pdb.with_suffix(".cif")

    # mmCIF is always written: no serial limit, and it is what OpenFE reads.
    with open(out_cif, "w") as handle:
        PDBxFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)

    # A PDB is always written too. Under 99,999 atoms OpenMM's writer is fine;
    # beyond that its serial column overflows, so use the hybrid-36 writer (the
    # standard large-system encoding, which VMD and Maestro both read).
    if n_atoms < 100000:
        with open(out_pdb, "w") as handle:
            # keepIds: preserve the input's chain/residue numbering. Without it
            # OpenMM renumbers every residue sequentially from 1, so ligand and
            # cofactor resids would no longer match the input structure.
            PDBFile.writeFile(
                modeller.topology, modeller.positions, handle, keepIds=True
            )
        # OpenMM's PDB writer truncates residue names to 3 columns, turning POPC
        # into POP so the lipid no longer matches its force-field template.
        try:
            from alchemforge.structure.pdbfix import (
                _restore_four_character_residue_names,
            )

            _restore_four_character_residue_names(out_pdb, modeller.topology)
        except Exception as exc:  # noqa: BLE001 - cosmetic, never block the build
            print(f"(could not restore 4-character residue names: {exc})",
                  file=sys.stderr)
    else:
        from alchemforge.structure.wrap_pbc import _write_pdb_hybrid36

        box = modeller.topology.getPeriodicBoxVectors().value_in_unit(nanometer)
        _write_pdb_hybrid36(
            out_pdb,
            modeller.topology,
            np.array(modeller.positions.value_in_unit(nanometer)),
            np.array([box[0][0], box[1][1], box[2][2]]),
            np,
        )
        print(
            f"NOTE: {n_atoms:,} atoms exceeds the 99,999-atom PDB limit, so the PDB "
            "was written with hybrid-36 serials (readable by VMD/Maestro, but NOT by "
            "OpenMM — feed the .cif to the next step).",
            file=sys.stderr,
        )
    print(f"Wrote membrane complex: {out_pdb}  and  {out_cif}")
    result = out_pdb

    print("Feed this into: chemflow openmm modeling relax-membrane")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build/trim or solvate a membrane complex for OpenMM.",
    )
    parser.add_argument(
        "--config", required=True, help="Membrane-build YAML configuration file"
    )
    args = parser.parse_args(argv)
    build_from_config(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
