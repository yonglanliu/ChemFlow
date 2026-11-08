"""Adapter around the reference MobleyLab SeparatedTopologies implementation."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from .config import SepTopGromacsConfig


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def prepare_edge(
    config: SepTopGromacsConfig, edge_id: str, ligand_a: str, ligand_b: str
) -> Path:
    """Build complex and solvent SepTop coordinates/topologies for one edge."""
    sys.path.insert(0, str(config.septop_source_dir))
    try:
        import boresch_restraints as br
        import combine_coordinates as cc
        import distance_solvent as ds
        import make_septop as ms
        import parmed as pmd
    finally:
        sys.path.pop(0)

    source_a = config.prepared_root / ligand_a
    source_b = config.prepared_root / ligand_b
    edge = config.project_dir / "edges" / edge_id
    complex_dir = edge / "complex"
    solvent_dir = edge / "solvent"
    complex_dir.mkdir(parents=True, exist_ok=True)
    solvent_dir.mkdir(parents=True, exist_ok=True)

    fit_b = complex_dir / "complex_B_fit.pdb"
    gro_b = complex_dir / "complex_B_fit.gro"
    complex_gro = complex_dir / "complex.gro"
    if config.complexes_pre_aligned:
        gro_b = source_b / "complex.gro"
    else:
        cc.align_complexes(
            str(source_a / "complex.pdb"),
            str(source_b / "complex.pdb"),
            str(fit_b),
        )
        cc.pdb2gro(str(fit_b), str(gro_b))
    cc.combine_ligands_gro(
        str(source_a / "complex.gro"),
        str(gro_b),
        str(complex_gro),
        ligand_A=config.complex_ligand_resname,
        ligand_B=config.complex_ligand_resname,
    )
    cc.edit_indices(str(complex_gro), str(complex_gro))

    traj_a = source_a / "complex.h5"
    traj_b = source_b / "complex.h5"
    if not traj_a.is_file():
        traj_a = source_a / "complex.gro"
    if not traj_b.is_file():
        traj_b = gro_b
    restraint_a, restraint_b, dg_a_off, dg_b_on = br.restrain_ligands(
        str(traj_a),
        str(traj_b),
        str(source_a / "ligand.mol2"),
        str(source_b / "ligand.mol2"),
        str(complex_dir / "boresch_restraints_A_on.itp"),
        str(complex_dir / "boresch_restraints_B_off.itp"),
        str(complex_dir / "boresch_restraints_A.itp"),
        str(complex_dir / "boresch_restraints_B.itp"),
        ligand_atoms=None,
        protein_atoms=None,
        substructure=None,
        ligand=config.complex_ligand_resname,
        top_A=str(source_a / "complex.gro"),
        top_B=str(gro_b),
    )
    complex_top = complex_dir / "complex.top"
    complex_step1 = complex_dir / "complex_1_scale.top"
    complex_step2 = complex_dir / "complex_2_scale.top"
    with _working_directory(complex_dir):
        ms.combine_ligands_top(
            str(source_a / "complex.top"),
            str(source_b / "complex.top"),
            str(complex_top),
            ligand=config.complex_ligand_resname,
        )
        ms.create_top(
            str(complex_top),
            str(complex_step1),
            config.complex_gamma,
            "vdwq_scaled-vdw",
            "dummy_scaled-vdwq",
            str(source_a / "complex.top"),
            str(source_b / "complex.top"),
            ligand=config.complex_ligand_resname,
        )
        br.include_itp_in_top(str(complex_step1), "boresch_restraints_B.itp")
        br.include_itp_in_top(str(complex_step1), "boresch_restraints_A_on.itp")
        ms.create_top(
            str(complex_top),
            str(complex_step2),
            config.complex_gamma,
            "scaled-vdw_dummy",
            "scaled-vdwq_vdwq",
            str(source_a / "complex.top"),
            str(source_b / "complex.top"),
            ligand=config.complex_ligand_resname,
        )
        br.include_itp_in_top(str(complex_step2), "boresch_restraints_B_off.itp")
        br.include_itp_in_top(str(complex_step2), "boresch_restraints_A.itp")

    solvent_gro = solvent_dir / "solvent.gro"
    cc.combine_ligands_gro(
        str(source_a / "solvent.gro"),
        str(source_b / "solvent.gro"),
        str(solvent_gro),
        ligand_A=config.solvent_ligand_resname,
        ligand_B=config.solvent_ligand_resname,
    )
    cc.edit_indices(str(solvent_gro), str(solvent_gro))
    solvent_top = solvent_dir / "solvent.top"
    with _working_directory(solvent_dir):
        ms.combine_ligands_top(
            str(source_a / "solvent.top"),
            str(source_b / "solvent.top"),
            str(solvent_top),
            ligand=config.solvent_ligand_resname,
        )
        for step, initial, final in (
            (1, "vdwq_scaled-vdw", "dummy_scaled-vdwq"),
            (2, "scaled-vdw_dummy", "scaled-vdwq_vdwq"),
        ):
            for suffix in ("", "_eq"):
                ms.create_top(
                    str(solvent_top),
                    str(solvent_dir / f"step{step}{suffix}.top"),
                    config.solvent_gamma,
                    initial,
                    final,
                    str(source_a / "solvent.top"),
                    str(source_b / "solvent.top"),
                    ligand=config.solvent_ligand_resname,
                )
        pmd.load_file(str(solvent_dir / "step1.top"), xyz=str(solvent_gro)).save(
            str(solvent_gro), overwrite=True
        )
        atoms, distance = ds.distance_restraint(
            str(solvent_gro), config.solvent_ligand_resname
        )
        ds.write_itp_restraints(
            atoms, distance, 1.0, 1.0, str(solvent_dir / "dist_restraint_eq.itp")
        )
        ds.write_itp_restraints(
            atoms, distance, 1000.0, 1000.0, str(solvent_dir / "dist_restraint.itp")
        )
        for step in (1, 2):
            br.include_itp_in_top(
                str(solvent_dir / f"step{step}.top"), "dist_restraint.itp"
            )
            br.include_itp_in_top(
                str(solvent_dir / f"step{step}_eq.top"), "dist_restraint_eq.itp"
            )

    metadata = edge / "restraint_corrections.json"
    metadata.write_text(
        json.dumps(
            {
                "ligand_a": ligand_a,
                "ligand_b": ligand_b,
                "boresch_dG_A_off": float(dg_a_off),
                "boresch_dG_B_on": float(dg_b_on),
                "restraint_atoms_a": [int(atom) for atom in restraint_a],
                "restraint_atoms_b": [int(atom) for atom in restraint_b],
            },
            indent=2,
        )
        + "\n"
    )
    return edge
