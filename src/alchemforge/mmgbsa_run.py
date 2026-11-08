"""Single-trajectory MM-GBSA energy evaluation (needs OpenMM + mdtraj).

For each frame of a complex trajectory this computes

    dG_frame = E_complex - E_receptor - E_ligand

in an implicit Generalized-Born + surface-area solvent, after stripping explicit
water and ions, and averages over frames. "Single trajectory" means the receptor
and ligand coordinates are taken from the complex frames (the usual MM-GBSA
approximation; it cancels most intramolecular terms and is cheap).

Generalized Born is set up generically so it works for a protein plus an
arbitrary small molecule / PROTAC: a vacuum system is built with the given force
field (the ligand parameterised through OpenMM-ForceFields if an SDF is given),
per-atom partial charges are read back from the NonbondedForce, and a
``GBSAOBCForce`` (OBC, igb=2/5-style) is added with element-based mbondi2 radii.

Run headlessly::

    python -m alchemforge.mmgbsa_run --config mmgbsa.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from alchemforge.mmgbsa_config import load_config, validate_config

# mbondi2 radii (Angstrom) and OBC scale factors, keyed by element symbol.
# Hydrogen is refined to 1.3 A when bonded to N/O (mbondi2), else 1.2 A.
_MBONDI2_RADII = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.5, "F": 1.5, "P": 1.85,
                  "S": 1.8, "Cl": 1.7, "Br": 1.85, "I": 1.98, "Mg": 1.18}
_OBC_SCALE = {"H": 0.85, "C": 0.72, "N": 0.79, "O": 0.85, "F": 0.88, "P": 0.86,
              "S": 0.96, "Cl": 0.80, "Br": 0.80, "I": 0.80, "Mg": 0.80}
_DEFAULT_RADIUS = 1.5
_DEFAULT_SCALE = 0.8


def _element_gb_params(topology):
    """(radius_nm, scale) per atom from mbondi2, with the H-on-N/O refinement."""
    params = []
    for atom in topology.atoms():
        symbol = (atom.element.symbol if atom.element is not None else "").strip()
        radius = _MBONDI2_RADII.get(symbol, _DEFAULT_RADIUS)
        scale = _OBC_SCALE.get(symbol, _DEFAULT_SCALE)
        if symbol == "H":
            neighbours = [b[1] if b[0].index == atom.index else b[0]
                          for b in topology.bonds() if atom.index in (b[0].index, b[1].index)]
            if any(n.element is not None and n.element.symbol in ("N", "O") for n in neighbours):
                radius = 1.3
        params.append((radius / 10.0, scale))      # Angstrom -> nm
    return params


def _build_context(topology, forcefield, conf):
    """Build an implicit-solvent OpenMM Context for ``topology``."""
    from openmm import GBSAOBCForce, NonbondedForce, Platform, VerletIntegrator
    from openmm.app import NoCutoff, Simulation

    system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff,
                                     constraints=None, rigidWater=False)

    nonbonded = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    charges = [nonbonded.getParticleParameters(i)[0]._value
               for i in range(nonbonded.getNumParticles())]

    gb = GBSAOBCForce()
    gb.setSoluteDielectric(float(conf.solute_dielectric))
    gb.setSolventDielectric(float(conf.solvent_dielectric))
    if not bool(getattr(conf, "surface_area", True)):
        gb.setSurfaceAreaEnergy(0.0)
    for charge, (radius, scale) in zip(charges, _element_gb_params(topology)):
        gb.addParticle(charge, radius, scale)
    system.addForce(gb)

    platform = Platform.getPlatformByName(str(getattr(conf, "platform", "CPU")))
    simulation = Simulation(topology, system, VerletIntegrator(0.001), platform)
    return simulation


def _potential_kcal(simulation, positions) -> float:
    from openmm.unit import kilocalorie_per_mole
    simulation.context.setPositions(positions)
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)


def run_from_config(config_path):
    import mdtraj as md
    import numpy as np
    from openmm.app import ForceField

    from chemflow.fe.openmm_modeling.membrane_relax import _register_small_molecule

    conf = load_config(config_path)
    validate_config(conf)
    workdir = Path(conf.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # ---- load trajectory, strip solvent/ions, split receptor/ligand ----
    traj = md.load(str(conf.trajectory), top=str(conf.topology))
    keep = traj.topology.select(f"not ({conf.strip_selection})")
    solute = traj.atom_slice(keep)
    lig_local = solute.topology.select(conf.ligand_selection)
    if len(lig_local) == 0:
        raise ValueError(f"ligand_selection matched no atoms: {conf.ligand_selection!r}")
    lig_set = set(int(i) for i in lig_local)
    rec_local = [a.index for a in solute.topology.atoms if a.index not in lig_set]

    complex_traj = solute
    ligand_traj = solute.atom_slice(sorted(lig_set))
    receptor_traj = solute.atom_slice(rec_local)

    # ---- frame selection ----
    start, stride, cap = int(conf.start_frame), int(conf.stride), int(conf.max_frames)
    frames = list(range(start, complex_traj.n_frames, stride))[:cap]
    if not frames:
        raise ValueError("no frames selected; check start_frame / stride / max_frames")

    # ---- build the three implicit-solvent contexts ----
    def _forcefield():
        ff = ForceField(*list(conf.forcefields))
        if getattr(conf, "ligand_sdf", None):
            _register_small_molecule(ff, conf.ligand_sdf, conf)
        return ff

    sims = {
        "complex": _build_context(complex_traj.topology.to_openmm(), _forcefield(), conf),
        "receptor": _build_context(receptor_traj.topology.to_openmm(), _forcefield(), conf),
        "ligand": _build_context(ligand_traj.topology.to_openmm(), _forcefield(), conf),
    }

    # ---- per-frame energies (mdtraj xyz is in nm, which OpenMM wants) ----
    from openmm.unit import nanometer

    per_frame = []
    for f in frames:
        e_cplx = _potential_kcal(sims["complex"], complex_traj.xyz[f] * nanometer)
        e_rec = _potential_kcal(sims["receptor"], receptor_traj.xyz[f] * nanometer)
        e_lig = _potential_kcal(sims["ligand"], ligand_traj.xyz[f] * nanometer)
        per_frame.append((f, e_cplx, e_rec, e_lig, e_cplx - e_rec - e_lig))

    dG = np.array([row[4] for row in per_frame], dtype=float)
    mean = float(dG.mean())
    sem = float(dG.std(ddof=1) / np.sqrt(len(dG))) if len(dG) > 1 else 0.0

    # ---- write per-frame + summary CSVs ----
    prefix = str(conf.output_prefix)
    frame_csv = workdir / f"{prefix}_frames.csv"
    with frame_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "E_complex", "E_receptor", "E_ligand", "dG"])
        writer.writerows(per_frame)

    summary_csv = workdir / f"{prefix}_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "dG_mean", "dG_sem", "n_frames",
                         "complex_mean", "receptor_mean", "ligand_mean"])
        writer.writerow([
            workdir.name,
            f"{mean:.3f}", f"{sem:.3f}", len(dG),
            f"{float(np.mean([r[1] for r in per_frame])):.3f}",
            f"{float(np.mean([r[2] for r in per_frame])):.3f}",
            f"{float(np.mean([r[3] for r in per_frame])):.3f}",
        ])

    print(f"MM-GBSA dG_bind = {mean:.2f} +/- {sem:.2f} kcal/mol over {len(dG)} frames")
    print(f"  per-frame: {frame_csv}")
    print(f"  summary:   {summary_csv}")
    return summary_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Single-trajectory MM-GBSA (OpenMM).")
    parser.add_argument("--config", required=True, help="MM-GBSA YAML configuration file")
    args = parser.parse_args(argv)
    run_from_config(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
