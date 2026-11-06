"""Unified MD runner for OpenMM.

Plain, unrestrained molecular dynamics for a prepared/relaxed complex, soluble or
membrane:

    energy minimization -> NVT equilibration -> NPT equilibration -> production

``system_type`` selects the production/NPT barostat:
  * ``soluble``  -> isotropic ``MonteCarloBarostat``
  * ``membrane`` -> ``MonteCarloMembraneBarostat`` (XY-isotropic, Z-free, NPgT)

The input PDB must carry a periodic box (CRYST1). Trajectory (DCD), energy log
(CSV), final PDB, state and checkpoint are written under the work directory.

Run headlessly with::

    python -m alchemforge.openmm_modeling.md_run --config md.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openmm import MonteCarloBarostat, MonteCarloMembraneBarostat
from openmm.app import ForceField, Modeller, PDBFile, Simulation
from openmm.unit import bar, kelvin, nanometer

# Reuse the relax engine helpers (shared OpenMM plumbing).
from alchemforge.openmm_modeling.membrane_relax import (
    _register_small_molecule,
    as_int_steps,
    attach_reporters,
    create_base_system,
    get_platform,
    logger,
    make_integrator,
    make_minimization_integrator,
    report_state_diagnostics,
    setup_logger,
    write_pdb,
)
from alchemforge.openmm_modeling.md_config import load_config, validate_config


def _add_barostat(system, conf, temperature):
    """Attach the pressure coupling appropriate for the system type."""
    if str(conf.system_type) == "membrane":
        barostat = MonteCarloMembraneBarostat(
            float(conf.pressure) * bar,
            float(conf.surface_tension) * bar * nanometer,
            float(temperature) * kelvin,
            MonteCarloMembraneBarostat.XYIsotropic,
            MonteCarloMembraneBarostat.ZFree,
        )
    else:
        barostat = MonteCarloBarostat(
            float(conf.pressure) * bar,
            float(temperature) * kelvin,
        )
    seed = getattr(conf, "seed", None)
    if seed is not None:
        barostat.setRandomNumberSeed(int(seed))
    system.addForce(barostat)
    return system


def _build_simulation(modeller, forcefield, conf, stage, platform, props, *,
                      add_barostat, positions, velocities=None):
    system = create_base_system(forcefield, modeller.topology)
    if add_barostat:
        _add_barostat(system, conf, stage.temperature)
    integrator = make_integrator(
        stage.temperature, stage.timestep, conf.friction, random_seed=conf.seed
    )
    simulation = Simulation(modeller.topology, system, integrator, platform, props)
    simulation.context.setPositions(positions)
    if velocities is not None:
        simulation.context.setVelocities(velocities)
    return simulation


def run_from_config(config_path):
    """Load + validate a YAML config and run the MD protocol."""
    conf = load_config(config_path)
    validate_config(conf)

    workdir = Path(conf.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file=workdir / "md.log")
    logger.info("MD working directory: %s", workdir)
    logger.info("System type: %s", conf.system_type)

    pdb = PDBFile(str(conf.input_pdb))
    if pdb.topology.getPeriodicBoxVectors() is None:
        raise ValueError(
            f"Input PDB has no periodic box (missing CRYST1): {conf.input_pdb}. "
            "Solvate/build the system first (build-soluble or build-membrane)."
        )
    modeller = Modeller(pdb.topology, pdb.positions)

    forcefield = ForceField(*list(conf.forcefields))
    if getattr(conf, "ligand_sdf", None):
        _register_small_molecule(forcefield, conf.ligand_sdf, conf)

    platform, props = get_platform(conf)
    temperature = float(conf.temperature)
    prefix = str(conf.output_prefix)
    report_interval = int(conf.report_interval)
    checkpoint_interval = int(conf.checkpoint_interval)

    # ---- give every equilibration/production stage the global temperature ----
    for key in ("nvt", "npt", "production"):
        setattr(getattr(conf, key), "temperature", temperature)

    # -------------------------------------------------- minimization (unrestrained)
    logger.info("=" * 60)
    logger.info("Energy minimization: %d steps", int(conf.minimization.steps))
    system = create_base_system(forcefield, modeller.topology)
    simulation = Simulation(
        modeller.topology, system,
        make_minimization_integrator(conf.nvt.timestep), platform, props,
    )
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=int(conf.minimization.steps))
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    report_state_diagnostics(simulation, "After minimization")
    write_pdb(simulation.topology, positions, workdir / f"{prefix}_minimized.pdb")

    # -------------------------------------------------- NVT equilibration
    stage = conf.nvt
    steps = as_int_steps(stage.simulation_time, stage.timestep)
    logger.info("NVT equilibration: %.1f ps (%d steps) at %.1f K",
                float(stage.simulation_time), steps, temperature)
    simulation = _build_simulation(
        modeller, forcefield, conf, stage, platform, props,
        add_barostat=False, positions=positions,
    )
    simulation.context.setVelocitiesToTemperature(temperature * kelvin)
    attach_reporters(simulation, prefix=str(workdir / f"{prefix}_nvt"),
                     report_interval=report_interval,
                     checkpoint_interval=checkpoint_interval, total_steps=steps)
    simulation.step(steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    positions, velocities = state.getPositions(), state.getVelocities()

    # -------------------------------------------------- NPT equilibration
    stage = conf.npt
    steps = as_int_steps(stage.simulation_time, stage.timestep)
    logger.info("NPT equilibration: %.1f ps (%d steps)", float(stage.simulation_time), steps)
    simulation = _build_simulation(
        modeller, forcefield, conf, stage, platform, props,
        add_barostat=True, positions=positions, velocities=velocities,
    )
    attach_reporters(simulation, prefix=str(workdir / f"{prefix}_npt"),
                     report_interval=report_interval,
                     checkpoint_interval=checkpoint_interval, total_steps=steps)
    simulation.step(steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    positions, velocities = state.getPositions(), state.getVelocities()
    write_pdb(simulation.topology, positions, workdir / f"{prefix}_equilibrated.pdb")

    # -------------------------------------------------- production
    stage = conf.production
    steps = as_int_steps(stage.simulation_time, stage.timestep)
    logger.info("Production: %.1f ps (%d steps)", float(stage.simulation_time), steps)
    simulation = _build_simulation(
        modeller, forcefield, conf, stage, platform, props,
        add_barostat=True, positions=positions, velocities=velocities,
    )
    attach_reporters(simulation, prefix=str(workdir / f"{prefix}_production"),
                     report_interval=report_interval,
                     checkpoint_interval=checkpoint_interval, total_steps=steps)
    simulation.step(steps)
    final = simulation.context.getState(getPositions=True, getVelocities=True)

    production_pdb = workdir / f"{prefix}_production.pdb"
    write_pdb(simulation.topology, final.getPositions(), production_pdb)
    simulation.saveState(str(workdir / f"{prefix}_production.xml"))
    simulation.saveCheckpoint(str(workdir / f"{prefix}_production.chk"))
    logger.info("Saved production PDB: %s", production_pdb)
    logger.info("MD protocol completed successfully.")
    return production_pdb


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Unrestrained MD (soluble or membrane) for OpenMM.",
    )
    parser.add_argument("--config", required=True, help="MD YAML configuration file")
    args = parser.parse_args(argv)
    run_from_config(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
