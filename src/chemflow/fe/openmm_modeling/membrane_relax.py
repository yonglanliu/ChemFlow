"""Membrane relaxation protocol for OpenMM.

CHARMM-GUI-style staged equilibration of a PRE-TRIMMED membrane+protein(+ligand)
complex, run before OpenFE free-energy calculations. The input PDB's periodic box
is preserved as-is (no extra solvent/membrane padding is added), so a minimal
membrane patch stays minimal and the FEP system stays small.

  Protein restraint:
      k * periodicdistance(x, y, z, x0, y0, z0)^2
  Bilayer lipid head-group Z restraint:
      k * periodicdistance(0, 0, z, 0, 0, z0)^2

Initial energy minimization is unrestrained; restraints are added only for the
subsequent staged equilibration.

Configuration units:
  timestep / simulation_time : ps     temperature    : K
  force constants            : kJ/mol/nm^2            pressure : bar
  surface_tension            : bar*nm

Run headlessly with::

    python -m chemflow.fe.openmm_modeling.membrane_relax --config membrane_relax.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from sys import stdout
from types import SimpleNamespace

import numpy as np

from openmm import (
    CustomExternalForce,
    LangevinMiddleIntegrator,
    MonteCarloMembraneBarostat,
    Platform,
    VerletIntegrator,
)
from openmm.app import (
    CheckpointReporter,
    DCDReporter,
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
    PME,
    Simulation,
    StateDataReporter,
)
from openmm.unit import (
    amu,
    bar,
    kelvin,
    kilojoules_per_mole,
    nanometer,
    nanometers,
    picosecond,
    picoseconds,
)

from chemflow.fe.openmm_modeling.membrane_relax_config import load_config, validate_config

logger = logging.getLogger("membrane_relaxation")


# ============================================================
# Logging
# ============================================================


def setup_logger(log_file=None, level=logging.INFO):
    """Configure console and optional file logging without duplicate handlers."""
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


# ============================================================
# Residue names
# ============================================================

PROTEIN_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "ASH", "GLH", "LYN", "CYX", "ACE", "NME", "NMA",
}

# Three-character lipid residue names used by the trimmed membrane topology.
LIPID_RESNAMES = {
    "POP",  # POPC / POPE represented as POP
    "DLP",  # DLPC / DLPE
    "DMP",  # DMPC
    "DOP",  # DOPC
    "DPP",  # DPPC
}

WATER_ION_RESNAMES = {
    "HOH", "WAT", "NA", "CL", "K", "CA", "MG",
}


# ============================================================
# Generic helpers
# ============================================================


def write_pdb(topology, positions, filename):
    with open(filename, "w") as handle:
        PDBFile.writeFile(topology, positions, handle)


def normalize_namespace(value):
    """Return a plain dictionary for dict/SimpleNamespace-like objects."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, SimpleNamespace):
        return vars(value).copy()
    if hasattr(value, "__dict__"):
        return vars(value).copy()
    return {}


def as_int_steps(simulation_time_ps, timestep_ps):
    """Convert ps / ps into an integer number of integration steps."""
    if float(timestep_ps) <= 0.0:
        raise ValueError("timestep must be > 0 ps")
    return int(round(float(simulation_time_ps) / float(timestep_ps)))


def make_integrator(temperature_K, timestep_ps, friction_ps_inv, random_seed=None):
    integrator = LangevinMiddleIntegrator(
        float(temperature_K) * kelvin,
        float(friction_ps_inv) / picosecond,
        float(timestep_ps) * picoseconds,
    )
    if random_seed is not None:
        integrator.setRandomNumberSeed(int(random_seed))
    return integrator


def make_minimization_integrator(timestep_ps):
    """Integrator required by Simulation; unused for dynamics during minimization."""
    return VerletIntegrator(float(timestep_ps) * picoseconds)


def nested_force_constant(restraint_spec, key, default=0.0):
    raw = normalize_namespace(restraint_spec)
    value = raw.get(key, None)
    if value is None:
        return float(default)
    if isinstance(value, dict):
        value = value.get("force_constants", default)
    elif hasattr(value, "force_constants"):
        value = value.force_constants
    return float(value)


def nested_vector_force_constant(restraint_spec, key, component=2, default=0.0):
    raw = normalize_namespace(restraint_spec)
    value = raw.get(key, None)
    if value is None:
        return float(default)
    if isinstance(value, dict):
        value = value.get("force_constants", default)
    elif hasattr(value, "force_constants"):
        value = value.force_constants
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(
                f"{key}.force_constants must contain exactly three values [kx, ky, kz]"
            )
        return float(value[component])
    return float(value)


def stage_has_restraints(stage):
    restraints = getattr(stage, "restraints", None)
    return restraints is not None and restraints is not False


def get_platform(conf):
    platform_name = str(conf.platform)
    platform = Platform.getPlatformByName(platform_name)
    properties = {}
    if platform.getName().upper() == "CUDA":
        properties["Precision"] = "mixed"
        device_index = getattr(conf, "device_index", None)
        if device_index is not None:
            properties["DeviceIndex"] = str(device_index)
    logger.info("Using OpenMM platform: %s", platform.getName())
    logger.info("Platform properties: %s", properties)
    return platform, properties


# ============================================================
# CHARMM-GUI-style restraint forces
# ============================================================


def make_protein_restraint_force():
    """Protein XYZ positional restraint, matching CHARMM-GUI."""
    force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2;")
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    return force


def make_membrane_z_restraint_force(force_constant):
    """Bilayer Z-only positional restraint, matching CHARMM-GUI."""
    force = CustomExternalForce("k*periodicdistance(0, 0, z, 0, 0, z0)^2;")
    force.addGlobalParameter("k", float(force_constant))
    force.addPerParticleParameter("z0")
    return force


def is_protein_backbone_atom(atom):
    return atom.name.strip().upper() in {"N", "CA", "C", "O"}


def is_membrane_head_atom(atom):
    """Approximate CHARMM-GUI lipid head-group selection using P/N atoms."""
    atom_name = atom.name.strip().upper()
    return atom_name.startswith("P") or atom_name.startswith("N")


def add_protein_restraints(system, topology, reference_positions, restraint_spec):
    """Add separate protein backbone and sidechain XYZ restraints."""
    fc_bb = nested_force_constant(restraint_spec, "backbone")
    fc_sc = nested_force_constant(restraint_spec, "sidechain")

    legacy = nested_force_constant(restraint_spec, "solute_heavy_atom")
    if fc_bb <= 0.0 and fc_sc <= 0.0 and legacy > 0.0:
        fc_bb = legacy
        fc_sc = legacy

    if fc_bb <= 0.0 and fc_sc <= 0.0:
        return system

    force = make_protein_restraint_force()
    n_bb = 0
    n_sc = 0

    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol == "H":
            continue
        resname = atom.residue.name.strip().upper()
        if resname not in PROTEIN_RESNAMES:
            continue
        if is_protein_backbone_atom(atom):
            k = fc_bb
            if k <= 0.0:
                continue
            n_bb += 1
        else:
            k = fc_sc
            if k <= 0.0:
                continue
            n_sc += 1
        pos = reference_positions[atom.index]
        force.addParticle(atom.index, [k, pos.x, pos.y, pos.z])

    if force.getNumParticles() > 0:
        system.addForce(force)

    logger.info(
        'Restraint "protein_backbone": %d atoms, k = %.1f kJ/mol/nm^2, XYZ',
        n_bb, fc_bb,
    )
    logger.info(
        'Restraint "protein_sidechain": %d atoms, k = %.1f kJ/mol/nm^2, XYZ',
        n_sc, fc_sc,
    )
    return system


def add_ligand_restraints(system, topology, reference_positions, restraint_spec):
    """Add XYZ positional restraints to ligand heavy atoms."""
    k = nested_force_constant(restraint_spec, "ligand")
    if k <= 0.0:
        return system

    force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2;")
    force.addGlobalParameter("k", k)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    count = 0
    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol == "H":
            continue
        resname = atom.residue.name.strip().upper()
        if resname in PROTEIN_RESNAMES:
            continue
        if resname in LIPID_RESNAMES:
            continue
        if resname in WATER_ION_RESNAMES:
            continue
        pos = reference_positions[atom.index]
        force.addParticle(atom.index, [pos.x, pos.y, pos.z])
        count += 1

    if count > 0:
        system.addForce(force)

    logger.info(
        'Restraint "ligand": %d atoms, k = %.1f kJ/mol/nm^2, XYZ', count, k,
    )
    return system


def add_membrane_z_restraint(system, topology, reference_positions, restraint_spec, key):
    """Add a Z-only CHARMM-GUI-style restraint to lipid head-group atoms."""
    k = nested_vector_force_constant(restraint_spec, key, component=2)
    if k <= 0.0:
        return system

    force = make_membrane_z_restraint_force(k)
    count = 0
    for atom in topology.atoms():
        resname = atom.residue.name.strip().upper()
        if resname not in LIPID_RESNAMES:
            continue
        if not is_membrane_head_atom(atom):
            continue
        pos = reference_positions[atom.index]
        force.addParticle(atom.index, [pos.z])
        count += 1

    if count > 0:
        system.addForce(force)

    logger.info(
        'Restraint "%s_Z": %d atoms, k = %.1f kJ/mol/nm^2, Z', key, count, k,
    )
    if count == 0:
        logger.warning('Restraint group "%s" selected 0 lipid head-group atoms.', key)
    return system


def add_stage_restraints(system, topology, reference_positions, restraint_spec):
    """Add exactly the restraint groups present in the stage configuration."""
    if restraint_spec is None or restraint_spec is False:
        logger.info("Restraints: none")
        return system

    raw = normalize_namespace(restraint_spec)
    if not raw:
        logger.info("Restraints: none")
        return system

    if any(key in raw for key in ("backbone", "sidechain", "solute_heavy_atom")):
        system = add_protein_restraints(
            system, topology, reference_positions, restraint_spec
        )
    if "ligand" in raw:
        system = add_ligand_restraints(
            system, topology, reference_positions, restraint_spec
        )
    if "membrane" in raw:
        system = add_membrane_z_restraint(
            system, topology, reference_positions, restraint_spec, "membrane"
        )
    if "membrane_head" in raw:
        system = add_membrane_z_restraint(
            system, topology, reference_positions, restraint_spec, "membrane_head"
        )
    return system


# ============================================================
# Diagnostics and stage logging
# ============================================================


def report_system_forces(system):
    logger.info("System has %d forces", system.getNumForces())
    for i in range(system.getNumForces()):
        logger.info("Force %d: %s", i, type(system.getForce(i)).__name__)


def report_state_diagnostics(simulation, label="state"):
    state = simulation.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, getForces=True,
    )
    positions = state.getPositions(asNumpy=True).value_in_unit(nanometers)
    forces = state.getForces(asNumpy=True).value_in_unit(
        kilojoules_per_mole / nanometer
    )
    magnitudes = np.linalg.norm(forces, axis=1)

    logger.info("%s potential energy: %s", label, state.getPotentialEnergy())
    logger.info("%s maximum force: %.6f kJ/mol/nm", label, magnitudes.max())
    logger.info("%s max-force atom: %d", label, int(magnitudes.argmax()))
    logger.info("%s NaN coordinates: %s", label, np.isnan(positions).any())
    logger.info("%s infinite coordinates: %s", label, np.isinf(positions).any())
    return state


def test_first_steps(simulation, n=1):
    simulation.step(int(n))
    logger.info("%d step(s) completed successfully", int(n))


def log_stage_info(stage, n_steps=None, temperature=None, duration_ps=None):
    """Log ensemble, timestep, temperature, duration, pressure and restraints."""
    if temperature is None:
        temperature = getattr(stage, "temperature", None)
    if duration_ps is None:
        duration_ps = getattr(stage, "simulation_time", None)
    if n_steps is None and duration_ps is not None:
        n_steps = as_int_steps(duration_ps, stage.timestep)

    logger.info("=" * 60)
    logger.info("%s", stage.title)
    logger.info("=" * 60)
    logger.info(
        "Ensemble: %s; timestep: %.4f ps; temperature: %.2f K; "
        "simulation time: %.1f ps; steps: %d",
        getattr(stage, "ensemble", "N/A"),
        float(stage.timestep),
        float(temperature) if temperature is not None else float("nan"),
        float(duration_ps) if duration_ps is not None else float("nan"),
        int(n_steps) if n_steps is not None else 0,
    )

    if str(getattr(stage, "ensemble", "")).upper() == "NPGT":
        logger.info(
            "Pressure: %.3f bar; surface tension: %.3f bar*nm",
            float(stage.pressure),
            float(stage.surface_tension),
        )

    restraints = getattr(stage, "restraints", None)
    if restraints is None or restraints is False:
        logger.info("Restraints: none")
        return
    restraint_dict = normalize_namespace(restraints)
    if not restraint_dict:
        logger.info("Restraints: none")
        return

    for name, config in restraint_dict.items():
        if isinstance(config, dict):
            fc = config.get("force_constants")
        elif hasattr(config, "force_constants"):
            fc = config.force_constants
        else:
            fc = config
        if isinstance(fc, (list, tuple)):
            logger.info(
                "Restraint: %s; k: X=%.1f, Y=%.1f, Z=%.1f kJ/mol/nm^2",
                name, float(fc[0]), float(fc[1]), float(fc[2]),
            )
        else:
            logger.info(
                "Restraint: %s; k: %.1f kJ/mol/nm^2", name, float(fc),
            )


# ============================================================
# System creation
# ============================================================


def create_base_system(forcefield, topology):
    """Create the base PME System without a barostat."""
    if not isinstance(forcefield, ForceField):
        raise TypeError(
            "forcefield must be an openmm.app.ForceField instance, "
            f"not {type(forcefield)!r}."
        )
    return forcefield.createSystem(
        topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
        rigidWater=True,
        hydrogenMass=3.0 * amu,
    )


def add_membrane_barostat(system, stage, random_seed=None):
    pressure = getattr(stage, "pressure", None)
    surface_tension = getattr(stage, "surface_tension", None)
    if pressure is None or surface_tension is None:
        return system
    barostat = MonteCarloMembraneBarostat(
        float(pressure) * bar,
        float(surface_tension) * bar * nanometer,
        float(stage.temperature) * kelvin,
        MonteCarloMembraneBarostat.XYIsotropic,
        MonteCarloMembraneBarostat.ZFree,
    )
    if random_seed is not None:
        barostat.setRandomNumberSeed(int(random_seed))
    system.addForce(barostat)
    return system


def build_simulation(
    modeller,
    forcefield,
    stage,
    conf,
    platform,
    platform_properties,
    reference_positions=None,
    add_restraints=False,
    add_barostat=False,
    velocities=None,
):
    """Build a fresh OpenMM Simulation from a clean System."""
    system = create_base_system(forcefield, modeller.topology)

    if add_restraints:
        system = add_stage_restraints(
            system, modeller.topology, reference_positions,
            getattr(stage, "restraints", None),
        )
    if add_barostat:
        system = add_membrane_barostat(
            system, stage, random_seed=getattr(conf, "seed", None)
        )

    integrator = make_integrator(
        stage.temperature, stage.timestep, conf.friction, random_seed=conf.seed,
    )
    simulation = Simulation(
        modeller.topology, system, integrator, platform, platform_properties,
    )
    if reference_positions is not None:
        simulation.context.setPositions(reference_positions)
    if velocities is not None:
        simulation.context.setVelocities(velocities)

    logger.info(
        "Actual simulation platform: %s",
        simulation.context.getPlatform().getName(),
    )
    report_system_forces(system)
    return simulation


def attach_reporters(
    simulation,
    prefix,
    report_interval=5000,
    checkpoint_interval=50000,
    total_steps=None,
    stdout_log=True,
):
    simulation.reporters.append(DCDReporter(f"{prefix}.dcd", report_interval))
    simulation.reporters.append(
        CheckpointReporter(f"{prefix}.chk", checkpoint_interval)
    )
    if stdout_log:
        simulation.reporters.append(
            StateDataReporter(
                stdout, report_interval, step=True, time=True,
                potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                temperature=True, volume=True, density=True,
                progress=(total_steps is not None),
                remainingTime=(total_steps is not None),
                speed=True,
                totalSteps=total_steps if total_steps is not None else 0,
                separator=",",
            )
        )
    simulation.reporters.append(
        StateDataReporter(
            f"{prefix}.csv", report_interval, step=True, time=True,
            potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
            temperature=True, volume=True, density=True, separator=",",
        )
    )


# ============================================================
# Input assembly (preserve the trimmed membrane box; no padding)
# ============================================================


def _register_small_molecule(forcefield, ligand_sdf, conf):
    """Register a template generator so ligand atoms in the PDB are parameterized."""
    try:
        from openff.toolkit import Molecule
        from openmmforcefields.generators import (
            GAFFTemplateGenerator,
            SMIRNOFFTemplateGenerator,
        )
    except ImportError as error:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "Parameterizing a ligand needs 'openff-toolkit' and "
            "'openmmforcefields'. Install them, or drop 'ligand_sdf' and supply "
            "a ligand force-field XML in 'forcefields' instead."
        ) from error

    molecules = Molecule.from_file(str(ligand_sdf), allow_undefined_stereo=True)
    if not isinstance(molecules, list):
        molecules = [molecules]

    ff_name = str(getattr(conf, "small_molecule_forcefield", "gaff-2.11"))
    lowered = ff_name.lower()
    if lowered.startswith(("openff", "smirnoff", "sage")):
        generator = SMIRNOFFTemplateGenerator(
            molecules=molecules, forcefield=ff_name
        )
    else:
        generator = GAFFTemplateGenerator(molecules=molecules, forcefield=ff_name)

    forcefield.registerTemplateGenerator(generator.generator)
    logger.info(
        "Registered %d ligand template(s) via %s", len(molecules), ff_name
    )


def build_system_inputs(conf):
    """Build a Modeller + ForceField from the config, preserving the input box."""
    input_pdb = Path(conf.input_pdb)
    pdb = PDBFile(str(input_pdb))
    modeller = Modeller(pdb.topology, pdb.positions)

    box = pdb.topology.getPeriodicBoxVectors()
    if box is None:
        raise ValueError(
            f"Input PDB has no periodic box (missing CRYST1): {input_pdb}. "
            "A trimmed membrane system must carry its own box; this protocol "
            "does not add solvent/membrane padding."
        )
    logger.info(
        "Preserving input periodic box (no padding added): %s",
        box,
    )

    forcefield = ForceField(*list(conf.forcefields))
    ligand_sdf = getattr(conf, "ligand_sdf", None)
    if ligand_sdf:
        _register_small_molecule(forcefield, ligand_sdf, conf)

    return modeller, forcefield


# ============================================================
# Main relaxation workflow
# ============================================================


def relax(modeller, forcefield, conf, debug_first_step=False):
    """Run the configured membrane relaxation protocol."""
    if not isinstance(forcefield, ForceField):
        raise TypeError(f"forcefield must be ForceField, got {type(forcefield)!r}")
    if not hasattr(conf, "friction"):
        raise AttributeError("conf must define friction")

    workdir = Path(getattr(conf, "workdir", ".")).expanduser().resolve()
    if not workdir.exists():
        raise FileNotFoundError(f"Working directory does not exist: {workdir}")
    if not workdir.is_dir():
        raise NotADirectoryError(f"Working directory is not a directory: {workdir}")

    setup_logger(log_file=workdir / "membrane_relaxation.log")
    logger.info("Working directory: %s", workdir)

    platform, platform_properties = get_platform(conf)

    # -------------------------------------------------- unrestrained minimization
    minimization = conf.minimization
    stage1 = conf.stage1

    logger.info("=" * 60)
    logger.info("%s: %d minimization steps", minimization.title, int(minimization.steps))
    logger.info("Initial energy minimization is UNRESTRAINED")
    logger.info("=" * 60)

    system = create_base_system(forcefield, modeller.topology)
    minimization_integrator = make_minimization_integrator(stage1.timestep)
    simulation = Simulation(
        modeller.topology, system, minimization_integrator,
        platform, platform_properties,
    )
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=int(minimization.steps))

    minimized_state = simulation.context.getState(
        getPositions=True, getEnergy=True, getForces=True,
    )
    report_state_diagnostics(simulation, "After minimization")
    minimized_positions = minimized_state.getPositions()
    minimized_pdb = workdir / minimization.out_pdb
    write_pdb(simulation.topology, minimized_positions, minimized_pdb)
    logger.info("Saved PDB after unrestrained minimization: %s", minimized_pdb)

    # -------------------------------------------------- Stage 1: restrained NVT
    stage = stage1
    stage_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=stage_steps)

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=minimized_positions,
        add_restraints=True, add_barostat=False,
    )
    simulation.context.setVelocitiesToTemperature(float(stage.temperature) * kelvin)

    if debug_first_step:
        report_state_diagnostics(simulation, "Before Stage 1 NVT")
        test_first_steps(simulation, 1)

    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=stage_steps, stdout_log=True,
    )
    logger.info("Running Stage 1 NVT...")
    simulation.step(stage_steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    write_pdb(simulation.topology, state.getPositions(), workdir / stage.out_pdb)

    # -------------------------------------------------- Stage 2: restrained NPgT
    stage = conf.stage2
    stage_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=stage_steps)
    previous_velocities = state.getVelocities()

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=state.getPositions(),
        add_restraints=True, add_barostat=True, velocities=previous_velocities,
    )
    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=stage_steps, stdout_log=True,
    )
    logger.info("Running Stage 2 restrained NPgT...")
    simulation.step(stage_steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    write_pdb(simulation.topology, state.getPositions(), workdir / stage.out_pdb)

    # -------------------------------------------------- Stage 3: restrained NPgT
    stage = conf.stage3
    stage_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=stage_steps)

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=state.getPositions(),
        add_restraints=True, add_barostat=True, velocities=state.getVelocities(),
    )
    if state.getVelocities() is not None:
        simulation.context.setVelocities(state.getVelocities())

    logger.info("Stage 3: short restrained minimization")
    simulation.minimizeEnergy(maxIterations=500)
    report_state_diagnostics(simulation, "Stage 3 after restrained minimization")

    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=stage_steps, stdout_log=True,
    )
    logger.info("Running Stage 3 restrained NPgT...")
    simulation.step(stage_steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    write_pdb(simulation.topology, state.getPositions(), workdir / stage.out_pdb)

    # -------------------------------------------------- Stage 4: NPgT anneal schedule
    stage = conf.stage4
    schedule = stage.temperature_schedule

    logger.info("=" * 60)
    logger.info("%s", stage.title)
    logger.info("=" * 60)
    logger.info("Stage 4 consists of %d temperature segments", len(schedule))

    for schedule_entry in schedule:
        temperature = float(schedule_entry.temperature)
        duration_ps = float(schedule_entry.duration)
        steps = as_int_steps(duration_ps, stage.timestep)
        logger.info(
            "Stage 4 segment: ensemble=%s; timestep=%.4f ps; temperature=%.2f K; "
            "duration=%.1f ps; steps=%d",
            stage.ensemble, stage.timestep, temperature, duration_ps, steps,
        )
        segment_stage = SimpleNamespace(**vars(stage))
        segment_stage.temperature = temperature
        log_stage_info(
            segment_stage, n_steps=steps,
            temperature=temperature, duration_ps=duration_ps,
        )
        simulation = build_simulation(
            modeller, forcefield, segment_stage, conf, platform, platform_properties,
            reference_positions=state.getPositions(),
            add_restraints=True, add_barostat=True, velocities=state.getVelocities(),
        )
        attach_reporters(
            simulation, prefix=str(workdir / f"{stage.name}_{temperature:g}K"),
            total_steps=steps, stdout_log=True,
        )
        logger.info("Running Stage 4 segment at %.2f K...", temperature)
        simulation.step(steps)
        state = simulation.context.getState(getPositions=True, getVelocities=True)

    write_pdb(simulation.topology, state.getPositions(), workdir / stage.out_pdb)

    # -------------------------------------------------- Stage 5: restrained NVT
    stage = conf.stage5
    stage_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=stage_steps)

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=state.getPositions(),
        add_restraints=True, add_barostat=False, velocities=state.getVelocities(),
    )
    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=stage_steps, stdout_log=True,
    )
    logger.info("Running Stage 5 restrained NVT...")
    simulation.step(stage_steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    write_pdb(simulation.topology, state.getPositions(), workdir / stage.out_pdb)

    # -------------------------------------------------- Stage 6: unrestrained NVT
    stage = conf.stage6
    stage_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=stage_steps)

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=state.getPositions(),
        add_restraints=False, add_barostat=False, velocities=state.getVelocities(),
    )
    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=stage_steps, stdout_log=True,
    )
    logger.info("Running Stage 6 unrestrained NVT...")
    simulation.step(stage_steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    write_pdb(simulation.topology, state.getPositions(), workdir / stage.out_pdb)

    # -------------------------------------------------- Production: unrestrained NPgT
    stage = conf.simulation
    prod_steps = as_int_steps(stage.simulation_time, stage.timestep)
    log_stage_info(stage, n_steps=prod_steps)

    simulation = build_simulation(
        modeller, forcefield, stage, conf, platform, platform_properties,
        reference_positions=state.getPositions(),
        add_restraints=False, add_barostat=True, velocities=state.getVelocities(),
    )
    attach_reporters(
        simulation, prefix=str(workdir / stage.name),
        total_steps=prod_steps, stdout_log=True,
    )
    logger.info("Running unrestrained production NPgT...")
    simulation.step(prod_steps)
    final_state = simulation.context.getState(getPositions=True, getVelocities=True)

    production_pdb = workdir / stage.out_pdb
    write_pdb(simulation.topology, final_state.getPositions(), production_pdb)
    # Also write the final relaxed system as mmCIF (OpenFE-ready; no PDB
    # >99,999-atom serial overflow for large membrane systems).
    from openmm.app import PDBxFile

    production_cif = production_pdb.with_suffix(".cif")
    with open(production_cif, "w") as handle:
        PDBxFile.writeFile(simulation.topology, final_state.getPositions(), handle, keepIds=True)
    simulation.saveState(str(workdir / stage.out_state))
    simulation.saveCheckpoint(str(workdir / stage.chk))

    logger.info("Saved final relaxed PDB: %s", production_pdb)
    logger.info("Saved final relaxed mmCIF: %s", production_cif)
    logger.info("Saved final state: %s", workdir / stage.out_state)
    logger.info("Saved final checkpoint: %s", workdir / stage.chk)
    logger.info("Membrane relaxation protocol completed successfully.")
    return production_pdb


def run_from_config(config_path, debug_first_step=False):
    """Load + validate a YAML config, build inputs, and run the relaxation."""
    conf = load_config(config_path)
    validate_config(conf)
    modeller, forcefield = build_system_inputs(conf)
    return relax(modeller, forcefield, conf, debug_first_step=debug_first_step)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CHARMM-GUI-style membrane relaxation for OpenMM.",
    )
    parser.add_argument(
        "--config", required=True, help="Membrane-relax YAML configuration file"
    )
    parser.add_argument(
        "--debug-first-step", action="store_true",
        help="Run one diagnostic step before Stage 1",
    )
    args = parser.parse_args(argv)
    run_from_config(Path(args.config), debug_first_step=args.debug_first_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
