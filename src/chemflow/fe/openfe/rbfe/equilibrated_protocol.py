"""RBFE with the NVT pre-equilibration stage openfe's RBFE protocol lacks.

Why this exists
---------------
On the same membrane system, OpenFE's ABFE and SepTop protocols run while
``RelativeHybridTopologyProtocol`` dies with a NaN at ``Propagating replica 0 at
state 0``. The protocols differ in how much relaxation happens before alchemical
sampling starts:

* SepTop/ABFE call ``_pre_equilibrate`` (``openmm_septop/base_units.py:94``),
  which runs plain MD with an ``equilibration_length_nvt`` stage, then NPT, then
  a short production — on the non-alchemical endstate — and only then builds the
  alchemical system.
* RBFE (``openmm_rfe/hybridtop_units.py:1226-1248``) goes straight from
  ``sampler.minimize()`` to ``sampler.equilibrate()``, already under the
  sampler's NPT thermodynamic states. There is no NVT stage, and no
  ``equilibration_length_nvt`` field exists in its ``simulation_settings``
  (verified against openfe 1.12.0).

An OpenFE maintainer named this gap in openfe#1974: "We're currently not doing a
careful pre-equilibration of the lambda windows before the free energy
calculation which could potentially be a cause." It is not fixed in 1.13.0.

A dense bilayer around a freshly inserted hybrid ligand needs to relax at
constant volume before a barostat starts rescaling the box; otherwise the first
integration steps can diverge even though minimisation reported success.

What it does
------------
:class:`EquilibratedHybridTopologySetupUnit` overrides exactly one method,
``_get_alchemical_system``. After openfe builds the hybrid factory, it runs
minimisation + NVT MD on the hybrid system and writes the relaxed coordinates
back into ``hybrid_factory.hybrid_positions``. The stock setup unit's ``run()``
serialises those positions (``hybridtop_units.py:805``), so the simulation unit
starts from the equilibrated structure with nothing else changed.

Three deliberate choices:

* **The hybrid is equilibrated, not stateA.** Hybrid positions mix stateA
  coordinates (core and unique-old atoms) with stateB's (unique-new). Relaxing
  stateA alone and writing it back would desynchronise the unique-new atoms —
  whose bonds to the core are at full strength even at lambda=0 — which is the
  failure this is meant to prevent.
* **NVT only.** The hybrid system openfe builds carries no barostat (pressure
  comes from the sampler's thermodynamic states), so this stage is NVT by
  construction, and openfe's own ``equilibrate()`` still provides the NPT stage.
* **Classes are defined at module level.** gufe serialises a Protocol by
  qualified name; a class created inside a factory function could not be
  reimported when the JSON is executed on a compute node.

Durations come from the environment, so gufe's serialised settings are
untouched:

    CHEMFLOW_RFE_PREEQ_NVT_PS     NVT length in ps   (default 100; 0 disables)
    CHEMFLOW_RFE_PREEQ_MIN_STEPS  minimisation steps (default 5000; 0 disables)

Enable it with ``pre_equilibrate_complex: true`` in the OpenFE config; it
applies to the complex leg only.
"""

from __future__ import annotations

import logging
import os
import uuid

import openmm
from openfe.protocols.openmm_rfe import (
    HybridTopologyMultiStateAnalysisUnit,
    HybridTopologyMultiStateSimulationUnit,
    HybridTopologySetupUnit,
    RelativeHybridTopologyProtocol,
)
from openfe.protocols.openmm_utils import omm_compute, system_validation
from openff.units.openmm import to_openmm

logger = logging.getLogger(__name__)

DEFAULT_NVT_PS = 100.0
DEFAULT_MIN_STEPS = 5000

# Grep-able marker so the stage is easy to confirm in a SLURM log:
#     grep chemflow-preeq rbfe_*.out
TAG = "[chemflow-preeq]"


def _say(message: str) -> None:
    """Print AND log.

    openfe configures its own logging, so an INFO record from this module may
    never reach the job's stdout; a flushed print always does.
    """
    print(f"{TAG} {message}", flush=True)
    logger.info(message)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _energy(context) -> str:
    """Potential energy of a Context as a short string (for the log)."""
    try:
        value = context.getState(getEnergy=True).getPotentialEnergy()
        return f"{value.value_in_unit(openmm.unit.kilojoule_per_mole):,.1f} kJ/mol"
    except Exception as exc:
        return f"(unavailable: {exc})"


def _check_seam(unit_cls) -> None:
    """Fail loudly if the openfe method we override no longer exists."""
    if not hasattr(unit_cls, "_get_alchemical_system"):
        raise RuntimeError(
            "This openfe version has no HybridTopologySetupUnit."
            "_get_alchemical_system, which chemflow's equilibrated RBFE protocol "
            "overrides. Re-check openfe/protocols/openmm_rfe/hybridtop_units.py "
            "and update chemflow.fe.openfe.rbfe.equilibrated_protocol."
        )


_check_seam(HybridTopologySetupUnit)


class EquilibratedHybridTopologySetupUnit(HybridTopologySetupUnit):
    """Setup unit that relaxes the hybrid system before sampling starts."""

    def _equilibrate_hybrid(self, hybrid_system, positions, settings):
        """Minimise, then run NVT MD on the hybrid at lambda=0.

        Returns the relaxed positions. The hybrid system's alchemical global
        parameters are left at their defaults, which is the stateA endstate
        that the sampler's replica 0 starts from.
        """
        nvt_ps = _env_float("CHEMFLOW_RFE_PREEQ_NVT_PS", DEFAULT_NVT_PS)
        min_steps = int(_env_float("CHEMFLOW_RFE_PREEQ_MIN_STEPS", DEFAULT_MIN_STEPS))
        if nvt_ps <= 0 and min_steps <= 0:
            _say("pre-equilibration DISABLED (CHEMFLOW_RFE_PREEQ_NVT_PS and "
                 "CHEMFLOW_RFE_PREEQ_MIN_STEPS are both 0) — running exactly as "
                 "stock openfe would")
            return positions

        timestep = to_openmm(settings["integrator_settings"].timestep)
        temperature = to_openmm(settings["thermo_settings"].temperature)
        friction = to_openmm(settings["integrator_settings"].langevin_collision_rate)
        n_steps = int(round(nvt_ps / timestep.value_in_unit(openmm.unit.picosecond)))

        restrict_cpu = (
            settings["forcefield_settings"].nonbonded_method.lower() == "nocutoff"
        )
        platform = omm_compute.get_openmm_platform(
            platform_name=settings["engine_settings"].compute_platform,
            gpu_device_index=settings["engine_settings"].gpu_device_index,
            restrict_cpu_count=restrict_cpu,
        )
        _say("=" * 62)
        _say("ChemFlow NVT pre-equilibration of the HYBRID system at lambda=0")
        _say("(openfe's RBFE protocol has no NVT stage; SepTop/ABFE do)")
        _say(f"  platform      : {platform.getName()}")
        _say(f"  particles     : {hybrid_system.getNumParticles()}")
        _say(f"  minimisation  : {min_steps} steps")
        _say(f"  NVT           : {nvt_ps:g} ps = {n_steps} steps at "
             f"{timestep.value_in_unit(openmm.unit.femtosecond):g} fs, "
             f"{temperature.value_in_unit(openmm.unit.kelvin):g} K")

        integrator = openmm.LangevinMiddleIntegrator(temperature, friction, timestep)
        context = openmm.Context(hybrid_system, integrator, platform)
        try:
            context.setPositions(positions)
            _say(f"  energy before : {_energy(context)}")
            if min_steps > 0:
                openmm.LocalEnergyMinimizer.minimize(context, maxIterations=min_steps)
                _say(f"  after minimise: {_energy(context)}")
            if n_steps > 0:
                context.setVelocitiesToTemperature(temperature)
                integrator.step(n_steps)
                _say(f"  after NVT     : {_energy(context)}")
            state = context.getState(getPositions=True, getEnergy=True)
            energy = state.getPotentialEnergy()
            if energy != energy:  # NaN
                _say("FAILED: the hybrid system is NaN after pre-equilibration")
                raise RuntimeError(
                    "the hybrid system is NaN after pre-equilibration: the "
                    "starting structure or the mapping is the problem, not the "
                    "amount of equilibration")
            _say("pre-equilibration COMPLETE — sampling starts from these "
                 "coordinates")
            _say("=" * 62)
            return state.getPositions(asNumpy=True)
        finally:
            del context, integrator

    def _get_alchemical_system(self, **kwargs):
        hybrid_factory, hybrid_system = super()._get_alchemical_system(**kwargs)
        settings = self._get_settings(self._inputs["protocol"].settings)
        hybrid_factory.hybrid_positions = self._equilibrate_hybrid(
            hybrid_system, hybrid_factory.hybrid_positions, settings
        )
        return hybrid_factory, hybrid_system


class EquilibratedRelativeHybridTopologyProtocol(RelativeHybridTopologyProtocol):
    """RBFE protocol whose setup unit pre-equilibrates the hybrid system.

    Settings and results are identical to the stock protocol; only the setup
    unit differs, so existing configs and analysis code are unaffected.
    """

    def _create(self, stateA, stateB, mapping, extends=None):
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)
        alchem_comps = system_validation.get_alchemical_components(stateA, stateB)
        ligandmapping = mapping[0] if isinstance(mapping, list) else mapping
        a_names = ",".join(c.name for c in alchem_comps["stateA"])
        b_names = ",".join(c.name for c in alchem_comps["stateB"])

        setup_units, simulation_units, analysis_units = [], [], []
        for i in range(self.settings.protocol_repeats):
            repeat_id = int(uuid.uuid4())
            setup = EquilibratedHybridTopologySetupUnit(
                protocol=self,
                stateA=stateA,
                stateB=stateB,
                ligandmapping=ligandmapping,
                alchemical_components=alchem_comps,
                generation=0,
                repeat_id=repeat_id,
                name=(f"HybridTopology Setup (pre-equilibrated): {a_names} to "
                      f"{b_names} repeat {i} generation 0"),
            )
            simulation = HybridTopologyMultiStateSimulationUnit(
                protocol=self,
                setup_results=setup,
                generation=0,
                repeat_id=repeat_id,
                name=(f"HybridTopology Simulation: {a_names} to {b_names} "
                      f"repeat {i} generation 0"),
            )
            analysis = HybridTopologyMultiStateAnalysisUnit(
                protocol=self,
                setup_results=setup,
                simulation_results=simulation,
                generation=0,
                repeat_id=repeat_id,
                name=(f"HybridTopology Analysis: {a_names} to {b_names} "
                      f"repeat {i} generation 0"),
            )
            setup_units.append(setup)
            simulation_units.append(simulation)
            analysis_units.append(analysis)
        return [*setup_units, *simulation_units, *analysis_units]


def get_protocol_class(equilibrated: bool = True):
    """Return the RBFE protocol class to use.

    ``equilibrated=False`` gives openfe's stock protocol, so the same edge can
    be run with and without the extra stage for comparison.
    """
    return (EquilibratedRelativeHybridTopologyProtocol if equilibrated
            else RelativeHybridTopologyProtocol)
