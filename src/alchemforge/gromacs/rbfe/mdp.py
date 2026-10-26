from __future__ import annotations

import re
from typing import TYPE_CHECKING

from alchemforge.gromacs.lambda_schedule import generate_lambda_windows

if TYPE_CHECKING:
    from .config import LigandRBFEConfig


rbfe_mdp = """
; ================================================================
; RBFE Production Run .mdp — Fully Annotated
; ================================================================

; ---------- RUN CONTROL ----------
integrator               = sd
; sd = stochastic dynamics (Langevin). Preferred for FEP because it
; provides built-in thermostatting with good sampling properties and
; stability during alchemical transformations. md-vv is an alternative
; if you need strict NVE-like behavior, but sd is the FEP community standard.

dt                        = 0.002
; 2 fs timestep. Requires h-bonds (or all-bonds) constraints to be stable.
; Do NOT increase above 0.002 unless using hydrogen mass repartitioning (HMR).

nsteps                    = 2500000
; Total steps for the run. 2,500,000 × 0.002 ps = 5000 ps = 5 ns.
; Adjust based on system size/convergence needs (5–20 ns/window is typical).

nstcomm                   = 100
; Frequency (steps) for removing center-of-mass motion. Prevents
; "flying ice cube" effect where kinetic energy drains into COM translation.

; ---------- OUTPUT CONTROL ----------
nstxout                   = 0
nstvout                   = 0
nstfout                   = 0
; Disable full-precision trajectory/velocity/force output to save disk
; space — not needed for FEP analysis, only dH/dλ values are needed.

nstlog                    = 5000
; How often (steps) to write to the .log file. 5000 steps = every 10 ps.

nstenergy                 = 1000
; How often to save energies to .edr file for later analysis (every 2 ps).

nstxout-compressed        = 5000
; Compressed trajectory (.xtc) output frequency — useful for visual
; inspection/RMSD checks without the disk cost of full precision.

; ---------- NEIGHBOR SEARCHING ----------
cutoff-scheme             = Verlet
; Required scheme in modern GROMACS (group scheme is deprecated).
; Enables better parallelization and GPU acceleration.

nstlist                   = 20
; Update neighbor list every 20 steps. Verlet scheme auto-tunes buffer
; so this can be larger than old "group scheme" defaults (safe up to ~40).

rlist                     = 1.0
; Neighbor list cutoff radius (nm). Should match or slightly exceed
; rcoulomb/rvdw; Verlet scheme adjusts this automatically if needed.

pbc                       = xyz
; Periodic boundary conditions in all three dimensions — standard for
; solvated simulations.

; ---------- ELECTROSTATICS / VDW ----------
coulombtype               = PME
; Particle Mesh Ewald — required for accurate long-range electrostatics
; in periodic systems. Essential for charged/polar ligand perturbations.

rcoulomb                  = 1.0
; Real-space cutoff (nm) for short-range Coulomb interactions before
; PME takes over for long-range part.

vdwtype                   = Cut-off
; Simple cutoff for van der Waals. Common in FEP since vdW is already
; perturbed via soft-core; avoid Potential-shift complications unless
; your force field (e.g., CHARMM) specifically calls for switch functions.

vdw-modifier              = Potential-shift-Verlet
; Amber-family defaults use the Verlet potential shift. CHARMM36-family
; force fields are rendered with a force switch over their prescribed range.

rvdw-switch               = 0.0
; Start of the CHARMM force-switch interval; unused for Amber-family settings.

rvdw                      = 1.0
; VdW cutoff radius (nm), typically matched to rcoulomb for simplicity
; and to satisfy force-field-recommended settings (e.g., 1.0 nm for Amber/OPLS,
; sometimes 1.2 nm for CHARMM).

DispCorr                  = EnerPres
; Long-range dispersion correction for energy and pressure — corrects
; for truncation error from the vdW cutoff. Important for consistent
; pressure/density, especially when comparing states with different vdW sizes.

; ---------- TEMPERATURE COUPLING ----------
tcoupl                    = V-rescale
; Velocity-rescaling thermostat (ignored when integrator=sd, since sd
; has its own built-in thermostat — included here in case you switch
; to integrator=md for equilibration stages).

tc-grps                   = System
; Which groups get independent temperature coupling. Using "System"
; (everything together) is fine for most RBFE setups since ligand is
; a small part of a solvated/protein system.

tau-t                     = 1.0
; Coupling time constant (ps) — how strongly/quickly temperature is
; corrected. 1.0–2.0 ps is standard; too small causes poor sampling,
; too large allows temperature drift.

ref-t                     = 298.15
; Reference temperature (K) — 298.15 K = 25°C, standard experimental
; comparison temperature. Match to your experimental binding data conditions.

; ---------- PRESSURE COUPLING ----------
pcoupl                    = C-rescale
; Stochastic cell rescaling barostat — modern replacement for
; Berendsen/Parrinello-Rahman with better pressure control for
; equilibrium production runs. (Use Berendsen only in early equilibration.)

pcoupltype                = isotropic
; Assumes uniform scaling in all directions — appropriate for a simple
; cubic/dodecahedral water box without membrane components.
; (Use semiisotropic if simulating a lipid bilayer system.)

tau-p                     = 5.0
; Pressure coupling time constant (ps). Longer than tau-t since pressure
; equilibrates more slowly than temperature; typical range 2–5 ps.

ref-p                     = 1.0
; Reference pressure (bar) — standard atmospheric pressure.

compressibility           = 4.5e-5
; Isothermal compressibility of water (bar⁻¹) at ~300K — tells the
; barostat how much volume change to expect per unit pressure change.

; ---------- CONSTRAINTS ----------
constraints               = h-bonds
; Constrains only bonds involving hydrogen (not heavy-atom bonds).
; Allows 2 fs timestep safely. Use "all-bonds" only if you have
; specific force field requirements.

constraint-algorithm      = LINCS
; Linear Constraint Solver — faster and more parallelizable than SHAKE;
; the GROMACS standard choice.

lincs-iter                = 1
; Number of iterations for constraint accuracy correction. 1 is
; sufficient for h-bonds constraints at 2 fs.

lincs-order               = 4
; Order of the matrix expansion used in LINCS. 4 is the standard,
; stable default; increase only for very stiff/large constrained systems.

; ================================================================
; ---------- FREE ENERGY / ALCHEMICAL PARAMETERS ----------
; ================================================================

free-energy               = yes
; Master switch — enables all alchemical perturbation machinery.

init-lambda-state         = 0
; Index into the lambda vectors below (0-based). THIS IS THE ONE
; PARAMETER YOU CHANGE PER WINDOW/SIMULATION FOLDER (0, 1, 2, ... N-1).
; Each lambda window is a separate GROMACS run with only this value differing.

delta-lambda              = 0
; Set to 0 for standard FEP (lambda fixed per window). Nonzero values
; would continuously change lambda during the run (slow-growth method,
; rarely used for RBFE — stick with 0).

; --- Lambda vectors (one column per window; index matches init-lambda-state) ---
coul-lambdas              = 0.00 0.00 0.00 0.00 0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 1.00
; Controls electrostatic (Coulomb) scaling of the perturbed/dummy atoms.
; Generated schedules keep charges off during the first portion, then ramp
; them after vdW is introduced. Manual schedules use the values supplied in
; the TOML configuration.

vdw-lambdas               = 0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
; Controls van der Waals scaling of perturbed atoms. Ramped up first
; while Coulomb is off, then held at 1.0 for the rest.
; Soft-core potentials (below) make this scaling smooth and non-singular.

bonded-lambdas            = 0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
; Controls interpolation of bonded terms (bonds/angles/dihedrals unique
; to state A vs B) — relevant for hybrid topologies where bonded terms
; differ between ligand A and B (e.g., ring vs. chain substituent).
; Generated schedules match this to the vdW schedule.

restraint-lambdas         = 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
; Controls scaling of any artificial restraints (e.g., Boresch restraints
; for absolute binding FE). Typically all zeros for standard RBFE since
; restraints aren't usually needed — relevant mainly for absolute FE calcs.

calc-lambda-neighbors     = -1
; How many neighboring lambda states to compute energies for (for BAR).
; -1 means calculate ALL states, not just adjacent ones — REQUIRED for
; MBAR analysis (which needs the full energy matrix). Set to 1 if you
; only plan to use simple BAR (saves computation but loses MBAR option).

; --- Soft-core potential parameters ---
sc-alpha                  = 0.5
; Soft-core alpha parameter — controls how "softened" the vdW potential
; becomes at intermediate lambda. 0 = no softening (hard-core, will crash
; at atom-overlap states). 0.5 is the standard starting value; increase
; to 0.7–1.0 for larger/bulkier perturbations where atoms overlap more.

sc-power                  = 1
; Exponent (p) in the soft-core functional form. 1 is standard GROMACS
; default; rarely needs changing.

sc-sigma                  = 0.3
; Soft-core sigma (nm) — effectively sets the minimum "radius" dummy
; atoms are treated as having, preventing singularities when they have
; zero vdW parameters (appearing/disappearing atoms, i.e. atoms with σ=0).

sc-coul                   = no
; Whether to apply soft-core to Coulomb interactions too. "no" means
; electrostatics are scaled linearly (standard approach) — only vdW
; gets the soft-core treatment. Setting "yes" is sometimes used for
; simultaneous decoupling schedules but adds complexity.

; --- Output ---
nstdhdl                   = 100
; How often (steps) to write dH/dλ and/or ΔU values to dhdl.xvg.
; Every 100 steps = every 0.2 ps — frequent enough for good statistics
; without excessive file size.

dhdl-print-energy         = potential
; Include total energy alongside dH/dλ in the output — useful for
; MBAR reweighting and diagnostics (also allows "potential" option).

separate-dhdl-file        = yes
; Write dH/dλ data to its own dhdl.xvg file rather than embedding in
; the main energy (.edr) file — easier for analysis pipelines (gmx bar,
; alchemlyb) to parse directly.

; --- Coupling parameters (for absolute FE / ligand decoupling; often blank for RBFE hybrid topology) ---
couple-moltype            =
; Specifies which molecule type to decouple entirely from the environment.
; Used in ABSOLUTE free energy calcs (ligand fully removed from system).
; For RBFE with a hybrid topology (A morphs into B), this is left BLANK —
; the perturbation is encoded directly in the topology's B-state parameters,
; not via this global decoupling mechanism.

couple-lambda0            = vdw-q
; (Only relevant if couple-moltype is set) Interactions active at lambda=0:
; vdw-q means both vdW and charge are ON (fully interacting state).

couple-lambda1            = vdw-q
; Interactions active at lambda=1. Also vdw-q here would mean no change —
; this pair of settings matters more for absolute FE (e.g., vdw-q → none).

couple-intramol            = no
; Whether intramolecular interactions within the decoupled molecule are
; also turned off. "no" keeps intramolecular interactions intact (treated
; via normal bonded/exclusions) — standard setting to avoid unphysical
; internal unfolding of the ligand during decoupling.
"""


def lambda_vectors(
    config: LigandRBFEConfig,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return Coulomb, van der Waals, bonded, and restraint schedules."""
    if config.lambda_schedule_mode == "manual":
        assert config.coul_lambdas is not None
        assert config.vdw_lambdas is not None
        assert config.bonded_lambdas is not None
        assert config.restraint_lambdas is not None
        return (
            list(config.coul_lambdas),
            list(config.vdw_lambdas),
            list(config.bonded_lambdas),
            list(config.restraint_lambdas),
        )

    progress = generate_lambda_windows(
        config.lambda_windows, config.lambda_function, config.lambda_power
    )
    # Introduce the new atoms with soft-core vdW/bonded terms before their
    # charges. This avoids electrostatic interactions on overlapping dummies.
    vdw = [min(2.0 * value, 1.0) for value in progress]
    coulomb = [max(2.0 * value - 1.0, 0.0) for value in progress]
    return coulomb, vdw, list(vdw), [0.0] * config.lambda_windows


def _values(items: list[float]) -> str:
    return " ".join(f"{item:.6f}" for item in items)


def _replace(text: str, key: str, value: object) -> str:
    pattern = rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$"
    result, count = re.subn(pattern, rf"\g<1>{value}", text, count=1)
    if count != 1:
        raise ValueError(f"RBFE MDP template does not define {key}")
    return result


def render_rbfe_mdp(
    config: LigandRBFEConfig,
    window: int,
    *,
    membrane: bool | None = None,
) -> str:
    """Render a production MDP for one lambda state."""
    if not 0 <= window < config.lambda_windows:
        raise ValueError(
            f"Lambda window must be between 0 and {config.lambda_windows - 1}"
        )
    coulomb, vdw, bonded, restraints = lambda_vectors(config)
    charmm = "charmm" in config.force_field.casefold()
    coulomb_cutoff = 1.2 if charmm else config.mdp_coulomb_cutoff_nm
    vdw_cutoff = 1.2 if charmm else config.mdp_vdw_cutoff_nm
    replacements: dict[str, object] = {
        "integrator": config.mdp_integrator,
        "tcoupl": "no" if config.mdp_integrator == "sd" else "V-rescale",
        "dt": config.timestep_ps,
        "nsteps": config.production_steps,
        "rlist": 1.2 if charmm else max(coulomb_cutoff, vdw_cutoff),
        "rcoulomb": coulomb_cutoff,
        "vdw-modifier": "Force-switch" if charmm else "Potential-shift-Verlet",
        "rvdw-switch": 1.0 if charmm else 0.0,
        "rvdw": vdw_cutoff,
        "DispCorr": "no" if charmm else "EnerPres",
        "ref-t": config.temperature_k,
        "ref-p": config.pressure_bar,
        "constraints": config.mdp_constraints,
        "init-lambda-state": window,
        "coul-lambdas": _values(coulomb),
        "vdw-lambdas": _values(vdw),
        "bonded-lambdas": _values(bonded),
        "restraint-lambdas": _values(restraints),
        "calc-lambda-neighbors": config.mdp_calc_lambda_neighbors,
        "sc-alpha": config.mdp_softcore_alpha,
        "nstdhdl": config.mdp_nstdhdl,
    }
    text = rbfe_mdp
    for key, value in replacements.items():
        text = _replace(text, key, value)
    # A hybrid RBFE topology carries its A/B parameters directly. The global
    # couple-* mechanism is for molecule decoupling (ABFE), not this workflow.
    text = re.sub(
        r"(?ms)^; --- Coupling parameters .*?\Z",
        "",
        text,
    )
    use_membrane_pressure = config.membrane.enabled if membrane is None else membrane
    if use_membrane_pressure:
        text = _replace(text, "pcoupltype", "semiisotropic")
        text = _replace(
            text, "ref-p", " ".join(map(str, config.membrane.reference_pressure_bar))
        )
        text = _replace(
            text,
            "compressibility",
            " ".join(map(str, config.membrane.compressibility_bar)),
        )
    return text.lstrip()


def _set_value(text: str, key: str, value: object) -> str:
    """Replace an MDP value, appending it when the production template omits it."""
    pattern = rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$"
    if re.search(pattern, text):
        return re.sub(pattern, rf"\g<1>{value}", text, count=1)
    return text.rstrip() + f"\n{key:<26}= {value}\n"


def render_rbfe_stage_mdp(
    config: LigandRBFEConfig,
    window: int,
    stage: str,
    *,
    membrane: bool | None = None,
) -> str:
    """Render one stage of the six-stage RBFE protocol."""
    stage = stage.casefold()
    if stage not in {"mini", "nvt", "npt", "transition", "final", "prod"}:
        raise ValueError(
            "RBFE stage must be mini, nvt, npt, transition, final, or prod"
        )
    text = render_rbfe_mdp(config, window, membrane=membrane)
    stage_index = {
        "mini": 0,
        "nvt": 1,
        "npt": 2,
        "transition": 3,
        "final": 4,
        "prod": 5,
    }[stage]
    integrator = config.stage_integrators[stage_index]
    timestep = config.stage_timesteps_ps[stage_index]
    temperature_coupling = config.stage_temperature_coupling[stage_index]
    pressure_coupling = config.stage_pressure_coupling[stage_index]

    def restraint_defines(forces: tuple[float, float, float]) -> str:
        macros: list[str] = []
        for macro, symbol, force in (
            ("POSRES", "POSRES_FC", forces[0]),
            ("POSRES_LIG", "POSRES_LIG_FC", forces[1]),
            ("POSRES_COFACTOR", "POSRES_COFACTOR_FC", forces[2]),
        ):
            if force > 0:
                macros.extend((f"-D{macro}", f"-D{symbol}={force:g}"))
        return " ".join(macros)

    if stage == "mini":
        values: dict[str, object] = {
            "define": restraint_defines(config.em_restraint_forces),
            "integrator": integrator,
            "nsteps": config.em_steps,
            "emtol": config.em_tolerance,
            "emstep": 0.01,
            "continuation": "no",
            "gen-vel": "no",
            "tcoupl": "no",
            "pcoupl": "no",
            "constraints": "none",
        }
    elif stage == "nvt":
        values = {
            "define": restraint_defines(config.nvt_restraint_forces),
            "integrator": integrator,
            "dt": timestep,
            "nsteps": config.nvt_steps,
            "continuation": "no",
            "gen-vel": "yes",
            "gen-temp": config.temperature_k,
            "gen-seed": -1,
            "tcoupl": temperature_coupling,
            "pcoupl": pressure_coupling,
        }
    elif stage == "npt":
        values = {
            "define": restraint_defines(config.npt_restraint_forces),
            "integrator": integrator,
            "dt": timestep,
            "nsteps": config.npt_steps,
            "continuation": "yes",
            "gen-vel": "no",
            "tcoupl": temperature_coupling,
            "pcoupl": pressure_coupling,
        }
    elif stage == "transition":
        values = {
            "define": restraint_defines(config.transition_restraint_forces),
            "integrator": integrator,
            "dt": timestep,
            "nsteps": config.transition_steps,
            "continuation": "yes",
            "gen-vel": "no",
            "tcoupl": temperature_coupling,
            "pcoupl": pressure_coupling,
        }
    elif stage == "final":
        values = {
            "define": restraint_defines(config.final_restraint_forces),
            "integrator": integrator,
            "dt": timestep,
            "nsteps": config.final_equilibration_steps,
            "continuation": "yes",
            "gen-vel": "no",
            "tcoupl": temperature_coupling,
            "pcoupl": pressure_coupling,
        }
    else:
        values = {
            "define": restraint_defines(config.production_restraint_forces),
            "integrator": integrator,
            "dt": timestep,
            "nsteps": config.production_steps,
            "continuation": "yes",
            "gen-vel": "no",
            "tcoupl": temperature_coupling,
            "pcoupl": pressure_coupling,
        }
    for key, value in values.items():
        text = _set_value(text, key, value)
    return text
