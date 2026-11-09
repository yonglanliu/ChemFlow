"""Qt-free config core for the OpenFE SepTop workflow page.

Mirrors AlchemForge's OpenFE SepTop setup (the single YAML consumed by
``chemflow.fe.openfe.septop.create_json``). SepTop is a RELATIVE method with
separated topologies: two ligands A and B are perturbed at once (A coupled while
B is decoupled, and vice versa), so every lambda column comes in an ``_A`` / ``_B``
pair. Like RBFE it plans a perturbation network (mapper/planner); like ABFE it
uses a host restraint. Workflow: charge → network → system → restraints → FEP
(dual A/B lambda schedules) → job.

No OpenMM / OpenFE / Qt imports, so it is fully testable. The SLURM array
generator and most helpers are shared with the RBFE/ABFE cores.
"""

from __future__ import annotations

from typing import Any

from chemflow.desktop.openfe_abfe_config import (  # shared helpers
    format_box_vectors,
    format_float_list,
    linspace,
    parse_box_vectors,
    parse_float_list,
)
from chemflow.desktop.openfe_rbfe_config import (  # shared SLURM array generator
    HPC_DEFAULTS,
    slurm_script,
)

__all__ = [
    "DEFAULTS",
    "HPC_DEFAULTS",
    "build_config",
    "to_yaml",
    "validate_setup",
    "parse_float_list",
    "parse_box_vectors",
    "format_float_list",
    "format_box_vectors",
    "linspace",
    "build_septop_lambda_schedule",
    "slurm_script",
    "LAMBDA_KEYS",
]

# Lambda schedule keys per leg (dual topology → _A/_B pairs). All lists within a
# leg must share one length.
LAMBDA_KEYS = {
    "complex": ("complex_lambda_elec_A", "complex_lambda_elec_B",
                "complex_lambda_restraints_A", "complex_lambda_restraints_B",
                "complex_lambda_vdw_A", "complex_lambda_vdw_B"),
    "solvent": ("solvent_lambda_elec_A", "solvent_lambda_elec_B",
                "solvent_lambda_restraints_A", "solvent_lambda_restraints_B",
                "solvent_lambda_vdw_A", "solvent_lambda_vdw_B"),
}


def _ramp(n: int) -> list[float]:
    """0 -> 1 over n points."""
    return linspace(n, 0.0, 1.0)


def _ramp_down(n: int) -> list[float]:
    """1 -> 0 over n points."""
    return linspace(n, 1.0, 0.0)


def build_septop_lambda_schedule(leg: str, n_phase1: int, n_phase2: int,
                                 n_phase3: int) -> dict[str, list[float]]:
    """Build a SepTop dual-topology lambda schedule from per-phase window counts.

    SepTop morphs ligand B into ligand A in three stages so only one kind of
    interaction changes at a time (good overlap):

    * **complex** leg —
      phase 1: switch on A's restraint and switch off B's van der Waals;
      phase 2: swap electrostatics (A on, B off);
      phase 3: switch on A's van der Waals and switch off B's restraint.
    * **solvent** leg — no restraints, so:
      phase 1: switch off B's vdw; phase 2: swap electrostatics; phase 3: switch
      on A's vdw. All ``*_restraints_*`` lists stay 0.

    Every ``*_lambda_*`` list within a leg shares one length
    (= n_phase1 + n_phase2 + n_phase3). ``n_phase*`` should be >= 2 so each phase
    actually ramps.
    """
    r1, r2, r3 = _ramp(n_phase1), _ramp(n_phase2), _ramp(n_phase3)
    d1, d2, d3 = _ramp_down(n_phase1), _ramp_down(n_phase2), _ramp_down(n_phase3)
    a, b, c = len(r1), len(r2), len(r3)
    total = a + b + c

    # Electrostatics swap happens in phase 2 for both legs.
    elec_A = [0.0] * a + r2 + [1.0] * c
    elec_B = [1.0] * a + d2 + [0.0] * c
    # A's vdw switches on in phase 3; B's vdw switches off in phase 1.
    vdw_A = [0.0] * (a + b) + r3
    vdw_B = d1 + [0.0] * (b + c)

    if leg == "complex":
        # A's restraint ramps on in phase 1; B's restraint ramps off in phase 3.
        restraints_A = r1 + [1.0] * (b + c)
        restraints_B = [1.0] * (a + b) + d3
        return {
            "complex_lambda_elec_A": elec_A,
            "complex_lambda_elec_B": elec_B,
            "complex_lambda_restraints_A": restraints_A,
            "complex_lambda_restraints_B": restraints_B,
            "complex_lambda_vdw_A": vdw_A,
            "complex_lambda_vdw_B": vdw_B,
        }
    if leg == "solvent":
        zeros = [0.0] * total
        return {
            "solvent_lambda_elec_A": elec_A,
            "solvent_lambda_elec_B": elec_B,
            "solvent_lambda_restraints_A": zeros,
            "solvent_lambda_restraints_B": list(zeros),
            "solvent_lambda_vdw_A": vdw_A,
            "solvent_lambda_vdw_B": vdw_B,
        }
    raise ValueError(f"leg must be 'complex' or 'solvent', got {leg!r}")


# Full default config, mirroring generate_cli + the user's PDE4B SepTop template.
# Keys not read by septop/create_json are harmless extras; most numeric defaults
# are also filled from the installed OpenFE by workflow_defaults at run time.
DEFAULTS: dict[str, Any] = {
    # --- Input files ---
    "ligand_sdf": "charged_ligands.sdf",
    "protein_pdb": "protein.pdb",
    "cofactor_sdf": None,
    "restart_json": None,
    # --- Output ---
    "workdir": ".",
    "jobname": "septop",
    "result_dir": "septop/results",
    "json_dir": "septop/json",
    "ligand_network_graphml": None,
    # --- Platform ---
    "platform": "CUDA",
    # --- Membrane or not ---
    "membrane_system": False,
    "complex_box_vectors": None,      # [[x,0,0],[0,y,0],[0,0,z]] nm when membrane
    "box_shape": "dodecahedron",      # cube | dodecahedron | octahedron (soluble + solvent)
    # --- General ---
    "protocol_repeats": 1,
    "pressure": 1.0,                  # bar
    "temperature": 310.0,             # kelvin
    # --- Force field / nonbonded ---
    "nonbonded_cutoff": 1.0,          # nanometer
    "nonbonded_method": "PME",
    "rigid_water": True,
    "small_molecule_forcefield": "openff-2.1.0",
    "hydrogen_mass": 3.0,             # dalton
    # --- Partial charges ---
    "off_toolkit_backend": "ambertools",
    "partial_charge_method": "am1bcc",
    # --- Alchemical settings ---
    "disable_alchemical_dispersion_correction": False,
    "annihilate_sterics": False,
    "softcore_a": 1.0,
    "softcore_b": 1.0,
    "softcore_c": 6.0,
    "softcore_alpha": 0.5,
    # --- Restraints ---
    "anchor_finding_strategy": "bonded",   # bonded | multi-residue
    "host_max_distance": 2.0,        # nanometer
    "host_min_distance": 0.3,        # nanometer
    "host_selection": "backbone",    # backbone | all
    "rmsf_cutoff": 0.1,              # nanometer
    "solvent_restraint_spring_constant": 1000.0,  # kJ/mol/nm^2
    # --- Equilibration settings ---
    "complex_equil_mini_steps": 5000,
    "complex_equil_equil_length": 0.1,       # ns
    "complex_equil_equil_length_nvt": 0.1,   # ns
    "complex_equil_production_length": 2.0,  # ns
    "solvent_equil_mini_steps": 5000,
    "solvent_equil_equil_length": 0.1,       # ns
    "solvent_equil_equil_length_nvt": 0.1,   # ns
    "solvent_equil_production_length": 2.0,  # ns
    # --- Solvation settings (non-membrane) ---
    "complex_solvation_padding": 1.5,  # nanometer
    "solvent_solvation_padding": 1.5,  # nanometer
    "solvent_model": "tip3p",
    # --- Ions / solvent charge ---
    "ion_concentration": 0.15,   # molar (SolventComponent salt concentration)
    "positive_ion": "Na+",
    "negative_ion": "Cl-",
    "neutralize": True,          # add counter-ions to neutralize net charge
    # --- Simulation settings ---
    "early_termination_target_error": 0.0,   # kcal/mol
    "complex_sim_mini_step": 5000,
    "complex_sim_equil_length": 1.0,         # ns
    "complex_sim_prod_length": 20.0,         # ns
    "solvent_sim_mini_step": 5000,
    "solvent_sim_equil_length": 5.0,         # ns
    "solvent_sim_prod_length": 30.0,         # ns
    # --- Lambda schedules (dual topology: _A / _B, each leg's lists share one length) ---
    **build_septop_lambda_schedule("complex", 7, 5, 7),
    **build_septop_lambda_schedule("solvent", 10, 8, 9),
    # --- Integrator settings ---
    "complex_timestep": 4.0,   # femtosecond
    "solvent_timestep": 4.0,   # femtosecond
    "barostat_frequency": 25.0,  # timestep
    "surface_tension": 300.0,  # bar*nm (membrane only)
    # --- Mapper / planner ---
    "mapper": "KartografAtomMapper",      # LomapAtomMapper | KartografAtomMapper
    "planner": "minimal_spanning",        # minimal_spanning | minimal_redundant | radial
}


def build_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge form overrides over the defaults (None values are ignored)."""
    conf = dict(DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                conf[key] = value
    return conf


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(v) for v in value) + "]"
    text = str(value)
    if text == "" or any(ch in text for ch in ' :#"\''):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def to_yaml(conf: dict[str, Any]) -> str:
    """Emit a flat YAML mapping for the create_json config."""
    return "".join(f"{key}: {_yaml_scalar(value)}\n" for key, value in conf.items())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_setup(conf: dict[str, Any]) -> list[str]:
    """Return a list of problems with the setup (empty = OK).

    SepTop is relative, so it needs at least two ligands in the SDF, a protein,
    a valid membrane box (vectors) when membrane, and — within each leg — all
    ``_A``/``_B`` lambda lists sharing one length.
    """
    issues: list[str] = []
    if not conf.get("ligand_sdf"):
        issues.append("Ligand SDF is required (SepTop perturbs pairs, so it needs >= 2 ligands).")
    if not conf.get("protein_pdb"):
        issues.append("Protein PDB is required.")

    if conf.get("membrane_system"):
        box = conf.get("complex_box_vectors")
        ok = (isinstance(box, (list, tuple)) and len(box) == 3
              and all(isinstance(v, (list, tuple)) and len(v) == 3
                      and all(_is_number(x) for x in v) for v in box))
        if not ok:
            issues.append("Membrane system requires complex_box_vectors = 3 vectors of 3 numbers "
                          "(nm), e.g. [[6.5,0,0],[0,6.5,0],[0,0,9.0]].")
        elif any(box[i][i] <= 0 for i in range(3)):
            issues.append("complex_box_vectors must have a positive diagonal (box lengths > 0).")

    for leg, keys in LAMBDA_KEYS.items():
        lengths = {}
        for key in keys:
            value = conf.get(key)
            if isinstance(value, (list, tuple)):
                lengths[key] = len(value)
        if lengths and len(set(lengths.values())) > 1:
            issues.append(f"{leg} lambda lists must share one length, got "
                          + ", ".join(f"{k}={v}" for k, v in lengths.items()) + ".")
        if lengths and min(lengths.values(), default=0) < 2:
            issues.append(f"{leg} lambda lists must have at least 2 windows.")
    return issues
