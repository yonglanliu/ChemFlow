"""Qt-free config core for the OpenFE ABFE workflow page.

Mirrors AlchemForge's OpenFE ABFE setup (the single YAML consumed by
``chemflow.fe.openfe.abfe.create_json``). ABFE decouples ONE ligand, so there is
no perturbation network — the workflow is charge → system (soluble or membrane)
→ restraints → FEP (alchemical + per-leg lambda schedules) → job. No OpenMM /
OpenFE / Qt imports, so it is fully testable.

The SLURM array generator is shared with the RBFE core (``openfe quickrun`` per
transformation JSON × replicate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    "build_abfe_lambda_schedule",
    "slurm_script",
    "LAMBDA_KEYS",
]

# Lambda schedule keys per leg; all lists within a leg must share one length.
LAMBDA_KEYS = {
    "complex": ("complex_lambda_elec", "complex_lambda_restraints", "complex_lambda_vdw"),
    "solvent": ("solvent_lambda_elec", "solvent_lambda_restraints", "solvent_lambda_vdw"),
}

# Full default config, mirroring generate_cli._ABFE_OPENFE_YAML (+ the user's
# capsaicin template). The form overrides a curated subset; everything else keeps
# these defaults so the YAML stays complete. Keys not read by abfe/create_json
# (e.g. solvent_restraint_spring_constant) are harmless extras.
DEFAULTS: dict[str, Any] = {
    # --- Input files ---
    "ligand_sdf": "charged_ligand.sdf",
    "protein_pdb": "complex.pdb",
    "cofactor_sdf": None,
    "restart_json": None,
    "name": "abfe_my_ligand",
    # --- Output ---
    "workdir": ".",
    "jobname": "abfe",
    "result_dir": "abfe/results",
    "json_dir": "abfe/json",
    # --- Platform ---
    "platform": "CUDA",
    # --- Membrane or not ---
    "membrane_system": False,
    "complex_box_vectors": None,     # [[x,0,0],[0,y,0],[0,0,z]] nm when membrane
    "box_shape": "dodecahedron",     # cube | dodecahedron | octahedron (non-membrane + solvent)
    # --- General ---
    "protocol_repeats": 1,
    "pressure": 1.0,                 # bar
    "temperature": 298.15,           # kelvin
    # --- Force field / nonbonded ---
    "nonbonded_cutoff": 1.0,         # nanometer
    "nonbonded_method": "PME",
    "rigid_water": True,
    "small_molecule_forcefield": "openff-2.1.0",
    "hydrogen_mass": 3.0,            # dalton
    # --- Partial charges ---
    "off_toolkit_backend": "ambertools",  # openeye | rdkit | ambertools
    "partial_charge_method": "am1bcc",    # am1bcc | nagl
    # --- Alchemical settings ---
    "disable_alchemical_dispersion_correction": False,
    "annihilate_sterics": False,
    "softcore_a": 1.0,
    "softcore_b": 1.0,
    "softcore_c": 6.0,
    "softcore_alpha": 0.5,
    # --- Restraints ---
    "anchor_finding_strategy": "multi-residue",  # bonded | multi-residue
    "host_max_distance": 2.0,        # nanometer
    "host_min_distance": 0.3,        # nanometer
    "host_selection": "backbone",    # backbone | all
    "rmsf_cutoff": 0.1,              # nanometer
    "solvent_restraint_spring_constant": 1000.0,  # kJ/mol/nm^2
    # --- Equilibration settings ---
    "complex_equil_mini_steps": 5000,
    "complex_equil_equil_length": 1.0,       # ns
    "complex_equil_equil_length_nvt": 1.0,   # ns
    "complex_equil_production_length": 2.0,   # ns (NPT production equilibration)
    "solvent_equil_mini_steps": 5000,
    "solvent_equil_equil_length": 1.0,       # ns
    "solvent_equil_equil_length_nvt": 1.0,   # ns
    "solvent_equil_production_length": 2.0,   # ns (NPT production equilibration)
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
    "complex_sim_prod_length": 10.0,         # ns
    "solvent_sim_mini_step": 5000,
    "solvent_sim_equil_length": 1.0,         # ns
    "solvent_sim_prod_length": 10.0,         # ns
    # --- Lambda schedules (each leg's elec/vdw/restraints lists share one length) ---
    "complex_lambda_elec": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4,
                            0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "complex_lambda_restraints": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "complex_lambda_vdw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4,
                           0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    "solvent_lambda_elec": [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0],
    "solvent_lambda_restraints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0],
    "solvent_lambda_vdw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.24, 0.36, 0.48, 0.6,
                           0.7, 0.77, 0.85, 1.0],
    # --- Integrator settings ---
    "complex_timestep": 4.0,   # femtosecond
    "solvent_timestep": 4.0,   # femtosecond
    "barostat_frequency": 25.0,  # timestep
    "surface_tension": 0.0,  # bar*nm (membrane only; 0 => tensionless)
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
        # Recurses, so nested lists (box vectors) emit as [[...], [...], [...]].
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

    Enforces: required inputs + name; a membrane system needs explicit
    ``complex_box_vectors`` (3 vectors of 3 numbers, positive diagonal); and each
    leg's lambda lists share one length.
    """
    issues: list[str] = []
    if not conf.get("ligand_sdf"):
        issues.append("Ligand SDF is required (ABFE decouples exactly one ligand).")
    if not conf.get("protein_pdb"):
        issues.append("Protein PDB is required.")
    if not conf.get("name"):
        issues.append("A transformation name is required (used as the output JSON basename).")

    if conf.get("membrane_system"):
        box = conf.get("complex_box_vectors")
        ok = (isinstance(box, (list, tuple)) and len(box) == 3
              and all(isinstance(v, (list, tuple)) and len(v) == 3
                      and all(_is_number(x) for x in v) for v in box))
        if not ok:
            issues.append("Membrane system requires complex_box_vectors = 3 vectors of 3 numbers "
                          "(nm), e.g. [[12.3,0,0],[0,12.2,0],[0,0,14.3]].")
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


def parse_float_list(text: str) -> list[float] | None:
    """Parse '0.0, 0.1, ...' (commas and/or whitespace) into floats; ''/None -> None."""
    if text is None or not str(text).strip():
        return None
    parts = str(text).replace(",", " ").split()
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"lambda list must be numbers, got {text!r}") from exc


def parse_box_vectors(text: str):
    """Parse box vectors into [[x,0,0],[0,y,0],[0,0,z]] (nm); ''/None -> None.

    Accepts three numbers (x y z -> a diagonal box) or nine numbers (full 3×3,
    row-major), separated by commas/whitespace/newlines. A path to a ``.txt`` file
    (or any existing file) is read and parsed the same way, so a membrane PBC file
    can be used directly.
    """
    if text is None or not str(text).strip():
        return None
    s = str(text).strip()
    path = Path(s)
    if s.lower().endswith(".txt") or path.is_file():
        if not path.is_file():
            raise ValueError(f"PBC file not found: {s}")
        s = path.read_text(encoding="utf-8")
    parts = s.replace(";", " ").replace(",", " ").split()
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"box vectors must be numbers or a .txt file, got {text!r}") from exc
    if len(values) == 3:
        x, y, z = values
        return [[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, z]]
    if len(values) == 9:
        return [values[0:3], values[3:6], values[6:9]]
    raise ValueError("box vectors need 3 numbers (x y z) or 9 numbers (full 3×3), "
                     f"got {len(values)}")


def format_float_list(values) -> str:
    """Render a list of numbers as a comma-separated string for a text field."""
    if not values:
        return ""
    return ", ".join(repr(float(v)) if isinstance(v, float) else str(v) for v in values)


def format_box_vectors(box) -> str:
    """Render box vectors back into the field; prefer 'x, y, z' for a diagonal box."""
    if not box:
        return ""
    try:
        if (len(box) == 3 and all(len(v) == 3 for v in box)
                and box[0][1] == 0 and box[0][2] == 0
                and box[1][0] == 0 and box[1][2] == 0
                and box[2][0] == 0 and box[2][1] == 0):
            return ", ".join(repr(float(box[i][i])) for i in range(3))
        return "\n".join(", ".join(repr(float(x)) for x in row) for row in box)
    except (TypeError, IndexError):
        return str(box)


def linspace(n: int, lo: float = 0.0, hi: float = 1.0, decimals: int = 4) -> list[float]:
    """n evenly spaced values from lo to hi inclusive (n>=2); n==1 -> [lo], n<=0 -> []."""
    if n <= 0:
        return []
    if n == 1:
        return [round(lo, decimals)]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, decimals) for i in range(n)]


def build_abfe_lambda_schedule(leg: str, n_restraints: int, n_elec: int,
                               n_vdw: int) -> dict[str, list[float]]:
    """Build an ABFE lambda schedule from per-phase window counts (a staircase).

    The absolute-BFE decoupling is done in stages so only one interaction is
    perturbed at a time, which keeps each window's overlap good:

    * **complex** leg — first ramp the Boresch *restraints* on (elec/vdw still
      fully coupled), then decouple *electrostatics*, then decouple *van der
      Waals*. All three lists have length ``n_restraints + n_elec + n_vdw``.
    * **solvent** leg — no restraints; decouple *electrostatics* then *vdw*.
      All three lists have length ``n_elec + n_vdw`` (restraints stay 0).

    Each ``*_lambda_*`` list within a leg shares one length, as the protocol
    requires. ``n_elec`` and ``n_vdw`` should be >= 2 so each phase actually
    ramps 0 -> 1.
    """
    elec_ramp = linspace(n_elec)
    vdw_ramp = linspace(n_vdw)
    if leg == "complex":
        rest_ramp = linspace(n_restraints)
        n_rest = len(rest_ramp)
        elec = [0.0] * n_rest + elec_ramp + [1.0] * len(vdw_ramp)
        restraints = rest_ramp + [1.0] * (len(elec_ramp) + len(vdw_ramp))
        vdw = [0.0] * (n_rest + len(elec_ramp)) + vdw_ramp
        return {
            "complex_lambda_elec": elec,
            "complex_lambda_restraints": restraints,
            "complex_lambda_vdw": vdw,
        }
    if leg == "solvent":
        total = len(elec_ramp) + len(vdw_ramp)
        elec = elec_ramp + [1.0] * len(vdw_ramp)
        vdw = [0.0] * len(elec_ramp) + vdw_ramp
        return {
            "solvent_lambda_elec": elec,
            "solvent_lambda_restraints": [0.0] * total,
            "solvent_lambda_vdw": vdw,
        }
    raise ValueError(f"leg must be 'complex' or 'solvent', got {leg!r}")

