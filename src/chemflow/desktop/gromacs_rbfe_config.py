"""Qt-free config core for the GROMACS Ligand-RBFE desktop page.

Builds the TOML config consumed by ``chemflow ligand-rbfe`` / ``chemflow rbfe``
(the PMX GROMACS relative binding free-energy pipeline) by merging the page's
overrides over the template defaults and emitting sectioned TOML. No GROMACS /
RDKit / Qt imports, so it is fully testable.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = [
    "DEFAULTS",
    "build_config",
    "to_toml",
    "validate_setup",
    "parse_list",
    "format_list",
]

# Nested defaults mirroring generate_cli._RBFE_TOML (sectioned).
DEFAULTS: dict[str, dict[str, Any]] = {
    "project": {"name": "rbfe_project", "work_dir": "work"},
    "system": {
        "type": "soluble",                       # soluble | membrane
        "protein": "protein.pdb",
        "force_field": "amber99sb-star-ildn-mut",
        "water_model": "tip3p",
        "gromacs_command": "gmx",
        "box_shape": "dodecahedron",
        "box_distance": 1.2,
        "neutralize": True,
        "ion_concentration": 0.15,
    },
    "ligands": {"library_sdf": "ligands.sdf"},
    "mapping": {
        "network_method": "minimal_spanning",   # lomap | minimal_spanning | minimal_redundant | radial
        "atom_mapper": "lomap",                  # lomap | kartograf
        "element_change": False,
        "max3d_angstrom": 1.0,
        "require_cycle_covering": True,
    },
    "topology": {
        "backend": "pmx",
        "charge_method": "bcc",                  # bcc | user
        "charge_change_method": "reject",        # reject | correction
    },
    "simulation": {
        "replicates": 3,
        "lambda_windows": 21,
        "timestep_ps": 0.002,
        "equilibration_ns": 1.0,
        "production_ns": 5.0,
        "sampling_method": "independent",        # independent | hrex
    },
    "execution": {"mdrun_options": ["-ntmpi", "1"]},
    "slurm": {
        "partition": "gpu",
        "nodes": 1,
        "cpus_per_task": 8,
        "gpus": 1,
        "memory": "16G",
        "walltime": "24:00:00",
    },
}

_NETWORK_METHODS = {"lomap", "minimal_spanning", "minimal_redundant", "radial"}


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge sectioned overrides over the defaults (None values ignored)."""
    return _deep_merge(DEFAULTS, overrides or {})


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def to_toml(conf: dict[str, Any]) -> str:
    """Emit sectioned TOML from a {section: {key: value}} mapping."""
    lines: list[str] = []
    for section, body in conf.items():
        lines.append(f"[{section}]")
        if isinstance(body, dict):
            for key, value in body.items():
                if value is None:
                    continue
                lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def validate_setup(conf: dict[str, Any]) -> list[str]:
    """Return a list of problems with the setup (empty = OK)."""
    issues: list[str] = []
    project = conf.get("project", {})
    system = conf.get("system", {})
    ligands = conf.get("ligands", {})
    mapping = conf.get("mapping", {})
    topology = conf.get("topology", {})
    simulation = conf.get("simulation", {})

    if not project.get("name"):
        issues.append("project.name is required (it names the work_dir/name project folder).")
    if not (system.get("protein") or system.get("prepared_complex_gro")):
        issues.append("system.protein (or a prepared_complex_gro) is required.")
    if not ligands.get("library_sdf"):
        issues.append("ligands.library_sdf is required (one SDF; each record a unique _Name).")

    system_type = system.get("type", "soluble")
    if system_type not in ("soluble", "membrane"):
        issues.append("system.type must be soluble | membrane.")
    preparation = conf.get("preparation", {})
    prep_on = bool(preparation.get("enabled"))
    if system_type == "membrane" and not (
        prep_on or system.get("prepared_complex_gro")
    ):
        issues.append(
            "Membrane systems need an assembled complex: enable preparation "
            "(prepare-system) or supply system.prepared_complex_gro/topology.")
    if prep_on and preparation.get("cofactor_force_field") == "GAFF2" \
            and not preparation.get("cofactor_structure"):
        issues.append(
            "preparation.cofactor_structure is required when cofactor_force_field = GAFF2.")

    method = mapping.get("network_method")
    if method not in _NETWORK_METHODS:
        issues.append(f"mapping.network_method must be one of {sorted(_NETWORK_METHODS)}.")
    if method == "radial" and not mapping.get("radial_hub"):
        issues.append("mapping.radial_hub is required when network_method = radial.")

    if topology.get("charge_change_method") == "correction" and not topology.get(
        "charge_correction_file"
    ):
        issues.append("topology.charge_correction_file is required when "
                      "charge_change_method = correction.")

    if int(simulation.get("lambda_windows", 0)) < 2:
        issues.append("simulation.lambda_windows must be >= 2.")
    if int(simulation.get("replicates", 0)) < 1:
        issues.append("simulation.replicates must be >= 1.")
    if simulation.get("sampling_method") not in ("independent", "hrex"):
        issues.append("simulation.sampling_method must be independent | hrex.")
    if simulation.get("sampling_method") == "hrex":
        execution = conf.get("execution", {})
        command = str(execution.get("hrex_gromacs_command") or "").strip()
        module = str(execution.get("hrex_module") or "").strip()
        if not command and not module:
            issues.append(
                "HREX needs an external MPI GROMACS build: set the 'HREX gmx command' "
                "(e.g. gmx_mpi or a full path) and/or the 'HREX module' on the Job tab. "
                "The conda env's thread-MPI gmx cannot run replica exchange.")
        elif command and command.replace("\\", "/").rsplit("/", 1)[-1] == "gmx":
            issues.append(
                "HREX gmx command is plain 'gmx' (thread-MPI); lambda-HREX requires an "
                "external MPI build such as gmx_mpi.")
    return issues


def parse_list(text: str) -> list[str] | None:
    """Parse whitespace/comma-separated tokens into a list; ''/None -> None."""
    if text is None or not str(text).strip():
        return None
    return [p for p in str(text).replace(",", " ").split()] or None


def format_list(items) -> str:
    """Render a list as space-separated tokens for a text field."""
    return " ".join(str(i) for i in (items or []))
