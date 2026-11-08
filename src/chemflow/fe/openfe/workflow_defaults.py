"""Build flat configuration defaults for the bundled OpenFE generators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _value(item: Any) -> Any:
    if isinstance(item, dict) and "val" in item:
        return item["val"]
    return item


def _common(settings: dict[str, Any]) -> dict[str, Any]:
    forcefield = settings["forcefield_settings"]
    thermo = settings["thermo_settings"]
    charge = settings["partial_charge_settings"]
    return {
        "platform": str(settings["engine_settings"]["compute_platform"]).upper(),
        "pressure": _value(thermo["pressure"]),
        "temperature": _value(thermo["temperature"]),
        "nonbonded_cutoff": _value(forcefield["nonbonded_cutoff"]),
        "nonbonded_method": forcefield["nonbonded_method"],
        "rigid_water": forcefield["rigid_water"],
        "ff_constraints": forcefield.get("constraints", "hbonds"),
        "hydrogen_mass": forcefield["hydrogen_mass"],
        "small_molecule_forcefield": forcefield["small_molecule_forcefield"],
        "forcefields": list(forcefield["forcefields"]),
        "partial_charge_method": charge["partial_charge_method"],
        "off_toolkit_backend": charge["off_toolkit_backend"],
        "membrane_system": False,
        "complex_box_vectors": None,
        "surface_tension": 0.0,
        "ligand_network_graphml": None,
    }


def _rbfe_defaults() -> dict[str, Any]:
    from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol

    settings = RelativeHybridTopologyProtocol.default_settings().model_dump()
    result = _common(settings)
    alchemical = settings["alchemical_settings"]
    simulation = settings["simulation_settings"]
    solvation = settings["solvation_settings"]
    integrator = settings["integrator_settings"]
    result.update(
        {
            "complex_protocol_repeats": 1,
            "solvent_protocol_repeats": 1,
            "endstate_dispersion_correction": alchemical[
                "endstate_dispersion_correction"
            ],
            # Prefer OpenFE's explicit finite-size correction for
            # charge-changing RBFE transformations. Users can still disable it
            # in the model-parameter editor when their protocol requires that.
            "explicit_charge_correction": True,
            "explicit_charge_correction_cutoff": _value(
                alchemical["explicit_charge_correction_cutoff"]
            ),
            "softcore_LJ": alchemical["softcore_LJ"],
            "softcore_alpha": alchemical["softcore_alpha"],
            "turn_off_core_unique_exceptions": alchemical[
                "turn_off_core_unique_exceptions"
            ],
            "use_dispersion_correction": alchemical["use_dispersion_correction"],
            "n_lambda_windows_complex_leg": settings["lambda_settings"][
                "lambda_windows"
            ],
            "n_lambda_windows_solvent_leg": settings["lambda_settings"][
                "lambda_windows"
            ],
            "complex_solvation_padding": _value(solvation["solvent_padding"]),
            "solvent_solvation_padding": _value(solvation["solvent_padding"]),
            "solvent_model": solvation["solvent_model"],
            "box_shape": solvation["box_shape"],
            "early_termination_target_error": _value(
                simulation["early_termination_target_error"]
            ),
            "complex_sim_equil_length": _value(simulation["equilibration_length"]),
            "solvent_sim_equil_length": _value(simulation["equilibration_length"]),
            "complex_sim_mini_step": simulation["minimization_steps"],
            "solvent_sim_mini_step": simulation["minimization_steps"],
            "complex_sim_prod_length": _value(simulation["production_length"]),
            "solvent_sim_prod_length": _value(simulation["production_length"]),
            "complex_timestep": _value(integrator["timestep"]),
            "solvent_timestep": _value(integrator["timestep"]),
            "barostat_frequency": _value(integrator["barostat_frequency"]),
            "alchemical_network": "all",
            "transformation": "both",
            "mapper": "LomapAtomMapper",
            "planner": "minimal_spanning",
        }
    )
    return result


def _subsettings(settings: dict[str, Any], *candidates: str,
                 must_have: str | None = None) -> dict[str, Any]:
    """Return the first sub-settings mapping matching one of ``candidates``.

    ``model_dump()`` key names for some nested settings (notably the SepTop
    restraint settings) vary between OpenFE versions, while attribute access on
    the live object stays stable. Try the given names first; otherwise fall back
    to any top-level sub-mapping that exposes the distinctive ``must_have`` field.
    Returns an empty mapping if nothing matches, so callers can use ``.get`` with
    defaults instead of crashing on a renamed/absent key.
    """
    for name in candidates:
        value = settings.get(name)
        if isinstance(value, dict):
            return value
    if must_have is not None:
        for value in settings.values():
            if isinstance(value, dict) and must_have in value:
                return value
    return {}


def _absolute_defaults(workflow: str) -> dict[str, Any]:
    if workflow == "abfe":
        from openfe.protocols.openmm_afe import AbsoluteBindingProtocol

        settings = AbsoluteBindingProtocol.default_settings().model_dump()
        restraint = _subsettings(settings, "restraint_settings",
                                 must_have="host_selection")
    else:
        from openfe.protocols.openmm_septop import SepTopProtocol

        settings = SepTopProtocol.default_settings().model_dump()
        restraint = _subsettings(settings, "complex_restraint_settings",
                                 must_have="host_selection")

    result = _common(settings)
    alchemical = settings["alchemical_settings"]
    complex_equil = settings["complex_equil_simulation_settings"]
    solvent_equil = settings["solvent_equil_simulation_settings"]
    complex_sim = settings["complex_simulation_settings"]
    solvent_sim = settings["solvent_simulation_settings"]
    complex_solvation = settings["complex_solvation_settings"]
    solvent_solvation = settings["solvent_solvation_settings"]
    complex_integrator = settings["complex_integrator_settings"]
    solvent_integrator = settings["solvent_integrator_settings"]
    result.update(
        {
            "protocol_repeats": 1,
            "disable_alchemical_dispersion_correction": alchemical[
                "disable_alchemical_dispersion_correction"
            ],
            "annihilate_sterics": alchemical["annihilate_sterics"],
            "softcore_a": alchemical["softcore_a"],
            "softcore_alpha": alchemical["softcore_alpha"],
            "softcore_b": alchemical["softcore_b"],
            "softcore_c": alchemical["softcore_c"],
            "anchor_finding_strategy": restraint.get("anchor_finding_strategy", "bonded"),
            "host_max_distance": _value(restraint.get("host_max_distance", 2.0)),
            "host_min_distance": _value(restraint.get("host_min_distance", 0.3)),
            "host_selection": restraint.get("host_selection", "backbone"),
            "rmsf_cutoff": _value(restraint.get("rmsf_cutoff", 0.1)),
            "complex_equil_mini_steps": complex_equil["minimization_steps"],
            "complex_equil_equil_length": _value(complex_equil["equilibration_length"]),
            "complex_equil_equil_length_nvt": _value(
                complex_equil["equilibration_length_nvt"]
            ),
            "complex_equil_production_length": _value(
                complex_equil["production_length"]
            ),
            "solvent_equil_mini_steps": solvent_equil["minimization_steps"],
            "solvent_equil_equil_length": _value(solvent_equil["equilibration_length"]),
            "solvent_equil_equil_length_nvt": _value(
                solvent_equil["equilibration_length_nvt"]
            ),
            "solvent_equil_production_length": _value(
                solvent_equil["production_length"]
            ),
            "early_termination_target_error": _value(
                complex_sim["early_termination_target_error"]
            ),
            "complex_sim_equil_length": _value(complex_sim["equilibration_length"]),
            "complex_sim_mini_step": complex_sim["minimization_steps"],
            "complex_sim_prod_length": _value(complex_sim["production_length"]),
            "solvent_sim_equil_length": _value(solvent_sim["equilibration_length"]),
            "solvent_sim_mini_step": solvent_sim["minimization_steps"],
            "solvent_sim_prod_length": _value(solvent_sim["production_length"]),
            "complex_solvation_padding": _value(complex_solvation["solvent_padding"]),
            "solvent_solvation_padding": _value(solvent_solvation["solvent_padding"]),
            "solvent_model": complex_solvation["solvent_model"],
            "box_shape": complex_solvation["box_shape"],
            "complex_timestep": _value(complex_integrator["timestep"]),
            "solvent_timestep": _value(solvent_integrator["timestep"]),
            "barostat_frequency": _value(complex_integrator["barostat_frequency"]),
        }
    )
    for leg in ("complex", "solvent"):
        for key, value in settings[f"{leg}_lambda_settings"].items():
            result[f"{leg}_{key}"] = value
    if workflow == "septop":
        solvent_restraint = _subsettings(settings, "solvent_restraint_settings",
                                         must_have="spring_constant")
        result.update(
            {
                "solvent_restraint_spring_constant": _value(
                    solvent_restraint.get("spring_constant", 1000.0)
                ),
                "mapper": "LomapAtomMapper",
                "planner": "minimal_spanning",
            }
        )
    return result


def workflow_defaults(workflow: str) -> dict[str, Any]:
    if workflow == "rbfe":
        return _rbfe_defaults()
    if workflow in {"abfe", "septop"}:
        return _absolute_defaults(workflow)
    raise ValueError(f"Unknown OpenFE workflow: {workflow}")


def load_workflow_config(path: str | Path, workflow: str) -> dict[str, Any]:
    """Merge a user configuration over defaults from the installed OpenFE."""
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"OpenFE configuration must be a YAML mapping: {source}")
    config = workflow_defaults(workflow)
    config.update(payload)
    return config
