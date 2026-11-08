"""Defaults and serialization helpers for the OpenFE desktop workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

OPENFE_LIGAND_FEP_CONFIG: dict[str, dict[str, Any]] = {
    "system_setup": {
        "ligand_sdf": "",
        "protein_pdb": "",
        "cofactor_sdf": None,
        "compute_charge": False,
        "input_json_files": "",
        "ff_constraints": "hbonds",
        "nonbonded_cutoff": 1.0,
        "nonbonded_method": "PME",
        "rigid_water": True,
        "protein_forcefield": [
            "amber/ff14SB.xml",
            "amber/tip3p_standard.xml",
        ],
        "off_toolkit_backend": "ambertools",
        "partial_charge_method": "am1bcc",
        "small_molecule_forcefield": "openff-2.2.1",
        "solvation_padding": 1.2,
        "complex_box_size": None,
        "solvent_model": "tip3p",
        "positive_ion": "Na+",
        "negative_ion": "Cl-",
        "neutralize": True,
        "ion_concentration": 0.15,
        "membrane_system": False,
        "hydrogen_mass": 3.0,
        "alchemical_network": "all",
        "transformation": "both",
        "mapper": "LomapAtomMapper",
        "planner": "minimal_spanning",
    },
    "alchemical_network_settings": {
        "replicates": 1,
        "start_replicates": 1,
        "pressure": 1.0,
        "temperature": 298.15,
        "endstate_dispersion_correction": False,
        "explicit_charge_correction": True,
        "explicit_charge_correction_cutoff": 0.8,
        "softcore_LJ": "gapsys",
        "softcore_alpha": 0.85,
        "turn_off_core_unique_exceptions": False,
        "use_dispersion_correction": True,
        "n_lambda_windows": 11,
        "early_termination_target_error": 0.0,
        "mini_step": 5000,
        "equil_length": 0.2,
        "prod_length": 5.0,
        "timestep": 4.0,
        "barostat_frequency": 25.0,
        "surface_tension": 0.0,
    },
    "job_setting": {
        "job_name": "LigAlchemFEP",
        "ligand_para_host": "localhost",
        "system_setup_host": "localhost",
        "simulation_host": "localhost",
        "platform": "CPU",
    },
}


def save_yaml(config: dict[str, Any], path: str | Path) -> Path:
    """Write a deterministic, human-readable OpenFE YAML configuration."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
    return destination
