"""Build flat configuration defaults for the bundled OpenFE generators."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _conf_get(conf: Any, name: str) -> Any:
    return conf.get(name) if isinstance(conf, dict) else getattr(conf, name, None)


def _conf_set(conf: Any, name: str, value: Any) -> None:
    if isinstance(conf, dict):
        conf[name] = value
    else:
        setattr(conf, name, value)


def resolve_layout(conf: Any, *, default_jobname: str = "job") -> dict[str, Path]:
    """Project layout: everything lives under ``<workdir>/<jobname>/``.

    Reads ``workdir`` (base directory, default ``.``) and ``jobname`` (falling back
    to ``job_name`` then ``default_jobname``) from a dict or namespace. Returns the
    ``root`` project folder plus its ``inputs``/``json``/``results`` subfolders.
    """
    workdir = Path(str(_conf_get(conf, "workdir") or "."))
    jobname = _conf_get(conf, "jobname") or _conf_get(conf, "job_name") or default_jobname
    root = workdir / str(jobname)
    return {
        "root": root,
        "inputs": root / "inputs",
        "json": root / "json",
        "results": root / "results",
    }


def prepare_project(conf: Any, conf_file: Any = None, *, default_jobname: str = "job",
                    input_attrs: list[tuple[str, str]] | None = None,
                    optional_file_attrs: list[tuple[str, str]] | None = None) -> dict[str, Path]:
    """Create the project layout and stage inputs/config into it.

    Creates ``<workdir>/<jobname>/{inputs,json,results}``, copies the config file
    into the project root for provenance, copies each named input file into
    ``inputs/``, and updates ``conf`` in place: the input attributes point at the
    copies, and ``json_dir``/``result_dir`` are set to the resolved job-dir paths
    (``workdir``/``jobname`` are left as given, so re-loading the written config
    recomputes the same root rather than nesting again). ``input_attrs`` is a list of ``(conf_attr, label)`` pairs that must be
    existing files. ``optional_file_attrs`` is for attributes whose value *may* be a
    file path (e.g. a ``complex_box_size`` that points at a ``pbc.txt`` instead of
    holding numbers) -- these are staged only when the value is an existing file,
    and left untouched otherwise. Returns the layout dict from :func:`resolve_layout`.
    """
    layout = resolve_layout(conf, default_jobname=default_jobname)
    for key in ("inputs", "json", "results"):
        layout[key].mkdir(parents=True, exist_ok=True)

    # Copy the config file into the project root (provenance).
    if conf_file:
        src = Path(conf_file)
        if src.is_file():
            dst = layout["root"] / src.name
            try:
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
            except OSError:
                pass

    def _stage(attr: str, label: str, *, required: bool) -> None:
        val = _conf_get(conf, attr)
        if val in (None, "", "null") or not isinstance(val, (str, Path)):
            return  # unset, or a non-path value (e.g. a numeric box list)
        src = Path(str(val))
        if not src.is_file():
            if required:
                raise FileNotFoundError(f"{label} file not found: {src}")
            return  # optional value that isn't a file (e.g. "6,6,9") -- leave as-is
        dst = layout["inputs"] / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        _conf_set(conf, attr, str(dst))
        print(f"Staged {label}: {dst}")

    for attr, label in (input_attrs or []):
        _stage(attr, label, required=True)
    for attr, label in (optional_file_attrs or []):
        _stage(attr, label, required=False)

    # Point the derived output dirs at the job folder. Note: workdir/jobname are
    # left as given (not overwritten) so the config stays idempotent -- writing it
    # out and re-loading recomputes the same root instead of nesting again.
    _conf_set(conf, "json_dir", str(layout["json"]))
    _conf_set(conf, "result_dir", str(layout["results"]))
    return layout


def box_vectors_matrix(box: Any) -> np.ndarray:
    """Return periodic box vectors as a unitless ``(3, 3)`` numpy array.

    OpenFE/gufe require box vectors in reduced 3x3 form
    (``gufe...._box_vectors_are_in_reduced_form`` asserts ``shape == (3, 3)``). This
    accepts:

    * a path to a ``.txt`` file holding the values (comma-separated), or a raw
      comma-separated string -- e.g. a PBC line exported for a membrane system;
    * 3 values ``[Lx, Ly, Lz]`` (orthorhombic -> diagonal);
    * 6 values ``[a, b, c, alpha, beta, gamma]`` (lengths in nm + angles in degrees
      -> reduced lower-triangular 3x3);
    * 9 values (-> reshaped to 3x3);
    * a full 3x3 matrix (returned as-is).

    Multiply the result by a length unit, e.g.
    ``box_vectors_matrix(box) * offunit.nanometer``.
    """
    # A string/path: read the file if it exists, then split on commas (tolerating
    # newlines/semicolons) into floats.
    if isinstance(box, str):
        text = box
        path = Path(box)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        text = text.replace(";", ",").replace("\n", ",")
        box = [float(part) for part in text.split(",") if part.strip()]

    arr = np.asarray(box, dtype=float)
    if arr.shape == (6,):
        # a, b, c (lengths) + alpha, beta, gamma (degrees) -> reduced 3x3 box
        # (OpenMM/GROMACS lower-triangular convention; 90/90/90 collapses to diag).
        a, b, c = arr[:3]
        alpha, beta, gamma = np.radians(arr[3:])
        bx, by = b * np.cos(gamma), b * np.sin(gamma)
        cx = c * np.cos(beta)
        cy = (b * c * np.cos(alpha) - bx * cx) / by if by else 0.0
        cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
        arr = np.array([[a, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])
        arr[np.abs(arr) < 1e-10] = 0.0
    if arr.shape == (9,):
        arr = arr.reshape(3, 3)
    if arr.shape == (3,):
        arr = np.diag(arr)
    if arr.shape != (3, 3):
        raise ValueError(
            "box vectors must be 3 ([Lx,Ly,Lz]), 6 ([a,b,c,alpha,beta,gamma]), 9, or "
            "a 3x3 matrix -- as numbers, a comma-separated string, or a .txt file; "
            f"got shape {arr.shape}"
        )
    return arr



def _is_mmcif(path: Any) -> bool:
    return str(path).lower().endswith((".cif", ".pdbx", ".mmcif"))


def load_protein_component(protein_path, *, membrane: bool = False, box_vectors=None):
    """Load a ProteinComponent / ProteinMembraneComponent from PDB **or mmCIF**.

    mmCIF (``.cif``/``.pdbx``/``.mmcif``) is preferred for large/membrane systems:
    PDB's atom-serial column overflows past 99,999 atoms (Hybrid36), which OpenMM
    cannot parse. The reader is chosen from the file extension; ``box_vectors`` is
    passed through for the membrane component exactly as ``from_pdb_file`` expects.
    """
    import openfe

    path = str(protein_path)
    cls = openfe.ProteinMembraneComponent if membrane else openfe.ProteinComponent
    kw = {"box_vectors": box_vectors} if membrane else {}

    if not _is_mmcif(path):
        return cls.from_pdb_file(path, **kw)

    if hasattr(cls, "from_pdbx_file"):
        return cls.from_pdbx_file(path, **kw)
    # Fallback for openfe builds without a pdbx loader: parse with OpenMM.
    from openmm.app import PDBxFile

    pdbx = PDBxFile(path)
    if hasattr(cls, "from_openmm_topology"):
        return cls.from_openmm_topology(pdbx.topology, pdbx.positions, **kw)
    raise ValueError(
        f"{cls.__name__} cannot read mmCIF in this openfe build "
        "(no from_pdbx_file / from_openmm_topology); upgrade openfe/gufe, or supply "
        "a PDB with <=99,999 atoms."
    )


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
        "input_check": "error",   # error | warn | off — preflight the solvated input
        "preparation": None,      # build + equilibrate the system in this run (see prepare_system.py)
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
