#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import string

import numpy as np
import openfe
import openmm
from gufe.protocols import execute_DAG
from openfe.protocols.openmm_afe import AbsoluteBindingProtocol
from openfe.protocols.openmm_utils.omm_settings import OpenFFPartialChargeSettings
from openff.toolkit import Molecule
from openff.units import unit as offunit
from openmm import Platform, System, VerletIntegrator, unit
from openmm.app import Simulation, Topology
from rdkit import Chem
from pydantic import ValidationError

from chemflow.fe.openfe.partial_charges import (
    ensure_partial_charges,
    harmonize_isomorphic_charges,
)
from chemflow.fe.openfe.workflow_defaults import (
    box_vectors_matrix,
    load_protein_component,
    load_workflow_config,
    prepare_project,
)


def test_cuda():
    print("=== OpenMM Platform Check ===")
    n = Platform.getNumPlatforms()

    for i in range(n):
        p = Platform.getPlatform(i)
        print(f"Platform {i}: {p.getName()}")

    try:
        platform = Platform.getPlatformByName("CUDA")
        print("\nTrying CUDA platform...")

        system = System()
        system.addParticle(12.0)  # dummy particle

        integrator = VerletIntegrator(0.001)

        simulation = Simulation(Topology(), system, integrator, platform)

        print("CUDA is available and working")

        # Print CUDA properties if available
        try:
            print("CUDA Device Index:", platform.getPropertyDefaultValue("DeviceIndex"))
            print("CUDA Precision:", platform.getPropertyDefaultValue("Precision"))
        except Exception:
            pass

        return True

    except Exception as e:
        print("CUDA test failed")
        print("Error:", e)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenFE SepTop RBFE workflow on HPC."
    )
    parser.add_argument(
        "--conf_file",
        type=pathlib.Path,
        default=pathlib.Path("./abfe_conf.yaml"),
        help="Directory to write transformation JSON.",
    )
    return parser.parse_args()


"""
In those cases, partial charges will be assigned within the selected Protocol.
As of writing, for SmallMoleculeComponents this is done using antechamber’s am1bcc charge assignment,
using the input conformer as the pre-optimized input for sqm’s AM1 calculation.
For ProteinComponents and SolventComponents this is done by assigning charges from the chosen protein and solvent force fields.
Unfortunately charge assignment can be both,
    a) time consuming for large molecules, and
    b) non-deterministic, especially when the molecule occupies a conformation far from a local minima in the AM1 energetic landscape.
The latter can be particularly problematic as this can lead to significant differences in assigned partial charges between Protocol simulation repeats.
To avoid these issues, we recommend applying user charges to SmallMoleculeComponents.
In its simplest form, this is done by converting the SmallMoleculeComponent to an OpenFF Molecule calling the OpenFF Molecule’s assign_partial_charges method
and then re-loading it back into a SmallMoleculeComponent before further manipulation.
"""


def load_ligands(ligand_sdf: pathlib.Path) -> list[openfe.SmallMoleculeComponent]:
    supp = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    ligands = [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]
    if not ligands:
        raise ValueError(f"No valid ligands found in {ligand_sdf}")
    return ligands


def load_cofactor(cofactor_sdf: pathlib.Path) -> list[openfe.SmallMoleculeComponent]:
    supp = Chem.SDMolSupplier(str(cofactor_sdf), removeHs=False)
    cofactors = [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]
    if not cofactors:
        raise ValueError(f"No valid cofactor found in {cofactor_sdf}")
    return cofactors


def _create_protocol(conf):
    # ---------------------------------------
    #           Create protocol
    # ---------------------------------------
    settings = AbsoluteBindingProtocol.default_settings()

    if conf.platform == "CUDA":
        # Honor the config literally; don't probe CUDA at generation time on a
        # GPU-less login node (that would bake CPU into the JSON and the GPU
        # compute node would run on CPU). OpenMM picks the device at run time.
        settings.engine_settings.compute_platform = "CUDA"
        print("Using CUDA (run on a GPU node; set platform: CPU to force CPU)")
    elif conf.platform == "OpenCL":
        settings.engine_settings.compute_platform = "OpenCL"
        print("Using OpenCL")
    elif conf.platform == "CPU":
        settings.engine_settings.compute_platform = "CPU"
        print("Using CPU")
    else:
        raise ValueError(f"Unsupported OpenMM platform: {conf.platform}")

    settings.protocol_repeats = conf.protocol_repeats

    # Thermo settings
    print("Thermo settings ...")
    settings.thermo_settings.pressure = conf.pressure * offunit.bar
    settings.thermo_settings.temperature = conf.temperature * offunit.kelvin

    # Alchemical settings
    settings.alchemical_settings.disable_alchemical_dispersion_correction = (
        conf.disable_alchemical_dispersion_correction
    )
    settings.alchemical_settings.annihilate_sterics = conf.annihilate_sterics
    settings.alchemical_settings.softcore_a = conf.softcore_a
    settings.alchemical_settings.softcore_alpha = conf.softcore_alpha
    settings.alchemical_settings.softcore_b = conf.softcore_b
    settings.alchemical_settings.softcore_c = conf.softcore_c

    # General FF settings
    settings.forcefield_settings.nonbonded_cutoff = (
        conf.nonbonded_cutoff * offunit.nanometer
    )
    settings.forcefield_settings.nonbonded_method = conf.nonbonded_method
    settings.forcefield_settings.rigid_water = conf.rigid_water
    settings.forcefield_settings.small_molecule_forcefield = (
        conf.small_molecule_forcefield
    )
    settings.forcefield_settings.forcefields = conf.forcefields
    settings.forcefield_settings.hydrogen_mass = conf.hydrogen_mass

    # Partial charge setting
    settings.partial_charge_settings.off_toolkit_backend = conf.off_toolkit_backend
    if conf.partial_charge_method == "am1bcc":
        settings.partial_charge_settings.partial_charge_method = "am1bcc"
    elif conf.partial_charge_method == "nagl":
        settings.partial_charge_settings.partial_charge_method = "nagl"
        settings.partial_charge_settings.nagl_model = "openff-gnn-am1bcc-0.1.0-rc.3.pt"

    # Restraint settings
    settings.restraint_settings.anchor_finding_strategy = conf.anchor_finding_strategy
    settings.restraint_settings.host_max_distance = (
        conf.host_max_distance * offunit.nanometer
    )
    settings.restraint_settings.host_min_distance = (
        conf.host_min_distance * offunit.nanometer
    )
    settings.restraint_settings.host_selection = conf.host_selection
    settings.restraint_settings.rmsf_cutoff = conf.rmsf_cutoff * offunit.nanometer

    # ---------------
    # Equil Settings
    # ---------------
    # Solvent
    settings.solvent_equil_simulation_settings.minimization_steps = (
        conf.solvent_equil_mini_steps
    )
    settings.solvent_equil_simulation_settings.equilibration_length_nvt = (
        conf.solvent_equil_equil_length_nvt * offunit.nanosecond
    )
    settings.solvent_equil_simulation_settings.equilibration_length = (
        conf.solvent_equil_equil_length * offunit.nanosecond
    )
    settings.solvent_equil_simulation_settings.production_length = (
        conf.solvent_equil_production_length * offunit.nanosecond
    )
    # Complex
    settings.complex_equil_simulation_settings.minimization_steps = (
        conf.complex_equil_mini_steps
    )
    settings.complex_equil_simulation_settings.equilibration_length = (
        conf.complex_equil_equil_length * offunit.nanosecond
    )
    settings.complex_equil_simulation_settings.equilibration_length_nvt = (
        conf.complex_equil_equil_length_nvt * offunit.nanosecond
    )
    settings.complex_equil_simulation_settings.production_length = (
        conf.complex_equil_production_length * offunit.nanosecond
    )

    # --------------------
    # Simulation settings
    # --------------------
    # complex
    settings.complex_simulation_settings.early_termination_target_error = (
        conf.early_termination_target_error * offunit.kilocalorie_per_mole
    )
    settings.complex_simulation_settings.equilibration_length = (
        conf.complex_sim_equil_length * offunit.nanosecond
    )
    settings.complex_simulation_settings.minimization_steps = conf.complex_sim_mini_step
    settings.complex_simulation_settings.production_length = (
        conf.complex_sim_prod_length * offunit.nanosecond
    )
    settings.complex_simulation_settings.n_replicas = len(conf.complex_lambda_elec)
    # solvent
    settings.solvent_simulation_settings.early_termination_target_error = (
        conf.early_termination_target_error * offunit.kilocalorie_per_mole
    )
    settings.solvent_simulation_settings.equilibration_length = (
        conf.solvent_sim_equil_length * offunit.nanosecond
    )
    settings.solvent_simulation_settings.minimization_steps = conf.solvent_sim_mini_step
    settings.solvent_simulation_settings.production_length = (
        conf.solvent_sim_prod_length * offunit.nanosecond
    )
    settings.solvent_simulation_settings.n_replicas = len(conf.solvent_lambda_elec)

    # -------------------
    # Solvation Settings
    # -------------------
    # complex
    # Membrane / non-membrane solvation
    if conf.membrane_system:
        if conf.complex_box_vectors is None:
            raise ValueError(
                "complex_box_vectors must be provided when membrane_system is True"
            )

        settings.complex_solvation_settings.box_shape = None
        settings.complex_solvation_settings.solvent_padding = None
        settings.complex_solvation_settings.box_vectors = (
            box_vectors_matrix(conf.complex_box_vectors) * offunit.nanometer
        )
    else:
        settings.complex_solvation_settings.box_shape = conf.box_shape
        settings.complex_solvation_settings.solvent_padding = (
            conf.complex_solvation_padding * offunit.nanometer
        )
    settings.complex_solvation_settings.solvent_model = conf.solvent_model

    # Solvent
    settings.solvent_solvation_settings.box_shape = conf.box_shape
    settings.solvent_solvation_settings.solvent_padding = (
        conf.solvent_solvation_padding * offunit.nanometer
    )
    settings.solvent_solvation_settings.solvent_model = conf.solvent_model

    # --------------------
    # Integrator settings
    # --------------------
    # complex
    if conf.membrane_system:
        settings.complex_integrator_settings.barostat = "MonteCarloMembraneBarostat"
        settings.complex_integrator_settings.surface_tension = (
            conf.surface_tension * offunit.bar * offunit.nanometer
        )
    settings.complex_integrator_settings.timestep = (
        conf.complex_timestep * offunit.femtosecond
    )
    settings.complex_integrator_settings.barostat_frequency = (
        conf.barostat_frequency * offunit.timestep
    )

    # solvent
    settings.solvent_integrator_settings.timestep = (
        conf.solvent_timestep * offunit.femtosecond
    )
    settings.solvent_integrator_settings.barostat_frequency = (
        conf.barostat_frequency * offunit.timestep
    )

    # --------------------
    # Lambda settings
    # --------------------
    settings.complex_lambda_settings.lambda_elec = conf.complex_lambda_elec
    settings.complex_lambda_settings.lambda_restraints = conf.complex_lambda_restraints
    settings.complex_lambda_settings.lambda_vdw = conf.complex_lambda_vdw

    settings.solvent_lambda_settings.lambda_elec = conf.solvent_lambda_elec
    settings.solvent_lambda_settings.lambda_restraints = conf.solvent_lambda_restraints
    settings.solvent_lambda_settings.lambda_vdw = conf.solvent_lambda_vdw

    protocol = AbsoluteBindingProtocol(settings=settings)

    return protocol


def dict_to_namespace(d):
    import argparse

    if isinstance(d, dict):
        return argparse.Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def main() -> None:
    args = parse_args()

    conf = dict_to_namespace(load_workflow_config(args.conf_file, "abfe"))

    # ---------------------------------------
    # Project layout: <workdir>/<jobname>/{inputs,json,results}; copies config +
    # input files into it and repoints conf at the copies.
    # ---------------------------------------
    prepare_project(
        conf,
        args.conf_file,
        default_jobname="abfe",
        input_attrs=[
            ("protein_pdb", "protein PDB"),
            ("ligand_sdf", "ligand"),
            ("cofactor_sdf", "cofactor"),
            ("cofactor_template_sdf", "cofactor template"),
        ],
        optional_file_attrs=[
            ("complex_box_vectors", "PBC file"),
            ("complex_box_size", "PBC file"),
        ],
    )

    # ---------------------------------------
    #           Create protocol
    # ---------------------------------------
    protocol = _create_protocol(conf)

    # -------------------------------------
    #         Create Chemical System
    # -------------------------------------
    charge_settings = OpenFFPartialChargeSettings(
        partial_charge_method=conf.partial_charge_method,
        off_toolkit_backend=conf.off_toolkit_backend,
    )
    ligands = load_ligands(conf.ligand_sdf)
    skip_charge = bool(getattr(conf, "skip_charge_assignment", False))
    if skip_charge:
        print("skip_charge_assignment: using stored charges as-is (no charge check).")
    else:
        ligands = ensure_partial_charges(
            ligands,
            molecule_kind="ligand",
            method=charge_settings.partial_charge_method,
            toolkit_backend=charge_settings.off_toolkit_backend,
            generate_n_conformers=charge_settings.number_of_conformers,
            nagl_model=charge_settings.nagl_model,
            processors=4,
        )
    solv = openfe.SolventComponent(
        positive_ion=getattr(conf, "positive_ion", "Na+"),
        negative_ion=getattr(conf, "negative_ion", "Cl-"),
        neutralize=getattr(conf, "neutralize", True),
        ion_concentration=getattr(conf, "ion_concentration", 0.15) * offunit.molar,
    )

    # Cofactor-from-template mode: build identical-charge cofactor copies from ONE
    # charged template SDF placed on the matching residues in the complex PDB, and
    # strip those residues from the protein PDB so the protein FF doesn't see them.
    protein_pdb_path = str(conf.protein_pdb)
    template_cofactors = None
    cof_template = getattr(conf, "cofactor_template_sdf", None)
    cof_resnames = getattr(conf, "cofactor_resnames", None)
    if cof_template and cof_resnames:
        import pathlib as _pl

        from chemflow.fe.openfe.cofactor_from_pdb import (
            build_cofactors_from_pdb,
            strip_residues,
        )
        template_cofactors = build_cofactors_from_pdb(
            cof_template, conf.protein_pdb, cof_resnames, charge_settings=charge_settings,
        )
        print(f"Built {len(template_cofactors)} cofactor copy/copies from template "
              f"'{cof_template}' on resname(s) {cof_resnames} (identical charges).")
        stripped = _pl.Path(conf.json_dir).parent / "inputs" / "protein_nocofactor.pdb"
        stripped.parent.mkdir(parents=True, exist_ok=True)
        strip_residues(conf.protein_pdb, cof_resnames, stripped)
        protein_pdb_path = str(stripped)

    if conf.membrane_system:
        box_vectors = box_vectors_matrix(conf.complex_box_vectors) * offunit.nanometer
        print(f"Complex is membrane system with PBC: {box_vectors}"
              f"Using openfe.ProteinMembraneComponent (PDB or mmCIF) loading structure")

        prot = load_protein_component(
            protein_pdb_path, membrane=True, box_vectors=box_vectors,
        )

        try:
            prot.validate()
        except ValidationError as e:
            print(f"Membrane PBC Validation failed: {e}")
        else:
            print("Membrane PBC Validation successful! Proceeding with next steps...")

    else:
        prot = load_protein_component(protein_pdb_path)

    components_dict = {
        "protein": prot,
        "solvent": solv,
    }

    if template_cofactors is not None:
        cofactors = template_cofactors
        print("Using template cofactors")
        for idx, cofactor in enumerate(cofactors):
            components_dict[f"cofactor_{idx}"] = cofactor
    elif conf.cofactor_sdf is not None:
        cofactors = load_cofactor(conf.cofactor_sdf)
        print(f"There are {len(cofactors)} cofactor molecule(s)")
        if not skip_charge:
            cofactors = ensure_partial_charges(
                cofactors,
                molecule_kind="cofactor",
                method=charge_settings.partial_charge_method,
                toolkit_backend=charge_settings.off_toolkit_backend,
                generate_n_conformers=charge_settings.number_of_conformers,
                nagl_model=charge_settings.nagl_model,
                processors=4,
            )
            # Identical (graph-isomorphic) cofactor copies must share one charge set.
            cofactors = harmonize_isomorphic_charges(cofactors, molecule_kind="cofactor")
        for idx, cofactor in enumerate(cofactors):
            components_dict[f"cofactor_{idx}"] = cofactor
    else:
        cofactors = None

    systemA = dict(components_dict)
    systemB = dict(components_dict)

    systemA["ligand"] = ligands[0]

    stateA = openfe.ChemicalSystem(systemA)

    stateB = openfe.ChemicalSystem(systemB)

    # Save protocol JSON
    transformation_name = conf.name
    transformation = openfe.Transformation(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
        protocol=protocol,
        name=transformation_name,
    )

    json_path = pathlib.Path(conf.json_dir) / f"{transformation_name}.json"
    transformation.to_json(json_path)
    print(f"Saved protocol to json file: {json_path}")


if __name__ == "__main__":
    main()
