#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
from typing import Literal

import numpy as np
import openfe
from openfe.protocols.openmm_rfe import (
    RelativeHybridTopologyProtocol,
)
from openfe.protocols.openmm_utils.omm_settings import OpenFFPartialChargeSettings
from openff.units import unit as offunit
from openmm import Platform, System, VerletIntegrator
from openmm.app import Simulation, Topology
from rdkit import Chem

from alchemforge.openfe.partial_charges import ensure_partial_charges
from alchemforge.openfe.workflow_defaults import load_workflow_config


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
    parser = argparse.ArgumentParser(description="Run OpenFE RBFE workflow on HPC.")

    parser.add_argument(
        "--conf_file",
        type=pathlib.Path,
        required=True,
        help="Path to configuration file.",
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

# def load_ligands(ligand_sdf: pathlib.Path) -> list[openfe.SmallMoleculeComponent]:
#     supp = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
#     ligands = [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]
#     if not ligands:
#         raise ValueError(f"No valid ligands found in {ligand_sdf}")
#     return ligands


def load_ligands(
    ligand_sdf: pathlib.Path,
    names: list[str] | None = None,
) -> list[openfe.SmallMoleculeComponent]:

    supp = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)

    mol_dict = {}

    for mol in supp:
        if mol is None:
            continue

        mol_name = None
        if mol.HasProp("_Name"):
            mol_name = mol.GetProp("_Name").strip()
        elif mol.HasProp("NAME"):
            mol_name = mol.GetProp("NAME").strip()

        if mol_name is None:
            continue

        mol_dict[mol_name] = openfe.SmallMoleculeComponent.from_rdkit(mol)

    if names is None:
        # return all in file order
        return list(mol_dict.values())

    # preserve input order
    ligands = []
    for name in names:
        if name not in mol_dict:
            raise ValueError(f"Ligand {name} not found in SDF")
        ligands.append(mol_dict[name])

    return ligands


def load_cofactor(cofactor_sdf: pathlib.Path) -> list[openfe.SmallMoleculeComponent]:
    supp = Chem.SDMolSupplier(str(cofactor_sdf), removeHs=False)
    cofactors = [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]
    if not cofactors:
        raise ValueError(f"No valid cofactor found in {cofactor_sdf}")
    return cofactors


# -------------------------------------------------
#    Helper for generating perturbation networks
# -------------------------------------------------
def _get_mapper(mapper_name: str):
    from openfe import setup

    if mapper_name == "LomapAtomMapper":
        return setup.LomapAtomMapper(max3d=1.0, element_change=False)
    if mapper_name == "KartografAtomMapper":
        return setup.KartografAtomMapper()
    raise ValueError(f"Unknown mapper: {mapper_name}")


def _get_ligand_network_planner(planner_name: str):
    import openfe

    planners = openfe.ligand_network_planning

    if planner_name == "minimal_spanning":
        return planners.generate_minimal_spanning_network, False
    if planner_name == "minimal_redundant":
        return planners.generate_minimal_redundant_network, False
    if planner_name == "radial":
        return planners.generate_radial_network, True

    raise ValueError(f"Unknown planner: {planner_name}")


def _default_lomap_scorer():
    try:
        from openfe import lomap_scorers

        return lomap_scorers.default_lomap_score
    except Exception:
        from openfe.setup.atom_mapping.lomap_scorers import default_lomap_score

        return default_lomap_score


def _create_ligand_network(
    ligands: list[openfe.SmallMoleculeComponent],
    mapper_name: Literal["LomapAtomMapper", "KartografAtomMapper"],
    planner_name: Literal["minimal_spanning", "minimal_redundant", "radial"],
    workdir: pathlib.Path,
):
    scorer = _default_lomap_scorer()
    network_planner, needs_center = _get_ligand_network_planner(
        planner_name=planner_name
    )
    mapper = _get_mapper(mapper_name)
    if needs_center:
        ligand_network = network_planner(
            ligands=ligands[1:],
            central_ligand=ligands[0],
            mappers=[mapper],
        )
    else:
        ligand_network = network_planner(
            ligands=ligands, mappers=[mapper], scorer=scorer
        )
    workdir.mkdir(exist_ok=True, parents=True)
    with open(workdir / "ligand_network.graphml", mode="w") as f:
        f.write(ligand_network.to_graphml())
    return ligand_network


def _load_ligand_network(graphml_path: pathlib.Path, workdir: pathlib.Path):
    source = pathlib.Path(graphml_path)
    if not source.is_file():
        raise FileNotFoundError(f"Ligand network GraphML not found: {source}")
    text = source.read_text(encoding="utf-8")
    network = openfe.LigandNetwork.from_graphml(text)
    workdir.mkdir(exist_ok=True, parents=True)
    (workdir / "ligand_network.graphml").write_text(text, encoding="utf-8")
    return network


def _create_alchemical_network(
    ligand_network, protein, solvent, cofactors, complex_protocol, solvent_protocol
):
    transformations = []
    for mapping in ligand_network.edges:
        for leg in ["solvent", "complex"]:
            # use the solvent and protein created above
            sysA_dict = {"ligand": mapping.componentA, "solvent": solvent}
            sysB_dict = {"ligand": mapping.componentB, "solvent": solvent}

            if cofactors is not None:
                for idx, cofactor in enumerate(cofactors):
                    sysA_dict[f"cofactor_{idx}"] = cofactor
                    sysB_dict[f"cofactor_{idx}"] = cofactor

            if leg == "complex":
                # If this is a complex transformation we use the complex protocol
                # and add in the protein to the chemical states
                protocol = complex_protocol
                sysA_dict["protein"] = protein
                sysB_dict["protein"] = protein
            else:
                # If this is a solvent transformation we just use the solvent protocol
                protocol = solvent_protocol

            # we don't have to name objects, but it can make things (like filenames) more convenient
            sysA = openfe.ChemicalSystem(
                sysA_dict, name=f"{mapping.componentA.name}_{leg}"
            )
            sysB = openfe.ChemicalSystem(
                sysB_dict, name=f"{mapping.componentB.name}_{leg}"
            )

            prefix = "rbfe_"  # prefix is only to exactly reproduce CLI

            transformation = openfe.Transformation(
                stateA=sysA,
                stateB=sysB,
                mapping=mapping,
                protocol=protocol,  # use protocol created above
                name=f"{prefix}{sysA.name}_{sysB.name}",
            )
            transformations.append(transformation)

    network = openfe.AlchemicalNetwork(transformations)
    return network


def _create_solvent_leg_protocol(conf):
    settings = RelativeHybridTopologyProtocol.default_settings()

    if conf.platform == "CUDA":
        if test_cuda():
            settings.engine_settings.compute_platform = "CUDA"
            print("CUDA is available; using CUDA")
        else:
            settings.engine_settings.compute_platform = "CPU"
            print("CUDA is unavailable; falling back to CPU")
    elif conf.platform == "OpenCL":
        settings.engine_settings.compute_platform = "OpenCL"
        print("Using OpenCL")
    elif conf.platform == "CPU":
        settings.engine_settings.compute_platform = "CPU"
        print("Using CPU")
    else:
        raise ValueError(f"Unsupported OpenMM platform: {conf.platform}")

    settings.protocol_repeats = conf.solvent_protocol_repeats

    # Thermo settings
    print("Thermo settings ...")
    settings.thermo_settings.pressure = conf.pressure * offunit.bar
    settings.thermo_settings.temperature = conf.temperature * offunit.kelvin

    # Alchemical settings
    settings.alchemical_settings.endstate_dispersion_correction = (
        conf.endstate_dispersion_correction
    )
    settings.alchemical_settings.explicit_charge_correction = (
        conf.explicit_charge_correction
    )
    settings.alchemical_settings.explicit_charge_correction_cutoff = (
        conf.explicit_charge_correction_cutoff * offunit.nanometer
    )
    settings.alchemical_settings.softcore_LJ = conf.softcore_LJ
    settings.alchemical_settings.softcore_alpha = conf.softcore_alpha
    settings.alchemical_settings.turn_off_core_unique_exceptions = (
        conf.turn_off_core_unique_exceptions
    )
    settings.alchemical_settings.use_dispersion_correction = (
        conf.use_dispersion_correction
    )
    settings.lambda_settings.lambda_windows = conf.n_lambda_windows_solvent_leg

    # General FF settings
    settings.forcefield_settings.constraints = conf.ff_constraints
    settings.forcefield_settings.hydrogen_mass = conf.hydrogen_mass
    settings.forcefield_settings.nonbonded_cutoff = (
        conf.nonbonded_cutoff * offunit.nanometer
    )
    settings.forcefield_settings.nonbonded_method = conf.nonbonded_method
    settings.forcefield_settings.rigid_water = conf.rigid_water
    settings.forcefield_settings.small_molecule_forcefield = (
        conf.small_molecule_forcefield
    )
    settings.forcefield_settings.forcefields = conf.forcefields
    settings.partial_charge_settings.off_toolkit_backend = conf.off_toolkit_backend

    if conf.partial_charge_method == "am1bcc":
        settings.partial_charge_settings.partial_charge_method = "am1bcc"
    elif conf.partial_charge_method == "nagl":
        settings.partial_charge_settings.partial_charge_method = "nagl"
        settings.partial_charge_settings.nagl_model = "openff-gnn-am1bcc-0.1.0-rc.3.pt"

    # Solvation settings:
    settings.solvation_settings.box_shape = conf.box_shape
    settings.solvation_settings.solvent_model = conf.solvent_model
    settings.solvation_settings.solvent_padding = (
        conf.solvent_solvation_padding * offunit.nanometer
    )

    # Simulation settings
    settings.simulation_settings.early_termination_target_error = (
        conf.early_termination_target_error * offunit.kilocalorie_per_mole
    )
    settings.simulation_settings.equilibration_length = (
        conf.solvent_sim_equil_length * offunit.nanosecond
    )
    settings.simulation_settings.minimization_steps = conf.solvent_sim_mini_step
    settings.simulation_settings.production_length = (
        conf.solvent_sim_prod_length * offunit.nanosecond
    )
    settings.simulation_settings.n_replicas = conf.n_lambda_windows_solvent_leg

    # Integrator settings
    settings.integrator_settings.timestep = conf.solvent_timestep * offunit.femtosecond
    settings.integrator_settings.barostat_frequency = (
        conf.barostat_frequency * offunit.timestep
    )

    return RelativeHybridTopologyProtocol(settings=settings)


def _create_complex_leg_protocol(conf):
    settings = RelativeHybridTopologyProtocol.default_settings()

    if conf.platform == "CUDA":
        if test_cuda():
            settings.engine_settings.compute_platform = "CUDA"
            print("CUDA is available; using CUDA")
        else:
            settings.engine_settings.compute_platform = "CPU"
            print("CUDA is unavailable; falling back to CPU")
    elif conf.platform == "OpenCL":
        settings.engine_settings.compute_platform = "OpenCL"
        print("Using OpenCL")
    elif conf.platform == "CPU":
        settings.engine_settings.compute_platform = "CPU"
        print("Using CPU")
    else:
        raise ValueError(f"Unsupported OpenMM platform: {conf.platform}")

    settings.protocol_repeats = conf.complex_protocol_repeats

    # Thermo settings
    print("Thermo settings ...")
    settings.thermo_settings.pressure = conf.pressure * offunit.bar
    settings.thermo_settings.temperature = conf.temperature * offunit.kelvin

    # Alchemical settings
    settings.alchemical_settings.endstate_dispersion_correction = (
        conf.endstate_dispersion_correction
    )
    settings.alchemical_settings.explicit_charge_correction = (
        conf.explicit_charge_correction
    )
    settings.alchemical_settings.explicit_charge_correction_cutoff = (
        conf.explicit_charge_correction_cutoff * offunit.nanometer
    )
    settings.alchemical_settings.softcore_LJ = conf.softcore_LJ
    settings.alchemical_settings.softcore_alpha = conf.softcore_alpha
    settings.alchemical_settings.turn_off_core_unique_exceptions = (
        conf.turn_off_core_unique_exceptions
    )
    settings.alchemical_settings.use_dispersion_correction = (
        conf.use_dispersion_correction
    )
    settings.lambda_settings.lambda_windows = conf.n_lambda_windows_complex_leg

    # General FF settings
    settings.forcefield_settings.constraints = conf.ff_constraints
    settings.forcefield_settings.hydrogen_mass = conf.hydrogen_mass
    settings.forcefield_settings.nonbonded_cutoff = (
        conf.nonbonded_cutoff * offunit.nanometer
    )
    settings.forcefield_settings.nonbonded_method = conf.nonbonded_method
    settings.forcefield_settings.rigid_water = conf.rigid_water
    settings.forcefield_settings.small_molecule_forcefield = (
        conf.small_molecule_forcefield
    )
    settings.forcefield_settings.forcefields = conf.forcefields
    settings.partial_charge_settings.off_toolkit_backend = conf.off_toolkit_backend

    if conf.partial_charge_method == "am1bcc":
        settings.partial_charge_settings.partial_charge_method = "am1bcc"
    elif conf.partial_charge_method == "nagl":
        settings.partial_charge_settings.partial_charge_method = "nagl"
        settings.partial_charge_settings.nagl_model = "openff-gnn-am1bcc-0.1.0-rc.3.pt"

    # Membrane / non-membrane solvation
    if conf.membrane_system:
        if conf.complex_box_vectors is None:
            raise ValueError(
                "complex_box_vectors must be provided when membrane_system is set"
            )

        settings.solvation_settings.box_shape = None
        settings.solvation_settings.solvent_padding = None
        settings.solvation_settings.box_vectors = (
            np.array(conf.complex_box_vectors) * offunit.nanometer
        )
    else:
        settings.solvation_settings.box_shape = conf.box_shape
        settings.solvation_settings.solvent_padding = (
            conf.complex_solvation_padding * offunit.nanometer
        )
    settings.solvation_settings.solvent_model = conf.solvent_model

    # Simulation settings
    settings.simulation_settings.early_termination_target_error = (
        conf.early_termination_target_error * offunit.kilocalorie_per_mole
    )
    settings.simulation_settings.equilibration_length = (
        conf.complex_sim_equil_length * offunit.nanosecond
    )
    settings.simulation_settings.minimization_steps = conf.complex_sim_mini_step
    settings.simulation_settings.production_length = (
        conf.complex_sim_prod_length * offunit.nanosecond
    )
    settings.simulation_settings.n_replicas = conf.n_lambda_windows_complex_leg

    # Integrator settings
    settings.integrator_settings.timestep = conf.complex_timestep * offunit.femtosecond
    settings.integrator_settings.barostat_frequency = (
        conf.barostat_frequency * offunit.timestep
    )
    if conf.membrane_system:
        settings.integrator_settings.surface_tension = (
            conf.surface_tension * offunit.bar * offunit.nanometer
        )

    return RelativeHybridTopologyProtocol(settings=settings)


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

    conf = dict_to_namespace(load_workflow_config(args.conf_file, "rbfe"))

    # ---------------------------------------
    # Check if every input file exists
    # ---------------------------------------
    if (
        not pathlib.Path(conf.protein_pdb).exists()
        or not pathlib.Path(conf.protein_pdb).is_file()
    ):
        raise FileNotFoundError(f"PDB file not found: {conf.protein_pdb}")
    print(f"Found PDB file: {conf.protein_pdb}")

    if (
        not pathlib.Path(conf.ligand_sdf).exists()
        or not pathlib.Path(conf.ligand_sdf).is_file()
    ):
        raise FileNotFoundError(f"Ligand file not found: {conf.ligand_sdf}")
    print(f"Found ligand file: {conf.ligand_sdf}")

    if conf.cofactor_sdf is not None:
        if (
            not pathlib.Path(conf.cofactor_sdf).exists()
            or not pathlib.Path(conf.cofactor_sdf).is_file()
        ):
            raise FileNotFoundError(f"Cofactor file not found: {conf.cofactor_sdf}")
        print(f"Found cofactor file: {conf.cofactor_sdf}")

    pathlib.Path(conf.workdir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(conf.json_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------
    #           Create protocol
    # ---------------------------------------
    sol_protocol = _create_solvent_leg_protocol(conf)
    comp_protocol = _create_complex_leg_protocol(conf)

    # -------------------------------------
    #         Create Chemical System
    # -------------------------------------
    charge_settings = OpenFFPartialChargeSettings(
        partial_charge_method=conf.partial_charge_method,
        off_toolkit_backend=conf.off_toolkit_backend,
    )

    if conf.alchemical_network == "all":
        names = None
    else:
        names = list(conf.alchemical_network)

    ligands = load_ligands(conf.ligand_sdf, names)

    ligands = ensure_partial_charges(
        ligands,
        molecule_kind="ligand",
        method=charge_settings.partial_charge_method,
        toolkit_backend=charge_settings.off_toolkit_backend,
        generate_n_conformers=charge_settings.number_of_conformers,
        nagl_model=charge_settings.nagl_model,
        processors=4,
    )

    ligand_network = (
        _load_ligand_network(
            pathlib.Path(conf.ligand_network_graphml),
            pathlib.Path(conf.json_dir),
        )
        if conf.ligand_network_graphml
        else _create_ligand_network(
            ligands,
            mapper_name=conf.mapper,
            planner_name=conf.planner,
            workdir=pathlib.Path(conf.json_dir),
        )
    )

    solv = openfe.SolventComponent()

    if conf.membrane_system:
        box_vectors = np.array(conf.complex_box_vectors) * offunit.nanometer

        prot = openfe.ProteinMembraneComponent.from_pdb_file(
            str(conf.protein_pdb),
            box_vectors=box_vectors,
        )
    else:
        prot = openfe.ProteinComponent.from_pdb_file(str(conf.protein_pdb))

    if conf.cofactor_sdf is not None:
        cofactors = load_cofactor(str(conf.cofactor_sdf))
        print(f"There are {len(cofactors)} cofactor molecule(s)")
        cofactors = ensure_partial_charges(
            cofactors,
            molecule_kind="cofactor",
            method=charge_settings.partial_charge_method,
            toolkit_backend=charge_settings.off_toolkit_backend,
            generate_n_conformers=charge_settings.number_of_conformers,
            nagl_model=charge_settings.nagl_model,
            processors=4,
        )
    else:
        cofactors = None

    network = _create_alchemical_network(
        ligand_network, prot, solv, cofactors, comp_protocol, sol_protocol
    )

    for transformation in network.edges:
        name = transformation.name

        if conf.transformation == "solvent":
            if "solvent" not in name:
                continue

        elif conf.transformation == "complex":
            if "complex" not in name:
                continue

        json_path = pathlib.Path(conf.json_dir) / f"{transformation.name}.json"
        transformation.to_json(json_path)
        print(f"Saved protocol to json file: {json_path}")


if __name__ == "__main__":
    main()
