"""Headless RBFE preparation CLI.

Exposes the four preparation steps as subcommands under ``alchemforge rbfe`` so
the whole flow can run without the desktop application:

    alchemforge rbfe fix-pdb       CONFIG   # 1. repair/assemble the receptor PDB
    alchemforge rbfe assign-charge CONFIG   # 2. charge + parameterize the ligands
    alchemforge rbfe plan          CONFIG   # 3. build the perturbation network
    alchemforge rbfe workflow      CONFIG   # 4. stage edges + Slurm submit scripts
    alchemforge rbfe all           CONFIG   # run steps 1-4 in order

Every step reads the same TOML configuration (the one consumed by
``alchemforge ligand-rbfe``). Two optional sections drive the inputs that the
RBFE config does not already carry:

    [protein_prep]                 # inputs for step 1 (fix-pdb)
    cofactor          = "metals.pdb"
    membrane          = "membrane.pdb"
    solvent           = "water.pdb"
    pbc_source        = "box.pdb"
    pdbfixer_command  = "pdbfixer"
    output_directory  = "prepared"   # default: <project>/input/prepared
    overwrite         = false

    [ligand_prep]                  # inputs for step 2 (assign-charge)
    atom_type         = "gaff2"
    net_charge        = 0            # single-ligand override
    net_charges       = [0, -1]      # per-ligand overrides (ordered)
    resnames          = ["L01", "L02"]
    obabel_command    = "obabel"
    acpype_command    = "acpype"
    output_directory  = "ligands"    # default: <project>/ligands

Relative paths in these sections are resolved against the config file's
directory. ``charge_method`` for step 2 comes from ``[topology].charge_method``
(the field the RBFE config already defines).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_toml(path: Path) -> dict:
    import tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve(base: Path, value: object) -> Path | None:
    if value is None:
        return None
    candidate = Path(str(value)).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _config_path(args: argparse.Namespace) -> Path:
    return Path(args.config).expanduser().resolve()


# ---------------------------------------------------------------------------
# Step 1: fix-pdb
# ---------------------------------------------------------------------------
def _cmd_fix_pdb(args: argparse.Namespace) -> int:
    from alchemforge.gromacs.rbfe.config import LigandRBFEConfig
    from alchemforge.structure import pdbfix

    cfg_path = _config_path(args)
    config = LigandRBFEConfig.load(cfg_path)
    data = _load_toml(cfg_path)
    base = cfg_path.parent

    protein = _resolve(base, data.get("system", {}).get("protein"))
    if protein is None:
        raise ValueError(
            "fix-pdb requires [system].protein in the configuration"
        )
    prep = data.get("protein_prep", {})
    output = _resolve(base, prep.get("output_directory")) or (
        config.project_dir / "input" / "prepared"
    )
    output.mkdir(parents=True, exist_ok=True)
    work = output / ".alchemforge-pdbfix"

    _, manifest, tasks = pdbfix.stage(
        None,
        output,
        work,
        include_components=False,
        protein_pdb=protein,
        cofactor_pdb=_resolve(base, prep.get("cofactor")),
        membrane_pdb=_resolve(base, prep.get("membrane")),
        solvent_pdb=_resolve(base, prep.get("solvent")),
        box_source_pdb=_resolve(base, prep.get("pbc_source")),
        assemble_system=True,
        overwrite=bool(prep.get("overwrite", False)),
        pdbfixer_command=str(prep.get("pdbfixer_command", "pdbfixer")),
    )
    result = pdbfix.run_manifest(manifest)
    if result == 0:
        print("Fixed components: " + ", ".join(task.output.name for task in tasks))
        print(f"Assembled system: {output / 'complex.pdb'}")
    return result


# ---------------------------------------------------------------------------
# Step 2: assign-charge
# ---------------------------------------------------------------------------
def _cmd_assign_charge(args: argparse.Namespace) -> int:
    from alchemforge.gromacs import ligand_prepare
    from alchemforge.gromacs.rbfe.config import LigandRBFEConfig

    cfg_path = _config_path(args)
    config = LigandRBFEConfig.load(cfg_path)
    data = _load_toml(cfg_path)
    base = cfg_path.parent

    if not config.ligand_library.is_file():
        raise ValueError(f"Ligand library not found: {config.ligand_library}")

    prep = data.get("ligand_prep", {})
    output = _resolve(base, prep.get("output_directory")) or (
        config.project_dir / "ligands"
    )
    net_charge = prep.get("net_charge")
    charge_method = config.charge_method if config.charge_method in {"bcc", "user"} else "bcc"
    ligand_config = ligand_prepare.LigandPreparationConfig(
        input_file=config.ligand_library,
        output_directory=output,
        charge_method=charge_method,
        atom_type=str(prep.get("atom_type", "gaff2")),
        net_charge=int(net_charge) if net_charge is not None else None,
        net_charges=tuple(int(value) for value in prep.get("net_charges", []) or ()),
        resnames=tuple(str(value) for value in prep.get("resnames", []) or ()),
        obabel_command=str(prep.get("obabel_command", "obabel")),
        acpype_command=str(prep.get("acpype_command", "acpype")),
    )
    outputs = ligand_prepare.prepare(ligand_config)
    print(
        f"Charged and parameterized {len(outputs)} ligand(s) "
        f"({charge_method}) in {output}"
    )
    return 0


# ---------------------------------------------------------------------------
# Step 3: plan
# ---------------------------------------------------------------------------
def _cmd_plan(args: argparse.Namespace) -> int:
    from alchemforge.gromacs.rbfe.config import LigandRBFEConfig
    from alchemforge.gromacs.rbfe.mapping import plan_network

    config = LigandRBFEConfig.load(_config_path(args))
    config.validate_inputs()
    network = plan_network(config)
    print(
        f"Planned network: {network.ligand_count} ligands, "
        f"{network.edge_count} edges"
    )
    print(f"Edge table: {network.edges_csv}")
    return 0


# ---------------------------------------------------------------------------
# Step 4: workflow
# ---------------------------------------------------------------------------
def _cmd_workflow(args: argparse.Namespace) -> int:
    from alchemforge.gromacs.rbfe.config import LigandRBFEConfig
    from alchemforge.gromacs.rbfe.workflow import stage

    config = LigandRBFEConfig.load(_config_path(args))
    config.validate_inputs()
    network_csv = config.project_dir / "network" / "edges.csv"
    # Reuse a network from a prior `plan` step when present; otherwise plan now.
    reuse = network_csv.is_file() and not args.replan
    generated, network = stage(config, reuse_network=reuse)
    action = "Reused" if reuse else "Planned"
    print(f"{action} network: {network.edge_count} edges")
    print(f"Staged project: {config.project_dir}")
    print(f"Submit script: {config.project_dir / 'submit_production_array.sh'}")
    print(f"Shell configuration: {generated}")
    return 0


# ---------------------------------------------------------------------------
# Pipeline: all
# ---------------------------------------------------------------------------
def _cmd_all(args: argparse.Namespace) -> int:
    steps = (
        ("fix-pdb", _cmd_fix_pdb),
        ("assign-charge", _cmd_assign_charge),
        ("plan", _cmd_plan),
        ("workflow", _cmd_workflow),
    )
    for name, handler in steps:
        print(f"\n=== rbfe {name} ===")
        result = handler(args)
        if result != 0:
            print(f"Step '{name}' failed with exit code {result}", flush=True)
            return result
    print("\nAll four preparation steps completed.")
    return 0


def add_rbfe_prep_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``alchemforge rbfe`` preparation command group."""
    parser = subparsers.add_parser(
        "rbfe",
        help="Headless RBFE prep: fix-pdb, assign-charge, plan, workflow",
        description=(
            "Run the four relative-binding-free-energy preparation steps from a "
            "single TOML configuration, without the desktop application."
        ),
    )
    steps = parser.add_subparsers(dest="rbfe_step", required=True)

    def _add(name: str, handler, help_text: str) -> None:
        step = steps.add_parser(name, help=help_text)
        step.add_argument("config", type=Path, help="RBFE TOML configuration file")
        if name in ("workflow", "all"):
            step.add_argument(
                "--replan",
                action="store_true",
                help="Re-plan the network instead of reusing an existing one",
            )
        step.set_defaults(func=handler)

    _add("fix-pdb", _cmd_fix_pdb, "Step 1: repair and assemble the receptor PDB")
    _add("assign-charge", _cmd_assign_charge, "Step 2: charge and parameterize ligands")
    _add("plan", _cmd_plan, "Step 3: build the perturbation network")
    _add("workflow", _cmd_workflow, "Step 4: stage edges and Slurm submit scripts")
    _add("all", _cmd_all, "Run steps 1-4 in sequence")
    return parser
