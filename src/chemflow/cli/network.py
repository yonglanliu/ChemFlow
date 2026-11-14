"""``chemflow network`` — inspect and edit a planned ligand network bundle.

    chemflow network list   DIR                         # show the edges
    chemflow network add    DIR --a LIG1 --b LIG2 [--ligand-sdf ligands.sdf]
    chemflow network remove DIR --drop LIG1:LIG2 [--drop ...]

A network *bundle* is a directory holding ``ligand_network.graphml`` and
``edges.csv`` (``add`` also needs ``network_manifest.json`` + ``ligands.sdf``).
Editing re-maps atoms / re-serialises the GraphML (needs OpenFE), so run it in
the alchemforge environment; ``list`` is pure Python.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _list(args: argparse.Namespace) -> int:
    from chemflow.fe.openfe.utils.network_edit import read_network_edges

    edges = read_network_edges(args.directory)
    print(f"{len(edges)} edges in {args.directory}:")
    for e in edges:
        print(f"  {e.ligand_a}  <->  {e.ligand_b}")
    return 0


def _plan(args: argparse.Namespace) -> int:
    from chemflow.fe.openfe.utils.network_edit import plan_network

    n = plan_network(args.ligands, args.out, mapper=args.mapper, planner=args.planner,
                     max3d=args.max3d, element_change=args.element_change)
    print(f"Planned {n} edges into {args.out} (graphml + edges.csv + ligands.sdf + manifest).")
    print(f"Review/edit with: chemflow network list {args.out}")
    return 0


def _add(args: argparse.Namespace) -> int:
    from chemflow.fe.gromacs.rbfe.mapping import add_network_edge

    directory = Path(args.directory).expanduser().resolve()
    if args.ligand_sdf:
        src = Path(args.ligand_sdf).expanduser().resolve()
        if not src.is_file():
            raise ValueError(f"ligand SDF not found: {src}")
        target = directory / "ligands.sdf"
        if not target.is_file():
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    result = add_network_edge(directory, args.a, args.b)
    print(f"Added edge {args.a} <-> {args.b}" + (f"  ({result})" if result else ""))
    return 0


def _remove(args: argparse.Namespace) -> int:
    from chemflow.fe.openfe.utils.network_edit import drop_edges

    pairs = []
    for spec in args.drop:
        if ":" not in spec:
            raise ValueError(f"--drop must be LIGA:LIGB, got {spec!r}")
        a, b = spec.split(":", 1)
        pairs.append((a.strip(), b.strip()))
    removed = drop_edges(args.directory, pairs)
    print(f"Removed {removed} edge(s); network remains connected.")
    return 0


def add_network_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "network", help="Inspect/edit a planned ligand perturbation network (add/remove edges)")
    commands = parser.add_subparsers(dest="network_command", required=True)

    plan = commands.add_parser("plan", help="Plan a network from a ligands SDF into an editable bundle")
    plan.add_argument("--ligands", required=True, help="ligands SDF (ideally charged, with hydrogens)")
    plan.add_argument("--out", required=True, help="output network bundle directory")
    plan.add_argument("--mapper", default="lomap", help="lomap | kartograf")
    plan.add_argument("--planner", default="minimal_spanning",
                      help="minimal_spanning | minimal_redundant")
    plan.add_argument("--max3d", type=float, default=1.0, help="max 3D atom displacement (A)")
    plan.add_argument("--element-change", action="store_true", help="allow element changes in mappings")
    plan.set_defaults(func=_plan)

    ls = commands.add_parser("list", help="List the edges in a network bundle")
    ls.add_argument("directory")
    ls.set_defaults(func=_list)

    add = commands.add_parser("add", help="Map and add one transformation edge")
    add.add_argument("directory")
    add.add_argument("--a", required=True, help="first ligand name")
    add.add_argument("--b", required=True, help="second ligand name")
    add.add_argument("--ligand-sdf", default=None,
                     help="ligands SDF to place in the bundle if missing (input for mapping)")
    add.set_defaults(func=_add)

    rm = commands.add_parser("remove", help="Remove edges (keeps the network connected)")
    rm.add_argument("directory")
    rm.add_argument("--drop", action="append", required=True, metavar="LIGA:LIGB",
                    help="edge to remove, e.g. --drop lig1:lig2 (repeatable)")
    rm.set_defaults(func=_remove)
