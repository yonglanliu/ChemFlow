"""Edit a planned ligand network bundle (edges.csv + ligand_network.graphml).

Qt-free so the ``chemflow network`` CLI can call it on a compute node; OpenFE
is imported lazily only to re-serialise the GraphML after dropping edges. Adding
an edge (which re-maps atoms) lives in :mod:`chemflow.fe.gromacs.rbfe.mapping`.
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NetworkEdge:
    ligand_a: str
    ligand_b: str


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def read_network_edges(network_dir: str | Path) -> list[NetworkEdge]:
    directory = Path(network_dir).expanduser().resolve()
    csv_path = directory / "edges.csv"
    if not csv_path.is_file():
        raise ValueError(f"edges.csv not found in {directory}")
    _fields, rows = _read_rows(csv_path)
    if not rows or not {"ligand_a", "ligand_b"}.issubset(rows[0]):
        raise ValueError("edges.csv must contain ligand_a and ligand_b columns")
    return [NetworkEdge(r["ligand_a"].strip(), r["ligand_b"].strip()) for r in rows]


def _validate_connected(edges: list[NetworkEdge], expected: set[str]) -> None:
    if not edges:
        raise ValueError("Keep at least one transformation")
    adjacency: dict[str, set[str]] = {name: set() for name in expected}
    for edge in edges:
        adjacency.setdefault(edge.ligand_a, set()).add(edge.ligand_b)
        adjacency.setdefault(edge.ligand_b, set()).add(edge.ligand_a)
    start = next(iter(expected))
    reached = {start}
    queue = deque([start])
    while queue:
        for neighbour in adjacency.get(queue.popleft(), set()) - reached:
            reached.add(neighbour)
            queue.append(neighbour)
    missing = expected - reached
    if missing:
        raise ValueError("Removing those edges would disconnect the network; "
                         "unreachable: " + ", ".join(sorted(missing)))


_MAPPERS = {
    "lomap": "LomapAtomMapper", "lomapatommapper": "LomapAtomMapper",
    "kartograf": "KartografAtomMapper", "kartografatommapper": "KartografAtomMapper",
}
_PLANNERS = {
    "minimal_spanning": "generate_minimal_spanning_network",
    "minimal_redundant": "generate_minimal_redundant_network",
}


def plan_network(ligands_sdf: str | Path, out_dir: str | Path, *, mapper: str = "lomap",
                 planner: str = "minimal_spanning", max3d: float = 1.0,
                 element_change: bool = False) -> int:
    """Plan a perturbation network from a ligands SDF and write an editable bundle.

    Writes ``ligand_network.graphml`` + ``edges.csv`` + ``network_manifest.json``
    and copies the SDF to ``ligands.sdf`` in ``out_dir`` — the bundle both
    ``create_json`` (GraphML) and ``add``/``drop`` (edges.csv + manifest) consume.
    Returns the number of edges planned.
    """
    import json
    import shutil

    import openfe
    from rdkit import Chem

    src = Path(ligands_sdf).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        raise ValueError(f"ligands SDF not found: {src}")

    molecules = [m for m in Chem.SDMolSupplier(str(src), removeHs=False) if m is not None]
    if len(molecules) < 2:
        raise ValueError(f"need at least 2 ligands to plan a network; found {len(molecules)} in {src}")
    components = [openfe.SmallMoleculeComponent.from_rdkit(m) for m in molecules]

    mapper_name = _MAPPERS.get(mapper.lower())
    if mapper_name == "LomapAtomMapper":
        atom_mapper = openfe.setup.LomapAtomMapper(threed=True, max3d=max3d, element_change=element_change)
    elif mapper_name == "KartografAtomMapper":
        atom_mapper = openfe.setup.KartografAtomMapper(atom_max_distance=max3d)
    else:
        raise ValueError(f"unknown mapper: {mapper}")

    planner_fn = _PLANNERS.get(planner)
    if planner_fn is None:
        raise ValueError(f"planner must be one of {sorted(_PLANNERS)} (got {planner!r})")
    scorer = openfe.lomap_scorers.default_lomap_score
    network = getattr(openfe.ligand_network_planning, planner_fn)(
        ligands=components, mappers=[atom_mapper], scorer=scorer)

    (out / "ligand_network.graphml").write_text(network.to_graphml(), encoding="utf-8")
    shutil.copy2(src, out / "ligands.sdf")
    (out / "network_manifest.json").write_text(
        json.dumps({"atom_mapper": mapper.lower(), "max3d_angstrom": max3d,
                    "element_change": element_change}), encoding="utf-8")
    edges = list(network.edges)
    mappings_dir = out / "mappings"
    mappings_dir.mkdir(exist_ok=True)
    with (out / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["edge_id", "ligand_a", "ligand_b", "lomap_score"])
        for i, edge in enumerate(edges):
            edge_id = f"edge_{i}"
            try:
                score = f"{scorer(edge):.4f}"
            except Exception:
                score = ""
            writer.writerow([edge_id, edge.componentA.name, edge.componentB.name, score])
            # Per-edge atom mapping for the viewer (componentA_to_componentB).
            try:
                amap = {int(k): int(v) for k, v in edge.componentA_to_componentB.items()}
                (mappings_dir / f"{edge_id}.json").write_text(
                    json.dumps({"componentA_to_componentB": {str(k): v for k, v in amap.items()}}),
                    encoding="utf-8")
            except Exception:
                pass
    return len(edges)


def drop_edges(network_dir: str | Path, drop_pairs) -> int:
    """Remove the given ``(ligand_a, ligand_b)`` pairs; rewrite GraphML + edges.csv.

    Validates that the remaining network stays connected. Returns how many edges
    were removed.
    """
    import openfe

    directory = Path(network_dir).expanduser().resolve()
    graphml = directory / "ligand_network.graphml"
    csv_path = directory / "edges.csv"
    if not graphml.is_file():
        raise ValueError(f"ligand_network.graphml not found in {directory}")
    fields, rows = _read_rows(csv_path)
    drop = {frozenset(pair) for pair in drop_pairs}
    kept = [r for r in rows if frozenset((r["ligand_a"], r["ligand_b"])) not in drop]
    removed = len(rows) - len(kept)
    if removed == 0:
        raise ValueError("No matching edges to remove")

    original = openfe.LigandNetwork.from_graphml(graphml.read_text(encoding="utf-8"))
    expected = {component.name for component in original.nodes}
    _validate_connected([NetworkEdge(r["ligand_a"], r["ligand_b"]) for r in kept], expected)

    keep_pairs = {frozenset((r["ligand_a"], r["ligand_b"])) for r in kept}
    mapped = [edge for edge in original.edges
              if frozenset((edge.componentA.name, edge.componentB.name)) in keep_pairs]
    edited = openfe.LigandNetwork(edges=mapped, nodes=original.nodes)
    graphml.write_text(edited.to_graphml(), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    return removed
