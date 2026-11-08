"""Qt-free reading/validation of an OpenFE/GROMACS ligand perturbation network.

A planned network is a *bundle* directory containing ``ligand_network.graphml``
and ``edges.csv`` (ligand_a, ligand_b, ...). This module reads the edges (from the
CSV or straight from the GraphML), checks connectivity, and computes the pairs to
keep/drop — the data layer for the desktop network editor. Mutating the network
(which re-maps atoms and rewrites the GraphML) needs OpenFE and is done through
the ``alchemforge network`` CLI; nothing here imports OpenFE or Qt.
"""

from __future__ import annotations

import csv
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NetworkEdge:
    ligand_a: str
    ligand_b: str
    score: str = ""


def read_edges_csv(path: str | Path) -> list[NetworkEdge]:
    """Read ``edges.csv`` (needs ligand_a + ligand_b columns)."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"edge table not found: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"ligand_a", "ligand_b"}.issubset(rows[0]):
        raise ValueError("edge table must contain ligand_a and ligand_b columns")
    return [
        NetworkEdge(r["ligand_a"].strip(), r["ligand_b"].strip(),
                    (r.get("lomap_score") or r.get("score") or "").strip())
        for r in rows
    ]


def parse_graphml_edges(text: str) -> list[NetworkEdge]:
    """Read edges from a ligand_network GraphML (maps node ids to ligand names)."""
    # node id -> name (name is a <data key="...">NAME</data> under each <node>)
    names: dict[str, str] = {}
    for node in re.finditer(r'<node\s+id="([^"]+)"\s*>(.*?)</node>', text, re.DOTALL):
        node_id, body = node.group(1), node.group(2)
        data = re.search(r'<data[^>]*>(.*?)</data>', body, re.DOTALL)
        names[node_id] = (data.group(1).strip() if data else node_id)
    edges: list[NetworkEdge] = []
    for edge in re.finditer(r'<edge\b[^>]*\bsource="([^"]+)"[^>]*\btarget="([^"]+)"', text):
        a, b = edge.group(1), edge.group(2)
        edges.append(NetworkEdge(names.get(a, a), names.get(b, b)))
    return edges


def read_network(bundle_dir: str | Path) -> list[NetworkEdge]:
    """Read edges from a bundle directory: prefer edges.csv, fall back to GraphML."""
    bundle = Path(bundle_dir).expanduser()
    csv_path = bundle / "edges.csv"
    if csv_path.is_file():
        return read_edges_csv(csv_path)
    graphml = bundle / "ligand_network.graphml"
    if graphml.is_file():
        return parse_graphml_edges(graphml.read_text(encoding="utf-8", errors="replace"))
    raise FileNotFoundError(f"no edges.csv or ligand_network.graphml in {bundle}")


def ligand_names(edges: list[NetworkEdge]) -> list[str]:
    """Sorted unique ligand names appearing in the edges."""
    names = {e.ligand_a for e in edges} | {e.ligand_b for e in edges}
    return sorted(names)


def connectivity_issue(edges: list[NetworkEdge], expected_names: set[str] | None = None) -> str | None:
    """Return a message if the network is empty/disconnected, else None."""
    if not edges:
        return "Network has no edges — keep at least one transformation."
    names = set(expected_names) if expected_names else {
        n for e in edges for n in (e.ligand_a, e.ligand_b)}
    adjacency: dict[str, set[str]] = {n: set() for n in names}
    for e in edges:
        adjacency.setdefault(e.ligand_a, set()).add(e.ligand_b)
        adjacency.setdefault(e.ligand_b, set()).add(e.ligand_a)
    start = next(iter(names))
    reached = {start}
    queue = deque([start])
    while queue:
        for nb in adjacency.get(queue.popleft(), set()) - reached:
            reached.add(nb)
            queue.append(nb)
    missing = names - reached
    if missing:
        return "Network is disconnected; unreachable: " + ", ".join(sorted(missing))
    return None


def edge_key(a: str, b: str) -> str:
    """Canonical, order-independent key for an edge pair (for CLI --drop)."""
    return ":".join(sorted((a, b)))


def would_disconnect(edges: list[NetworkEdge], drop_index: int) -> bool:
    """True if removing ``edges[drop_index]`` leaves the network disconnected."""
    kept = [e for i, e in enumerate(edges) if i != drop_index]
    names = {n for e in edges for n in (e.ligand_a, e.ligand_b)}
    return connectivity_issue(kept, names) is not None
