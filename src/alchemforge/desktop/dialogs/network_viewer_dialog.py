"""Visualize a planned ligand perturbation network in the desktop."""

from __future__ import annotations

import csv
import html
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QDir, Qt, QTemporaryDir, QUrl, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from alchemforge.desktop.modeless import show_modeless
from alchemforge.gromacs.rbfe.mapping import add_network_edge


@dataclass(frozen=True)
class NetworkEdge:
    """One planned alchemical transformation."""

    ligand_a: str
    ligand_b: str
    mapped_atoms: str = ""
    score: str = ""


def read_network_edges(path: str | Path) -> list[NetworkEdge]:
    """Read the edge table emitted by the GROMACS network planner."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Network edge table does not exist: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ligand_a", "ligand_b"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Network edge table must contain ligand_a and ligand_b")
    return [
        NetworkEdge(
            row["ligand_a"].strip(),
            row["ligand_b"].strip(),
            row.get("mapped_atoms", "").strip(),
            row.get("lomap_score", "").strip(),
        )
        for row in rows
    ]


def _read_edge_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _validate_connected(edges: list[NetworkEdge], expected_names: set[str]) -> None:
    if not edges:
        raise ValueError("Keep at least one transformation")
    adjacency = {name: set() for name in expected_names}
    for edge in edges:
        adjacency[edge.ligand_a].add(edge.ligand_b)
        adjacency[edge.ligand_b].add(edge.ligand_a)
    reached = {next(iter(expected_names))}
    queue = deque(reached)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            queue.append(neighbor)
    missing = expected_names - reached
    if missing:
        raise ValueError(
            "The edited network is disconnected. Restore an edge connecting: "
            + ", ".join(sorted(missing))
        )


def write_edited_network(
    path: str | Path, enabled_rows: set[int], expected_names: set[str]
) -> int:
    """Persist an enabled subset while retaining mapping IDs and metadata."""
    destination = Path(path).expanduser().resolve()
    fields, rows = _read_edge_rows(destination)
    kept_rows = [row for index, row in enumerate(rows) if index in enabled_rows]
    edges = [NetworkEdge(row["ligand_a"], row["ligand_b"]) for row in kept_rows]
    _validate_connected(edges, expected_names)
    graphml = destination.with_name("ligand_network.graphml")
    if graphml.is_file():
        import openfe

        original = openfe.LigandNetwork.from_graphml(
            graphml.read_text(encoding="utf-8")
        )
        enabled_pairs = {
            frozenset((row["ligand_a"], row["ligand_b"])) for row in kept_rows
        }
        mapped_edges = [
            edge
            for edge in original.edges
            if frozenset((edge.componentA.name, edge.componentB.name)) in enabled_pairs
        ]
        if len(mapped_edges) != len(kept_rows):
            raise ValueError(
                "The edited edge table does not match the atom mappings in GraphML"
            )
        edited = openfe.LigandNetwork(edges=mapped_edges, nodes=original.nodes)
        graphml_text = edited.to_graphml()
    else:
        node_ids = {
            name: f"n{index}" for index, name in enumerate(sorted(expected_names))
        }
        graphml_text = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
            '  <key id="name" for="node" attr.name="name" attr.type="string"/>\n'
            '  <graph id="ligand_network" edgedefault="undirected">\n'
            + "".join(
                f'    <node id="{node_ids[name]}"><data key="name">'
                f"{html.escape(name)}</data></node>\n"
                for name in sorted(expected_names)
            )
            + "".join(
                f'    <edge id="{html.escape(row.get("edge_id", ""))}" '
                f'source="{node_ids[row["ligand_a"]]}" '
                f'target="{node_ids[row["ligand_b"]]}"/>\n'
                for row in kept_rows
            )
            + "  </graph>\n</graphml>\n"
        )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept_rows)
    graphml.write_text(graphml_text, encoding="utf-8")
    manifest = destination.with_name("network_manifest.json")
    if manifest.is_file():
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_data["edge_count"] = len(kept_rows)
        manifest_data["edited"] = True
        manifest.write_text(
            json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8"
        )
    return len(kept_rows)


def _net_charge_label(molecule) -> tuple[str, str]:
    """Return a transparent partial-charge sum or an explicit formal fallback."""
    from rdkit import Chem

    for property_name in (
        "PartialCharge",
        "_TriposPartialCharge",
        "partial_charge",
        "_GasteigerCharge",
    ):
        if molecule.GetNumAtoms() and all(
            atom.HasProp(property_name) for atom in molecule.GetAtoms()
        ):
            try:
                values = [
                    float(atom.GetProp(property_name)) for atom in molecule.GetAtoms()
                ]
            except ValueError:
                continue
            if all(math.isfinite(value) for value in values):
                total = sum(values)
                return f"PC {total:+.3f} e", property_name
    formal = Chem.GetFormalCharge(molecule)
    return f"PC n/a · FC {formal:+d} e", "formal charge fallback"


def _ligand_depictions(path: str | Path) -> dict[str, tuple[str, str, str]]:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Ligand SDF does not exist: {source}")
    depictions = {}
    for index, molecule in enumerate(
        Chem.SDMolSupplier(str(source), removeHs=False), start=1
    ):
        if molecule is None:
            continue
        name = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
        name = name or f"record {index}"
        drawing_molecule = Chem.RemoveHs(molecule)
        rdDepictor.Compute2DCoords(drawing_molecule)
        drawer = rdMolDraw2D.MolDraw2DSVG(170, 105)
        drawer.drawOptions().clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, drawing_molecule)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg[svg.find("<svg") :]
        svg = svg.replace("<svg", '<svg class="molecule"', 1)
        charge_label, charge_source = _net_charge_label(molecule)
        depictions[name] = (svg, charge_label, charge_source)
    return depictions


def _ligands_by_name(path: Path):
    from rdkit import Chem

    ligands = {}
    for index, molecule in enumerate(
        Chem.SDMolSupplier(str(path), removeHs=False), start=1
    ):
        if molecule is None:
            continue
        name = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
        ligands[name or f"record {index}"] = molecule
    return ligands


def _mapped_molecule_svg(
    molecule, atom_colors: dict[int, tuple[float, float, float]]
) -> str:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    drawing_molecule = Chem.Mol(molecule)
    rdDepictor.Compute2DCoords(drawing_molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 360)
    drawer.drawOptions().addAtomIndices = True
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        drawing_molecule,
        highlightAtoms=sorted(atom_colors),
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.find("<svg") :]


def render_atom_mapping_html(
    edges_csv: str | Path,
    ligand_sdf: str | Path,
    selected_a: str,
    selected_b: str,
) -> str:
    """Render the saved atom mapping for one directly connected ligand pair."""
    edge_path = Path(edges_csv).expanduser().resolve()
    _fields, rows = _read_edge_rows(edge_path)
    row = next(
        (
            candidate
            for candidate in rows
            if {candidate["ligand_a"], candidate["ligand_b"]}
            == {selected_a, selected_b}
        ),
        None,
    )
    if row is None:
        raise ValueError(
            f"{selected_a!r} and {selected_b!r} are not directly connected "
            "in the enabled network"
        )
    mapping_path = edge_path.parent / "mappings" / f"{row['edge_id']}.json"
    if not mapping_path.is_file():
        raise ValueError(f"Atom-mapping file does not exist: {mapping_path}")
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    raw_mapping = payload.get("componentA_to_componentB")
    if not raw_mapping:
        pairs_path = Path(payload.get("pairs1", ""))
        if pairs_path.is_file():
            pairs = [
                line.split()[:2]
                for line in pairs_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith(("#", ";"))
            ]
            # PMX pair files use one-based RDKit atom indices.
            mapping = {int(left) - 1: int(right) - 1 for left, right in pairs}
        else:
            status = payload.get("mapping_status", "unavailable")
            raise ValueError(
                f"The atom mapping for {row['edge_id']} is {status}. If PMX was "
                "staged, run run_pmx_mappings.sh before inspecting the final mapping."
            )
    else:
        mapping = {int(left): int(right) for left, right in raw_mapping.items()}
    if not mapping:
        raise ValueError(
            f"The atom mapping for {row['edge_id']} contains no atom pairs"
        )
    ligands = _ligands_by_name(Path(ligand_sdf).expanduser().resolve())
    name_a, name_b = row["ligand_a"], row["ligand_b"]
    if name_a not in ligands or name_b not in ligands:
        raise ValueError("Mapped ligand names could not be found in the ligand SDF")
    palette = [
        (0.90, 0.22, 0.21),
        (0.12, 0.47, 0.71),
        (0.20, 0.63, 0.17),
        (1.00, 0.50, 0.05),
        (0.58, 0.40, 0.74),
        (0.55, 0.34, 0.29),
        (0.89, 0.47, 0.76),
        (0.50, 0.50, 0.50),
        (0.74, 0.74, 0.13),
        (0.09, 0.75, 0.81),
    ]
    colors_a = {
        atom_a: palette[index % len(palette)] for index, atom_a in enumerate(mapping)
    }
    colors_b = {
        atom_b: palette[index % len(palette)]
        for index, atom_b in enumerate(mapping.values())
    }
    svg_a = _mapped_molecule_svg(ligands[name_a], colors_a)
    svg_b = _mapped_molecule_svg(ligands[name_b], colors_b)
    pairs = " ".join(
        f'<span class="pair">{atom_a} ↔ {atom_b}</span>'
        for atom_a, atom_b in mapping.items()
    )
    score = html.escape(row.get("lomap_score", ""))
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;padding:18px;background:#f4f8fc;color:#254b6b;font-family:-apple-system,
BlinkMacSystemFont,'Segoe UI',sans-serif}}h2{{margin:0 0 5px;color:#174f80}}
.meta{{color:#6d8193;margin-bottom:14px}}.structures{{display:grid;
grid-template-columns:1fr 1fr;gap:16px}}.ligand{{background:white;border:1px solid #bed8ec;
border-radius:12px;padding:10px;text-align:center}}.ligand h3{{margin:2px 0 8px}}
.ligand svg{{width:100%;height:auto;max-height:430px}}.pairs{{margin-top:15px;line-height:2}}
.pair{{display:inline-block;background:white;border:1px solid #c7dced;border-radius:5px;
padding:1px 7px;margin:2px;font-family:monospace}}</style></head><body>
<h2>{html.escape(row["edge_id"])}: atom mapping</h2>
<div class="meta">{len(mapping)} mapped atoms · LoMap score {score}. Atom indices are shown.</div>
<div class="structures"><div class="ligand"><h3>{html.escape(name_a)}</h3>{svg_a}</div>
<div class="ligand"><h3>{html.escape(name_b)}</h3>{svg_b}</div></div>
<div class="pairs">{pairs}</div></body></html>"""


def _network_layout(
    names: list[str], edges: list[NetworkEdge]
) -> dict[str, tuple[float, float]]:
    """Create a deterministic force-directed layout without extra dependencies."""
    count = len(names)
    radius = max(260.0, count * 48.0)
    positions = {
        name: [
            radius * math.cos(2.0 * math.pi * index / count),
            radius * math.sin(2.0 * math.pi * index / count),
        ]
        for index, name in enumerate(names)
    }
    if count < 3:
        return {name: tuple(position) for name, position in positions.items()}
    area = (2.0 * radius) ** 2
    ideal = math.sqrt(area / count)
    for iteration in range(180):
        displacement = {name: [0.0, 0.0] for name in names}
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                dx = positions[left][0] - positions[right][0]
                dy = positions[left][1] - positions[right][1]
                distance = max(math.hypot(dx, dy), 1.0)
                force = ideal * ideal / distance
                fx, fy = dx / distance * force, dy / distance * force
                displacement[left][0] += fx
                displacement[left][1] += fy
                displacement[right][0] -= fx
                displacement[right][1] -= fy
        for edge in edges:
            dx = positions[edge.ligand_a][0] - positions[edge.ligand_b][0]
            dy = positions[edge.ligand_a][1] - positions[edge.ligand_b][1]
            distance = max(math.hypot(dx, dy), 1.0)
            force = distance * distance / ideal
            fx, fy = dx / distance * force, dy / distance * force
            displacement[edge.ligand_a][0] -= fx
            displacement[edge.ligand_a][1] -= fy
            displacement[edge.ligand_b][0] += fx
            displacement[edge.ligand_b][1] += fy
        temperature = radius * 0.08 * (1.0 - iteration / 180.0)
        for name in names:
            length = max(math.hypot(*displacement[name]), 1.0)
            positions[name][0] += (
                displacement[name][0] / length * min(length, temperature)
            )
            positions[name][1] += (
                displacement[name][1] / length * min(length, temperature)
            )
    return {name: tuple(position) for name, position in positions.items()}


def render_network_html(edges_csv: str | Path, ligand_sdf: str | Path) -> str:
    """Render an interactive, self-contained HTML network visualization."""
    edges = read_network_edges(edges_csv)
    names = sorted({name for edge in edges for name in (edge.ligand_a, edge.ligand_b)})
    if len(names) < 2:
        raise ValueError("A ligand network must contain at least two ligands")
    depictions = _ligand_depictions(ligand_sdf)
    positions = _network_layout(names, edges)
    card_width, card_height, margin = 190.0, 168.0, 170.0
    min_x = min(value[0] for value in positions.values()) - card_width / 2 - margin
    max_x = max(value[0] for value in positions.values()) + card_width / 2 + margin
    min_y = min(value[1] for value in positions.values()) - card_height / 2 - margin
    max_y = max(value[1] for value in positions.values()) + card_height / 2 + margin

    edge_markup = []
    for edge in edges:
        x1, y1 = positions[edge.ligand_a]
        x2, y2 = positions[edge.ligand_b]
        details = []
        if edge.score:
            details.append(f"score {edge.score}")
        if edge.mapped_atoms:
            details.append(f"{edge.mapped_atoms} mapped atoms")
        label = " · ".join(details)
        edge_markup.append(
            f'<g class="edge" data-a="{html.escape(edge.ligand_a, quote=True)}" '
            f'data-b="{html.escape(edge.ligand_b, quote=True)}">'
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}"/><title>'
            f"{html.escape(edge.ligand_a)} → {html.escape(edge.ligand_b)}"
            f"{html.escape(' · ' + label) if label else ''}</title>"
            f'<text x="{(x1 + x2) / 2:.2f}" y="{(y1 + y2) / 2 - 7:.2f}">'
            f"{html.escape(edge.score)}</text></g>"
        )

    node_markup = []
    for name in names:
        x, y = positions[name]
        depiction, charge_label, charge_source = depictions.get(
            name,
            (
                "",
                "PC unavailable",
                "ligand record was not found in the SDF",
            ),
        )
        content = (
            depiction
            if depiction
            else '<div class="missing">No matching SDF record</div>'
        )
        node_markup.append(
            f'<g class="node" data-name="{html.escape(name, quote=True)}" '
            f'data-x="{x:.2f}" data-y="{y:.2f}" '
            f'data-original-x="{x:.2f}" data-original-y="{y:.2f}" '
            f'transform="translate({x - card_width / 2:.2f},'
            f'{y - card_height / 2:.2f})"><foreignObject width="{card_width}" '
            f'height="{card_height}"><div xmlns="http://www.w3.org/1999/xhtml" '
            f'class="card"><div class="name">{html.escape(name)}</div>'
            f'<div class="charge" title="Charge source: '
            f'{html.escape(charge_source, quote=True)}">'
            f"{html.escape(charge_label)}</div>{content}"
            "</div></foreignObject></g>"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#f4f8fc;
font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
#canvas{{width:100%;height:100%;cursor:grab;user-select:none}}
#canvas.dragging{{cursor:grabbing}}
.edge line{{stroke:#80a9ce;stroke-width:3.5}}
.edge text{{font-size:12px;fill:#4e7294;text-anchor:middle;paint-order:stroke;
stroke:#f4f8fc;stroke-width:5px;stroke-linejoin:round}}
.card{{box-sizing:border-box;width:{card_width}px;height:{card_height}px;
background:white;border:2px solid #6ba5d8;border-radius:12px;overflow:hidden;
box-shadow:0 3px 10px rgba(30,75,115,.18)}}
.node{{cursor:move}}.node.selected .card{{border:4px solid #e64b3c;
box-shadow:0 0 0 4px rgba(230,75,60,.18)}}
.name{{height:28px;line-height:28px;text-align:center;font-size:14px;font-weight:700;
color:#174f80;background:#e9f4fd;border-bottom:1px solid #c5deef;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap;padding:0 7px}}
.charge{{height:21px;line-height:21px;text-align:center;font-size:11px;font-weight:600;
color:#315f7d;background:#f4f9fd;border-bottom:1px solid #dbeaf5;white-space:nowrap}}
.molecule{{display:block;width:170px!important;height:105px!important;margin:2px auto}}
.missing{{font-size:12px;color:#8a99a8;text-align:center;padding-top:42px}}
</style></head><body>
<svg id="canvas" viewBox="{min_x:.2f} {min_y:.2f} {max_x - min_x:.2f} {max_y - min_y:.2f}">
<g id="scene">{"".join(edge_markup)}{"".join(node_markup)}</g></svg>
<script>
const svg=document.getElementById('canvas');
const initialBox={{x:{min_x:.2f},y:{min_y:.2f},w:{max_x - min_x:.2f},h:{max_y - min_y:.2f}}};
let box={{...initialBox}},drag=false,last=null,nodeDrag=null;
function apply(){{svg.setAttribute('viewBox',`${{box.x}} ${{box.y}} ${{box.w}} ${{box.h}}`);}}
function zoomAt(factor,clientX=null,clientY=null){{
  const r=svg.getBoundingClientRect(),scale=1/factor;
  const nextW=Math.min(initialBox.w*8,Math.max(initialBox.w*.08,box.w*scale));
  const actualScale=nextW/box.w,nextH=box.h*actualScale;
  const px=clientX===null?box.x+box.w/2:box.x+(clientX-r.left)/r.width*box.w;
  const py=clientY===null?box.y+box.h/2:box.y+(clientY-r.top)/r.height*box.h;
  box.x=px-(px-box.x)*actualScale;box.y=py-(py-box.y)*actualScale;
  box.w=nextW;box.h=nextH;apply();
}}
window.zoomNetwork=factor=>zoomAt(factor);
window.resetNetworkZoom=()=>{{box={{...initialBox}};apply();}};
function nodeByName(name){{return [...document.querySelectorAll('.node')].find(n=>n.dataset.name===name);}}
function placeNode(node,x,y){{node.dataset.x=x;node.dataset.y=y;
  node.setAttribute('transform',`translate(${{x-{card_width / 2}}},${{y-{card_height / 2}}})`);}}
function updateEdges(){{document.querySelectorAll('.edge').forEach(edge=>{{
  const a=nodeByName(edge.dataset.a),b=nodeByName(edge.dataset.b);if(!a||!b)return;
  const x1=+a.dataset.x,y1=+a.dataset.y,x2=+b.dataset.x,y2=+b.dataset.y;
  const line=edge.querySelector('line'),label=edge.querySelector('text');
  line.setAttribute('x1',x1);line.setAttribute('y1',y1);line.setAttribute('x2',x2);line.setAttribute('y2',y2);
  label.setAttribute('x',(x1+x2)/2);label.setAttribute('y',(y1+y2)/2-7);
}});}}
svg.addEventListener('wheel',e=>{{e.preventDefault();
  zoomAt(e.deltaY>0?1/1.12:1/.89,e.clientX,e.clientY);
}},{{passive:false}});
svg.addEventListener('mousedown',e=>{{const node=e.target.closest?.('.node');last=[e.clientX,e.clientY];
  if(node){{nodeDrag={{node:node,x:+node.dataset.x,y:+node.dataset.y,start:[e.clientX,e.clientY],moved:false}};return;}}
  drag=true;svg.classList.add('dragging');}});
window.addEventListener('mouseup',()=>{{
  if(nodeDrag&&!nodeDrag.moved)toggleNodeSelection(nodeDrag.node);
  nodeDrag=null;drag=false;svg.classList.remove('dragging');}});
window.addEventListener('mousemove',e=>{{const r=svg.getBoundingClientRect();
  if(nodeDrag){{const dx=(e.clientX-nodeDrag.start[0])/r.width*box.w,dy=(e.clientY-nodeDrag.start[1])/r.height*box.h;
    if(Math.hypot(e.clientX-nodeDrag.start[0],e.clientY-nodeDrag.start[1])>3)nodeDrag.moved=true;
    placeNode(nodeDrag.node,nodeDrag.x+dx,nodeDrag.y+dy);updateEdges();return;}}
  if(!drag)return;box.x-=(e.clientX-last[0])/r.width*box.w;box.y-=(e.clientY-last[1])/r.height*box.h;
  last=[e.clientX,e.clientY];apply();}});
window.selectedLigands=[];
function toggleNodeSelection(node){{
  const name=node.dataset.name,index=window.selectedLigands.indexOf(name);
  if(index>=0)window.selectedLigands.splice(index,1);else{{
    if(window.selectedLigands.length>=2){{const old=window.selectedLigands.shift();
      [...document.querySelectorAll('.node')].find(n=>n.dataset.name===old)?.classList.remove('selected');}}
    window.selectedLigands.push(name);
  }}
  node.classList.toggle('selected',window.selectedLigands.includes(name));
}}
window.resetNetworkLayout=()=>{{document.querySelectorAll('.node').forEach(node=>
  placeNode(node,+node.dataset.originalX,+node.dataset.originalY));updateEdges();box={{...initialBox}};apply();}};
</script></body></html>"""


class LigandNetworkViewerDialog(QDialog):
    """Interactive viewer for a planned ligand transformation network."""

    network_saved = pyqtSignal(str)

    def __init__(
        self,
        edges_csv: str | Path,
        ligand_sdf: str | Path,
        parent=None,
    ):
        super().__init__(parent)
        self.edges_csv = Path(edges_csv).expanduser().resolve()
        self.ligand_sdf = Path(ligand_sdf).expanduser().resolve()
        edges = read_network_edges(self.edges_csv)
        ligand_count = len(
            {name for edge in edges for name in (edge.ligand_a, edge.ligand_b)}
        )
        self.setWindowTitle(f"AlchemForge Ligand Network — {self.edges_csv.parent.name}")
        self.resize(1150, 800)
        self._all_names = {
            name for edge in edges for name in (edge.ligand_a, edge.ligand_b)
        }
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        summary = QLabel(f"{ligand_count} ligands · {len(edges)} transformations")
        summary.setStyleSheet("font-size: 13pt; font-weight: 700; color: #174f80;")
        header.addWidget(summary)
        header.addStretch()
        zoom_out = QPushButton("−")
        zoom_out.setToolTip("Zoom out")
        zoom_out.clicked.connect(lambda: self._zoom(0.85))
        zoom_in = QPushButton("+")
        zoom_in.setToolTip("Zoom in")
        zoom_in.clicked.connect(lambda: self._zoom(1.15))
        reset = QPushButton("Reset Zoom")
        reset.clicked.connect(
            lambda: self.web.page().runJavaScript("window.resetNetworkZoom?.()")
        )
        reset_layout = QPushButton("Reset Layout")
        reset_layout.clicked.connect(
            lambda: self.web.page().runJavaScript("window.resetNetworkLayout?.()")
        )
        show_mapping = QPushButton("Show Selected Mapping")
        show_mapping.setToolTip(
            "Click two ligand nodes, then inspect their atom mapping"
        )
        show_mapping.clicked.connect(self._show_selected_mapping)
        add_edge = QPushButton("Add Selected Edge")
        add_edge.setToolTip(
            "Click two unconnected ligand nodes, then generate and add their mapping"
        )
        add_edge.clicked.connect(self._add_selected_edge)
        remove_edge = QPushButton("Remove Selected Edge")
        remove_edge.setToolTip(
            "Click two connected ligand nodes, then remove their transformation"
        )
        remove_edge.clicked.connect(self._remove_selected_edge)
        header.addWidget(add_edge)
        header.addWidget(remove_edge)
        header.addWidget(show_mapping)
        header.addWidget(zoom_out)
        header.addWidget(zoom_in)
        header.addWidget(reset)
        header.addWidget(reset_layout)
        layout.addLayout(header)
        bundle_path = QLabel(f"Bundle: {self.edges_csv.parent}")
        bundle_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bundle_path.setToolTip(str(self.edges_csv.parent))
        bundle_path.setStyleSheet("color:#526b82")
        layout.addWidget(bundle_path)
        hint = QLabel(
            "Drag a ligand card to move it; drag the background to pan. Use the mouse "
            "wheel to zoom and hover over an edge for details. Edge labels are LoMap scores."
        )
        hint.setStyleSheet("color:#71839a")
        layout.addWidget(hint)
        splitter = QSplitter(Qt.Horizontal)
        self.web = QWebEngineView()
        splitter.addWidget(self.web)
        editor = self._build_edge_editor(edges)
        splitter.addWidget(editor)
        splitter.setSizes([850, 300])
        splitter.setStretchFactor(0, 1)
        layout.addWidget(splitter, 1)
        temporary_template = str(
            Path(QDir.tempPath()) / "alchemforge_network_XXXXXX"
        )
        self._html_directory = QTemporaryDir(temporary_template)
        if not self._html_directory.isValid():
            raise RuntimeError("Could not create a temporary network-view directory")
        self._scene_generation = 0
        self._load_network_scene()

    def _cleanup_html_directory(self) -> None:
        """Release the web page and remove its generated HTML directory."""

        directory = getattr(self, "_html_directory", None)
        if directory is None:
            return

        # The parent dialogs retain the viewer Python object after it closes,
        # so relying on QTemporaryDir's destructor leaves alchemforge_network_*
        # directories behind for the rest of the desktop session.
        self.web.stop()
        self.web.setUrl(QUrl("about:blank"))
        self._html_directory = None
        directory.remove()

    def done(self, result: int) -> None:
        """Clean generated network HTML for Accept, Reject, and window close."""

        self._cleanup_html_directory()
        super().done(result)

    def _load_network_scene(self) -> None:
        self._scene_generation += 1
        destination = (
            Path(self._html_directory.path())
            / f"network_{self._scene_generation:04d}.html"
        )
        destination.write_text(
            render_network_html(self.edges_csv, self.ligand_sdf), encoding="utf-8"
        )
        self.web.load(QUrl.fromLocalFile(str(destination)))

    def _add_selected_edge(self) -> None:
        self.web.page().runJavaScript(
            "window.selectedLigands || []", self._add_edge_for_selection
        )

    def _add_edge_for_selection(self, selected) -> None:
        if not isinstance(selected, list) or len(selected) != 2:
            QMessageBox.information(
                self,
                "Select Two Ligands",
                "Click two unconnected ligand nodes, then choose Add Selected Edge.",
            )
            return
        try:
            row = add_network_edge(self.edges_csv.parent, selected[0], selected[1])
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Add Network Edge", str(error))
            return
        table_row = self.edge_table.rowCount()
        self.edge_table.insertRow(table_row)
        enabled = QTableWidgetItem()
        enabled.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        enabled.setCheckState(Qt.Checked)
        self.edge_table.setItem(table_row, 0, enabled)
        self.edge_table.setItem(table_row, 1, QTableWidgetItem(row["ligand_a"]))
        self.edge_table.setItem(table_row, 2, QTableWidgetItem(row["ligand_b"]))
        self.edge_table.setItem(table_row, 3, QTableWidgetItem(row["lomap_score"]))
        self.edge_table.resizeColumnsToContents()
        self._load_network_scene()
        self.network_saved.emit(str(self.edges_csv))
        mapping_note = (
            " PMX mapping was staged and must be run before inspection."
            if row["mapped_atoms"] == "pending"
            else f" {row['mapped_atoms']} atoms were mapped."
        )
        QMessageBox.information(
            self,
            "Network Edge Added",
            f"Added {row['ligand_a']} ↔ {row['ligand_b']}." + mapping_note,
        )

    def _remove_selected_edge(self) -> None:
        self.web.page().runJavaScript(
            "window.selectedLigands || []", self._remove_edge_for_selection
        )

    def _remove_edge_for_selection(self, selected) -> None:
        if not isinstance(selected, list) or len(selected) != 2:
            QMessageBox.information(
                self,
                "Select Two Ligands",
                "Click the two nodes joined by the edge you want to remove.",
            )
            return
        edges = read_network_edges(self.edges_csv)
        selected_pair = frozenset(selected)
        matching_rows = [
            index
            for index, edge in enumerate(edges)
            if frozenset((edge.ligand_a, edge.ligand_b)) == selected_pair
        ]
        if not matching_rows:
            QMessageBox.information(
                self,
                "No Selected Edge",
                f"{selected[0]} and {selected[1]} are not directly connected.",
            )
            return
        row = matching_rows[0]
        answer = QMessageBox.question(
            self,
            "Remove Network Edge",
            f"Remove the transformation {selected[0]} ↔ {selected[1]}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        enabled_rows = set(range(len(edges))) - {row}
        try:
            count = write_edited_network(
                self.edges_csv, enabled_rows, self._all_names
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Remove Network Edge", str(error))
            return
        self.edge_table.removeRow(row)
        self._load_network_scene()
        self.network_saved.emit(str(self.edges_csv))
        QMessageBox.information(
            self,
            "Network Edge Removed",
            f"Removed {selected[0]} ↔ {selected[1]}. "
            f"The network now contains {count} transformations.",
        )

    def _show_selected_mapping(self) -> None:
        self.web.page().runJavaScript(
            "window.selectedLigands || []", self._open_mapping_for_selection
        )

    def _open_mapping_for_selection(self, selected) -> None:
        if not isinstance(selected, list) or len(selected) != 2:
            QMessageBox.information(
                self,
                "Select Two Ligands",
                "Click two ligand nodes in the network, then choose Show Selected Mapping.",
            )
            return
        try:
            mapping_html = render_atom_mapping_html(
                self.edges_csv, self.ligand_sdf, selected[0], selected[1]
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "Cannot Show Atom Mapping", str(error))
            return
        self._mapping_dialog = AtomMappingDialog(
            mapping_html, selected[0], selected[1], self
        )
        show_modeless(self, self._mapping_dialog)

    def _build_edge_editor(self, edges: list[NetworkEdge]):
        from PyQt5.QtWidgets import QWidget

        panel = QWidget()
        panel.setMinimumWidth(280)
        layout = QVBoxLayout(panel)
        title = QLabel("Edit Transformations")
        title.setStyleSheet("font-size: 12pt; font-weight: 700; color: #174f80")
        layout.addWidget(title)
        note = QLabel(
            "Uncheck unwanted planned edges, then save. The network must remain "
            "connected. To add a transformation, select two unconnected ligand "
            "cards in the graph and choose Add Selected Edge."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#71839a")
        layout.addWidget(note)
        self.edge_table = QTableWidget(len(edges), 4)
        self.edge_table.setHorizontalHeaderLabels(
            ["Use", "Ligand A", "Ligand B", "Score"]
        )
        self.edge_table.verticalHeader().setVisible(False)
        for row, edge in enumerate(edges):
            enabled = QTableWidgetItem()
            enabled.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            enabled.setCheckState(Qt.Checked)
            self.edge_table.setItem(row, 0, enabled)
            self.edge_table.setItem(row, 1, QTableWidgetItem(edge.ligand_a))
            self.edge_table.setItem(row, 2, QTableWidgetItem(edge.ligand_b))
            self.edge_table.setItem(row, 3, QTableWidgetItem(edge.score))
        self.edge_table.resizeColumnsToContents()
        layout.addWidget(self.edge_table, 1)
        save = QPushButton("Save Edited Network")
        save.clicked.connect(self._save_edits)
        layout.addWidget(save)
        return panel

    def _save_edits(self) -> None:
        enabled = {
            row
            for row in range(self.edge_table.rowCount())
            if self.edge_table.item(row, 0).checkState() == Qt.Checked
        }
        try:
            count = write_edited_network(self.edges_csv, enabled, self._all_names)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Save Network", str(error))
            return
        self.network_saved.emit(str(self.edges_csv))
        QMessageBox.information(
            self,
            "Network Saved",
            f"Saved {count} enabled transformations. Existing atom-mapping IDs "
            "were preserved.",
        )
        self.accept()

    def _zoom(self, multiplier: float) -> None:
        self.web.page().runJavaScript(f"window.zoomNetwork?.({multiplier})")


class AtomMappingDialog(QDialog):
    """Display corresponding atom highlights for one planned network edge."""

    def __init__(self, mapping_html: str, ligand_a: str, ligand_b: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Atom Mapping: {ligand_a} ↔ {ligand_b}")
        self.resize(980, 720)
        layout = QVBoxLayout(self)
        view = QWebEngineView()
        view.setHtml(mapping_html)
        layout.addWidget(view)
