"""RDKit-only rendering of a ligand network and its per-edge atom mappings.

No OpenFE or Qt imports, so the ChemFlow desktop (which has RDKit) can render a
planned network bundle in-process. Reads ``edges.csv`` (edge_id, ligand_a,
ligand_b, lomap_score), ``ligands.sdf`` and ``mappings/<edge_id>.json``
(``componentA_to_componentB``) written by ``plan_network`` / ``add_network_edge``.

* :func:`render_network_html` — an interactive SVG: nodes = ligands, edges are
  clickable and call ``window.bridge.on_edge_clicked(a, b)`` (host injects the
  bridge, like the linker designer).
* :func:`render_atom_mapping_html` — the two mapped ligands side by side with the
  corresponding atoms coloured and indexed.
"""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

from chemflow.fe.openfe.utils.network_edit import NetworkEdge, read_network_edges

_PALETTE = [
    (0.90, 0.22, 0.21), (0.12, 0.47, 0.71), (0.20, 0.63, 0.17), (1.00, 0.50, 0.05),
    (0.58, 0.40, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13), (0.09, 0.75, 0.81),
]


def _read_edge_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _ligands_by_name(path: Path):
    from rdkit import Chem

    ligands = {}
    for index, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False), start=1):
        if mol is None:
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        ligands[name or f"record {index}"] = mol
    return ligands


def _network_layout(names, edges):
    """Deterministic force-directed layout (no extra dependencies)."""
    count = len(names)
    radius = max(260.0, count * 48.0)
    pos = {name: [radius * math.cos(2 * math.pi * i / count),
                  radius * math.sin(2 * math.pi * i / count)]
           for i, name in enumerate(names)}
    if count < 3:
        return {n: tuple(p) for n, p in pos.items()}
    ideal = math.sqrt((2 * radius) ** 2 / count)
    for it in range(180):
        disp = {n: [0.0, 0.0] for n in names}
        for li, left in enumerate(names):
            for right in names[li + 1:]:
                dx, dy = pos[left][0] - pos[right][0], pos[left][1] - pos[right][1]
                dist = max(math.hypot(dx, dy), 1.0)
                f = ideal * ideal / dist
                disp[left][0] += dx / dist * f; disp[left][1] += dy / dist * f
                disp[right][0] -= dx / dist * f; disp[right][1] -= dy / dist * f
        for e in edges:
            dx, dy = pos[e.ligand_a][0] - pos[e.ligand_b][0], pos[e.ligand_a][1] - pos[e.ligand_b][1]
            dist = max(math.hypot(dx, dy), 1.0)
            f = dist * dist / ideal
            disp[e.ligand_a][0] -= dx / dist * f; disp[e.ligand_a][1] -= dy / dist * f
            disp[e.ligand_b][0] += dx / dist * f; disp[e.ligand_b][1] += dy / dist * f
        temp = radius * 0.08 * (1.0 - it / 180.0)
        for n in names:
            length = max(math.hypot(*disp[n]), 1.0)
            pos[n][0] += disp[n][0] / length * min(length, temp)
            pos[n][1] += disp[n][1] / length * min(length, temp)
    return {n: tuple(p) for n, p in pos.items()}


def render_network_html(bundle_dir) -> str:
    """Interactive SVG of the network; clicking an edge calls bridge.on_edge_clicked."""
    bundle = Path(bundle_dir).expanduser().resolve()
    edges = read_network_edges(bundle)
    names = sorted({n for e in edges for n in (e.ligand_a, e.ligand_b)})
    if len(names) < 2:
        raise ValueError("A ligand network needs at least two ligands")
    pos = _network_layout(names, edges)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs) - 120, max(xs) + 120
    min_y, max_y = min(ys) - 80, max(ys) + 80

    # edge scores (for labels) read from the CSV
    _fields, rows = _read_edge_rows(bundle / "edges.csv")
    scores = {frozenset((r["ligand_a"], r["ligand_b"])):
              (r.get("lomap_score") or r.get("score") or "") for r in rows}
    edge_markup = []
    for e in edges:
        x1, y1 = pos[e.ligand_a]
        x2, y2 = pos[e.ligand_b]
        score = html.escape(scores.get(frozenset((e.ligand_a, e.ligand_b)), ""))
        edge_markup.append(
            f'<g class="edge" data-a="{html.escape(e.ligand_a, quote=True)}" '
            f'data-b="{html.escape(e.ligand_b, quote=True)}">'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'
            f'<text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 - 6:.1f}">{score}</text>'
            f'<title>{html.escape(e.ligand_a)} ↔ {html.escape(e.ligand_b)}</title></g>')
    node_markup = []
    for name in names:
        x, y = pos[name]
        node_markup.append(
            f'<g class="node" data-name="{html.escape(name, quote=True)}" '
            f'data-x="{x:.1f}" data-y="{y:.1f}" transform="translate({x:.1f},{y:.1f})">'
            f'<circle r="30"/><text y="4">{html.escape(name[:12])}</text></g>')

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;height:100%;background:#f4f8fc;font-family:-apple-system,sans-serif}}
#canvas{{width:100%;height:100%;cursor:grab}}#canvas.panning{{cursor:grabbing}}
.edge line{{stroke:#80a9ce;stroke-width:4;cursor:pointer}}
.edge:hover line{{stroke:#e64b3c;stroke-width:6}}
.edge text{{font-size:12px;fill:#4e7294;text-anchor:middle;pointer-events:none}}
.node{{cursor:move}}
.node circle{{fill:#e9f4fd;stroke:#6ba5d8;stroke-width:2}}
.node.sel circle{{fill:#ffe6b3;stroke:#e6902b;stroke-width:4}}
.node text{{font-size:11px;fill:#174f80;text-anchor:middle;font-weight:700;pointer-events:none}}
</style></head><body>
<svg id="canvas" viewBox="{min_x:.1f} {min_y:.1f} {max_x - min_x:.1f} {max_y - min_y:.1f}">
{''.join(edge_markup)}{''.join(node_markup)}</svg>
<script>
window.bridge=window.bridge||null;
const svg=document.getElementById('canvas');
const vb={{x:{min_x:.1f},y:{min_y:.1f},w:{max_x - min_x:.1f},h:{max_y - min_y:.1f}}};
const applyVB=()=>svg.setAttribute('viewBox',`${{vb.x}} ${{vb.y}} ${{vb.w}} ${{vb.h}}`);
const nodeByName=n=>[...document.querySelectorAll('.node')].find(g=>g.dataset.name===n);
function place(node,x,y){{node.dataset.x=x;node.dataset.y=y;node.setAttribute('transform',`translate(${{x}},${{y}})`);}}
function updateEdges(){{document.querySelectorAll('.edge').forEach(e=>{{
  const a=nodeByName(e.dataset.a),b=nodeByName(e.dataset.b);if(!a||!b)return;
  const x1=+a.dataset.x,y1=+a.dataset.y,x2=+b.dataset.x,y2=+b.dataset.y;
  const ln=e.querySelector('line'),tx=e.querySelector('text');
  ln.setAttribute('x1',x1);ln.setAttribute('y1',y1);ln.setAttribute('x2',x2);ln.setAttribute('y2',y2);
  tx.setAttribute('x',(x1+x2)/2);tx.setAttribute('y',(y1+y2)/2-6);}});}}
let pan=false,last=null,nodeDrag=null;
svg.addEventListener('wheel',e=>{{e.preventDefault();const f=e.deltaY>0?1.12:0.89;
  const r=svg.getBoundingClientRect();
  const px=vb.x+(e.clientX-r.left)/r.width*vb.w,py=vb.y+(e.clientY-r.top)/r.height*vb.h;
  vb.w*=f;vb.h*=f;vb.x=px-(px-vb.x)*f;vb.y=py-(py-vb.y)*f;applyVB();}},{{passive:false}});
svg.addEventListener('mousedown',e=>{{last=[e.clientX,e.clientY];
  const node=e.target.closest&&e.target.closest('.node');
  if(node){{nodeDrag={{node,x:+node.dataset.x,y:+node.dataset.y,start:[e.clientX,e.clientY],moved:false}};return;}}
  const edge=e.target.closest&&e.target.closest('.edge');
  if(edge){{if(window.bridge)window.bridge.on_edge_clicked(edge.dataset.a,edge.dataset.b);return;}}
  pan=true;svg.classList.add('panning');}});
window.addEventListener('mousemove',e=>{{const r=svg.getBoundingClientRect();
  if(nodeDrag){{const dx=(e.clientX-nodeDrag.start[0])/r.width*vb.w,dy=(e.clientY-nodeDrag.start[1])/r.height*vb.h;
    if(Math.hypot(e.clientX-nodeDrag.start[0],e.clientY-nodeDrag.start[1])>3)nodeDrag.moved=true;
    place(nodeDrag.node,nodeDrag.x+dx,nodeDrag.y+dy);updateEdges();return;}}
  if(pan){{vb.x-=(e.clientX-last[0])/r.width*vb.w;vb.y-=(e.clientY-last[1])/r.height*vb.h;last=[e.clientX,e.clientY];applyVB();}}}});
window.addEventListener('mouseup',()=>{{
  if(nodeDrag&&!nodeDrag.moved){{nodeDrag.node.classList.toggle('sel');
    if(window.bridge)window.bridge.on_node_clicked(nodeDrag.node.dataset.name);}}
  nodeDrag=null;pan=false;svg.classList.remove('panning');}});
</script></body></html>"""


def render_atom_mapping_html(bundle_dir, selected_a: str, selected_b: str) -> str:
    """Two mapped ligands side by side (corresponding atoms coloured + indexed)."""
    bundle = Path(bundle_dir).expanduser().resolve()
    edges_csv = bundle / "edges.csv"
    ligand_sdf = bundle / "ligands.sdf"
    _fields, rows = _read_edge_rows(edges_csv)
    row = next((r for r in rows if {r["ligand_a"], r["ligand_b"]} == {selected_a, selected_b}), None)
    if row is None:
        raise ValueError(f"{selected_a!r} and {selected_b!r} are not directly connected")
    mapping_path = bundle / "mappings" / f"{row['edge_id']}.json"
    if not mapping_path.is_file():
        raise ValueError(f"atom-mapping file not found: {mapping_path} "
                         "(re-plan the network to generate mappings)")
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    raw = payload.get("componentA_to_componentB") or {}
    mapping = {int(k): int(v) for k, v in raw.items()}
    if not mapping:
        raise ValueError(f"the atom mapping for {row['edge_id']} is empty")

    ligands = _ligands_by_name(ligand_sdf)
    name_a, name_b = row["ligand_a"], row["ligand_b"]
    if name_a not in ligands or name_b not in ligands:
        raise ValueError("mapped ligand names not found in ligands.sdf")
    colors_a = {a: _PALETTE[i % len(_PALETTE)] for i, a in enumerate(mapping)}
    colors_b = {b: _PALETTE[i % len(_PALETTE)] for i, b in enumerate(mapping.values())}
    svg_a = _mapped_molecule_svg(ligands[name_a], colors_a)
    svg_b = _mapped_molecule_svg(ligands[name_b], colors_b)
    pairs = " ".join(f'<span class="pair">{a} ↔ {b}</span>' for a, b in mapping.items())
    score = html.escape(row.get("lomap_score", ""))
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;padding:16px;background:#f4f8fc;color:#254b6b;font-family:-apple-system,sans-serif}}
h2{{margin:0 0 4px;color:#174f80}}.meta{{color:#6d8193;margin-bottom:12px}}
.structures{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.ligand{{background:white;border:1px solid #bed8ec;border-radius:12px;padding:10px;text-align:center}}
.ligand svg{{width:100%;height:auto;max-height:430px}}
.pairs{{margin-top:14px;line-height:2}}.pair{{display:inline-block;background:white;
border:1px solid #c7dced;border-radius:5px;padding:1px 7px;margin:2px;font-family:monospace}}
</style></head><body>
<h2>{html.escape(row['edge_id'])}: atom mapping</h2>
<div class="meta">{len(mapping)} mapped atoms · LoMap score {score}. Atom indices shown.</div>
<div class="structures"><div class="ligand"><h3>{html.escape(name_a)}</h3>{svg_a}</div>
<div class="ligand"><h3>{html.escape(name_b)}</h3>{svg_b}</div></div>
<div class="pairs">{pairs}</div></body></html>"""


def _mapped_molecule_svg(molecule, atom_colors) -> str:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    drawing = Chem.Mol(molecule)
    rdDepictor.Compute2DCoords(drawing)
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 360)
    drawer.drawOptions().addAtomIndices = True
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, drawing, highlightAtoms=sorted(atom_colors), highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.find("<svg"):]
