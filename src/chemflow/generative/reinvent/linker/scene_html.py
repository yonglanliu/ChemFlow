"""Qt-free HTML for the linker-designer 3D views.

Shared by the AlchemForge (PyQt5) and ChemFlow (PySide6) desktop dialogs so
there is one copy of the picking scene. The page expects ``qwebchannel.js`` and a
``window.bridge`` object to be injected by the host *after* the page loads (the
proven pattern in ``chemflow.desktop.widgets.ketcher_editor``); clicking a ligand
atom calls ``window.bridge.on_atom_clicked(head, atom.index)``.

Atom numbering: build each model's MolBlock from the same RDKit Mol later passed
to :func:`chemflow.generative.reinvent.linker.heads.make_head`, so the clicked
``atom.index`` equals the backend's atom index.
"""

from __future__ import annotations

import json

_3DMOL_CDN = "https://cdn.jsdelivr.net/npm/3dmol@2.1.0/build/3Dmol-min.js"
_PICK_COLORS = {0: "0x2ca02c", 1: "0x1f77b4"}      # head1 green, head2 blue
_PICK_PALETTE = ["0x2ca02c", "0x1f77b4", "0xff7f0e", "0xd62728", "0x9467bd", "0x8c564b"]


def molblock_for_viewer(mol) -> str:
    from rdkit import Chem

    return Chem.MolToMolBlock(mol)


def build_workspace_html(
    ligands,
    *,
    proteins=None,
    selected: dict[int, int] | None = None,
    colors: dict[int, str] | None = None,
    background: str = "white",
) -> str:
    """Maestro-style workspace: display any number of included structures.

    ``ligands`` is a list of ``(entry_id, molblock)`` for clickable small
    molecules; clicking an atom calls ``window.bridge.on_atom_clicked(entry_id,
    atom.index)``. ``proteins`` is a list of PDB text shown as cartoons.
    ``selected`` maps ``entry_id`` to the picked attachment atom index, drawn as a
    coloured sphere; ``colors`` optionally maps ``entry_id`` to a ``0xRRGGBB``
    sphere colour (e.g. by role), else a palette is used.
    """
    ligands = list(ligands)
    proteins = proteins or []
    selected = selected or {}
    colors = colors or {}
    parts = []
    for pos, (eid, molblock) in enumerate(ligands):
        parts.append(
            f"var m{pos} = viewer.addModel({json.dumps(molblock)}, 'sdf');\n"
            f"m{pos}.setStyle({{}}, {{stick:{{radius:0.14, colorscheme:'Jmol'}}}});\n"
            f"m{pos}.setClickable({{}}, true, (function(eid){{"
            f"return function(atom){{ if (window.bridge) window.bridge.on_atom_clicked(eid, atom.index); }};"
            f"}})({int(eid)}));\n"
        )
        if eid in selected:
            color = colors.get(eid) or _PICK_PALETTE[pos % len(_PICK_PALETTE)]
            parts.append(
                f"viewer.addStyle({{model:{pos}, index:{int(selected[eid])}}}, "
                f"{{sphere:{{radius:0.6, color:'{color}'}}}});\n"
            )
    for pdb in proteins:
        parts.append(
            f"viewer.addModel({json.dumps(pdb)}, 'pdb')"
            ".setStyle({}, {cartoon:{color:'spectrum', opacity:0.65}});\n"
        )
    if not parts:
        parts.append("")
    return _page(background, "".join(parts))


def build_pick_scene_html(
    head1_molblock: str | None,
    head2_molblock: str | None,
    *,
    protein_pdb: str | None = None,
    selected: dict[int, int] | None = None,
    background: str = "white",
) -> str:
    """Scene with up to two clickable ligands and an optional protein cartoon."""
    selected = selected or {}
    parts = []
    for head, molblock in ((0, head1_molblock), (1, head2_molblock)):
        if molblock is None:
            continue
        parts.append(
            f"var m{head} = viewer.addModel({json.dumps(molblock)}, 'sdf');\n"
            f"m{head}.setStyle({{}}, {{stick:{{radius:0.14, colorscheme:'Jmol'}}}});\n"
            f"m{head}.setClickable({{}}, true, (function(head){{"
            f"return function(atom){{ if (window.bridge) window.bridge.on_atom_clicked(head, atom.index); }};"
            f"}})({head}));\n"
        )
    for head, idx in selected.items():
        color = _PICK_COLORS.get(int(head), "0xff7f0e")
        parts.append(
            f"viewer.addStyle({{model:{int(head)}, index:{int(idx)}}}, "
            f"{{sphere:{{radius:0.6, color:'{color}'}}}});\n"
        )
    if protein_pdb:
        parts.append(
            f"viewer.addModel({json.dumps(protein_pdb)}, 'pdb')"
            ".setStyle({}, {cartoon:{color:'spectrum', opacity:0.65}});\n"
        )
    return _page(background, "".join(parts))


def build_pose_html(molblock: str, *, protein_pdb: str | None = None, background: str = "white") -> str:
    """Scene showing one assembled PROTAC pose (not interactive)."""
    parts = [
        f"viewer.addModel({json.dumps(molblock)}, 'sdf')"
        ".setStyle({}, {stick:{radius:0.15, colorscheme:'Jmol'}});\n"
    ]
    if protein_pdb:
        parts.append(
            f"viewer.addModel({json.dumps(protein_pdb)}, 'pdb')"
            ".setStyle({}, {cartoon:{color:'spectrum', opacity:0.6}});\n"
        )
    return _page(background, "".join(parts))


def _page(background: str, body: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>html,body{{width:100%;height:100%;margin:0;overflow:hidden;background:{background};}}
#viewport{{width:100%;height:100%;position:relative;}}</style>
<script src="{_3DMOL_CDN}"></script>
</head><body><div id="viewport"></div>
<script>
window.bridge = window.bridge || null;   // the host injects the real bridge after load
var viewer = $3Dmol.createViewer(document.getElementById('viewport'), {{backgroundColor: '{background}'}});
{body}viewer.zoomTo();
viewer.render();
</script></body></html>"""
