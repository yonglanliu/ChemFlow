"""Interactive 3D structure / binding-pose viewer (3Dmol.js in QtWebEngine).

Renders a predicted complex (PDB / mmCIF) with the receptor as a cartoon and the
ligand (HETATM / non-polymer) as sticks, zoomed to the binding pose — the Qt
desktop equivalent of a py3Dmol viewer.

Asset resolution (offline-first, so it works on HPC compute nodes with no
internet):
  1. ``CHEMFLOW_3DMOL_PATH`` env var (a ``3Dmol-min.js`` file), or
  2. a vendored copy under ``<project>/.chemflow/vendor/3dmol/3Dmol-min.js`` or
     ``~/.cache/chemflow/3dmol/3Dmol-min.js`` — inlined into the page, or
  3. the 3Dmol.js CDN (needs internet; a workstation convenience fallback).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from chemflow.config import PROJECT_ROOT

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

_CDN_URL = "https://3Dmol.org/build/3Dmol-min.js"


def find_3dmol_js() -> Path | None:
    """Return a local ``3Dmol-min.js`` bundle for fully-offline rendering."""
    configured = os.environ.get("CHEMFLOW_3DMOL_PATH", "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        PROJECT_ROOT / ".chemflow" / "vendor" / "3dmol" / "3Dmol-min.js",
        Path.home() / ".cache" / "chemflow" / "3dmol" / "3Dmol-min.js",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        js = candidate if candidate.name.endswith(".js") else candidate / "3Dmol-min.js"
        if js.is_file():
            return js.resolve()
    return None


def _fmt_for(path: Path) -> str:
    """Map a structure file extension to a 3Dmol model format."""
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif", ".mcif"}:
        return "mmcif"
    if suffix in {".sdf", ".mol"}:
        return "sdf"
    if suffix in {".mol2"}:
        return "mol2"
    if suffix in {".xyz"}:
        return "xyz"
    return "pdb"


class Molecule3DView(QWidget):
    """A 3Dmol.js viewer for predicted complexes / binding poses."""

    def __init__(self, parent=None, *, min_height: int = 360):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.web_view = QWebEngineView(self)
        self.web_view.setMinimumHeight(min_height)
        layout.addWidget(self.web_view)
        self._ready = False
        self._pending: tuple[str, str] | None = None
        self.web_view.loadFinished.connect(self._on_load_finished)
        self.web_view.setHtml(self._page_html(), QUrl("about:blank"))

    # -- public API -------------------------------------------------------
    def load_structure(self, path: str | Path) -> None:
        """Load and render a structure file (PDB / mmCIF / SDF / mol2 / xyz)."""
        target = Path(path).expanduser()
        if not target.is_file():
            self._show_message(f"File not found: {target}")
            return
        data = target.read_text(encoding="utf-8", errors="replace")
        self.load_data(data, _fmt_for(target))

    def load_data(self, data: str, fmt: str = "pdb") -> None:
        """Render structure text already in memory."""
        if not self._ready:
            self._pending = (data, fmt)
            return
        script = f"window.renderModel({json.dumps(data)}, {json.dumps(fmt)});"
        self.web_view.page().runJavaScript(script)

    def clear(self) -> None:
        if self._ready:
            self.web_view.page().runJavaScript("window.clearModel && window.clearModel();")

    # -- internals --------------------------------------------------------
    def _on_load_finished(self, ok: bool) -> None:
        self._ready = bool(ok)
        if ok and self._pending is not None:
            data, fmt = self._pending
            self._pending = None
            self.load_data(data, fmt)

    def _show_message(self, message: str) -> None:
        if self._ready:
            self.web_view.page().runJavaScript(
                f"window.showMessage && window.showMessage({json.dumps(message)});"
            )

    def _three_dmol_script_tag(self) -> str:
        local = find_3dmol_js()
        if local is not None:
            try:
                return "<script>\n" + local.read_text(encoding="utf-8") + "\n</script>"
            except OSError:
                pass
        # No local bundle. Only try the CDN when explicitly allowed — on an offline
        # HPC node a <script src> to the CDN hangs the page load and the viewer is
        # left showing the stale placeholder. By default we skip it so the page
        # loads instantly and renderModel() can show an actionable message.
        if os.environ.get("CHEMFLOW_3DMOL_ALLOW_CDN", "").strip() not in ("", "0", "false", "False"):
            return f'<script src="{_CDN_URL}"></script>'
        return ""

    def _page_html(self) -> str:
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>"
            "html,body{margin:0;padding:0;height:100%;background:#0b1622;}"
            "#viewer{width:100%;height:100vh;position:relative;}"
            "#msg{position:absolute;top:8px;left:12px;color:#9fb4c8;"
            "font:13px/1.4 -apple-system,Segoe UI,sans-serif;z-index:5;}"
            "</style>"
            f"{self._three_dmol_script_tag()}"
            "</head><body>"
            "<div id='viewer'><div id='msg'>Run a docking job, then load its "
            "predicted complex to see the binding pose.</div></div>"
            "<script>"
            "var _viewer=null;"
            "function _ensure(){"
            "  if(typeof $3Dmol==='undefined'){return null;}"
            "  if(!_viewer){_viewer=$3Dmol.createViewer('viewer',{backgroundColor:'0x0b1622'});}"
            "  return _viewer;"
            "}"
            "window.showMessage=function(t){var m=document.getElementById('msg');"
            "if(m){m.textContent=t;m.style.display='block';}};"
            "window.clearModel=function(){var v=_ensure();if(v){v.clear();v.render();}};"
            "window.renderModel=function(data,fmt){"
            "  var v=_ensure();"
            "  if(!v){window.showMessage('3Dmol.js not found. Run  chemflow vendor-3dmol  "
            "on a node with internet (or set CHEMFLOW_3DMOL_PATH), then reopen this page.');return;}"
            "  var m=document.getElementById('msg');if(m){m.style.display='none';}"
            "  v.clear();"
            "  try{v.addModel(data,fmt);}catch(e){window.showMessage('Parse error: '+e);return;}"
            "  v.setStyle({},{cartoon:{color:'spectrum'}});"
            "  v.setStyle({hetflag:true},{stick:{colorscheme:'greenCarbon',radius:0.18}});"
            "  v.addStyle({hetflag:true},{sphere:{scale:0.28}});"
            "  var sel={hetflag:true};"
            "  try{v.zoomTo(sel);}catch(e){v.zoomTo();}"
            "  v.render();"
            "};"
            "</script>"
            "</body></html>"
        )
