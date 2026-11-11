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
from chemflow.desktop.pose_scene import COMPLEX, VIEWER_JS, build_scene

from PySide6.QtCore import QUrl, Signal
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
    """A 3Dmol.js viewer for predicted complexes / binding poses.

    Renders a *scene* of objects (see :mod:`chemflow.desktop.pose_scene`): a Boltz/Chai complex, a
    DiffDock ligand pose + its separate receptor, and/or a multi-frame diffusion trajectory.
    """

    frames_changed = Signal(int)   # number of trajectory frames in the loaded scene (0 = none)

    def __init__(self, parent=None, *, min_height: int = 360):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.web_view = QWebEngineView(self)
        self.web_view.setMinimumHeight(min_height)
        layout.addWidget(self.web_view)
        self._ready = False
        self._pending: tuple[list, bool] | None = None
        self.web_view.loadFinished.connect(self._on_load_finished)
        self.web_view.setHtml(self._page_html(), QUrl("about:blank"))

    # -- public API -------------------------------------------------------
    def load_scene(self, objects: list[dict], autoplay: bool = True) -> None:
        """Render objects ``[{data, fmt, role}]`` (role: complex | receptor | ligand | trajectory)."""
        if not self._ready:
            self._pending = (objects, autoplay)
            return
        script = (f"window.renderScene({json.dumps(objects)}, {json.dumps(bool(autoplay))});")
        self.web_view.page().runJavaScript(script, self._emit_frames)

    def load_files(self, pose: str | None = None, receptor: str | None = None,
                   trajectory: str | None = None, autoplay: bool = True) -> list[str]:
        """Render whichever files are given (complex | ligand pose + receptor | trajectory).

        Returns human-readable notes (e.g. "ligand-only file: choose a receptor").
        """
        objects, notes = build_scene(pose, receptor, trajectory)
        if objects:
            self.load_scene(objects, autoplay)
        else:
            self._show_message("Nothing to show — choose a pose / complex file first.")
        return notes

    def load_structure(self, path: str | Path) -> None:
        """Load and render a single structure file (PDB / mmCIF / SDF / mol2 / xyz)."""
        target = Path(path).expanduser()
        if not target.is_file():
            self._show_message(f"File not found: {target}")
            return
        self.load_files(pose=str(target))

    def load_data(self, data: str, fmt: str = "pdb") -> None:
        """Render structure text already in memory (treated as a complex)."""
        self.load_scene([{"data": data, "fmt": fmt, "role": COMPLEX}])

    def play(self) -> None:
        if self._ready:
            self.web_view.page().runJavaScript("window.playTraj && window.playTraj();")

    def pause(self) -> None:
        if self._ready:
            self.web_view.page().runJavaScript("window.pauseTraj && window.pauseTraj();")

    def set_frame(self, index: int) -> None:
        if self._ready:
            self.web_view.page().runJavaScript(f"window.setTrajFrame && window.setTrajFrame({int(index)});")

    def clear(self) -> None:
        if self._ready:
            self.web_view.page().runJavaScript("window.clearModel && window.clearModel();")

    # -- internals --------------------------------------------------------
    def _on_load_finished(self, ok: bool) -> None:
        self._ready = bool(ok)
        if ok and self._pending is not None:
            objects, autoplay = self._pending
            self._pending = None
            self.load_scene(objects, autoplay)

    def _emit_frames(self, n) -> None:
        try:
            self.frames_changed.emit(max(0, int(n)))
        except (TypeError, ValueError):
            self.frames_changed.emit(0)

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
            "result (a complex, or a ligand pose + receptor) to see the binding pose.</div></div>"
            "<script>" + VIEWER_JS + "</script>"
            "</body></html>"
        )
