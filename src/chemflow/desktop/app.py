"""Application entry point for ChemFlow Studio."""

from __future__ import annotations

import faulthandler
import os
import sys

# Print a C + Python traceback to stderr if the process receives a fatal signal
# (e.g. SIGSEGV), so native Qt crashes are diagnosable instead of a bare
# "Segmentation fault: 11".
faulthandler.enable()


def _configure_webengine_rendering() -> None:
    """Fall back to software rendering for the embedded Chromium (QtWebEngine).

    On a remote/headless Linux session (HPC over SSH/X11 or VNC) WebEngine's GPU
    process cannot create a command buffer
    ("ContextResult::kTransientFailure: Failed to send GpuControl.CreateCommandBuffer"),
    which leaves the RBFE/ABFE network graphs, linker designer and 3D viewers blank.
    Disabling GPU acceleration fixes it; the cost is negligible for these views.

    Applied on Linux unless the user opts into hardware GL (``CHEMFLOW_DESKTOP_GPU=1``)
    or has already set ``QTWEBENGINE_CHROMIUM_FLAGS`` themselves. Must run before any
    QtWebEngine import, so it lives at module import time, ahead of the Qt imports
    below.
    """
    if sys.platform != "linux":
        return
    if str(os.environ.get("CHEMFLOW_DESKTOP_GPU", "")).lower() in ("1", "true", "yes"):
        return
    os.environ.setdefault(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        "--disable-gpu --disable-gpu-compositing --no-sandbox",
    )
    os.environ.setdefault("QT_OPENGL", "software")


_configure_webengine_rendering()

from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

from chemflow.config import PROJECT_ROOT
from chemflow.desktop.main_window import MainWindow
from chemflow.desktop.theme import COLORS, stylesheet


def _dark_palette() -> QPalette:
    """Provide colors for native popup surfaces not fully covered by QSS."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["canvas"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS["surface_alt"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS["surface"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#245f6c"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLORS["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(COLORS["muted"]))
    return palette


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ChemFlow Studio")
    _logo = PROJECT_ROOT / "assets" / "images" / "chemflow_logo.svg"
    if _logo.is_file():
        app.setWindowIcon(QIcon(str(_logo)))
    app.setOrganizationName("ChemFlow")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
