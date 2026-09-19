"""Application entry point for ChemFlow Studio."""

from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QPalette, QSurfaceFormat
from PySide6.QtWidgets import QApplication

from src.desktop.main_window import MainWindow
from src.desktop.theme import COLORS, stylesheet


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
    surface = QSurfaceFormat()
    surface.setVersion(3, 3)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setDepthBufferSize(24)
    surface.setSamples(4)
    QSurfaceFormat.setDefaultFormat(surface)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ChemFlow Studio")
    app.setOrganizationName("ChemFlow")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
