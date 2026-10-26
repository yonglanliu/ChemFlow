"""Helpers for independent, non-blocking AlchemForge tool windows."""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt


def show_modeless(owner: Any, dialog: Any) -> Any:
    """Show an independent tool window and retain it until the user closes it."""
    registry = getattr(owner, "_modeless_windows", None)
    if registry is None:
        registry = {}
        owner._modeless_windows = registry
    key = id(dialog)
    registry[key] = dialog
    # A QObject/QWidget parent makes macOS place dialogs in one native window
    # group. Detaching the tool window avoids platform-specific behavior where
    # closing a child also dismisses its parent workflow window. The registry
    # above provides the Python lifetime ownership instead.
    flags = dialog.windowFlags() | Qt.Window
    dialog.setParent(None, flags)
    dialog.setModal(False)
    dialog.setWindowModality(Qt.NonModal)
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    dialog.destroyed.connect(
        lambda _object=None, window_key=key: registry.pop(window_key, None)
    )
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
