"""Offline Ketcher molecule editor embedded in the Qt desktop."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QFile, QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


KETCHER_VERSION = "3.12.0"


def find_ketcher_index() -> Path | None:
    """Return the installed standalone Ketcher entry point, if available."""
    configured = os.environ.get("CHEMFLOW_KETCHER_PATH", "").strip()
    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        Path(configured).expanduser() if configured else None,
        project_root
        / ".chemflow"
        / "vendor"
        / f"ketcher-{KETCHER_VERSION}"
        / "standalone",
        Path.home()
        / ".cache"
        / "chemflow"
        / f"ketcher-{KETCHER_VERSION}"
        / "standalone",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        index = candidate if candidate.name == "index.html" else candidate / "index.html"
        if index.is_file():
            return index.resolve()
    return None


class _KetcherBridge(QObject):
    smiles_received = Signal(str)
    error_received = Signal(str)
    ready = Signal()

    @Slot(str)
    def acceptSmiles(self, value: str) -> None:  # Qt/JavaScript API name
        self.smiles_received.emit(value)

    @Slot(str)
    def reportError(self, message: str) -> None:  # Qt/JavaScript API name
        self.error_received.emit(message)

    @Slot()
    def editorReady(self) -> None:  # Qt/JavaScript API name
        self.ready.emit()


class KetcherEditor(QWidget):
    """Full chemical sketcher with SMILES transfer to the prediction form."""

    smiles_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.index_path = find_ketcher_index()
        if self.index_path is None:
            message = QLabel(
                "Ketcher is not installed. Run `chemflow-install-ketcher`, then restart "
                "the desktop application."
            )
            message.setWordWrap(True)
            message.setObjectName("Muted")
            layout.addWidget(message)
            self.web_view = None
            self.smiles = QLineEdit()
            self.smiles.setReadOnly(True)
            layout.addWidget(self.smiles)
            return

        self.web_view = QWebEngineView(self)
        self.web_view.setMinimumSize(720, 520)
        self._bridge = _KetcherBridge(self)
        self._channel = QWebChannel(self.web_view.page())
        self._channel.registerObject("chemflowBridge", self._bridge)
        self.web_view.page().setWebChannel(self._channel)
        self.web_view.loadFinished.connect(self._on_load_finished)
        layout.addWidget(self.web_view)

        controls = QHBoxLayout()
        self.status = QLabel("Loading Ketcher…")
        self.status.setObjectName("Muted")
        self.use_button = QPushButton("Use structure for prediction")
        self.use_button.setEnabled(False)
        self.clear_button = QPushButton("Clear")
        controls.addWidget(self.status)
        controls.addStretch()
        controls.addWidget(self.clear_button)
        controls.addWidget(self.use_button)
        layout.addLayout(controls)

        self.smiles = QLineEdit()
        self.smiles.setReadOnly(True)
        self.smiles.setPlaceholderText(
            "Draw a molecule, then choose “Use structure for prediction”"
        )
        layout.addWidget(self.smiles)

        self._bridge.smiles_received.connect(self._accept_smiles)
        self._bridge.error_received.connect(self._show_error)
        self._bridge.ready.connect(self._editor_ready)
        self.use_button.clicked.connect(self._request_smiles)
        self.clear_button.clicked.connect(self.clear)
        self.web_view.load(QUrl.fromLocalFile(str(self.index_path)))

    def _on_load_finished(self, succeeded: bool) -> None:
        if not succeeded:
            self._show_error("Ketcher failed to load its local assets.")
            return
        resource = QFile(":/qtwebchannel/qwebchannel.js")
        if not resource.open(QFile.OpenModeFlag.ReadOnly):
            self._show_error("Qt WebChannel could not be initialized.")
            return
        library = bytes(resource.readAll()).decode("utf-8")
        resource.close()
        bootstrap = """
new QWebChannel(qt.webChannelTransport, function(channel) {
  window.chemflowBridge = channel.objects.chemflowBridge;
  const waitUntilReady = function() {
    if (window.ketcher) {
      window.chemflowBridge.editorReady();
    } else {
      window.setTimeout(waitUntilReady, 200);
    }
  };
  waitUntilReady();
});
"""
        self.web_view.page().runJavaScript(library + "\n" + bootstrap)

    @Slot()
    def _editor_ready(self) -> None:
        self.status.setText("Ketcher ready · structures stay on this computer")
        self.use_button.setEnabled(True)

    def _request_smiles(self) -> None:
        script = """
(async function() {
  try {
    if (!window.ketcher) throw new Error('Ketcher is still loading');
    const value = await window.ketcher.getSmiles();
    window.chemflowBridge.acceptSmiles(value || '');
  } catch (error) {
    window.chemflowBridge.reportError(String(error));
  }
})();
"""
        self.web_view.page().runJavaScript(script)

    @Slot(str)
    def _accept_smiles(self, value: str) -> None:
        self._value = str(value).strip()
        self.smiles.setText(self._value)
        self.status.setText("Structure transferred to prediction input")
        self.smiles_changed.emit(self._value)

    @Slot(str)
    def _show_error(self, message: str) -> None:
        if hasattr(self, "status"):
            self.status.setText(message)

    def clear(self) -> None:
        self._value = ""
        self.smiles.clear()
        self.smiles_changed.emit("")
        if self.web_view is not None:
            self.web_view.page().runJavaScript(
                "if (window.ketcher) { window.ketcher.setMolecule(''); }"
            )

    def value(self) -> str:
        return self._value
