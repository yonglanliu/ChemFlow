"""Standalone ligand-network planning desktop workflow."""

from __future__ import annotations

import re
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from alchemforge.desktop.dialogs.network_viewer_dialog import (
    LigandNetworkViewerDialog,
)
from alchemforge.desktop.modeless import show_modeless
from alchemforge.gromacs.rbfe.mapping import LigandNetworkSettings, plan_network


def resolve_network_bundle(path: str | Path) -> Path:
    """Accept either a network bundle directory or its parent project."""
    selected = Path(path).expanduser().resolve()
    candidates = (selected, selected / "network")
    required = ("edges.csv", "ligands.sdf", "ligand_network.graphml")
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in required):
            return candidate
    raise ValueError(
        "The selected directory is not a ligand network bundle. Expected "
        "edges.csv, ligands.sdf, and ligand_network.graphml."
    )


class LigandNetworkPlannerDialog(QDialog):
    """Plan and inspect a portable ligand-only perturbation network."""

    network_created = pyqtSignal(str)

    def __init__(
        self,
        working_directory: str | None = None,
        parent=None,
        *,
        embedded: bool = False,
    ):
        super().__init__(parent)
        self.embedded = embedded
        if embedded:
            self.setWindowFlags(Qt.Widget)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self._network_viewer = None
        self._network_dir: Path | None = None
        self.setWindowTitle("Ligand Network Planning")
        self.resize(760, 660)
        root = QVBoxLayout(self)

        title = QLabel("Ligand Network Planning")
        title.setStyleSheet("font-size:18pt;font-weight:700;color:#123f76")
        root.addWidget(title)
        description = QLabel(
            "Create the ligand graph and atom mappings before system setup. The "
            "portable network bundle can be reused by both OpenFE and GROMACS "
            "workflows with different proteins, membranes, or simulation protocols."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color:#71839a")
        root.addWidget(description)

        source_group = QGroupBox("1. Ligand Library and Output")
        source_form = QFormLayout(source_group)
        source_form.setLabelAlignment(Qt.AlignRight)
        self.ligands = QLineEdit()
        self.output_root = QLineEdit(self.working_directory)
        self.name = QLineEdit("ligand_network")
        source_form.addRow("Multi-molecule SDF:", self._file_row(self.ligands))
        source_form.addRow("Output root:", self._directory_row(self.output_root))
        source_form.addRow("Network name:", self.name)
        root.addWidget(source_group)

        planning_group = QGroupBox("2. Graph and Atom Mapping")
        planning_form = QFormLayout(planning_group)
        planning_form.setLabelAlignment(Qt.AlignRight)
        self.target = QComboBox()
        self.target.addItems(["OpenFE + GROMACS", "OpenFE", "GROMACS"])
        self.target.setToolTip(
            "OpenFE/LoMap graph bundles work with both engines. PMX mappings are "
            "available only for GROMACS."
        )
        self.method = QComboBox()
        self.method.addItems(
            ["lomap", "minimal_redundant", "minimal_spanning", "radial"]
        )
        self.mapper = QComboBox()
        self.mapper.addItems(["lomap", "kartograf", "pmx"])
        self.max3d = QDoubleSpinBox()
        self.max3d.setRange(0.1, 100.0)
        self.max3d.setValue(1.0)
        self.max3d.setSuffix(" Å")
        self.element_change = QCheckBox("Allow element changes")
        self.cycle_covering = QCheckBox("Require every ligand in a cycle")
        self.cycle_covering.setChecked(True)
        self.redundant = QSpinBox()
        self.redundant.setRange(1, 20)
        self.redundant.setValue(2)
        self.radial_hub = QLineEdit()
        self.radial_hub.setPlaceholderText("Exact ligand _Name")
        self.pmx_mode = QComboBox()
        self.pmx_mode.addItems(["stage", "run"])
        planning_form.addRow("Intended engine:", self.target)
        planning_form.addRow("Network topology:", self.method)
        planning_form.addRow("Atom mapper:", self.mapper)
        planning_form.addRow("3D mapping cutoff:", self.max3d)
        planning_form.addRow("", self.element_change)
        planning_form.addRow("", self.cycle_covering)
        planning_form.addRow("Redundant spanning trees:", self.redundant)
        planning_form.addRow("Radial hub:", self.radial_hub)
        planning_form.addRow("PMX action:", self.pmx_mode)
        root.addWidget(planning_group)

        bundle_note = QLabel(
            "Output bundle: ligands.sdf, ligand_network.graphml, edges.csv, mapping "
            "JSON files, and a manifest. Network editing preserves existing mapping IDs."
        )
        bundle_note.setWordWrap(True)
        bundle_note.setStyleSheet("color:#5f7387;background:#eef6fc;padding:9px")
        root.addWidget(bundle_note)
        root.addStretch()

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.show_button = QPushButton("Show / Edit Network")
        self.show_button.setToolTip(
            "Browse to a saved network bundle, then open it for inspection or editing"
        )
        self.show_button.clicked.connect(self.show_network)
        plan = QPushButton("Create Ligand Network")
        plan.clicked.connect(self.create_network)
        buttons.addWidget(self.show_button)
        buttons.addWidget(plan)
        root.addLayout(buttons)

        self.method.currentTextChanged.connect(self._update_options)
        self.mapper.currentTextChanged.connect(self._update_options)
        self.target.currentTextChanged.connect(self._update_options)
        self._update_options()

    @staticmethod
    def _row(field: QLineEdit, button: QPushButton) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _file_row(self, field: QLineEdit) -> QWidget:
        button = QPushButton("Browse…")

        def browse() -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select ligand library", self.working_directory, "SDF (*.sdf)"
            )
            if path:
                field.setText(path)

        button.clicked.connect(browse)
        return self._row(field, button)

    def _directory_row(self, field: QLineEdit) -> QWidget:
        button = QPushButton("Browse…")

        def browse() -> None:
            path = QFileDialog.getExistingDirectory(
                self, "Select output root", self.working_directory
            )
            if path:
                field.setText(path)

        button.clicked.connect(browse)
        return self._row(field, button)

    def _update_options(self) -> None:
        method = self.method.currentText()
        self.radial_hub.setEnabled(method == "radial")
        self.redundant.setEnabled(method == "minimal_redundant")
        self.cycle_covering.setEnabled(method == "lomap")
        gromacs_allowed = self.target.currentText() == "GROMACS"
        if not gromacs_allowed and self.mapper.currentText() == "pmx":
            self.mapper.setCurrentText("lomap")
        model = self.mapper.model()
        model.item(2).setEnabled(gromacs_allowed)
        self.pmx_mode.setEnabled(self.mapper.currentText() == "pmx")

    def create_network(self) -> None:
        try:
            name = self.name.text().strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                raise ValueError(
                    "Network name must use only letters, numbers, '.', '_' or '-'"
                )
            ligand_sdf = Path(self.ligands.text()).expanduser().resolve()
            if not ligand_sdf.is_file() or ligand_sdf.suffix.lower() != ".sdf":
                raise ValueError("Select an existing multi-molecule SDF")
            project_dir = Path(self.output_root.text()).expanduser().resolve() / name
            settings = LigandNetworkSettings(
                project_dir=project_dir,
                ligand_library=ligand_sdf,
                network_method=self.method.currentText(),
                atom_mapper=self.mapper.currentText(),
                max3d_angstrom=self.max3d.value(),
                element_change=self.element_change.isChecked(),
                require_cycle_covering=self.cycle_covering.isChecked(),
                radial_hub=self.radial_hub.text().strip() or None,
                redundant_networks=self.redundant.value(),
                pmx_mapping_mode=self.pmx_mode.currentText(),
            )
            result = plan_network(settings)
        except (OSError, RuntimeError, TypeError, ValueError, ImportError) as error:
            QMessageBox.critical(self, "Network Planning Failed", str(error))
            return
        self._network_dir = result.edges_csv.parent
        self.network_created.emit(str(self._network_dir))
        QMessageBox.information(
            self,
            "Ligand Network Created",
            f"Bundle: {self._network_dir}\nLigands: {result.ligand_count}\n"
            f"Transformations: {result.edge_count}\n\nUse this bundle in the OpenFE "
            "or GROMACS system-setup window.",
        )
        self._open_network_viewer(self._network_dir)

    def show_network(self) -> None:
        """Browse every time so an earlier selection cannot be reopened silently."""
        suggested = (
            Path(self.output_root.text()).expanduser()
            / self.name.text().strip()
            / "network"
        )
        start = self._network_dir or (
            suggested if suggested.is_dir() else suggested.parent
        )
        if not start.is_dir():
            start = Path(self.working_directory)
        selected = QFileDialog.getExistingDirectory(
            self, "Open Ligand Network Bundle", str(start)
        )
        if not selected:
            return
        try:
            network_dir = resolve_network_bundle(selected)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid Network Bundle", str(error))
            return
        self._network_dir = network_dir
        self._open_network_viewer(network_dir)

    def _open_network_viewer(self, network_dir: Path) -> None:
        try:
            self._network_viewer = LigandNetworkViewerDialog(
                network_dir / "edges.csv",
                network_dir / "ligands.sdf",
                self,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Show Network", str(error))
            return
        show_modeless(self, self._network_viewer)
