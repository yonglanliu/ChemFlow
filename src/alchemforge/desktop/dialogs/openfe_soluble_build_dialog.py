"""Desktop dialog for the OpenFE soluble-system builder."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from alchemforge.openmm_modeling.soluble_build_config import (
    DEFAULT_FORCEFIELDS,
    SUPPORTED_BOX_SHAPES,
    SUPPORTED_WATER_MODELS,
    stage as stage_soluble_build,
)


class OpenFESolubleBuildDialog(QDialog):
    """Solvate + ionize a soluble protein(+ligand) complex in a water box."""

    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("OpenMM · Soluble System Builder")
        self.resize(800, 560)

        root = QVBoxLayout(self)
        title = QLabel("Soluble System Builder (OpenMM addSolvent)")
        title.setStyleSheet("font-size:18pt; font-weight:700; color:#123f76;")
        root.addWidget(title)
        note = QLabel(
            "Solvate a soluble protein(+ligand) complex in a water box and add "
            "ions. Output feeds the MD runner or an OpenFE FEP workflow."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.job_name = QLineEdit("soluble_build")
        form.addRow("Job name:", self.job_name)

        self.input_pdb = QLineEdit()
        self.input_pdb.setPlaceholderText("Protein PDB (optionally with a bound ligand)")
        form.addRow("Complex PDB:", self._file_row(self.input_pdb, "Select PDB", "PDB (*.pdb)"))

        self.ligand_sdf = QLineEdit()
        self.ligand_sdf.setPlaceholderText("Optional — parameterize a ligand in the PDB")
        form.addRow("Ligand SDF:", self._file_row(self.ligand_sdf, "Select ligand SDF", "SDF (*.sdf)"))

        self.small_molecule_ff = QLineEdit("gaff-2.11")
        form.addRow("Ligand force field:", self.small_molecule_ff)

        self.output_pdb = QLineEdit("soluble_complex.pdb")
        form.addRow("Output PDB:", self.output_pdb)

        self.solvent_model = QComboBox()
        self.solvent_model.addItems(sorted(SUPPORTED_WATER_MODELS))
        self.solvent_model.setCurrentText("tip3p")
        form.addRow("Water model:", self.solvent_model)

        self.box_mode = QComboBox()
        self.box_mode.addItems(["padding", "boxsize"])
        self.box_mode.currentTextChanged.connect(self._sync_mode)
        form.addRow("Box mode:", self.box_mode)

        self.padding = QDoubleSpinBox()
        self.padding.setRange(0.1, 100.0)
        self.padding.setSingleStep(0.1)
        self.padding.setDecimals(2)
        self.padding.setValue(1.0)
        form.addRow("Padding (nm):", self.padding)

        self.box_shape = QComboBox()
        self.box_shape.addItems(sorted(SUPPORTED_BOX_SHAPES))
        self.box_shape.setCurrentText("cube")
        form.addRow("Box shape:", self.box_shape)

        self.box_x = self._dim_spin()
        self.box_y = self._dim_spin()
        self.box_z = self._dim_spin()
        box_row = QHBoxLayout()
        for widget in (self.box_x, self.box_y, self.box_z):
            box_row.addWidget(widget)
        box_container = QWidget()
        box_container.setLayout(box_row)
        form.addRow("Box size X/Y/Z (nm):", box_container)

        self.positive_ion = QLineEdit("Na+")
        self.negative_ion = QLineEdit("Cl-")
        form.addRow("Positive ion:", self.positive_ion)
        form.addRow("Negative ion:", self.negative_ion)

        self.ionic_strength = QDoubleSpinBox()
        self.ionic_strength.setRange(0.0, 5.0)
        self.ionic_strength.setDecimals(3)
        self.ionic_strength.setValue(0.15)
        form.addRow("Ionic strength (M):", self.ionic_strength)

        self.neutralize = QCheckBox("Neutralize net charge with counterions")
        self.neutralize.setChecked(True)
        form.addRow("", self.neutralize)

        self.forcefields = QPlainTextEdit("\n".join(DEFAULT_FORCEFIELDS))
        self.forcefields.setMaximumHeight(110)
        form.addRow("System force fields:", self.forcefields)

        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        create = QPushButton("Create Build Job")
        create.clicked.connect(self.create_job)
        buttons.addWidget(cancel)
        buttons.addWidget(create)
        root.addLayout(buttons)

        self._sync_mode(self.box_mode.currentText())

    def _dim_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(1.0, 100.0)
        spin.setDecimals(2)
        spin.setValue(8.0)
        return spin

    def _sync_mode(self, mode: str) -> None:
        is_padding = mode == "padding"
        self.padding.setEnabled(is_padding)
        self.box_shape.setEnabled(is_padding)
        for widget in (self.box_x, self.box_y, self.box_z):
            widget.setEnabled(not is_padding)

    def _file_row(self, field: QLineEdit, title: str, file_filter: str) -> QWidget:
        button = QPushButton("Browse…")

        def select() -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, title, self.working_directory, file_filter
            )
            if path:
                field.setText(path)

        button.clicked.connect(select)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)
        container = QWidget()
        container.setLayout(layout)
        return container

    def _build_config(self) -> dict:
        forcefields = [
            line.strip()
            for line in self.forcefields.toPlainText().splitlines()
            if line.strip()
        ]
        if not forcefields:
            raise ValueError("Provide at least one system force-field XML.")

        mode = self.box_mode.currentText()
        box: dict = {"mode": mode}
        if mode == "padding":
            box["padding"] = float(self.padding.value())
            box["box_shape"] = self.box_shape.currentText()
        else:
            box["boxsize"] = [
                float(self.box_x.value()),
                float(self.box_y.value()),
                float(self.box_z.value()),
            ]

        config: dict = {
            "input_pdb": self.input_pdb.text().strip(),
            "output_pdb": self.output_pdb.text().strip() or "soluble_complex.pdb",
            "solvent_model": self.solvent_model.currentText(),
            "positive_ion": self.positive_ion.text().strip() or "Na+",
            "negative_ion": self.negative_ion.text().strip() or "Cl-",
            "ionic_strength": float(self.ionic_strength.value()),
            "neutralize": self.neutralize.isChecked(),
            "forcefields": forcefields,
            "box": box,
        }
        ligand = self.ligand_sdf.text().strip()
        if ligand:
            config["ligand_sdf"] = ligand
            config["small_molecule_forcefield"] = self.small_molecule_ff.text().strip()
        return config

    def create_job(self) -> None:
        import yaml

        job_name = self.job_name.text().strip()
        if not job_name:
            QMessageBox.warning(self, "Missing Job Name", "Please enter a job name.")
            return
        if not self.input_pdb.text().strip():
            QMessageBox.warning(self, "Missing Complex", "Please select the complex PDB.")
            return
        try:
            config = self._build_config()
            task = Path(self.working_directory).expanduser().resolve() / job_name
            task.mkdir(parents=True, exist_ok=True)
            config_path = task / f"{job_name}_soluble_build.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            runner, _ = stage_soluble_build(config_path, task)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Create Job", str(error))
            return
        self.job_requested.emit(str(task), str(runner), str(config_path))
        self.accept()


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = OpenFESolubleBuildDialog()
    dialog.show()
    sys.exit(app.exec_())
