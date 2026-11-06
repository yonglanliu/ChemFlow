"""Desktop dialog for the OpenFE membrane builder (MembraneModeller)."""

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

from alchemforge.openmm_modeling.membrane_build_config import (
    DEFAULT_FORCEFIELDS,
    SUPPORTED_LIPIDS,
    SUPPORTED_WATER_MODELS,
    stage as stage_membrane_build,
)


class OpenFEMembraneBuildDialog(QDialog):
    """Build + trim a minimal membrane around a protein (or solvate an existing one)."""

    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("OpenMM · Membrane Builder")
        self.resize(820, 680)

        root = QVBoxLayout(self)
        title = QLabel("Membrane Builder (OpenMM MembraneModeller)")
        title.setStyleSheet("font-size:18pt; font-weight:700; color:#123f76;")
        root.addWidget(title)
        note = QLabel(
            "build mode tiles a lipid patch around the protein and trims it to "
            "protein-XY + padding (a minimal membrane), then solvates and adds "
            "ions. existing mode only solvates + ionizes a system that already "
            "has a membrane. Output feeds Membrane Relaxation. Membrane normal is Z."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.job_name = QLineEdit("membrane_build")
        form.addRow("Job name:", self.job_name)

        self.input_pdb = QLineEdit()
        self.input_pdb.setPlaceholderText("Protein PDB (optionally with a bound ligand)")
        form.addRow("Protein PDB:", self._file_row(self.input_pdb, "Select protein PDB", "PDB (*.pdb)"))

        self.ligand_sdf = QLineEdit()
        self.ligand_sdf.setPlaceholderText("Optional — parameterize a ligand in the PDB")
        form.addRow("Ligand SDF:", self._file_row(self.ligand_sdf, "Select ligand SDF", "SDF (*.sdf)"))

        self.small_molecule_ff = QLineEdit("gaff-2.11")
        self.small_molecule_ff.setToolTip("gaff-2.11 or an openff-x.y.z SMIRNOFF line")
        form.addRow("Ligand force field:", self.small_molecule_ff)

        self.mode = QComboBox()
        self.mode.addItems(["build", "existing"])
        self.mode.setToolTip(
            "build: construct + trim a membrane around the protein. "
            "existing: input already contains a membrane; only solvate + ionize."
        )
        self.mode.currentTextChanged.connect(self._sync_mode)
        form.addRow("Mode:", self.mode)

        self.platform = QComboBox()
        self.platform.addItems(["CUDA", "OpenCL", "CPU"])
        form.addRow("OpenMM platform:", self.platform)

        self.output_pdb = QLineEdit("membrane_complex.pdb")
        form.addRow("Output PDB:", self.output_pdb)

        self.lipid_type = QComboBox()
        self.lipid_type.addItems(sorted(SUPPORTED_LIPIDS))
        self.lipid_type.setCurrentText("POPC")
        form.addRow("Lipid type:", self.lipid_type)

        self.scale_factor = QDoubleSpinBox()
        self.scale_factor.setRange(0.05, 1.0)
        self.scale_factor.setSingleStep(0.05)
        self.scale_factor.setDecimals(2)
        self.scale_factor.setValue(0.6)
        self.scale_factor.setToolTip("XY shrink of the solute while packing lipids")
        form.addRow("Scale factor:", self.scale_factor)

        self.x_padding = self._nm_spin(1.0)
        self.y_padding = self._nm_spin(1.0)
        self.z_padding = self._nm_spin(1.0)
        form.addRow("X padding (nm):", self.x_padding)
        form.addRow("Y padding (nm):", self.y_padding)
        form.addRow("Z padding (nm, water):", self.z_padding)

        self.membrane_center_z = QDoubleSpinBox()
        self.membrane_center_z.setRange(-100.0, 100.0)
        self.membrane_center_z.setDecimals(2)
        self.membrane_center_z.setValue(0.0)
        form.addRow("Membrane center Z (nm):", self.membrane_center_z)

        self.existing_padding = self._nm_spin(1.0)
        form.addRow("Existing-mode padding (nm):", self.existing_padding)

        self.solvent_model = QComboBox()
        self.solvent_model.addItems(sorted(SUPPORTED_WATER_MODELS))
        self.solvent_model.setCurrentText("tip3p")
        form.addRow("Water model:", self.solvent_model)

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

        self.write_intermediates = QCheckBox("Write addMembrane_*.pdb checkpoints")
        form.addRow("", self.write_intermediates)

        self.forcefields = QPlainTextEdit("\n".join(DEFAULT_FORCEFIELDS))
        self.forcefields.setMaximumHeight(120)
        self.forcefields.setToolTip("One system force-field XML per line.")
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

        self._sync_mode(self.mode.currentText())

    def _nm_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.1, 100.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(2)
        spin.setValue(value)
        return spin

    def _sync_mode(self, mode: str) -> None:
        is_build = mode == "build"
        for widget in (
            self.lipid_type, self.scale_factor, self.x_padding, self.y_padding,
            self.z_padding, self.membrane_center_z,
        ):
            widget.setEnabled(is_build)
        self.existing_padding.setEnabled(not is_build)

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

        config: dict = {
            "input_pdb": self.input_pdb.text().strip(),
            "mode": self.mode.currentText(),
            "platform": self.platform.currentText(),
            "output_pdb": self.output_pdb.text().strip() or "membrane_complex.pdb",
            "write_intermediates": self.write_intermediates.isChecked(),
            "solvent_model": self.solvent_model.currentText(),
            "positive_ion": self.positive_ion.text().strip() or "Na+",
            "negative_ion": self.negative_ion.text().strip() or "Cl-",
            "ionic_strength": float(self.ionic_strength.value()),
            "neutralize": self.neutralize.isChecked(),
            "forcefields": forcefields,
            "build": {
                "lipid_type": self.lipid_type.currentText(),
                "scale_factor": float(self.scale_factor.value()),
                "membrane_center_z": float(self.membrane_center_z.value()),
                "x_padding": float(self.x_padding.value()),
                "y_padding": float(self.y_padding.value()),
                "z_padding": float(self.z_padding.value()),
            },
            "existing": {
                "padding": float(self.existing_padding.value()),
            },
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
            QMessageBox.warning(self, "Missing Protein", "Please select the protein PDB.")
            return

        try:
            config = self._build_config()
            task = Path(self.working_directory).expanduser().resolve() / job_name
            task.mkdir(parents=True, exist_ok=True)
            config_path = task / f"{job_name}_membrane_build.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            runner, _ = stage_membrane_build(config_path, task)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Create Job", str(error))
            return

        self.job_requested.emit(str(task), str(runner), str(config_path))
        self.accept()


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = OpenFEMembraneBuildDialog()
    dialog.show()
    sys.exit(app.exec_())
