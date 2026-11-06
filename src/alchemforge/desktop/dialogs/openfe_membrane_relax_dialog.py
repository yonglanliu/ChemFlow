"""Desktop dialog for the OpenFE membrane relaxation protocol."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from alchemforge.openmm_modeling.membrane_relax_config import (
    DEFAULT_FORCEFIELDS,
    SUPPORTED_PLATFORMS,
    stage as stage_membrane_relax,
)


class OpenFEMembraneRelaxDialog(QDialog):
    """Configure and launch a CHARMM-GUI-style membrane relaxation (OpenMM)."""

    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("OpenMM · Membrane Relaxation")
        self.resize(820, 640)

        root = QVBoxLayout(self)
        title = QLabel("Membrane Relaxation (CHARMM-GUI-style, OpenMM)")
        title.setStyleSheet("font-size:18pt; font-weight:700; color:#123f76;")
        root.addWidget(title)
        note = QLabel(
            "Staged, restrained pre-equilibration of a PRE-TRIMMED "
            "membrane+protein(+ligand) complex before OpenFE FEP. The input PDB's "
            "periodic box is preserved (no padding is added), so a minimal "
            "membrane patch stays small."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.job_name = QLineEdit("membrane_relax")
        form.addRow("Job name:", self.job_name)

        self.input_pdb = QLineEdit()
        self.input_pdb.setPlaceholderText("Prepared complex PDB (must contain a CRYST1 box)")
        form.addRow("Complex PDB:", self._file_row(self.input_pdb, "Select complex PDB", "PDB (*.pdb)"))

        self.ligand_sdf = QLineEdit()
        self.ligand_sdf.setPlaceholderText("Optional — parameterize ligand atoms in the complex")
        form.addRow("Ligand SDF:", self._file_row(self.ligand_sdf, "Select ligand SDF", "SDF (*.sdf)"))

        self.small_molecule_ff = QLineEdit("gaff-2.11")
        self.small_molecule_ff.setToolTip(
            "Ligand force field: a GAFF line (gaff-2.11) or an OpenFF SMIRNOFF "
            "line (openff-2.1.0). Only used when a ligand SDF is provided."
        )
        form.addRow("Ligand force field:", self.small_molecule_ff)

        self.platform = QComboBox()
        self.platform.addItems(["CUDA", "OpenCL", "CPU"])
        form.addRow("OpenMM platform:", self.platform)

        self.forcefields = QPlainTextEdit("\n".join(DEFAULT_FORCEFIELDS))
        self.forcefields.setMaximumHeight(120)
        self.forcefields.setToolTip("One system force-field XML per line.")
        form.addRow("System force fields:", self.forcefields)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(1.0, 1000.0)
        self.temperature.setDecimals(2)
        self.temperature.setValue(300.0)
        form.addRow("Temperature (K):", self.temperature)

        self.friction = QDoubleSpinBox()
        self.friction.setRange(0.001, 1000.0)
        self.friction.setDecimals(3)
        self.friction.setValue(1.0)
        form.addRow("Langevin friction (1/ps):", self.friction)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(1)
        form.addRow("Random seed:", self.seed)

        self.minimization_steps = QSpinBox()
        self.minimization_steps.setRange(1, 100_000_000)
        self.minimization_steps.setValue(10_000)
        form.addRow("Minimization steps:", self.minimization_steps)

        self.production_time = QDoubleSpinBox()
        self.production_time.setRange(0.1, 1_000_000.0)
        self.production_time.setDecimals(1)
        self.production_time.setValue(5000.0)
        self.production_time.setToolTip("Final unrestrained NPgT production length (ps).")
        form.addRow("Production time (ps):", self.production_time)

        root.addLayout(form)

        hint = QLabel(
            "The six-stage CHARMM-GUI restraint ladder and the Stage-4 annealing "
            "schedule use built-in defaults; edit the generated YAML to tune them."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555;")
        root.addWidget(hint)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        create = QPushButton("Create Relaxation Job")
        create.clicked.connect(self.create_job)
        buttons.addWidget(cancel)
        buttons.addWidget(create)
        root.addLayout(buttons)

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
        if self.platform.currentText() not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Unsupported platform: {self.platform.currentText()}")

        temperature = float(self.temperature.value())
        config: dict = {
            "input_pdb": self.input_pdb.text().strip(),
            "platform": self.platform.currentText(),
            "friction": float(self.friction.value()),
            "seed": int(self.seed.value()),
            "forcefields": forcefields,
            "minimization": {"steps": int(self.minimization_steps.value())},
            "simulation": {
                "simulation_time": float(self.production_time.value()),
                "temperature": temperature,
            },
        }
        # Apply the chosen temperature to every equilibration stage; leave the
        # Stage-4 annealing schedule to the YAML defaults (editable afterwards).
        for stage_key in ("stage1", "stage2", "stage3", "stage5", "stage6"):
            config[stage_key] = {"temperature": temperature}
        config["stage4"] = {"temperature": temperature}

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
            QMessageBox.warning(
                self, "Missing Complex", "Please select the prepared complex PDB."
            )
            return

        try:
            config = self._build_config()
            task = Path(self.working_directory).expanduser().resolve() / job_name
            task.mkdir(parents=True, exist_ok=True)
            config_path = task / f"{job_name}_membrane_relax.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            runner, _ = stage_membrane_relax(config_path, task)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Create Job", str(error))
            return

        self.job_requested.emit(str(task), str(runner), str(config_path))
        self.accept()


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = OpenFEMembraneRelaxDialog()
    dialog.show()
    sys.exit(app.exec_())
