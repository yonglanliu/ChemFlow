"""Desktop dialog for the OpenFE MD runner (soluble or membrane)."""

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

from alchemforge.openmm_modeling.md_config import (
    DEFAULT_FORCEFIELDS,
    SUPPORTED_SYSTEM_TYPES,
    stage as stage_md,
)


class OpenFEMDDialog(QDialog):
    """Run unrestrained MD (min/NVT/NPT/production) for a soluble or membrane system."""

    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("OpenMM · MD Simulation")
        self.resize(820, 640)

        root = QVBoxLayout(self)
        title = QLabel("MD Simulation (OpenMM)")
        title.setStyleSheet("font-size:18pt; font-weight:700; color:#123f76;")
        root.addWidget(title)
        note = QLabel(
            "Unrestrained MD on a solvated/relaxed complex: minimization -> NVT -> "
            "NPT -> production. system_type picks the barostat (soluble = isotropic; "
            "membrane = NPgT, XY-isotropic/Z-free). Input must carry a periodic box."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.job_name = QLineEdit("md")
        form.addRow("Job name:", self.job_name)

        self.input_pdb = QLineEdit()
        self.input_pdb.setPlaceholderText("Solvated/relaxed complex PDB (with a box)")
        form.addRow("Input PDB:", self._file_row(self.input_pdb, "Select PDB", "PDB (*.pdb)"))

        self.ligand_sdf = QLineEdit()
        self.ligand_sdf.setPlaceholderText("Optional — parameterize a ligand in the PDB")
        form.addRow("Ligand SDF:", self._file_row(self.ligand_sdf, "Select ligand SDF", "SDF (*.sdf)"))

        self.small_molecule_ff = QLineEdit("gaff-2.11")
        form.addRow("Ligand force field:", self.small_molecule_ff)

        self.system_type = QComboBox()
        self.system_type.addItems(sorted(SUPPORTED_SYSTEM_TYPES))
        self.system_type.setCurrentText("soluble")
        self.system_type.currentTextChanged.connect(self._sync_type)
        form.addRow("System type:", self.system_type)

        self.platform = QComboBox()
        self.platform.addItems(["CUDA", "OpenCL", "CPU"])
        form.addRow("OpenMM platform:", self.platform)

        self.temperature = self._spin(300.0, 1.0, 1000.0, 2)
        form.addRow("Temperature (K):", self.temperature)
        self.pressure = self._spin(1.0, 0.001, 1000.0, 3)
        form.addRow("Pressure (bar):", self.pressure)
        self.surface_tension = self._spin(0.0, -1000.0, 1000.0, 3)
        form.addRow("Surface tension (bar·nm):", self.surface_tension)
        self.friction = self._spin(1.0, 0.001, 1000.0, 3)
        form.addRow("Langevin friction (1/ps):", self.friction)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(1)
        form.addRow("Random seed:", self.seed)

        self.min_steps = QSpinBox()
        self.min_steps.setRange(1, 100_000_000)
        self.min_steps.setValue(10_000)
        form.addRow("Minimization steps:", self.min_steps)

        self.nvt_time = self._spin(250.0, 0.1, 1_000_000.0, 1)
        form.addRow("NVT equil time (ps):", self.nvt_time)
        self.npt_time = self._spin(500.0, 0.1, 1_000_000.0, 1)
        form.addRow("NPT equil time (ps):", self.npt_time)
        self.equil_timestep = self._spin(0.002, 0.0001, 0.01, 4)
        form.addRow("Equil timestep (ps):", self.equil_timestep)

        self.prod_time = self._spin(10000.0, 0.1, 100_000_000.0, 1)
        form.addRow("Production time (ps):", self.prod_time)
        self.prod_timestep = self._spin(0.004, 0.0001, 0.01, 4)
        form.addRow("Production timestep (ps):", self.prod_timestep)

        self.report_interval = QSpinBox()
        self.report_interval.setRange(1, 100_000_000)
        self.report_interval.setValue(5000)
        form.addRow("Report interval (steps):", self.report_interval)

        self.forcefields = QPlainTextEdit("\n".join(DEFAULT_FORCEFIELDS))
        self.forcefields.setMaximumHeight(100)
        form.addRow("System force fields:", self.forcefields)

        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        create = QPushButton("Create MD Job")
        create.clicked.connect(self.create_job)
        buttons.addWidget(cancel)
        buttons.addWidget(create)
        root.addLayout(buttons)

        self._sync_type(self.system_type.currentText())

    def _spin(self, value, lo, hi, decimals) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _sync_type(self, system_type: str) -> None:
        self.surface_tension.setEnabled(system_type == "membrane")

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
            "system_type": self.system_type.currentText(),
            "platform": self.platform.currentText(),
            "friction": float(self.friction.value()),
            "seed": int(self.seed.value()),
            "temperature": float(self.temperature.value()),
            "pressure": float(self.pressure.value()),
            "surface_tension": float(self.surface_tension.value()),
            "report_interval": int(self.report_interval.value()),
            "output_prefix": self.job_name.text().strip() or "md",
            "forcefields": forcefields,
            "minimization": {"steps": int(self.min_steps.value())},
            "nvt": {
                "timestep": float(self.equil_timestep.value()),
                "simulation_time": float(self.nvt_time.value()),
            },
            "npt": {
                "timestep": float(self.equil_timestep.value()),
                "simulation_time": float(self.npt_time.value()),
            },
            "production": {
                "timestep": float(self.prod_timestep.value()),
                "simulation_time": float(self.prod_time.value()),
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
            QMessageBox.warning(self, "Missing Input", "Please select the input PDB.")
            return
        try:
            config = self._build_config()
            task = Path(self.working_directory).expanduser().resolve() / job_name
            task.mkdir(parents=True, exist_ok=True)
            config_path = task / f"{job_name}_md.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            runner, _ = stage_md(config_path, task)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Create Job", str(error))
            return
        self.job_requested.emit(str(task), str(runner), str(config_path))
        self.accept()


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = OpenFEMDDialog()
    dialog.show()
    sys.exit(app.exec_())
