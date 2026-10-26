"""Desktop dialog for GROMACS absolute binding free-energy projects."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from alchemforge.desktop.dialogs.form_tabs import form_step_tabs
from alchemforge.desktop.dialogs.gromacs_ligand_prepare_dialog import (
    GromacsLigandPrepareDialog,
)
from alchemforge.desktop.modeless import show_modeless
from alchemforge.gromacs.abfe.config import ABFEConfig
from alchemforge.gromacs.abfe.workflow import stage


class ABFEGromacsDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("GROMACS · LigandFEP-ABFE Workflow")
        self.resize(820, 760)
        self._build_ui()

    def _file_row(self, field: QLineEdit, title: str, file_filter: str) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        button = QPushButton("Browse...")
        button.clicked.connect(lambda: self._browse(field, title, file_filter))
        layout.addWidget(button)
        container = QWidget()
        container.setLayout(layout)
        return container

    def _open_ligand_preparation(self) -> None:
        self._ligand_preparation_dialog = GromacsLigandPrepareDialog(
            working_directory=self.working_directory,
            parent=self,
        )
        self._ligand_preparation_dialog.job_requested.connect(
            self.job_requested.emit
        )
        show_modeless(self, self._ligand_preparation_dialog)

    def _browse(self, field: QLineEdit, title: str, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, title, self.working_directory, file_filter
        )
        if path:
            field.setText(path)

    def _directory_row(self, field: QLineEdit, title: str) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        button = QPushButton("Browse...")
        button.clicked.connect(
            lambda: field.setText(
                QFileDialog.getExistingDirectory(self, title, self.working_directory)
                or field.text()
            )
        )
        layout.addWidget(button)
        container = QWidget()
        container.setLayout(layout)
        return container

    def _spin(self, value: float, maximum: float = 100000.0) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(0.001, maximum)
        widget.setDecimals(3)
        widget.setValue(value)
        return widget

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        title = QLabel("GROMACS · LigandFEP-ABFE Workflow")
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")
        note = QLabel(
            "Prepare complex and solvent decoupling legs with Boresch restraints. "
            "Local execution uses the GROMACS installed in the active environment; "
            "Slurm is optional."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #71839a;")
        root.addWidget(title)
        root.addWidget(note)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.job_name = QLineEdit("ABFE_GROMACS")
        self.protein = QLineEdit()
        self.ligand = QLineEdit()
        self.resname = QLineEdit("LIG")
        self.charge = QSpinBox()
        self.charge.setRange(-20, 20)
        self.force_field = QLineEdit("amber99sb-ildn")
        self.water_model = QLineEdit("tip3p")
        self.system_type = QComboBox()
        self.system_type.addItems(["soluble", "membrane"])
        self.prepared_setup = QLineEdit()
        form.addRow("Job name:", self.job_name)
        form.addRow(
            "Protein PDB:",
            self._file_row(self.protein, "Select protein", "PDB files (*.pdb)"),
        )
        form.addRow(
            "Ligand SDF:",
            self._file_row(self.ligand, "Select ligand", "SDF files (*.sdf)"),
        )
        form.addRow("Ligand residue name:", self.resname)
        form.addRow("Ligand charge:", self.charge)
        form.addRow("Protein force field:", self.force_field)
        form.addRow("Water model:", self.water_model)
        form.addRow("System type:", self.system_type)
        form.addRow(
            "Prepared ABFE setup:",
            self._directory_row(
                self.prepared_setup, "Select prepared ABFE setup directory"
            ),
        )

        self.k_distance = self._spin(4184.0, 1000000.0)
        self.k_angle = self._spin(41.84)
        self.k_dihedral = self._spin(41.84)

        self.replicas = QSpinBox()
        self.replicas.setRange(1, 1000)
        self.replicas.setValue(1)
        self.windows = QSpinBox()
        self.windows.setRange(2, 1000)
        self.windows.setValue(21)
        self.lambda_function = QComboBox()
        self.lambda_function.addItems(["cosine", "linear", "power"])
        self.lambda_power = self._spin(2.0, 20.0)
        self.equilibration = self._spin(1.0)
        self.production = self._spin(5.0)
        self.decouple_order = QComboBox()
        self.decouple_order.addItems(["Coulomb then vdW", "vdW then Coulomb"])
        self.softcore_alpha = self._spin(0.5, 5.0)

        self.simulation_tabs = QTabWidget()
        stage_page = QWidget()
        stage_layout = QVBoxLayout(stage_page)
        self.stage_protocol_table = QTableWidget(4, 5)
        self.stage_protocol_table.setHorizontalHeaderLabels(
            ["Stage", "Integrator", "dt (ps)", "Temperature", "Pressure"]
        )
        for row, values in enumerate(
            (
                ("EM", "steep", "—", "No coupling", "No coupling"),
                ("NVT", "md", "0.002", "V-rescale", "No coupling"),
                ("NPT", "md", "0.002", "V-rescale", "C-rescale"),
                ("Production", "md", "0.002", "V-rescale", "PR / membrane barostat"),
            )
        ):
            for column, value in enumerate(values):
                self.stage_protocol_table.setItem(row, column, QTableWidgetItem(value))
        self.stage_protocol_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stage_protocol_table.verticalHeader().setVisible(False)
        self.stage_protocol_table.horizontalHeader().setStretchLastSection(True)
        stage_layout.addWidget(self.stage_protocol_table)
        stage_form = QFormLayout()
        stage_form.addRow("Total NVT + NPT equilibration (ns):", self.equilibration)
        stage_form.addRow("Production/window (ns):", self.production)
        stage_layout.addLayout(stage_form)
        self.simulation_tabs.addTab(stage_page, "Stage Protocol")

        restraint_page = QWidget()
        restraint_layout = QFormLayout(restraint_page)
        restraint_layout.addRow("Boresch distance k:", self.k_distance)
        restraint_layout.addRow("Boresch angle k:", self.k_angle)
        restraint_layout.addRow("Boresch dihedral k:", self.k_dihedral)
        self.simulation_tabs.addTab(restraint_page, "Restraints")

        lambda_page = QWidget()
        lambda_layout = QFormLayout(lambda_page)
        lambda_layout.addRow("Replicates:", self.replicas)
        lambda_layout.addRow("Lambda windows:", self.windows)
        lambda_layout.addRow("Lambda function:", self.lambda_function)
        lambda_layout.addRow("Lambda power:", self.lambda_power)
        self.simulation_tabs.addTab(lambda_page, "Lambda Schedule")

        alchemical_page = QWidget()
        alchemical_layout = QFormLayout(alchemical_page)
        alchemical_layout.addRow("Decoupling order:", self.decouple_order)
        alchemical_layout.addRow("Soft-core alpha:", self.softcore_alpha)
        self.simulation_tabs.addTab(alchemical_page, "Alchemical Controls")
        form.addRow(self.simulation_tabs)

        self.execution_mode = QComboBox()
        self.execution_mode.addItems(["local", "slurm"])
        self.execution_mode.currentTextChanged.connect(self._update_execution_ui)
        self.worker = QLineEdit()
        self.partition = QLineEdit("gpu")
        self.cpus = QSpinBox()
        self.cpus.setRange(1, 512)
        self.cpus.setValue(8)
        self.gpus = QSpinBox()
        self.gpus.setRange(0, 16)
        self.gpus.setValue(1)
        self.memory = QLineEdit("16G")
        self.walltime = QLineEdit("24:00:00")
        self.max_jobs = QSpinBox()
        self.max_jobs.setRange(1, 1000000)
        self.max_jobs.setValue(3)
        form.addRow("Execution:", self.execution_mode)
        form.addRow(
            "ABFE worker:",
            self._file_row(self.worker, "Select ABFE worker", "Shell scripts (*.sh)"),
        )
        form.addRow("Slurm partition:", self.partition)
        form.addRow("CPUs per task:", self.cpus)
        form.addRow("GPUs per task:", self.gpus)
        form.addRow("Memory:", self.memory)
        form.addRow("Walltime:", self.walltime)
        form.addRow("Maximum parallel jobs:", self.max_jobs)
        self.slurm_widgets = [
            self.partition,
            self.cpus,
            self.gpus,
            self.memory,
            self.walltime,
        ]
        self.tabs = form_step_tabs(
            form,
            (
                ("System", 9),
                ("FEP Simulation", 1),
                ("Job Setting", 8),
            ),
        )
        self.ligand_preparation = GromacsLigandPrepareDialog(
            working_directory=self.working_directory,
            parent=self,
            embedded=True,
        )
        self.ligand_preparation.job_requested.connect(self.job_requested.emit)
        self.tabs.insertTab(0, self.ligand_preparation, "Ligand Preparation")
        self.tabs.setCurrentIndex(0)
        root.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        create = QPushButton("Create LigandFEP-ABFE Workflow")
        create.clicked.connect(self.create_task)
        run = QPushButton("Run")
        run.clicked.connect(self.run_task)
        buttons.addWidget(create)
        buttons.addWidget(run)
        root.addLayout(buttons)
        self._update_execution_ui()

    def _update_execution_ui(self) -> None:
        enabled = self.execution_mode.currentText() == "slurm"
        for widget in self.slurm_widgets:
            widget.setEnabled(enabled)

    def _write_config(self) -> Path:
        name = self.job_name.text().strip()
        if not name:
            raise ValueError("Job name is required")
        project = Path(self.working_directory) / name
        project.mkdir(parents=True, exist_ok=True)
        path = project / f"{name}_abfe_gromacs.toml"
        worker = self.worker.text().strip()
        text = (
            f"""
[project]
name = {json.dumps(name)}
work_dir = {json.dumps(self.working_directory)}

[system]
type = {json.dumps(self.system_type.currentText())}
protein = {json.dumps(self.protein.text().strip())}
gromacs_command = "gmx"
force_field = {json.dumps(self.force_field.text().strip())}
water_model = {json.dumps(self.water_model.text().strip())}
box_distance = 1.2
ion_concentration = 0.15
temperature_k = 300.0
pressure_bar = 1.0
{f"prepared_setup_dir = {json.dumps(self.prepared_setup.text().strip())}" if self.prepared_setup.text().strip() else ""}
membrane_normal = "z"
reference_pressure_bar = [1.0, 1.0]
compressibility_bar = [4.5e-5, 4.5e-5]
refcoord_scaling = "com"

[ligand]
sdf = {json.dumps(self.ligand.text().strip())}
resname = {json.dumps(self.resname.text().strip())}
charge = {self.charge.value()}

[topology]
backend = "pmx"
charge_method = "bcc"

[restraints]
method = "boresch"
seed = 2026
distance_force_constant = {self.k_distance.value()}
angle_force_constant = {self.k_angle.value()}
dihedral_force_constant = {self.k_dihedral.value()}

[simulation]
replicas = {self.replicas.value()}
lambda_windows = {self.windows.value()}
lambda_function = {json.dumps(self.lambda_function.currentText())}
lambda_power = {self.lambda_power.value()}
timestep_ps = 0.002
equilibration_ns = {self.equilibration.value()}
production_ns = {self.production.value()}
decouple_coulomb_first = {str(self.decouple_order.currentIndex() == 0).lower()}
softcore_alpha = {self.softcore_alpha.value()}

[execution]
mode = {json.dumps(self.execution_mode.currentText())}
max_parallel_jobs = {self.max_jobs.value()}
mdrun_options = {json.dumps(["-ntmpi", "1", "-nb", "gpu", "-pme", "gpu"] if self.gpus.value() > 0 else ["-ntmpi", "1"])}
{f"worker_script = {json.dumps(worker)}" if worker else ""}

[slurm]
partition = {json.dumps(self.partition.text().strip())}
nodes = 1
tasks = 1
cpus_per_task = {self.cpus.value()}
gpus = {self.gpus.value()}
memory = {json.dumps(self.memory.text().strip())}
walltime = {json.dumps(self.walltime.text().strip())}
""".strip()
            + "\n"
        )
        path.write_text(text, encoding="utf-8")
        return path

    def create_task(self) -> None:
        try:
            config = ABFEConfig.load(self._write_config())
            config.validate_inputs()
            generated = stage(config)
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.critical(self, "ABFE Task Creation Failed", str(error))
            return
        QMessageBox.information(
            self,
            "ABFE Task Created",
            f"Project: {config.project_dir}\nConfiguration: {generated}",
        )

    def run_task(self) -> None:
        try:
            config = ABFEConfig.load(self._write_config())
            config.validate_inputs(require_worker=True)
            generated = stage(config)
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Run ABFE", str(error))
            return
        self.job_requested.emit(
            str(config.project_dir),
            str(config.project_dir / "run_abfe.sh"),
            str(generated),
        )
        self.accept()
