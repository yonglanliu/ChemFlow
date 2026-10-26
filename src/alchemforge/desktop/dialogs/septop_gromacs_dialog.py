"""Desktop project creator for GROMACS separated-topologies RBFE."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
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
from alchemforge.desktop.dialogs.ligand_network_planner_dialog import (
    LigandNetworkPlannerDialog,
)
from alchemforge.desktop.dialogs.network_viewer_dialog import (
    LigandNetworkViewerDialog,
)
from alchemforge.desktop.modeless import show_modeless
from alchemforge.gromacs.septop.config import SepTopGromacsConfig
from alchemforge.gromacs.septop.workflow import stage


class SepTopGromacsDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self._planned_edges: Path | None = None
        self._planned_ligands: Path | None = None
        self._network_viewer = None
        self.setWindowTitle("GROMACS · LigandSepTop-RBFE Workflow")
        self.resize(820, 650)
        root = QVBoxLayout(self)
        title = QLabel("GROMACS · LigandSepTop-RBFE Workflow")
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")
        note = QLabel(
            "Configure prepared complex/solvent systems and the SepTop protocol "
            "using a saved ligand network bundle. Ligand graph planning and editing "
            "are handled separately. Preparation requires OpenEye."
        )
        note.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(note)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.name = QLineEdit("SepTopRBFE_GROMACS")
        self.library = QLineEdit()
        self.network_bundle = QLineEdit()
        self.prepared = QLineEdit()
        self.source = QLineEdit()
        self.network = QComboBox()
        self.network.addItems(
            ["minimal_spanning", "lomap", "minimal_redundant", "radial"]
        )
        self.replicates = QSpinBox()
        self.replicates.setRange(1, 100)
        self.replicates.setValue(1)
        self.equil = QDoubleSpinBox()
        self.equil.setRange(0.001, 1000)
        self.equil.setValue(1.0)
        self.pressure_equil = QDoubleSpinBox()
        self.pressure_equil.setRange(0.0, 1000)
        self.pressure_equil.setValue(1.0)
        self.production = QDoubleSpinBox()
        self.production.setRange(0.001, 1000)
        self.production.setValue(5.0)
        self.max_jobs = QSpinBox()
        self.max_jobs.setRange(1, 100000)
        self.max_jobs.setValue(3)
        self.partition = QLineEdit("gpu")
        self.nodes = QSpinBox()
        self.nodes.setRange(1, 1000)
        self.nodes.setValue(1)
        self.tasks = QSpinBox()
        self.tasks.setRange(1, 1000)
        self.tasks.setValue(1)
        self.cpus = QSpinBox()
        self.cpus.setRange(1, 512)
        self.cpus.setValue(4)
        self.gpus = QSpinBox()
        self.gpus.setRange(0, 16)
        self.gpus.setValue(1)
        self.memory = QLineEdit("16G")
        self.walltime = QLineEdit("24:00:00")
        self.system_type = QComboBox()
        self.system_type.addItems(["soluble", "membrane"])
        self.complex_gamma = QDoubleSpinBox()
        self.complex_gamma.setRange(0.001, 10.0)
        self.complex_gamma.setValue(0.5)
        self.solvent_gamma = QDoubleSpinBox()
        self.solvent_gamma.setRange(0.001, 10.0)
        self.solvent_gamma.setValue(0.3)
        self.complex_resname = QLineEdit("LIG")
        self.solvent_resname = QLineEdit("UNL")
        self.timestep = QDoubleSpinBox()
        self.timestep.setRange(0.0001, 0.01)
        self.timestep.setDecimals(4)
        self.timestep.setValue(0.002)
        self.allow_charge_change = QCheckBox(
            "Allow only when an explicit charge correction is handled separately"
        )
        form.addRow("Job name:", self.name)
        form.addRow(
            "Ligand network bundle:",
            self._network_directory_row(),
        )
        form.addRow(
            "Prepared ligand root:",
            self._directory_row(self.prepared, "Select prepared ligand root"),
        )
        form.addRow(
            "SeparatedTopologies source:",
            self._directory_row(self.source, "Select MobleyLab source checkout"),
        )
        form.addRow("System type:", self.system_type)

        self.simulation_tabs = QTabWidget()
        stage_page = QWidget()
        stage_layout = QVBoxLayout(stage_page)
        self.stage_protocol_table = QTableWidget(4, 5)
        self.stage_protocol_table.setHorizontalHeaderLabels(
            ["Stage", "Integrator", "dt (ps)", "Temperature", "Pressure"]
        )
        for row, values in enumerate(
            (
                ("EM", "Template", "—", "No coupling", "No coupling"),
                ("NVT", "Template", "0.002", "Template", "No coupling"),
                ("NPT (membrane)", "Template", "0.002", "Template", "Semi-isotropic"),
                ("Production", "Template", "0.002", "Template", "System-dependent"),
            )
        ):
            for column, value in enumerate(values):
                self.stage_protocol_table.setItem(row, column, QTableWidgetItem(value))
        self.stage_protocol_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stage_protocol_table.verticalHeader().setVisible(False)
        self.stage_protocol_table.horizontalHeader().setStretchLastSection(True)
        stage_layout.addWidget(self.stage_protocol_table)
        stage_form = QFormLayout()
        stage_form.addRow("NVT per window (ns):", self.equil)
        stage_form.addRow("Membrane NPT per window (ns):", self.pressure_equil)
        stage_form.addRow("Production per window (ns):", self.production)
        stage_layout.addLayout(stage_form)
        self.simulation_tabs.addTab(stage_page, "Stage Protocol")

        restraint_page = QWidget()
        restraint_layout = QFormLayout(restraint_page)
        restraint_layout.addRow("Complex Boresch gamma:", self.complex_gamma)
        restraint_layout.addRow("Solvent distance gamma:", self.solvent_gamma)
        restraint_layout.addRow("Complex ligand residue:", self.complex_resname)
        restraint_layout.addRow("Solvent ligand residue:", self.solvent_resname)
        self.simulation_tabs.addTab(restraint_page, "Restraints")

        lambda_page = QWidget()
        lambda_layout = QFormLayout(lambda_page)
        lambda_layout.addRow("Replicates:", self.replicates)
        lambda_note = QLabel(
            "Lambda states are read from the selected SeparatedTopologies MDP templates."
        )
        lambda_note.setWordWrap(True)
        lambda_layout.addRow("Schedule source:", lambda_note)
        self.simulation_tabs.addTab(lambda_page, "Lambda Schedule")

        alchemical_page = QWidget()
        alchemical_layout = QFormLayout(alchemical_page)
        alchemical_layout.addRow("Timestep (ps):", self.timestep)
        alchemical_layout.addRow("Charge-changing edges:", self.allow_charge_change)
        self.simulation_tabs.addTab(alchemical_page, "Alchemical Controls")
        form.addRow(self.simulation_tabs)
        form.addRow("Maximum Slurm jobs:", self.max_jobs)
        form.addRow("Slurm partition:", self.partition)
        form.addRow("Nodes per job:", self.nodes)
        form.addRow("Tasks per job:", self.tasks)
        form.addRow("CPUs per task:", self.cpus)
        form.addRow("GPUs per job:", self.gpus)
        form.addRow("Memory per job:", self.memory)
        form.addRow("Walltime:", self.walltime)
        self.tabs = form_step_tabs(
            form,
            (("System", 5), ("FEP Simulation", 1), ("Job Setting", 8)),
        )
        self.ligand_preparation = GromacsLigandPrepareDialog(
            working_directory=self.working_directory,
            parent=self,
            embedded=True,
        )
        self.ligand_preparation.job_requested.connect(self.job_requested.emit)
        self.tabs.insertTab(0, self.ligand_preparation, "Ligand Preparation")
        self.network_planner = LigandNetworkPlannerDialog(
            working_directory=self.working_directory,
            parent=self,
            embedded=True,
        )
        self.network_planner.target.setCurrentText("GROMACS")
        self.network_planner.network_created.connect(self._use_network_bundle)
        self.tabs.insertTab(1, self.network_planner, "Network Creation")
        self.tabs.setCurrentIndex(0)
        root.addWidget(self.tabs, 1)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.show_network_button = QPushButton("Show Network")
        self.show_network_button.setEnabled(False)
        self.show_network_button.clicked.connect(self.show_network)
        create = QPushButton("Create LigandSepTop-RBFE Workflow")
        create.clicked.connect(self.create_project)
        buttons.addWidget(self.show_network_button)
        buttons.addWidget(create)
        root.addLayout(buttons)

    def _open_ligand_preparation(self) -> None:
        self._ligand_preparation_dialog = GromacsLigandPrepareDialog(
            working_directory=self.working_directory,
            parent=self,
        )
        self._ligand_preparation_dialog.job_requested.connect(
            self.job_requested.emit
        )
        show_modeless(self, self._ligand_preparation_dialog)

    def _file_row(self, field: QLineEdit, title: str, file_filter: str) -> QWidget:
        button = QPushButton("Browse...")
        button.clicked.connect(lambda: self._browse_file(field, title, file_filter))
        return self._row(field, button)

    def _directory_row(self, field: QLineEdit, title: str) -> QWidget:
        button = QPushButton("Browse...")
        button.clicked.connect(lambda: self._browse_directory(field, title))
        return self._row(field, button)

    def _network_directory_row(self) -> QWidget:
        button = QPushButton("Browse...")

        def select() -> None:
            value = QFileDialog.getExistingDirectory(
                self, "Select ligand network bundle", self.working_directory
            )
            if value:
                self._use_network_bundle(value)

        button.clicked.connect(select)
        return self._row(self.network_bundle, button)

    def _use_network_bundle(self, path: str) -> None:
        bundle = Path(path)
        self.network_bundle.setText(str(bundle))
        ligand_copy = bundle / "ligands.sdf"
        if ligand_copy.is_file():
            self.library.setText(str(ligand_copy))
        self._planned_edges = bundle / "edges.csv"
        self._planned_ligands = ligand_copy
        self.show_network_button.setEnabled(
            self._planned_edges.is_file() and ligand_copy.is_file()
        )

    @staticmethod
    def _row(field: QLineEdit, button: QPushButton) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _browse_file(self, field: QLineEdit, title: str, file_filter: str) -> None:
        value, _ = QFileDialog.getOpenFileName(
            self, title, self.working_directory, file_filter
        )
        if value:
            field.setText(value)

    def _browse_directory(self, field: QLineEdit, title: str) -> None:
        value = QFileDialog.getExistingDirectory(self, title, self.working_directory)
        if value:
            field.setText(value)

    def _write_config(self) -> Path:
        name = self.name.text().strip()
        if not name:
            raise ValueError("Job name is required")
        network_bundle = self.network_bundle.text().strip()
        if not network_bundle:
            raise ValueError(
                "Select a ligand network bundle created by Ligand Network Planning"
            )
        directory = Path(self.working_directory) / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}_septop.toml"
        path.write_text(
            "[project]\n"
            + f"name = {json.dumps(name)}\nwork_dir = {json.dumps(self.working_directory)}\n\n"
            "[inputs]\n"
            + f"ligand_library = {json.dumps(self.library.text().strip())}\nprepared_root = {json.dumps(self.prepared.text().strip())}\nseptop_source_dir = {json.dumps(self.source.text().strip())}\n\n"
            "[system]\n"
            + f'type = {json.dumps(self.system_type.currentText())}\nmembrane_normal = "z"\ncomplexes_pre_aligned = {str(self.system_type.currentText() == "membrane").lower()}\nreference_pressure_bar = [1.0, 1.0]\ncompressibility_bar = [4.5e-5, 4.5e-5]\nrefcoord_scaling = "com"\n\n'
            "[network]\n"
            + f"existing_network_dir = {json.dumps(network_bundle)}\n"
            + f'method = {json.dumps(self.network.currentText())}\natom_mapper = "lomap"\nmax3d_angstrom = 1.0\nelement_change = false\nrequire_cycle_covering = true\n\n'
            "[protocol]\n"
            + f"complex_gamma = {self.complex_gamma.value()}\n"
            + f"solvent_gamma = {self.solvent_gamma.value()}\n"
            + f"complex_ligand_resname = {json.dumps(self.complex_resname.text().strip())}\n"
            + f"solvent_ligand_resname = {json.dumps(self.solvent_resname.text().strip())}\n"
            + f"allow_charge_change = {str(self.allow_charge_change.isChecked()).lower()}\n"
            + f"replicates = {self.replicates.value()}\nstart_replicate = 1\ntimestep_ps = {self.timestep.value()}\nequilibration_ns = {self.equil.value()}\npressure_equilibration_ns = {self.pressure_equil.value()}\nproduction_ns = {self.production.value()}\n\n"
            '[execution]\ngromacs_command = "gmx"\n'
            + (
                'mdrun_options = ["-ntmpi", "1", "-nb", "gpu", "-pme", "gpu"]\n'
                if self.gpus.value() > 0
                else 'mdrun_options = ["-ntmpi", "1"]\n'
            )
            + f"max_parallel_jobs = {self.max_jobs.value()}\n\n"
            "[slurm]\n"
            + f"partition = {json.dumps(self.partition.text().strip())}\n"
            + f"time = {json.dumps(self.walltime.text().strip())}\n"
            + f"nodes = {self.nodes.value()}\n"
            + f"tasks = {self.tasks.value()}\n"
            + f"gpus = {self.gpus.value()}\n"
            + f"cpus_per_task = {self.cpus.value()}\n"
            + f"memory = {json.dumps(self.memory.text().strip())}\n",
            encoding="utf-8",
        )
        return path

    def create_project(self) -> None:
        try:
            config = SepTopGromacsConfig.load(self._write_config())
            config.validate_inputs()
            result = stage(config)
        except (OSError, TypeError, ValueError, ImportError) as error:
            QMessageBox.critical(self, "SepTop Staging Failed", str(error))
            return
        self._planned_edges = result.network.edges_csv
        self._planned_ligands = config.ligand_library
        self.show_network_button.setEnabled(True)
        QMessageBox.information(
            self,
            "SepTop Project Staged",
            f"Edges: {result.network.edge_count}\nTasks: {result.task_count}\n\n{result.slurm_script}",
        )
        self.show_network()

    def show_network(self) -> None:
        if self._planned_edges is None or self._planned_ligands is None:
            QMessageBox.information(
                self, "No Planned Network", "Plan the ligand network first."
            )
            return
        try:
            self._network_viewer = LigandNetworkViewerDialog(
                self._planned_edges, self._planned_ligands, self
            )
            self._network_viewer.network_saved.connect(self._restage_edited_network)
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Show Network", str(error))
            return
        show_modeless(self, self._network_viewer)
        self._network_viewer.raise_()
        self._network_viewer.activateWindow()

    def _restage_edited_network(self, _path: str) -> None:
        try:
            config = SepTopGromacsConfig.load(self._write_config())
            result = stage(config, reuse_network=True)
        except (OSError, TypeError, ValueError, ImportError) as error:
            QMessageBox.warning(
                self,
                "Network Saved but Restaging Failed",
                f"The edited edge table was saved, but tasks could not be rebuilt: {error}",
            )
            return
        QMessageBox.information(
            self,
            "Edited Network Restaged",
            f"Rebuilt {result.task_count} tasks for {result.network.edge_count} edges.",
        )
