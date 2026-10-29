"""Desktop setup dialog for multi-ligand GROMACS RBFE projects."""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import shutil
import sys
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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from alchemforge.desktop.dialogs.asl_restraint_dialog import (
    ASLRestraintDialog,
    parse_restraint_asl,
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
from alchemforge.gromacs.lambda_schedule import generate_lambda_windows
from alchemforge.gromacs.membrane import count_lipid_molecules_pdb
from alchemforge.gromacs.rbfe.config import LigandRBFEConfig
from alchemforge.gromacs.rbfe.workflow import stage, write_slurm_array_files
from alchemforge.gromacs.system_prepare import (
    GromacsSystemPreparationConfig,
    installed_amber_lipid_force_fields,
    installed_gromacs_force_fields,
    stage_gromacs_system,
)


class LigandRBFEGromacsDialog(QDialog):
    """Configure a GROMACS RBFE system using a saved ligand network."""

    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self._planned_edges: Path | None = None
        self._planned_ligands: Path | None = None
        self._network_viewer = None
        self._stage_restraint_rules = self._default_stage_restraint_rules()
        self._restraint_editors: dict[str, ASLRestraintDialog] = {}
        self.setWindowTitle("GROMACS · LigandFEP-RBFE Workflow")
        self.resize(900, 780)
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

    def _network_directory_row(self) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.network_bundle)
        button = QPushButton("Browse...")

        def browse() -> None:
            path = QFileDialog.getExistingDirectory(
                self, "Select ligand network bundle", self.working_directory
            )
            if path:
                self._use_network_bundle(path)

        button.clicked.connect(browse)
        layout.addWidget(button)
        container = QWidget()
        container.setLayout(layout)
        return container

    def _use_network_bundle(self, path: str) -> None:
        bundle = Path(path)
        self.network_bundle.setText(str(bundle))
        ligand_copy = bundle / "ligands.sdf"
        self._planned_edges = bundle / "edges.csv"
        self._planned_ligands = ligand_copy
        self.show_network_button.setEnabled(
            self._planned_edges.is_file() and ligand_copy.is_file()
        )
        self._update_expected_array_tasks()

    def _update_expected_array_tasks(self, *_args) -> None:
        """Show edges × legs × replicates × lambda windows for the array."""
        bundle_text = self.network_bundle.text().strip()
        edges_path = Path(bundle_text) / "edges.csv" if bundle_text else None
        edge_count = 0
        if edges_path is not None and edges_path.is_file():
            try:
                with edges_path.open(newline="", encoding="utf-8") as handle:
                    edge_count = sum(1 for _row in csv.DictReader(handle))
            except OSError:
                edge_count = 0
        if self.lambda_schedule_mode.currentText() == "manual":
            windows = len(
                [
                    token
                    for token in re.split(
                        r"[\s,;]+", self.coul_lambdas.text().strip()
                    )
                    if token
                ]
            )
        else:
            windows = self.lambda_windows.value()
        if edge_count == 0 or windows == 0:
            self.expected_array_tasks.setText(
                "Select/create a network and define the lambda schedule"
            )
            return
        if self.sampling_method.currentData() == "hrex":
            total = edge_count * 2 * self.replicates.value()
            last_replicate = (
                self.start_replicate.value() + self.replicates.value() - 1
            )
            self.expected_array_tasks.setText(
                f"{edge_count} edges × 2 legs × {self.replicates.value()} "
                f"replicates (rep_{self.start_replicate.value()}–"
                f"rep_{last_replicate}) = {total} HREX groups; "
                f"{windows} lambda replicas/group; "
                f"array 0-{total - 1}"
            )
        else:
            total = edge_count * 2 * self.replicates.value() * windows
            last_replicate = (
                self.start_replicate.value() + self.replicates.value() - 1
            )
            self.expected_array_tasks.setText(
                f"{edge_count} edges × 2 legs × {self.replicates.value()} replicates "
                f"(rep_{self.start_replicate.value()}–rep_{last_replicate}) "
                f"× {windows} windows = {total} tasks; array 0-{total - 1}"
            )

    def _update_sampling_method(self, *_args) -> None:
        hrex = self.sampling_method.currentData() == "hrex"
        self.exchange_interval.setEnabled(hrex)
        self.exchange_interval.setVisible(hrex)
        self.exchange_interval_label.setVisible(hrex)
        self.ntmpi.setEnabled(not hrex)
        self.hrex_module.setEnabled(hrex)
        self.hrex_mpi_launcher.setEnabled(hrex)
        self.hrex_gmx_command.setEnabled(hrex)
        self.tasks.setEnabled(not hrex)
        self.tasks.setValue(self.lambda_windows.value() if hrex else 1)
        self._update_expected_array_tasks()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        title = QLabel("GROMACS · LigandFEP-RBFE Workflow")
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")
        note = QLabel(
            "Prepare ligands, create their alchemical network, and configure the "
            "protein or membrane system, GROMACS protocol, lambda schedule, and "
            "execution settings in order."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #71839a;")
        root.addWidget(title)
        root.addWidget(note)
        parameter_help_note = QLabel(
            "ⓘ Hover over any parameter to see what it controls and the units used."
        )
        parameter_help_note.setStyleSheet("color:#52708f; font-style:italic")
        root.addWidget(parameter_help_note)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.job_name = QLineEdit("LigRBFE_GROMACS")
        self.protein_path = QLineEdit()
        self.protein_path.textChanged.connect(self._synchronize_system_type_with_input)
        self.ligand_library = QLineEdit()
        self.ligand_library.setPlaceholderText(
            "Folder created by GROMACS Ligand Preparation"
        )
        self.network_bundle = QLineEdit()
        self.force_field_family = QComboBox()
        self.force_field_family.addItem("Amber", "amber")
        installed = installed_gromacs_force_fields()
        charmm36m = "charmm36m"
        status = "installed" if charmm36m in installed else "not installed"
        self.force_field_family.addItem(f"CHARMM36m ({status})", charmm36m)
        self.force_field = QComboBox()
        self.force_field.addItems(["ff14SB", "ff19SB"])
        self.force_field_table = QTableWidget(0, 3)
        self.force_field_table.setHorizontalHeaderLabels(
            ["Component", "Force field", "Lipid parameters"]
        )
        self.force_field_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.force_field_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.force_field_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.force_field_table.verticalHeader().setVisible(False)
        self.force_field_table.horizontalHeader().setStretchLastSection(True)
        self.force_field_table.setMaximumHeight(165)
        self.force_field_table.setToolTip(
            "Select the PMX mutation force field. Amber entries can use Lipid17 or Lipid21; "
            "CHARMM36m uses the lipid definitions included in its .ff bundle."
        )
        self.add_force_field_button = QPushButton("Add Force Field")
        self.delete_force_field_button = QPushButton("Delete Force Field")
        self.add_force_field_button.clicked.connect(self._add_force_field)
        self.delete_force_field_button.clicked.connect(self._delete_force_field)
        self.force_field_discovery_status = QLabel()
        self.force_field_discovery_status.setWordWrap(True)
        self._populate_force_field_table()
        self.lipid_force_field = QComboBox()
        self.lipid_force_field.addItems(
            installed_amber_lipid_force_fields()
            + ["Included in selected CHARMM .ff"]
        )
        self.nucleic_acid_force_field = QComboBox()
        self.nucleic_acid_force_field.addItems(
            [
                "DNA OL21 + RNA OL3",
                "DNA bsc1 + RNA OL3",
                "Included in selected CHARMM .ff",
                "None",
            ]
        )
        self.water_model = QComboBox()
        self.water_model.addItems(
            [
                "TIP3P",
                "OPC",
                "OPC3",
                "SPC/E",
                "TIP4P-Ew",
            ]
        )
        self.water_model.setToolTip(
            "Only solvent models supplied by the selected force-field family are "
            "shown. Matching water and ion parameters are selected together and "
            "the resulting topology is converted to GROMACS format."
        )
        self.ion_force_field = QComboBox()
        self.ion_force_field.addItems(
            ["Joung-Cheatham TIP3P", "Included in selected CHARMM .ff"]
        )
        self.ligand_force_field = QComboBox()
        self.ligand_force_field.addItems(["GAFF2", "CGenFF topology required"])
        self.system_type = QComboBox()
        self.system_type.addItems(["soluble", "membrane"])
        self.override_membrane_box = QCheckBox("Override membrane periodic box")
        self.override_membrane_box.setToolTip(
            "Replace the final GRO box without moving or scaling coordinates. "
            "Use the box from the original membrane simulation."
        )
        self.membrane_box_lengths = QLineEdit()
        self.membrane_box_lengths.setPlaceholderText(
            "X Y Z in nm, e.g. 12.5 12.5 15.0"
        )
        self.membrane_box_angles = QLineEdit("90 90 90")
        self.membrane_box_angles.setPlaceholderText(
            "alpha beta gamma in degrees"
        )
        self.box_shape = QComboBox()
        self.box_shape.addItems(["dodecahedron", "cubic", "octahedron"])
        self.box_shape.setToolTip(
            "Applied to the ligand-only solvent leg and to the complex leg for "
            "soluble systems. A prepared membrane complex keeps its existing box."
        )
        self.box_distance = QDoubleSpinBox()
        self.box_distance.setRange(0.1, 10.0)
        self.box_distance.setDecimals(2)
        self.box_distance.setSingleStep(0.1)
        self.box_distance.setValue(1.2)
        self.box_distance.setSuffix(" nm")
        self.neutralize_system = QCheckBox("Neutralize before adding salt")
        self.neutralize_system.setChecked(True)
        self.ion_strength = QDoubleSpinBox()
        self.ion_strength.setRange(0.0, 5.0)
        self.ion_strength.setDecimals(3)
        self.ion_strength.setSingleStep(0.05)
        self.ion_strength.setValue(0.15)
        self.ion_strength.setSuffix(" M")
        self.prepared_complex_gro = QLineEdit()
        self.prepared_complex_topology = QLineEdit()
        self._prepared_ligands: list[dict[str, str]] = []
        self.charge_change_method = QComboBox()
        self.charge_change_method.addItem("Co-ion (preserve cell charge)", "coion")
        self.charge_change_method.addItem("Correction table", "correction")
        self.charge_change_method.addItem("Reject charge-changing edges", "reject")
        self.charge_change_method.setToolTip(
            "Uses the assigned integer net charge for each prepared ligand. "
            "Co-ion stages a physical-ion to neutral-ion topology for both legs; "
            "Correction table requires validated leg-specific corrections."
        )
        self.charge_correction_file = QLineEdit()
        self.cofactor_sdf = QLineEdit()
        self.prepared_cofactor_directory: Path | None = None
        self.prepared_cofactor_itp: Path | None = None
        form.addRow("Job name:", self.job_name)
        form.addRow(
            "Protein PDB:",
            self._file_row(
                self.protein_path, "Select protein PDB", "PDB files (*.pdb)"
            ),
        )
        form.addRow("Ligand network bundle:", self._network_directory_row())
        force_fields = QGroupBox("Compatible force-field components")
        force_fields.setToolTip(
            "Select one protein/mutation force-field bundle. The lipid and "
            "small-molecule components are shown on the same list."
        )
        force_field_layout = QGridLayout(force_fields)
        force_field_layout.addWidget(self.force_field_table, 0, 0, 1, 4)
        force_field_buttons = QHBoxLayout()
        force_field_buttons.addStretch()
        force_field_buttons.addWidget(self.add_force_field_button)
        force_field_buttons.addWidget(self.delete_force_field_button)
        force_field_layout.addLayout(force_field_buttons, 1, 0, 1, 4)
        force_field_layout.addWidget(self.force_field_discovery_status, 2, 0, 1, 4)
        force_field_note = QLabel(
            "Amber entries use TIP3P/Joung–Cheatham ions, GAFF2 ligands, and "
            "Lipid17 or Lipid21. CHARMM36m uses the matching components included in its bundle."
        )
        force_field_note.setWordWrap(True)
        force_field_note.setStyleSheet("color:#6b7280")
        force_field_layout.addWidget(force_field_note, 3, 0, 1, 4)
        form.addRow(force_fields)
        form.addRow("System type:", self.system_type)
        system_builder = QGroupBox("System preparation")
        system_builder_layout = QGridLayout(system_builder)
        system_builder_layout.addWidget(QLabel("Prepared ligand folder:"), 0, 0)
        system_builder_layout.addWidget(self._prepared_ligand_row(), 0, 1, 1, 3)
        system_builder_layout.addWidget(QLabel("Solvent model:"), 1, 0)
        system_builder_layout.addWidget(self.water_model, 1, 1, 1, 3)
        system_builder_layout.addWidget(QLabel("Box shape:"), 2, 0)
        system_builder_layout.addWidget(self.box_shape, 2, 1)
        system_builder_layout.addWidget(QLabel("Box padding:"), 2, 2)
        system_builder_layout.addWidget(self.box_distance, 2, 3)
        system_builder_layout.addWidget(self.neutralize_system, 3, 1, 1, 3)
        system_builder_layout.addWidget(QLabel("Ion strength:"), 4, 0)
        system_builder_layout.addWidget(self.ion_strength, 4, 1)
        system_builder_layout.addWidget(QLabel("Charge-changing edges:"), 5, 0)
        system_builder_layout.addWidget(self.charge_change_method, 5, 1, 1, 3)
        self.charge_correction_label = QLabel("Correction CSV:")
        system_builder_layout.addWidget(self.charge_correction_label, 6, 0)
        self.charge_correction_row = self._file_row(
            self.charge_correction_file,
            "Select charge correction table",
            "CSV files (*.csv)",
        )
        system_builder_layout.addWidget(self.charge_correction_row, 6, 1, 1, 3)
        system_builder_layout.addWidget(self.override_membrane_box, 7, 0, 1, 4)
        system_builder_layout.addWidget(QLabel("Membrane box lengths:"), 8, 0)
        system_builder_layout.addWidget(self.membrane_box_lengths, 8, 1)
        system_builder_layout.addWidget(QLabel("Box angles:"), 8, 2)
        system_builder_layout.addWidget(self.membrane_box_angles, 8, 3)
        self.prepare_system_button = QPushButton("Prepare / Validate System")
        self.prepare_system_button.clicked.connect(self._prepare_system)
        system_builder_layout.addWidget(self.prepare_system_button, 9, 2, 1, 2)
        self.system_build_status = QLabel("Not prepared")
        self.system_build_status.setStyleSheet("color:#71839a")
        system_builder_layout.addWidget(self.system_build_status, 9, 0, 1, 2)
        self.charge_change_method.currentIndexChanged.connect(
            self._update_charge_correction_controls
        )
        self.system_type.currentTextChanged.connect(self._update_box_override_controls)
        self.override_membrane_box.toggled.connect(self._update_box_override_controls)
        self._update_charge_correction_controls()
        self._update_box_override_controls()
        form.addRow(system_builder)

        self.start_replicate = QSpinBox()
        self.start_replicate.setRange(1, 1_000_000)
        self.start_replicate.setValue(1)
        self.replicates = QSpinBox()
        self.replicates.setRange(1, 1000)
        self.replicates.setValue(1)
        self.sampling_method = QComboBox()
        self.sampling_method.addItem("Independent lambda windows", "independent")
        self.sampling_method.addItem("Lambda Hamiltonian replica exchange", "hrex")
        self.sampling_method.setToolTip(
            "Independent mode submits every lambda window separately. Lambda-HREX "
            "coordinates all lambda states with GROMACS -multidir and -replex and "
            "requires an external-MPI gmx_mpi executable."
        )
        self.exchange_interval = QSpinBox()
        self.exchange_interval.setRange(1, 10_000_000)
        self.exchange_interval.setValue(500)
        self.exchange_interval.setSuffix(" steps")
        self.exchange_interval.setToolTip(
            "Attempt exchanges between neighboring lambda Hamiltonians at this "
            "production-step interval."
        )
        self.lambda_windows = QSpinBox()
        self.lambda_windows.setRange(2, 1000)
        self.lambda_windows.setValue(21)
        self.lambda_schedule_mode = QComboBox()
        self.lambda_schedule_mode.addItems(["generated", "manual"])
        self.lambda_schedule_mode.setToolTip(
            "Generated uses a linear, cosine, or power progression and stages "
            "vdW/bonded changes before Coulomb changes. Manual accepts four "
            "explicit lists of values between 0 and 1."
        )
        self.lambda_function = QComboBox()
        self.lambda_function.addItems(["cosine", "linear", "power"])
        self.lambda_power = QDoubleSpinBox()
        self.lambda_power.setRange(0.01, 20.0)
        self.lambda_power.setDecimals(3)
        self.lambda_power.setValue(2.0)
        self.coul_lambdas = QLineEdit()
        self.vdw_lambdas = QLineEdit()
        self.bonded_lambdas = QLineEdit()
        self.restraint_lambdas = QLineEdit()
        for field in (
            self.coul_lambdas,
            self.vdw_lambdas,
            self.bonded_lambdas,
            self.restraint_lambdas,
        ):
            field.setPlaceholderText("Space- or comma-separated values, e.g. 0, 0.5, 1")
        self.lambda_preview = QPlainTextEdit()
        self.lambda_preview.setReadOnly(True)
        self.lambda_preview.setMaximumHeight(112)
        self.timestep = QDoubleSpinBox()
        self.timestep.setRange(0.0001, 0.01)
        self.timestep.setDecimals(4)
        self.timestep.setValue(0.002)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(1.0, 1000.0)
        self.temperature.setDecimals(2)
        self.temperature.setSingleStep(1.0)
        self.temperature.setValue(300.0)
        self.temperature.setSuffix(" K")
        self.em_steps = QSpinBox()
        self.em_steps.setRange(1, 100_000_000)
        self.em_steps.setValue(5_000)
        self.em_tolerance = QDoubleSpinBox()
        self.em_tolerance.setRange(0.001, 1_000_000.0)
        self.em_tolerance.setDecimals(3)
        self.em_tolerance.setValue(200.0)
        self.nvt_duration = QDoubleSpinBox()
        self.nvt_duration.setRange(0.001, 100000.0)
        self.nvt_duration.setDecimals(3)
        self.nvt_duration.setValue(0.5)
        self.npt_duration = QDoubleSpinBox()
        self.npt_duration.setRange(0.001, 100000.0)
        self.npt_duration.setDecimals(3)
        self.npt_duration.setValue(1.0)
        self.transition_duration = QDoubleSpinBox()
        self.transition_duration.setRange(0.001, 100000.0)
        self.transition_duration.setDecimals(3)
        self.transition_duration.setValue(0.5)
        self.final_equilibration_duration = QDoubleSpinBox()
        self.final_equilibration_duration.setRange(0.001, 100000.0)
        self.final_equilibration_duration.setDecimals(3)
        self.final_equilibration_duration.setValue(0.5)
        self.production = QDoubleSpinBox()
        self.production.setRange(0.001, 100000.0)
        self.production.setValue(5.0)
        self.partition = QLineEdit("gpu")
        self.nodes = QSpinBox()
        self.nodes.setRange(1, 1000)
        self.nodes.setValue(1)
        self.tasks = QSpinBox()
        self.tasks.setRange(1, 1000)
        self.tasks.setValue(1)
        self.cpus = QSpinBox()
        self.cpus.setRange(1, 512)
        self.cpus.setValue(8)
        self.ntmpi = QSpinBox()
        self.ntmpi.setRange(1, 512)
        self.ntmpi.setValue(1)
        self.ntmpi.setToolTip(
            "GROMACS thread-MPI ranks inside each array task; one is recommended "
            "for one GPU per alchemical window."
        )
        self.hrex_gmx_command = QLineEdit("gmx_mpi")
        self.hrex_gmx_command.setToolTip(
            "External-MPI GROMACS executable used only for lambda-HREX. System "
            "preparation continues to use gmx."
        )
        self.hrex_module = QComboBox()
        self.hrex_module.setEditable(True)
        self.hrex_module.addItems(
            [
                "",
                "gromacs/2026.3-gcc13.2",
                "gromacs/2026.3-intel2024",
                "gromacs/2024.4-gcc11.3",
                "gromacs/2024.4-intel2022",
                "gromacs/2024.4+plumed2.9.3",
                "gromacs/2024.1-intel2024",
                "gromacs/2024-intel2022",
                "gromacs/2023.2-gcc9.2.0",
                "gromacs/2023.2-intel2022",
                "gromacs/2022.4",
                "gromacs/2022.4+plumed2.8.2",
                "gromacs/2021.3",
            ]
        )
        self.hrex_module.setToolTip(
            "Optional HPC environment module loaded only by HREX Slurm jobs. "
            "The field is editable for modules not listed here."
        )
        self.hrex_mpi_launcher = QComboBox()
        self.hrex_mpi_launcher.addItems(["mpirun", "srun"])
        self.hrex_mpi_launcher.setToolTip(
            "MPI launcher used to start all HREX replicas as one MPI universe. "
            "Use mpirun for the listed OpenMPI GROMACS modules."
        )
        self.gpus = QSpinBox()
        self.gpus.setRange(0, 16)
        self.gpus.setValue(1)
        self.memory = QLineEdit("16G")
        self.walltime = QLineEdit("24:00:00")
        self.pme_target = QComboBox()
        self.pme_target.addItems(["cpu", "gpu", "auto"])
        self.pme_target.setToolTip(
            "Use CPU for the Conda OpenCL GROMACS build; use GPU for a "
            "CUDA-enabled GROMACS build after testing it."
        )
        self.conda_init = QLineEdit()
        self.conda_environment = QLineEdit()
        self.expected_array_tasks = QLabel("Select or create a ligand network")
        self.expected_array_tasks.setToolTip(
            "Edges × 2 thermodynamic legs × replicates × lambda windows"
        )
        self.worker_script = QLineEdit()
        self.mdp_integrator = QComboBox()
        self.mdp_integrator.addItems(["sd", "md"])
        self.mdp_constraints = QComboBox()
        self.mdp_constraints.addItems(["h-bonds", "all-bonds", "none"])
        self.mdp_coulomb_cutoff = QDoubleSpinBox()
        self.mdp_coulomb_cutoff.setRange(0.1, 5.0)
        self.mdp_coulomb_cutoff.setDecimals(3)
        self.mdp_coulomb_cutoff.setValue(1.0)
        self.mdp_vdw_cutoff = QDoubleSpinBox()
        self.mdp_vdw_cutoff.setRange(0.1, 5.0)
        self.mdp_vdw_cutoff.setDecimals(3)
        self.mdp_vdw_cutoff.setValue(1.0)
        self.mdp_softcore_alpha = QDoubleSpinBox()
        self.mdp_softcore_alpha.setRange(0.01, 5.0)
        self.mdp_softcore_alpha.setDecimals(3)
        self.mdp_softcore_alpha.setValue(0.5)
        self.mdp_lambda_neighbors = QSpinBox()
        self.mdp_lambda_neighbors.setRange(-1, 1000)
        self.mdp_lambda_neighbors.setValue(-1)
        self.mdp_lambda_neighbors.setToolTip(
            "-1 evaluates every state for MBAR; 1 evaluates adjacent states for BAR."
        )
        self.mdp_nstdhdl = QSpinBox()
        self.mdp_nstdhdl.setRange(1, 10_000_000)
        self.mdp_nstdhdl.setValue(100)
        self.simulation_tabs = QTabWidget()

        lambda_page = QWidget()
        lambda_layout = QFormLayout(lambda_page)
        lambda_layout.setLabelAlignment(Qt.AlignRight)
        lambda_layout.addRow("Sampling method:", self.sampling_method)
        self.exchange_interval_label = QLabel("HREX exchange interval:")
        lambda_layout.addRow(self.exchange_interval_label, self.exchange_interval)
        lambda_layout.addRow("Start replicate:", self.start_replicate)
        lambda_layout.addRow("Replicates:", self.replicates)
        lambda_layout.addRow("Lambda schedule:", self.lambda_schedule_mode)
        lambda_layout.addRow("Lambda windows:", self.lambda_windows)
        lambda_layout.addRow("Generated function:", self.lambda_function)
        lambda_layout.addRow("Power exponent:", self.lambda_power)
        lambda_layout.addRow("Manual Coulomb lambdas:", self.coul_lambdas)
        lambda_layout.addRow("Manual vdW lambdas:", self.vdw_lambdas)
        lambda_layout.addRow("Manual bonded lambdas:", self.bonded_lambdas)
        lambda_layout.addRow("Manual restraint lambdas:", self.restraint_lambdas)
        lambda_layout.addRow("Schedule preview:", self.lambda_preview)
        stage_page = QWidget()
        stage_layout = QVBoxLayout(stage_page)
        self.stage_protocol_table = QTableWidget(4, 5)
        self.stage_protocol_table.setHorizontalHeaderLabels(
            ["Stage", "Integrator", "dt (ps)", "Temperature", "Pressure"]
        )
        stage_rows = (
            ("EM", "steep", "—", "no", "no"),
            ("Early NVT", "md", "0.001", "V-rescale", "no"),
            ("Early NPT", "md", "0.001", "V-rescale", "C-rescale"),
            ("Production", "sd", "0.002", "no", "C-rescale"),
        )
        self._stage_protocol_widgets: list[dict[str, QWidget]] = []
        for row, values in enumerate(stage_rows):
            stage_item = QTableWidgetItem(values[0])
            stage_item.setFlags(stage_item.flags() & ~Qt.ItemIsEditable)
            self.stage_protocol_table.setItem(row, 0, stage_item)
            integrator = QComboBox()
            integrator.addItems(
                ["steep", "cg", "l-bfgs"]
                if row == 0
                else ["md", "md-vv", "md-vv-avek", "sd"]
            )
            integrator.setCurrentText(values[1])
            timestep = QDoubleSpinBox()
            timestep.setRange(0.0 if row == 0 else 0.000001, 0.1)
            timestep.setDecimals(6)
            timestep.setSingleStep(0.0005)
            timestep.setValue(0.0 if row == 0 else float(values[2]))
            timestep.setSuffix(" ps")
            temperature = QComboBox()
            temperature.addItem("No coupling / SD", "no")
            if row > 0:
                for label, data in (
                    ("Berendsen (legacy)", "Berendsen"),
                    ("V-rescale", "V-rescale"),
                    ("Nose-Hoover", "Nose-Hoover"),
                ):
                    temperature.addItem(label, data)
            temperature.setCurrentIndex(temperature.findData(values[3]))
            pressure = QComboBox()
            pressure.addItem("No coupling", "no")
            if row >= 2:
                for label, data in (
                    ("Berendsen (legacy)", "Berendsen"),
                    ("C-rescale (recommended)", "C-rescale"),
                    ("Parrinello-Rahman", "Parrinello-Rahman"),
                ):
                    pressure.addItem(label, data)
            pressure.setCurrentIndex(pressure.findData(values[4]))
            for column, widget in enumerate(
                (integrator, timestep, temperature, pressure), start=1
            ):
                self.stage_protocol_table.setCellWidget(row, column, widget)
            integrator.currentTextChanged.connect(
                lambda _value, index=row: self._update_stage_protocol_row(index)
            )
            self._stage_protocol_widgets.append(
                {
                    "integrator": integrator,
                    "timestep": timestep,
                    "temperature": temperature,
                    "pressure": pressure,
                }
            )
        self.stage_protocol_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.stage_protocol_table.verticalHeader().setVisible(False)
        self.stage_protocol_table.horizontalHeader().setStretchLastSection(True)
        self.stage_protocol_table.setMaximumHeight(220)
        for row in range(len(stage_rows)):
            self._update_stage_protocol_row(row)
        stage_layout.addWidget(self.stage_protocol_table)
        stage_form = QFormLayout()
        stage_form.setLabelAlignment(Qt.AlignRight)
        stage_form.addRow("Simulation temperature:", self.temperature)
        stage_form.addRow("EM maximum steps:", self.em_steps)
        stage_form.addRow("EM tolerance (kJ/mol/nm):", self.em_tolerance)
        stage_form.addRow("NVT duration (ns):", self.nvt_duration)
        stage_form.addRow("NPT duration (ns):", self.npt_duration)
        stage_form.addRow("Production duration (ns):", self.production)
        stage_layout.addLayout(stage_form)
        stage_layout.addStretch()
        self.simulation_tabs.addTab(stage_page, "Stage Protocol")

        restraint_page = QWidget()
        restraint_layout = QVBoxLayout(restraint_page)
        restraint_note = QLabel(
            "Configure restraints separately for each equilibration stage. "
            "Each editor uses ASL selections against the structure already "
            "loaded in the System tab using PyMOL-style selections; production "
            "is always unrestrained."
        )
        restraint_note.setWordWrap(True)
        restraint_layout.addWidget(restraint_note)
        self.restraint_protocol_table = QTableWidget(2, 4)
        self.restraint_protocol_table.setHorizontalHeaderLabels(
            ["Stage", "Rules", "Restraint summary", "Edit"]
        )
        self._restraint_stage_rows = (
            ("nvt", "Early NVT"),
            ("npt", "Early NPT"),
        )
        for row, (stage_key, stage_label) in enumerate(self._restraint_stage_rows):
            for column, value in enumerate((stage_label, "", "")):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.restraint_protocol_table.setItem(row, column, item)
            edit = QPushButton("Configure Selection…")
            edit.clicked.connect(
                lambda _checked=False, key=stage_key, label=stage_label: self._open_asl_restraint_editor(
                    key, label
                )
            )
            self.restraint_protocol_table.setCellWidget(row, 3, edit)
        self.restraint_protocol_table.setToolTip(
            "Open a stage to add protein, ligand, or cofactor positional "
            "restraints using topology-stable PyMOL-style selections."
        )
        self.restraint_protocol_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.restraint_protocol_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.restraint_protocol_table.cellDoubleClicked.connect(
            self._open_asl_restraint_editor_for_row
        )
        self.restraint_protocol_table.verticalHeader().setVisible(False)
        self.restraint_protocol_table.horizontalHeader().setStretchLastSection(True)
        self.restraint_protocol_table.setMaximumHeight(220)
        restraint_layout.addWidget(self.restraint_protocol_table)
        self._refresh_restraint_stage_table()
        restraint_units = QLabel(
            "Selections use stable residue and atom identities; force constants are "
            "kJ mol⁻¹ nm⁻². Production: all positional restraints released."
        )
        restraint_units.setStyleSheet("color:#6b7280")
        restraint_units.setWordWrap(True)
        restraint_layout.addWidget(restraint_units)
        restraint_layout.addStretch()
        self.simulation_tabs.addTab(restraint_page, "Restraints")
        self.simulation_tabs.addTab(lambda_page, "Lambda Schedule")

        alchemical_page = QWidget()
        alchemical_layout = QFormLayout(alchemical_page)
        alchemical_layout.setLabelAlignment(Qt.AlignRight)
        alchemical_layout.addRow("MDP constraints:", self.mdp_constraints)
        alchemical_layout.addRow("Coulomb cutoff (nm):", self.mdp_coulomb_cutoff)
        alchemical_layout.addRow("vdW cutoff (nm):", self.mdp_vdw_cutoff)
        alchemical_layout.addRow("Soft-core alpha:", self.mdp_softcore_alpha)
        alchemical_layout.addRow("Lambda neighbors:", self.mdp_lambda_neighbors)
        alchemical_layout.addRow("dH/dλ interval (steps):", self.mdp_nstdhdl)
        self.simulation_tabs.addTab(alchemical_page, "Alchemical Controls")

        form.addRow(self.simulation_tabs)
        form.addRow("Slurm partition:", self.partition)
        form.addRow("Nodes per job:", self.nodes)
        form.addRow("Tasks per job:", self.tasks)
        form.addRow("CPUs per task:", self.cpus)
        form.addRow("GROMACS thread-MPI ranks:", self.ntmpi)
        form.addRow("HREX GROMACS module:", self.hrex_module)
        form.addRow("HREX MPI launcher:", self.hrex_mpi_launcher)
        form.addRow("HREX GROMACS command:", self.hrex_gmx_command)
        form.addRow("GPUs per job:", self.gpus)
        form.addRow("PME target:", self.pme_target)
        form.addRow("Memory per job:", self.memory)
        form.addRow("Walltime:", self.walltime)
        form.addRow("Conda initialization:", self.conda_init)
        form.addRow("Conda environment:", self.conda_environment)
        form.addRow("Expected array tasks:", self.expected_array_tasks)
        form.addRow(
            "Execution worker:",
            self._file_row(
                self.worker_script, "Select GROMACS worker", "Shell scripts (*.sh)"
            ),
        )
        self.tabs = form_step_tabs(
            form,
            (
                ("System", 6),
                ("FEP Simulation", 1),
                ("Job Setting", 16),
            ),
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
        self.show_network_button.setToolTip(
            "Show ligand structures and planned alchemical transformations"
        )
        self.show_network_button.clicked.connect(self.show_network)
        plan = QPushButton("Create LigandFEP-RBFE Workflow")
        plan.clicked.connect(self.create_task)
        run = QPushButton("Run Worker")
        run.clicked.connect(self.run_worker)
        buttons.addWidget(self.show_network_button)
        buttons.addWidget(plan)
        buttons.addWidget(run)
        root.addLayout(buttons)
        self.lambda_schedule_mode.currentTextChanged.connect(self._update_lambda_ui)
        self.lambda_function.currentTextChanged.connect(self._update_lambda_ui)
        self.lambda_windows.valueChanged.connect(self._update_lambda_ui)
        self.lambda_windows.valueChanged.connect(self._update_sampling_method)
        self.sampling_method.currentIndexChanged.connect(
            self._update_sampling_method
        )
        self.replicates.valueChanged.connect(self._update_expected_array_tasks)
        self.start_replicate.valueChanged.connect(
            self._update_expected_array_tasks
        )
        self.coul_lambdas.textChanged.connect(self._update_expected_array_tasks)
        self.network_bundle.editingFinished.connect(
            self._update_expected_array_tasks
        )
        self.lambda_power.valueChanged.connect(self._update_lambda_ui)
        self.cofactor_sdf.textChanged.connect(self._cofactor_input_changed)
        self.ligand_library.editingFinished.connect(
            self._reload_prepared_ligand_directory
        )
        self.force_field_family.currentIndexChanged.connect(
            self._update_force_field_family
        )
        self.water_model.currentTextChanged.connect(
            self._update_water_model_components
        )
        self.force_field_table.itemSelectionChanged.connect(
            self._update_force_field_selection
        )
        self._update_lambda_ui()
        self._update_sampling_method()
        self._update_expected_array_tasks()
        self._update_force_field_selection()
        self._update_embedded_system_builder()
        self._install_parameter_help()

    def _install_parameter_help(self) -> None:
        """Attach concise scientific and execution help to every input field."""
        help_text = {
            "job_name": "Project folder and SLURM job name. Use letters, numbers, '.', '_' or '-'.",
            "protein_path": "Protein or receptor PDB used to prepare the complex leg.",
            "ligand_library": "Directory containing separately parameterized GROMACS ligand folders and their manifest.",
            "network_bundle": "Ligand-network folder containing edges.csv, ligands.sdf, mappings, and the network manifest.",
            "system_type": "Soluble rebuilds and solvates the complex; membrane preserves a prepared membrane/solvent box.",
            "prepared_complex_gro": "Coordinates for an already parameterized complex, including membrane and solvent when applicable.",
            "prepared_complex_topology": "GROMACS topology matching the prepared complex GRO exactly.",
            "force_field_family": "Compatible force-field family used for protein, ligand, lipid, water, ions, and cofactors.",
            "force_field": "Installed PMX/GROMACS protein force field containing mutation-capable parameters.",
            "force_field_table": "Force-field components that will be combined for the system. Each row identifies its role, source, and compatibility notes.",
            "lipid_force_field": "Lipid parameter set; it must be compatible with the selected protein force-field family.",
            "nucleic_acid_force_field": "DNA/RNA parameter set included only when nucleic acids are present.",
            "water_model": "Explicit solvent model used to build soluble boxes and interpret compatible ion parameters.",
            "ion_force_field": "Ion parameter set matched to the selected water model.",
            "ligand_force_field": "Small-molecule parameter family used by the prepared ligand topologies.",
            "cofactor_sdf": "Optional cofactor structure used during system parameterization.",
            "prepared_cofactor_directory": "Optional existing cofactor parameter directory, such as an ACPYPE result.",
            "prepared_cofactor_itp": "Optional pre-generated cofactor topology include file.",
            "box_shape": "Periodic box geometry for newly solvated soluble systems.",
            "box_distance": "Minimum solute-to-box-edge distance in nanometres.",
            "neutralize_system": "Add counterions until the initial system has zero net charge.",
            "ion_strength": "Target bulk salt concentration in mol/L after neutralization.",
            "charge_change_method": "Treatment for ligand transformations whose integer net charge changes.",
            "charge_correction_file": "CSV containing a validated finite-size correction when correction-table mode is selected.",
            "override_membrane_box": "Replace missing or incorrect membrane box vectors with explicitly supplied values.",
            "membrane_box_lengths": "Membrane box lengths a, b, c in nanometres.",
            "membrane_box_angles": "Box angles alpha, beta, gamma in degrees.",
            "sampling_method": "Independent runs one job per lambda window; HREX exchanges configurations between lambda Hamiltonians.",
            "exchange_interval": "Number of MD steps between neighboring Hamiltonian-exchange attempts during HREX production.",
            "start_replicate": "First replicate identifier. For example, start 3 with 3 replicates creates rep_3, rep_4, and rep_5.",
            "replicates": "Independent repeats per edge and thermodynamic leg, initialized separately.",
            "lambda_schedule_mode": "Generated constructs a schedule automatically; manual uses the four explicit lambda vectors below.",
            "lambda_windows": "Number of alchemical states between ligand A and ligand B.",
            "lambda_function": "Shape of the generated lambda progression: cosine, linear, or power.",
            "lambda_power": "Exponent used only for a generated power schedule.",
            "coul_lambdas": "Electrostatic interpolation value for every manual lambda state.",
            "vdw_lambdas": "Van der Waals interpolation value for every manual lambda state.",
            "bonded_lambdas": "Bond, angle, and dihedral interpolation value for every manual lambda state.",
            "restraint_lambdas": "Alchemical scaling vector for topology restraint terms; standard RBFE normally uses zeros.",
            "lambda_preview": "Resolved Coulomb, vdW, bonded, and restraint vectors written to the MDP files.",
            "timestep": "Base integration timestep in picoseconds. Individual protocol stages may use a smaller timestep as shown in the stage table.",
            "temperature": "Target simulation temperature in kelvin, used for velocity generation and every thermostatted stage.",
            "em_steps": "Maximum steepest-descent energy-minimization steps.",
            "em_tolerance": "Energy-minimization convergence threshold in kJ mol⁻¹ nm⁻¹.",
            "nvt_duration": "Constant-volume temperature-equilibration duration in nanoseconds.",
            "npt_duration": "Initial pressure-equilibration duration in nanoseconds.",
            "transition_duration": "Stochastic-dynamics transition duration while strong restraints are reduced.",
            "final_equilibration_duration": "Final equilibration duration with only minimal selected restraints.",
            "production": "Production sampling time per lambda window in nanoseconds.",
            "stage_protocol_table": "Exact integrator, timestep, thermostat, and barostat used for each sequential stage.",
            "restraint_protocol_table": "Five stage-specific restraint sections. Double-click a row or use Configure Selection to edit PyMOL-style rules and force constants in kJ mol⁻¹ nm⁻².",
            "mdp_integrator": "Base production integrator for generated alchemical MDP settings; the stage protocol table shows the effective integrator for every stage.",
            "mdp_constraints": "Bond constraints used by LINCS; h-bonds permits a standard 2 fs timestep.",
            "mdp_coulomb_cutoff": "Real-space electrostatic cutoff in nanometres before PME treatment.",
            "mdp_vdw_cutoff": "Van der Waals cutoff in nanometres; force-field-specific overrides may apply.",
            "mdp_softcore_alpha": "Soft-core strength used to avoid endpoint singularities for appearing/disappearing atoms.",
            "mdp_lambda_neighbors": "Energy states evaluated from each trajectory: -1 for all-state MBAR, 1 for adjacent BAR.",
            "mdp_nstdhdl": "MD-step interval for writing dH/dλ and expanded energy data.",
            "partition": "SLURM partition used for production jobs.",
            "nodes": "Number of compute nodes requested for each scheduler job.",
            "tasks": "SLURM tasks per independent-window job; HREX derives one MPI task per lambda replica.",
            "cpus": "CPU cores assigned to each SLURM task.",
            "ntmpi": "Thread-MPI ranks used by independent-window GROMACS runs.",
            "hrex_module": "Optional HPC module loaded before HREX execution, for example gromacs/2026.3-gcc13.2.",
            "hrex_mpi_launcher": "Starts all HREX replicas in one MPI universe; use mpirun for OpenMPI modules and srun only when the cluster supports direct MPI integration.",
            "hrex_gmx_command": "External-MPI GROMACS executable required for -multidir/-replex HREX, usually gmx_mpi.",
            "gpus": "GPUs requested for each SLURM array job.",
            "pme_target": "Run PME on CPU, GPU, or let GROMACS choose automatically.",
            "memory": "Total memory requested per SLURM job, for example 16G.",
            "walltime": "Maximum scheduler runtime in HH:MM:SS form.",
            "conda_init": "Shell initialization script used before activating the calculation environment.",
            "conda_environment": "Conda environment name or absolute environment path.",
            "expected_array_tasks": "Calculated scheduler workload from edges, legs, replicates, windows, and sampling method.",
            "worker_script": "Optional custom execution worker; leave blank to use the generated built-in runner.",
        }
        self._parameter_help = help_text
        for name, explanation in help_text.items():
            widget = getattr(self, name, None)
            if widget is None:
                continue
            widget.setToolTip(explanation)
            widget.setWhatsThis(explanation)
            widget.setStatusTip(explanation)
            widget.setAccessibleDescription(explanation)

    def _update_stage_protocol_row(self, row: int) -> None:
        widgets = self._stage_protocol_widgets[row]
        integrator = widgets["integrator"].currentText()
        minimization = row == 0
        stochastic = integrator == "sd"
        widgets["timestep"].setEnabled(not minimization)
        widgets["temperature"].setEnabled(not minimization and not stochastic)
        widgets["pressure"].setEnabled(not minimization and row >= 2)
        if minimization or stochastic:
            widgets["temperature"].setCurrentIndex(
                widgets["temperature"].findData("no")
            )
        elif widgets["temperature"].currentData() == "no":
            widgets["temperature"].setCurrentIndex(
                widgets["temperature"].findData("V-rescale")
            )
        tooltips = {
            "integrator": "GROMACS integrator for this stage.",
            "timestep": "Integration timestep in ps; minimization has no MD timestep.",
            "temperature": (
                "Temperature coupling. With integrator=sd, GROMACS uses stochastic "
                "dynamics and ignores tcoupl."
            ),
            "pressure": (
                "Pressure coupling. C-rescale is suitable for equilibration and "
                "production; Berendsen is retained only as a legacy option."
            ),
        }
        for name, widget in widgets.items():
            widget.setToolTip(tooltips[name])

    def _stage_protocol_settings(self) -> dict[str, list[object]]:
        integrators: list[str] = []
        timesteps: list[float] = []
        temperatures: list[str] = []
        pressures: list[str] = []
        for row, widgets in enumerate(self._stage_protocol_widgets):
            integrator = widgets["integrator"].currentText()
            integrators.append(integrator)
            timesteps.append(
                0.0 if row == 0 else widgets["timestep"].value()
            )
            temperatures.append(str(widgets["temperature"].currentData()))
            pressures.append(str(widgets["pressure"].currentData()))
        # The configuration loader retains its historical six-slot schema for
        # compatibility. Map the visible Production row onto the unused legacy
        # transition/final slots as well; generated workflows execute only EM,
        # NVT, NPT, and Production.
        order = (0, 1, 2, 3, 3, 3)
        return {
            "integrators": [integrators[index] for index in order],
            "timesteps_ps": [timesteps[index] for index in order],
            "temperature_coupling": [temperatures[index] for index in order],
            "pressure_coupling": [pressures[index] for index in order],
        }

    @staticmethod
    def _pmx_force_field_directories() -> list[Path]:
        candidates: list[Path] = []
        try:
            import pmx
        except ImportError:
            pmx = None
        if pmx is not None:
            candidates.append(Path(pmx.__file__).resolve().parent / "data" / "mutff")
        candidates.extend(
            Path(sys.prefix).glob("lib/python*/site-packages/pmx/data/mutff")
        )
        executable = shutil.which("pmx")
        if executable:
            prefix = Path(executable).resolve().parent.parent
            candidates.extend(
                prefix.glob("lib/python*/site-packages/pmx/data/mutff")
            )
            candidates.append(prefix / "share" / "pmx" / "data" / "mutff")
        pmx_home = os.environ.get("PMX_HOME")
        if pmx_home:
            home = Path(pmx_home).expanduser()
            candidates.extend((home / "data" / "mutff", home / "mutff"))
        gmxlibrary = os.environ.get("GMXLIB")
        if gmxlibrary:
            candidates.extend(
                Path(value).expanduser()
                for value in gmxlibrary.split(os.pathsep)
                if value
            )
        directories: list[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_dir() and resolved not in directories:
                directories.append(resolved)
        return directories

    @staticmethod
    def _lipid_force_field_display(path: Path) -> str:
        if path.name.casefold().startswith("charmm36m"):
            return "Included CHARMM36m lipids"
        if path.name.casefold().startswith("amber"):
            return "Select installed Amber lipid set below"
        return "Verify compatibility"

    @staticmethod
    def _force_field_component_display(path: Path) -> str:
        name = path.name.casefold()
        if "bsc1-mut" in name:
            return "Protein + DNA mutation (bsc1)"
        if "dna-mut" in name:
            return "DNA mutation"
        if name.startswith("charmm36m"):
            return "Protein + DNA mutation"
        if name.startswith("amber"):
            return "Protein mutation"
        return "Custom mutation bundle"

    def _append_force_field(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        for row in range(self.force_field_table.rowCount()):
            item = self.force_field_table.item(row, 0)
            if item is not None and Path(item.data(Qt.UserRole)) == resolved:
                self.force_field_table.selectRow(row)
                return
        row = self.force_field_table.rowCount()
        self.force_field_table.insertRow(row)
        component = QTableWidgetItem(self._force_field_component_display(resolved))
        component.setData(Qt.UserRole, str(resolved))
        component.setData(Qt.UserRole + 1, "primary")
        component.setToolTip(str(resolved))
        self.force_field_table.setItem(row, 0, component)
        self.force_field_table.setItem(row, 1, QTableWidgetItem(resolved.name))
        self.force_field_table.setItem(
            row, 2, QTableWidgetItem(self._lipid_force_field_display(resolved))
        )

    def _populate_force_field_table(self) -> None:
        directories = self._pmx_force_field_directories()
        fields = sorted(
            {
                path
                for directory in directories
                for path in directory.glob("*.ff")
                if path.name.casefold().startswith("amber99")
                or path.name.casefold().startswith("amber14")
                or path.name.casefold() == "charmm36m-mut.ff"
            },
            key=lambda path: path.name.casefold(),
        )
        for path in fields:
            self._append_force_field(path)
        preferred = "amber99sb-star-ildn-mut.ff"
        for row in range(self.force_field_table.rowCount()):
            item = self.force_field_table.item(row, 0)
            if item is not None and Path(item.data(Qt.UserRole)).name == preferred:
                self.force_field_table.selectRow(row)
                break
        if self.force_field_table.currentRow() < 0 and fields:
            self.force_field_table.selectRow(0)
        if fields:
            self.force_field_discovery_status.setText(
                f"Detected {len(fields)} PMX mutation force fields in: "
                + ", ".join(str(path) for path in directories)
            )
            self.force_field_discovery_status.setStyleSheet("color:#52708f")
        else:
            searched = [
                str(Path(sys.prefix) / "lib/python*/site-packages/pmx/data/mutff"),
                "PMX_HOME/data/mutff",
                "GMXLIB",
            ]
            self.force_field_discovery_status.setText(
                "No PMX mutation force fields were detected. Active Python: "
                f"{sys.executable}. Searched: " + ", ".join(searched) + ". "
                "Install PMX in this environment or use Add Force Field."
            )
            self.force_field_discovery_status.setStyleSheet("color:#b42318")

    def _add_force_field(self) -> None:
        choice = QMessageBox(self)
        choice.setWindowTitle("Add Force Field")
        choice.setText("What do you want to add to the force-field box?")
        directory_button = choice.addButton(
            "PMX/GROMACS .ff Directory", QMessageBox.ActionRole
        )
        cofactor_button = choice.addButton(
            "Cofactor SDF/MOL2 (GAFF2)", QMessageBox.ActionRole
        )
        cofactor_itp_button = choice.addButton(
            "Existing Cofactor ITP", QMessageBox.ActionRole
        )
        choice.addButton(QMessageBox.Cancel)
        choice.exec_()
        clicked = choice.clickedButton()
        if clicked is cofactor_itp_button:
            if self.force_field_family.currentData() != "amber":
                QMessageBox.warning(
                    self,
                    "Unsupported Cofactor Topology",
                    "Importing an ACPYPE ITP is currently supported by the Amber "
                    "system builder. A CHARMM36m system requires matching CGenFF "
                    "parameters.",
                )
                return
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select existing cofactor ITP",
                self.working_directory,
                "GROMACS topology include (*.itp)",
            )
            if not path:
                return
            itp = Path(path).expanduser().resolve()
            parameter_directory = itp.parent
            if not list(parameter_directory.rglob("*.mol2")) or not list(
                parameter_directory.rglob("*.frcmod")
            ):
                QMessageBox.warning(
                    self,
                    "Incomplete Cofactor Parameters",
                    "The ITP must be selected from its complete ACPYPE output "
                    "folder, which must also contain the assigned-charge MOL2 "
                    "and matching frcmod required to assemble the Amber system.",
                )
                return
            self.cofactor_sdf.clear()
            self.prepared_cofactor_directory = parameter_directory
            self.prepared_cofactor_itp = itp
            self._add_prepared_cofactor_row()
            self._update_embedded_system_builder()
            return
        if clicked is cofactor_button:
            if self.force_field_family.currentData() != "amber":
                QMessageBox.warning(
                    self,
                    "Unsupported Cofactor Parameters",
                    "Automatic cofactor parameterization currently uses Amber "
                    "GAFF2. Select an Amber force field first. CHARMM36m requires "
                    "a separately prepared compatible CGenFF topology.",
                )
                return
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select cofactor molecule",
                self.working_directory,
                "Molecule files (*.sdf *.mol2)",
            )
            if path:
                self.cofactor_sdf.setText(path)
            return
        if clicked is not directory_button:
            return
        path = QFileDialog.getExistingDirectory(
            self, "Select PMX/GROMACS .ff directory", self.working_directory
        )
        if not path:
            return
        force_field = Path(path)
        if force_field.suffix.casefold() != ".ff":
            QMessageBox.warning(
                self,
                "Invalid Force Field",
                "Select a force-field directory whose name ends in .ff.",
            )
            return
        self._append_force_field(force_field)

    def _delete_force_field(self) -> None:
        row = self.force_field_table.currentRow()
        if row >= 0:
            item = self.force_field_table.item(row, 0)
            if item is None:
                return
            if item.data(Qt.UserRole + 1) == "cofactor":
                self.cofactor_sdf.clear()
                return
            if item.data(Qt.UserRole + 1) == "cofactor_itp":
                self.prepared_cofactor_directory = None
                self.prepared_cofactor_itp = None
                self.force_field_table.removeRow(row)
                self._update_embedded_system_builder()
                return
            removed_path = Path(item.data(Qt.UserRole))
            self.force_field_table.removeRow(row)
            if getattr(self, "_active_force_field", None) == removed_path:
                self._active_force_field = None
        if self.force_field_table.rowCount() and self.force_field_table.currentRow() < 0:
            self.force_field_table.selectRow(0)

    def _selected_force_field_path(self) -> Path | None:
        row = self.force_field_table.currentRow()
        item = self.force_field_table.item(row, 0) if row >= 0 else None
        if item is not None and item.data(Qt.UserRole + 1) == "primary":
            return Path(item.data(Qt.UserRole))
        active = getattr(self, "_active_force_field", None)
        if active is not None:
            return active
        for row in range(self.force_field_table.rowCount()):
            item = self.force_field_table.item(row, 0)
            if item is not None and item.data(Qt.UserRole + 1) == "primary":
                return Path(item.data(Qt.UserRole))
        return None

    def _selected_force_field_name(self) -> str:
        path = self._selected_force_field_path()
        if path is None:
            raise ValueError("Select a force field")
        return path.name.removesuffix(".ff")

    def _update_force_field_selection(self, *_args) -> None:
        row = self.force_field_table.currentRow()
        current = self.force_field_table.item(row, 0) if row >= 0 else None
        if current is not None and current.data(Qt.UserRole + 1) != "primary":
            self.delete_force_field_button.setEnabled(True)
            return
        path = self._selected_force_field_path()
        if path is None:
            self.delete_force_field_button.setEnabled(False)
            return
        self.delete_force_field_button.setEnabled(True)
        self._active_force_field = path
        family = "charmm36m" if path.name.casefold().startswith("charmm") else "amber"
        index = self.force_field_family.findData(family)
        if index >= 0:
            self.force_field_family.setCurrentIndex(index)
        self._update_force_field_family()
        self.force_field.clear()
        self.force_field.addItem(path.name.removesuffix(".ff"))

    def _cofactor_input_changed(self, *_args) -> None:
        for row in reversed(range(self.force_field_table.rowCount())):
            item = self.force_field_table.item(row, 0)
            if item is not None and item.data(Qt.UserRole + 1) in {
                "cofactor",
                "cofactor_itp",
            }:
                self.force_field_table.removeRow(row)
        path_text = self.cofactor_sdf.text().strip()
        if path_text:
            self.prepared_cofactor_directory = None
            self.prepared_cofactor_itp = None
            path = Path(path_text).expanduser()
            row = self.force_field_table.rowCount()
            self.force_field_table.insertRow(row)
            component = QTableWidgetItem("Cofactor")
            component.setData(Qt.UserRole, str(path))
            component.setData(Qt.UserRole + 1, "cofactor")
            component.setToolTip(str(path))
            self.force_field_table.setItem(row, 0, component)
            self.force_field_table.setItem(
                row, 1, QTableWidgetItem(f"GAFF2 from {path.name}")
            )
            self.force_field_table.setItem(
                row, 2, QTableWidgetItem("Uses assigned net charge")
            )
        self._update_embedded_system_builder()

    def _add_prepared_cofactor_row(self) -> None:
        if self.prepared_cofactor_itp is None:
            return
        for row in reversed(range(self.force_field_table.rowCount())):
            item = self.force_field_table.item(row, 0)
            if item is not None and item.data(Qt.UserRole + 1) in {
                "cofactor",
                "cofactor_itp",
            }:
                self.force_field_table.removeRow(row)
        row = self.force_field_table.rowCount()
        self.force_field_table.insertRow(row)
        component = QTableWidgetItem("Cofactor")
        component.setData(Qt.UserRole, str(self.prepared_cofactor_itp))
        component.setData(Qt.UserRole + 1, "cofactor_itp")
        component.setToolTip(str(self.prepared_cofactor_itp))
        self.force_field_table.setItem(row, 0, component)
        self.force_field_table.setItem(
            row,
            1,
            QTableWidgetItem(f"Existing ITP: {self.prepared_cofactor_itp.name}"),
        )
        self.force_field_table.setItem(
            row, 2, QTableWidgetItem("Reuses charged MOL2 + frcmod")
        )

    def _update_force_field_family(self, *_args) -> None:
        family = self.force_field_family.currentData()
        amber = family == "amber"
        if amber:
            current_lipid = self.lipid_force_field.currentText()
            current = self.force_field.currentText()
            self.force_field.clear()
            self.force_field.addItems(["ff14SB", "ff19SB"])
            if current in {"ff14SB", "ff19SB"}:
                self.force_field.setCurrentText(current)
            installed_lipids = installed_amber_lipid_force_fields()
            if installed_lipids:
                self.lipid_force_field.setCurrentText(
                    current_lipid
                    if current_lipid in installed_lipids
                    else installed_lipids[0]
                )
            self.nucleic_acid_force_field.setCurrentText("DNA OL21 + RNA OL3")
            self.water_model.clear()
            self.water_model.addItems(
                [
                    "TIP3P",
                    "OPC",
                    "OPC3",
                    "SPC/E",
                    "TIP4P-Ew",
                ]
            )
            self.ligand_force_field.setCurrentText("GAFF2")
        else:
            self.force_field.clear()
            self.force_field.addItem(str(family))
            self.lipid_force_field.setCurrentText("Included in selected CHARMM .ff")
            self.nucleic_acid_force_field.setCurrentText(
                "Included in selected CHARMM .ff"
            )
            self.water_model.clear()
            self.water_model.addItems(["TIP3P", "TIP4P", "SPC", "SPC/E"])
            self.ion_force_field.setCurrentText("Included in selected CHARMM .ff")
            self.ligand_force_field.setCurrentText("CGenFF topology required")
        self.force_field.setEnabled(amber)
        for combo in (
            self.lipid_force_field,
            self.nucleic_acid_force_field,
            self.ion_force_field,
            self.ligand_force_field,
        ):
            combo.setEnabled(amber)
        self.water_model.setEnabled(True)
        self._update_water_model_components()
        self._update_embedded_system_builder()

    def _update_water_model_components(self, *_args) -> None:
        if self.force_field_family.currentData() == "amber":
            ions = {
                "TIP3P": "Joung-Cheatham TIP3P",
                "OPC": "Li-Merz OPC",
                "OPC3": "Li-Merz OPC3",
                "SPC/E": "Joung-Cheatham SPC/E",
                "TIP4P-Ew": "Joung-Cheatham TIP4P-Ew",
            }.get(self.water_model.currentText())
            if ions is None:
                return
            self.ion_force_field.clear()
            self.ion_force_field.addItem(ions)
        else:
            self.ion_force_field.clear()
            self.ion_force_field.addItem("Included in selected CHARMM .ff")

    def _load_prepared_ligands(self, directory: Path) -> None:
        self._prepared_ligands = []
        try:
            manifest = directory / "ligand_manifest.tsv"
            records: list[tuple[str, str, str, int]] = []
            recorded_directories: set[str] = set()
            if manifest.is_file():
                with manifest.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle, delimiter="\t"):
                        name = row.get("ligand_name", "").strip()
                        resname = row.get("resname", "").strip()
                        folder = row.get("directory", "").strip()
                        target = directory / folder
                        if (
                            name
                            and resname
                            and folder
                            and target.is_dir()
                            and any(target.rglob("*.mol2"))
                            and any(target.rglob("*.frcmod"))
                        ):
                            try:
                                net_charge = int(row.get("net_charge", "0"))
                            except ValueError:
                                continue
                            records.append((name, resname, folder, net_charge))
                            recorded_directories.add(folder)
            for folder in sorted(directory.iterdir()):
                if (
                    not folder.is_dir()
                    or folder.name in recorded_directories
                    or not any(folder.rglob("*.mol2"))
                    or not any(folder.rglob("*.frcmod"))
                ):
                    continue
                acpype_directories = sorted(folder.glob("*.acpype"))
                resname = (
                    acpype_directories[0].stem
                    if acpype_directories
                    else folder.name
                )
                if not re.fullmatch(r"[A-Za-z0-9]{1,5}", resname):
                    mol2 = next(iter(sorted(folder.rglob("*.mol2"))))
                    resname = self._mol2_resname(mol2) or "LIG"
                mol2 = next(iter(sorted(folder.rglob("*.mol2"))))
                records.append(
                    (folder.name, resname.upper(), folder.name, self._mol2_net_charge(mol2))
                )
            for name, resname, folder, net_charge in records:
                ligand_directory = (directory / folder).resolve()
                self._prepared_ligands.append(
                    {
                        "name": name,
                        "resname": resname,
                        "directory": folder,
                        "path": str(ligand_directory),
                        "net_charge": str(net_charge),
                    }
                )
        except OSError:
            return

    def _reload_prepared_ligand_directory(self) -> None:
        path_text = self.ligand_library.text().strip()
        if not path_text:
            self._prepared_ligands = []
            return
        directory = Path(path_text).expanduser()
        if directory.is_dir():
            self._load_prepared_ligands(directory)

    def _prepared_ligand_row(self) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.ligand_library)
        button = QPushButton("Browse...")

        def browse() -> None:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select prepared GROMACS ligand directory",
                self.working_directory,
            )
            if path:
                self.ligand_library.setText(path)
                self._load_prepared_ligands(Path(path))

        button.clicked.connect(browse)
        layout.addWidget(button)
        container = QWidget()
        container.setLayout(layout)
        return container

    def _update_embedded_system_builder(self, *_args) -> None:
        pass

    def _update_charge_correction_controls(self, *_args) -> None:
        enabled = self.charge_change_method.currentData() == "correction"
        self.charge_correction_label.setVisible(enabled)
        self.charge_correction_row.setVisible(enabled)

    def _update_box_override_controls(self, *_args) -> None:
        membrane = self.system_type.currentText() == "membrane"
        enabled = membrane and self.override_membrane_box.isChecked()
        self.override_membrane_box.setEnabled(membrane)
        self.membrane_box_lengths.setEnabled(enabled)
        self.membrane_box_angles.setEnabled(enabled)

    @staticmethod
    def _box_triplet(text: str, label: str) -> tuple[float, float, float]:
        fields = text.replace(",", " ").split()
        if len(fields) != 3:
            raise ValueError(f"{label} must contain exactly three numbers")
        try:
            return tuple(float(value) for value in fields)  # type: ignore[return-value]
        except ValueError as error:
            raise ValueError(f"{label} contains a non-numeric value") from error

    def _system_anchor_ligand(self) -> dict[str, str]:
        """Choose the bound-system anchor without requiring a global reference."""
        if not self._prepared_ligands:
            path_text = self.ligand_library.text().strip()
            if path_text:
                self._load_prepared_ligands(Path(path_text).expanduser())
        if not self._prepared_ligands:
            raise ValueError(
                "The prepared ligand folder contains no usable ligand parameters"
            )

        structure_text = self.protein_path.text().strip()
        if structure_text:
            try:
                pdb_residues = {
                    line[17:20].strip().upper()
                    for line in Path(structure_text)
                    .expanduser()
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines()
                    if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 20
                }
            except OSError:
                pdb_residues = set()
            for record in self._prepared_ligands:
                if record["resname"].upper() in pdb_residues:
                    return record

        bundle_text = self.network_bundle.text().strip()
        ligand_sdf = Path(bundle_text).expanduser() / "ligands.sdf"
        if ligand_sdf.is_file():
            from rdkit import Chem

            network_names = [
                molecule.GetProp("_Name").strip()
                for molecule in Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
                if molecule is not None and molecule.HasProp("_Name")
            ]
            by_name = {record["name"]: record for record in self._prepared_ligands}
            for name in network_names:
                if name in by_name:
                    return by_name[name]
        return self._prepared_ligands[0]

    def _system_preparation_mode(self) -> str:
        return (
            "amber_pdb_sdf"
            if self.force_field_family.currentData() == "amber"
            else "pdb2gmx"
        )

    def _workflow_water_model(self) -> str:
        return {
            "TIP3P": "tip3p",
            "OPC": "opc",
            "OPC3": "opc3",
            "TIP4P-Ew": "tip4pew",
            "TIP4P": "tip4p",
            "SPC": "spc",
            "SPC/E": "spce",
        }[self.water_model.currentText()]

    def _synchronize_system_type_with_input(self, *_args) -> None:
        """Prevent a lipid-containing PDB from being staged as a soluble system."""
        text = self.protein_path.text().strip()
        if not text:
            return
        path = Path(text).expanduser()
        if not path.is_file() or path.suffix.casefold() != ".pdb":
            return
        try:
            lipid_count = count_lipid_molecules_pdb(path)
        except OSError:
            return
        if lipid_count < 20:
            return
        self.system_type.setCurrentText("membrane")
        self.system_type.setToolTip(
            f"Membrane selected automatically: {lipid_count} lipid molecules "
            "were detected in the input PDB."
        )
        if hasattr(self, "system_build_status"):
            self.system_build_status.setText(
                f"Membrane detected ({lipid_count} lipids); system not prepared"
            )
            self.system_build_status.setStyleSheet("color:#71839a")

    @staticmethod
    def _mol2_resname(path: Path) -> str | None:
        """Read the first Tripos substructure name from a MOL2 file."""
        in_atoms = False
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if line.startswith("@<TRIPOS>"):
                in_atoms = False
            if in_atoms:
                fields = line.split()
                if len(fields) >= 8 and re.fullmatch(r"[A-Za-z0-9]{1,5}", fields[7]):
                    return fields[7].upper()
        return None

    @staticmethod
    def _mol2_net_charge(path: Path) -> int:
        """Sum assigned MOL2 atom charges and require an integer molecular charge."""
        in_atoms = False
        total = 0.0
        found = False
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if line.startswith("@<TRIPOS>"):
                in_atoms = False
            if in_atoms:
                fields = line.split()
                if len(fields) >= 9:
                    try:
                        total += float(fields[8])
                        found = True
                    except ValueError:
                        continue
        if not found:
            return 0
        rounded = round(total)
        if abs(total - rounded) > 0.1:
            raise ValueError(
                f"Assigned MOL2 charges in {path} sum to {total:.4f}, not an "
                "integer molecular charge"
            )
        return rounded

    def _write_charge_manifest(self, project: Path) -> Path:
        """Write the normalized assigned-charge list consumed by edge planning."""
        directory_text = self.ligand_library.text().strip()
        if not directory_text:
            raise ValueError("Select the prepared GROMACS ligand folder")
        self._load_prepared_ligands(Path(directory_text).expanduser())
        if not self._prepared_ligands:
            raise ValueError("The prepared ligand folder has no usable parameters")
        manifest = project / "prepared_ligand_charges.tsv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                delimiter="\t",
                fieldnames=("index", "ligand_name", "resname", "net_charge", "directory"),
            )
            writer.writeheader()
            for index, record in enumerate(self._prepared_ligands, start=1):
                writer.writerow(
                    {
                        "index": index,
                        "ligand_name": record["name"],
                        "resname": record["resname"],
                        "net_charge": record["net_charge"],
                        "directory": record["path"],
                    }
                )
        return manifest

    def _cofactor_identity(self) -> tuple[str, int]:
        """Derive the PDB residue name and integer charge from cofactor input."""
        if self.prepared_cofactor_directory is not None:
            mol2_files = sorted(self.prepared_cofactor_directory.rglob("*.mol2"))
            if not mol2_files:
                raise ValueError("The prepared cofactor folder contains no MOL2 file")
            directory_candidates = [
                self.prepared_cofactor_directory.name.removesuffix(".acpype").upper(),
                *(path.stem.removesuffix(".acpype").upper() for path in self.prepared_cofactor_directory.glob("*.acpype")),
            ]
            resname = next(
                (
                    candidate
                    for candidate in directory_candidates
                    if re.fullmatch(r"[A-Za-z0-9]{1,5}", candidate)
                    and candidate not in {"UNL", "UNL1", "LIG", "LIGA", "COF"}
                ),
                None,
            )
            if resname is None:
                resname = self._mol2_resname(mol2_files[0])
            if resname is None:
                raise ValueError(
                    "Could not derive the cofactor PDB residue name from the prepared "
                    "MOL2. Its Tripos substructure name must match the PDB residue."
                )
            return resname, 0

        path_text = self.cofactor_sdf.text().strip()
        if not path_text:
            return "", 0
        path = Path(path_text).expanduser().resolve()
        from rdkit import Chem

        if path.suffix.casefold() == ".sdf":
            molecules = [
                molecule
                for molecule in Chem.SDMolSupplier(str(path), removeHs=False)
                if molecule is not None
            ]
        else:
            molecule = Chem.MolFromMol2File(str(path), removeHs=False)
            molecules = [molecule] if molecule is not None else []
        if len(molecules) != 1:
            raise ValueError("The cofactor file must contain exactly one molecule")
        molecule = molecules[0]
        candidates = []
        if path.suffix.casefold() == ".mol2":
            candidates.append(self._mol2_resname(path))
        if molecule.HasProp("_Name"):
            candidates.append(molecule.GetProp("_Name").strip().upper())
        candidates.append(path.stem.upper())
        resname = next(
            (
                candidate
                for candidate in candidates
                if candidate and re.fullmatch(r"[A-Za-z0-9]{1,5}", candidate)
            ),
            None,
        )
        if resname is None:
            raise ValueError(
                "Could not derive a 1-5 character cofactor residue name. Name the "
                "SDF/MOL2 molecule (or file) with its matching PDB residue name."
            )
        return resname, int(Chem.GetFormalCharge(molecule))

    def _stage_system_preparation(self) -> tuple[Path, Path, Path]:
        self._synchronize_system_type_with_input()
        job_name = self.job_name.text().strip() or "LigRBFE_GROMACS"
        output = Path(self.working_directory) / job_name / "prepared_system"
        mode = self._system_preparation_mode()
        protein_choice = self.force_field.currentText().strip().casefold()
        nucleic_acid = self.nucleic_acid_force_field.currentText()
        dna = (
            "None"
            if nucleic_acid in {"None", "From prepared topology"}
            else ("bsc1" if "bsc1" in nucleic_acid else "OL21")
        )
        rna = (
            "None"
            if nucleic_acid in {"None", "From prepared topology"}
            else "OL3"
        )
        structure_text = self.protein_path.text().strip()
        topology_text = ""
        family = self.force_field_family.currentData()
        selected_force_field = self._selected_force_field_path()
        force_field_name = self._selected_force_field_name()
        anchor_ligand = self._system_anchor_ligand()
        reference_name = anchor_ligand["name"]
        ligand_resname = anchor_ligand.get("resname", "LIG")
        cofactor_resname, cofactor_charge = self._cofactor_identity()
        override_box = (
            self.system_type.currentText() == "membrane"
            and self.override_membrane_box.isChecked()
        )
        box_lengths = (
            self._box_triplet(
                self.membrane_box_lengths.text(), "Membrane box lengths"
            )
            if override_box
            else None
        )
        box_angles = (
            self._box_triplet(self.membrane_box_angles.text(), "Membrane box angles")
            if override_box
            else (90.0, 90.0, 90.0)
        )
        config = GromacsSystemPreparationConfig(
            mode=mode,
            structure=Path(structure_text).expanduser().resolve(),
            topology=(
                Path(topology_text).expanduser().resolve()
                if topology_text
                else None
            ),
            output_directory=output.resolve(),
            system_type=self.system_type.currentText(),
            force_field=force_field_name,
            force_field_directory=(
                selected_force_field if mode == "pdb2gmx" else None
            ),
            water_model=self._workflow_water_model(),
            protein_force_field=(
                "ff99SBildn"
                if "amber99" in protein_choice
                else ("ff19SB" if "19" in protein_choice else "ff14SB")
            ),
            lipid_force_field=self.lipid_force_field.currentText(),
            dna_force_field=dna,
            rna_force_field=rna,
            ion_parameters=self.ion_force_field.currentText(),
            ligand_force_field="GAFF2",
            prepared_ligand_directory=(
                Path(self.ligand_library.text().strip()).expanduser().resolve()
                if self.ligand_library.text().strip()
                else None
            ),
            reference_ligand=reference_name or None,
            ligand_resname=ligand_resname,
            ligand_charge=int(anchor_ligand.get("net_charge", "0")),
            exclude_ligand_from_system=True,
            neutralize=self.neutralize_system.isChecked(),
            cofactor_force_field=(
                "GAFF2"
                if family == "amber"
                and (
                    self.cofactor_sdf.text().strip()
                    or self.prepared_cofactor_directory is not None
                )
                else "None"
            ),
            cofactor_structure=(
                Path(self.cofactor_sdf.text().strip()).expanduser().resolve()
                if self.cofactor_sdf.text().strip()
                else None
            ),
            prepared_cofactor_directory=self.prepared_cofactor_directory,
            cofactor_resname=cofactor_resname,
            cofactor_charge=cofactor_charge,
            box_lengths_nm=box_lengths,
            box_angles_deg=box_angles,
        )
        runner, manifest = stage_gromacs_system(config)
        self.prepared_complex_gro.setText(str(output / "system.gro"))
        self.prepared_complex_topology.setText(str(output / "topol.top"))
        return output, runner, manifest

    def _prepare_system(self) -> None:
        try:
            output, runner, manifest = self._stage_system_preparation()
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Prepare GROMACS System", str(error))
            return
        self.system_build_status.setText("Preparation job created")
        self.system_build_status.setStyleSheet("color:#16823b;font-weight:600")
        self.job_requested.emit(str(output), str(runner), str(manifest))

    def _validate_system_form(self) -> None:
        self._synchronize_system_type_with_input()
        if self.system_type.currentText() != "membrane":
            return
        gro = self.prepared_complex_gro.text().strip()
        topology = self.prepared_complex_topology.text().strip()
        if not gro or not topology:
            raise ValueError(
                "A membrane RBFE calculation requires an already parameterized "
                "complete system. Use Prepare / Validate System first to generate "
                "the matching system.gro and topol.top from the selected inputs."
            )

    @staticmethod
    def _parse_lambda_values(text: str, label: str) -> list[float]:
        tokens = [token for token in re.split(r"[\s,;]+", text.strip()) if token]
        if not tokens:
            raise ValueError(f"{label} is required for a manual schedule")
        try:
            return [float(token) for token in tokens]
        except ValueError as error:
            raise ValueError(f"{label} contains a non-numeric value") from error

    @staticmethod
    def _parse_restraint_names(text: str) -> list[str]:
        return [
            token.upper()
            for token in re.split(r"[\s,;]+", text.strip())
            if token
        ]

    @staticmethod
    def _parse_residue_numbers(text: str) -> list[int]:
        values: set[int] = set()
        for token in (
            item for item in re.split(r"[\s,;]+", text.strip()) if item
        ):
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                try:
                    start, end = int(start_text), int(end_text)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid protein residue range: {token}"
                    ) from error
                if start < 1 or end < start:
                    raise ValueError(f"Invalid protein residue range: {token}")
                values.update(range(start, end + 1))
            else:
                try:
                    value = int(token)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid protein residue number: {token}"
                    ) from error
                if value < 1:
                    raise ValueError("Protein residue numbers must be positive")
                values.add(value)
        return sorted(values)

    @staticmethod
    def _default_stage_restraint_rules() -> dict[str, list[dict[str, object]]]:
        def rules(protein: float, ligand: float, cofactor: float):
            return [
                {
                    "name": "Protein heavy atoms",
                    "asl": "protein and not element H",
                    "force": protein,
                },
                {
                    "name": "Ligand heavy atoms",
                    "asl": "ligand and not element H",
                    "force": ligand,
                },
                {
                    "name": "Cofactor heavy atoms",
                    "asl": "cofactor and not element H",
                    "force": cofactor,
                },
            ]

        return {
            "nvt": rules(1000.0, 1000.0, 1000.0),
            "npt": rules(500.0, 500.0, 500.0),
        }

    def _restraint_structure_description(self) -> str:
        if self.prepared_complex_gro.text().strip():
            return self.prepared_complex_gro.text().strip()
        if self.protein_path.text().strip():
            return self.protein_path.text().strip()
        return "System-tab structure inputs (select the protein/complex first)"

    def _open_asl_restraint_editor_for_row(self, row: int, _column: int) -> None:
        if not 0 <= row < len(self._restraint_stage_rows):
            return
        self._open_asl_restraint_editor(*self._restraint_stage_rows[row])

    def _open_asl_restraint_editor(self, stage_key: str, stage_label: str) -> None:
        editor = ASLRestraintDialog(
            stage_key=stage_key,
            stage_label=stage_label,
            rules=self._stage_restraint_rules[stage_key],
            structure_description=self._restraint_structure_description(),
            parent=self,
        )
        editor.rules_saved.connect(self._save_stage_restraint_rules)
        self._restraint_editors[stage_key] = editor
        editor.finished.connect(
            lambda _result, key=stage_key: self._restraint_editors.pop(key, None)
        )
        show_modeless(self, editor)

    def _save_stage_restraint_rules(
        self, stage_key: str, rules: list[dict[str, object]]
    ) -> None:
        self._stage_restraint_rules[stage_key] = rules
        component_templates: dict[str, list[dict[str, object]]] = {}
        for rule in rules:
            component = str(
                parse_restraint_asl(str(rule["asl"]))["component"]
            )
            component_templates.setdefault(component, []).append(rule)
        for component, templates in component_templates.items():
            for other_stage, other_rules in self._stage_restraint_rules.items():
                if other_stage == stage_key:
                    continue
                matching = [
                    other
                    for other in other_rules
                    if parse_restraint_asl(str(other["asl"]))["component"]
                    == component
                ]
                preserved_force = (
                    float(matching[0]["force"]) if matching else 0.0
                )
                other_rules[:] = [
                    other
                    for other in other_rules
                    if parse_restraint_asl(str(other["asl"]))["component"]
                    != component
                ]
                other_rules.extend(
                    {
                        "name": template["name"],
                        "asl": template["asl"],
                        "force": preserved_force,
                    }
                    for template in templates
                )
        self._refresh_restraint_stage_table()

    def _refresh_restraint_stage_table(self) -> None:
        for row, (stage_key, _stage_label) in enumerate(self._restraint_stage_rows):
            rules = self._stage_restraint_rules[stage_key]
            active = [rule for rule in rules if float(rule["force"]) > 0]
            self.restraint_protocol_table.item(row, 1).setText(str(len(active)))
            summary = []
            for rule in active:
                component = str(parse_restraint_asl(str(rule["asl"]))["component"])
                summary.append(f"{component.title()} {float(rule['force']):g}")
            self.restraint_protocol_table.item(row, 2).setText(
                "; ".join(summary) if summary else "Released"
            )

    def _resolved_restraint_configuration(self) -> dict[str, object]:
        forces_by_stage: dict[str, list[float]] = {}
        component_selections: dict[str, list[dict[str, object]]] = {
            "protein": [],
            "ligand": [],
            "cofactor": [],
        }
        for stage_key, _stage_label in self._restraint_stage_rows:
            stage_forces = {name: 0.0 for name in component_selections}
            for rule in self._stage_restraint_rules[stage_key]:
                parsed = parse_restraint_asl(str(rule["asl"]))
                component = str(parsed["component"])
                force = float(rule["force"])
                if (
                    force > 0
                    and stage_forces[component] > 0
                    and stage_forces[component] != force
                ):
                    raise ValueError(
                        f"All {component} restraint groups in {stage_key} must "
                        "use the same force constant"
                    )
                if force <= 0:
                    continue
                stage_forces[component] = force
                if parsed not in component_selections[component]:
                    component_selections[component].append(parsed)
            forces_by_stage[stage_key] = [
                stage_forces["protein"],
                stage_forces["ligand"],
                stage_forces["cofactor"],
            ]
        released = [0.0, 0.0, 0.0]
        forces = [
            list(released),
            forces_by_stage["nvt"],
            forces_by_stage["npt"],
            list(released),
            list(released),
            list(released),
        ]

        selections: dict[str, dict[str, object]] = {}
        for component, parsed_rules in component_selections.items():
            if not parsed_rules:
                selections[component] = {
                    "component": component,
                    "residue_ids": [],
                    "residue_names": [],
                    "atom_names": [],
                }
                continue
            if component == "protein" and any(
                parsed["residue_names"] for parsed in parsed_rules
            ):
                raise ValueError("Protein selections support resi and name, not resn")
            if component in {"ligand", "cofactor"} and any(
                parsed["residue_ids"] for parsed in parsed_rules
            ):
                raise ValueError(
                    f"{component.title()} selections support resn and name, not resi"
                )
            dimensions = (
                ("residue_ids", "resi"),
                ("residue_names", "resn"),
                ("atom_names", "name"),
            )
            varying = []
            merged: dict[str, object] = {"component": component}
            for key, label in dimensions:
                values = [tuple(parsed[key]) for parsed in parsed_rules]
                if any(not value for value in values):
                    merged[key] = []
                    continue
                unique = set(values)
                if len(unique) > 1:
                    varying.append(label)
                merged[key] = sorted({item for value in values for item in value})
            if len(varying) > 1:
                raise ValueError(
                    f"The {component} groups vary in both {' and '.join(varying)}. "
                    "Split them using a common atom-name or residue selection so "
                    "they can be represented exactly in the GROMACS topology."
                )
            selections[component] = merged
        return {"forces": forces, "selections": selections}

    def _restraint_force_rows(self) -> list[list[float]]:
        return self._resolved_restraint_configuration()["forces"]

    @staticmethod
    def _lambda_text(values: list[float]) -> str:
        return " ".join(f"{value:.4f}" for value in values)

    def _update_lambda_ui(self, *_args) -> None:
        manual = self.lambda_schedule_mode.currentText() == "manual"
        self.lambda_windows.setEnabled(not manual)
        self.lambda_function.setEnabled(not manual)
        self.lambda_power.setEnabled(
            not manual and self.lambda_function.currentText() == "power"
        )
        for field in (
            self.coul_lambdas,
            self.vdw_lambdas,
            self.bonded_lambdas,
            self.restraint_lambdas,
        ):
            field.setEnabled(manual)
        self._update_expected_array_tasks()
        if manual:
            self.lambda_preview.setPlainText(
                "Enter all four schedules. The number of Coulomb values sets "
                "the window count; all lists must have the same length."
            )
            return
        progress = generate_lambda_windows(
            self.lambda_windows.value(),
            self.lambda_function.currentText(),
            self.lambda_power.value(),
        )
        vdw = [min(2.0 * value, 1.0) for value in progress]
        coulomb = [max(2.0 * value - 1.0, 0.0) for value in progress]
        self.lambda_preview.setPlainText(
            f"Coul: {self._lambda_text(coulomb)}\n"
            f"vdW:  {self._lambda_text(vdw)}\n"
            f"Bond: {self._lambda_text(vdw)}\n"
            f"Rest: {self._lambda_text([0.0] * len(progress))}"
        )

    def _write_config(self) -> Path:
        self._synchronize_system_type_with_input()
        name = self.job_name.text().strip()
        if not name:
            raise ValueError("Job name is required")
        network_bundle = self.network_bundle.text().strip()
        if not network_bundle:
            raise ValueError(
                "Select a ligand network bundle created by Ligand Network Planning"
            )
        network_ligands = Path(network_bundle) / "ligands.sdf"
        if not network_ligands.is_file():
            raise ValueError(
                "The selected network bundle does not contain ligands.sdf"
            )
        project = Path(self.working_directory) / name
        project.mkdir(parents=True, exist_ok=True)
        charge_manifest = self._write_charge_manifest(project)
        charge_change_method = str(self.charge_change_method.currentData())
        correction_file = self.charge_correction_file.text().strip()
        cofactor_resname, _cofactor_charge = self._cofactor_identity()
        restraint_configuration = self._resolved_restraint_configuration()
        restraint_forces = restraint_configuration["forces"]
        restraint_selections = restraint_configuration["selections"]
        stage_protocol = self._stage_protocol_settings()
        protein_residue_ids = restraint_selections["protein"]["residue_ids"]
        protein_atom_names = restraint_selections["protein"]["atom_names"]
        ligand_resnames = restraint_selections["ligand"]["residue_names"]
        ligand_atom_names = restraint_selections["ligand"]["atom_names"]
        cofactor_resnames = restraint_selections["cofactor"]["residue_names"] or (
            [cofactor_resname] if cofactor_resname else []
        )
        cofactor_atom_names = restraint_selections["cofactor"]["atom_names"]
        if charge_change_method == "correction" and not correction_file:
            raise ValueError(
                "Select a charge correction CSV when using the correction-table method"
            )
        path = project / f"{name}_ligand_rbfe.toml"
        planner = self.network_planner
        hub = planner.radial_hub.text().strip()
        worker = self.worker_script.text().strip()
        schedule_mode = self.lambda_schedule_mode.currentText()
        manual_text = ""
        windows = self.lambda_windows.value()
        if schedule_mode == "manual":
            vectors = {
                "coul_lambdas": self._parse_lambda_values(
                    self.coul_lambdas.text(), "Manual Coulomb lambdas"
                ),
                "vdw_lambdas": self._parse_lambda_values(
                    self.vdw_lambdas.text(), "Manual vdW lambdas"
                ),
                "bonded_lambdas": self._parse_lambda_values(
                    self.bonded_lambdas.text(), "Manual bonded lambdas"
                ),
                "restraint_lambdas": self._parse_lambda_values(
                    self.restraint_lambdas.text(), "Manual restraint lambdas"
                ),
            }
            windows = len(vectors["coul_lambdas"])
            manual_text = "".join(
                f"{key} = {json.dumps(value)}\n" for key, value in vectors.items()
            )
        text = (
            "[project]\n"
            f"name = {json.dumps(name)}\n"
            f"work_dir = {json.dumps(self.working_directory)}\n\n"
            "[system]\n"
            f"type = {json.dumps(self.system_type.currentText())}\n"
            f"protein = {json.dumps(self.protein_path.text().strip())}\n"
            'gromacs_command = "gmx"\n'
            f"force_field = {json.dumps(self.force_field.currentText().strip())}\n"
            f"water_model = {json.dumps(self._workflow_water_model())}\n"
            f"box_shape = {json.dumps(self.box_shape.currentText())}\n"
            f"box_distance = {self.box_distance.value()}\n"
            f"neutralize = {str(self.neutralize_system.isChecked()).lower()}\n"
            f"ion_concentration = {self.ion_strength.value()}\n"
            f"temperature_k = {self.temperature.value()}\npressure_bar = 1.0\n"
            + (
                f"prepared_complex_gro = {json.dumps(self.prepared_complex_gro.text().strip())}\n"
                if self.prepared_complex_gro.text().strip()
                else ""
            )
            + (
                f"prepared_complex_topology = {json.dumps(self.prepared_complex_topology.text().strip())}\n"
                if self.prepared_complex_topology.text().strip()
                else ""
            )
            + 'membrane_normal = "z"\nreference_pressure_bar = [1.0, 1.0]\ncompressibility_bar = [4.5e-5, 4.5e-5]\nrefcoord_scaling = "com"\n\n'
            "[force_field_components]\n"
            f"family = {json.dumps(self.force_field_family.currentText())}\n"
            f"protein = {json.dumps(self.force_field.currentText().strip())}\n"
            f"lipid = {json.dumps(self.lipid_force_field.currentText())}\n"
            f"nucleic_acid = {json.dumps(self.nucleic_acid_force_field.currentText())}\n"
            f"water = {json.dumps(self.water_model.currentText())}\n"
            f"ions = {json.dumps(self.ion_force_field.currentText())}\n"
            f"ligand = {json.dumps(self.ligand_force_field.currentText())}\n"
            f"cofactor = {json.dumps('GAFF2' if self.force_field_family.currentData() == 'amber' and self.cofactor_sdf.text().strip() else 'None')}\n\n"
            "[ligands]\n"
            f"library_sdf = {json.dumps(str(network_ligands))}\n"
            f"charge_manifest = {json.dumps(str(charge_manifest))}\n\n"
            "[mapping]\n"
            f"existing_network_dir = {json.dumps(network_bundle)}\n"
            f"network_method = {json.dumps(planner.method.currentText())}\n"
            f"atom_mapper = {json.dumps(planner.mapper.currentText())}\n"
            'pmx_command = "pmx"\n'
            f"pmx_mapping_mode = {json.dumps(planner.pmx_mode.currentText())}\n"
            "pmx_h2h_polar = true\n"
            f"element_change = {str(planner.element_change.isChecked()).lower()}\n"
            f"max3d_angstrom = {planner.max3d.value()}\n"
            f"require_cycle_covering = {str(planner.cycle_covering.isChecked()).lower()}\n"
            f"redundant_networks = {planner.redundant.value()}\n"
            + (f"radial_hub = {json.dumps(hub)}\n" if hub else "")
            + '\n[topology]\nbackend = "pmx"\ncharge_method = "assigned"\n'
            + f"charge_change_method = {json.dumps(charge_change_method)}\n"
            + (
                f"charge_correction_file = {json.dumps(correction_file)}\n"
                if correction_file and charge_change_method == "correction"
                else ""
            )
            + "\n"
            f"[simulation]\nstart_replicate = {self.start_replicate.value()}\n"
            f"replicates = {self.replicates.value()}\n"
            f"sampling_method = {json.dumps(str(self.sampling_method.currentData()))}\n"
            f"exchange_interval_steps = {self.exchange_interval.value()}\n"
            f"lambda_windows = {windows}\n"
            f"lambda_schedule_mode = {json.dumps(schedule_mode)}\n"
            f"lambda_function = {json.dumps(self.lambda_function.currentText())}\n"
            f"lambda_power = {self.lambda_power.value()}\n"
            + manual_text
            + f"timestep_ps = {self.timestep.value()}\n"
            f"em_steps = {self.em_steps.value()}\n"
            f"em_tolerance = {self.em_tolerance.value()}\n"
            f"nvt_ns = {self.nvt_duration.value()}\n"
            f"npt_ns = {self.npt_duration.value()}\n"
            f"production_ns = {self.production.value()}\n\n"
            "[stage_protocol]\n"
            f"integrators = {json.dumps(stage_protocol['integrators'])}\n"
            f"timesteps_ps = {json.dumps(stage_protocol['timesteps_ps'])}\n"
            f"temperature_coupling = {json.dumps(stage_protocol['temperature_coupling'])}\n"
            f"pressure_coupling = {json.dumps(stage_protocol['pressure_coupling'])}\n\n"
            "[restraints]\n"
            f"em_forces = {json.dumps(restraint_forces[0])}\n"
            f"nvt_forces = {json.dumps(restraint_forces[1])}\n"
            f"npt_forces = {json.dumps(restraint_forces[2])}\n"
            f"transition_forces = {json.dumps(restraint_forces[3])}\n"
            f"final_forces = {json.dumps(restraint_forces[4])}\n"
            f"production_forces = {json.dumps(restraint_forces[5])}\n"
            f"protein_residue_ids = {json.dumps(protein_residue_ids)}\n"
            f"protein_atom_names = {json.dumps(protein_atom_names)}\n"
            f"ligand_resnames = {json.dumps(ligand_resnames)}\n"
            f"ligand_atom_names = {json.dumps(ligand_atom_names)}\n"
            f"cofactor_resnames = {json.dumps(cofactor_resnames)}\n"
            f"cofactor_atom_names = {json.dumps(cofactor_atom_names)}\n\n"
            f"asl_rules_json = {json.dumps(json.dumps(self._stage_restraint_rules, sort_keys=True))}\n\n"
            "[mdp]\n"
            f"integrator = {json.dumps(self.mdp_integrator.currentText())}\n"
            f"constraints = {json.dumps(self.mdp_constraints.currentText())}\n"
            f"coulomb_cutoff_nm = {self.mdp_coulomb_cutoff.value()}\n"
            f"vdw_cutoff_nm = {self.mdp_vdw_cutoff.value()}\n"
            f"softcore_alpha = {self.mdp_softcore_alpha.value()}\n"
            f"calc_lambda_neighbors = {self.mdp_lambda_neighbors.value()}\n"
            f"nstdhdl = {self.mdp_nstdhdl.value()}\n\n"
            "[execution]\n"
            f"hrex_module = {json.dumps(self.hrex_module.currentText().strip())}\n"
            f"hrex_mpi_launcher = {json.dumps(self.hrex_mpi_launcher.currentText())}\n"
            f"hrex_gromacs_command = {json.dumps(self.hrex_gmx_command.text().strip())}\n"
            + (
                "mdrun_options = "
                + json.dumps(
                    [
                        "-ntmpi",
                        str(self.ntmpi.value()),
                        "-nb",
                        "gpu",
                        "-pme",
                        self.pme_target.currentText(),
                    ]
                )
                + "\n"
                if self.gpus.value() > 0
                else "mdrun_options = "
                + json.dumps(["-ntmpi", str(self.ntmpi.value())])
                + "\n"
            )
            + (f"worker_script = {json.dumps(worker)}\n" if worker else "")
            + "\n[slurm]\n"
            f"partition = {json.dumps(self.partition.text().strip())}\n"
            f"nodes = {self.nodes.value()}\n"
            f"tasks = {self.tasks.value()}\n"
            f"cpus_per_task = {self.cpus.value()}\n"
            f"gpus = {self.gpus.value()}\n"
            f"memory = {json.dumps(self.memory.text().strip())}\n"
            f"walltime = {json.dumps(self.walltime.text().strip())}\n"
            f"conda_init = {json.dumps(self.conda_init.text().strip())}\n"
            f"conda_environment = {json.dumps(self.conda_environment.text().strip())}\n"
        )
        path.write_text(text, encoding="utf-8")
        return path

    def create_task(self) -> None:
        try:
            output, preparation_runner, preparation_manifest = (
                self._stage_system_preparation()
            )
            config_path = self._write_config()
            project = config_path.parent
            workflow_config = LigandRBFEConfig.load(config_path)
            edges_path = Path(self.network_bundle.text().strip()) / "edges.csv"
            with edges_path.open(newline="", encoding="utf-8") as handle:
                edge_count = sum(1 for _row in csv.DictReader(handle))
            if edge_count > 0:
                write_slurm_array_files(workflow_config, edge_count=edge_count)
            combined_runner = project / "build_gromacs_rbfe_workflow.sh"
            combined_runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"bash {shlex.quote(str(preparation_runner))} "
                f"{shlex.quote(str(preparation_manifest))}\n"
                f"test -s {shlex.quote(str(output / 'system.gro'))}\n"
                f"test -s {shlex.quote(str(output / 'topol.top'))}\n"
                f"{shlex.quote(sys.executable)} -m alchemforge.cli "
                f"ligand-rbfe stage {shlex.quote(str(config_path))}\n"
                f"test -s {shlex.quote(str(project / 'build_hybrid_systems.sh'))} "
                "|| { echo 'RBFE staging did not create build_hybrid_systems.sh' "
                ">&2; exit 1; }\n"
                f"bash {shlex.quote(str(project / 'build_hybrid_systems.sh'))}\n",
                encoding="utf-8",
            )
            combined_runner.chmod(combined_runner.stat().st_mode | 0o100)
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.critical(self, "System Setup Failed", str(error))
            return
        self.system_build_status.setText("System build running in Job Monitor")
        self.system_build_status.setStyleSheet("color:#b26a00;font-weight:600")
        self.job_requested.emit(
            str(project), str(combined_runner), str(config_path)
        )
        QMessageBox.information(
            self,
            "GROMACS RBFE Build Started",
            f"Project: {project}\n\n"
            "The Job Monitor will assemble and parameterize the complete system, "
            "validate the ligand-free base system, build one PMX hybrid topology "
            "per network transformation, and create both complex and solvent legs.",
        )

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
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Show Network", str(error))
            return
            show_modeless(self, self._network_viewer)
        self._network_viewer.raise_()
        self._network_viewer.activateWindow()

    def run_worker(self) -> None:
        try:
            self._validate_system_form()
            config = LigandRBFEConfig.load(self._write_config())
            config.validate_inputs(require_runtime=True)
            reuse_network = (
                self._planned_edges is not None
                and self._planned_edges.is_file()
                and self._planned_edges == config.project_dir / "network" / "edges.csv"
            )
            generated, _ = stage(config, reuse_network=reuse_network)
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Run Ligand RBFE", str(error))
            return
        worker = config.project_dir / "LigandRBFE_GROMACS.sh"
        self.job_requested.emit(str(config.project_dir), str(worker), str(generated))
        self.accept()
