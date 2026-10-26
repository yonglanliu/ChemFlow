"""Desktop dialogs for OpenFE RBFE, ABFE, and SepTop JSON generation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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

from alchemforge.desktop.dialogs.ligand_network_planner_dialog import (
    LigandNetworkPlannerDialog,
)
from alchemforge.desktop.dialogs.network_viewer_dialog import (
    LigandNetworkViewerDialog,
)
from alchemforge.desktop.dialogs.openfe_charge_dialog import (
    OpenFEChargeMoleculesDialog,
)
from alchemforge.desktop.dialogs.openfe_execute_dialog import OpenFEExecuteDialog
from alchemforge.desktop.modeless import show_modeless
from alchemforge.openfe.workflow import (
    DEFAULT_FORCEFIELDS,
    OpenFEWorkflowConfig,
    stage,
)
from alchemforge.openfe.workflow_defaults import workflow_defaults

WORKFLOW_TITLES = {
    "rbfe": "OpenFE · LigandFEP-RBFE Workflow",
    "abfe": "OpenFE · LigandFEP-ABFE Workflow",
    "septop": "OpenFE · LigandSepTop-RBFE Workflow",
}

_DEFAULTS_CACHE: dict[str, dict[str, object]] = {}


class _OpenFEDefaultsLoader(QThread):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, workflow: str, parent=None):
        super().__init__(parent)
        self.workflow = workflow

    def run(self) -> None:
        try:
            defaults = workflow_defaults(self.workflow)
        except (
            AttributeError,
            ImportError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            self.failed.emit(str(error))
            return
        self.loaded.emit(defaults)


def orthorhombic_box_vectors(
    x_nm: float, y_nm: float, z_nm: float
) -> list[list[float]]:
    """Convert simple box lengths to the 3x3 vectors expected by OpenFE."""
    if min(x_nm, y_nm, z_nm) <= 0:
        raise ValueError("Enter positive X, Y, and Z box lengths in nanometers")
    return [[x_nm, 0.0, 0.0], [0.0, y_nm, 0.0], [0.0, 0.0, z_nm]]


_CORE_MODEL_PARAMETERS = {
    "platform",
    "temperature",
    "pressure",
    "membrane_system",
    "complex_box_vectors",
    "ligand_network_graphml",
    "mapper",
    "planner",
    "transformation",
    "alchemical_network",
    "protocol_repeats",
    "complex_protocol_repeats",
    "solvent_protocol_repeats",
    "forcefields",
    "solvent_model",
    "box_shape",
}


def _parameter_category(name: str) -> str:
    if "lambda" in name:
        return "Lambda Schedules"
    if any(token in name for token in ("restraint", "anchor", "host_", "rmsf")):
        return "Restraints"
    if any(token in name for token in ("solvation", "solvent_model", "ion_")):
        return "Solvation"
    if any(
        token in name
        for token in (
            "softcore",
            "annihilate",
            "dispersion",
            "charge_correction",
            "core_unique",
        )
    ):
        return "Alchemical"
    if any(
        token in name
        for token in (
            "forcefield",
            "nonbonded",
            "rigid_water",
            "constraint",
            "hydrogen_mass",
            "partial_charge",
            "toolkit",
        )
    ):
        return "Force Field & Charges"
    if any(
        token in name
        for token in (
            "mini_",
            "equil_",
            "prod_",
            "timestep",
            "barostat",
            "termination",
            "surface_tension",
        )
    ):
        return "Simulation"
    return "Other"


class OpenFEModelSettingsDialog(QDialog):
    """Typed editor for every non-core setting consumed by create_json.py."""

    def __init__(self, parameters: dict, defaults: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenFE Model Setup — All Parameters")
        self.resize(920, 700)
        self._original = dict(defaults or parameters)
        self._tables: list[QTableWidget] = []
        layout = QVBoxLayout(self)
        note = QLabel(
            "Values come from the installed OpenFE protocol defaults. Lists use "
            "JSON/YAML syntax; units are shown in parameter names or tooltips in "
            "the main workflow."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        tabs = QTabWidget()
        categories: dict[str, list[tuple[str, object]]] = {}
        for name, value in parameters.items():
            categories.setdefault(_parameter_category(name), []).append((name, value))
        order = (
            "Force Field & Charges",
            "Alchemical",
            "Simulation",
            "Solvation",
            "Restraints",
            "Lambda Schedules",
            "Other",
        )
        for category in order:
            entries = sorted(categories.get(category, []))
            if not entries:
                continue
            table = QTableWidget(len(entries), 3)
            table.setHorizontalHeaderLabels(["Parameter", "Value", "Type"])
            table.verticalHeader().setVisible(False)
            table.setProperty("parameter_category", category)
            for row, (name, value) in enumerate(entries):
                key = QTableWidgetItem(name)
                key.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                value_item = QTableWidgetItem(json.dumps(value))
                value_item.setData(Qt.UserRole, name)
                value_type = QTableWidgetItem(type(value).__name__)
                value_type.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(row, 0, key)
                table.setItem(row, 1, value_item)
                table.setItem(row, 2, value_type)
            table.resizeColumnsToContents()
            table.horizontalHeader().setStretchLastSection(False)
            table.setColumnWidth(1, 390)
            tabs.addTab(table, category)
            self._tables.append(table)
        layout.addWidget(tabs, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.RestoreDefaults
            | QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        layout.addWidget(buttons)
        self._values = dict(parameters)

    def _restore_defaults(self) -> None:
        for table in self._tables:
            for row in range(table.rowCount()):
                name = table.item(row, 0).text()
                table.item(row, 1).setText(json.dumps(self._original[name]))

    def _validate_and_accept(self) -> None:
        values = {}
        try:
            for table in self._tables:
                for row in range(table.rowCount()):
                    name = table.item(row, 0).text()
                    value = yaml.safe_load(table.item(row, 1).text())
                    expected = self._original[name]
                    if isinstance(expected, bool) and not isinstance(value, bool):
                        raise TypeError(f"{name} must be true or false")
                    if isinstance(expected, list) and not isinstance(value, list):
                        raise TypeError(f"{name} must be a list")
                    if isinstance(expected, str) and not isinstance(value, str):
                        raise TypeError(f"{name} must be text")
                    if (
                        isinstance(expected, int)
                        and not isinstance(expected, bool)
                        and (not isinstance(value, int) or isinstance(value, bool))
                    ):
                        raise TypeError(f"{name} must be an integer")
                    if isinstance(expected, float) and not isinstance(
                        value, (int, float)
                    ):
                        raise TypeError(f"{name} must be numeric")
                    values[name] = value
        except (TypeError, ValueError, yaml.YAMLError) as error:
            QMessageBox.warning(self, "Invalid OpenFE Parameter", str(error))
            return
        self._values = values
        self.accept()

    def values(self) -> dict:
        return dict(self._values)


class OpenFEWorkflowDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(
        self, workflow: str, working_directory: str | None = None, parent=None
    ):
        super().__init__(parent)
        self.workflow = workflow
        self._network_viewer = None
        self.model_parameters: dict[str, object] = {}
        self._installed_model_defaults: dict[str, object] = {}
        self._defaults_loader: _OpenFEDefaultsLoader | None = None
        self.working_directory = str(Path(working_directory or ".").resolve())
        title_text = WORKFLOW_TITLES[workflow]
        self.setWindowTitle(title_text)
        self.resize(860, 760)

        root = QVBoxLayout(self)
        title = QLabel(title_text)
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")
        description = QLabel(
            "Configure the molecular system and OpenFE protocol. Ligand graph "
            "planning is a separate reusable step; RBFE and SepTop consume a "
            "previously saved network bundle."
        )
        description.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(description)
        self.tabs = QTabWidget()
        system_page = QWidget()
        system_layout = QVBoxLayout(system_page)
        model_page = QWidget()
        model_page_layout = QVBoxLayout(model_page)
        job_page = QWidget()
        job_page_layout = QVBoxLayout(job_page)
        execution_page = QWidget()
        execution_layout = QVBoxLayout(execution_page)
        self.ligand_preparation = OpenFEChargeMoleculesDialog(
            working_directory=self.working_directory,
            parent=self,
            embedded=True,
        )
        self.ligand_preparation.job_requested.connect(self.job_requested.emit)
        self.tabs.addTab(self.ligand_preparation, "Ligand Preparation")
        if workflow in {"rbfe", "septop"}:
            self.network_planner = LigandNetworkPlannerDialog(
                working_directory=self.working_directory,
                parent=self,
                embedded=True,
            )
            self.network_planner.target.setCurrentText("OpenFE")
            self.network_planner.network_created.connect(self._use_network_bundle)
            self.tabs.addTab(self.network_planner, "Network Creation")
        self.tabs.addTab(system_page, "System")
        self.tabs.addTab(model_page, "Model Setup")
        self.tabs.addTab(job_page, "Job Setting")
        self.tabs.addTab(execution_page, "Execution")
        root.addWidget(self.tabs, 1)

        inputs = QGroupBox("Inputs and Output")
        input_form = QFormLayout(inputs)
        input_form.setLabelAlignment(Qt.AlignRight)
        self.protein = QLineEdit()
        self.ligands = QLineEdit()
        self.network_bundle = QLineEdit()
        self.cofactor = QLineEdit()
        self.task_root = QLineEdit(self.working_directory)
        default_name = {
            "rbfe": "openfe_rbfe",
            "abfe": "openfe_abfe",
            "septop": "openfe_septop_rbfe",
        }[workflow]
        self.job_name = QLineEdit(default_name)
        input_form.addRow(
            "Biomolecular structure (PDB):",
            self._file_row(self.protein, "PDB (*.pdb)"),
        )
        ligand_label = (
            "Ligand SDF (one molecule):"
            if workflow == "abfe"
            else "Ligand SDF (multiple molecules):"
        )
        if workflow == "abfe":
            input_form.addRow(ligand_label, self._file_row(self.ligands, "SDF (*.sdf)"))
        else:
            input_form.addRow(
                "Ligand network bundle:",
                self._network_directory_row(self.network_bundle),
            )
        input_form.addRow(
            "Cofactor SDF (optional):", self._file_row(self.cofactor, "SDF (*.sdf)")
        )
        system_layout.addWidget(inputs)

        protocol = QGroupBox("Core Protocol Settings")
        protocol_form = QFormLayout(protocol)
        protocol_form.setLabelAlignment(Qt.AlignRight)
        self.platform = QComboBox()
        self.platform.addItems(["CPU", "CUDA", "OpenCL"])
        self.repeats = QSpinBox()
        self.repeats.setRange(1, 100)
        self.repeats.setValue(1)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(1.0, 1000.0)
        self.temperature.setDecimals(2)
        self.temperature.setValue(298.15)
        self.pressure = QDoubleSpinBox()
        self.pressure.setRange(0.001, 1000.0)
        self.pressure.setDecimals(3)
        self.pressure.setValue(1.0)
        protocol_form.addRow("OpenMM platform:", self.platform)
        protocol_form.addRow("Protocol repeats:", self.repeats)
        protocol_form.addRow("Temperature (K):", self.temperature)
        protocol_form.addRow("Pressure (bar):", self.pressure)

        self.mapper = QComboBox()
        self.mapper.addItems(["LomapAtomMapper", "KartografAtomMapper"])
        self.planner = QComboBox()
        self.planner.addItems(["minimal_spanning", "minimal_redundant", "radial"])
        self.transformation = QComboBox()
        self.transformation.addItems(["both", "complex", "solvent"])
        if workflow == "rbfe":
            protocol_form.addRow("Legs to generate:", self.transformation)
        model_page_layout.addWidget(protocol)

        system_forcefield = QGroupBox("System Force Fields and Solvation")
        system_forcefield_form = QFormLayout(system_forcefield)
        system_forcefield_form.setLabelAlignment(Qt.AlignRight)
        self.forcefields = QPlainTextEdit("\n".join(DEFAULT_FORCEFIELDS))
        self.forcefields.setMaximumHeight(115)
        self.forcefields.setToolTip(
            "One OpenMM XML file per line. Include compatible biomolecule, water, "
            "ion, lipid, or RNA parameter files required by the complete system."
        )
        self.solvent_model = QComboBox()
        self.solvent_model.addItems(["tip3p", "spce", "tip4pew", "tip5p"])
        self.box_shape = QComboBox()
        self.box_shape.addItems(["dodecahedron", "cube", "octahedron"])
        system_forcefield_form.addRow("System force-field XMLs:", self.forcefields)
        system_forcefield_form.addRow("Water model:", self.solvent_model)
        system_forcefield_form.addRow("Water-box shape:", self.box_shape)
        model_page_layout.addWidget(system_forcefield)

        model_setup = QGroupBox("Complete OpenFE Model Setup")
        model_layout = QHBoxLayout(model_setup)
        self.model_note = QLabel(
            "Additional protocol parameters are loaded from the installed OpenFE "
            "version only when you open the model editor."
        )
        self.model_note.setWordWrap(True)
        self.edit_model_button = QPushButton("Edit All Model Parameters…")
        self.edit_model_button.clicked.connect(self._edit_model_parameters)
        model_layout.addWidget(self.model_note, 1)
        model_layout.addWidget(self.edit_model_button)
        model_page_layout.addWidget(model_setup)
        model_page_layout.addStretch()

        membrane = QGroupBox("Membrane System")
        membrane_form = QFormLayout(membrane)
        membrane_form.setLabelAlignment(Qt.AlignRight)
        self.membrane_system = QCheckBox(
            "Protein PDB already contains protein + membrane + solvent"
        )
        self.box_x = self._box_length_spinbox()
        self.box_y = self._box_length_spinbox()
        self.box_z = self._box_length_spinbox()
        for field in (self.box_x, self.box_y, self.box_z):
            field.setEnabled(False)
            self.membrane_system.toggled.connect(field.setEnabled)
        self.triclinic_box = QCheckBox("Use advanced triclinic vectors")
        self.triclinic_box.setEnabled(False)
        self.advanced_box_vectors = QLineEdit()
        self.advanced_box_vectors.setPlaceholderText(
            "[[a_x,a_y,a_z],[b_x,b_y,b_z],[c_x,c_y,c_z]] nm"
        )
        self.advanced_box_vectors.setVisible(False)
        self.membrane_system.toggled.connect(self.triclinic_box.setEnabled)
        self.triclinic_box.toggled.connect(self.advanced_box_vectors.setVisible)
        membrane_form.addRow("Membrane input:", self.membrane_system)
        membrane_form.addRow("Box lengths (nm):", self._box_lengths_row())
        membrane_form.addRow("", self.triclinic_box)
        membrane_form.addRow("", self.advanced_box_vectors)
        system_layout.addWidget(membrane)
        system_layout.addStretch()

        job_settings = QGroupBox("Output and Job")
        job_form = QFormLayout(job_settings)
        job_form.setLabelAlignment(Qt.AlignRight)
        job_form.addRow("Task root:", self._directory_row(self.task_root))
        job_form.addRow("Job name:", self.job_name)
        self.slurm_partition = QLineEdit("gpu")
        self.slurm_nodes = QSpinBox()
        self.slurm_nodes.setRange(1, 1024)
        self.slurm_nodes.setValue(1)
        self.slurm_tasks = QSpinBox()
        self.slurm_tasks.setRange(1, 65536)
        self.slurm_tasks.setValue(1)
        self.slurm_cpus = QSpinBox()
        self.slurm_cpus.setRange(1, 1024)
        self.slurm_cpus.setValue(8)
        self.slurm_gpus = QSpinBox()
        self.slurm_gpus.setRange(0, 64)
        self.slurm_gpus.setValue(1)
        self.slurm_memory = QLineEdit("16G")
        self.slurm_walltime = QLineEdit("24:00:00")
        self.max_parallel_jobs = QSpinBox()
        self.max_parallel_jobs.setRange(1, 256)
        self.max_parallel_jobs.setValue(3)
        job_form.addRow("Slurm partition:", self.slurm_partition)
        job_form.addRow("Slurm nodes:", self.slurm_nodes)
        job_form.addRow("Slurm tasks:", self.slurm_tasks)
        job_form.addRow("Slurm CPUs per task:", self.slurm_cpus)
        job_form.addRow("Slurm GPUs:", self.slurm_gpus)
        job_form.addRow("Slurm memory:", self.slurm_memory)
        job_form.addRow("Slurm walltime:", self.slurm_walltime)
        job_form.addRow("Parallel transformations:", self.max_parallel_jobs)
        job_page_layout.addWidget(job_settings)

        note = QLabel(
            "RBFE and SepTop require at least two named molecules in the SDF. "
            "ABFE requires exactly one. Ligands should already contain partial charges."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #6b7280;")
        job_page_layout.addWidget(note)
        job_page_layout.addStretch()

        buttons = QHBoxLayout()
        buttons.addStretch()
        validate = QPushButton("Validate System")
        validate.clicked.connect(self.validate_system)
        execute = QPushButton("Run Generated JSONs…")
        execute.setToolTip(
            "Execute previously generated OpenFE transformations, three in parallel by default"
        )
        execute.clicked.connect(self._open_execution)
        create = QPushButton("Create OpenFE Job")
        create.clicked.connect(self.create_job)
        buttons.addWidget(validate)
        buttons.addWidget(execute)
        buttons.addWidget(create)
        execution_note = QLabel(
            "First create the OpenFE transformation JSON files. Then run the "
            "generated JSON directory locally with resumable parallel execution."
        )
        execution_note.setWordWrap(True)
        execution_note.setStyleSheet("color:#607991;")
        execution_layout.addWidget(execution_note)
        execution_layout.addStretch()
        execution_layout.addLayout(buttons)

    @staticmethod
    def _row(field: QLineEdit, button: QPushButton) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _open_ligand_preparation(self) -> None:
        self._ligand_preparation_dialog = OpenFEChargeMoleculesDialog(
            working_directory=self.working_directory,
            parent=self,
        )
        self._ligand_preparation_dialog.job_requested.connect(self.job_requested.emit)
        show_modeless(self, self._ligand_preparation_dialog)

    def _open_execution(self) -> None:
        self._execution_dialog = OpenFEExecuteDialog(
            working_directory=self.working_directory,
            parent=self,
        )
        self._execution_dialog.job_requested.connect(self.job_requested.emit)
        show_modeless(self, self._execution_dialog)

    @staticmethod
    def _box_length_spinbox() -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setRange(0.0, 1000.0)
        field.setDecimals(4)
        field.setSingleStep(0.1)
        field.setSuffix(" nm")
        field.setSpecialValueText("not set")
        return field

    def _box_lengths_row(self) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        for label, field in (
            ("X", self.box_x),
            ("Y", self.box_y),
            ("Z", self.box_z),
        ):
            layout.addWidget(QLabel(label))
            layout.addWidget(field)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _file_row(self, field: QLineEdit, file_filter: str) -> QWidget:
        button = QPushButton("Browse...")

        def select() -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select input file", self.working_directory, file_filter
            )
            if path:
                field.setText(path)

        button.clicked.connect(select)
        return self._row(field, button)

    def _edit_model_parameters(self) -> None:
        cached = _DEFAULTS_CACHE.get(self.workflow)
        if cached is not None and not self._installed_model_defaults:
            self._set_installed_defaults(cached)
        if not self._installed_model_defaults:
            self.edit_model_button.setEnabled(False)
            self.edit_model_button.setText("Loading OpenFE defaults…")
            self.model_note.setText(
                "Loading protocol defaults from the installed OpenFE version. "
                "The workflow window remains usable."
            )
            self._defaults_loader = _OpenFEDefaultsLoader(self.workflow, self)
            self._defaults_loader.loaded.connect(self._defaults_loaded)
            self._defaults_loader.failed.connect(self._defaults_failed)
            self._defaults_loader.start()
            return
        self._show_model_editor()

    def _defaults_loaded(self, defaults: object) -> None:
        values = dict(defaults)
        _DEFAULTS_CACHE[self.workflow] = values
        self._set_installed_defaults(values)
        self._show_model_editor()

    def _set_installed_defaults(self, defaults: dict[str, object]) -> None:
        self.model_parameters = {
            key: value
            for key, value in defaults.items()
            if key not in _CORE_MODEL_PARAMETERS
        }
        self._installed_model_defaults = dict(self.model_parameters)
        self.model_note.setText(
            f"{len(self.model_parameters)} additional parameters from the installed "
            "OpenFE defaults are available for this workflow."
        )
        self.edit_model_button.setText("Edit All Model Parameters…")
        self.edit_model_button.setEnabled(True)

    def _defaults_failed(self, message: str) -> None:
        self.edit_model_button.setText("Retry Loading Model Parameters…")
        self.edit_model_button.setEnabled(True)
        self.model_note.setText("Could not load the installed OpenFE defaults.")
        QMessageBox.warning(self, "Cannot Load OpenFE Defaults", message)

    def _show_model_editor(self) -> None:
        dialog = OpenFEModelSettingsDialog(
            self.model_parameters, self._installed_model_defaults, self
        )
        if dialog.exec_() == QDialog.Accepted:
            self.model_parameters = dialog.values()

    def _preview_network(self) -> None:
        bundle = Path(self.network_bundle.text()).expanduser().resolve()
        try:
            self._network_viewer = LigandNetworkViewerDialog(
                bundle / "edges.csv", bundle / "ligands.sdf", self
            )
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Preview Network", str(error))
            return
        show_modeless(self, self._network_viewer)

    def _directory_row(self, field: QLineEdit) -> QWidget:
        button = QPushButton("Browse...")

        def select() -> None:
            path = QFileDialog.getExistingDirectory(
                self, "Select task root", self.working_directory
            )
            if path:
                field.setText(path)

        button.clicked.connect(select)
        return self._row(field, button)

    def _network_directory_row(self, field: QLineEdit) -> QWidget:
        button = QPushButton("Browse...")

        def select() -> None:
            path = QFileDialog.getExistingDirectory(
                self, "Select ligand network bundle", self.working_directory
            )
            if path:
                self._use_network_bundle(path)

        button.clicked.connect(select)
        preview = QPushButton("Preview")
        preview.clicked.connect(self._preview_network)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)
        layout.addWidget(preview)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _use_network_bundle(self, path: str) -> None:
        bundle = Path(path)
        self.network_bundle.setText(str(bundle))
        ligand_copy = bundle / "ligands.sdf"
        if ligand_copy.is_file():
            self.ligands.setText(str(ligand_copy))

    def _box_vectors(self) -> list[list[float]] | None:
        if not self.membrane_system.isChecked():
            return None
        if self.triclinic_box.isChecked():
            text = self.advanced_box_vectors.text().strip()
            if not text:
                raise ValueError("Enter the three triclinic box vectors")
            value = ast.literal_eval(text)
            try:
                vectors = [[float(number) for number in row] for row in value]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Triclinic vectors must be a numeric 3 by 3 array"
                ) from error
            if len(vectors) != 3 or any(len(row) != 3 for row in vectors):
                raise ValueError("Triclinic vectors must be a numeric 3 by 3 array")
            return vectors
        return orthorhombic_box_vectors(
            self.box_x.value(), self.box_y.value(), self.box_z.value()
        )

    @staticmethod
    def _optional_path(field: QLineEdit) -> Path | None:
        value = field.text().strip()
        return Path(value) if value else None

    def _workflow_config(self) -> OpenFEWorkflowConfig:
        if (
            self.workflow in {"rbfe", "septop"}
            and not self.network_bundle.text().strip()
        ):
            raise ValueError(
                "Select a ligand network bundle created by Ligand Network Planning"
            )
        job_name = self.job_name.text().strip()
        task_root = Path(self.task_root.text().strip()).expanduser().resolve()
        task = task_root / job_name
        return OpenFEWorkflowConfig(
            workflow=self.workflow,
            job_name=job_name,
            protein_pdb=Path(self.protein.text().strip()).expanduser().resolve(),
            ligand_sdf=Path(self.ligands.text().strip()).expanduser().resolve(),
            cofactor_sdf=self._optional_path(self.cofactor),
            ligand_network_graphml=(
                Path(self.network_bundle.text()).expanduser().resolve()
                / "ligand_network.graphml"
                if self.workflow in {"rbfe", "septop"}
                else None
            ),
            task_directory=task,
            platform=self.platform.currentText(),
            membrane_system=self.membrane_system.isChecked(),
            complex_box_vectors=self._box_vectors(),
            mapper=self.mapper.currentText(),
            planner=self.planner.currentText(),
            transformation=self.transformation.currentText(),
            protocol_repeats=self.repeats.value(),
            temperature=self.temperature.value(),
            pressure=self.pressure.value(),
            forcefields=tuple(
                line.strip()
                for line in self.forcefields.toPlainText().splitlines()
                if line.strip()
            ),
            solvent_model=self.solvent_model.currentText(),
            box_shape=self.box_shape.currentText(),
            slurm_partition=self.slurm_partition.text().strip(),
            slurm_nodes=self.slurm_nodes.value(),
            slurm_tasks=self.slurm_tasks.value(),
            slurm_cpus_per_task=self.slurm_cpus.value(),
            slurm_gpus=self.slurm_gpus.value(),
            slurm_memory=self.slurm_memory.text().strip(),
            slurm_walltime=self.slurm_walltime.text().strip(),
            max_parallel_jobs=self.max_parallel_jobs.value(),
            model_parameters=self.model_parameters,
        )

    def validate_system(self) -> None:
        try:
            self._workflow_config().validate()
        except (OSError, SyntaxError, TypeError, ValueError) as error:
            QMessageBox.critical(self, "System Validation Failed", str(error))
            return
        QMessageBox.information(
            self,
            "System Valid",
            "The workflow inputs, force fields, solvation settings, box, and "
            "Slurm resources are internally consistent.",
        )

    def create_job(self) -> None:
        try:
            config = self._workflow_config()
            runner, config_path = stage(config)
        except (OSError, SyntaxError, TypeError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Create OpenFE Job", str(error))
            return
        self.job_requested.emit(
            str(config.task_directory), str(runner), str(config_path)
        )
        self.accept()


class OpenFERBFEDialog(OpenFEWorkflowDialog):
    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__("rbfe", working_directory, parent)


class OpenFEABFEDialog(OpenFEWorkflowDialog):
    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__("abfe", working_directory, parent)


class OpenFESepTopDialog(OpenFEWorkflowDialog):
    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__("septop", working_directory, parent)
