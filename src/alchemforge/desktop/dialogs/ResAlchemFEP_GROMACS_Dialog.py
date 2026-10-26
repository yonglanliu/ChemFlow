import shutil
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

from alchemforge.desktop.dialogs.gromacs_ligand_prepare_dialog import (
    GromacsLigandPrepareDialog,
)
from alchemforge.desktop.modeless import show_modeless


class ResAlchemFEPDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory=None, parent=None):
        super().__init__(parent)

        self.working_directory = working_directory

        self.setWindowTitle("ResAlchemFEP Setup")
        self.resize(900, 680)

        self.build_ui()

    def build_ui(self):
        root = QVBoxLayout(self)

        self.setWindowTitle("GROMACS · ResFEP-RBFE Workflow")
        title = QLabel("GROMACS · ResFEP-RBFE Workflow")
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")

        subtitle = QLabel(
            "Configure the residue mutation, system, simulation, "
            "and JOB SETTING parameters."
        )
        subtitle.setStyleSheet("color: #71839a;")

        root.addWidget(title)
        root.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.ligand_preparation = GromacsLigandPrepareDialog(
            working_directory=self.working_directory,
            parent=self,
            embedded=True,
        )
        self.ligand_preparation.job_requested.connect(self.job_requested.emit)
        self.tabs.addTab(self.ligand_preparation, "Ligand Preparation")
        self.tabs.addTab(self.build_system_tab(), "System")
        self.tabs.addTab(self.build_mutation_tab(), "Mutation")
        self.tabs.addTab(self.build_simulation_tab(), "FEP Simulation")
        self.tabs.addTab(self.build_job_setting_tab(), "Job Setting")

        root.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()

        run_btn = QPushButton("Create ResFEP-RBFE Workflow")
        run_btn.setStyleSheet(
            """
            QPushButton {
                background: #2f7ed8;
                color: white;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #286fbe;
            }
            """
        )
        run_btn.clicked.connect(self.create_task)

        submit_btn = QPushButton("Submit Job")
        submit_btn.setStyleSheet(
            """
            QPushButton {
                background: #28a745;
                color: white;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: 700;
            }
            """
        )
        submit_btn.clicked.connect(self.submit_job)

        buttons.addStretch()
        buttons.addWidget(run_btn)
        buttons.addWidget(submit_btn)

        root.addLayout(buttons)

    def _open_ligand_preparation(self):
        self._ligand_preparation_dialog = GromacsLigandPrepareDialog(
            working_directory=self.working_directory,
            parent=self,
        )
        self._ligand_preparation_dialog.job_requested.connect(
            self.job_requested.emit
        )
        show_modeless(self, self._ligand_preparation_dialog)

    def build_system_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setLabelAlignment(Qt.AlignRight)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(10)

        # Protein structure
        self.structure_edit = QLineEdit()
        self.structure_edit.setPlaceholderText("Select protein structure...")

        structure_row = QHBoxLayout()
        structure_row.setContentsMargins(0, 0, 0, 0)
        structure_row.addWidget(self.structure_edit)

        protein_browse_btn = QPushButton("Browse...")
        protein_browse_btn.clicked.connect(self.browse_structure)
        structure_row.addWidget(protein_browse_btn)

        protein_structure_container = QWidget()
        protein_structure_container.setLayout(structure_row)

        self.system_type_combo = QComboBox()
        self.system_type_combo.addItems(["soluble", "membrane"])
        self.membrane_gro_edit = QLineEdit()
        self.membrane_top_edit = QLineEdit()
        membrane_gro_row = QHBoxLayout()
        membrane_gro_row.setContentsMargins(0, 0, 0, 0)
        membrane_gro_row.addWidget(self.membrane_gro_edit)
        membrane_gro_button = QPushButton("Browse...")
        membrane_gro_button.clicked.connect(
            lambda: self.browse_into(
                self.membrane_gro_edit, "Select prepared membrane GRO", "GRO (*.gro)"
            )
        )
        membrane_gro_row.addWidget(membrane_gro_button)
        membrane_gro_container = QWidget()
        membrane_gro_container.setLayout(membrane_gro_row)
        membrane_top_row = QHBoxLayout()
        membrane_top_row.setContentsMargins(0, 0, 0, 0)
        membrane_top_row.addWidget(self.membrane_top_edit)
        membrane_top_button = QPushButton("Browse...")
        membrane_top_button.clicked.connect(
            lambda: self.browse_into(
                self.membrane_top_edit,
                "Select prepared membrane topology",
                "TOP (*.top)",
            )
        )
        membrane_top_row.addWidget(membrane_top_button)
        membrane_top_container = QWidget()
        membrane_top_container.setLayout(membrane_top_row)

        # Ligand structure
        self.ligand_edit = QLineEdit()
        self.ligand_edit.setPlaceholderText("Select ligand structure...")

        ligand_row = QHBoxLayout()
        ligand_row.setContentsMargins(0, 0, 0, 0)
        ligand_row.addWidget(self.ligand_edit)

        ligand_browse_btn = QPushButton("Browse...")
        ligand_browse_btn.clicked.connect(self.browse_ligand)
        ligand_row.addWidget(ligand_browse_btn)

        ligand_container = QWidget()
        ligand_container.setLayout(ligand_row)

        # Force field
        self.forcefield_combo = QComboBox()
        self.forcefield_combo.addItems(
            [
                "amber99sb-star-ildn-mut",
                "amber14sb",
                "charmm36m",
            ]
        )

        # Water model
        self.water_combo = QComboBox()
        self.water_combo.addItems(["TIP3P", "SPC/E"])

        # Water box
        self.box_shape_combo = QComboBox()
        self.box_shape_combo.addItems(["Octahedral", "Cubic", "Triclinic"])

        self.box_size_spin = QDoubleSpinBox()
        self.box_size_spin.setRange(0.1, 100.0)
        self.box_size_spin.setDecimals(2)
        self.box_size_spin.setSingleStep(0.1)
        self.box_size_spin.setValue(1.0)
        self.box_size_spin.setSuffix(" nm")

        # Neutralization / salt
        self.neutralize_checkbox = QCheckBox("Neutralize system")
        self.neutralize_checkbox.setChecked(True)

        self.add_salt_checkbox = QCheckBox("Add salt to target concentration")
        self.add_salt_checkbox.setChecked(True)

        self.positive_ion_combo = QComboBox()
        self.positive_ion_combo.addItems(["NA", "K"])

        self.negative_ion_combo = QComboBox()
        self.negative_ion_combo.addItems(["CL"])

        self.ion_strength_spin = QDoubleSpinBox()
        self.ion_strength_spin.setRange(0.0, 5.0)
        self.ion_strength_spin.setDecimals(3)
        self.ion_strength_spin.setSingleStep(0.05)
        self.ion_strength_spin.setValue(0.15)
        self.ion_strength_spin.setSuffix(" M")

        self.neutralize_checkbox.toggled.connect(self.update_ion_controls)
        self.add_salt_checkbox.toggled.connect(self.update_ion_controls)

        # Rows
        layout.addRow("Input Protein structure:", protein_structure_container)
        layout.addRow("System type:", self.system_type_combo)
        layout.addRow("Prepared membrane GRO:", membrane_gro_container)
        layout.addRow("Prepared membrane topology:", membrane_top_container)
        layout.addRow("Input Ligand structure:", ligand_container)
        layout.addRow("Protein force field:", self.forcefield_combo)
        layout.addRow("Water model:", self.water_combo)
        layout.addRow("Water box shape:", self.box_shape_combo)
        layout.addRow("Water box size:", self.box_size_spin)
        layout.addRow("", self.neutralize_checkbox)
        layout.addRow("", self.add_salt_checkbox)
        layout.addRow("Positive ion:", self.positive_ion_combo)
        layout.addRow("Negative ion:", self.negative_ion_combo)
        layout.addRow("Ion strength:", self.ion_strength_spin)

        self.update_ion_controls()
        return widget

    def update_ion_controls(self):
        use_ions = (
            self.neutralize_checkbox.isChecked() or self.add_salt_checkbox.isChecked()
        )

        self.positive_ion_combo.setEnabled(use_ions)
        self.negative_ion_combo.setEnabled(use_ions)
        self.ion_strength_spin.setEnabled(self.add_salt_checkbox.isChecked())

    def build_mutation_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setLabelAlignment(Qt.AlignRight)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(10)

        self.chain_edit = QLineEdit()
        self.chain_edit.setPlaceholderText("A")

        self.residue_spin = QSpinBox()
        self.residue_spin.setRange(1, 999999)

        amino_acids = [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ]

        self.to_residue_combo = QComboBox()
        self.to_residue_combo.addItems(amino_acids)

        layout.addRow("Chain:", self.chain_edit)
        layout.addRow("Residue number:", self.residue_spin)
        layout.addRow("To residue:", self.to_residue_combo)

        return widget

    def build_simulation_tab(self):
        widget = QWidget()
        root = QVBoxLayout(widget)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0, 1000)
        self.temperature_spin.setDecimals(1)
        self.temperature_spin.setValue(300.0)
        self.temperature_spin.setSuffix(" K")

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setDecimals(4)
        self.dt_spin.setRange(0.0001, 0.01)
        self.dt_spin.setSingleStep(0.001)
        self.dt_spin.setValue(0.002)
        self.dt_spin.setSuffix(" ps")

        self.lambda_mode_combo = QComboBox()
        self.lambda_mode_combo.addItems(["single"])

        self.nlambda_spin = QSpinBox()
        self.nlambda_spin.setRange(1, 1000)
        self.nlambda_spin.setValue(17)

        self.nrep_spin = QSpinBox()
        self.nrep_spin.setRange(1, 1000)
        self.nrep_spin.setValue(1)

        self.start_rep_spin = QSpinBox()
        self.start_rep_spin.setRange(1, 1000)
        self.start_rep_spin.setValue(1)

        self.fep_lambda_function_combo = QComboBox()
        self.fep_lambda_function_combo.addItems(["cosine", "linear", "power"])

        self.fep_lambda_power_spin = QDoubleSpinBox()
        self.fep_lambda_power_spin.setRange(0.1, 10.0)
        self.fep_lambda_power_spin.setDecimals(2)
        self.fep_lambda_power_spin.setValue(2.0)

        self.fep_nvt_time_spin = QDoubleSpinBox()
        self.fep_nvt_time_spin.setRange(0.001, 100000.0)
        self.fep_nvt_time_spin.setDecimals(3)
        self.fep_nvt_time_spin.setValue(0.5)
        self.fep_nvt_time_spin.setSuffix(" ns")

        self.fep_npt_time_spin = QDoubleSpinBox()
        self.fep_npt_time_spin.setRange(0.001, 100000.0)
        self.fep_npt_time_spin.setDecimals(3)
        self.fep_npt_time_spin.setValue(1.0)
        self.fep_npt_time_spin.setSuffix(" ns")

        self.fep_production_time_spin = QDoubleSpinBox()
        self.fep_production_time_spin.setRange(0.001, 100000.0)
        self.fep_production_time_spin.setDecimals(3)
        self.fep_production_time_spin.setValue(10.0)
        self.fep_production_time_spin.setSuffix(" ns")

        self.max_fep_jobs_spin = QSpinBox()
        self.max_fep_jobs_spin.setRange(1, 1000000)
        self.max_fep_jobs_spin.setValue(3)

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
                ("NVT", "MD", "Configured", "Temperature coupling", "No coupling"),
                ("NPT", "MD", "Configured", "Temperature coupling", "System-dependent"),
                ("Production", "FEP MD", "Configured", "Temperature coupling", "System-dependent"),
            )
        ):
            for column, value in enumerate(values):
                self.stage_protocol_table.setItem(row, column, QTableWidgetItem(value))
        self.stage_protocol_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stage_protocol_table.verticalHeader().setVisible(False)
        self.stage_protocol_table.horizontalHeader().setStretchLastSection(True)
        stage_layout.addWidget(self.stage_protocol_table)
        stage_form = QFormLayout()
        stage_form.setLabelAlignment(Qt.AlignRight)
        stage_form.addRow("FEP NVT time:", self.fep_nvt_time_spin)
        stage_form.addRow("FEP NPT time:", self.fep_npt_time_spin)
        stage_form.addRow("FEP production time:", self.fep_production_time_spin)
        stage_layout.addLayout(stage_form)
        self.simulation_tabs.addTab(stage_page, "Stage Protocol")

        restraint_page = QWidget()
        restraint_layout = QVBoxLayout(restraint_page)
        restraint_note = QLabel(
            "ResFEP uses the protein/residue restraints supplied by the prepared "
            "PMX mutation topology. No Boresch ligand restraint is used."
        )
        restraint_note.setWordWrap(True)
        restraint_layout.addWidget(restraint_note)
        restraint_layout.addStretch()
        self.simulation_tabs.addTab(restraint_page, "Restraints")

        lambda_page = QWidget()
        lambda_layout = QFormLayout(lambda_page)
        lambda_layout.setLabelAlignment(Qt.AlignRight)
        lambda_layout.addRow("Lambda mode:", self.lambda_mode_combo)
        lambda_layout.addRow("Lambda windows:", self.nlambda_spin)
        lambda_layout.addRow("Replicates:", self.nrep_spin)
        lambda_layout.addRow("Start replicate:", self.start_rep_spin)
        lambda_layout.addRow("Lambda function:", self.fep_lambda_function_combo)
        lambda_layout.addRow("Lambda power:", self.fep_lambda_power_spin)
        self.simulation_tabs.addTab(lambda_page, "Lambda Schedule")

        alchemical_page = QWidget()
        alchemical_layout = QFormLayout(alchemical_page)
        alchemical_layout.setLabelAlignment(Qt.AlignRight)
        alchemical_layout.addRow("Temperature:", self.temperature_spin)
        alchemical_layout.addRow("Time step:", self.dt_spin)
        alchemical_layout.addRow("Max FEP jobs:", self.max_fep_jobs_spin)
        self.simulation_tabs.addTab(alchemical_page, "Alchemical Controls")

        root.addWidget(self.simulation_tabs)

        return widget

    def build_job_setting_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        common_form = QFormLayout()
        self.job_name_edit = QLineEdit("ResAlchemFEP")
        common_form.addRow("Job name:", self.job_name_edit)
        layout.addLayout(common_form)

        self.task_settings = {}
        task_tabs = QTabWidget()
        for task_name, title in (
            ("ligand", "Ligand Parameterization"),
            ("system", "System Setup"),
            ("fep", "Simulation"),
        ):
            task_tabs.addTab(self.build_task_settings(task_name), title)
        layout.addWidget(task_tabs)

        return widget

    def build_task_settings(self, task_name):
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setLabelAlignment(Qt.AlignRight)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(10)

        host_combo = QComboBox()
        host_combo.addItem("localhost")

        if task_name in {"ligand", "system"}:
            layout.addRow("Host:", QLabel("localhost"))
            self.task_settings[task_name] = {"host": host_combo}
            return widget

        host_combo.addItem("slurm")
        host_combo.setCurrentText("slurm")
        partition_edit = QLineEdit("gpu")

        nodes_spin = QSpinBox()
        nodes_spin.setRange(1, 1000)
        nodes_spin.setValue(1)
        tasks_spin = QSpinBox()
        tasks_spin.setRange(1, 1000)
        tasks_spin.setValue(1)
        cpus_spin = QSpinBox()
        cpus_spin.setRange(1, 512)
        cpus_spin.setValue(8)
        gpus_spin = QSpinBox()
        gpus_spin.setRange(0, 16)
        gpus_spin.setValue(1)
        memory_edit = QLineEdit("16G")
        walltime_edit = QLineEdit("08:00:00")

        layout.addRow("Host:", host_combo)
        layout.addRow("SLURM partition:", partition_edit)
        layout.addRow("Nodes:", nodes_spin)
        layout.addRow("Tasks:", tasks_spin)
        layout.addRow("CPUs per task:", cpus_spin)
        layout.addRow("GPUs:", gpus_spin)
        layout.addRow("Memory:", memory_edit)
        layout.addRow("Walltime:", walltime_edit)

        self.task_settings[task_name] = {
            "host": host_combo,
            "partition": partition_edit,
            "nodes": nodes_spin,
            "tasks": tasks_spin,
            "cpus_per_task": cpus_spin,
            "gpus": gpus_spin,
            "memory": memory_edit,
            "walltime": walltime_edit,
        }

        return widget

    def browse_structure(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Protein Structure",
            self.working_directory or "",
            "Protein Structure Files (*.pdb *.gro *.mae *.maegz);;All Files (*)",
        )

        if path:
            self.structure_edit.setText(path)

    def browse_ligand(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ligand Structure",
            self.working_directory or "",
            "Ligand Structure Files (*.sdf *.mol2 *.mol *.pdb *.mae *.maegz);;All Files (*)",
        )

        if path:
            self.ligand_edit.setText(path)

    def browse_into(self, field, title, file_filter):
        path, _ = QFileDialog.getOpenFileName(
            self, title, self.working_directory or "", file_filter
        )
        if path:
            field.setText(path)

    def collect_config(self):
        return {
            "system": {
                "protein_structure": self.structure_edit.text().strip(),
                "system_type": self.system_type_combo.currentText(),
                "prepared_complex_gro": self.membrane_gro_edit.text().strip(),
                "prepared_complex_topology": self.membrane_top_edit.text().strip(),
                "ligand_structure": self.ligand_edit.text().strip(),
                "forcefield": self.forcefield_combo.currentText(),
                "water_model": self.water_combo.currentText(),
                "box_shape": self.box_shape_combo.currentText(),
                "box_size_nm": self.box_size_spin.value(),
                "neutralize": self.neutralize_checkbox.isChecked(),
                "add_salt": self.add_salt_checkbox.isChecked(),
                "positive_ion": self.positive_ion_combo.currentText(),
                "negative_ion": self.negative_ion_combo.currentText(),
                "ion_strength_m": self.ion_strength_spin.value(),
            },
            "mutation": {
                "chain": self.chain_edit.text().strip(),
                "residue_number": self.residue_spin.value(),
                "to_residue": self.to_residue_combo.currentText(),
            },
            "simulation": {
                "temperature": self.temperature_spin.value(),
                "dt": self.dt_spin.value(),
                "lambda_mode": self.lambda_mode_combo.currentText(),
                "nlambda": self.nlambda_spin.value(),
                "nrep": self.nrep_spin.value(),
                "start_rep": self.start_rep_spin.value(),
                "fep_lambda_function": self.fep_lambda_function_combo.currentText(),
                "fep_lambda_power": self.fep_lambda_power_spin.value(),
                "fep_nvt_time": self.fep_nvt_time_spin.value(),
                "fep_npt_time": self.fep_npt_time_spin.value(),
                "fep_production_time": self.fep_production_time_spin.value(),
                "max_fep_jobs": self.max_fep_jobs_spin.value(),
            },
            "job_setting": {
                "job_name": self.job_name_edit.text().strip(),
                "tasks": {
                    task_name: (
                        {"host": fields["host"].currentText()}
                        if task_name in {"ligand", "system"}
                        else {
                            "host": fields["host"].currentText(),
                            "partition": fields["partition"].text().strip(),
                            "nodes": fields["nodes"].value(),
                            "tasks": fields["tasks"].value(),
                            "cpus_per_task": fields["cpus_per_task"].value(),
                            "gpus": fields["gpus"].value(),
                            "memory": fields["memory"].text().strip(),
                            "walltime": fields["walltime"].text().strip(),
                        }
                    )
                    for task_name, fields in self.task_settings.items()
                },
            },
        }

    @staticmethod
    def extract_ligand_info(mol2_path):
        """Extract the ligand residue name and net charge from a MOL2 file."""
        residue_name = None
        total_charge = 0.0
        atom_count = 0
        in_atom_section = False

        with open(mol2_path, "r", encoding="utf-8") as mol2_file:
            for line in mol2_file:
                section = line.strip()
                if section.startswith("@<TRIPOS>"):
                    in_atom_section = section == "@<TRIPOS>ATOM"
                    continue

                if not in_atom_section or not section:
                    continue

                fields = section.split()
                if len(fields) < 9:
                    raise ValueError(
                        f"Invalid MOL2 atom record in {mol2_path}: {line.strip()}"
                    )

                if residue_name is None:
                    residue_name = fields[7]

                try:
                    total_charge += float(fields[8])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid MOL2 atom charge in {mol2_path}: {line.strip()}"
                    ) from exc

                atom_count += 1

        if not residue_name or atom_count == 0:
            raise ValueError(
                f"No atom records were found in the MOL2 file: {mol2_path}"
            )

        return residue_name, total_charge

    def save_config(self, checked=False, file_path=None, config=None):
        config = config or self.collect_config()

        job_name = config["job_setting"]["job_name"]
        if not job_name:
            QMessageBox.warning(
                self,
                "Missing Job Name",
                "Please enter a job name.",
            )
            return

        task_directory = (
            Path(self.working_directory or ".").expanduser().resolve() / job_name
        )
        try:
            task_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Could not create task directory:\n\n{exc}",
            )
            return

        if file_path is None:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save ResAlchemFEP Configuration",
                str(task_directory / f"{job_name}_config.inp"),
                "Input Configuration (*.inp);;All Files (*)",
            )

        # User clicked Cancel
        if not file_path:
            return

        file_path = str(file_path)

        # Ensure .inp extension
        if not file_path.endswith(".inp"):
            file_path += ".inp"

        system = config["system"]
        mutation = config["mutation"]
        simulation = config["simulation"]
        job_setting = config["job_setting"]
        task_settings = job_setting["tasks"]

        neutralize = str(system["neutralize"]).lower()

        add_salt = str(system["add_salt"]).lower()

        work_directory = str(Path(self.working_directory or ".").expanduser().resolve())
        protein_path = system["protein_structure"]
        ligand_path = system["ligand_structure"]
        ligand_stem = Path(ligand_path).stem
        ligand_itp = str(Path(ligand_path).with_name(f"{ligand_stem}-out.itp"))
        ligand_gro = str(Path(ligand_path).with_name(f"{ligand_stem}-out.gro"))
        water_model = system["water_model"].lower()
        system_setup_mode = (
            "SKIP" if system["system_type"] == "membrane" else "OVERWRITE"
        )
        if system["system_type"] == "membrane" and not (
            system["prepared_complex_gro"] and system["prepared_complex_topology"]
        ):
            QMessageBox.critical(
                self,
                "Missing Membrane Inputs",
                "Membrane residue FEP requires prepared GRO and topology files.",
            )
            return

        try:
            ligand_resname, ligand_charge = self.extract_ligand_info(ligand_path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self,
                "Invalid Ligand MOL2",
                f"Could not extract ligand residue name and charge:\n\n{exc}",
            )
            return

        ligand_charge = int(round(ligand_charge))

        script = f"""#!/bin/bash
# ============================================================
# Configuration
# ============================================================

WORK_DIR="{work_directory}"  # Project root directory containing the workflow and input files

JOB_NAME="{job_setting["job_name"]}"  # Name of the project/job directory under ROOT

GROMACS_MODULE="gromacs/2024.4-gcc11.3"
AMBERTOOLS_MODULE="amber/22-ambertools23.gcc"

LIGAND_HOST="{task_settings["ligand"]["host"]}"

SYSTEM_HOST="{task_settings["system"]["host"]}"

FEP_HOST="{task_settings["fep"]["host"]}"
FEP_PARTITION="{task_settings["fep"]["partition"]}"
FEP_NODES={task_settings["fep"]["nodes"]}
FEP_NTASKS={task_settings["fep"]["tasks"]}
FEP_CPUS_PER_TASK={task_settings["fep"]["cpus_per_task"]}
FEP_GPUS={task_settings["fep"]["gpus"]}
FEP_MEMORY="{task_settings["fep"]["memory"]}"
FEP_WALLTIME="{task_settings["fep"]["walltime"]}"


# ============================================================
# Mutation definition
# ============================================================

CHAIN="{mutation["chain"]}"  # Chain ID of the residue being mutated
RESID={mutation["residue_number"]}  # Residue number to mutate
MUT="{mutation["to_residue"]}"  # Target residue name after mutation


# ============================================================
# Force field
# ============================================================

FF="{system["forcefield"]}"  # Force field used for the protein and hybrid topology
WATER="{water_model}"  # Water model used during solvation


# ============================================================
# Input Information
# ============================================================

COMPLEX="{protein_path}"  # Protein-ligand complex structure used as the starting point

SYSTEM_TYPE="{system["system_type"]}"
PREPARED_COMPLEX_GRO="{system["prepared_complex_gro"]}"
PREPARED_COMPLEX_TOPOLOGY="{system["prepared_complex_topology"]}"
MEMBRANE_NORMAL="z"
MEMBRANE_BAROSTAT="Parrinello-Rahman"
MEMBRANE_REF_P="1.0 1.0"
MEMBRANE_COMPRESSIBILITY="4.5e-5 4.5e-5"
MEMBRANE_REFCOORD_SCALING="com"

LIGAND_MOL2="{ligand_path}"  # Input ligand structure in MOL2 format

LIGAND_CHARGE_METHOD="bcc"  # Charge assignment method for the ligand: bcc or user

LIGAND_ITP="{ligand_itp}"  # Ligand topology file (ITP) used in the bound setup

LIGAND_GRO="{ligand_gro}"  # Ligand coordinate file in GRO format

LIGAND_RESNAME="{ligand_resname}"  # Residue name of the ligand as defined in the topology

LIGAND_CHARGE={ligand_charge:.4f}  # Expected net charge of the ligand


# ============================================================
# System preparation
# ============================================================

# Minimum distance between solute and box edge, nm
BOX_DISTANCE={system["box_size_nm"]}  # Box padding around the solute, in nanometers

# Salt concentration, mol/L
ION_CONC={system["ion_strength_m"]}  # Final ionic strength used for neutralization, mol/L

# Temperature, K
TEMPERATURE={simulation["temperature"]}  # Simulation temperature in kelvin

# Pressure, bar
PRESSURE=1.0  # Simulation pressure in bar


# ============================================================
# GROMACS
# ============================================================

GMX="gmx"  # GROMACS executable to use for setup and MD runs


# ============================================================
# CPU / GPU settings
# ============================================================
MDRUN_EM_OPTIONS="
-ntmpi 1
"  # MPI settings for the energy minimization step

MDRUN_MD_OPTIONS="
-ntmpi 1
-nb gpu
-pme gpu
"  # GPU-enabled settings for NVT/NPT equilibration and general MD runs


# Equilibrium FEP production
MDRUN_FEP_OPTIONS="
-ntmpi 1
-nb gpu
-pme gpu
"  # GPU settings used for the production FEP window runs


# Nonequilibrium switching
MDRUN_SWITCH_OPTIONS="
-ntmpi 1
-nb gpu
-pme gpu
"  # GPU settings used for nonequilibrium switching runs if enabled


# ============================================================
# General MD settings
# ============================================================
NREP={simulation["nrep"]}             # Number of independent replicate simulations per leg
START_REP={simulation["start_rep"]}        # First replicate index used for the job set

DT_PS={simulation["dt"]}        # MD timestep in picoseconds


# ------------------------------------------------------------
# FEP simulation
# ------------------------------------------------------------
LAMBDA_MODE="{simulation["lambda_mode"]}"  # Lambda schedule mode for the alchemical transformation
NLAMBDA={simulation["nlambda"]}             # Number of lambda windows for the FEP schedule

FEP_LAMBDA_FUNCTION="{simulation["fep_lambda_function"]}"  # Lambda spacing scheme: cosine, linear, or power
FEP_LAMBDA_POWER={simulation["fep_lambda_power"]}           # Exponent used when lambda function is set to power

FEP_DT_PS={simulation["dt"]}                 # Timestep used for the FEP MD integration, in ps

FEP_NVT_PS={simulation["fep_nvt_time"]}                  # NVT equilibration length, in ns
FEP_NPT_PS={simulation["fep_npt_time"]}                  # NPT equilibration length, in ns
FEP_PROD_NS={simulation["fep_production_time"]}                  # Production run length, in ns

MAX_FEP_JOBS={simulation["max_fep_jobs"]}  # Maximum simultaneous lambda-window jobs allowed

FEP_NVT_STEPS=$(
    awk \
        -v time_ps="${{FEP_NVT_PS}}" \
        -v dt="${{FEP_DT_PS}}" \
        'BEGIN {{
            printf "%.0f", time_ps * 1000.0 / dt
        }}'
)

FEP_NPT_STEPS=$(
    awk \
        -v time_ps="${{FEP_NPT_PS}}" \
        -v dt="${{FEP_DT_PS}}" \
        'BEGIN {{
            printf "%.0f", time_ps * 1000.0 / dt
        }}'
)

FEP_PROD_STEPS=$(
    awk \
        -v time_ns="${{FEP_PROD_NS}}" \
        -v dt="${{FEP_DT_PS}}" \
        'BEGIN {{
            printf "%.0f", time_ns * 1000.0 / dt
        }}'
)

N_FEP_TASKS=$(( 2 * NREP * NLAMBDA ))  # Expected equilibrium FEP array size

# Preparation and simulation modes
LIGAND_PREP_MODE="OVERWRITE"  # Mode for ligand preparation: OVERWRITE or SKIP
SYSTEM_SETUP_MODE="{system_setup_mode}"  # Membranes must preserve prepared topology
SIMULATION_MODE="OVERWRITE"  # Mode for simulation setup: OVERWRITE, SKIP, or RESUME

"""

        try:
            with open(
                file_path,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(script)

            QMessageBox.information(
                self,
                "Configuration Saved",
                f"Configuration saved successfully:\n\n{file_path}",
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Could not save configuration:\n\n{exc}",
            )

    def create_task(self):
        config = self.collect_config()

        if not config["system"]["protein_structure"]:
            QMessageBox.warning(
                self,
                "Missing Protein Structure",
                "Please select an input protein structure.",
            )
            return

        if not config["system"]["ligand_structure"]:
            QMessageBox.warning(
                self,
                "Missing Ligand Structure",
                "Please select an input ligand structure.",
            )
            return

        job_name = config["job_setting"]["job_name"]
        if not job_name:
            QMessageBox.warning(
                self,
                "Missing Job Name",
                "Please enter a job name.",
            )
            return

        task_directory = (
            Path(self.working_directory or ".").expanduser().resolve() / job_name
        )
        protein_source = (
            Path(config["system"]["protein_structure"]).expanduser().resolve()
        )
        ligand_source = (
            Path(config["system"]["ligand_structure"]).expanduser().resolve()
        )
        protein_destination = (
            task_directory / f"{job_name}_protein{protein_source.suffix}"
        )
        ligand_destination = task_directory / f"{job_name}_ligand{ligand_source.suffix}"
        submission_directory = (
            Path(__file__).resolve().parents[2] / "gromacs" / "resFEP"
        )
        submission_files = ("para_ligand.sh", "setup.sh", "run_pipeline.sh")
        submission_names = {
            "para_ligand.sh": "01_para_ligand.sh",
            "setup.sh": "02_system_setup.sh",
            "run_pipeline.sh": "03_run_fep_pipeline.sh",
        }
        ions_mdp_source = submission_directory / "mdp" / "ions.mdp"
        ions_mdp_destination = task_directory / "mdp" / "ions.mdp"
        fep_mdp_source = submission_directory / "mdp" / "fep_base.mdp"
        fep_mdp_destination = task_directory / "mdp" / "fep_base.mdp"

        try:
            task_directory.mkdir(parents=True, exist_ok=True)
            ions_mdp_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(protein_source, protein_destination)
            shutil.copy2(ligand_source, ligand_destination)
            shutil.copy2(ions_mdp_source, ions_mdp_destination)
            shutil.copy2(fep_mdp_source, fep_mdp_destination)
            for submission_file in submission_files:
                shutil.copy2(
                    submission_directory / submission_file,
                    task_directory
                    / submission_names.get(
                        submission_file,
                        submission_file,
                    ),
                )
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Task Creation Failed",
                f"Could not copy task files:\n\n{exc}",
            )
            return

        config["system"]["protein_structure"] = str(protein_destination)
        config["system"]["ligand_structure"] = str(ligand_destination)

        self.save_config(
            file_path=task_directory / f"{job_name}_config.inp",
            config=config,
        )

        QMessageBox.information(
            self,
            "Task Created",
            f"Task files saved to:\n\n{task_directory}\n\n"
            "Preparation scripts and MDP files were staged from the installed "
            "AlchemForge package.",
        )

    def submit_job(self):
        config = self.collect_config()

        if not config["system"]["protein_structure"]:
            QMessageBox.warning(
                self,
                "Missing Protein",
                "Please select a protein structure.",
            )
            return

        job_name = config["job_setting"]["job_name"]
        if not job_name:
            QMessageBox.warning(
                self,
                "Missing Job Name",
                "Please enter a job name.",
            )
            return

        task_directory = (
            Path(self.working_directory or ".").expanduser().resolve() / job_name
        )
        script_path = task_directory / "03_run_fep_pipeline.sh"
        config_path = task_directory / f"{job_name}_config.inp"
        worker_path = task_directory / "ResAlchemFEP.sh"

        if not script_path.is_file() or not config_path.is_file():
            QMessageBox.warning(
                self,
                "Task Not Created",
                "Create the task before submitting the job.",
            )
            return

        if not worker_path.is_file():
            QMessageBox.warning(
                self,
                "Missing FEP Worker",
                "Submission requires ResAlchemFEP.sh in the task directory. "
                "The downloaded workflow contains only the three orchestration "
                "scripts, so add the simulation worker before submitting.",
            )
            return

        self.job_requested.emit(
            str(task_directory),
            str(script_path),
            str(config_path),
        )
        self.accept()
