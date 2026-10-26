import ast
from copy import deepcopy
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
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from alchemforge.desktop.modeless import show_modeless
from alchemforge.openfe.config import OPENFE_LIGAND_FEP_CONFIG, save_yaml


class OpenFESetupSectionDialog(QDialog):
    def __init__(self, title, section, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 520)

        layout = QVBoxLayout(self)
        layout.addWidget(section)
        section.show()

        close_button = QPushButton("Done")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


class LigAlchemFEPDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory=None, parent=None):
        super().__init__(parent)

        self.working_directory = working_directory
        self.openfe_config = deepcopy(OPENFE_LIGAND_FEP_CONFIG)

        self.setWindowTitle("LigAlchemFEP Workflow Using OpenFE")
        self.resize(600, 700)

        self.build_ui()

    def build_ui(self):
        root = QVBoxLayout(self)

        title = QLabel("Ligand Alchemical FEP Workflow Using OpenFE")
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")

        subtitle = QLabel(
            "Ligand relative binding free-energy calculations using OpenFE."
        )
        subtitle.setStyleSheet("color: #71839a;")

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addSpacing(5)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_openfe_system_tab(), "System Setup")
        self.tabs.addTab(self.build_openfe_simulation_tab(), "FEP Simulation Protocol")
        self.tabs.addTab(self.build_openfe_job_setting_tab(), "Job Setting")
        root.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()

        run_btn = QPushButton("Create Task")
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

        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self.save_openfe_config)

        buttons.addStretch()
        buttons.addWidget(run_btn)
        buttons.addWidget(self.save_config_btn)
        root.addLayout(buttons)

    def build_openfe_system_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        input_group = QGroupBox("Input")
        input_layout = QFormLayout(input_group)
        input_layout.setLabelAlignment(Qt.AlignRight)
        input_layout.setHorizontalSpacing(14)
        input_layout.setVerticalSpacing(10)

        forcefield_group = QGroupBox("Force Field")
        forcefield_layout = QFormLayout(forcefield_group)
        forcefield_layout.setLabelAlignment(Qt.AlignRight)
        forcefield_layout.setHorizontalSpacing(14)
        forcefield_layout.setVerticalSpacing(10)

        solvation_group = QGroupBox("Solvation")
        solvation_layout = QFormLayout(solvation_group)
        solvation_layout.setLabelAlignment(Qt.AlignRight)
        solvation_layout.setHorizontalSpacing(14)
        solvation_layout.setVerticalSpacing(10)

        perturbation_group = QGroupBox("Perturbation Network")
        perturbation_layout = QFormLayout(perturbation_group)
        perturbation_layout.setLabelAlignment(Qt.AlignRight)
        perturbation_layout.setHorizontalSpacing(14)
        perturbation_layout.setVerticalSpacing(10)

        system_setup = self.openfe_config["system_setup"]

        self.openfe_compute_charge_checkbox = QCheckBox("Compute Partial Charges")
        self.openfe_compute_charge_checkbox.setChecked(False)

        input_note = QLabel(
            """
            <p align="justify">
            If selected, the workflow will compute the
            ligand partial charges. Otherwise, existing partial charges will be used.
            If no partial charges are found, they will be computed automatically.
            </p>
            """
        )
        input_note.setWordWrap(True)
        input_note.setStyleSheet("color: #707070; font-size: 11px; ")
        input_note.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.openfe_ligand_edit = QLineEdit()
        self.openfe_ligand_edit.setPlaceholderText("Ligand SDF")

        self.openfe_protein_edit = QLineEdit()
        self.openfe_protein_edit.setPlaceholderText("Protein PDB")

        self.openfe_cofactor_edit = QLineEdit()
        self.openfe_cofactor_edit.setPlaceholderText("Cofactor SDF")

        self.openfe_json_edit = QLineEdit(system_setup["input_json_files"])

        self.openfe_constraints_combo = QComboBox()
        self.openfe_constraints_combo.addItems(["hbonds", "allbonds", "none"])
        self.openfe_constraints_combo.setCurrentText(system_setup["ff_constraints"])

        self.openfe_cutoff_spin = QDoubleSpinBox()
        self.openfe_cutoff_spin.setRange(0.0, 10.0)
        self.openfe_cutoff_spin.setDecimals(3)
        self.openfe_cutoff_spin.setValue(system_setup["nonbonded_cutoff"])

        self.openfe_method_combo = QComboBox()
        self.openfe_method_combo.addItems(
            ["PME", "NoCutoff", "CutoffPeriodic", "CutoffNonPeriodic"]
        )
        self.openfe_method_combo.setCurrentText(system_setup["nonbonded_method"])
        self.openfe_rigid_water_checkbox = QCheckBox()
        self.openfe_rigid_water_checkbox.setChecked(system_setup["rigid_water"])
        self.openfe_forcefield_edit = QPlainTextEdit()
        self.openfe_forcefield_edit.setPlainText(
            "\n".join(system_setup["protein_forcefield"])
        )
        self.openfe_forcefield_edit.setMaximumHeight(90)
        self.openfe_backend_edit = QLineEdit(system_setup["off_toolkit_backend"])
        self.openfe_charge_method_edit = QLineEdit(
            system_setup["partial_charge_method"]
        )
        self.openfe_small_molecule_ff_edit = QLineEdit(
            system_setup["small_molecule_forcefield"]
        )
        self.openfe_padding_spin = QDoubleSpinBox()
        self.openfe_padding_spin.setRange(0.0, 100.0)
        self.openfe_padding_spin.setDecimals(3)
        self.openfe_padding_spin.setValue(system_setup["solvation_padding"])
        self.openfe_solvent_model_edit = QLineEdit(system_setup["solvent_model"])
        self.openfe_positive_ion_edit = QLineEdit(system_setup["positive_ion"])
        self.openfe_negative_ion_edit = QLineEdit(system_setup["negative_ion"])
        self.openfe_neutralize_checkbox = QCheckBox()
        self.openfe_neutralize_checkbox.setChecked(system_setup["neutralize"])
        self.openfe_ion_concentration_spin = QDoubleSpinBox()
        self.openfe_ion_concentration_spin.setRange(0.0, 10.0)
        self.openfe_ion_concentration_spin.setDecimals(3)
        self.openfe_ion_concentration_spin.setValue(system_setup["ion_concentration"])
        self.openfe_membrane_checkbox = QCheckBox()
        self.openfe_membrane_checkbox.setChecked(system_setup["membrane_system"])
        self.openfe_membrane_checkbox.toggled.connect(self.update_openfe_box_visibility)
        self.openfe_hydrogen_mass_spin = QDoubleSpinBox()
        self.openfe_hydrogen_mass_spin.setRange(0.0, 100.0)
        self.openfe_hydrogen_mass_spin.setDecimals(3)
        self.openfe_hydrogen_mass_spin.setValue(system_setup["hydrogen_mass"])
        alchemical_network = system_setup["alchemical_network"]
        alchemical_network_text = (
            str(alchemical_network)
            if isinstance(alchemical_network, (list, tuple))
            else str(alchemical_network)
        )
        self.openfe_network_edit = QLineEdit(alchemical_network_text)
        self.openfe_network_edit.setPlaceholderText("all or [ligand_a, ligand_b]")
        self.openfe_network_help = QPushButton("?")
        self.openfe_network_help.setFixedWidth(28)
        self.openfe_network_help.setToolTip(
            "Use 'all' to include every ligand, or enter an array of ligand names, "
            "for example [ligand_a, ligand_b]."
        )
        network_row = QHBoxLayout()
        network_row.setContentsMargins(0, 0, 0, 0)
        network_row.addWidget(self.openfe_network_edit)
        network_row.addWidget(self.openfe_network_help)
        self.openfe_transformation_combo = QComboBox()
        self.openfe_transformation_combo.addItems(["both", "complex", "solvent"])
        self.openfe_transformation_combo.setCurrentText(system_setup["transformation"])
        self.openfe_mapper_combo = QComboBox()
        self.openfe_mapper_combo.addItems(["LomapAtomMapper", "KartografAtomMapper"])
        self.openfe_mapper_combo.setCurrentText(system_setup["mapper"])
        self.openfe_planner_combo = QComboBox()
        self.openfe_planner_combo.addItems(
            [
                "minimal_spanning",
                "minimal_redundant",
                "radial",
            ]
        )
        self.openfe_planner_combo.setCurrentText(system_setup["planner"])
        self.openfe_complex_box_label = QLabel("Complex box size:")
        complex_box_size = system_setup["complex_box_size"]
        complex_box_text = (
            str(complex_box_size) if isinstance(complex_box_size, (list, tuple)) else ""
        )
        self.openfe_complex_box_edit = QLineEdit(complex_box_text)
        self.openfe_complex_box_edit.setPlaceholderText("[x, y, z]")

        ligand_row = self.build_openfe_file_row(
            self.openfe_ligand_edit,
            "Select ligand SDF",
            "SDF files (*.sdf);;All files (*)",
        )
        protein_row = self.build_openfe_file_row(
            self.openfe_protein_edit,
            "Select protein PDB",
            "PDB files (*.pdb);;All files (*)",
        )
        cofactor_row = self.build_openfe_file_row(
            self.openfe_cofactor_edit,
            "Select cofactor SDF",
            "SDF files (*.sdf);;All files (*)",
        )

        input_layout.addRow("Protein PDB:", protein_row)

        input_layout.addRow("Ligand SDF:", ligand_row)
        input_layout.addRow("Cofactor SDF:", cofactor_row)

        charge_container = QWidget()
        charge_layout = QVBoxLayout(charge_container)

        charge_layout.setContentsMargins(0, 0, 0, 0)
        charge_layout.setSpacing(2)

        charge_layout.addWidget(self.openfe_compute_charge_checkbox)
        charge_layout.addWidget(input_note)

        input_layout.addRow("Partial charges:", charge_container)

        # input_layout.addRow(
        #     "Partial charges:",
        #     self.openfe_compute_charge_checkbox
        # )
        # input_layout.addRow("", input_note)

        # input_layout.addRow("Input JSON files:", self.openfe_json_edit)

        forcefield_layout.addRow(
            "Force field constraints:", self.openfe_constraints_combo
        )
        forcefield_layout.addRow(
            "Nonbonded cutoff:", self.build_unit_row(self.openfe_cutoff_spin, "nm")
        )
        forcefield_layout.addRow("Nonbonded method:", self.openfe_method_combo)
        forcefield_layout.addRow("Rigid water:", self.openfe_rigid_water_checkbox)
        forcefield_layout.addRow("Protein force field:", self.openfe_forcefield_edit)
        forcefield_layout.addRow("OpenFF backend:", self.openfe_backend_edit)
        forcefield_layout.addRow(
            "Partial charge method:", self.openfe_charge_method_edit
        )
        forcefield_layout.addRow(
            "Small molecule force field:", self.openfe_small_molecule_ff_edit
        )

        solvation_layout.addRow(
            "Solvation padding:", self.build_unit_row(self.openfe_padding_spin, "nm")
        )
        solvation_layout.addRow("Solvent model:", self.openfe_solvent_model_edit)
        solvation_layout.addRow("Positive ion:", self.openfe_positive_ion_edit)
        solvation_layout.addRow("Negative ion:", self.openfe_negative_ion_edit)
        solvation_layout.addRow("Neutralize:", self.openfe_neutralize_checkbox)
        solvation_layout.addRow(
            "Ion concentration:",
            self.build_unit_row(self.openfe_ion_concentration_spin, "M"),
        )
        solvation_layout.addRow("Membrane system:", self.openfe_membrane_checkbox)
        solvation_layout.addRow(
            self.openfe_complex_box_label, self.openfe_complex_box_edit
        )
        solvation_layout.addRow(
            "Hydrogen mass:", self.build_unit_row(self.openfe_hydrogen_mass_spin, "Da")
        )

        perturbation_layout.addRow("Alchemical network:", network_row)
        perturbation_layout.addRow("Transformation:", self.openfe_transformation_combo)
        perturbation_layout.addRow("Ligand Mapping Method:", self.openfe_mapper_combo)
        perturbation_layout.addRow("Planner Method:", self.openfe_planner_combo)

        self.openfe_system_layout = layout
        self.openfe_system_panel = widget
        self.openfe_input_group = input_group
        self.openfe_forcefield_group = forcefield_group
        self.openfe_solvation_group = solvation_group
        self.openfe_perturbation_group = perturbation_group

        layout.addWidget(input_group)
        layout.addWidget(forcefield_group)
        layout.addWidget(solvation_group)
        layout.addWidget(perturbation_group)

        for section, title in (
            (forcefield_group, "OpenFE Force Field"),
            (solvation_group, "OpenFE Solvation"),
        ):
            section.hide()
            button = QPushButton(title)
            button.clicked.connect(
                lambda checked=False, section=section, title=title: (
                    self.open_openfe_section(section, title)
                )
            )
            layout.addWidget(button)
        layout.addStretch()
        self.update_openfe_box_visibility(self.openfe_membrane_checkbox.isChecked())

        return widget

    def open_openfe_section(self, section, title):
        self.open_openfe_section_dialog(
            section,
            title,
            self.openfe_system_layout,
            self.openfe_system_panel,
        )

    def open_openfe_section_dialog(self, section, title, section_layout, panel):
        section_index = section_layout.indexOf(section)
        section_layout.removeWidget(section)
        section.setParent(None)

        dialog = OpenFESetupSectionDialog(title, section, self)

        def restore_section(_result=None):
            dialog.layout().removeWidget(section)
            section.setParent(panel)
            section_layout.insertWidget(section_index, section)
            section.hide()

        dialog.finished.connect(restore_section)
        show_modeless(self, dialog)

    def build_openfe_file_row(self, line_edit, dialog_title, file_filter):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(
            lambda: self.browse_openfe_file(line_edit, dialog_title, file_filter)
        )
        row.addWidget(browse_button)

        container = QWidget()
        container.setLayout(row)
        return container

    def browse_openfe_file(self, line_edit, dialog_title, file_filter):
        path, _ = QFileDialog.getOpenFileName(
            self,
            dialog_title,
            self.working_directory or "",
            file_filter,
        )
        if path:
            line_edit.setText(path)

    def update_openfe_box_visibility(self, membrane_enabled):
        self.openfe_complex_box_label.setVisible(membrane_enabled)
        self.openfe_complex_box_edit.setVisible(membrane_enabled)

    def get_openfe_complex_box_size(self):
        value = self.openfe_complex_box_edit.text().strip()
        if not value:
            return None

        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
            raise ValueError("Complex box size must be an array with three values")

        return [float(item) for item in parsed]

    def get_openfe_alchemical_network(self):
        value = self.openfe_network_edit.text().strip()
        if value == "all":
            return value

        parsed = ast.literal_eval(value)
        if (
            not isinstance(parsed, (list, tuple))
            or not parsed
            or not all(isinstance(item, str) and item.strip() for item in parsed)
        ):
            raise ValueError(
                "Alchemical network must be 'all' or an array of ligand names"
            )

        return [item.strip() for item in parsed]

    def build_unit_row(self, widget, unit):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        row.addWidget(widget)
        row.addWidget(QLabel(unit))
        row.addStretch()

        container = QWidget()
        container.setLayout(row)

        # Prevent this row from expanding vertically
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        return container

    def build_compact_parameter_row(self, *parameters):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for index, (label, widget, unit) in enumerate(parameters):
            if index:
                row.addSpacing(14)
            row.addWidget(QLabel(f"{label}:"))
            row.addWidget(widget)
            if unit:
                row.addWidget(QLabel(unit))
        row.addStretch()
        container = QWidget()
        container.setLayout(row)
        return container

    def build_openfe_simulation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        core_group = QGroupBox("Core Simulation")
        core_layout = QFormLayout(core_group)
        core_layout.setLabelAlignment(Qt.AlignLeft)
        # core_layout.setHorizontalSpacing(14)
        # core_layout.setVerticalSpacing(10)

        details_group = QGroupBox("Simulation Details")
        details_layout = QFormLayout(details_group)
        details_layout.setLabelAlignment(Qt.AlignLeft)
        details_layout.setHorizontalSpacing(14)
        details_layout.setVerticalSpacing(10)

        network = self.openfe_config["alchemical_network_settings"]

        self.openfe_temperature_spin = QDoubleSpinBox()
        self.openfe_temperature_spin.setRange(0.0, 1000.0)
        self.openfe_temperature_spin.setDecimals(1)
        self.openfe_temperature_spin.setValue(network["temperature"])
        self.openfe_lambda_windows_spin = QSpinBox()
        self.openfe_lambda_windows_spin.setRange(1, 1000)
        self.openfe_lambda_windows_spin.setValue(network["n_lambda_windows"])
        self.openfe_production_length_spin = QDoubleSpinBox()
        self.openfe_production_length_spin.setRange(0.0, 100000.0)
        self.openfe_production_length_spin.setDecimals(2)
        self.openfe_production_length_spin.setValue(network["prod_length"])
        self.openfe_replicates_spin = QSpinBox()
        self.openfe_replicates_spin.setRange(1, 1000)
        self.openfe_replicates_spin.setValue(network["replicates"])
        self.openfe_start_replicates_spin = QSpinBox()
        self.openfe_start_replicates_spin.setRange(1, 1000)
        self.openfe_start_replicates_spin.setValue(network["start_replicates"])
        self.openfe_pressure_spin = QDoubleSpinBox()
        self.openfe_pressure_spin.setRange(0.0, 10000.0)
        self.openfe_pressure_spin.setDecimals(3)
        self.openfe_pressure_spin.setValue(network["pressure"])
        self.openfe_endstate_dispersion_checkbox = QCheckBox()
        self.openfe_endstate_dispersion_checkbox.setChecked(
            network["endstate_dispersion_correction"]
        )
        self.openfe_charge_correction_checkbox = QCheckBox()
        self.openfe_charge_correction_checkbox.setChecked(
            network["explicit_charge_correction"]
        )
        self.openfe_charge_cutoff_spin = QDoubleSpinBox()
        self.openfe_charge_cutoff_spin.setRange(0.0, 100.0)
        self.openfe_charge_cutoff_spin.setDecimals(3)
        self.openfe_charge_cutoff_spin.setValue(
            network["explicit_charge_correction_cutoff"]
        )
        self.openfe_softcore_combo = QComboBox()
        self.openfe_softcore_combo.addItems(["gapsys", "beutler"])
        self.openfe_softcore_combo.setCurrentText(network["softcore_LJ"])
        self.openfe_softcore_alpha_spin = QDoubleSpinBox()
        self.openfe_softcore_alpha_spin.setRange(0.0, 100.0)
        self.openfe_softcore_alpha_spin.setDecimals(3)
        self.openfe_softcore_alpha_spin.setValue(network["softcore_alpha"])
        self.openfe_unique_exceptions_checkbox = QCheckBox()
        self.openfe_unique_exceptions_checkbox.setChecked(
            network["turn_off_core_unique_exceptions"]
        )
        self.openfe_dispersion_checkbox = QCheckBox()
        self.openfe_dispersion_checkbox.setChecked(network["use_dispersion_correction"])
        self.openfe_early_termination_spin = QDoubleSpinBox()
        self.openfe_early_termination_spin.setRange(0.0, 100000.0)
        self.openfe_early_termination_spin.setDecimals(4)
        self.openfe_early_termination_spin.setValue(
            network["early_termination_target_error"]
        )
        self.openfe_mini_step_spin = QSpinBox()
        self.openfe_mini_step_spin.setRange(0, 100000000)
        self.openfe_mini_step_spin.setValue(network["mini_step"])
        self.openfe_equil_length_spin = QDoubleSpinBox()
        self.openfe_equil_length_spin.setRange(0.0, 100000.0)
        self.openfe_equil_length_spin.setDecimals(3)
        self.openfe_equil_length_spin.setValue(network["equil_length"])
        self.openfe_timestep_spin = QDoubleSpinBox()
        self.openfe_timestep_spin.setRange(0.0, 1000.0)
        self.openfe_timestep_spin.setDecimals(3)
        self.openfe_timestep_spin.setValue(network["timestep"])
        self.openfe_lambda_windows_spin.setFixedWidth(
            self.openfe_timestep_spin.sizeHint().width()
        )
        self.openfe_barostat_frequency_spin = QDoubleSpinBox()
        self.openfe_barostat_frequency_spin.setRange(0.0, 100000.0)
        self.openfe_barostat_frequency_spin.setDecimals(3)
        self.openfe_barostat_frequency_spin.setValue(network["barostat_frequency"])
        self.openfe_surface_tension_spin = QDoubleSpinBox()
        self.openfe_surface_tension_spin.setRange(0.0, 100000.0)
        self.openfe_surface_tension_spin.setDecimals(3)
        self.openfe_surface_tension_spin.setValue(network["surface_tension"])
        core_layout.addRow("Replicates:", self.openfe_replicates_spin)

        core_layout.addRow("Start replicate:", self.openfe_start_replicates_spin)
        core_layout.addRow(
            "Pressure:", self.build_unit_row(self.openfe_pressure_spin, "bar")
        )

        core_layout.addRow(
            "Temperature:", self.build_unit_row(self.openfe_temperature_spin, "K")
        )

        core_layout.addRow("Lambda windows:", self.openfe_lambda_windows_spin)

        core_layout.addRow(
            "Minimization steps:",
            self.build_unit_row(self.openfe_mini_step_spin, "steps"),
        )

        core_layout.addRow(
            "Equilibration length:",
            self.build_unit_row(self.openfe_equil_length_spin, "ns"),
        )

        core_layout.addRow(
            "Production length:",
            self.build_unit_row(self.openfe_production_length_spin, "ns"),
        )

        core_layout.addRow(
            "Timestep:", self.build_unit_row(self.openfe_timestep_spin, "fs")
        )
        details_layout.addRow(
            "Endstate dispersion correction:", self.openfe_endstate_dispersion_checkbox
        )
        details_layout.addRow(
            "Explicit charge correction:", self.openfe_charge_correction_checkbox
        )
        details_layout.addRow(
            "Charge correction cutoff:",
            self.build_unit_row(self.openfe_charge_cutoff_spin, "nm"),
        )
        details_layout.addRow("Softcore LJ:", self.openfe_softcore_combo)
        details_layout.addRow("Softcore alpha:", self.openfe_softcore_alpha_spin)
        details_layout.addRow(
            "Turn off core unique exceptions:", self.openfe_unique_exceptions_checkbox
        )
        details_layout.addRow(
            "Use dispersion correction:", self.openfe_dispersion_checkbox
        )
        details_layout.addRow(
            "Early termination target error:",
            self.build_unit_row(self.openfe_early_termination_spin, "kcal/mol"),
        )
        details_layout.addRow(
            "Barostat frequency:",
            self.build_unit_row(self.openfe_barostat_frequency_spin, "steps"),
        )
        details_layout.addRow(
            "Surface tension:",
            self.build_unit_row(self.openfe_surface_tension_spin, "bar nm"),
        )

        layout.addWidget(core_group)
        layout.addWidget(details_group)
        details_group.hide()
        details_button = QPushButton("More Details")

        details_button.clicked.connect(
            lambda: self.open_openfe_section_dialog(
                details_group,
                "OpenFE Simulation Details",
                layout,
                widget,
            )
        )
        layout.addWidget(details_button)
        layout.addStretch()
        return widget

    def build_openfe_job_setting_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setLabelAlignment(Qt.AlignRight)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(10)

        job_setting = self.openfe_config["job_setting"]
        self.openfe_job_name_edit = QLineEdit(job_setting["job_name"])
        self.openfe_ligand_para_host = QLineEdit(job_setting["ligand_para_host"])
        self.openfe_system_setup_host = QLineEdit(job_setting["system_setup_host"])

        self.openfe_simulation_host = QComboBox()
        self.openfe_simulation_host.addItems(["localhost", "remote"])
        self.openfe_simulation_host.setCurrentText(job_setting["simulation_host"])

        self.openfe_platform_combo = QComboBox()
        self.openfe_platform_combo.addItems(["CUDA", "CPU", "OpenCL"])
        self.openfe_platform_combo.setCurrentText(job_setting["platform"])

        layout.addRow("Job name:", self.openfe_job_name_edit)
        layout.addRow("Ligand parameterization host:", self.openfe_ligand_para_host)
        layout.addRow("System setup host:", self.openfe_system_setup_host)
        layout.addRow("Simulation host:", self.openfe_simulation_host)
        layout.addRow("Simulation Platform:", self.openfe_platform_combo)
        return widget

    def save_openfe_config(self, checked=False, file_path=None, config=None):
        config = config or self.collect_openfe_config()
        job_name = config["job_setting"]["job_name"]

        if file_path is None:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save OpenFE LigAlchemFEP Configuration",
                str(
                    Path(self.working_directory or ".").expanduser().resolve()
                    / "openfe_config.yaml"
                ),
                "YAML Configuration (*.yaml);;All Files (*)",
            )

        # User clicked Cancel
        if not file_path:
            return

        file_path = str(file_path)

        # Ensure .yaml extension
        if not file_path.endswith(".yaml"):
            file_path += ".yaml"

        save_yaml(config, file_path)

    def collect_openfe_config(self):
        config = deepcopy(self.openfe_config)
        config["job_setting"].update(
            {
                "job_name": self.openfe_job_name_edit.text().strip(),
                "ligand_para_host": self.openfe_ligand_para_host.text().strip(),
                "system_setup_host": self.openfe_system_setup_host.text().strip(),
                "simulation_host": self.openfe_simulation_host.currentText(),
                "platform": self.openfe_platform_combo.currentText(),
            }
        )
        config["system_setup"].update(
            {
                "ligand_sdf": self.openfe_ligand_edit.text().strip(),
                "compute_charge": self.openfe_compute_charge_checkbox.isChecked(),
                "protein_pdb": self.openfe_protein_edit.text().strip(),
                "cofactor_sdf": self.openfe_cofactor_edit.text().strip() or None,
                "input_json_files": self.openfe_json_edit.text().strip(),
                "ff_constraints": self.openfe_constraints_combo.currentText(),
                "nonbonded_cutoff": self.openfe_cutoff_spin.value(),
                "nonbonded_method": self.openfe_method_combo.currentText(),
                "rigid_water": self.openfe_rigid_water_checkbox.isChecked(),
                "protein_forcefield": [
                    line.strip()
                    for line in self.openfe_forcefield_edit.toPlainText().splitlines()
                    if line.strip()
                ],
                "off_toolkit_backend": self.openfe_backend_edit.text().strip(),
                "partial_charge_method": self.openfe_charge_method_edit.text().strip(),
                "small_molecule_forcefield": self.openfe_small_molecule_ff_edit.text().strip(),
                "solvation_padding": self.openfe_padding_spin.value(),
                "complex_box_size": (
                    self.get_openfe_complex_box_size()
                    if self.openfe_membrane_checkbox.isChecked()
                    else None
                ),
                "solvent_model": self.openfe_solvent_model_edit.text().strip(),
                "positive_ion": self.openfe_positive_ion_edit.text().strip(),
                "negative_ion": self.openfe_negative_ion_edit.text().strip(),
                "neutralize": self.openfe_neutralize_checkbox.isChecked(),
                "ion_concentration": self.openfe_ion_concentration_spin.value(),
                "membrane_system": self.openfe_membrane_checkbox.isChecked(),
                "hydrogen_mass": self.openfe_hydrogen_mass_spin.value(),
                "alchemical_network": self.get_openfe_alchemical_network(),
                "transformation": self.openfe_transformation_combo.currentText(),
                "mapper": self.openfe_mapper_combo.currentText(),
                "planner": self.openfe_planner_combo.currentText(),
            }
        )
        config["alchemical_network_settings"].update(
            {
                "replicates": self.openfe_replicates_spin.value(),
                "start_replicates": self.openfe_start_replicates_spin.value(),
                "pressure": self.openfe_pressure_spin.value(),
                "temperature": self.openfe_temperature_spin.value(),
                "endstate_dispersion_correction": self.openfe_endstate_dispersion_checkbox.isChecked(),
                "explicit_charge_correction": self.openfe_charge_correction_checkbox.isChecked(),
                "explicit_charge_correction_cutoff": self.openfe_charge_cutoff_spin.value(),
                "softcore_LJ": self.openfe_softcore_combo.currentText(),
                "softcore_alpha": self.openfe_softcore_alpha_spin.value(),
                "turn_off_core_unique_exceptions": self.openfe_unique_exceptions_checkbox.isChecked(),
                "use_dispersion_correction": self.openfe_dispersion_checkbox.isChecked(),
                "n_lambda_windows": self.openfe_lambda_windows_spin.value(),
                "early_termination_target_error": self.openfe_early_termination_spin.value(),
                "mini_step": self.openfe_mini_step_spin.value(),
                "equil_length": self.openfe_equil_length_spin.value(),
                "prod_length": self.openfe_production_length_spin.value(),
                "timestep": self.openfe_timestep_spin.value(),
                "barostat_frequency": self.openfe_barostat_frequency_spin.value(),
                "surface_tension": self.openfe_surface_tension_spin.value(),
            }
        )
        return config

    def create_task(self):
        try:
            config = self.collect_openfe_config()
        except (ValueError, SyntaxError) as exc:
            QMessageBox.warning(self, "Invalid Configuration", str(exc))
            return

        protein = config["system_setup"]["protein_pdb"]
        ligand = config["system_setup"]["ligand_sdf"]
        job_name = config["job_setting"]["job_name"]

        if not protein:
            QMessageBox.warning(
                self, "Missing Protein", "Please select an input protein PDB."
            )
            return

        if not ligand:
            QMessageBox.warning(
                self, "Missing Ligand", "Please select an input ligand SDF."
            )
            return

        if not job_name:
            QMessageBox.warning(self, "Missing Job Name", "Please enter a job name.")
            return

        task_directory = (
            Path(self.working_directory or ".").expanduser().resolve() / job_name
        )

        try:
            task_directory.mkdir(parents=True, exist_ok=True)
            config_path = task_directory / f"{job_name}_openfe_config.yaml"
            save_yaml(config, str(config_path))
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Task Creation Failed",
                f"Could not create OpenFE task:\n\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Task Created",
            f"OpenFE task configuration saved to:\n\n{config_path}",
        )


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = LigAlchemFEPDialog()
    dialog.show()
    sys.exit(app.exec_())
