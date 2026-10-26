"""Desktop interface for preparing complete GROMACS systems."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from alchemforge.gromacs.system_prepare import (
    GromacsSystemPreparationConfig,
    installed_amber_lipid_force_fields,
    installed_gromacs_force_fields,
    stage_gromacs_system,
)


class GromacsSystemPrepareDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("GROMACS System Builder")
        self.resize(850, 700)
        root = QVBoxLayout(self)
        title = QLabel("GROMACS System Builder")
        title.setStyleSheet("font-size:18pt;font-weight:700;color:#123f76")
        root.addWidget(title)
        note = QLabel(
            "Package and validate an existing parameterized membrane/solvent system, "
            "or run pdb2gmx on an assembled PDB. Existing solvent and membrane "
            "coordinates are preserved; this module does not resolvate the system."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#71839a")
        root.addWidget(note)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.mode = QComboBox()
        self.mode.addItem("Existing GRO + topology (recommended)", "existing")
        self.mode.addItem("Amber membrane PDB + ligand SDF (ResFEP-style)", "amber_pdb_sdf")
        self.mode.addItem("Build topology from assembled PDB", "pdb2gmx")
        self.system_type = QComboBox()
        self.system_type.addItems(["membrane", "soluble"])
        self.override_box = QCheckBox("Override periodic box from the input structure")
        self.override_box.setToolTip(
            "Set the final GRO box explicitly without moving or scaling atoms. "
            "Lengths are in nm; angles are in degrees."
        )
        self.box_lengths = QLineEdit()
        self.box_lengths.setPlaceholderText("X Y Z in nm, e.g. 12.5 12.5 15.0")
        self.box_angles = QLineEdit("90 90 90")
        self.box_angles.setPlaceholderText("alpha beta gamma in degrees")
        self.structure = QLineEdit()
        self.topology = QLineEdit()
        self.ligands = QLineEdit()
        self.reference_ligand = QComboBox()
        self.reference_ligand.setEditable(False)
        self.ligand_resname = QLineEdit("LIG")
        self.ligand_charge = QSpinBox()
        self.ligand_charge.setRange(-20, 20)
        self.cofactor_topology = QLineEdit()
        self.cofactor_topology.setPlaceholderText(
            "Optional GROMACS ITP for one cofactor type"
        )
        self.cofactor_count = QSpinBox()
        self.cofactor_count.setRange(1, 10000)
        self.cofactor_count.setValue(1)
        self.amber_cofactor = QLineEdit()
        self.amber_cofactor_resname = QLineEdit("COF")
        self.amber_cofactor_charge = QSpinBox()
        self.amber_cofactor_charge.setRange(-20, 20)
        self.output = QLineEdit(str(Path(self.working_directory) / "gromacs_system"))
        self.gmx = QLineEdit("gmx")
        self.force_field = QComboBox()
        self.force_field.setEditable(True)
        self.force_field.addItems(installed_gromacs_force_fields())
        self.force_field.setCurrentText("charmm27")
        self.protein_ff = self._combo(["ff14SB", "ff19SB"])
        self.lipid_ff = self._combo(installed_amber_lipid_force_fields())
        self.dna_ff = self._combo(["OL21", "bsc1", "None"])
        self.rna_ff = self._combo(["OL3", "None"])
        self.water = self._combo(["TIP3P"])
        self.ion_ff = self._combo(["Joung-Cheatham TIP3P"])
        self.ligand_ff = self._combo(["GAFF2"])
        self.cofactor_ff = self._combo(["None", "GAFF2"])
        self.structure_label = QLabel("Prepared GRO:")
        self.topology_label = QLabel("Matching topology:")
        form.addRow("Mode:", self.mode)
        form.addRow("System type:", self.system_type)
        form.addRow("", self.override_box)
        form.addRow("PBC box lengths:", self.box_lengths)
        form.addRow("PBC box angles:", self.box_angles)
        form.addRow(self.structure_label, self._file_row(self.structure))
        form.addRow(self.topology_label, self._file_row(self.topology, topology=True))
        self.ligands_label = QLabel("Ligand library SDF:")
        self.reference_label = QLabel("Reference ligand:")
        self.resname_label = QLabel("PDB ligand residue:")
        self.charge_label = QLabel("Reference net charge:")
        form.addRow(self.ligands_label, self._ligand_row())
        form.addRow(self.reference_label, self.reference_ligand)
        form.addRow(self.resname_label, self.ligand_resname)
        form.addRow(self.charge_label, self.ligand_charge)
        self.cofactor_topology_label = QLabel("Cofactor topology ITP:")
        self.cofactor_count_label = QLabel("Cofactor molecule count:")
        form.addRow(
            self.cofactor_topology_label,
            self._file_row(
                self.cofactor_topology,
                file_filter="GROMACS topology include (*.itp)",
            ),
        )
        form.addRow(self.cofactor_count_label, self.cofactor_count)
        self.amber_cofactor_label = QLabel("Cofactor SDF (optional):")
        self.amber_cofactor_resname_label = QLabel("Cofactor PDB residue:")
        self.amber_cofactor_charge_label = QLabel("Cofactor net charge:")
        self.amber_cofactor_row = self._file_row(
            self.amber_cofactor, file_filter="SDF (*.sdf)"
        )
        form.addRow(self.amber_cofactor_label, self.amber_cofactor_row)
        form.addRow(self.amber_cofactor_resname_label, self.amber_cofactor_resname)
        form.addRow(self.amber_cofactor_charge_label, self.amber_cofactor_charge)
        form.addRow("Output system directory:", self._directory_row(self.output))
        form.addRow("GROMACS command:", self.gmx)
        self.force_field_label = QLabel("GROMACS .ff bundle:")
        form.addRow(self.force_field_label, self.force_field)
        root.addLayout(form)
        self.component_group = QGroupBox("Amber force-field components")
        component_layout = QGridLayout(self.component_group)
        component_fields = (
            ("Protein", self.protein_ff),
            ("Lipids", self.lipid_ff),
            ("DNA", self.dna_ff),
            ("RNA", self.rna_ff),
            ("Water", self.water),
            ("Ions", self.ion_ff),
            ("Ligands", self.ligand_ff),
            ("Cofactor (optional)", self.cofactor_ff),
        )
        for index, (label, field) in enumerate(component_fields):
            row, column = divmod(index, 2)
            component_layout.addWidget(QLabel(f"{label}:"), row, column * 2)
            component_layout.addWidget(field, row, column * 2 + 1)
        root.addWidget(self.component_group)
        warning = QLabel(
            "PDB build mode succeeds only when the chosen GROMACS force field "
            "recognizes every protein, lipid, solvent, ion, cofactor, and ligand "
            "residue. For CHARMM-GUI or another parameterized membrane system, use "
            "Existing GRO + topology."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("background:#fff5d9;color:#765b16;padding:9px")
        root.addWidget(warning)
        root.addStretch()
        buttons = QHBoxLayout()
        buttons.addStretch()
        run = QPushButton("Prepare and Validate System")
        run.clicked.connect(self.create_job)
        buttons.addWidget(run)
        root.addLayout(buttons)
        self.mode.currentIndexChanged.connect(self._update_mode)
        self.system_type.currentTextChanged.connect(self._update_box_override)
        self.override_box.toggled.connect(self._update_box_override)
        self.cofactor_ff.currentTextChanged.connect(self._update_cofactor)
        self._update_mode()

    @staticmethod
    def _combo(items: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItems(items)
        return combo

    @staticmethod
    def _row(field: QLineEdit, button: QPushButton) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _file_row(
        self,
        field: QLineEdit,
        *,
        topology: bool = False,
        file_filter: str | None = None,
    ) -> QWidget:
        button = QPushButton("Browse…")

        def browse() -> None:
            selected_filter = file_filter or (
                "Topology (*.top)" if topology else "Structure (*.gro *.pdb)"
            )
            path, _ = QFileDialog.getOpenFileName(
                self, "Select GROMACS input", self.working_directory, selected_filter
            )
            if path:
                field.setText(path)

        button.clicked.connect(browse)
        return self._row(field, button)

    def _directory_row(self, field: QLineEdit) -> QWidget:
        button = QPushButton("Browse…")

        def browse() -> None:
            path = QFileDialog.getExistingDirectory(
                self, "Select output directory", self.working_directory
            )
            if path:
                field.setText(path)

        button.clicked.connect(browse)
        return self._row(field, button)

    def _ligand_row(self) -> QWidget:
        button = QPushButton("Browse…")

        def browse() -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select multi-molecule ligand library",
                self.working_directory,
                "SDF (*.sdf)",
            )
            if not path:
                return
            self.ligands.setText(path)
            self.reference_ligand.clear()
            try:
                from rdkit import Chem

                names = [
                    molecule.GetProp("_Name").strip()
                    for molecule in Chem.SDMolSupplier(path, removeHs=False)
                    if molecule is not None
                    and molecule.HasProp("_Name")
                    and molecule.GetProp("_Name").strip()
                ]
            except (OSError, RuntimeError) as error:
                QMessageBox.warning(self, "Cannot Read Ligands", str(error))
                return
            self.reference_ligand.addItems(names)

        button.clicked.connect(browse)
        return self._row(self.ligands, button)

    def _update_mode(self) -> None:
        existing = self.mode.currentData() == "existing"
        amber = self.mode.currentData() == "amber_pdb_sdf"
        self.structure_label.setText("Prepared GRO:" if existing else "Complex PDB:")
        self.topology.setEnabled(existing)
        self.topology_label.setEnabled(existing)
        for widget in (
            self.ligands,
            self.reference_ligand,
            self.ligand_resname,
            self.ligand_charge,
            self.ligands_label,
            self.reference_label,
            self.resname_label,
            self.charge_label,
        ):
            widget.setEnabled(amber)
        for widget in (
            self.cofactor_topology,
            self.cofactor_count,
            self.cofactor_topology_label,
            self.cofactor_count_label,
        ):
            widget.setEnabled(existing)
        self.force_field.setEnabled(not existing and not amber)
        self.force_field_label.setEnabled(not existing and not amber)
        self.component_group.setVisible(amber)
        self._update_cofactor()
        if amber:
            self.force_field.setCurrentText("Amber component preset")
        elif self.force_field.currentText() == "Amber component preset":
            self.force_field.setCurrentText("charmm27")
        self._update_box_override()

    def _update_box_override(self, *_args) -> None:
        enabled = self.override_box.isChecked()
        self.box_lengths.setEnabled(enabled)
        self.box_angles.setEnabled(enabled)

    @staticmethod
    def _triplet(text: str, label: str) -> tuple[float, float, float]:
        fields = text.replace(",", " ").split()
        if len(fields) != 3:
            raise ValueError(f"{label} must contain exactly three numbers")
        try:
            return tuple(float(value) for value in fields)  # type: ignore[return-value]
        except ValueError as error:
            raise ValueError(f"{label} contains a non-numeric value") from error

    def _update_cofactor(self) -> None:
        enabled = (
            self.mode.currentData() == "amber_pdb_sdf"
            and self.cofactor_ff.currentText() == "GAFF2"
        )
        for widget in (
            self.amber_cofactor,
            self.amber_cofactor_row,
            self.amber_cofactor_label,
            self.amber_cofactor_resname,
            self.amber_cofactor_resname_label,
            self.amber_cofactor_charge,
            self.amber_cofactor_charge_label,
        ):
            widget.setEnabled(enabled)

    def create_job(self) -> None:
        try:
            output_text = self.output.text().strip()
            if not output_text:
                raise ValueError("Select an output system directory")
            topology_text = self.topology.text().strip()
            box_lengths = (
                self._triplet(self.box_lengths.text(), "PBC box lengths")
                if self.override_box.isChecked()
                else None
            )
            box_angles = (
                self._triplet(self.box_angles.text(), "PBC box angles")
                if self.override_box.isChecked()
                else (90.0, 90.0, 90.0)
            )
            config = GromacsSystemPreparationConfig(
                mode=self.mode.currentData(),
                structure=Path(self.structure.text().strip()).expanduser().resolve(),
                topology=(
                    Path(topology_text).expanduser().resolve()
                    if topology_text
                    else None
                ),
                output_directory=Path(output_text).expanduser().resolve(),
                system_type=self.system_type.currentText(),
                gmx_command=self.gmx.text().strip(),
                force_field=self.force_field.currentText().strip(),
                water_model=self.water.currentText().strip().lower(),
                protein_force_field=self.protein_ff.currentText(),
                lipid_force_field=self.lipid_ff.currentText(),
                dna_force_field=self.dna_ff.currentText(),
                rna_force_field=self.rna_ff.currentText(),
                ion_parameters=self.ion_ff.currentText(),
                ligand_force_field=self.ligand_ff.currentText(),
                cofactor_force_field=self.cofactor_ff.currentText(),
                cofactor_structure=(
                    Path(self.amber_cofactor.text().strip()).expanduser().resolve()
                    if self.amber_cofactor.text().strip()
                    else None
                ),
                cofactor_resname=self.amber_cofactor_resname.text().strip().upper(),
                cofactor_charge=self.amber_cofactor_charge.value(),
                ligand_library=(
                    Path(self.ligands.text().strip()).expanduser().resolve()
                    if self.ligands.text().strip()
                    else None
                ),
                reference_ligand=self.reference_ligand.currentText() or None,
                ligand_resname=self.ligand_resname.text().strip().upper(),
                ligand_charge=self.ligand_charge.value(),
                cofactor_topology=(
                    Path(self.cofactor_topology.text().strip()).expanduser().resolve()
                    if self.cofactor_topology.text().strip()
                    else None
                ),
                cofactor_count=self.cofactor_count.value(),
                box_lengths_nm=box_lengths,
                box_angles_deg=box_angles,
            )
            runner, manifest = stage_gromacs_system(config)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Prepare GROMACS System", str(error))
            return
        self.job_requested.emit(str(config.output_directory), str(runner), str(manifest))
        self.accept()
