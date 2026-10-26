"""Desktop dialog for hydrogen removal followed by PDBFixer repair."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from alchemforge.structure.pdbfix import stage


class PDBFixerDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("Fix PDB Structures")
        self.resize(800, 500)
        root = QVBoxLayout(self)
        title = QLabel("Remove Hydrogens and Repair PDB Files")
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")
        description = QLabel(
            "First removes hydrogen ATOM/HETATM records, then runs PDBFixer "
            "with --add-atoms=all. Select the protein PDB and optional cofactor, "
            "membrane, and solvent PDB files. The final complex.pdb contains all "
            "selected components."
        )
        description.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(description)
        warning = QLabel(
            "Cofactor and membrane PDBs are preserved without PDBFixer because "
            "PDBFixer can corrupt or replace nonstandard residues."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #9a5b00;")
        root.addWidget(warning)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.protein_pdb = QLineEdit()
        self.cofactor_pdb = QLineEdit()
        self.membrane_pdb = QLineEdit()
        self.solvent_pdb = QLineEdit()
        self.pbc_source_pdb = QLineEdit()
        self.output_directory = QLineEdit()
        self.pdbfixer_command = QLineEdit("pdbfixer")
        self.overwrite = QCheckBox("Overwrite existing fixed output files")

        self.protein_pdb.setPlaceholderText("Required: protein PDB")
        self.cofactor_pdb.setPlaceholderText(
            "Optional: one PDB containing all retained cofactors"
        )
        self.membrane_pdb.setPlaceholderText("Optional: membrane/lipid-only PDB")
        self.solvent_pdb.setPlaceholderText("Optional: water/ion-only PDB")
        self.pbc_source_pdb.setPlaceholderText(
            "Optional: CMS-converted PDB containing CRYST1"
        )
        self.output_directory.setPlaceholderText("Directory for fixed PDB files")

        protein_help = (
            "Select the protein PDB to repair. This file is required and will "
            "produce fixed_protein.pdb."
        )
        cofactor_help = (
            "Select a PDB containing the cofactor or cofactors that must remain "
            "bound in the simulated complex. It is copied without PDBFixer to "
            "preserve atom names and coordinates. Matching force-field "
            "parameters are still required during system preparation."
        )
        membrane_help = (
            "For a membrane system, select the separate membrane/lipid-only "
            "PDB. You must also select a solvent PDB. Leave both empty for a "
            "soluble protein. The membrane is copied unchanged to "
            "fixed_membrane.pdb rather than processed by PDBFixer."
        )
        solvent_help = (
            "For a membrane system, select the separate solvent PDB containing "
            "water and/or ions. You must also select a membrane PDB. It will "
            "produce fixed_solvent.pdb."
        )
        pbc_help = (
            "Select the full-system PDB converted from the Desmond CMS file. "
            "Only its CRYST1 periodic box is read; its atoms are not added again. "
            "If left empty, the assembler searches the selected component PDBs "
            "for CRYST1 automatically."
        )
        output_help = (
            "Choose the directory where the fixed PDB files will be written. "
            "The original input files are not modified."
        )
        command_help = (
            "Command used to launch PDBFixer. Keep 'pdbfixer' unless the "
            "executable has a different name or you need to enter its full path."
        )
        form.addRow(
            "Protein PDB:",
            self._pdb_row(self.protein_pdb, "Select protein PDB", protein_help, True),
        )
        form.addRow(
            "Cofactor PDB (optional):",
            self._pdb_row(self.cofactor_pdb, "Select cofactor PDB", cofactor_help),
        )
        form.addRow(
            "Membrane PDB (optional):",
            self._pdb_row(self.membrane_pdb, "Select membrane PDB", membrane_help),
        )
        form.addRow(
            "Solvent PDB (optional):",
            self._pdb_row(self.solvent_pdb, "Select solvent PDB", solvent_help),
        )
        form.addRow(
            "PBC source PDB (optional):",
            self._pdb_row(
                self.pbc_source_pdb,
                "Select CMS-converted PDB with CRYST1",
                pbc_help,
            ),
        )
        form.addRow("Output directory:", self._output_row(output_help))
        form.addRow(
            "PDBFixer command:",
            self._field_with_help(self.pdbfixer_command, command_help),
        )
        self.overwrite.setToolTip(
            "Enable this only when existing fixed output files may be replaced."
        )
        form.addRow("", self.overwrite)
        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        run = QPushButton("Fix Structures")
        run.clicked.connect(self.create_job)
        buttons.addWidget(run)
        root.addLayout(buttons)

    def _help_button(self, help_text: str) -> QToolButton:
        button = QToolButton()
        button.setText("?")
        button.setToolTip(help_text)
        button.setWhatsThis(help_text)
        button.setFixedSize(24, 24)
        button.clicked.connect(
            lambda _checked=False: QMessageBox.information(
                self, "Field Help", help_text
            )
        )
        return button

    def _row(self, field: QLineEdit, button: QPushButton, help_text: str) -> QWidget:
        field.setToolTip(help_text)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)
        layout.addWidget(self._help_button(help_text))
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _field_with_help(self, field: QLineEdit, help_text: str) -> QWidget:
        field.setToolTip(help_text)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(self._help_button(help_text))
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _pdb_row(
        self,
        field: QLineEdit,
        title: str,
        help_text: str,
        set_default_output: bool = False,
    ) -> QWidget:
        button = QPushButton("Browse...")
        button.clicked.connect(
            lambda: self._select_pdb(field, title, set_default_output)
        )
        return self._row(field, button, help_text)

    def _output_row(self, help_text: str) -> QWidget:
        button = QPushButton("Browse...")
        button.clicked.connect(self._select_output)
        return self._row(self.output_directory, button, help_text)

    def _select_pdb(
        self, field: QLineEdit, title: str, set_default_output: bool = False
    ) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, title, self.working_directory, "PDB files (*.pdb);;All files (*)"
        )
        if path:
            field.setText(path)
            if set_default_output and not self.output_directory.text().strip():
                self.output_directory.setText(str(Path(path).parent / "fixed"))

    @staticmethod
    def _optional_path(field: QLineEdit) -> Path | None:
        value = field.text().strip()
        return Path(value) if value else None

    def _select_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select output directory", self.working_directory
        )
        if path:
            self.output_directory.setText(path)

    def create_job(self) -> None:
        try:
            protein_pdb = self._optional_path(self.protein_pdb)
            if protein_pdb is None:
                raise ValueError("Select a protein PDB file")
            cofactor_pdb = self._optional_path(self.cofactor_pdb)
            membrane_pdb = self._optional_path(self.membrane_pdb)
            solvent_pdb = self._optional_path(self.solvent_pdb)
            pbc_source_pdb = self._optional_path(self.pbc_source_pdb)
            if (membrane_pdb is None) != (solvent_pdb is None):
                raise ValueError(
                    "Select both membrane and solvent PDBs for a membrane system, "
                    "or leave both empty for a protein-only system"
                )
            output_text = self.output_directory.text().strip()
            if not output_text:
                raise ValueError("Select an output directory")
            output = Path(output_text).expanduser().resolve()
            task = Path(self.working_directory) / "pdbfixer" / protein_pdb.stem
            runner, manifest, tasks = stage(
                None,
                output,
                task,
                include_components=False,
                protein_pdb=protein_pdb,
                cofactor_pdb=cofactor_pdb,
                membrane_pdb=membrane_pdb,
                solvent_pdb=solvent_pdb,
                box_source_pdb=pbc_source_pdb,
                assemble_system=True,
                overwrite=self.overwrite.isChecked(),
                pdbfixer_command=self.pdbfixer_command.text().strip(),
            )
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Fix PDB Files", str(error))
            return
        labels = ", ".join(item.label for item in tasks)
        QMessageBox.information(
            self,
            "PDBFixer Job Created",
            f"Files scheduled: {labels}\nFinal system: {output / 'complex.pdb'}"
            f"\nVisualization: {output / 'complex.cif'}",
        )
        self.job_requested.emit(str(task), str(runner), str(manifest))
        self.accept()
