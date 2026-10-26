"""Desktop dialog for standalone GROMACS ligand parameterization."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QByteArray, Qt, pyqtSignal
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from alchemforge.gromacs.ligand_prepare import LigandPreparationConfig, stage


class GromacsLigandPrepareDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(
        self,
        working_directory: str | None = None,
        parent=None,
        *,
        embedded: bool = False,
    ):
        super().__init__(parent)
        self.embedded = embedded
        if embedded:
            self.setWindowFlags(Qt.Widget)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self._molecules = []
        self._loaded_input: Path | None = None
        self.setWindowTitle("Prepare Ligands for GROMACS")
        self.resize(980, 720)
        root = QVBoxLayout(self)
        title = QLabel("GROMACS Ligand Preparation")
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")
        description = QLabel(
            "Prepare one MOL2 ligand or every molecule in an SDF with Open Babel "
            "and ACPYPE. Results are reusable by ABFE, ligand RBFE, SepTop, and "
            "residue FEP workflows."
        )
        description.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(description)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.input_file = QLineEdit()
        self.output_directory = QLineEdit()
        self.charge_method = QComboBox()
        self.charge_method.addItems(["bcc", "user"])
        self.atom_type = QComboBox()
        self.atom_type.addItems(["gaff2", "gaff"])
        self.obabel_command = QLineEdit("obabel")
        self.acpype_command = QLineEdit("acpype")
        form.addRow("Ligand SDF or MOL2:", self._input_row())
        form.addRow("Output directory:", self._output_row())
        form.addRow("Charge method:", self.charge_method)
        form.addRow("Atom type:", self.atom_type)
        form.addRow("Open Babel command:", self.obabel_command)
        form.addRow("ACPYPE command:", self.acpype_command)
        root.addLayout(form)

        splitter = QSplitter(Qt.Horizontal)
        self.compound_table = QTableWidget(0, 4)
        self.compound_table.setHorizontalHeaderLabels(
            ["Compound", "Formal charge", "Assigned net charge", "Residue name"]
        )
        self.compound_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.compound_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.compound_table.verticalHeader().setVisible(False)
        self.compound_table.itemSelectionChanged.connect(self._show_selected_structure)
        splitter.addWidget(self.compound_table)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        self.preview_name = QLabel("Select a compound")
        self.preview_name.setAlignment(Qt.AlignCenter)
        self.preview_name.setStyleSheet(
            "font-size:13pt;font-weight:700;color:#174f80"
        )
        self.structure_preview = QSvgWidget()
        self.structure_preview.setMinimumSize(360, 300)
        preview_layout.addWidget(self.preview_name)
        preview_layout.addWidget(self.structure_preview, 1)
        splitter.addWidget(preview)
        splitter.setSizes([540, 400])
        root.addWidget(splitter, 1)

        outputs = QLabel(
            "Each ligand directory and its output filenames use the selected "
            "residue name, for example 4DY/4DY.sdf, 4DY/4DY.mol2, "
            "4DY/4DY.itp, 4DY/4DY.gro, and 4DY/4DY.top."
        )
        outputs.setWordWrap(True)
        outputs.setStyleSheet("color: #6b7280;")
        root.addWidget(outputs)

        buttons = QHBoxLayout()
        buttons.addStretch()
        create = QPushButton("Prepare Ligands")
        create.clicked.connect(self.create_job)
        buttons.addWidget(create)
        root.addLayout(buttons)

    @staticmethod
    def _row(field: QLineEdit, button: QPushButton) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _input_row(self) -> QWidget:
        button = QPushButton("Browse...")
        button.clicked.connect(self._select_input)
        return self._row(self.input_file, button)

    def _output_row(self) -> QWidget:
        button = QPushButton("Browse...")
        button.clicked.connect(self._select_output)
        return self._row(self.output_directory, button)

    def _select_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ligand input",
            self.working_directory,
            "Ligands (*.sdf *.mol2);;All files (*)",
        )
        if path:
            self.input_file.setText(path)
            try:
                self._load_compounds(Path(path))
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "Cannot Load Ligands", str(error))
                return
            if not self.output_directory.text().strip():
                source = Path(path)
                self.output_directory.setText(str(source.parent / "gromacs_ligands"))

    def _load_compounds(self, path: Path) -> None:
        from rdkit import Chem

        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"Ligand input does not exist: {resolved}")
        if resolved.suffix.lower() == ".sdf":
            molecules = [
                molecule
                for molecule in Chem.SDMolSupplier(str(resolved), removeHs=False)
                if molecule is not None
            ]
        elif resolved.suffix.lower() == ".mol2":
            molecule = Chem.MolFromMol2File(str(resolved), removeHs=False)
            molecules = [molecule] if molecule is not None else []
        else:
            raise ValueError("Ligand input must be SDF or MOL2")
        if not molecules:
            raise ValueError("No valid compounds were found in the selected file")
        self._molecules = molecules
        self._loaded_input = resolved
        self.compound_table.setRowCount(len(molecules))
        for row, molecule in enumerate(molecules):
            name = (
                molecule.GetProp("_Name").strip()
                if molecule.HasProp("_Name") and molecule.GetProp("_Name").strip()
                else f"ligand_{row + 1:03d}"
            )
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            formal_item = QTableWidgetItem(str(Chem.GetFormalCharge(molecule)))
            formal_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            assigned_item = QTableWidgetItem("0")
            assigned_item.setTextAlignment(Qt.AlignCenter)
            assigned_item.setToolTip(
                "Editable integer net charge used to constrain ACPYPE partial charges"
            )
            default_resname = f"L{row + 1:02d}" if len(molecules) > 1 else "LIG"
            resname_item = QTableWidgetItem(default_resname)
            resname_item.setTextAlignment(Qt.AlignCenter)
            resname_item.setToolTip(
                "Editable 1-5 character residue name written by ACPYPE, e.g. LIG or 4DY"
            )
            self.compound_table.setItem(row, 0, name_item)
            self.compound_table.setItem(row, 1, formal_item)
            self.compound_table.setItem(row, 2, assigned_item)
            self.compound_table.setItem(row, 3, resname_item)
        self.compound_table.resizeColumnsToContents()
        self.compound_table.selectRow(0)

    def _show_selected_structure(self) -> None:
        rows = self.compound_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if row >= len(self._molecules):
            return
        name = self.compound_table.item(row, 0).text()
        self.preview_name.setText(name)
        try:
            from rdkit import Chem
            from rdkit.Chem import rdDepictor
            from rdkit.Chem.Draw import rdMolDraw2D

            molecule = Chem.Mol(self._molecules[row])
            rdDepictor.Compute2DCoords(molecule)
            drawer = rdMolDraw2D.MolDraw2DSVG(420, 330)
            drawer.drawOptions().clearBackground = False
            drawer.DrawMolecule(molecule)
            drawer.FinishDrawing()
            self.structure_preview.load(
                QByteArray(drawer.GetDrawingText().encode("utf-8"))
            )
            self.structure_preview.setToolTip("")
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            # A mismatched HPC Cairo/FreeType runtime must not abort the entire
            # Qt process merely because an optional 2D preview cannot render.
            fallback = b"""<svg xmlns="http://www.w3.org/2000/svg" width="420" height="330">
<rect width="100%" height="100%" fill="#f4f6f8"/>
<text x="210" y="150" text-anchor="middle" fill="#8b1e1e" font-size="16">Structure preview unavailable</text>
<text x="210" y="178" text-anchor="middle" fill="#4a5568" font-size="12">Check the RDKit Cairo/FreeType installation</text>
</svg>"""
            self.structure_preview.load(QByteArray(fallback))
            self.structure_preview.setToolTip(str(error))

    def _assigned_charges(self) -> tuple[int, ...]:
        charges = []
        for row in range(self.compound_table.rowCount()):
            text = self.compound_table.item(row, 2).text().strip()
            try:
                charge = int(text)
            except ValueError as error:
                name = self.compound_table.item(row, 0).text()
                raise ValueError(
                    f"Assigned net charge for {name!r} must be an integer"
                ) from error
            if not -20 <= charge <= 20:
                raise ValueError("Assigned net charges must be between -20 and 20")
            charges.append(charge)
        return tuple(charges)

    def _assigned_resnames(self) -> tuple[str, ...]:
        resnames = []
        for row in range(self.compound_table.rowCount()):
            name = self.compound_table.item(row, 0).text()
            resname = self.compound_table.item(row, 3).text().strip().upper()
            if not resname or len(resname) > 5 or not resname.isalnum():
                raise ValueError(
                    f"Residue name for {name!r} must contain 1-5 letters or numbers"
                )
            resnames.append(resname)
        return tuple(resnames)

    def _select_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select ligand output directory", self.working_directory
        )
        if path:
            self.output_directory.setText(path)

    def create_job(self) -> None:
        try:
            input_file = Path(self.input_file.text().strip()).expanduser().resolve()
            if self._loaded_input != input_file:
                self._load_compounds(input_file)
            output_text = self.output_directory.text().strip()
            if not output_text:
                raise ValueError("Select an output directory")
            output = Path(output_text).expanduser().resolve()
            config = LigandPreparationConfig(
                input_file=input_file,
                output_directory=output,
                charge_method=self.charge_method.currentText(),
                atom_type=self.atom_type.currentText(),
                net_charges=self._assigned_charges(),
                resnames=self._assigned_resnames(),
                obabel_command=self.obabel_command.text().strip(),
                acpype_command=self.acpype_command.text().strip(),
            )
            task = (
                Path(self.working_directory)
                / "gromacs_ligand_prepare"
                / input_file.stem
            )
            runner, manifest = stage(config, task)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Prepare Ligands", str(error))
            return
        self.job_requested.emit(str(task), str(runner), str(manifest))
        if not self.embedded:
            self.accept()
