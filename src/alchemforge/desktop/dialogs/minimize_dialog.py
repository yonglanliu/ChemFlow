"""Desktop dialogs for standalone GROMACS and OpenMM minimization."""

from __future__ import annotations

import ast
import shlex
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
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
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from alchemforge.structure.minimize import (
    DEFAULT_OPENMM_FORCEFIELDS,
    GromacsMinimizationConfig,
    OpenMMMinimizationConfig,
    stage_gromacs,
    stage_openmm,
)


class _MinimizeDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())

    @staticmethod
    def _row(field: QWidget, button: QPushButton) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _file_row(self, field: QLineEdit, title: str, file_filter: str) -> QWidget:
        button = QPushButton("Browse...")

        def select() -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, title, self.working_directory, file_filter
            )
            if path:
                field.setText(path)

        button.clicked.connect(select)
        return self._row(field, button)

    def _output_row(self, field: QLineEdit) -> QWidget:
        button = QPushButton("Browse...")

        def select() -> None:
            path = QFileDialog.getExistingDirectory(
                self, "Select minimization output directory", self.working_directory
            )
            if path:
                field.setText(path)

        button.clicked.connect(select)
        return self._row(field, button)


class GromacsMinimizeDialog(_MinimizeDialog):
    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(working_directory, parent)
        self.setWindowTitle("Energy Minimization with GROMACS")
        self.resize(780, 500)
        root = QVBoxLayout(self)
        title = QLabel("GROMACS Energy Minimization")
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")
        description = QLabel(
            "Minimize an already parameterized system. The structure and topology "
            "must describe exactly the same atoms and molecules."
        )
        description.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(description)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.structure = QLineEdit()
        self.topology = QLineEdit()
        self.output_directory = QLineEdit()
        self.gmx_command = QLineEdit("gmx")
        self.steps = QSpinBox()
        self.steps.setRange(1, 10000000)
        self.steps.setValue(50000)
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.001, 1000000.0)
        self.tolerance.setValue(1000.0)
        self.step_size = QDoubleSpinBox()
        self.step_size.setRange(0.000001, 1.0)
        self.step_size.setDecimals(6)
        self.step_size.setValue(0.01)
        self.mdrun_options = QLineEdit()
        self.mdrun_options.setPlaceholderText("Example: -ntmpi 1 -nb gpu -pme gpu")
        form.addRow(
            "Structure:",
            self._file_row(
                self.structure, "Select GRO or PDB", "Structures (*.gro *.pdb)"
            ),
        )
        form.addRow(
            "Topology:",
            self._file_row(
                self.topology, "Select GROMACS topology", "Topology (*.top)"
            ),
        )
        form.addRow("Output directory:", self._output_row(self.output_directory))
        form.addRow("GROMACS command:", self.gmx_command)
        form.addRow("Maximum steps:", self.steps)
        form.addRow("Tolerance (kJ/mol/nm):", self.tolerance)
        form.addRow("Step size (nm):", self.step_size)
        form.addRow("mdrun options:", self.mdrun_options)
        root.addLayout(form)
        root.addStretch()
        button_row = QHBoxLayout()
        button_row.addStretch()
        run = QPushButton("Minimize with GROMACS")
        run.clicked.connect(self.create_job)
        button_row.addWidget(run)
        root.addLayout(button_row)

    def create_job(self) -> None:
        try:
            structure = Path(self.structure.text().strip()).expanduser().resolve()
            output_text = self.output_directory.text().strip()
            if not output_text:
                raise ValueError("Select an output directory")
            output = Path(output_text).expanduser().resolve()
            config = GromacsMinimizationConfig(
                structure=structure,
                topology=Path(self.topology.text().strip()).expanduser().resolve(),
                output_directory=output,
                gmx_command=self.gmx_command.text().strip(),
                steps=self.steps.value(),
                tolerance=self.tolerance.value(),
                step_size=self.step_size.value(),
                mdrun_options=tuple(shlex.split(self.mdrun_options.text().strip())),
            )
            task = (
                Path(self.working_directory)
                / "minimization"
                / "gromacs"
                / structure.stem
            )
            runner, manifest = stage_gromacs(config, task)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Minimize System", str(error))
            return
        self.job_requested.emit(str(task), str(runner), str(manifest))
        self.accept()


class OpenMMMinimizeDialog(_MinimizeDialog):
    def __init__(
        self,
        working_directory: str | None = None,
        parent=None,
        *,
        parameterization_mode: bool = False,
    ):
        super().__init__(working_directory, parent)
        self.parameterization_mode = parameterization_mode
        self.setWindowTitle(
            "PDB + SDF to GROMACS Parameterization"
            if parameterization_mode
            else "Energy Minimization with OpenFE / OpenMM"
        )
        self.resize(860, 720)
        self._small_molecule_paths: list[Path] = []
        root = QVBoxLayout(self)
        title = QLabel(
            "PDB + SDF → GROMACS Parameterization"
            if parameterization_mode
            else "OpenFE / OpenMM Energy Minimization"
        )
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")
        description = QLabel(
            (
                "Parameterize an assembled PDB/mmCIF with Amber protein/lipid "
                "force fields and charged ligand/cofactor SDF files, minimize it, "
                "and export system.gro plus topol.top."
            )
            if parameterization_mode
            else (
                "Build an OpenMM system using the force fields below and minimize "
                "it. Every residue—including lipids, ions, cofactors, and "
                "ligands—must have a matching force-field template. Use "
                "complex.cif for systems with more than 99,999 atoms."
            )
        )
        description.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(description)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.structure = QLineEdit()
        self.small_molecules = QLineEdit()
        self.small_molecules.setReadOnly(True)
        self.small_molecules.setPlaceholderText(
            "Optional: charged ligand/cofactor SDF or MOL2 files"
        )
        self.output_directory = QLineEdit()
        self.platform = QComboBox()
        self.platform.addItems(["CPU", "CUDA", "OpenCL", "Reference"])
        self.steps = QSpinBox()
        self.steps.setRange(1, 10000000)
        self.steps.setValue(50000)
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.0001, 1000000.0)
        self.tolerance.setDecimals(4)
        self.tolerance.setValue(10.0)
        self.periodic = QCheckBox("Use PME periodic minimization")
        self.periodic.setChecked(parameterization_mode)
        self.normalize_lipids = QCheckBox(
            "Normalize recognized Desmond lipid names for OpenMM"
        )
        self.normalize_lipids.setChecked(True)
        self.normalize_lipids.setToolTip(
            "Currently converts Desmond-style POPC headgroup atom names to the "
            "Amber lipid17/CHARMM POPC convention without changing coordinates."
        )
        self.export_gromacs = QCheckBox(
            "Write system.gro and topol.top for GROMACS"
        )
        self.export_gromacs.setChecked(parameterization_mode)
        self.box_vectors = QLineEdit()
        self.box_vectors.setPlaceholderText(
            "Optional: [[a_x,a_y,a_z],[b_x,b_y,b_z],[c_x,c_y,c_z]] nm"
        )
        self.forcefields = QPlainTextEdit("\n".join(DEFAULT_OPENMM_FORCEFIELDS))
        self.forcefields.setMaximumHeight(120)
        self.template_generator = QComboBox()
        self.template_generator.addItem("SMIRNOFF (OpenFF)", "smirnoff")
        self.template_generator.addItem("GAFF (AmberTools)", "gaff")
        self.small_molecule_forcefield = QLineEdit("openff-2.2.1")
        self.template_generator.currentIndexChanged.connect(
            self._set_template_forcefield_default
        )
        form.addRow(
            "Assembled structure:",
            self._file_row(
                self.structure,
                "Select assembled PDB or mmCIF",
                "Structures (*.cif *.mmcif *.pdb)",
            ),
        )
        form.addRow("Ligand/cofactor definitions:", self._small_molecule_row())
        form.addRow("Template generator:", self.template_generator)
        form.addRow("Small-molecule force field:", self.small_molecule_forcefield)
        form.addRow("Output directory:", self._output_row(self.output_directory))
        form.addRow("OpenMM platform:", self.platform)
        form.addRow("Maximum iterations:", self.steps)
        form.addRow("Tolerance (kJ/mol/nm):", self.tolerance)
        form.addRow("Periodic system:", self.periodic)
        form.addRow("Desmond lipids:", self.normalize_lipids)
        form.addRow("GROMACS export:", self.export_gromacs)
        form.addRow("Box vectors:", self.box_vectors)
        form.addRow("Force-field XML files:", self.forcefields)
        root.addLayout(form)
        warning = QLabel(
            "For membrane systems, enable periodic minimization and supply box "
            "vectors when the PDB has no CRYST1 record. Template errors mean the "
            "selected force fields do not support one or more residues. Add a "
            "charged SDF/MOL2 for every nonstandard ligand or cofactor."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #9a5b00;")
        root.addWidget(warning)
        root.addStretch()
        button_row = QHBoxLayout()
        button_row.addStretch()
        run = QPushButton(
            "Parameterize and Validate"
            if parameterization_mode
            else "Minimize with OpenMM"
        )
        run.clicked.connect(self.create_job)
        button_row.addWidget(run)
        root.addLayout(button_row)

    def _small_molecule_row(self) -> QWidget:
        button = QPushButton("Browse...")
        button.clicked.connect(self._select_small_molecules)
        return self._row(self.small_molecules, button)

    def _select_small_molecules(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select ligand/cofactor definitions",
            self.working_directory,
            "Small molecules (*.sdf *.mol2);;SDF (*.sdf);;MOL2 (*.mol2)",
        )
        if paths:
            self._small_molecule_paths = [Path(path).resolve() for path in paths]
            self.small_molecules.setText(
                "; ".join(str(path) for path in self._small_molecule_paths)
            )

    def _set_template_forcefield_default(self) -> None:
        generator = self.template_generator.currentData()
        self.small_molecule_forcefield.setText(
            "openff-2.2.1" if generator == "smirnoff" else "gaff-2.11"
        )

    def _parsed_vectors(self) -> tuple[tuple[float, float, float], ...] | None:
        text = self.box_vectors.text().strip()
        if not text:
            return None
        value = ast.literal_eval(text)
        try:
            vectors = tuple(tuple(float(number) for number in row) for row in value)
        except (TypeError, ValueError) as error:
            raise ValueError("Box vectors must be a numeric 3 by 3 array") from error
        if len(vectors) != 3 or any(len(vector) != 3 for vector in vectors):
            raise ValueError("Box vectors must be a numeric 3 by 3 array")
        return vectors

    def create_job(self) -> None:
        try:
            structure = Path(self.structure.text().strip()).expanduser().resolve()
            output_text = self.output_directory.text().strip()
            if not output_text:
                raise ValueError("Select an output directory")
            output = Path(output_text).expanduser().resolve()
            forcefields = tuple(
                line.strip()
                for line in self.forcefields.toPlainText().splitlines()
                if line.strip()
            )
            config = OpenMMMinimizationConfig(
                structure=structure,
                output_directory=output,
                forcefields=forcefields,
                platform=self.platform.currentText(),
                steps=self.steps.value(),
                tolerance=self.tolerance.value(),
                periodic=self.periodic.isChecked(),
                box_vectors_nm=self._parsed_vectors(),
                small_molecule_files=tuple(self._small_molecule_paths),
                template_generator=str(self.template_generator.currentData()),
                small_molecule_forcefield=self.small_molecule_forcefield.text().strip(),
                normalize_desmond_lipids=self.normalize_lipids.isChecked(),
                export_gromacs=self.export_gromacs.isChecked(),
            )
            task = (
                Path(self.working_directory)
                / "minimization"
                / "openmm"
                / structure.stem
            )
            runner, manifest = stage_openmm(config, task)
        except (OSError, SyntaxError, TypeError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Minimize System", str(error))
            return
        self.job_requested.emit(str(task), str(runner), str(manifest))
        self.accept()
