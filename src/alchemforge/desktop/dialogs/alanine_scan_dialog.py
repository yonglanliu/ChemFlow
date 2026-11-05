"""Compact GROMACS alanine-scanning dialog.

Collects the scan inputs, writes a scan.toml plus a runner that invokes the
``alchemforge alanine-scan`` CLI, and emits a backend job. The CLI does the
heavy lifting (expand selection -> one resFEP run per residue, then submit).
"""

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
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
    QVBoxLayout,
    QWidget,
)


class AlanineScanDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory=None, parent=None):
        super().__init__(parent)
        self.working_directory = working_directory or ""
        self.setWindowTitle("GROMACS · Alanine Scanning")
        self.resize(640, 560)
        self._build_ui()

    def _file_row(self, field, file_filter):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        button = QPushButton("Browse…")

        def choose():
            path, _ = QFileDialog.getOpenFileName(
                self, "Select file", self.working_directory, file_filter
            )
            if path:
                field.setText(path)

        button.clicked.connect(choose)
        layout.addWidget(button)
        return row

    def _build_ui(self):
        root = QVBoxLayout(self)
        title = QLabel("GROMACS · Alanine Scanning")
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")
        root.addWidget(title)
        root.addWidget(
            QLabel(
                "Each selected residue is mutated to ALA in its own resFEP run. "
                "ALA, GLY, and PRO are skipped automatically."
            )
        )

        form = QFormLayout()
        self.name_edit = QLineEdit("alanine_scan")
        self.workdir_edit = QLineEdit(self.working_directory or "work")
        self.complex_edit = QLineEdit()
        self.ligand_edit = QLineEdit()
        self.ligand_resname_edit = QLineEdit("LIG")
        self.forcefield_edit = QLineEdit("amber99sb")
        self.water_edit = QLineEdit("tip3p")
        self.chain_edit = QLineEdit("A")
        self.residues_edit = QLineEdit()
        self.residues_edit.setPlaceholderText("e.g. 40, 41, 72 (optional)")
        self.range_edit = QLineEdit()
        self.range_edit.setPlaceholderText("e.g. 45-55 (optional)")
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.0, 5.0)
        self.cutoff_spin.setSingleStep(0.1)
        self.cutoff_spin.setDecimals(2)
        self.cutoff_spin.setValue(0.0)
        self.cutoff_spin.setSuffix(" nm (0 = off)")
        self.replicates_spin = QSpinBox()
        self.replicates_spin.setRange(1, 100)
        self.replicates_spin.setValue(2)
        self.windows_spin = QSpinBox()
        self.windows_spin.setRange(2, 100)
        self.windows_spin.setValue(17)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["RESUME", "OVERWRITE", "SKIP"])

        form.addRow("Project name:", self.name_edit)
        form.addRow("Work directory:", self.workdir_edit)
        form.addRow("Complex PDB:", self._file_row(self.complex_edit, "PDB (*.pdb)"))
        form.addRow("Ligand MOL2:", self._file_row(self.ligand_edit, "MOL2 (*.mol2)"))
        form.addRow("Ligand resname:", self.ligand_resname_edit)
        form.addRow("Force field:", self.forcefield_edit)
        form.addRow("Water model:", self.water_edit)
        form.addRow("Chain:", self.chain_edit)
        form.addRow("Residues:", self.residues_edit)
        form.addRow("Residue range:", self.range_edit)
        form.addRow("Interface cutoff:", self.cutoff_spin)
        form.addRow("Replicates:", self.replicates_spin)
        form.addRow("Lambda windows:", self.windows_spin)
        form.addRow("Mode:", self.mode_combo)
        root.addLayout(form)

        buttons = QHBoxLayout()
        create = QPushButton("Create Alanine-Scan Job")
        create.clicked.connect(self.create_job)
        buttons.addStretch(1)
        buttons.addWidget(create)
        root.addLayout(buttons)

    def _scan_toml(self) -> str:
        lines = [
            "[project]",
            f'name = "{self.name_edit.text().strip()}"',
            f'work_dir = "{self.workdir_edit.text().strip()}"',
            "",
            "[system]",
            f'complex = "{self.complex_edit.text().strip()}"',
            f'force_field = "{self.forcefield_edit.text().strip()}"',
            f'water_model = "{self.water_edit.text().strip()}"',
            "",
            "[ligand]",
            f'mol2 = "{self.ligand_edit.text().strip()}"',
            f'resname = "{self.ligand_resname_edit.text().strip()}"',
            "charge = 0",
            "",
            "[simulation]",
            f"replicates = {self.replicates_spin.value()}",
            f"lambda_windows = {self.windows_spin.value()}",
            f'mode = "{self.mode_combo.currentText()}"',
            "",
            "[execution]",
            "max_parallel_jobs = 8",
            'mdrun_options = ["-ntmpi", "1"]',
            "",
            "[scan]",
            f'chain = "{self.chain_edit.text().strip()}"',
        ]
        residues = [
            r.strip() for r in self.residues_edit.text().split(",") if r.strip()
        ]
        if residues:
            lines.append("residues = [" + ", ".join(residues) + "]")
        if self.range_edit.text().strip():
            lines.append(f'residue_range = "{self.range_edit.text().strip()}"')
        if self.cutoff_spin.value() > 0:
            lines.append(f"interface_cutoff_nm = {self.cutoff_spin.value()}")
        return "\n".join(lines) + "\n"

    def create_job(self):
        name = self.name_edit.text().strip()
        workdir = Path(self.workdir_edit.text().strip() or "work").expanduser()
        if not name or not self.complex_edit.text().strip():
            QMessageBox.critical(
                self, "Missing input", "Project name and complex PDB are required."
            )
            return
        try:
            workdir.mkdir(parents=True, exist_ok=True)
            scan_toml = workdir / f"{name}.toml"
            scan_toml.write_text(self._scan_toml(), encoding="utf-8")
            runner = workdir / f"{name}_run.sh"
            runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'alchemforge alanine-scan submit "$1"\n',
                encoding="utf-8",
            )
            runner.chmod(runner.stat().st_mode | 0o100)
        except OSError as error:
            QMessageBox.critical(self, "Cannot Create Job", str(error))
            return
        self.job_requested.emit(str(workdir), str(runner), str(scan_toml))
        self.accept()
