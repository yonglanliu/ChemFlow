"""Desktop dialog for AlchemFold docking (multi-engine co-folding)."""

from __future__ import annotations

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

from alchemfold.dock_config import stage as stage_dock

# Display label -> engine key.
_MODELS = {"Boltz-2": "boltz", "Chai-1": "chai", "DiffDock": "diffdock"}


class AlchemFoldDockingDialog(QDialog):
    """Dock a ligand into a receptor by co-folding, with a selectable model."""

    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("AlchemFold · Docking")
        self.resize(840, 760)

        root = QVBoxLayout(self)
        title = QLabel("AlchemFold — Docking (Co-Folding)")
        title.setStyleSheet("font-size:18pt; font-weight:700; color:#123f76;")
        root.addWidget(title)
        note = QLabel(
            "Jointly predict a receptor+ligand complex with the chosen model. "
            "Optionally steer the ligand into a known binding-site pocket. The "
            "predicted complex feeds OpenMM modeling -> FEP."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.job_name = QLineEdit("docking")
        form.addRow("Job name:", self.job_name)

        self.model = QComboBox()
        self.model.addItems(list(_MODELS))
        self.model.currentTextChanged.connect(self._sync_model)
        form.addRow("Model:", self.model)

        self.protein_id = QLineEdit("A")
        form.addRow("Receptor chain id:", self.protein_id)

        self.sequence = QPlainTextEdit()
        self.sequence.setPlaceholderText("One-letter receptor sequence (no header)")
        self.sequence.setMaximumHeight(100)
        form.addRow("Receptor sequence:", self.sequence)

        self.receptor_pdb = QLineEdit()
        self.receptor_pdb.setPlaceholderText("Receptor PDB structure (DiffDock; optional if sequence given)")
        form.addRow("Receptor PDB:", self._file_row(self.receptor_pdb, "Select receptor PDB", "PDB (*.pdb *.cif)"))

        self.ligand_id = QLineEdit("L")
        form.addRow("Ligand id:", self.ligand_id)
        self.ligand_kind = QComboBox()
        self.ligand_kind.addItems(["smiles", "ccd"])
        form.addRow("Ligand input:", self.ligand_kind)
        self.ligand_value = QLineEdit()
        self.ligand_value.setPlaceholderText("SMILES string, or a PDB CCD code if 'ccd'")
        form.addRow("Ligand value:", self.ligand_value)

        # Ligand library file (CSV / SMILES) — screens one run per entry and
        # overrides the single ligand above when set.
        self.ligand_file = QLineEdit()
        self.ligand_file.setPlaceholderText("Optional: CSV / .smi / .cxsmiles library (.bz2/.gz ok) to screen")
        self.ligand_file.textChanged.connect(self._sync_ligand_file)
        form.addRow("Ligand file:", self._file_row(
            self.ligand_file, "Select ligand library",
            "Ligand libraries (*.csv *.smi *.txt *.cxsmiles *.bz2 *.gz)"))
        self.ligand_file_format = QComboBox()
        self.ligand_file_format.addItems(["auto", "csv", "smiles", "cxsmiles"])
        form.addRow("Ligand file format:", self.ligand_file_format)

        # Pocket / binding-site constraint
        self.use_pocket = QCheckBox("Restrain the ligand to a binding-site pocket")
        self.use_pocket.toggled.connect(self._sync_pocket)
        form.addRow("", self.use_pocket)
        self.pocket_contacts = QLineEdit()
        self.pocket_contacts.setPlaceholderText("Pocket residues, e.g. A:123, A:145, A:201")
        form.addRow("Pocket residues:", self.pocket_contacts)
        self.pocket_distance = QDoubleSpinBox()
        self.pocket_distance.setRange(1.0, 50.0)
        self.pocket_distance.setDecimals(1)
        self.pocket_distance.setValue(6.0)
        form.addRow("Pocket max distance (Å):", self.pocket_distance)

        self.predict_affinity = QCheckBox("Predict binding affinity (Boltz-2 only)")
        form.addRow("", self.predict_affinity)

        self.use_msa_server = QCheckBox("Compute MSAs via the model server (needs internet)")
        self.use_msa_server.setChecked(True)
        form.addRow("", self.use_msa_server)

        self.accelerator = QComboBox()
        self.accelerator.addItems(["gpu", "cpu", "tpu"])
        form.addRow("Accelerator:", self.accelerator)
        self.devices = self._spin(1, 1, 64)
        form.addRow("Devices:", self.devices)

        self.diffusion_samples = self._spin(1, 1, 100)
        form.addRow("Diffusion samples (poses):", self.diffusion_samples)
        self.recycling_steps = self._spin(3, 1, 20)
        form.addRow("Recycling steps:", self.recycling_steps)
        self.sampling_steps = self._spin(200, 1, 10000)
        form.addRow("Sampling steps (Boltz):", self.sampling_steps)
        self.seed = self._spin(1, 0, 2_000_000_000)
        form.addRow("Seed:", self.seed)

        self.output_format = QComboBox()
        self.output_format.addItems(["pdb", "mmcif"])
        form.addRow("Output format (Boltz):", self.output_format)

        self.engine_command = QLineEdit("boltz")
        form.addRow("Engine command:", self.engine_command)
        self.out_dir = QLineEdit("docking_output")
        form.addRow("Output directory:", self.out_dir)

        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        create = QPushButton("Create Docking Job")
        create.clicked.connect(self.create_job)
        buttons.addWidget(cancel)
        buttons.addWidget(create)
        root.addLayout(buttons)

        self._sync_model(self.model.currentText())
        self._sync_pocket(self.use_pocket.isChecked())

    def _spin(self, value, lo, hi) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(value)
        return spin

    def _file_row(self, field: QLineEdit, title: str, file_filter: str) -> QWidget:
        button = QPushButton("Browse…")

        def select() -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, title, self.working_directory, file_filter
            )
            if path:
                field.setText(path)

        button.clicked.connect(select)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)
        container = QWidget()
        container.setLayout(layout)
        return container

    def _sync_model(self, label: str) -> None:
        engine = _MODELS.get(label, "boltz")
        is_boltz = engine == "boltz"
        is_diffdock = engine == "diffdock"
        is_cofold = not is_diffdock  # boltz or chai (sequence-based co-folding)

        # Affinity, sampling steps and output format are Boltz-only.
        self.predict_affinity.setEnabled(is_boltz)
        if not is_boltz:
            self.predict_affinity.setChecked(False)
        self.sampling_steps.setEnabled(is_boltz)
        self.output_format.setEnabled(is_boltz)

        # DiffDock: blind docking into a receptor structure — no pocket, no MSA,
        # no recycling; a receptor PDB is the primary input.
        if is_diffdock and self.use_pocket.isChecked():
            self.use_pocket.setChecked(False)
        self.use_pocket.setEnabled(is_cofold)
        self.use_msa_server.setEnabled(is_cofold)
        self.recycling_steps.setEnabled(is_cofold)

        # Reset the engine command default when it still holds a stock default.
        if self.engine_command.text().strip() in {"", "boltz", "chai-lab", "python -m inference"}:
            self.engine_command.setText(
                {"boltz": "boltz", "chai": "chai-lab", "diffdock": "python -m inference"}[engine]
            )

    def _sync_pocket(self, on: bool) -> None:
        self.pocket_contacts.setEnabled(on)
        self.pocket_distance.setEnabled(on)

    def _sync_ligand_file(self, text: str) -> None:
        has_file = bool(text.strip())
        self.ligand_value.setEnabled(not has_file)
        self.ligand_kind.setEnabled(not has_file)
        self.ligand_file_format.setEnabled(has_file)

    def _parse_contacts(self) -> list[dict]:
        contacts: list[dict] = []
        for token in self.pocket_contacts.text().replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"pocket residue '{token}' must be chain:residue, e.g. A:123")
            chain, residue = token.split(":", 1)
            contacts.append({"chain": chain.strip(), "residue": int(residue.strip())})
        if not contacts:
            raise ValueError("Enter at least one pocket residue (chain:residue).")
        return contacts

    def _build_config(self) -> dict:
        engine = _MODELS[self.model.currentText()]
        sequence = "".join(self.sequence.toPlainText().split())
        receptor_pdb = self.receptor_pdb.text().strip()
        if engine == "diffdock":
            if not sequence and not receptor_pdb:
                raise ValueError("DiffDock needs a receptor PDB (or a sequence to fold).")
        elif not sequence:
            raise ValueError("Enter a receptor sequence.")

        ligand_file = self.ligand_file.text().strip()
        ligand_value = self.ligand_value.text().strip()
        if not ligand_file and not ligand_value:
            raise ValueError("Enter a ligand SMILES/CCD, or select a ligand file.")

        protein: dict = {"id": self.protein_id.text().strip() or "A"}
        if sequence:
            protein["sequence"] = sequence
        ligand_id = self.ligand_id.text().strip() or "L"

        config: dict = {
            "engine": engine,
            "out_dir": self.out_dir.text().strip() or "docking_output",
            "output_format": self.output_format.currentText(),
            "use_msa_server": self.use_msa_server.isChecked(),
            "seed": int(self.seed.value()),
            "accelerator": self.accelerator.currentText(),
            "devices": int(self.devices.value()),
            "diffusion_samples": int(self.diffusion_samples.value()),
            "recycling_steps": int(self.recycling_steps.value()),
            "sampling_steps": int(self.sampling_steps.value()),
            "predict_affinity": self.predict_affinity.isChecked(),
            "proteins": [protein],
        }
        if ligand_file:
            # Screen a library: one run per entry; ids come from the file.
            config["ligand_file"] = ligand_file
            config["ligand_file_format"] = self.ligand_file_format.currentText()
        else:
            ligand: dict = {"id": ligand_id}
            ligand[self.ligand_kind.currentText()] = ligand_value
            config["ligands"] = [ligand]

        command = self.engine_command.text().strip()
        command_key = {
            "boltz": "boltz_command",
            "chai": "chai_command",
            "diffdock": "diffdock_command",
        }[engine]
        default_cmd = {"boltz": "boltz", "chai": "chai-lab", "diffdock": "python -m inference"}[engine]
        config[command_key] = command or default_cmd

        if receptor_pdb:
            config["receptor_pdb"] = receptor_pdb
        # affinity_binder / pocket.binder only apply to the inline single ligand;
        # for a screened library the runner assigns them per entry.
        if self.predict_affinity.isChecked() and not ligand_file:
            config["affinity_binder"] = ligand_id
        if self.use_pocket.isChecked():
            pocket: dict = {
                "max_distance": float(self.pocket_distance.value()),
                "contacts": self._parse_contacts(),
            }
            if not ligand_file:
                pocket["binder"] = ligand_id
            config["pocket"] = pocket
        return config

    def create_job(self) -> None:
        import yaml

        job_name = self.job_name.text().strip()
        if not job_name:
            QMessageBox.warning(self, "Missing Job Name", "Please enter a job name.")
            return
        try:
            config = self._build_config()
            task = Path(self.working_directory).expanduser().resolve() / job_name
            task.mkdir(parents=True, exist_ok=True)
            config_path = task / f"{job_name}_docking.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            runner, _ = stage_dock(config_path, task)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Create Job", str(error))
            return
        self.job_requested.emit(str(task), str(runner), str(config_path))
        self.accept()


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = AlchemFoldDockingDialog()
    dialog.show()
    sys.exit(app.exec_())
