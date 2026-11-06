"""Desktop dialog for Boltz-2 co-folding (protein + ligand)."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from alchemfold.boltz_config import stage as stage_boltz


class CoFoldBoltzDialog(QDialog):
    """Configure and launch a Boltz-2 co-folding job (one protein + one ligand)."""

    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("AlchemFold · Boltz-2")
        self.resize(820, 680)

        root = QVBoxLayout(self)
        title = QLabel("AlchemFold — Co-Folding (Boltz-2)")
        title.setStyleSheet("font-size:18pt; font-weight:700; color:#123f76;")
        root.addWidget(title)
        note = QLabel(
            "Jointly predict a protein+ligand complex (and, optionally, its binding "
            "affinity). The predicted complex feeds the OpenMM modeling steps "
            "(build-soluble / build-membrane) and then FEP. Requires Boltz-2."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.job_name = QLineEdit("cofold_boltz")
        form.addRow("Job name:", self.job_name)

        self.protein_id = QLineEdit("A")
        form.addRow("Protein chain id:", self.protein_id)

        self.sequence = QPlainTextEdit()
        self.sequence.setPlaceholderText("One-letter protein sequence (no header)")
        self.sequence.setMaximumHeight(110)
        form.addRow("Protein sequence:", self.sequence)

        self.ligand_id = QLineEdit("L")
        form.addRow("Ligand id:", self.ligand_id)

        self.ligand_input_kind = QComboBox()
        self.ligand_input_kind.addItems(["smiles", "ccd"])
        form.addRow("Ligand input:", self.ligand_input_kind)

        self.ligand_value = QLineEdit()
        self.ligand_value.setPlaceholderText("SMILES string, or a PDB CCD code if 'ccd'")
        form.addRow("Ligand value:", self.ligand_value)

        self.predict_affinity = QCheckBox("Predict binding affinity (Boltz-2)")
        self.predict_affinity.setChecked(True)
        form.addRow("", self.predict_affinity)

        self.use_msa_server = QCheckBox("Compute MSAs via the Boltz server (needs internet)")
        self.use_msa_server.setChecked(True)
        form.addRow("", self.use_msa_server)

        self.output_format = QComboBox()
        self.output_format.addItems(["pdb", "mmcif"])
        form.addRow("Output format:", self.output_format)

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
        form.addRow("Sampling steps:", self.sampling_steps)
        self.seed = self._spin(1, 0, 2_000_000_000)
        form.addRow("Seed:", self.seed)

        self.boltz_command = QLineEdit("boltz")
        form.addRow("Boltz command:", self.boltz_command)
        self.out_dir = QLineEdit("cofold_output")
        form.addRow("Output directory:", self.out_dir)

        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        create = QPushButton("Create Co-Folding Job")
        create.clicked.connect(self.create_job)
        buttons.addWidget(cancel)
        buttons.addWidget(create)
        root.addLayout(buttons)

    def _spin(self, value, lo, hi) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(value)
        return spin

    def _build_config(self) -> dict:
        sequence = "".join(self.sequence.toPlainText().split())
        if not sequence:
            raise ValueError("Enter a protein sequence.")
        ligand_value = self.ligand_value.text().strip()
        if not ligand_value:
            raise ValueError("Enter a ligand SMILES (or a CCD code).")

        ligand: dict = {"id": self.ligand_id.text().strip() or "L"}
        ligand[self.ligand_input_kind.currentText()] = ligand_value

        config: dict = {
            "boltz_command": self.boltz_command.text().strip() or "boltz",
            "out_dir": self.out_dir.text().strip() or "cofold_output",
            "output_format": self.output_format.currentText(),
            "use_msa_server": self.use_msa_server.isChecked(),
            "diffusion_samples": int(self.diffusion_samples.value()),
            "recycling_steps": int(self.recycling_steps.value()),
            "sampling_steps": int(self.sampling_steps.value()),
            "seed": int(self.seed.value()),
            "accelerator": self.accelerator.currentText(),
            "devices": int(self.devices.value()),
            "predict_affinity": self.predict_affinity.isChecked(),
            "proteins": [
                {"id": self.protein_id.text().strip() or "A", "sequence": sequence}
            ],
            "ligands": [ligand],
        }
        if self.predict_affinity.isChecked():
            config["affinity_binder"] = ligand["id"]
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
            config_path = task / f"{job_name}_cofold_boltz.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            runner, _ = stage_boltz(config_path, task)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Create Job", str(error))
            return
        self.job_requested.emit(str(task), str(runner), str(config_path))
        self.accept()


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = CoFoldBoltzDialog()
    dialog.show()
    sys.exit(app.exec_())
