"""Desktop dialog for AlchemFold virtual screening (sharded, HPC, ranked)."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
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

from alchemfold.screen_config import write_stage_runner

_MODELS = {"Boltz-2": "boltz", "Chai-1": "chai", "DiffDock": "diffdock"}


class AlchemFoldScreenDialog(QDialog):
    """Dock a ligand library across a SLURM array, then rank the top hits."""

    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("AlchemFold · Virtual Screening")
        self.resize(860, 760)

        root = QVBoxLayout(self)
        title = QLabel("AlchemFold — Virtual Screening")
        title.setStyleSheet("font-size:18pt; font-weight:700; color:#123f76;")
        root.addWidget(title)
        note = QLabel(
            "Dock a ligand library against the receptor in parallel: the library is "
            "split into shards, each shard is docked as one SLURM array task "
            "(across GPUs/nodes), then all results are ranked and the top hits kept. "
            "This stages the jobs; submit them on the cluster with submit_screen.sh."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.job_name = QLineEdit("screen")
        form.addRow("Job name:", self.job_name)

        self.model = QComboBox()
        self.model.addItems(list(_MODELS))
        form.addRow("Model:", self.model)

        self.protein_id = QLineEdit("A")
        form.addRow("Receptor chain id:", self.protein_id)
        self.sequence = QPlainTextEdit()
        self.sequence.setPlaceholderText("One-letter receptor sequence")
        self.sequence.setMaximumHeight(90)
        form.addRow("Receptor sequence:", self.sequence)
        self.receptor_pdb = QLineEdit()
        self.receptor_pdb.setPlaceholderText("Receptor PDB (DiffDock; optional if sequence given)")
        form.addRow("Receptor PDB:", self._file_row(self.receptor_pdb, "Select receptor PDB", "PDB (*.pdb *.cif)"))

        self.ligand_files = QPlainTextEdit()
        self.ligand_files.setPlaceholderText(
            "One library file per line (CSV / .smi / .cxsmiles, .bz2/.gz ok)"
        )
        self.ligand_files.setMaximumHeight(90)
        add_files = QPushButton("Add files…")
        add_files.clicked.connect(self._add_files)
        files_row = QHBoxLayout()
        files_row.setContentsMargins(0, 0, 0, 0)
        files_row.addWidget(self.ligand_files)
        files_row.addWidget(add_files)
        files_container = QWidget()
        files_container.setLayout(files_row)
        form.addRow("Ligand library files:", files_container)

        self.n_shards = self._spin(0, 0, 1_000_000)
        form.addRow("Number of shards (0 = use size):", self.n_shards)
        self.shard_size = self._spin(1000, 0, 10_000_000)
        form.addRow("Shard size (ligands/shard):", self.shard_size)

        # SLURM resources
        self.partition = QLineEdit("gpu")
        form.addRow("SLURM partition:", self.partition)
        self.account = QLineEdit()
        form.addRow("SLURM account:", self.account)
        self.gpus_per_task = self._spin(1, 0, 16)
        form.addRow("GPUs per task (0 = CPU):", self.gpus_per_task)
        self.gpu_type = QLineEdit()
        self.gpu_type.setPlaceholderText("Optional, e.g. a100")
        form.addRow("GPU type:", self.gpu_type)
        self.cpus_per_task = self._spin(8, 1, 256)
        form.addRow("CPUs per task:", self.cpus_per_task)
        self.nodes = self._spin(1, 1, 64)
        form.addRow("Nodes per task:", self.nodes)
        self.mem = QLineEdit("32G")
        form.addRow("Memory per task:", self.mem)
        self.walltime = QLineEdit("24:00:00")
        form.addRow("Walltime:", self.walltime)
        self.conda_env = QLineEdit("alchemfold")
        form.addRow("Conda env:", self.conda_env)

        # Ranking
        self.ranking_metric = QComboBox()
        self.ranking_metric.addItems(["auto", "confidence", "affinity", "ptm", "iptm", "plddt"])
        form.addRow("Ranking metric:", self.ranking_metric)
        self.top_k = self._spin(100, 1, 10_000_000)
        form.addRow("Keep top-K:", self.top_k)

        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        create = QPushButton("Stage Screening Jobs")
        create.clicked.connect(self.create_job)
        buttons.addWidget(cancel)
        buttons.addWidget(create)
        root.addLayout(buttons)

    def _spin(self, value, lo, hi) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(value)
        return spin

    def _file_row(self, field: QLineEdit, title: str, flt: str) -> QWidget:
        button = QPushButton("Browse…")

        def select() -> None:
            path, _ = QFileDialog.getOpenFileName(self, title, self.working_directory, flt)
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

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select ligand library files", self.working_directory,
            "Ligand libraries (*.csv *.smi *.txt *.cxsmiles *.bz2 *.gz)",
        )
        if paths:
            existing = self.ligand_files.toPlainText().strip()
            lines = ([existing] if existing else []) + paths
            self.ligand_files.setPlainText("\n".join(lines))

    def _build_config(self) -> dict:
        engine = _MODELS[self.model.currentText()]
        sequence = "".join(self.sequence.toPlainText().split())
        receptor_pdb = self.receptor_pdb.text().strip()
        if engine == "diffdock":
            if not sequence and not receptor_pdb:
                raise ValueError("DiffDock needs a receptor PDB (or a sequence).")
        elif not sequence:
            raise ValueError("Enter a receptor sequence.")

        files = [l.strip() for l in self.ligand_files.toPlainText().splitlines() if l.strip()]
        if not files:
            raise ValueError("Add at least one ligand library file.")

        protein: dict = {"id": self.protein_id.text().strip() or "A"}
        if sequence:
            protein["sequence"] = sequence

        config: dict = {
            "engine": engine,
            "screen_out": "screen_output",
            "proteins": [protein],
            "ligand_files": files,
            "n_shards": int(self.n_shards.value()),
            "shard_size": int(self.shard_size.value()),
            "slurm": {
                "partition": self.partition.text().strip() or "gpu",
                "account": self.account.text().strip(),
                "nodes": int(self.nodes.value()),
                "gpus_per_task": int(self.gpus_per_task.value()),
                "gpu_type": self.gpu_type.text().strip(),
                "cpus_per_task": int(self.cpus_per_task.value()),
                "mem": self.mem.text().strip() or "32G",
                "time": self.walltime.text().strip() or "24:00:00",
                "conda_env": self.conda_env.text().strip(),
            },
            "ranking": {
                "metric": self.ranking_metric.currentText(),
                "top_k": int(self.top_k.value()),
                "order": "auto",
                "output": "ranking.csv",
            },
        }
        if receptor_pdb:
            config["receptor_pdb"] = receptor_pdb
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
            config_path = task / f"{job_name}_screen.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            runner, _ = write_stage_runner(config_path, task)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Stage Screening", str(error))
            return
        self.job_requested.emit(str(task), str(runner), str(config_path))
        self.accept()


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = AlchemFoldScreenDialog()
    dialog.show()
    sys.exit(app.exec_())
