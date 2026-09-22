"""Local and Slurm training workspace for ChemFlow Studio."""

from __future__ import annotations

import json
import shlex
import sys
import uuid
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, QTimer, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from chemflow.config import PROJECT_ROOT
from chemflow.desktop.training_parameters import (
    MODEL_PARAMETER_SCHEMAS,
    write_configuration,
)
from chemflow.desktop.widgets.forms import Card, PageHeader, PathField, action_button, field
from chemflow.desktop.widgets.training_config_editor import TrainingConfigurationEditor


MODEL_COMMANDS = {
    "Conventional ML": "ml",
    "Graphormer": "hf-graphormer",
    "CheMeleon": "chemeleon",
    "ChemBERTa": "chemberta",
    "Chemprop": "chemprop",
    "KERMT": "kermt",
    "GPT": "gpt",
    "LSTM": "lstm",
}

ACTIVE_SLURM_STATES = {
    "CONFIGURING", "COMPLETING", "PENDING", "PREEMPTED", "REQUEUED",
    "RESIZING", "RUNNING", "SUSPENDED",
}


@dataclass
class TrainingJob:
    key: str
    name: str
    model: str
    backend: str
    config: str
    started: str
    status: str = "Queued"
    external_id: str = ""
    host: str = ""
    log: str = ""
    remote_log: str = ""
    remote_output: str = ""
    process: QProcess | None = dataclass_field(default=None, repr=False, compare=False)

    def persisted(self) -> dict[str, str]:
        return {
            "key": self.key, "name": self.name, "model": self.model,
            "backend": self.backend, "config": self.config,
            "started": self.started, "status": self.status,
            "external_id": self.external_id, "host": self.host,
            "log": self.log[-20_000:], "remote_log": self.remote_log,
            "remote_output": self.remote_output[-20_000:],
        }


class TrainingCenterPage(QWidget):
    """Launch and monitor local or remote model-training jobs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("ChemFlow", "ChemFlow Studio")
        self.jobs: list[TrainingJob] = []
        self.auxiliary_processes: list[QProcess] = []
        self._build_ui()
        self._restore_settings()
        self._restore_jobs()
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(15_000)
        self.poll_timer.timeout.connect(self.refresh_jobs)
        self.poll_timer.start()

    def _build_ui(self) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(18)
        layout.addWidget(PageHeader(
            "Model studio", "Training center",
            "Configure, launch, and monitor local or Slurm experiments from one workspace.",
        ))

        launcher = Card(
            "New training job",
            "Use the same ChemFlow configuration and CLI locally or on the HPC.",
        )
        self.model = QComboBox(); self.model.addItems(MODEL_COMMANDS)
        self.backend = QComboBox(); self.backend.addItems(["Local", "Slurm HPC"])
        self.config_source = QComboBox()
        self.config_source.addItems(["Existing configuration", "Build configuration"])
        self.job_name = QLineEdit("chemflow-training")
        self.seed = QSpinBox(); self.seed.setRange(0, 2_147_483_647); self.seed.setValue(42)
        self.local_config = PathField("TOML, YAML, or JSON configuration")
        common = QGridLayout(); common.setHorizontalSpacing(18); common.setVerticalSpacing(13)
        common.addWidget(field("Model family", self.model), 0, 0)
        common.addWidget(field("Execution", self.backend), 0, 1)
        common.addWidget(field("Job name", self.job_name), 1, 0)
        common.addWidget(field("Random seed", self.seed, "Applied to GPT; other models read their configuration."), 1, 1)
        common.addWidget(field("Configuration source", self.config_source), 2, 0, 1, 2)
        self.local_config_field = field("Configuration file", self.local_config)
        common.addWidget(self.local_config_field, 3, 0, 1, 2)
        launcher.body.addLayout(common)

        self.builder_panel = QWidget()
        builder_layout = QVBoxLayout(self.builder_panel)
        builder_layout.setContentsMargins(0, 4, 0, 0)
        builder_layout.setSpacing(10)
        self.config_editor = TrainingConfigurationEditor()
        self.config_editor.toolbox.setMinimumHeight(390)
        self.generated_config = PathField(
            "Generated configuration path",
            mode="save",
            save_filter="TOML (*.toml);;JSON (*.json)",
        )
        save_row = QHBoxLayout()
        save_row.addWidget(field("Save generated configuration", self.generated_config), 1)
        save_config = QPushButton("Save configuration")
        save_config.clicked.connect(lambda: self._save_generated_configuration())
        save_row.addWidget(save_config, alignment=Qt.AlignmentFlag.AlignBottom)
        builder_layout.addWidget(self.config_editor)
        builder_layout.addLayout(save_row)
        self.builder_panel.setVisible(False)
        launcher.body.addWidget(self.builder_panel)

        self.local_panel = QWidget(); local_layout = QVBoxLayout(self.local_panel)
        local_layout.setContentsMargins(0, 4, 0, 0)
        local_help = QLabel("Runs in this desktop's Python environment. Closing the desktop stops active local jobs.")
        local_help.setObjectName("Muted"); local_help.setWordWrap(True)
        local_layout.addWidget(local_help)
        launcher.body.addWidget(self.local_panel)

        self.remote_panel = QWidget(); remote_grid = QGridLayout(self.remote_panel)
        remote_grid.setContentsMargins(0, 4, 0, 0); remote_grid.setHorizontalSpacing(18); remote_grid.setVerticalSpacing(13)
        self.ssh_host = QLineEdit(); self.ssh_host.setPlaceholderText("user@login.cluster.example")
        self.remote_repo = QLineEdit("/vf/users/liuy48/ChemFlow")
        self.remote_config = QLineEdit(); self.remote_config.setPlaceholderText("/data/project/config.toml")
        self.conda_init = QLineEdit("$HOME/bin/myconda")
        self.conda_env = QLineEdit("chemflow")
        self.partition = QLineEdit("gpu")
        self.account = QLineEdit()
        self.gpus = QSpinBox(); self.gpus.setRange(0, 64); self.gpus.setValue(1)
        self.cpus = QSpinBox(); self.cpus.setRange(1, 512); self.cpus.setValue(8)
        self.memory = QLineEdit("64G"); self.walltime = QLineEdit("24:00:00")
        remote_grid.addWidget(field("SSH host", self.ssh_host), 0, 0)
        remote_grid.addWidget(field("Remote repository", self.remote_repo), 0, 1)
        remote_grid.addWidget(field("Remote configuration", self.remote_config), 1, 0, 1, 2)
        remote_grid.addWidget(field("Conda initialization", self.conda_init), 2, 0)
        remote_grid.addWidget(field("Conda environment", self.conda_env), 2, 1)
        remote_grid.addWidget(field("Partition", self.partition), 3, 0)
        remote_grid.addWidget(field("Account (optional)", self.account), 3, 1)
        remote_grid.addWidget(field("GPUs", self.gpus), 4, 0)
        remote_grid.addWidget(field("CPUs", self.cpus), 4, 1)
        remote_grid.addWidget(field("Memory", self.memory), 5, 0)
        remote_grid.addWidget(field("Wall time", self.walltime), 5, 1)
        self.remote_panel.setVisible(False)
        launcher.body.addWidget(self.remote_panel)
        launch = action_button("Launch training"); launch.clicked.connect(self.launch)
        launcher.body.addWidget(launch, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(launcher)

        monitor = Card("Job monitor", "Local and Slurm states refresh automatically every 15 seconds.")
        controls = QHBoxLayout()
        refresh = QPushButton("Refresh"); refresh.clicked.connect(self.refresh_jobs)
        stop = QPushButton("Cancel selected"); stop.setObjectName("Danger"); stop.clicked.connect(self.cancel_selected)
        clear = QPushButton("Remove finished"); clear.clicked.connect(self.remove_finished)
        controls.addWidget(refresh); controls.addWidget(stop); controls.addWidget(clear); controls.addStretch()
        monitor.body.addLayout(controls)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Job", "Model", "Backend", "Status", "ID / PID", "Started", "Configuration"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False); self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(210); self.table.itemSelectionChanged.connect(self._show_selected_log)
        monitor.body.addWidget(self.table)
        self.output = QPlainTextEdit(); self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(8000); self.output.setMinimumHeight(190)
        self.output.setPlaceholderText("Select a job to inspect submission and training output.")
        monitor.body.addWidget(QLabel("Selected job output")); monitor.body.addWidget(self.output)
        layout.addWidget(monitor); layout.addStretch()

        scroll = QScrollArea(self); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setWidget(content)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.addWidget(scroll)
        self.backend.currentIndexChanged.connect(self._backend_changed)
        self.model.currentTextChanged.connect(self._model_changed)
        self.config_source.currentIndexChanged.connect(self._config_source_changed)
        self._model_changed(self.model.currentText())

    def _backend_changed(self, index: int) -> None:
        self.local_panel.setVisible(index == 0)
        self.remote_panel.setVisible(index == 1)

    def _config_source_changed(self, index: int) -> None:
        self.local_config_field.setVisible(index == 0)
        self.builder_panel.setVisible(index == 1)

    def _model_changed(self, model_label: str) -> None:
        self.seed.setEnabled(MODEL_COMMANDS[model_label] == "gpt")
        self.config_editor.set_model(model_label)
        schema = MODEL_PARAMETER_SCHEMAS[model_label]
        stem = MODEL_COMMANDS[model_label].replace("hf-", "")
        self.generated_config.setText(str(Path.cwd() / f"{stem}_training{schema.extension}"))

    def _save_generated_configuration(self, *, notify: bool = True) -> Path | None:
        destination = self.generated_config.text()
        if not destination:
            self._warn("Choose where to save the generated configuration.")
            return None
        try:
            path = write_configuration(
                self.model.currentText(),
                self.config_editor.values(),
                destination,
            )
        except (OSError, TypeError, ValueError) as error:
            self._warn(f"Could not save configuration: {error}")
            return None
        self.generated_config.setText(str(path))
        if notify:
            QMessageBox.information(self, "Configuration saved", str(path))
        return path

    def _restore_settings(self) -> None:
        values = {
            self.ssh_host: "training/ssh_host", self.remote_repo: "training/remote_repo",
            self.conda_init: "training/conda_init", self.conda_env: "training/conda_env",
            self.partition: "training/partition", self.account: "training/account",
            self.memory: "training/memory", self.walltime: "training/walltime",
        }
        for widget, key in values.items():
            stored = self.settings.value(key)
            if stored is not None:
                widget.setText(str(stored))

    def _save_settings(self) -> None:
        values = {
            "training/ssh_host": self.ssh_host.text().strip(),
            "training/remote_repo": self.remote_repo.text().strip(),
            "training/conda_init": self.conda_init.text().strip(),
            "training/conda_env": self.conda_env.text().strip(),
            "training/partition": self.partition.text().strip(),
            "training/account": self.account.text().strip(),
            "training/memory": self.memory.text().strip(),
            "training/walltime": self.walltime.text().strip(),
        }
        for key, value in values.items(): self.settings.setValue(key, value)

    def _restore_jobs(self) -> None:
        try: stored = json.loads(str(self.settings.value("training/jobs", "[]")))
        except (TypeError, ValueError, json.JSONDecodeError): stored = []
        for item in stored:
            if not isinstance(item, dict): continue
            try: job = TrainingJob(**item)
            except TypeError: continue
            if job.backend == "Local" and job.status in {"Queued", "Running"}: job.status = "Disconnected"
            self.jobs.append(job)
        self._render_jobs()

    def _persist_jobs(self) -> None:
        self.settings.setValue("training/jobs", json.dumps([job.persisted() for job in self.jobs[-100:]]))

    def launch(self) -> None:
        model_label = self.model.currentText(); model_command = MODEL_COMMANDS[model_label]
        name = self.job_name.text().strip() or f"chemflow-{model_command}"
        generated_path = None
        if self.config_source.currentIndex() == 1:
            generated_path = self._save_generated_configuration(notify=False)
            if generated_path is None:
                return
        if self.backend.currentText() == "Local":
            config = str(generated_path) if generated_path else self.local_config.text()
            if not config: self._warn("Choose a local training configuration."); return
            if not Path(config).expanduser().is_file(): self._warn(f"Configuration file does not exist: {config}"); return
            self._launch_local(name, model_label, model_command, config); return
        required = {
            "SSH host": self.ssh_host.text().strip(), "remote repository": self.remote_repo.text().strip(),
            "remote configuration": self.remote_config.text().strip(), "Conda environment": self.conda_env.text().strip(),
            "partition": self.partition.text().strip(), "memory": self.memory.text().strip(), "wall time": self.walltime.text().strip(),
        }
        missing = [label for label, value in required.items() if not value]
        if missing: self._warn("Complete these HPC fields: " + ", ".join(missing)); return
        self._save_settings()
        if generated_path is not None:
            self._upload_configuration(
                name,
                model_label,
                model_command,
                generated_path,
            )
        else:
            self._launch_slurm(name, model_label, model_command)

    def _upload_configuration(
        self,
        name: str,
        model_label: str,
        model_command: str,
        local_path: Path,
    ) -> None:
        host = self.ssh_host.text().strip()
        remote_path = self.remote_config.text().strip()
        if any(character.isspace() for character in remote_path):
            self._warn("Remote configuration paths containing whitespace are not supported.")
            return
        remote_parent = str(Path(remote_path).parent)
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.finished.connect(
            lambda code, _status, p=process: self._remote_directory_ready(
                p,
                int(code),
                name,
                model_label,
                model_command,
                local_path,
            )
        )
        self.auxiliary_processes.append(process)
        process.start(
            "ssh",
            self._ssh_arguments(host, f"mkdir -p {shlex.quote(remote_parent)}"),
        )

    def _remote_directory_ready(
        self,
        process: QProcess,
        exit_code: int,
        name: str,
        model_label: str,
        model_command: str,
        local_path: Path,
    ) -> None:
        if process in self.auxiliary_processes:
            self.auxiliary_processes.remove(process)
        detail = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        process.deleteLater()
        if exit_code != 0:
            self._warn(f"Could not create the remote configuration directory:\n{detail}")
            return
        upload = QProcess(self)
        upload.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        upload.finished.connect(
            lambda code, _status, p=upload: self._configuration_uploaded(
                p,
                int(code),
                name,
                model_label,
                model_command,
            )
        )
        self.auxiliary_processes.append(upload)
        upload.start(
            "scp",
            [
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                str(local_path),
                f"{self.ssh_host.text().strip()}:{self.remote_config.text().strip()}",
            ],
        )

    def _configuration_uploaded(
        self,
        process: QProcess,
        exit_code: int,
        name: str,
        model_label: str,
        model_command: str,
    ) -> None:
        if process in self.auxiliary_processes:
            self.auxiliary_processes.remove(process)
        detail = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        process.deleteLater()
        if exit_code != 0:
            self._warn(f"Could not upload the generated configuration:\n{detail}")
            return
        self._launch_slurm(name, model_label, model_command)

    def _new_job(self, name: str, model: str, backend: str, config: str) -> TrainingJob:
        job = TrainingJob(uuid.uuid4().hex, name, model, backend, config, datetime.now().astimezone().isoformat(timespec="seconds"))
        self.jobs.insert(0, job); self._render_jobs(); self.table.selectRow(0); return job

    def _launch_local(self, name: str, model_label: str, model_command: str, config: str) -> None:
        job = self._new_job(name, model_label, "Local", str(Path(config).expanduser().resolve()))
        process = QProcess(self); process.setWorkingDirectory(str(PROJECT_ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        arguments = ["-m", "chemflow.cli.main", "train", model_command, job.config]
        if model_command == "gpt": arguments.extend(["--seed", str(self.seed.value())])
        job.log = "$ " + shlex.join([sys.executable, *arguments]) + "\n"; job.process = process
        process.started.connect(lambda key=job.key: self._local_started(key))
        process.readyReadStandardOutput.connect(lambda key=job.key: self._read_process(key))
        process.finished.connect(lambda code, _status, key=job.key: self._local_finished(key, int(code)))
        process.errorOccurred.connect(lambda _error, key=job.key: self._process_error(key))
        process.start(sys.executable, arguments)

    def _launch_slurm(self, name: str, model_label: str, model_command: str) -> None:
        host = self.ssh_host.text().strip(); config = self.remote_config.text().strip()
        job = self._new_job(name, model_label, "Slurm HPC", config); job.host = host; job.status = "Submitting"
        train_args = ["chemflow", "train", model_command, config]
        if model_command == "gpt": train_args.extend(["--seed", str(self.seed.value())])
        initialization = self.conda_init.text().strip()
        init_command = f"source {self._remote_path(initialization)} && " if initialization else ""
        run_command = (
            f"{init_command}conda activate {shlex.quote(self.conda_env.text().strip())} && "
            f"cd {shlex.quote(self.remote_repo.text().strip())} && srun {shlex.join(train_args)}"
        )
        sbatch = ["sbatch", "--parsable", f"--job-name={name}", f"--partition={self.partition.text().strip()}",
                  f"--cpus-per-task={self.cpus.value()}", f"--mem={self.memory.text().strip()}",
                  f"--time={self.walltime.text().strip()}", "--output=chemflow-%j.out"]
        if self.gpus.value() > 0: sbatch.append(f"--gres=gpu:{self.gpus.value()}")
        if self.account.text().strip(): sbatch.append(f"--account={self.account.text().strip()}")
        sbatch.extend(["--wrap", run_command])
        remote_command = f"cd {shlex.quote(self.remote_repo.text().strip())} && {shlex.join(sbatch)}"
        job.log = f"$ ssh {shlex.quote(host)} {shlex.quote(remote_command)}\n"
        process = QProcess(self); process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels); job.process = process
        process.readyReadStandardOutput.connect(lambda key=job.key: self._read_process(key))
        process.finished.connect(lambda code, _status, key=job.key: self._submission_finished(key, int(code)))
        process.errorOccurred.connect(lambda _error, key=job.key: self._process_error(key))
        process.start("ssh", self._ssh_arguments(host, remote_command)); self._render_jobs()

    @staticmethod
    def _remote_path(value: str) -> str:
        if value == "$HOME": return '"$HOME"'
        if value.startswith("$HOME/"): return '"$HOME"/' + shlex.quote(value[6:])
        return shlex.quote(value)

    @staticmethod
    def _ssh_arguments(host: str, command: str) -> list[str]:
        return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command]

    def _job(self, key: str) -> TrainingJob | None:
        return next((job for job in self.jobs if job.key == key), None)

    def _local_started(self, key: str) -> None:
        job = self._job(key)
        if job and job.process:
            job.status = "Running"; job.external_id = str(job.process.processId()); self._changed(job)

    def _read_process(self, key: str) -> None:
        job = self._job(key)
        if not job or not job.process: return
        text = bytes(job.process.readAllStandardOutput()).decode("utf-8", errors="replace"); job.log += text
        if self._selected_job() is job:
            self.output.moveCursor(QTextCursor.MoveOperation.End); self.output.insertPlainText(text); self.output.ensureCursorVisible()

    def _local_finished(self, key: str, exit_code: int) -> None:
        job = self._job(key)
        if not job: return
        job.status = "Completed" if exit_code == 0 else f"Failed ({exit_code})"; job.process = None; self._changed(job)

    def _process_error(self, key: str) -> None:
        job = self._job(key)
        if not job or not job.process: return
        job.log += f"\nProcess error: {job.process.errorString()}\n"; job.status = "Could not start"; job.process = None; self._changed(job)

    def _submission_finished(self, key: str, exit_code: int) -> None:
        job = self._job(key)
        if not job: return
        job.process = None
        if exit_code != 0: job.status = f"Submission failed ({exit_code})"; self._changed(job); return
        lines = [line.strip() for line in job.log.splitlines() if line.strip()]
        candidate = lines[-1].split(";", 1)[0] if lines else ""
        if not candidate.isdigit(): job.status = "Submission response unclear"; self._changed(job); return
        job.external_id = candidate
        remote_repo = self.remote_repo.text().strip().rstrip("/")
        job.remote_log = f"{remote_repo}/chemflow-{candidate}.out"
        job.status = "Pending"; self._changed(job); self.refresh_jobs()

    def refresh_jobs(self) -> None:
        queryable = ACTIVE_SLURM_STATES | {"SUBMITTING", "UNKNOWN"}
        for job in self.jobs:
            if job.backend != "Slurm HPC" or not job.external_id or job.status.upper() not in queryable: continue
            command = (f"state=$(squeue -h -j {shlex.quote(job.external_id)} -o %T | head -n 1); "
                       f"if [ -z \"$state\" ]; then sacct -n -j {shlex.quote(job.external_id)} --format=State -X | head -n 1; "
                       "else printf '%s\\n' \"$state\"; fi; "
                       "printf '%s\\n' __CHEMFLOW_LOG__; "
                       f"tail -n 250 {shlex.quote(job.remote_log)} 2>/dev/null || true")
            process = QProcess(self); process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            process.finished.connect(lambda code, _status, p=process, key=job.key: self._poll_finished(key, p, int(code)))
            self.auxiliary_processes.append(process)
            process.start("ssh", self._ssh_arguments(job.host, command))

    def _poll_finished(self, key: str, process: QProcess, exit_code: int) -> None:
        if process in self.auxiliary_processes: self.auxiliary_processes.remove(process)
        job = self._job(key)
        response = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
        process.deleteLater()
        if not job: return
        state_text, marker, remote_output = response.partition("__CHEMFLOW_LOG__")
        state_text = state_text.strip()
        job.status = "Unknown" if exit_code != 0 or not state_text else state_text.splitlines()[0].strip().split()[0].title()
        if marker:
            job.remote_output = remote_output.lstrip("\r\n")
        self._changed(job)

    def cancel_selected(self) -> None:
        job = self._selected_job()
        if not job: self._warn("Select a job first."); return
        if job.backend == "Local":
            if job.process and job.process.state() != QProcess.ProcessState.NotRunning:
                job.status = "Stopping"; job.process.terminate()
                QTimer.singleShot(5000, lambda p=job.process: p.kill() if p and p.state() != QProcess.ProcessState.NotRunning else None)
                self._changed(job)
            return
        if not job.external_id: self._warn("This Slurm job has no job ID to cancel."); return
        process = QProcess(self); process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.finished.connect(lambda code, _status, p=process, key=job.key: self._cancel_finished(key, p, int(code)))
        self.auxiliary_processes.append(process)
        process.start("ssh", self._ssh_arguments(job.host, f"scancel {shlex.quote(job.external_id)}"))

    def _cancel_finished(self, key: str, process: QProcess, exit_code: int) -> None:
        if process in self.auxiliary_processes: self.auxiliary_processes.remove(process)
        job = self._job(key); response = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace"); process.deleteLater()
        if job:
            job.log += response; job.status = "Cancelled" if exit_code == 0 else "Cancellation failed"; self._changed(job)

    def remove_finished(self) -> None:
        active = {"Queued", "Running", "Submitting", "Pending", "Stopping", "Unknown"}
        self.jobs = [job for job in self.jobs if job.status in active]; self._render_jobs(); self._persist_jobs()

    def _changed(self, job: TrainingJob) -> None:
        self._render_jobs(selected_key=job.key); self._persist_jobs()

    def _render_jobs(self, selected_key: str | None = None) -> None:
        if selected_key is None:
            selected = self._selected_job(); selected_key = selected.key if selected else None
        self.table.setRowCount(len(self.jobs)); selected_row = -1
        for row, job in enumerate(self.jobs):
            values = [job.name, job.model, job.backend, job.status, job.external_id or "—", job.started.replace("T", " "), job.config]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value); item.setData(Qt.ItemDataRole.UserRole, job.key); self.table.setItem(row, column, item)
            if job.key == selected_key: selected_row = row
        self.table.resizeColumnsToContents()
        if selected_row >= 0: self.table.selectRow(selected_row)

    def _selected_job(self) -> TrainingJob | None:
        row = self.table.currentRow()
        if row < 0 or not self.table.item(row, 0): return None
        return self._job(str(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)))

    def _show_selected_log(self) -> None:
        job = self._selected_job()
        text = job.log if job else ""
        if job and job.remote_output:
            text += "\n--- Slurm output ---\n" + job.remote_output
        self.output.setPlainText(text)
        self.output.moveCursor(QTextCursor.MoveOperation.End)

    def _warn(self, message: str) -> None:
        QMessageBox.warning(self, "Training center", message)
