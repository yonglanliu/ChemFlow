#!/usr/bin/env python3
"""
ResFEP_GROMACS analysis dialog for AlchemForge.

Analysis modes
--------------
all          : convergence -> matrix -> dG
convergency  : forward/reverse BAR convergence
matrix       : MBAR overlap matrix/QC
dG           : combine replicate dG values and calculate final DDG

Execution
---------
localhost    : run analysis in a local QProcess
slurm        : write and submit one CPU SLURM job

Expected AlchemForge console commands
-------------------------------------
resfep_convergence PROJECT_DIR
resfep_matrix -input_dir PROJECT_DIR -output_dir PROJECT_DIR/analysis/overlap
resfep_dg PROJECT_DIR

Change the three command constants below if your pyproject.toml uses different
console-script names.
"""

import os
import shlex
from pathlib import Path

from PyQt5.QtCore import QProcess, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
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


CONVERGENCE_COMMAND = "get_resfep_convergence"
MATRIX_COMMAND = "plot_resrbfe_mbar_matrix"
DG_COMMAND = "get_resrbfe_results"


class ResFEPGromacsAnalysisDialog(QDialog):
    """GUI for ResFEP_GROMACS post-processing."""

    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory=None, parent=None):
        super().__init__(parent)

        self.working_directory = (
            str(Path(working_directory).expanduser().resolve())
            if working_directory
            else str(Path.cwd())
        )

        self.process = None
        self._local_script = None

        self.setWindowTitle("ResFEP_GROMACS Analysis")
        self.resize(760, 700)

        self.build_ui()
        self.update_execution_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QLabel("ResFEP_GROMACS Analysis")
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")

        subtitle = QLabel(
            "Run convergence, MBAR overlap matrix, and dG analyses "
            "for a ResFEP_GROMACS project."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #71839a;")

        root.addWidget(title)
        root.addWidget(subtitle)

        # ---------------- Project ----------------
        project_group = QGroupBox("Project")
        project_form = QFormLayout(project_group)
        project_form.setLabelAlignment(Qt.AlignRight)
        project_form.setHorizontalSpacing(14)
        project_form.setVerticalSpacing(10)

        self.project_edit = QLineEdit()
        self.project_edit.setPlaceholderText(
            "Select ResFEP_GROMACS project directory..."
        )

        project_row = QHBoxLayout()
        project_row.setContentsMargins(0, 0, 0, 0)
        project_row.addWidget(self.project_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_project)
        project_row.addWidget(browse_btn)

        project_container = QWidget()
        project_container.setLayout(project_row)

        self.analysis_combo = QComboBox()
        self.analysis_combo.addItem("All", "all")
        self.analysis_combo.addItem("Convergency", "convergency")
        self.analysis_combo.addItem("Matrix", "matrix")
        self.analysis_combo.addItem("dG", "dG")

        project_form.addRow("Project directory:", project_container)
        project_form.addRow("Analysis:", self.analysis_combo)

        root.addWidget(project_group)

        # ---------------- Execution ----------------
        execution_group = QGroupBox("Execution")
        execution_form = QFormLayout(execution_group)
        execution_form.setLabelAlignment(Qt.AlignRight)
        execution_form.setHorizontalSpacing(14)
        execution_form.setVerticalSpacing(10)

        self.host_combo = QComboBox()
        self.host_combo.addItem("Localhost", "localhost")
        self.host_combo.addItem("SLURM / Other CPUs", "slurm")
        self.host_combo.currentIndexChanged.connect(self.update_execution_ui)

        self.partition_edit = QLineEdit("norm")
        self.partition_edit.setPlaceholderText("e.g. norm")

        self.cpus_spin = QSpinBox()
        self.cpus_spin.setRange(1, 512)
        self.cpus_spin.setValue(4)

        self.memory_edit = QLineEdit("8G")
        self.walltime_edit = QLineEdit("02:00:00")

        self.job_name_edit = QLineEdit("resfep_analysis")

        execution_form.addRow("Run on:", self.host_combo)
        execution_form.addRow("Job name:", self.job_name_edit)
        execution_form.addRow("SLURM partition:", self.partition_edit)
        execution_form.addRow("CPUs per task:", self.cpus_spin)
        execution_form.addRow("Memory:", self.memory_edit)
        execution_form.addRow("Walltime:", self.walltime_edit)

        self.slurm_widgets = [
            self.partition_edit,
            self.cpus_spin,
            self.memory_edit,
            self.walltime_edit,
        ]

        root.addWidget(execution_group)

        # ---------------- Preview ----------------
        preview_group = QGroupBox("Commands")
        preview_layout = QVBoxLayout(preview_group)

        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumHeight(125)
        self.command_preview.setFont(QFont("Monospace"))
        preview_layout.addWidget(self.command_preview)

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self.update_command_preview)
        preview_layout.addWidget(preview_btn, alignment=Qt.AlignRight)

        root.addWidget(preview_group)

        # ---------------- Output ----------------
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Monospace"))
        output_layout.addWidget(self.output_text)

        root.addWidget(output_group, 1)

        # ---------------- Buttons ----------------
        buttons = QHBoxLayout()

        self.clear_btn = QPushButton("Clear Output")
        self.clear_btn.clicked.connect(self.output_text.clear)

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setStyleSheet(
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
        self.run_btn.clicked.connect(self.run_analysis)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)

        buttons.addWidget(self.clear_btn)
        buttons.addStretch()
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.close_btn)

        root.addLayout(buttons)

        # Update preview whenever the main choices change.
        self.analysis_combo.currentIndexChanged.connect(self.update_command_preview)
        self.host_combo.currentIndexChanged.connect(self.update_command_preview)
        self.project_edit.textChanged.connect(self.update_command_preview)

        self.update_command_preview()

    def update_execution_ui(self):
        use_slurm = self.host_combo.currentData() == "slurm"
        for widget in getattr(self, "slurm_widgets", []):
            widget.setEnabled(use_slurm)

        if hasattr(self, "run_btn"):
            self.run_btn.setText("Submit Job" if use_slurm else "Run Analysis")

    def browse_project(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select ResFEP_GROMACS Project",
            self.project_edit.text().strip() or self.working_directory,
        )
        if directory:
            self.project_edit.setText(directory)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def get_project_dir(self):
        text = self.project_edit.text().strip()
        if not text:
            raise ValueError("Please select a project directory.")

        project_dir = Path(text).expanduser().resolve()
        if not project_dir.is_dir():
            raise ValueError(f"Project directory does not exist:\n{project_dir}")

        # ResFEP projects normally contain apo and bound.
        missing = [
            name for name in ("apo", "bound") if not (project_dir / name).is_dir()
        ]
        if missing:
            raise ValueError(
                "This does not look like a ResFEP_GROMACS project.\n\n"
                "Missing director"
                + ("y: " if len(missing) == 1 else "ies: ")
                + ", ".join(missing)
            )

        return project_dir

    def build_analysis_commands(self, project_dir):
        """
        Return shell commands in dependency-safe order.

        dG consumes the convergence CSV files, so 'all' intentionally runs
        convergence before dG.
        """
        mode = self.analysis_combo.currentData()

        convergence = (
            f"{shlex.quote(CONVERGENCE_COMMAND)} {shlex.quote(str(project_dir))}"
        )

        overlap_dir = project_dir / "analysis" / "overlap"
        matrix = (
            f"{shlex.quote(MATRIX_COMMAND)} "
            f"-input_dir {shlex.quote(str(project_dir))} "
            f"-output_dir {shlex.quote(str(overlap_dir))}"
        )

        dg = f"{shlex.quote(DG_COMMAND)} {shlex.quote(str(project_dir))}"

        if mode == "all":
            return [
                ("Convergency", convergence),
                ("Matrix", matrix),
                ("dG", dg),
            ]
        if mode == "convergency":
            return [("Convergency", convergence)]
        if mode == "matrix":
            return [("Matrix", matrix)]
        if mode == "dG":
            return [("dG", dg)]

        raise ValueError(f"Unknown analysis mode: {mode}")

    def build_shell_script(self, project_dir):
        commands = self.build_analysis_commands(project_dir)

        lines = [
            "set -euo pipefail",
            "",
            f"PROJECT_DIR={shlex.quote(str(project_dir))}",
            'mkdir -p "${PROJECT_DIR}/analysis"',
            "",
        ]

        for label, command in commands:
            lines.extend(
                [
                    'echo "============================================================"',
                    f'echo "Running {label} analysis"',
                    'echo "============================================================"',
                    command,
                    "",
                ]
            )

        lines.append('echo "All requested analyses completed."')
        return "\n".join(lines) + "\n"

    def update_command_preview(self):
        if not hasattr(self, "command_preview"):
            return

        project_text = self.project_edit.text().strip()
        if not project_text:
            self.command_preview.setPlainText(
                "Select a ResFEP_GROMACS project directory."
            )
            return

        project_dir = Path(project_text).expanduser()
        try:
            commands = self.build_analysis_commands(project_dir)
        except Exception as exc:
            self.command_preview.setPlainText(str(exc))
            return

        text = "\n".join(command for _, command in commands)
        self.command_preview.setPlainText(text)

    # ------------------------------------------------------------------
    # Run / submit
    # ------------------------------------------------------------------

    def run_analysis(self):
        try:
            project_dir = self.get_project_dir()
            script = self.build_shell_script(project_dir)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Analysis Setup", str(exc))
            return

        if self.host_combo.currentData() == "localhost":
            self.run_local(script, project_dir)
        else:
            self.submit_slurm(script, project_dir)

    def run_local(self, script, project_dir):
        if self.process is not None and (self.process.state() != QProcess.NotRunning):
            QMessageBox.information(
                self,
                "Analysis Running",
                "An analysis process is already running.",
            )
            return

        self.output_text.clear()
        self.append_output(f"Running locally\nProject: {project_dir}\n\n")

        self.run_btn.setEnabled(False)

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(project_dir))
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        # A login shell is useful on HPC systems because it inherits the
        # user's normal shell initialization and the active AlchemForge PATH.
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.finished.connect(self.local_finished)

        self.process.start("bash", ["-lc", script])

        if not self.process.waitForStarted(5000):
            self.run_btn.setEnabled(True)
            QMessageBox.critical(
                self,
                "Launch Failed",
                "Could not start the local analysis process.",
            )

    def read_process_output(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput())
        if data:
            self.append_output(data.decode(errors="replace"))

    def local_finished(self, exit_code, exit_status):
        self.read_process_output()
        self.run_btn.setEnabled(True)

        if exit_code == 0:
            self.append_output("\nAnalysis completed successfully.\n")
        else:
            self.append_output(f"\nAnalysis failed with exit code {exit_code}.\n")

    def submit_slurm(self, analysis_script, project_dir):
        analysis_dir = project_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        job_name = self.job_name_edit.text().strip() or "resfep_analysis"
        partition = self.partition_edit.text().strip()
        memory = self.memory_edit.text().strip()
        walltime = self.walltime_edit.text().strip()
        cpus = self.cpus_spin.value()

        if not partition:
            QMessageBox.warning(
                self, "Missing Partition", "Please enter a SLURM partition."
            )
            return

        if not memory:
            QMessageBox.warning(
                self, "Missing Memory", "Please enter requested memory."
            )
            return

        if not walltime:
            QMessageBox.warning(self, "Missing Walltime", "Please enter a walltime.")
            return

        slurm_file = analysis_dir / "submit_resfep_analysis.sh"
        stdout_file = analysis_dir / f"{job_name}_%j.out"
        stderr_file = analysis_dir / f"{job_name}_%j.err"

        slurm_script = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={walltime}
#SBATCH --output={stdout_file}
#SBATCH --error={stderr_file}

{analysis_script}
"""

        slurm_file.write_text(slurm_script, encoding="utf-8")
        slurm_file.chmod(0o755)

        self.output_text.clear()
        self.append_output(
            "Submitting CPU analysis job\n"
            f"Project: {project_dir}\n"
            f"Script:  {slurm_file}\n\n"
        )

        submit = QProcess(self)
        submit.setWorkingDirectory(str(project_dir))
        submit.setProcessChannelMode(QProcess.MergedChannels)
        submit.finished.connect(
            lambda code, status, p=submit: self.slurm_submit_finished(p, code)
        )
        submit.start("sbatch", [str(slurm_file)])

        if not submit.waitForStarted(5000):
            QMessageBox.critical(
                self,
                "Submission Failed",
                "Could not start sbatch. Make sure SLURM is available.",
            )

    def slurm_submit_finished(self, process, exit_code):
        data = bytes(process.readAllStandardOutput()).decode(errors="replace")
        if data:
            self.append_output(data)

        if exit_code == 0:
            self.append_output("\nSLURM analysis job submitted successfully.\n")
        else:
            self.append_output(f"\nsbatch failed with exit code {exit_code}.\n")

        process.deleteLater()

    def append_output(self, text):
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.output_text.setTextCursor(cursor)
        self.output_text.ensureCursorVisible()

    def closeEvent(self, event):
        if self.process is not None and (self.process.state() != QProcess.NotRunning):
            answer = QMessageBox.question(
                self,
                "Analysis Running",
                "A local analysis is still running. Stop it and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished(3000)

        event.accept()


# Optional standalone test
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = ResFEPGromacsAnalysisDialog()
    dialog.show()
    sys.exit(app.exec_())
