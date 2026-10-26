"""Desktop launcher for executing generated OpenFE transformations."""

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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from alchemforge.openfe.execute import OpenFEExecutionConfig, stage


class OpenFEExecuteDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(self, working_directory: str | None = None, parent=None):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("Run OpenFE Transformations")
        self.resize(760, 360)
        root = QVBoxLayout(self)
        title = QLabel("Execute OpenFE Transformation JSON Files")
        title.setStyleSheet("font-size:18pt; font-weight:700; color:#123f76;")
        root.addWidget(title)
        note = QLabel(
            "Run every transformation JSON in the selected directory. Each "
            "transformation receives an independent result directory."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.json_directory = QLineEdit()
        self.output_directory = QLineEdit()
        self.openfe_command = QLineEdit("openfe")
        self.max_parallel = QSpinBox()
        self.max_parallel.setRange(1, 256)
        self.max_parallel.setValue(3)
        self.resume = QCheckBox("Resume from existing OpenFE cache when possible")
        self.resume.setChecked(True)
        form.addRow("Transformation JSON directory:", self._directory_row(self.json_directory))
        form.addRow("Results directory:", self._directory_row(self.output_directory))
        form.addRow("OpenFE command:", self.openfe_command)
        form.addRow("Parallel transformations:", self.max_parallel)
        form.addRow("Resume:", self.resume)
        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        run = QPushButton("Run OpenFE Calculations")
        run.clicked.connect(self.create_job)
        buttons.addWidget(cancel)
        buttons.addWidget(run)
        root.addLayout(buttons)

    def _directory_row(self, field: QLineEdit) -> QWidget:
        button = QPushButton("Browse…")

        def select() -> None:
            path = QFileDialog.getExistingDirectory(
                self, "Select directory", self.working_directory
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

    def create_job(self) -> None:
        try:
            json_directory = Path(self.json_directory.text().strip()).expanduser().resolve()
            output_directory = Path(self.output_directory.text().strip()).expanduser().resolve()
            config = OpenFEExecutionConfig(
                json_directory=json_directory,
                output_directory=output_directory,
                max_parallel=self.max_parallel.value(),
                openfe_command=self.openfe_command.text().strip(),
                resume=self.resume.isChecked(),
            )
            task = output_directory / ".alchemforge_openfe_execution"
            runner, config_path = stage(config, task)
        except (KeyError, OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Run OpenFE", str(error))
            return
        self.job_requested.emit(str(task), str(runner), str(config_path))
        self.accept()
