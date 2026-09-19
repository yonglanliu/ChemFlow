"""Non-blocking ChemFlow CLI process console."""

from __future__ import annotations

import shlex
import sys

from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


class ProcessConsole(QWidget):
    state_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[3]))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)

        self.status = QLabel("Ready")
        self.status.setObjectName("Muted")
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Training and prediction output will appear here…")
        self.output.setMaximumBlockCount(5000)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.output.clear)
        self.cancel = QPushButton("Stop")
        self.cancel.setObjectName("Danger")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.process.terminate)

        bar = QHBoxLayout()
        bar.addWidget(self.status)
        bar.addStretch()
        bar.addWidget(clear)
        bar.addWidget(self.cancel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.output)

    def run(self, arguments: list[str]) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            raise RuntimeError("A ChemFlow task is already running.")
        command = ["-m", "src.cli.main", *arguments]
        self.output.appendPlainText("\n$ " + shlex.join([sys.executable, *command]))
        self.status.setText("Running")
        self.cancel.setEnabled(True)
        self.state_changed.emit(True)
        self.process.start(sys.executable, command)

    def _read_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.insertPlainText(text)
        self.output.ensureCursorVisible()

    def _finished(self, exit_code: int, _status) -> None:
        self.status.setText("Complete" if exit_code == 0 else f"Failed ({exit_code})")
        self.cancel.setEnabled(False)
        self.state_changed.emit(False)

    def _process_error(self, _error) -> None:
        message = self.process.errorString()
        self.output.appendPlainText(f"\nCould not run task: {message}")
        self.status.setText("Could not start")
        self.cancel.setEnabled(False)
        self.state_changed.emit(False)
