"""Non-blocking ChemFlow CLI process console."""

from __future__ import annotations

import shlex
import sys

from chemflow.config import PROJECT_ROOT

from PySide6.QtCore import QProcess, QProcessEnvironment, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


class ProcessConsole(QWidget):
    state_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
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
        self.output.setMaximumBlockCount(100000)
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

    def set_working_directory(self, path: str) -> None:
        """Run subsequent tasks in ``path`` so their outputs land there."""
        self.process.setWorkingDirectory(path)

    def append_line(self, text: str) -> None:
        """Append output produced by an in-process task (e.g. the linker designer)."""
        self.output.moveCursor(QTextCursor.MoveOperation.End)
        self.output.appendPlainText(text)
        self.output.ensureCursorVisible()

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def run(self, arguments: list[str]) -> None:
        self.run_module("chemflow.cli.main", arguments)

    def run_alchemforge(self, arguments: list[str]) -> None:
        """Run ``alchemforge.cli`` through the free-energy env's interpreter.

        OpenFE/AlchemForge usually live in their own conda env; this resolves that
        interpreter (env var CHEMFLOW_FE_PYTHON / a dedicated conda env, else the
        current one) so same-env and linked-env installs both work.
        """
        from chemflow.desktop.fe_env import find_fe_python

        self.run_module("alchemforge.cli", arguments, python=find_fe_python())

    def run_chemflow_fe(self, arguments: list[str]) -> None:
        """Run ``chemflow.cli.main`` through the free-energy (OpenFE) env.

        The OpenFE alchemical workflows (``chemflow openfe`` / ``chemflow network``)
        were migrated into chemflow.fe.openfe but still need the OpenFE/RDKit/OpenMM
        stack, which lives in the dedicated conda env — so run them through that
        interpreter, not the desktop's.
        """
        from chemflow.desktop.fe_env import find_fe_python

        self.run_module("chemflow.cli.main", arguments, python=find_fe_python())

    def run_chemflow_gmx(self, arguments: list[str]) -> None:
        """Run ``chemflow.cli.main`` through the GROMACS/PMX env.

        The GROMACS workflows (``chemflow ligand-rbfe`` / ``rbfe`` …) need the
        GROMACS toolchain (Open Babel, ACPYPE, AmberTools, GROMACS, PMX), kept in
        its own conda env (gmxfe) separate from the OpenFE env — so run them there.
        """
        from chemflow.desktop.gmx_env import find_gmx_python

        self.run_module("chemflow.cli.main", arguments, python=find_gmx_python())

    def run_rosetta(self, arguments: list[str]) -> None:
        """Run a chemflow command through the PyRosetta env's interpreter.

        Used for the PROTAC commands that import pyrosetta (pyrosetta-dock, KIC
        fix-structure); resolves the dedicated env or falls back to the current one.
        """
        from chemflow.desktop.rosetta_env import find_rosetta_python

        self.run_module("chemflow.cli.main", arguments, python=find_rosetta_python())

    def run_alchemfold(self, arguments: list[str], *, cwd: str | None = None) -> None:
        """Run the ``alchemfold`` CLI (docking / screening).

        AlchemFold runs in whichever env can import it — normally ``chemflow``,
        since the shared ``alchemfold`` env is retired. The ENGINES live in
        their own envs (``boltz``, ``chai``, ``diffdock``) and are resolved per
        run from the config. ``cwd`` sets the working directory so outputs land
        next to the config.
        """
        from chemflow.desktop.alchemfold_env import find_alchemfold_python

        self.run_module("alchemfold", arguments, python=find_alchemfold_python(), cwd=cwd)

    def run_module(self, module: str, arguments: list[str], *, python: str | None = None,
                   cwd: str | None = None) -> None:
        """Run ``python -m <module> <arguments>`` (defaults to this interpreter)."""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            raise RuntimeError("A ChemFlow task is already running.")
        interpreter = python or sys.executable
        command = ["-m", module, *arguments]
        if cwd:
            self.process.setWorkingDirectory(cwd)
        self.output.appendPlainText("\n$ " + shlex.join([interpreter, *command]))
        self.status.setText("Running")
        self.cancel.setEnabled(True)
        self.state_changed.emit(True)
        self.process.start(interpreter, command)

    def run_program(self, program: str, arguments: list[str], *, cwd: str | None = None) -> None:
        """Run an arbitrary external program (e.g. ``sbatch``), optionally in ``cwd``."""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            raise RuntimeError("A ChemFlow task is already running.")
        if cwd:
            self.process.setWorkingDirectory(cwd)
        self.output.appendPlainText("\n$ " + shlex.join([program, *arguments]))
        self.status.setText("Running")
        self.cancel.setEnabled(True)
        self.state_changed.emit(True)
        self.process.start(program, arguments)

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
