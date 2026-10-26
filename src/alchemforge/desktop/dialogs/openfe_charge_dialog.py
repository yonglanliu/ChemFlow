"""Desktop dialog for assigning ligand partial charges with OpenFE."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView
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

from alchemforge.desktop.modeless import show_modeless
from alchemforge.openfe.charge import (
    DEFAULT_CHARGE_SETTINGS,
    ChargeMoleculesConfig,
    stage,
)
from alchemforge.openfe.charge_qc import render_charge_qc_html


class ChargeQCDialog(QDialog):
    def __init__(self, report_html: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenFE Partial-Charge QC")
        self.resize(1050, 820)
        layout = QVBoxLayout(self)
        view = QWebEngineView()
        view.setHtml(report_html)
        layout.addWidget(view)


class OpenFEChargeMoleculesDialog(QDialog):
    job_requested = pyqtSignal(str, str, str)

    def __init__(
        self,
        working_directory: str | None = None,
        parent=None,
        *,
        embedded: bool = False,
    ):
        super().__init__(parent)
        self.embedded = embedded
        if embedded:
            self.setWindowFlags(Qt.Widget)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.setWindowTitle("Charge Ligands Using OpenFE")
        self.resize(760, 360)
        root = QVBoxLayout(self)
        title = QLabel("Assign Partial Charges with OpenFE")
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #123f76;")
        description = QLabel(
            "Run openfe charge-molecules on every molecule in an SDF. "
            "The desktop remains responsive and reports completion in Jobs."
        )
        description.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(description)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.input_sdf = QLineEdit()
        self.output_sdf = QLineEdit()
        self.settings_yaml = QLineEdit(str(DEFAULT_CHARGE_SETTINGS))
        self.openfe_command = QLineEdit("openfe")
        self.n_cores = QSpinBox()
        self.n_cores.setRange(1, 256)
        self.n_cores.setValue(4)
        self.overwrite = QCheckBox("Replace charges already stored in the input SDF")
        self.overwrite.setChecked(True)
        form.addRow("Input SDF:", self._input_row())
        form.addRow("Output SDF:", self._output_row())
        form.addRow("Charge settings YAML:", self._settings_row())
        form.addRow("OpenFE command:", self.openfe_command)
        form.addRow("Processes:", self.n_cores)
        form.addRow("", self.overwrite)
        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        review = QPushButton("Review Charged SDF")
        review.setToolTip(
            "Compare the input and charged SDF, including summed partial charges"
        )
        review.clicked.connect(self.review_charges)
        run = QPushButton("Charge Molecules")
        run.clicked.connect(self.create_job)
        buttons.addWidget(review)
        buttons.addWidget(run)
        root.addLayout(buttons)

    @staticmethod
    def _row(field: QLineEdit, button: QPushButton) -> QWidget:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        layout.addWidget(button)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _input_row(self) -> QWidget:
        button = QPushButton("Browse...")
        button.clicked.connect(self._select_input)
        return self._row(self.input_sdf, button)

    def _output_row(self) -> QWidget:
        button = QPushButton("Browse...")
        button.clicked.connect(self._select_output)
        return self._row(self.output_sdf, button)

    def _settings_row(self) -> QWidget:
        button = QPushButton("Browse...")
        button.clicked.connect(self._select_settings)
        return self._row(self.settings_yaml, button)

    def _select_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select input SDF", self.working_directory, "SDF (*.sdf)"
        )
        if path:
            self.input_sdf.setText(path)
            if not self.output_sdf.text().strip():
                source = Path(path)
                self.output_sdf.setText(
                    str(source.with_name(f"{source.stem}_charged.sdf"))
                )

    def _select_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Select charged SDF output", self.working_directory, "SDF (*.sdf)"
        )
        if path:
            self.output_sdf.setText(
                path if path.lower().endswith(".sdf") else path + ".sdf"
            )

    def _select_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OpenFE charge settings",
            self.working_directory,
            "YAML (*.yaml *.yml)",
        )
        if path:
            self.settings_yaml.setText(path)

    def review_charges(self) -> None:
        try:
            original = Path(self.input_sdf.text().strip()).expanduser().resolve()
            charged = Path(self.output_sdf.text().strip()).expanduser().resolve()
            if not original.is_file():
                raise ValueError("Select the original input SDF")
            if not charged.is_file():
                raise ValueError(
                    "The charged output SDF does not exist yet. Run charging first "
                    "or select an existing charged SDF."
                )
            report = render_charge_qc_html(original, charged)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Review Charges", str(error))
            return
        self._qc_dialog = ChargeQCDialog(report, self)
        show_modeless(self, self._qc_dialog)

    def create_job(self) -> None:
        try:
            config = ChargeMoleculesConfig(
                input_sdf=Path(self.input_sdf.text().strip()).expanduser().resolve(),
                output_sdf=Path(self.output_sdf.text().strip()).expanduser().resolve(),
                settings_yaml=Path(self.settings_yaml.text().strip())
                .expanduser()
                .resolve(),
                n_cores=self.n_cores.value(),
                overwrite_charges=self.overwrite.isChecked(),
                openfe_command=self.openfe_command.text().strip(),
            )
            task = (
                Path(self.working_directory) / "openfe_charge" / config.output_sdf.stem
            )
            script, config_path = stage(config, task)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot Charge Molecules", str(error))
            return
        self.job_requested.emit(str(task), str(script), str(config_path))
        if not self.embedded:
            self.accept()
