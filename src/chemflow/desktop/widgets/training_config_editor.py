"""Schema-driven model configuration editor for the training center."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.training_parameters import (
    MODEL_PARAMETER_SCHEMAS,
    ModelParameters,
    ParameterSpec,
)
from chemflow.desktop.widgets.forms import PathField


FILE_PARAMETERS = {
    "dataset_path",
    "test_dataset_path",
    "data_file",
    "checkpoint_path",
    "resume_checkpoint",
    "base_checkpoint",
    "transfer_checkpoint",
    "pretrained_path",
}

DIRECTORY_PARAMETERS = {
    "workdir",
    "save_dir",
    "model_name",
    "tokenizer_name",
}


class TrainingConfigurationEditor(QWidget):
    """Render typed controls and help affordances from a model schema."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_label = ""
        self.schema: ModelParameters | None = None
        self.editors: dict[str, tuple[ParameterSpec, QWidget]] = {}
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(10)
        self.description = QLabel()
        self.description.setObjectName("Muted")
        self.description.setWordWrap(True)
        self.toolbox = QTabWidget()
        self.root_layout.addWidget(self.description)
        self.root_layout.addWidget(self.toolbox)

    def set_model(self, model_label: str) -> None:
        self.model_label = model_label
        self.schema = MODEL_PARAMETER_SCHEMAS[model_label]
        self.description.setText(
            f"{self.schema.description} Adjust the canonical parameters below; "
            "hover over or select ? for an explanation."
        )
        while self.toolbox.count():
            widget = self.toolbox.widget(0)
            self.toolbox.removeTab(0)
            widget.deleteLater()
        self.editors.clear()
        sections: OrderedDict[str, list[ParameterSpec]] = OrderedDict()
        for parameter in self.schema.parameters:
            sections.setdefault(parameter.section, []).append(parameter)
        for section, parameters in sections.items():
            page = QWidget()
            form = QFormLayout(page)
            form.setContentsMargins(14, 12, 14, 14)
            form.setHorizontalSpacing(14)
            form.setVerticalSpacing(9)
            for parameter in parameters:
                editor = self._editor(parameter)
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                row_layout.addWidget(editor, 1)
                help_button = QToolButton()
                help_button.setText("?")
                help_button.setObjectName("HelpButton")
                help_button.setFixedSize(28, 28)
                help_button.setToolTip(parameter.help)
                help_button.clicked.connect(
                    lambda _checked=False, item=parameter: QMessageBox.information(
                        self,
                        item.label,
                        f"{item.path}\n\n{item.help}",
                    )
                )
                row_layout.addWidget(help_button)
                label = QLabel(parameter.label)
                label.setToolTip(parameter.path)
                form.addRow(label, row)
                self.editors[parameter.path] = (parameter, editor)
            self.toolbox.addTab(page, section)
        if self.toolbox.count():
            self.toolbox.setCurrentIndex(0)

    def _editor(self, parameter: ParameterSpec) -> QWidget:
        if parameter.key in FILE_PARAMETERS:
            editor = PathField(
                "Browse locally or type a path available on the HPC",
            )
            editor.setText(str(parameter.default))
        elif parameter.key in DIRECTORY_PARAMETERS:
            editor = PathField(
                "Browse locally or type a path available on the HPC",
                mode="directory",
            )
            editor.setText(str(parameter.default))
        elif parameter.kind == "bool":
            editor = QCheckBox("Enabled")
            editor.setChecked(bool(parameter.default))
        elif parameter.kind == "choice":
            editor = QComboBox()
            editor.addItems(parameter.options)
            editor.setCurrentText(str(parameter.default))
        elif parameter.kind == "int":
            editor = QSpinBox()
            editor.setRange(int(parameter.minimum), int(parameter.maximum))
            editor.setValue(int(parameter.default))
        elif parameter.kind == "float":
            editor = QDoubleSpinBox()
            editor.setDecimals(8)
            editor.setRange(float(parameter.minimum), float(parameter.maximum))
            value = float(parameter.default)
            editor.setSingleStep(max(abs(value) / 10.0, 1e-6))
            editor.setValue(value)
        else:
            editor = QLineEdit()
            if parameter.kind == "list":
                editor.setText(", ".join(str(item) for item in parameter.default))
                editor.setPlaceholderText("Comma-separated values")
            else:
                editor.setText(str(parameter.default))
        editor.setToolTip(parameter.help)
        return editor

    def values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for path, (parameter, editor) in self.editors.items():
            if parameter.kind == "bool":
                value = editor.isChecked()
            elif parameter.kind == "choice":
                value = editor.currentText()
            elif parameter.kind == "int":
                value = editor.value()
            elif parameter.kind == "float":
                value = editor.value()
            elif parameter.kind == "list":
                value = [item.strip() for item in editor.text().split(",") if item.strip()]
                if value and all(self._is_number(item) for item in value):
                    value = [self._number(item) for item in value]
            else:
                value = editor.text().strip()
            values[path] = value
        return values

    @staticmethod
    def _is_number(value: str) -> bool:
        try:
            float(value)
        except ValueError:
            return False
        return True

    @staticmethod
    def _number(value: str) -> int | float:
        number = float(value)
        return int(number) if number.is_integer() else number
