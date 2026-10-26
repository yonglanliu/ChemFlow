"""Native deployment page for the bundled LogD and solubility models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.adme_page import (
    export_prediction_csv,
    load_structure_frame,
    parse_named_smiles,
)
from chemflow.desktop.compound_agent import CompoundSelection
from chemflow.desktop.compound_agent_window import CompoundAgentDialog
from chemflow.desktop.physchem_model import (
    PhysChemModel,
    bundled_physchem_checkpoints,
)
from chemflow.desktop.widgets.forms import (
    Card,
    PageHeader,
    PathField,
    action_button,
    field,
)
from chemflow.desktop.widgets.molecule_sketcher import MoleculePreview


class _PhysChemWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        model: PhysChemModel | None,
        model_key: tuple[object, ...],
        smiles: list[str],
        properties: tuple[str, ...],
        batch_size: int,
        num_workers: int,
        mc_dropout_samples: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_key = model_key
        self.smiles = smiles
        self.properties = properties
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mc_dropout_samples = mc_dropout_samples

    @Slot()
    def run(self) -> None:
        try:
            if self.model is None:
                paths = dict(self.model_key[0])
                device = (
                    None if self.model_key[1] == "auto" else str(self.model_key[1])
                )
                self.model = PhysChemModel(
                    paths,
                    device=device,
                    applicability_domain=bool(self.model_key[2]),
                )
            predictions = self.model.predict_frame(
                self.smiles,
                properties=self.properties,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                mc_dropout_samples=self.mc_dropout_samples,
            )
            self.completed.emit((self.model, self.model_key, predictions))
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class PhysChemDeploymentPage(QWidget):
    """Predict LogD plus kinetic and thermodynamic aqueous solubility."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _PhysChemWorker | None = None
        self._model: PhysChemModel | None = None
        self._model_key: tuple[object, ...] | None = None
        self._result: pd.DataFrame | None = None
        self._agent_dialog: CompoundAgentDialog | None = None

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(18)
        layout.addWidget(
            PageHeader(
                "Physicochemical deployment",
                "Predict LogD and solubility",
                "Run the bundled Chemprop models locally. Kinetic and thermodynamic solubility are reported as log10(mol/L) and µM, with optional MC-dropout uncertainty.",
            )
        )
        layout.addWidget(self._property_card())
        layout.addWidget(self._input_card())
        layout.addWidget(self._results_card())
        layout.addStretch()

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _property_card(self) -> Card:
        card = Card(
            "Deployed properties",
            "Models load on the first prediction and remain cached for this session.",
        )
        self.logd = QCheckBox("LogD")
        self.logd.setChecked(True)
        self.solubility = QCheckBox("Kinetic solubility")
        self.solubility.setChecked(True)
        self.thermodynamic_solubility = QCheckBox("Thermodynamic solubility")
        self.thermodynamic_solubility.setChecked(True)
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "mps", "cuda"])
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 4096)
        self.batch_size.setValue(64)
        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 64)
        self.num_workers.setValue(0)
        self.mc_dropout = QCheckBox("MC-dropout prediction and uncertainty")
        self.mc_dropout.setChecked(True)
        self.mc_samples = QSpinBox()
        self.mc_samples.setRange(2, 200)
        self.mc_samples.setValue(30)
        self.mc_dropout.toggled.connect(self.mc_samples.setEnabled)
        self.applicability_domain = QCheckBox("Similarity and OOD analysis")
        self.applicability_domain.setChecked(True)
        grid = QGridLayout()
        grid.addWidget(self.logd, 0, 0)
        grid.addWidget(self.solubility, 0, 1)
        grid.addWidget(self.thermodynamic_solubility, 1, 0, 1, 2)
        grid.addWidget(field("Device", self.device), 2, 0)
        grid.addWidget(field("Batch size", self.batch_size), 2, 1)
        grid.addWidget(field("Data workers", self.num_workers), 3, 0)
        grid.addWidget(self.mc_dropout, 3, 1)
        grid.addWidget(field("MC samples", self.mc_samples), 4, 1)
        grid.addWidget(self.applicability_domain, 4, 0)
        card.body.addLayout(grid)
        return card

    def _input_card(self) -> Card:
        card = Card(
            "Molecular input", "Enter SMILES or load a molecular data file."
        )
        self.input_tabs = QTabWidget()
        smiles_tab = QWidget()
        smiles_layout = QHBoxLayout(smiles_tab)
        smiles_layout.setContentsMargins(0, 10, 0, 0)
        self.smiles_input = QPlainTextEdit()
        self.smiles_input.setPlaceholderText(
            "SMILES followed by an optional molecule name\n"
            "CC(=O)Oc1ccccc1C(=O)O aspirin\nCCO ethanol"
        )
        self.smiles_input.setMinimumHeight(210)
        self.preview = MoleculePreview()
        smiles_layout.addWidget(self.smiles_input, 11)
        smiles_layout.addWidget(self.preview, 9)
        self.input_tabs.addTab(smiles_tab, "SMILES")

        file_tab = QWidget()
        file_layout = QGridLayout(file_tab)
        file_layout.setContentsMargins(0, 10, 0, 0)
        self.input_file = PathField("CSV, TSV, Parquet, SMI, TXT, or SDF")
        self.structure_column = QLineEdit("SMILES")
        file_layout.addWidget(field("Molecule file", self.input_file), 0, 0, 1, 2)
        file_layout.addWidget(field("Structure column", self.structure_column), 1, 0)
        self.input_tabs.addTab(file_tab, "File")
        card.body.addWidget(self.input_tabs)

        controls = QHBoxLayout()
        self.status = QLabel("Ready")
        self.status.setObjectName("Muted")
        self.run_button = action_button("Predict properties")
        controls.addWidget(self.status)
        controls.addStretch()
        controls.addWidget(self.run_button)
        card.body.addLayout(controls)
        self.smiles_input.textChanged.connect(self._update_preview)
        self.run_button.clicked.connect(self._start_prediction)
        return card

    def _results_card(self) -> Card:
        card = Card(
            "Physicochemical predictions",
            "Includes MC uncertainty, nearest-training-compound similarity, and validation-calibrated OOD scoring. Invalid structures are retained with blank predictions.",
        )
        self.table = QTableWidget(0, 0)
        self.table.setMinimumHeight(300)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        card.body.addWidget(self.table)
        controls = QHBoxLayout()
        self.row_summary = QLabel("No predictions yet")
        self.row_summary.setObjectName("Muted")
        self.agent_button = QPushButton("Open selection agent…")
        self.agent_button.setToolTip(
            "Run physicochemical predictions first, then use the agent to select rows."
        )
        self.agent_button.clicked.connect(self._open_selection_agent)
        save_button = QPushButton("Save CSV as…")
        save_button.clicked.connect(self._export_as)
        controls.addWidget(self.row_summary)
        controls.addStretch()
        controls.addWidget(self.agent_button)
        controls.addWidget(save_button)
        card.body.addLayout(controls)
        return card

    def _selected_properties(self) -> tuple[str, ...]:
        selected = []
        if self.logd.isChecked():
            selected.append("LOGD")
        if self.solubility.isChecked():
            selected.append("SOLUBILITY")
        if self.thermodynamic_solubility.isChecked():
            selected.append("THERMODYNAMIC_SOLUBILITY")
        if not selected:
            raise ValueError("Select at least one physicochemical property.")
        return tuple(selected)

    def _input_frame(self) -> tuple[pd.DataFrame, str]:
        if self.input_tabs.currentIndex() == 0:
            return parse_named_smiles(self.smiles_input.toPlainText()), "SMILES"
        if not self.input_file.text():
            raise ValueError("Choose a molecular input file.")
        structure = self.structure_column.text().strip() or "SMILES"
        return load_structure_frame(self.input_file.text(), structure), structure

    def _update_preview(self) -> None:
        first = next(
            (line.strip() for line in self.smiles_input.toPlainText().splitlines() if line.strip()),
            "",
        )
        self.preview.set_smiles(first.split(maxsplit=1)[0] if first else "")

    def _start_prediction(self) -> None:
        if self._thread is not None:
            return
        if self._agent_dialog is not None:
            if self._agent_dialog.is_running:
                QMessageBox.warning(
                    self,
                    "Selection agent is working",
                    "Wait for the agent request to finish before replacing predictions.",
                )
                return
            self._agent_dialog.close()
            self._agent_dialog = None
        try:
            properties = self._selected_properties()
            all_paths = bundled_physchem_checkpoints()
            paths = {name: all_paths[name] for name in properties}
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError("Bundled model is missing: " + ", ".join(missing))
            input_frame, structure_column = self._input_frame()
            smiles = input_frame[structure_column].astype(str).tolist()
            model_key = (
                tuple((name, str(path.resolve())) for name, path in paths.items()),
                self.device.currentText(),
                self.applicability_domain.isChecked(),
            )
            cached = self._model if model_key == self._model_key else None
        except Exception as error:
            QMessageBox.warning(self, "Cannot predict", str(error))
            return

        self._pending_input = input_frame
        self.run_button.setEnabled(False)
        self.status.setText(f"Predicting {len(smiles):,} molecule(s)…")
        self._thread = QThread(self)
        self._worker = _PhysChemWorker(
            cached,
            model_key,
            smiles,
            properties,
            self.batch_size.value(),
            self.num_workers.value(),
            self.mc_samples.value() if self.mc_dropout.isChecked() else 0,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.completed.connect(self._prediction_complete)
        self._worker.failed.connect(self._prediction_failed)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.completed.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_finished)
        self._thread.start()

    @Slot(object)
    def _prediction_complete(self, payload: object) -> None:
        model, model_key, predictions = payload
        self._model = model
        self._model_key = model_key
        self._result = pd.concat(
            [
                self._pending_input.reset_index(drop=True),
                predictions.drop(columns=["SMILES"]).reset_index(drop=True),
            ],
            axis=1,
        )
        self.agent_button.setEnabled(True)
        self._populate_table(self._result)
        prediction_columns = [
            column
            for column in (
                "LogD_prediction",
                "Kinetic_LogS_prediction (log10 mol/L)",
                "Thermodynamic_LogS_prediction (log10 mol/L)",
            )
            if column in self._result
        ]
        valid = int(
            self._result[prediction_columns]
            .apply(pd.to_numeric, errors="coerce")
            .notna()
            .all(axis=1)
            .sum()
        )
        ood = int(
            self._result.get(
                "OOD_flag", pd.Series(False, index=self._result.index)
            ).sum()
        )
        self.status.setText("Prediction complete")
        self.row_summary.setText(
            f"{len(self._result):,} rows · {valid:,} valid predictions · "
            f"{ood:,} OOD"
        )

    @Slot(str)
    def _prediction_failed(self, message: str) -> None:
        self.status.setText("Prediction failed")
        QMessageBox.critical(self, "Physicochemical prediction failed", message)

    @Slot()
    def _thread_finished(self) -> None:
        self.run_button.setEnabled(True)
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None

    def _populate_table(self, frame: pd.DataFrame) -> None:
        display = frame.head(500)
        self.table.clear()
        self.table.setColumnCount(len(display.columns))
        self.table.setHorizontalHeaderLabels([str(value) for value in display.columns])
        self.table.setRowCount(len(display))
        for row_index, row in display.iterrows():
            is_ood = str(row.get("Domain", "")).upper() == "OOD"
            is_selected = bool(row.get("AI_Selected", False))
            for column_index, value in enumerate(row):
                if pd.isna(value):
                    text = "—"
                elif isinstance(value, (float, np.floating)):
                    text = f"{float(value):.5g}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                if is_selected:
                    item.setBackground(QColor("#245f6c"))
                    item.setToolTip("Selected by the Compound Selection Agent")
                elif is_ood:
                    item.setBackground(QColor("#51263a"))
                    item.setToolTip("Outside the validation-calibrated domain")
                self.table.setItem(row_index, column_index, item)
        self.table.resizeColumnsToContents()

    @Slot()
    def _open_selection_agent(self) -> None:
        if self._result is None or self._result.empty:
            QMessageBox.warning(
                self, "No candidates", "Run physicochemical predictions first."
            )
            return
        if self._agent_dialog is not None:
            self._agent_dialog.show()
            self._agent_dialog.raise_()
            self._agent_dialog.activateWindow()
            return
        try:
            dialog = CompoundAgentDialog(self._result.copy(), self.window())
            dialog.selection_applied.connect(self._apply_agent_selection)
            dialog.finished.connect(self._agent_dialog_finished)
            self._agent_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        except Exception as error:
            self._agent_dialog = None
            QMessageBox.critical(
                self,
                "Cannot open Compound Selection Agent",
                f"{type(error).__name__}: {error}",
            )

    @Slot(object)
    def _apply_agent_selection(self, selection: CompoundSelection) -> None:
        if self._result is None:
            return
        selected = np.zeros(len(self._result), dtype=bool)
        valid_row_ids = [
            row_id for row_id in selection.row_ids if 0 <= row_id < len(selected)
        ]
        selected[valid_row_ids] = True
        self._result["AI_Selected"] = selected
        self._result["AI_Selection_Rationale"] = np.where(
            selected, selection.rationale, ""
        )
        self._populate_table(self._result)
        self.row_summary.setText(
            f"{len(self._result):,} rows · {int(selected.sum()):,} selected by agent"
        )

    @Slot(int)
    def _agent_dialog_finished(self, _result: int) -> None:
        self._agent_dialog = None

    @Slot()
    def _export_as(self) -> None:
        if self._result is None:
            QMessageBox.warning(self, "Nothing to export", "Run a prediction first.")
            return
        selected, _ = QFileDialog.getSaveFileName(
            self.window(),
            "Export physicochemical predictions",
            str(Path.home() / "physicochemical_predictions.csv"),
            "CSV (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not selected:
            return
        try:
            output = export_prediction_csv(self._result, selected)
        except Exception as error:
            QMessageBox.critical(self, "Cannot export predictions", str(error))
            return
        self.status.setText(f"Saved CSV: {output}")
