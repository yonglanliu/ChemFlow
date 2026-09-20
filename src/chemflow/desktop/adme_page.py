"""Qt deployment workspace for the CheMeleon clearance ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from rdkit import Chem

from chemflow.deep_learning.chemeleon.adme_model import ADMEModel
from chemflow.desktop.adme_config import (
    find_default_adme_config,
    load_adme_deployment_config,
)
from chemflow.desktop.widgets.forms import Card, PageHeader, PathField, action_button, field
from chemflow.desktop.widgets.ketcher_editor import KetcherEditor
from chemflow.desktop.widgets.molecule_sketcher import MoleculePreview


QUALITY_ROW_COLORS = {
    "OOD + HIGH UNCERTAINTY": "#64243a",
    "OOD": "#51263a",
    "HIGH UNCERTAINTY": "#59451f",
    "NOT ASSESSED": "#3e4652",
}


def parse_named_smiles(text: str) -> pd.DataFrame:
    """Parse ``SMILES [name]`` lines and preserve names as row identifiers."""
    rows: list[dict[str, str]] = []
    has_name = False
    for line_number, line in enumerate(str(text).splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        parts = value.split(maxsplit=1)
        name = parts[1].strip() if len(parts) > 1 else ""
        has_name |= bool(name)
        rows.append(
            {
                "Molecule Name": name or f"row_{line_number}",
                "SMILES": parts[0],
            }
        )
    if not rows:
        raise ValueError("Enter at least one SMILES string.")
    frame = pd.DataFrame(rows)
    return frame if has_name else frame[["SMILES"]]


def load_structure_frame(path: str | Path, structure_column: str) -> pd.DataFrame:
    """Load a molecular table while retaining its identifying columns."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".tsv", ".tab"}:
        frame = pd.read_csv(source, sep="\t")
    elif suffix == ".parquet":
        frame = pd.read_parquet(source)
    elif suffix in {".smi", ".smiles", ".txt"}:
        rows = []
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                parts = value.split(maxsplit=1)
                rows.append(
                    {
                        structure_column: parts[0],
                        "Molecule Name": parts[1] if len(parts) > 1 else f"row_{line_number}",
                    }
                )
        frame = pd.DataFrame(rows)
    elif suffix in {".sdf", ".sd"}:
        rows = []
        for index, molecule in enumerate(Chem.SDMolSupplier(str(source))):
            if molecule is None:
                continue
            name = molecule.GetProp("_Name") if molecule.HasProp("_Name") else f"molecule_{index + 1}"
            row = {
                "Molecule Name": name,
                structure_column: Chem.MolToSmiles(
                    molecule, canonical=True, isomericSmiles=True
                ),
            }
            row.update({key: molecule.GetProp(key) for key in molecule.GetPropNames()})
            rows.append(row)
        frame = pd.DataFrame(rows)
    else:
        raise ValueError(
            "Unsupported input format. Use CSV, TSV, Parquet, SMI, TXT, or SDF."
        )

    if structure_column not in frame.columns:
        raise KeyError(
            f"Structure column {structure_column!r} was not found. "
            f"Available columns: {frame.columns.tolist()}"
        )
    if frame.empty:
        raise ValueError("The selected molecular file contains no usable rows.")
    return frame.reset_index(drop=True)


def export_prediction_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write prediction results to a normalized CSV destination."""
    output = Path(path).expanduser()
    if output.suffix.lower() != ".csv":
        output = output.with_suffix(".csv")
    output = output.resolve()
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Export directory does not exist: {output.parent}")
    frame.to_csv(output, index=False)
    if not output.is_file():
        raise OSError(f"CSV export did not create the requested file: {output}")
    return output


def count_valid_log_predictions(
    frame: pd.DataFrame, selected_properties: tuple[str, ...]
) -> int:
    """Count rows with finite raw log predictions for every selected endpoint."""
    columns = [
        f"Log_{str(property_name).upper()}_CLint_prediction"
        for property_name in selected_properties
    ]
    if not columns or any(column not in frame for column in columns):
        return 0
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    return int(np.isfinite(values).all(axis=1).sum())


def add_prediction_quality_flags(
    frame: pd.DataFrame,
    selected_properties: tuple[str, ...],
    *,
    min_tanimoto_similarity: float,
    max_embedding_distance: float,
    max_log_interval_width: float,
    ood_score_threshold: float = 0.95,
) -> pd.DataFrame:
    """Add concise uncertainty/domain diagnostics and a conservative review flag."""
    result = frame.copy()
    row_count = len(result)
    properties = tuple(str(value).upper() for value in selected_properties)
    prefixes = list(properties)

    ood_values = [
        pd.to_numeric(result[f"{prefix}_ood_score"], errors="coerce").to_numpy()
        for prefix in prefixes
        if f"{prefix}_ood_score" in result
    ]
    similarity_values: list[np.ndarray] = []
    distance_values: list[np.ndarray] = []
    for prefix in prefixes:
        similarity_column = f"{prefix}_train_max_tanimoto"
        distance_column = f"{prefix}_train_embedding_cosine_distance"
        if similarity_column not in result or distance_column not in result:
            continue
        similarity = pd.to_numeric(result[similarity_column], errors="coerce").to_numpy()
        distance = pd.to_numeric(result[distance_column], errors="coerce").to_numpy()
        similarity_values.append(similarity)
        distance_values.append(distance)

    if ood_values:
        ood_matrix = np.column_stack(ood_values)
        finite_ood = np.isfinite(ood_matrix)
        domain_assessed = finite_ood.any(axis=1)
        best_ood_score = np.where(finite_ood, ood_matrix, np.inf).min(axis=1)
        is_ood = domain_assessed & (best_ood_score >= ood_score_threshold)
    elif similarity_values:
        similarity_matrix = np.column_stack(similarity_values)
        distance_matrix = np.column_stack(distance_values)
        finite_pairs = np.isfinite(similarity_matrix) & np.isfinite(distance_matrix)
        domain_assessed = finite_pairs.any(axis=1)
        best_similarity = np.where(finite_pairs, similarity_matrix, -np.inf).max(
            axis=1
        )
        best_distance = np.where(finite_pairs, distance_matrix, np.inf).min(axis=1)
        is_ood = domain_assessed & (
            (best_similarity < min_tanimoto_similarity)
            | (best_distance > max_embedding_distance)
        )
    else:
        domain_assessed = np.zeros(row_count, dtype=bool)
        is_ood = np.zeros(row_count, dtype=bool)

    local_columns = [
        f"{property_name}_local_calibration_used"
        for property_name in properties
        if f"{property_name}_local_calibration_used" in result
    ]
    if local_columns:
        local_calibration_used = result[local_columns].astype(bool).all(axis=1).to_numpy()
        is_ood |= ~local_calibration_used
    else:
        local_calibration_used = np.zeros(row_count, dtype=bool)

    uncertainty_assessed = np.ones(row_count, dtype=bool)
    high_uncertainty = np.zeros(row_count, dtype=bool)
    uncertainty_widths: list[np.ndarray] = []
    for property_name in properties:
        lower_columns = sorted(
            column
            for column in result
            if column.startswith(f"Log_{property_name}_CLint_lower_")
        )
        upper_columns = sorted(
            column
            for column in result
            if column.startswith(f"Log_{property_name}_CLint_upper_")
        )
        if not lower_columns or not upper_columns:
            uncertainty_assessed[:] = False
            continue
        lower = pd.to_numeric(result[lower_columns[0]], errors="coerce").to_numpy()
        upper = pd.to_numeric(result[upper_columns[0]], errors="coerce").to_numpy()
        width = upper - lower
        result[f"{property_name}_uncertainty_log_width"] = width
        finite = np.isfinite(width)
        uncertainty_assessed &= finite
        high_uncertainty |= finite & (width > max_log_interval_width)
        uncertainty_widths.append(width)

    if uncertainty_widths:
        width_matrix = np.column_stack(uncertainty_widths)
        finite_widths = np.isfinite(width_matrix)
        maximum_width = np.where(finite_widths, width_matrix, -np.inf).max(axis=1)
        maximum_width[~finite_widths.any(axis=1)] = np.nan
        result["uncertainty_max_log_width"] = maximum_width
    else:
        result["uncertainty_max_log_width"] = np.nan

    fully_assessed = domain_assessed & uncertainty_assessed
    review_required = is_ood | high_uncertainty | ~fully_assessed
    quality_flag = np.full(row_count, "PASS", dtype=object)
    quality_flag[~fully_assessed] = "NOT ASSESSED"
    quality_flag[high_uncertainty] = "HIGH UNCERTAINTY"
    quality_flag[is_ood] = "OOD"
    quality_flag[is_ood & high_uncertainty] = "OOD + HIGH UNCERTAINTY"

    result["quality_flag"] = quality_flag
    result["review_required"] = review_required
    result["is_ood"] = is_ood
    result["high_uncertainty"] = high_uncertainty
    result["domain_assessed"] = domain_assessed
    result["uncertainty_assessed"] = uncertainty_assessed
    result["local_calibration_used"] = local_calibration_used
    return result


def simplify_adme_results(
    frame: pd.DataFrame,
    input_columns: list[str],
    selected_properties: tuple[str, ...],
) -> pd.DataFrame:
    """Present deployment results without exposing internal checkpoint columns."""
    simplified = frame[input_columns].copy()
    properties = tuple(str(value).upper() for value in selected_properties)
    for property_name in properties:
        calibrated = f"{property_name}_CLint_calibrated (mL/min/kg)"
        raw = f"{property_name}_CLint_prediction (mL/min/kg)"
        if raw in frame:
            simplified[f"{property_name}_pred_raw"] = frame[raw]
        if calibrated in frame:
            simplified[f"{property_name}_pred_calibrated"] = frame[calibrated]

        lower = next(
            (
                column
                for column in frame
                if column.startswith(f"{property_name}_CLint_lower_")
                and column.endswith("(mL/min/kg)")
            ),
            None,
        )
        upper = next(
            (
                column
                for column in frame
                if column.startswith(f"{property_name}_CLint_upper_")
                and column.endswith("(mL/min/kg)")
            ),
            None,
        )
        if lower is not None:
            simplified[f"{property_name}_low"] = frame[lower]
        if upper is not None:
            simplified[f"{property_name}_high"] = frame[upper]
        uncertainty = f"{property_name}_uncertainty_log_width"
        if uncertainty in frame:
            simplified[f"{property_name}_uncertainty_log"] = frame[uncertainty]
        mc_standard_deviation = f"Log_{property_name}_CLint_mc_std"
        if mc_standard_deviation in frame:
            simplified[f"{property_name}_MC_std_log"] = frame[
                mc_standard_deviation
            ]

    prefixes = list(properties)
    for prefix in prefixes:
        similarity = f"{prefix}_train_max_tanimoto"
        distance = f"{prefix}_train_embedding_cosine_distance"
        if similarity in frame:
            simplified[f"{prefix}_FP_sim"] = frame[similarity]
        if distance in frame:
            simplified[f"{prefix}_EB_sim"] = 1.0 - pd.to_numeric(
                frame[distance], errors="coerce"
            )
        ood_score = f"{prefix}_ood_score"
        if ood_score in frame:
            simplified[f"{prefix}_OOD_score"] = frame[ood_score]
        local_count = f"{prefix}_local_calibration_count"
        local_used = f"{prefix}_local_calibration_used"
        if local_count in frame:
            simplified[f"{prefix}_local_n"] = frame[local_count]
        if local_used in frame:
            simplified[f"{prefix}_calibration"] = np.where(
                frame[local_used].astype(bool), "local", "global_fallback"
            )
    similarities = [
        f"{prefix}_train_max_tanimoto"
        for prefix in prefixes
        if f"{prefix}_train_max_tanimoto" in frame
    ]
    distances = [
        f"{prefix}_train_embedding_cosine_distance"
        for prefix in prefixes
        if f"{prefix}_train_embedding_cosine_distance" in frame
    ]
    if similarities:
        simplified["FP_sim_best"] = frame[similarities].max(axis=1, skipna=True)
        simplified["FP_sim_worst"] = frame[similarities].min(axis=1, skipna=True)
    if distances:
        embedding_similarities = 1.0 - frame[distances].apply(
            pd.to_numeric, errors="coerce"
        )
        simplified["EB_sim_best"] = embedding_similarities.max(
            axis=1, skipna=True
        )
        simplified["EB_sim_worst"] = embedding_similarities.min(
            axis=1, skipna=True
        )
    ood_scores = [
        f"{prefix}_ood_score"
        for prefix in prefixes
        if f"{prefix}_ood_score" in frame
    ]
    if ood_scores:
        simplified["OOD_score_best"] = frame[ood_scores].min(axis=1, skipna=True)
        simplified["OOD_score_worst"] = frame[ood_scores].max(axis=1, skipna=True)

    display_names = {
        "quality_flag": "Flag",
        "is_ood": "OOD",
        "high_uncertainty": "High_uncertainty",
    }
    for source, destination in display_names.items():
        if source in frame:
            simplified[destination] = frame[source]
    return simplified


class _PredictionWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        model: ADMEModel | None,
        model_key: tuple[object, ...],
        smiles: list[str],
        properties: tuple[str, ...],
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_key = model_key
        self.smiles = smiles
        self.properties = properties
        self.batch_size = batch_size
        self.num_workers = num_workers

    @Slot()
    def run(self) -> None:
        try:
            if self.model is None:
                device = None if self.model_key[2] == "auto" else self.model_key[2]
                self.model = ADMEModel(
                    multitask_checkpoint=self.model_key[0] or None,
                    mlm_single_checkpoint=self.model_key[1] or None,
                    device=device,
                    applicability_domain=bool(self.model_key[3]),
                    embedding_dimensions=int(self.model_key[4]),
                    calibration_confidence=float(self.model_key[5]),
                    similarity_radius=int(self.model_key[6]),
                    similarity_bits=int(self.model_key[7]),
                    mc_dropout_samples=int(self.model_key[8]),
                )
            predictions = self.model.predict_frame(
                self.smiles,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                properties=self.properties,
            )
            self.completed.emit((self.model, self.model_key, predictions))
        except Exception as error:  # displayed in the GUI with actionable context
            self.failed.emit(f"{type(error).__name__}: {error}")


class ADMEDeploymentPage(QWidget):
    """Deploy two CheMeleon checkpoints as one clearance prediction product."""

    def __init__(self, parent=None, config_path: str | Path | None = None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _PredictionWorker | None = None
        self._model: ADMEModel | None = None
        self._model_key: tuple[object, ...] | None = None
        self._result: pd.DataFrame | None = None
        self.selected_properties: tuple[str, ...] = ("HLM", "RLM", "MLM")
        self.min_tanimoto_similarity = 0.35
        self.max_embedding_distance = 0.35
        self.max_log_interval_width = 1.0
        self.ood_score_threshold = 0.95
        self.mc_dropout_samples = 0

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(18)
        layout.addWidget(
            PageHeader(
                "ADMET deployment",
                "Predict microsomal clearance",
                "Combine multitask HLM/RLM and single-task MLM CheMeleon models, with automatic conversion from log10 values to mL/min/kg.",
            )
        )
        self.selection_banner = QLabel()
        self.selection_banner.setObjectName("SelectionBanner")
        layout.addWidget(self.selection_banner)
        # The model configuration remains live but is intentionally hidden from
        # the prediction workflow. It is populated by the auto-discovered TOML.
        self.model_settings_card = self._model_card()
        self.model_settings_card.hide()
        layout.addWidget(self.model_settings_card)
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
        selected_config = (
            config_path
            if config_path is not None
            else find_default_adme_config()
        )
        if selected_config is not None:
            self.deployment_config.setText(str(selected_config))
            self._apply_deployment_config(show_message=False)

    def _model_card(self) -> Card:
        card = Card(
            "Deployed models",
            "The multitask checkpoint must contain one HLM and one RLM regression target; the single-task checkpoint must contain MLM.",
        )
        self.deployment_config = PathField("Optional ADMET deployment TOML")
        self.load_config_button = QPushButton("Apply config")
        self.load_config_button.clicked.connect(self._apply_deployment_config)
        self.multitask_checkpoint = PathField("Multitask HLM/RLM best.ckpt")
        self.mlm_checkpoint = PathField("Single-task MLM best.ckpt")
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda", "mps"])
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 4096)
        self.batch_size.setValue(64)
        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 64)
        self.num_workers.setValue(0)
        self.applicability_domain = QCheckBox("Calculate calibration and domain")
        self.applicability_domain.setChecked(True)
        self.embedding_dimensions = QSpinBox()
        self.embedding_dimensions.setRange(1, 128)
        self.embedding_dimensions.setValue(128)
        self.calibration_confidence = QDoubleSpinBox()
        self.calibration_confidence.setRange(0.50, 0.99)
        self.calibration_confidence.setDecimals(2)
        self.calibration_confidence.setSingleStep(0.05)
        self.calibration_confidence.setValue(0.90)
        self.similarity_radius = QSpinBox()
        self.similarity_radius.setRange(1, 6)
        self.similarity_radius.setValue(2)
        self.similarity_bits = QComboBox()
        self.similarity_bits.addItems(["512", "1024", "2048", "4096"])
        self.similarity_bits.setCurrentText("2048")
        for control in (
            self.embedding_dimensions,
            self.calibration_confidence,
            self.similarity_radius,
            self.similarity_bits,
        ):
            self.applicability_domain.toggled.connect(control.setEnabled)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(12)
        self.multitask_checkpoint_field = field(
            "Multitask checkpoint", self.multitask_checkpoint
        )
        self.mlm_checkpoint_field = field("MLM checkpoint", self.mlm_checkpoint)
        grid.addWidget(field("Deployment config", self.deployment_config), 0, 0)
        grid.addWidget(
            self.load_config_button, 0, 1, 1, 1, Qt.AlignmentFlag.AlignBottom
        )
        grid.addWidget(self.multitask_checkpoint_field, 1, 0, 1, 2)
        grid.addWidget(self.mlm_checkpoint_field, 2, 0, 1, 2)
        grid.addWidget(field("Device", self.device), 3, 0)
        grid.addWidget(field("Batch size", self.batch_size), 3, 1)
        grid.addWidget(field("Data workers", self.num_workers), 4, 0)
        grid.addWidget(self.applicability_domain, 4, 1)
        grid.addWidget(field("Embedding dimensions", self.embedding_dimensions), 5, 0)
        grid.addWidget(field("Calibration confidence", self.calibration_confidence), 5, 1)
        grid.addWidget(field("Morgan radius", self.similarity_radius), 6, 0)
        grid.addWidget(field("Morgan bits", self.similarity_bits), 6, 1)
        card.body.addLayout(grid)
        self._update_property_state()
        return card

    def _apply_deployment_config(self, _checked=False, *, show_message: bool = True) -> None:
        try:
            if not self.deployment_config.text():
                raise ValueError("Choose an ADMET deployment TOML file.")
            config = load_adme_deployment_config(self.deployment_config.text())
            self.deployment_config.setText(str(config.source))
            self.multitask_checkpoint.setText(config.multitask_checkpoint)
            self.mlm_checkpoint.setText(config.mlm_checkpoint)
            self.device.setCurrentText(config.device)
            self.batch_size.setValue(config.batch_size)
            self.num_workers.setValue(config.num_workers)
            self.applicability_domain.setChecked(config.applicability_domain)
            self.embedding_dimensions.setValue(config.embedding_dimensions)
            self.calibration_confidence.setValue(config.calibration_confidence)
            self.similarity_radius.setValue(config.similarity_radius)
            self.similarity_bits.setCurrentText(str(config.similarity_bits))
            self.mc_dropout_samples = config.mc_dropout_samples
            self.min_tanimoto_similarity = config.min_tanimoto_similarity
            self.max_embedding_distance = config.max_embedding_distance
            self.max_log_interval_width = config.max_log_interval_width
            self.ood_score_threshold = config.ood_score_threshold
        except Exception as error:
            if show_message:
                QMessageBox.warning(self, "Cannot load deployment config", str(error))
                return
            raise
        if show_message:
            self.status.setText(f"Loaded deployment config: {config.source}")

    def set_selected_properties(self, properties: tuple[str, ...]) -> None:
        normalized = tuple(str(value).upper() for value in properties)
        allowed = {"HLM", "RLM", "MLM"}
        if not normalized or not set(normalized).issubset(allowed):
            raise ValueError(f"Unsupported ADMET property selection: {properties}")
        self.selected_properties = normalized
        self._update_property_state()

    def _update_property_state(self) -> None:
        if not hasattr(self, "selection_banner") or not hasattr(
            self, "multitask_checkpoint_field"
        ):
            return
        needs_multitask = bool({"HLM", "RLM"}.intersection(self.selected_properties))
        needs_mlm = "MLM" in self.selected_properties
        self.multitask_checkpoint_field.setEnabled(needs_multitask)
        self.mlm_checkpoint_field.setEnabled(needs_mlm)
        names = " · ".join(self.selected_properties)
        self.selection_banner.setText(f"ACTIVE PREDICTION  ·  {names}")
        if hasattr(self, "status"):
            self.status.setText(
                f"Selected: {', '.join(self.selected_properties)}. Models load on first use."
            )

    def _input_card(self) -> Card:
        card = Card("Molecular input", "Enter SMILES, browse a molecular file, or sketch a structure.")
        self.input_tabs = QTabWidget()

        smiles_tab = QWidget()
        smiles_layout = QHBoxLayout(smiles_tab)
        smiles_layout.setContentsMargins(0, 10, 0, 0)
        self.smiles_input = QPlainTextEdit()
        self.smiles_input.setPlaceholderText(
            "SMILES followed by an optional molecule name\n"
            "CC(=O)Oc1ccccc1C(=O)O aspirin\n"
            "CN1CCC[C@H]1c1cccnc1 nicotine"
        )
        self.smiles_input.setMinimumHeight(220)
        self.preview = MoleculePreview()
        preview_panel = QFrame()
        preview_panel.setObjectName("MoleculePreviewPanel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(10, 8, 10, 10)
        preview_layout.setSpacing(6)
        preview_title = QLabel("STRUCTURE PREVIEW")
        preview_title.setObjectName("MoleculePreviewTitle")
        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.preview, 1)
        smiles_layout.setSpacing(14)
        smiles_layout.addWidget(self.smiles_input, 11)
        smiles_layout.addWidget(preview_panel, 9)
        self.input_tabs.addTab(smiles_tab, "SMILES")

        file_tab = QWidget()
        file_layout = QGridLayout(file_tab)
        file_layout.setContentsMargins(0, 10, 0, 0)
        self.input_file = PathField("CSV, TSV, Parquet, SMI, TXT, or SDF")
        self.structure_column = QLineEdit("SMILES")
        file_layout.addWidget(field("Molecule file", self.input_file), 0, 0, 1, 2)
        file_layout.addWidget(field("Structure column", self.structure_column), 1, 0)
        file_layout.setRowStretch(2, 1)
        self.input_tabs.addTab(file_tab, "Search file")

        self.sketcher = KetcherEditor()
        sketch_tab = QWidget()
        sketch_layout = QVBoxLayout(sketch_tab)
        sketch_layout.setContentsMargins(0, 10, 0, 0)
        sketch_layout.addWidget(self.sketcher)
        self.input_tabs.addTab(sketch_tab, "Ketcher editor")
        card.body.addWidget(self.input_tabs)

        controls = QHBoxLayout()
        self.run_button = action_button("Predict clearance")
        self.status = QLabel("Models load on the first prediction and remain cached.")
        self.status.setObjectName("Muted")
        controls.addWidget(self.status)
        controls.addStretch()
        controls.addWidget(self.run_button)
        card.body.addLayout(controls)

        self.smiles_input.textChanged.connect(self._update_preview)
        self.sketcher.smiles_changed.connect(self.preview.set_smiles)
        self.run_button.clicked.connect(self._start_prediction)
        return card

    def _results_card(self) -> Card:
        card = Card(
            "Clearance predictions",
            "Raw and calibrated predictions and interval bounds use mL/min/kg; "
            "uncertainty is the local (or global fallback) log10 interval width. "
            "FP and embedding similarity show endpoint-specific domain support.",
        )
        self.table = QTableWidget(0, 0)
        self.table.setMinimumHeight(300)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        card.body.addWidget(self.table)
        controls = QHBoxLayout()
        self.row_summary = QLabel("No predictions yet")
        self.row_summary.setObjectName("Muted")
        self.save_as_button = QPushButton("Save CSV as…")
        self.save_as_button.clicked.connect(self._export_as)
        controls.addWidget(self.row_summary)
        controls.addStretch()
        controls.addWidget(self.save_as_button)
        card.body.addLayout(controls)
        return card

    def _update_preview(self) -> None:
        first = next(
            (line.strip() for line in self.smiles_input.toPlainText().splitlines() if line.strip()),
            "",
        )
        self.preview.set_smiles(first.split(maxsplit=1)[0] if first else "")

    def _input_frame(self) -> pd.DataFrame:
        tab = self.input_tabs.currentIndex()
        if tab == 0:
            return parse_named_smiles(self.smiles_input.toPlainText())
        if tab == 1:
            if not self.input_file.text():
                raise ValueError("Choose a molecular input file.")
            return load_structure_frame(
                self.input_file.text(),
                self.structure_column.text().strip() or "SMILES",
            )
        value = self.sketcher.value()
        if not value:
            raise ValueError(
                "Draw a molecule in Ketcher and click “Use structure for prediction”."
            )
        return pd.DataFrame({"SMILES": [value]})

    def _start_prediction(self) -> None:
        if self._thread is not None:
            return
        try:
            needs_multitask = bool(
                {"HLM", "RLM"}.intersection(self.selected_properties)
            )
            needs_mlm = "MLM" in self.selected_properties
            multitask = (
                str(Path(self.multitask_checkpoint.text()).expanduser().resolve())
                if needs_multitask and self.multitask_checkpoint.text()
                else ""
            )
            mlm = (
                str(Path(self.mlm_checkpoint.text()).expanduser().resolve())
                if needs_mlm and self.mlm_checkpoint.text()
                else ""
            )
            if needs_multitask and (not multitask or not Path(multitask).is_file()):
                raise FileNotFoundError("Choose a valid multitask HLM/RLM checkpoint.")
            if needs_mlm and (not mlm or not Path(mlm).is_file()):
                raise FileNotFoundError("Choose a valid single-task MLM checkpoint.")
            input_frame = self._input_frame()
            structure_column = (
                self.structure_column.text().strip() or "SMILES"
                if self.input_tabs.currentIndex() == 1
                else "SMILES"
            )
            smiles = input_frame[structure_column].astype(str).tolist()
            model_key = (
                multitask,
                mlm,
                self.device.currentText(),
                self.applicability_domain.isChecked(),
                self.embedding_dimensions.value(),
                self.calibration_confidence.value(),
                self.similarity_radius.value(),
                int(self.similarity_bits.currentText()),
                self.mc_dropout_samples,
            )
            cached_model = self._model if model_key == self._model_key else None
        except Exception as error:
            QMessageBox.warning(self, "Cannot predict", str(error))
            return

        self._pending_input = input_frame
        self.run_button.setEnabled(False)
        self.status.setText(f"Predicting {len(smiles):,} molecule(s)…")
        self._thread = QThread(self)
        self._worker = _PredictionWorker(
            model=cached_model,
            model_key=model_key,
            smiles=smiles,
            properties=self.selected_properties,
            batch_size=self.batch_size.value(),
            num_workers=self.num_workers.value(),
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
        prediction_columns = predictions.drop(columns=["SMILES"])
        self._result = pd.concat(
            [self._pending_input.reset_index(drop=True), prediction_columns], axis=1
        )
        self._result = add_prediction_quality_flags(
            self._result,
            self.selected_properties,
            min_tanimoto_similarity=self.min_tanimoto_similarity,
            max_embedding_distance=self.max_embedding_distance,
            max_log_interval_width=self.max_log_interval_width,
            ood_score_threshold=self.ood_score_threshold,
        )
        valid = count_valid_log_predictions(
            self._result, self.selected_properties
        )
        input_columns = list(self._pending_input.columns)
        flagged = int(self._result["review_required"].sum())
        self._result = simplify_adme_results(
            self._result, input_columns, self.selected_properties
        )
        self._populate_table(self._result)
        self.status.setText("Prediction complete")
        self.row_summary.setText(
            f"{len(self._result):,} rows · {valid:,} valid for "
            f"{', '.join(self.selected_properties)} · {flagged:,} flagged for review"
        )

    @Slot(str)
    def _prediction_failed(self, message: str) -> None:
        self.status.setText("Prediction failed")
        QMessageBox.critical(self, "ADME prediction failed", message)

    @Slot()
    def _thread_finished(self) -> None:
        self.run_button.setEnabled(True)
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None

    def _populate_table(self, frame: pd.DataFrame) -> None:
        display = frame.head(500)
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(display.columns))
        self.table.setHorizontalHeaderLabels([str(column) for column in display.columns])
        self.table.setRowCount(len(display))
        for row_index, row in display.iterrows():
            flag = str(row.get("Flag", "")).strip().upper()
            row_color = QUALITY_ROW_COLORS.get(flag)
            for column_index, value in enumerate(row):
                if pd.isna(value):
                    text = "—"
                elif isinstance(value, (float, np.floating)):
                    text = f"{float(value):.5g}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                if row_color is not None:
                    item.setBackground(QColor(row_color))
                    item.setToolTip(f"Flagged for review: {flag}")
                self.table.setItem(row_index, column_index, item)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    @Slot()
    def _export_as(self) -> None:
        if self._result is None:
            QMessageBox.warning(self, "Nothing to export", "Run a prediction first.")
            return
        selected, _ = QFileDialog.getSaveFileName(
            self.window(),
            "Export clearance predictions",
            str(Path.home() / "adme_clearance_predictions.csv"),
            "CSV (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not selected:
            self.status.setText("Export cancelled")
            return
        self._save_export(selected)

    def _save_export(self, selected: str | Path) -> None:
        if self._result is None:
            return
        try:
            output = export_prediction_csv(self._result, selected)
        except Exception as error:
            self.status.setText("CSV export failed")
            QMessageBox.critical(
                self,
                "Cannot export predictions",
                f"{type(error).__name__}: {error}",
            )
            return
        self.status.setText(f"Saved CSV: {output}")
        self.row_summary.setText(
            f"Exported {len(self._result):,} rows to {output.name}"
        )
