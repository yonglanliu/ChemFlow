"""Interactive multi-object py3Dmol viewer for the AlchemForge desktop."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import Qt, QTemporaryDir, QTimer, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineSettings, QWebEngineView
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from alchemforge.structure.ligand_alignment import (
    align_ligand_files,
    ligand_record_names,
)

SUPPORTED_FORMATS = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".sdf": "sdf",
    ".mol": "sdf",
    ".mol2": "mol2",
    ".xyz": "xyz",
    ".gro": "gro",
}

WATER_RESIDUES = ["HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE"]
LIPID_RESIDUES = [
    "POPC",
    "POPE",
    "POPG",
    "POPS",
    "DPPC",
    "DOPC",
    "DOPE",
    "DLPC",
    "DMPC",
    "CHOL",
]
OBJECT_ROLES = [
    "Use default",
    "Protein",
    "Ligand",
    "Membrane",
    "Cofactor",
    "Solvent",
]


@dataclass
class ViewerObject:
    """One independently named and styled structure in the viewer."""

    path: Path
    name: str
    visible: bool = True
    role: str = "Use default"
    structure_data: str | None = None
    structure_format: str | None = None


def guess_object_role(path: str | Path) -> str:
    """Choose a useful initial role from a conventional structure filename."""
    source = Path(path)
    name = source.stem.lower()
    if any(token in name for token in ("protein", "receptor")):
        return "Protein"
    if any(token in name for token in ("membrane", "lipid", "bilayer")):
        return "Membrane"
    if any(token in name for token in ("solvent", "water")):
        return "Solvent"
    if any(token in name for token in ("cofactor", "heme")):
        return "Cofactor"
    if "ligand" in name or source.suffix.lower() in {".sdf", ".mol", ".mol2"}:
        return "Ligand"
    return "Use default"


def _read_structure(
    path: Path,
    structure_data: str | None = None,
    structure_format: str | None = None,
) -> tuple[str, str, int]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Structure file does not exist: {source}")
    file_format = structure_format or SUPPORTED_FORMATS.get(source.suffix.lower())
    if file_format is None:
        raise ValueError(
            "Unsupported structure format. Choose PDB, SDF, MOL, MOL2, XYZ, or GRO."
        )
    structure = (
        structure_data
        if structure_data is not None
        else source.read_text(encoding="utf-8", errors="replace")
    )
    if not structure.strip():
        raise ValueError(f"Structure file is empty: {source}")
    model_count = max(structure.count("$$$$"), 1) if file_format == "sdf" else 1
    return structure, file_format, model_count


def _model_selection(start: int, count: int) -> dict[str, int | list[int]]:
    models = list(range(start, start + count))
    return {"model": models[0] if len(models) == 1 else models}


def _style_object(
    viewer, selection: dict, role: str, default: str, file_format: str
) -> None:
    chosen = default if role == "Use default" else role
    if chosen == "Automatic":
        chosen = "Protein + ligands" if file_format in {"pdb", "gro"} else "Sticks"

    if chosen in {"Protein", "Protein + ligands"}:
        viewer.setStyle(selection, {"line": {"colorscheme": "Jmol"}})
        viewer.setStyle(
            {**selection, "hetflag": False},
            {"cartoon": {"color": "spectrum"}},
        )
        if chosen == "Protein + ligands":
            viewer.setStyle(
                {**selection, "hetflag": True},
                {"stick": {"radius": 0.2, "colorscheme": "Jmol"}},
            )
    elif chosen in {"Ligand", "Cofactor", "Sticks"}:
        viewer.setStyle(selection, {"stick": {"radius": 0.18, "colorscheme": "Jmol"}})
    elif chosen in {"Membrane", "Membrane system"}:
        viewer.setStyle(selection, {"line": {"colorscheme": "Jmol"}})
        if chosen == "Membrane system":
            viewer.setStyle(
                {**selection, "hetflag": False},
                {"cartoon": {"color": "spectrum"}},
            )
            viewer.setStyle(
                {**selection, "resn": LIPID_RESIDUES},
                {"line": {"colorscheme": "Jmol"}},
            )
    elif chosen in {"Solvent", "Lines"}:
        viewer.setStyle(selection, {"line": {"colorscheme": "Jmol"}})
    elif chosen == "Spheres":
        viewer.setStyle(selection, {"sphere": {"scale": 0.3, "colorscheme": "Jmol"}})
    else:
        viewer.setStyle(selection, {"cartoon": {"color": "spectrum"}})


def render_structure_objects_html(
    objects: list[ViewerObject],
    *,
    representation: str = "Automatic",
    background: str = "White",
    show_water: bool = False,
    render_quality: str = "High",
) -> str:
    """Create one py3Dmol scene containing several independently styled files."""
    if not objects:
        raise ValueError("Add at least one structure file")
    try:
        import py3Dmol
    except ImportError as error:
        raise RuntimeError(
            "py3Dmol is not installed. Install requirements-alchemforge.txt "
            "and restart AlchemForge."
        ) from error

    quality_options = {
        "Performance": {"antialias": False, "upscale": False, "cartoonQuality": 5},
        "Balanced": {"antialias": True, "upscale": False, "cartoonQuality": 8},
        "High": {"antialias": True, "upscale": True, "cartoonQuality": 14},
        "Ultra": {"antialias": True, "upscale": True, "cartoonQuality": 24},
    }
    if render_quality not in quality_options:
        raise ValueError("Render quality must be Performance, Balanced, High, or Ultra")
    viewer = py3Dmol.view(width=1000, height=700)
    viewer.setBackgroundColor(background.lower().replace(" ", ""))
    model_index = 0
    selections: list[tuple[ViewerObject, dict]] = []
    for item in objects:
        structure, file_format, model_count = _read_structure(
            item.path, item.structure_data, item.structure_format
        )
        if model_count > 1:
            viewer.addModels(structure, file_format)
        else:
            viewer.addModel(structure, file_format)
        selection = _model_selection(model_index, model_count)
        selections.append((item, selection))
        _style_object(viewer, selection, item.role, representation, file_format)
        model_index += model_count

    if not show_water:
        viewer.setStyle({"resn": WATER_RESIDUES}, {})
    for item, selection in selections:
        if not item.visible:
            viewer.setStyle(selection, {})
    viewer.zoomTo()
    viewer.zoom(0.9)
    fragment = viewer._make_html()
    # py3Dmol's ``options`` argument configures model loading, not GLViewer.
    # Inject documented ViewerSpec values into createViewer itself.
    viewer_options = {
        "backgroundColor": background.lower().replace(" ", ""),
        **quality_options[render_quality],
    }
    fragment = fragment.replace(
        '{backgroundColor:"white"}', json.dumps(viewer_options), 1
    )
    color = background.lower().replace(" ", "")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>html,body{width:100%;height:100%;margin:0;overflow:hidden;}"
        "div[id^='3dmolviewer_']{width:100%!important;height:100%!important;}"
        "</style></head>"
        f"<body style='background:{color};'>{fragment}</body></html>"
    )


def render_structure_html(
    path: str | Path,
    *,
    representation: str = "Automatic",
    background: str = "White",
    show_water: bool = False,
    render_quality: str = "High",
) -> str:
    """Backward-compatible single-structure rendering helper."""
    source = Path(path).expanduser().resolve()
    return render_structure_objects_html(
        [ViewerObject(source, source.stem)],
        representation=representation,
        background=background,
        show_water=show_water,
        render_quality=render_quality,
    )


class StructureViewerDialog(QDialog):
    """Interactively inspect multiple named structure objects with py3Dmol."""

    def __init__(
        self,
        working_directory: str | None = None,
        structure_path: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.objects: list[ViewerObject] = []
        self._alignment_selection: set[tuple[int, int]] = set()
        self._known_alignment_records: set[tuple[int, int]] = set()
        self._alignment_records: dict[int, list[str]] = {}
        self._html_directory = QTemporaryDir("alchemforge_3dmol_XXXXXX")
        if not self._html_directory.isValid():
            raise RuntimeError(
                "Could not create a temporary directory for the 3D viewer"
            )
        self._scene_generation = 0
        self.setWindowTitle("AlchemForge 3D Structure Viewer")
        self.resize(1280, 840)

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        add_files = QPushButton("Add Structure Files…")
        add_files.clicked.connect(self._browse)
        refresh = QPushButton("Refresh Scene")
        refresh.clicked.connect(self.load_structure)
        self.representation = QComboBox()
        self.representation.addItems(
            [
                "Automatic",
                "Protein + ligands",
                "Membrane system",
                "Cartoon",
                "Sticks",
                "Lines",
                "Spheres",
            ]
        )
        self.background = QComboBox()
        self.background.addItems(["White", "Black", "Light gray"])
        self.render_quality = QComboBox()
        self.render_quality.addItems(["Performance", "Balanced", "High", "Ultra"])
        self.render_quality.setCurrentText("High")
        self.render_quality.setToolTip(
            "High and Ultra enable antialiasing, 2× upscaling, and smoother "
            "cartoon geometry. Performance is intended for very large systems."
        )
        self.show_water = QCheckBox("Show water")
        controls.addWidget(add_files)
        controls.addWidget(refresh)
        controls.addSpacing(15)
        controls.addWidget(QLabel("Default style:"))
        controls.addWidget(self.representation)
        controls.addWidget(QLabel("Background:"))
        controls.addWidget(self.background)
        controls.addWidget(QLabel("Quality:"))
        controls.addWidget(self.render_quality)
        controls.addWidget(self.show_water)
        controls.addStretch()
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)
        self.viewer = QWebEngineView()
        self.viewer.settings().setAttribute(
            QWebEngineSettings.LocalContentCanAccessRemoteUrls, True
        )
        self.viewer.settings().setAttribute(
            QWebEngineSettings.LocalContentCanAccessFileUrls, True
        )
        self.viewer.setHtml(
            "<html><body style='font-family:sans-serif;color:#60758d;"
            "display:flex;align-items:center;justify-content:center'>"
            "Click <b>Add Structure Files…</b> to create a scene."
            "</body></html>"
        )
        splitter.addWidget(self.viewer)
        splitter.addWidget(self._build_object_panel())
        splitter.setSizes([930, 350])
        splitter.setStretchFactor(0, 1)
        layout.addWidget(splitter, 1)

        self.status = QLabel(
            "Left-drag to rotate, wheel to zoom, and right-drag to translate."
        )
        self.status.setStyleSheet("color: #71839a;")
        layout.addWidget(self.status)

        self.representation.currentTextChanged.connect(self._reload_if_loaded)
        self.background.currentTextChanged.connect(self._reload_if_loaded)
        self.render_quality.currentTextChanged.connect(self._reload_if_loaded)
        self.show_water.toggled.connect(self._reload_if_loaded)
        if structure_path:
            self._add_paths([structure_path])

    def _build_object_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("RightCard")
        panel.setMinimumWidth(310)
        panel_layout = QVBoxLayout(panel)
        title = QLabel("Scene Objects")
        title.setStyleSheet("font-size: 13pt; font-weight: 700;")
        panel_layout.addWidget(title)
        help_text = QLabel(
            "Rename each object, select its molecular role, or hide it. Choose "
            "any ligand below as the alignment reference."
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: #71839a;")
        panel_layout.addWidget(help_text)

        self.object_table = QTableWidget(0, 3)
        self.object_table.setHorizontalHeaderLabels(["Show", "Object name", "Role"])
        self.object_table.verticalHeader().setVisible(False)
        self.object_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.object_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.object_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.object_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.object_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.object_table.itemChanged.connect(self._object_item_changed)
        panel_layout.addWidget(self.object_table, 1)

        alignment_options = QHBoxLayout()
        alignment_options.addWidget(QLabel("Reference ligand:"))
        alignment_options.addStretch()
        alignment_options.addWidget(QLabel("MCS timeout:"))
        self.mcs_timeout = QSpinBox()
        self.mcs_timeout.setRange(1, 600)
        self.mcs_timeout.setValue(30)
        self.mcs_timeout.setSuffix(" s")
        self.mcs_timeout.setToolTip(
            "Maximum time allowed for each pairwise common-substructure search"
        )
        alignment_options.addWidget(self.mcs_timeout)
        panel_layout.addLayout(alignment_options)
        self.alignment_reference = QComboBox()
        self.alignment_reference.setToolTip(
            "Only checked ligands below are available as the reference"
        )
        panel_layout.addWidget(self.alignment_reference)

        self.sdf_tree_label = QLabel("Ligands to align:")
        self.sdf_tree_label.setToolTip(
            "Check only chemically related ligands; click one to use it as reference"
        )
        panel_layout.addWidget(self.sdf_tree_label)
        self.sdf_tree = QTreeWidget()
        self.sdf_tree.setHeaderHidden(True)
        self.sdf_tree.setMaximumHeight(190)
        self.sdf_tree.itemClicked.connect(self._sdf_molecule_clicked)
        self.sdf_tree.itemChanged.connect(self._alignment_item_changed)
        panel_layout.addWidget(self.sdf_tree)
        self.sdf_tree_label.hide()
        self.sdf_tree.hide()

        buttons = QHBoxLayout()
        align = QPushButton("Align Ligands")
        align.setToolTip(
            "Align ligand objects to the selected reference using an "
            "RDKit maximum-common-substructure atom mapping"
        )
        align.clicked.connect(self._align_ligands)
        reset = QPushButton("Reset Alignment")
        reset.clicked.connect(self._reset_alignment)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_selected)
        clear = QPushButton("Clear All")
        clear.clicked.connect(self._clear_objects)
        buttons.addWidget(align)
        buttons.addWidget(reset)
        panel_layout.addLayout(buttons)

        object_buttons = QHBoxLayout()
        object_buttons.addWidget(remove)
        object_buttons.addWidget(clear)
        panel_layout.addLayout(object_buttons)
        return panel

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Molecular Structures",
            self.working_directory,
            "Molecular structures (*.pdb *.ent *.sdf *.mol *.mol2 *.xyz *.gro);;"
            "All files (*)",
        )
        if paths:
            self.working_directory = str(Path(paths[0]).parent)
            self._add_paths(paths)

    def _add_paths(self, paths: list[str]) -> None:
        for value in paths:
            source = Path(value).expanduser().resolve()
            name = source.stem
            existing = {item.name for item in self.objects}
            if name in existing:
                suffix = 2
                while f"{name}_{suffix}" in existing:
                    suffix += 1
                name = f"{name}_{suffix}"
            self.objects.append(
                ViewerObject(source, name, role=guess_object_role(source))
            )
        self._refresh_object_table()
        self.load_structure()

    def _refresh_object_table(self) -> None:
        self.object_table.blockSignals(True)
        self.object_table.setRowCount(len(self.objects))
        for row, item in enumerate(self.objects):
            visible = QTableWidgetItem()
            visible.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            )
            visible.setCheckState(Qt.Checked if item.visible else Qt.Unchecked)
            visible.setToolTip(str(item.path))
            self.object_table.setItem(row, 0, visible)

            name = QTableWidgetItem(item.name)
            name.setToolTip(str(item.path))
            self.object_table.setItem(row, 1, name)

            role = QComboBox()
            role.addItems(OBJECT_ROLES)
            role.setCurrentText(item.role)
            role.currentTextChanged.connect(
                lambda value, combo=role: self._role_changed(combo, value)
            )
            self.object_table.setCellWidget(row, 2, role)
        self.object_table.blockSignals(False)
        self._refresh_alignment_references()

    def _object_item_changed(self, table_item: QTableWidgetItem) -> None:
        row = table_item.row()
        if not 0 <= row < len(self.objects):
            return
        if table_item.column() == 0:
            self.objects[row].visible = table_item.checkState() == Qt.Checked
            self.load_structure()
        elif table_item.column() == 1:
            name = table_item.text().strip()
            if name:
                self.objects[row].name = name
                self._refresh_alignment_references()
            else:
                table_item.setText(self.objects[row].name)

    def _role_changed(self, combo: QComboBox, value: str) -> None:
        for row in range(self.object_table.rowCount()):
            if self.object_table.cellWidget(row, 2) is combo:
                self.objects[row].role = value
                self._refresh_alignment_references()
                self.load_structure()
                return

    def _refresh_alignment_references(self) -> None:
        previous = self.alignment_reference.currentData()
        records_by_object: dict[int, list[str]] = {}
        for object_index in self._ligand_object_indices():
            item = self.objects[object_index]
            try:
                records = ligand_record_names(item.path)
            except (OSError, RuntimeError, ValueError):
                records = [item.path.stem]
            records_by_object[object_index] = records
        valid_records = {
            (object_index, molecule_index)
            for object_index, records in records_by_object.items()
            for molecule_index in range(len(records))
        }
        new_records = valid_records - self._known_alignment_records
        self._alignment_selection.intersection_update(valid_records)
        self._alignment_selection.update(new_records)
        self._known_alignment_records = valid_records
        self._alignment_records = records_by_object
        self._refresh_alignment_tree()
        self._populate_alignment_references(previous)

    def _populate_alignment_references(self, previous=None) -> None:
        if previous is None:
            previous = self.alignment_reference.currentData()
        self.alignment_reference.blockSignals(True)
        self.alignment_reference.clear()
        for object_index, records in self._alignment_records.items():
            item = self.objects[object_index]
            for molecule_index, record_name in enumerate(records):
                reference = (object_index, molecule_index)
                if reference not in self._alignment_selection:
                    continue
                label = (
                    item.name if len(records) == 1 else f"{item.name}: {record_name}"
                )
                self.alignment_reference.addItem(label, reference)
        if previous is not None:
            restored = self.alignment_reference.findData(previous)
            if restored >= 0:
                self.alignment_reference.setCurrentIndex(restored)
        self.alignment_reference.blockSignals(False)

    def _refresh_alignment_tree(self) -> None:
        self.sdf_tree.blockSignals(True)
        self.sdf_tree.clear()
        for object_index, records in self._alignment_records.items():
            if len(records) == 1:
                item = QTreeWidgetItem(self.sdf_tree, [self.objects[object_index].name])
                item.setData(0, Qt.UserRole, (object_index, 0))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    Qt.Checked
                    if (object_index, 0) in self._alignment_selection
                    else Qt.Unchecked,
                )
                item.setToolTip(0, str(self.objects[object_index].path))
                continue
            parent = QTreeWidgetItem(
                self.sdf_tree,
                [f"{self.objects[object_index].name} ({len(records)} molecules)"],
            )
            parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
            parent.setToolTip(0, str(self.objects[object_index].path))
            parent.setExpanded(False)
            for molecule_index, record_name in enumerate(records):
                child = QTreeWidgetItem(parent, [record_name])
                child.setData(0, Qt.UserRole, (object_index, molecule_index))
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(
                    0,
                    Qt.Checked
                    if (object_index, molecule_index) in self._alignment_selection
                    else Qt.Unchecked,
                )
                child.setToolTip(
                    0, "Check to include in alignment; click to use as reference"
                )
        self.sdf_tree.blockSignals(False)
        visible = bool(self._alignment_records)
        self.sdf_tree_label.setVisible(visible)
        self.sdf_tree.setVisible(visible)

    def _alignment_item_changed(self, _item: QTreeWidgetItem, _column: int) -> None:
        selected: set[tuple[int, int]] = set()
        for top_index in range(self.sdf_tree.topLevelItemCount()):
            top = self.sdf_tree.topLevelItem(top_index)
            candidates = (
                [top.child(index) for index in range(top.childCount())]
                if top.childCount()
                else [top]
            )
            for candidate in candidates:
                reference = candidate.data(0, Qt.UserRole)
                if reference is not None and candidate.checkState(0) == Qt.Checked:
                    selected.add(tuple(reference))
        previous = self.alignment_reference.currentData()
        self._alignment_selection = selected
        self._populate_alignment_references(previous)

    def _sdf_molecule_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        reference = item.data(0, Qt.UserRole)
        if reference is None:
            return
        if item.checkState(0) != Qt.Checked:
            item.setCheckState(0, Qt.Checked)
        index = self.alignment_reference.findData(reference)
        if index >= 0:
            self.alignment_reference.setCurrentIndex(index)

    def _ligand_object_indices(self) -> list[int]:
        return [
            index for index, item in enumerate(self.objects) if item.role == "Ligand"
        ]

    def _align_ligands(self) -> None:
        ligand_indices = self._ligand_object_indices()
        if not ligand_indices:
            QMessageBox.information(
                self,
                "No Ligands to Align",
                "Assign at least one scene object the Ligand role.",
            )
            return
        reference = self.alignment_reference.currentData()
        if reference is None:
            QMessageBox.information(
                self, "Select a Ligand Reference", "Choose a reference ligand."
            )
            return
        reference_index, reference_molecule_index = reference
        reference_position = ligand_indices.index(reference_index)
        selected_by_file = [
            {
                molecule_index
                for object_index, molecule_index in self._alignment_selection
                if object_index == ligand_index
            }
            for ligand_index in ligand_indices
        ]
        try:
            results = align_ligand_files(
                [self.objects[index].path for index in ligand_indices],
                reference_file_index=reference_position,
                reference_molecule_index=reference_molecule_index,
                selected_molecule_indices=selected_by_file,
                timeout_seconds=self.mcs_timeout.value(),
            )
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "Ligand Alignment Failed", str(error))
            return
        rmsds = []
        molecule_count = 0
        for object_index, result in zip(ligand_indices, results, strict=True):
            if not result.aligned_count:
                continue
            item = self.objects[object_index]
            item.structure_data = result.sdf
            item.structure_format = "sdf"
            rmsds.extend(value for value in result.rmsds if math.isfinite(value))
            molecule_count += result.aligned_count
        self.load_structure()
        mean_rmsd = sum(rmsds) / len(rmsds)
        self.status.setText(
            f"Aligned {molecule_count} selected ligand molecules to "
            f"{self.alignment_reference.currentText()!r}; mean mapped-atom RMSD "
            f"{mean_rmsd:.3f} Å. Source files were not changed."
        )

    def _reset_alignment(self) -> None:
        changed = False
        for item in self.objects:
            if item.structure_data is not None:
                item.structure_data = None
                item.structure_format = None
                changed = True
        if changed:
            self.load_structure()
            self.status.setText("Restored the original ligand coordinates")

    def _remove_selected(self) -> None:
        rows = sorted(
            {
                index.row()
                for index in self.object_table.selectionModel().selectedRows()
            },
            reverse=True,
        )
        for row in rows:
            self.objects.pop(row)
        self._alignment_selection.clear()
        self._known_alignment_records.clear()
        self._refresh_object_table()
        self._show_scene_or_welcome()

    def _clear_objects(self) -> None:
        self.objects.clear()
        self._alignment_selection.clear()
        self._known_alignment_records.clear()
        self._refresh_object_table()
        self._show_scene_or_welcome()

    def _show_scene_or_welcome(self) -> None:
        if self.objects:
            self.load_structure()
        else:
            self.viewer.setHtml(
                "<html><body style='font-family:sans-serif;color:#60758d;"
                "display:flex;align-items:center;justify-content:center'>"
                "Click <b>Add Structure Files…</b> to create a scene."
                "</body></html>"
            )
            self.status.setText("No objects loaded")

    def _display_scene_html(self, html: str) -> int:
        """Load from a file so large PDB systems avoid QWebEngine.setHtml limits."""
        self._scene_generation += 1
        generation = self._scene_generation
        destination = Path(self._html_directory.path()) / f"scene_{generation:06d}.html"
        destination.write_text(html, encoding="utf-8")
        self.viewer.load(QUrl.fromLocalFile(str(destination)))
        QTimer.singleShot(5000, lambda: self._check_renderer(generation))
        return generation

    def _check_renderer(self, generation: int) -> None:
        if generation != self._scene_generation or not self.objects:
            return
        script = """
        (() => {
          const canvas = document.createElement('canvas');
          const webgl = !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
          const loaded = typeof $3Dmol !== 'undefined';
          const rendered = document.querySelectorAll("div[id^='3dmolviewer_'] canvas").length > 0;
          return {loaded: loaded, webgl: webgl, rendered: rendered};
        })();
        """
        self.viewer.page().runJavaScript(
            script,
            lambda result: self._report_renderer_status(generation, result),
        )

    def _report_renderer_status(self, generation: int, result) -> None:
        if generation != self._scene_generation or not isinstance(result, dict):
            return
        if not result.get("webgl"):
            self.status.setText(
                "The structure loaded, but WebGL is unavailable in Qt WebEngine. "
                "Hardware acceleration may be disabled."
            )
        elif not result.get("loaded"):
            self.status.setText(
                "The structure loaded, but 3Dmol.js could not be downloaded. "
                "Check the internet connection or firewall access to cdn.jsdelivr.net."
            )
        elif not result.get("rendered"):
            self.status.setText(
                "3Dmol.js loaded but did not create a canvas. Check whether the "
                "selected file contains valid atomic coordinates."
            )

    def _reload_if_loaded(self, *_args) -> None:
        if self.objects:
            self.load_structure()

    def load_structure(self) -> None:
        try:
            html = render_structure_objects_html(
                self.objects,
                representation=self.representation.currentText(),
                background=self.background.currentText(),
                show_water=self.show_water.isChecked(),
                render_quality=self.render_quality.currentText(),
            )
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Display Structure", str(error))
            return
        self._display_scene_html(html)
        visible = sum(item.visible for item in self.objects)
        self.status.setText(
            f"Loaded {len(self.objects)} objects ({visible} visible). "
            "Edit their names and roles in the Scene Objects panel."
        )
