"""ChemFlow Studio page: design a PROTAC linker with a Maestro-style workspace.

Load any number of structures (SMILES, .sdf, .mol, .mol2, .pdb). Each becomes an
entry in a table where you can:

* **Include** (the "In" checkbox) — show/hide it in the 3D window; several can be
  displayed at once (ligands as sticks, proteins as cartoon).
* **Select** (click the row) — mark entries so you can delete them.

To design a linker, click one atom on each of the two warheads you want to join
(the picked atom turns into a coloured sphere); LinkInvent then proposes linkers
scored for length, properties and 3D pocket fit.

The chemistry is the shared backend in
:mod:`chemflow.generative.reinvent.linker`; the scene HTML is the shared
:mod:`chemflow.generative.reinvent.linker.scene_html`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import QFile, QObject, Qt, QTemporaryDir, QThread, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.widgets.forms import (
    NoScrollComboBox,
    NoScrollDoubleSpinBox,
    NoScrollSpinBox,
)
from chemflow.generative.reinvent.linker.scene_html import (
    build_pose_html,
    build_workspace_html,
)

_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

# Role labels shown in the table and the colour used for each warhead's picked atom.
_ROLE_NONE = "—"
_ROLE_A = "Ligand A"
_ROLE_B = "Ligand B"
_ROLE_COLORS = {"A": "0x2ca02c", "B": "0x1f77b4"}   # A green, B blue


class _Entry:
    """One loaded structure in the workspace."""

    __slots__ = ("id", "title", "kind", "mol", "pdb_text", "path", "has_3d", "atom", "included", "role")

    def __init__(self, eid, title, kind, *, mol=None, pdb_text=None, path=None, has_3d=False):
        self.id = eid
        self.title = title
        self.kind = kind            # "ligand" or "protein"
        self.mol = mol              # RDKit Mol (ligands)
        self.pdb_text = pdb_text    # PDB text (proteins)
        self.path = path            # source file (proteins: used for fit geometry)
        self.has_3d = has_3d
        self.atom = None            # picked attachment atom index (ligands)
        self.included = True        # shown in the 3D window
        self.role = None            # "A" or "B" for the two warheads; None otherwise


class _ClickBridge(QObject):
    atom_clicked = Signal(int, int)

    @Slot(int, int)
    def on_atom_clicked(self, entry_id: int, index: int) -> None:
        self.atom_clicked.emit(entry_id, index)


class _DesignWorker(QThread):
    finished_ok = Signal(list, bool)
    failed = Signal(str)
    progressed = Signal(str)

    def __init__(self, head1, head2, opts, models, fit, fit_top, parent=None, *, library=None):
        super().__init__(parent)
        self._args = (head1, head2, opts, models, fit, fit_top)
        self._library = library      # (path, min_length, max_length) for library search, else None

    def run(self):
        head1, head2, opts, models, fit, fit_top = self._args
        log = lambda line: self.progressed.emit(line)
        try:
            if self._library is not None:
                from chemflow.generative.reinvent.linker.library import design_from_library

                path, min_len, max_len = self._library
                candidates = design_from_library(
                    head1, head2, path, models=models, fit=fit, fit_top=fit_top,
                    min_length=min_len, max_length=max_len, log=log,
                )
            else:
                from chemflow.generative.reinvent.linker.designer import design_linkers

                candidates = design_linkers(
                    head1, head2, opts, models=models, fit=fit, fit_top=fit_top, log=log,
                )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(candidates, fit is not None)


class LinkerDesignerPage(QWidget):
    """PROTAC linker designer page for ChemFlow Studio (Maestro-style workspace)."""

    def __init__(self, parent=None, console=None):
        super().__init__(parent)
        self._console = console
        self.working_directory = str(Path(".").resolve())
        self._entries: list[_Entry] = []
        self._next_id = 1
        self._refreshing = False
        self._candidates: list = []
        self._with_fit = False
        self._html_dir = QTemporaryDir("chemflow_linker_XXXXXX")
        if self._html_dir.isValid() and self._html_dir.path():
            self._scene_dir = Path(self._html_dir.path())
        else:
            import tempfile
            self._scene_dir = Path(tempfile.mkdtemp(prefix="chemflow_linker_"))
        self._scene_dir.mkdir(parents=True, exist_ok=True)

        root = QVBoxLayout(self)
        title = QLabel("PROTAC Linker Designer")
        title.setObjectName("PageTitle")
        root.addWidget(title)
        subtitle = QLabel(
            "Load any structures (SMILES, SDF, MOL, MOL2, PDB). Tick “In” to show one "
            "in the 3D view. Assign two of them as Ligand A and Ligand B (the two "
            "warheads) in the Role column, click an attachment atom on each, then "
            "Generate Linkers. Proteins load as background."
        )
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # -- load controls --------------------------------------------------
        top = QHBoxLayout()
        b_file = QPushButton("Load File…")
        b_file.setToolTip("Load one or more .sdf / .mol / .mol2 / .pdb / .smi files")
        b_file.clicked.connect(self._load_files)
        top.addWidget(b_file)

        self.smiles_edit = QLineEdit()
        self.smiles_edit.setPlaceholderText("Paste a SMILES and press Add…")
        self.smiles_edit.returnPressed.connect(self._add_smiles_from_box)
        top.addWidget(self.smiles_edit, 1)
        b_smi = QPushButton("Add SMILES")
        b_smi.clicked.connect(self._add_smiles_from_box)
        top.addWidget(b_smi)

        bex = QPushButton("Try Example")
        bex.setToolTip("Load two example warheads with their attachment atoms pre-picked")
        bex.clicked.connect(self._load_example)
        top.addWidget(bex)
        root.addLayout(top)

        # -- viewer + panel -------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.viewer = QWebEngineView()
        try:
            from PySide6.QtWebEngineCore import QWebEngineSettings

            self.viewer.settings().setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            self.viewer.settings().setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        except Exception:
            pass
        self.channel = QWebChannel(self.viewer.page())
        self.bridge = _ClickBridge()
        self.bridge.atom_clicked.connect(self._on_atom_clicked)
        self.channel.registerObject("bridge", self.bridge)
        self.viewer.page().setWebChannel(self.channel)
        self.viewer.loadFinished.connect(self._inject_bridge)
        self._show_welcome()
        splitter.addWidget(self.viewer)
        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QFrame.Shape.NoFrame)
        panel_scroll.setWidget(self._build_panel())
        panel_scroll.setMinimumWidth(420)
        splitter.addWidget(panel_scroll)
        splitter.setSizes([780, 460])
        splitter.setStretchFactor(0, 1)
        root.addWidget(splitter, 1)

        self.status = QLabel("Load structures to begin.")
        self.status.setObjectName("Muted")
        root.addWidget(self.status)

    # -- panel -------------------------------------------------------------
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        col = QVBoxLayout(panel)

        col.addWidget(QLabel("Workspace entries:"))
        self.entry_table = QTableWidget(0, 5)
        self.entry_table.setHorizontalHeaderLabels(["In", "Title", "Type", "Role", "Atom"])
        self.entry_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.entry_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.entry_table.verticalHeader().setVisible(False)
        self.entry_table.itemChanged.connect(self._entry_item_changed)
        hh = self.entry_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.entry_table.setMinimumHeight(170)
        self.entry_table.setMaximumHeight(260)
        self.entry_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        col.addWidget(self.entry_table)

        row = QHBoxLayout()
        del_btn = QPushButton("Delete Selected")
        del_btn.setToolTip("Remove the selected entries from the workspace")
        del_btn.clicked.connect(self._delete_selected)
        clr_btn = QPushButton("Clear Atom")
        clr_btn.setToolTip("Clear the picked attachment atom on the selected ligand(s)")
        clr_btn.clicked.connect(self._clear_atom_selected)
        row.addWidget(del_btn)
        row.addWidget(clr_btn)
        row.addStretch()
        col.addLayout(row)

        col.addWidget(QLabel("Linker source:"))
        self.source_combo = NoScrollComboBox()
        self.source_combo.addItems(["LinkInvent (de novo generation)", "Linker library (search, e.g. Enamine)"])
        self.source_combo.currentIndexChanged.connect(self._update_source)
        col.addWidget(self.source_combo)

        # --- generative (LinkInvent) fields ---
        self.generative_box = QWidget()
        gbox = QVBoxLayout(self.generative_box)
        gbox.setContentsMargins(0, 0, 0, 0)
        gbox.addWidget(QLabel("REINVENT4 executable:"))
        from chemflow.generative.reinvent.locate import find_reinvent_command
        self.reinvent_edit = QLineEdit(find_reinvent_command())
        gbox.addWidget(self.reinvent_edit)
        install_reinvent = QPushButton("Install REINVENT4 env…")
        install_reinvent.setToolTip("Create a separate 'reinvent' conda env from the bundled source "
                                    "(runs in the Activity console); its executable is auto-detected")
        install_reinvent.clicked.connect(self._install_reinvent_env)
        gbox.addWidget(install_reinvent)
        gbox.addWidget(QLabel("linkinvent.prior:"))
        prior_row = QHBoxLayout()
        self.prior_edit = QLineEdit(os.environ.get("LINKINVENT_PRIOR", ""))
        self.prior_edit.textChanged.connect(lambda _=None: self._update_ready())
        prior_browse = QPushButton("Browse…")
        prior_browse.clicked.connect(self._browse_prior)
        prior_row.addWidget(self.prior_edit, 1)
        prior_row.addWidget(prior_browse)
        gbox.addLayout(prior_row)
        col.addWidget(self.generative_box)

        # --- library (search) fields ---
        self.library_box = QWidget()
        lbox = QVBoxLayout(self.library_box)
        lbox.setContentsMargins(0, 0, 0, 0)
        lbox.addWidget(QLabel("Linker library file (.smi / .csv / .sdf of bifunctional linkers):"))
        lib_row = QHBoxLayout()
        self.library_edit = QLineEdit(os.environ.get("LINKER_LIBRARY", ""))
        self.library_edit.textChanged.connect(lambda _=None: self._update_ready())
        lib_browse = QPushButton("Browse…")
        lib_browse.clicked.connect(self._browse_library)
        lib_row.addWidget(self.library_edit, 1)
        lib_row.addWidget(lib_browse)
        lbox.addLayout(lib_row)
        len_row = QHBoxLayout()
        len_row.addWidget(QLabel("Linker length (bonds): min"))
        self.min_len_spin = NoScrollSpinBox()
        self.min_len_spin.setRange(0, 60)
        self.min_len_spin.setValue(0)
        self.min_len_spin.setToolTip("0 = no minimum")
        len_row.addWidget(self.min_len_spin)
        len_row.addWidget(QLabel("max"))
        self.max_len_spin = NoScrollSpinBox()
        self.max_len_spin.setRange(0, 60)
        self.max_len_spin.setValue(0)
        self.max_len_spin.setToolTip("0 = no maximum")
        len_row.addWidget(self.max_len_spin)
        len_row.addStretch()
        lbox.addLayout(len_row)
        col.addWidget(self.library_box)
        self.library_box.setVisible(False)

        self.sel_label = QLabel("No attachment atoms picked yet.")
        self.sel_label.setWordWrap(True)
        col.addWidget(self.sel_label)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("Linkers:"))
        self.num_spin = NoScrollSpinBox()
        self.num_spin.setRange(10, 5000)
        self.num_spin.setValue(200)
        opts.addWidget(self.num_spin)
        opts.addWidget(QLabel("3D fit top:"))
        self.fit_spin = NoScrollSpinBox()
        self.fit_spin.setRange(0, 1000)
        self.fit_spin.setValue(50)
        opts.addWidget(self.fit_spin)
        opts.addWidget(QLabel("Device:"))
        self.device = NoScrollComboBox()
        self.device.addItems(["cpu", "cuda:0", "mps"])
        opts.addWidget(self.device)
        col.addLayout(opts)

        self.generate = QPushButton("Generate Linkers")
        self.generate.clicked.connect(self._generate)
        self.generate.setEnabled(False)
        col.addWidget(self.generate)

        hint = QLabel("Progress streams to the Activity console below.")
        hint.setObjectName("Muted")
        col.addWidget(hint)

        col.addWidget(QLabel("Generated linkers:"))
        self.table = QTableWidget(0, 0)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._row_selected)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(150)
        col.addWidget(self.table, 1)

        self.export_btn = QPushButton("Export (CSV / SMILES / poses)…")
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)
        col.addWidget(self.export_btn)
        return panel

    # -- entry bookkeeping -------------------------------------------------
    def _new_id(self) -> int:
        eid = self._next_id
        self._next_id += 1
        return eid

    def _find(self, eid: int) -> _Entry | None:
        for e in self._entries:
            if e.id == eid:
                return e
        return None

    def _add_entry(self, entry: _Entry) -> None:
        self._entries.append(entry)
        self._refresh_table()
        self._refresh_scene()
        self._update_ready()

    def _add_ligand(self, title: str, mol, *, has_3d: bool) -> None:
        self._add_entry(_Entry(self._new_id(), title, "ligand", mol=mol, has_3d=has_3d))

    # -- loading -----------------------------------------------------------
    def _load_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load structures", self.working_directory,
            "Structures (*.sdf *.sd *.mol *.mol2 *.pdb *.ent *.smi *.smiles)")
        for path in paths:
            self._load_one(path)

    def _load_one(self, path: str) -> None:
        from chemflow.generative.reinvent.linker.heads import load_molecule, mol_from_smiles

        p = Path(path)
        suffix = p.suffix.lower()
        try:
            if suffix in (".smi", ".smiles"):
                added = 0
                for i, line in enumerate(p.read_text(errors="replace").splitlines()):
                    parts = line.split()
                    if not parts:
                        continue
                    name = parts[1] if len(parts) > 1 else f"{p.stem}:{i + 1}"
                    self._add_ligand(name, mol_from_smiles(parts[0]), has_3d=False)
                    added += 1
                if not added:
                    QMessageBox.warning(self, "Empty file", f"No SMILES found in {p.name}")
            elif suffix in (".pdb", ".ent"):
                text = p.read_text(errors="replace")
                if self._pdb_is_protein(text):
                    self._add_entry(_Entry(self._new_id(), p.name, "protein", pdb_text=text, path=str(p)))
                else:
                    mol = load_molecule(path)
                    self._add_ligand(p.name, mol, has_3d=bool(mol.GetNumConformers()))
            else:
                mol = load_molecule(path)
                self._add_ligand(p.name, mol, has_3d=bool(mol.GetNumConformers()))
        except Exception as exc:
            QMessageBox.warning(self, "Cannot load", f"{p.name}:\n{exc}")

    def _add_smiles_from_box(self) -> None:
        from chemflow.generative.reinvent.linker.heads import mol_from_smiles

        smi = self.smiles_edit.text().strip()
        if not smi:
            return
        try:
            mol = mol_from_smiles(smi)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid SMILES", str(exc))
            return
        title = smi if len(smi) <= 28 else smi[:25] + "…"
        self._add_ligand(title, mol, has_3d=False)
        self.smiles_edit.clear()

    def _load_example(self) -> None:
        from chemflow.generative.reinvent.linker.example_data import ensure_example
        from chemflow.generative.reinvent.linker.heads import load_molecule

        dest = self._scene_dir / "example"
        try:
            warheads = ensure_example(dest)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot load example", f"{exc}")
            return
        for i, info in enumerate(warheads[:2]):
            mol = load_molecule(info["path"])
            entry = _Entry(self._new_id(), info["name"], "ligand",
                           mol=mol, has_3d=bool(mol.GetNumConformers()))
            entry.atom = int(info["attach_index"])
            entry.role = "A" if i == 0 else "B"
            self._entries.append(entry)
        self._refresh_table()
        self._refresh_scene()
        self._update_ready()
        names = " + ".join(w["name"] for w in warheads[:2])
        self.status.setText(f"Example loaded: {names}. Attachment atoms pre-picked — set the "
                            "prior path, then Generate Linkers.")

    @staticmethod
    def _pdb_is_protein(text: str) -> bool:
        for line in text.splitlines():
            if line.startswith(("SEQRES", "HELIX", "SHEET")):
                return True
            if line.startswith("ATOM") and line[17:20].strip() in _AMINO_ACIDS:
                return True
        return False

    def _browse_prior(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select linkinvent.prior", self.working_directory, "Priors (*.prior);;All files (*)")
        if path:
            self.prior_edit.setText(path)

    def _browse_library(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select linker library", self.working_directory,
            "Linker libraries (*.smi *.smiles *.csv *.txt *.sdf *.sd);;All files (*)")
        if path:
            self.library_edit.setText(path)

    def _install_reinvent_env(self) -> None:
        if self._console is not None:
            self._console.run(["reinvent-env", "install"])
            self.status.setText("Installing the REINVENT4 env in the Activity console… when it "
                                "finishes, reopen this page so the executable is auto-detected.")

    def _update_source(self) -> None:
        library_mode = self.source_combo.currentIndex() == 1
        self.generative_box.setVisible(not library_mode)
        self.library_box.setVisible(library_mode)
        self.generate.setText("Search Library" if library_mode else "Generate Linkers")
        self._update_ready()

    # -- entry table -------------------------------------------------------
    def _refresh_table(self) -> None:
        self._refreshing = True
        self.entry_table.setRowCount(len(self._entries))
        for r, e in enumerate(self._entries):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                         | Qt.ItemFlag.ItemIsSelectable)
            chk.setCheckState(Qt.CheckState.Checked if e.included else Qt.CheckState.Unchecked)
            chk.setData(Qt.ItemDataRole.UserRole, e.id)
            self.entry_table.setItem(r, 0, chk)
            self.entry_table.setItem(r, 1, QTableWidgetItem(e.title))
            if e.kind == "protein":
                kind = "protein"
            else:
                kind = "ligand (3D)" if e.has_3d else "ligand"
            self.entry_table.setItem(r, 2, QTableWidgetItem(kind))

            if e.kind == "protein":
                role_item = QTableWidgetItem("Protein (bg)")
                role_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.entry_table.setItem(r, 3, role_item)
                self.entry_table.removeCellWidget(r, 3)
            else:
                combo = NoScrollComboBox()
                combo.addItems([_ROLE_NONE, _ROLE_A, _ROLE_B])
                combo.setCurrentText(
                    _ROLE_A if e.role == "A" else _ROLE_B if e.role == "B" else _ROLE_NONE)
                combo.currentTextChanged.connect(
                    lambda text, eid=e.id: self._set_role(eid, text))
                self.entry_table.setItem(r, 3, QTableWidgetItem())
                self.entry_table.setCellWidget(r, 3, combo)

            atom = "—" if (e.kind == "protein" or e.atom is None) else str(e.atom)
            self.entry_table.setItem(r, 4, QTableWidgetItem(atom))
        self._refreshing = False

    def _set_role(self, eid: int, text: str) -> None:
        if self._refreshing:
            return
        entry = self._find(eid)
        if entry is None or entry.kind != "ligand":
            return
        role = "A" if text == _ROLE_A else "B" if text == _ROLE_B else None
        if role in ("A", "B"):
            # A role is exclusive: clear it from any other ligand.
            for other in self._entries:
                if other is not entry and other.role == role:
                    other.role = None
        entry.role = role
        self._refresh_table()
        self._refresh_scene()
        self._update_ready()

    def _entry_item_changed(self, item: QTableWidgetItem) -> None:
        if self._refreshing or item.column() != 0:
            return
        eid = item.data(Qt.ItemDataRole.UserRole)
        entry = self._find(int(eid)) if eid is not None else None
        if entry is None:
            return
        entry.included = item.checkState() == Qt.CheckState.Checked
        self._refresh_scene()
        self._update_ready()

    def _selected_ids(self) -> list[int]:
        ids = []
        for idx in self.entry_table.selectionModel().selectedRows():
            cell = self.entry_table.item(idx.row(), 0)
            if cell is not None:
                data = cell.data(Qt.ItemDataRole.UserRole)
                if data is not None:
                    ids.append(int(data))
        return ids

    def _delete_selected(self) -> None:
        ids = set(self._selected_ids())
        if not ids:
            self.status.setText("Select one or more rows first, then Delete Selected.")
            return
        self._entries = [e for e in self._entries if e.id not in ids]
        self._refresh_table()
        self._refresh_scene()
        self._update_ready()

    def _clear_atom_selected(self) -> None:
        ids = set(self._selected_ids())
        changed = False
        for e in self._entries:
            if e.id in ids and e.kind == "ligand" and e.atom is not None:
                e.atom = None
                changed = True
        if changed:
            self._refresh_table()
            self._refresh_scene()
            self._update_ready()

    # -- scene -------------------------------------------------------------
    def _display_molblock(self, entry: _Entry) -> str:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = entry.mol
        if mol.GetNumConformers():
            return Chem.MolToMolBlock(mol)
        work = Chem.Mol(mol)
        try:
            AllChem.Compute2DCoords(work)
        except Exception:
            pass
        return Chem.MolToMolBlock(work)

    def _included_proteins(self) -> list[str]:
        return [e.pdb_text for e in self._entries
                if e.kind == "protein" and e.included and e.pdb_text]

    def _refresh_scene(self) -> None:
        ligands = [(e.id, self._display_molblock(e)) for e in self._entries
                   if e.kind == "ligand" and e.included]
        proteins = self._included_proteins()
        if not ligands and not proteins:
            self._show_welcome()
            return
        selected = {e.id: e.atom for e in self._entries
                    if e.kind == "ligand" and e.included and e.atom is not None}
        role_colors = {e.id: _ROLE_COLORS[e.role] for e in self._entries
                       if e.kind == "ligand" and e.included and e.role in ("A", "B")}
        html = build_workspace_html(ligands, proteins=proteins, selected=selected, colors=role_colors)
        self._load_html(html)

    def _show_welcome(self) -> None:
        self.viewer.setHtml(
            "<html><body style='font-family:sans-serif;color:#8aa;"
            "display:flex;align-items:center;justify-content:center'>"
            "Load structures and tick “In” to display them here.</body></html>")

    def _load_html(self, html: str) -> None:
        base = QUrl.fromLocalFile(str(self._scene_dir) + "/")
        self.viewer.setHtml(html, base)

    def _inject_bridge(self, ok: bool) -> None:
        if not ok:
            return
        resource = QFile(":/qtwebchannel/qwebchannel.js")
        if not resource.open(QFile.OpenModeFlag.ReadOnly):
            self.status.setText("Qt web channel could not be initialised; clicking is disabled.")
            return
        library = bytes(resource.readAll()).decode("utf-8")
        resource.close()
        bootstrap = (
            "new QWebChannel(qt.webChannelTransport, function(channel){"
            " window.bridge = channel.objects.bridge; });"
        )
        self.viewer.page().runJavaScript(library + "\n" + bootstrap)

    # -- picking -----------------------------------------------------------
    def _on_atom_clicked(self, entry_id: int, index: int) -> None:
        entry = self._find(entry_id)
        if entry is None or entry.kind != "ligand":
            return
        entry.atom = index
        # If this ligand has no role yet, give it the first free warhead slot.
        if entry.role is None:
            taken = {e.role for e in self._entries if e.kind == "ligand" and e is not entry}
            if "A" not in taken:
                entry.role = "A"
            elif "B" not in taken:
                entry.role = "B"
        self._refresh_table()
        self._refresh_scene()
        self._update_ready()

    def _role_entry(self, role: str) -> _Entry | None:
        for e in self._entries:
            if e.kind == "ligand" and e.role == role:
                return e
        return None

    def _update_ready(self) -> None:
        a, b = self._role_entry("A"), self._role_entry("B")
        lines = []
        for label, e in (("Ligand A", a), ("Ligand B", b)):
            if e is None:
                lines.append(f"{label}: not assigned")
            elif e.atom is None:
                lines.append(f"{label}: {e.title} — click an atom in the 3D view")
            else:
                atom = e.mol.GetAtomWithIdx(e.atom)
                incl = "" if e.included else " (not shown — tick “In”)"
                lines.append(f"{label}: {e.title}, atom {e.atom} ({atom.GetSymbol()}){incl}")
        self.sel_label.setText("\n".join(lines))
        warheads_ready = (
            a is not None and b is not None
            and a.atom is not None and b.atom is not None
            and a.included and b.included
        )
        if self.source_combo.currentIndex() == 1:
            source_ready = bool(self.library_edit.text().strip())
        else:
            source_ready = bool(self.prior_edit.text().strip())
        self.generate.setEnabled(warheads_ready and source_ready)

    # -- generate ----------------------------------------------------------
    def _generate(self) -> None:
        from chemflow.generative.reinvent.linker.designer import DesignOptions
        from chemflow.generative.reinvent.linker.fit3d import FitContext, read_pdb_heavy_atoms
        from chemflow.generative.reinvent.linker.heads import make_head

        heads = [self._role_entry("A"), self._role_entry("B")]
        if heads[0] is None or heads[1] is None:
            QMessageBox.warning(self, "Assign both warheads",
                                "Set one ligand to “Ligand A” and another to “Ligand B” "
                                "(use the Role column or just click an atom on each).")
            return
        if heads[0].atom is None or heads[1].atom is None:
            QMessageBox.warning(self, "Pick attachment atoms",
                                "Click one attachment atom on both Ligand A and Ligand B.")
            return

        library_mode = self.source_combo.currentIndex() == 1
        prior = self.prior_edit.text().strip()
        library = self.library_edit.text().strip()
        if library_mode:
            if not library or not Path(library).expanduser().is_file():
                QMessageBox.warning(self, "Missing library",
                                    "Choose a linker library file (.smi/.csv/.sdf) to search.")
                return
        elif not prior:
            QMessageBox.warning(self, "Missing prior", "Set the linkinvent.prior path first.")
            return

        protein_paths = [e.path for e in self._entries
                         if e.kind == "protein" and e.included and e.path]
        try:
            protein_xyz = read_pdb_heavy_atoms(protein_paths) if protein_paths else None
            head1 = make_head(heads[0].mol, heads[0].atom, name="head1", protein_xyz=protein_xyz)
            head2 = make_head(heads[1].mol, heads[1].atom, name="head2", protein_xyz=protein_xyz)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot build attachment points", str(exc))
            return

        fit = None
        fit_top = self.fit_spin.value()
        if fit_top > 0:
            if head1.has_3d and head2.has_3d:
                fit = FitContext(head1=head1, head2=head2, protein_xyz=protein_xyz, same_frame=True)
            else:
                self.status.setText("3D fit skipped: one or both warheads lack 3D coordinates.")

        self.generate.setEnabled(False)
        if self._console is not None:
            self._console.append_line("\n=== PROTAC linker designer ===")

        if library_mode:
            min_len = self.min_len_spin.value() or None
            max_len = self.max_len_spin.value() or None
            self.status.setText("Searching the linker library…")
            self._log(f"Searching linker library: {library}")
            worker = _DesignWorker(head1, head2, None, [], fit, fit_top,
                                   library=(library, min_len, max_len))
        else:
            opts = DesignOptions(
                reinvent_command=self.reinvent_edit.text().strip() or "reinvent", prior_file=prior,
                workdir=self.working_directory,
                out_dir=str(Path(self.working_directory) / "linker_design"),
                num_smiles=self.num_spin.value(), device=self.device.currentText(),
            )
            self.status.setText("Running LinkInvent… (first run loads the model; please wait)")
            self._log("Starting LinkInvent…")
            worker = _DesignWorker(head1, head2, opts, [], fit, fit_top)

        self._worker = worker
        self._worker.progressed.connect(self._on_progress)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_ok.connect(self._on_results)
        self._worker.start()

    def _log(self, line: str) -> None:
        """Write a line to the shared Activity console."""
        if self._console is not None:
            self._console.append_line(line)

    def _on_progress(self, line: str) -> None:
        self._log(line)
        self.status.setText(line[:120])
        if self._console is not None:
            self._console.set_status("Linker designer: running")

    def _on_failed(self, message: str) -> None:
        self._update_ready()
        self._log("FAILED: " + message)
        if self._console is not None:
            self._console.set_status("Linker designer: failed")
        QMessageBox.warning(self, "Linker generation failed", message)
        self.status.setText("Generation failed.")

    def _on_results(self, candidates: list, with_fit: bool) -> None:
        from chemflow.generative.reinvent.linker.designer import table_columns, table_rows

        self._update_ready()
        self._candidates = candidates
        self._with_fit = with_fit
        if not candidates:
            self._log("Finished, but no valid linkers were generated.")
            self.status.setText("No valid linkers were generated.")
            return
        if with_fit:
            # Rank by pocket fit: fits first, then fewest clashes, least strain, then NLL.
            def _fit_key(c):
                s = c.scores
                ok_rank = 0 if s.get("fit_ok") else (2 if s.get("fit_ok") is None else 1)
                clashes = s.get("fit_clashes")
                strain = s.get("fit_strain")
                return (
                    ok_rank,
                    clashes if clashes is not None else 9999,
                    strain if strain is not None else 9e9,
                    c.nll if c.nll is not None else 9e9,
                )
            candidates = sorted(candidates, key=_fit_key)
            self._candidates = candidates
        columns = table_columns(candidates, with_fit=with_fit)
        headers = ["id", "linker", "full molecule", "NLL"] + [label for _k, label, _f in columns]
        rows = table_rows(candidates, columns)
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            # table_rows gives [id, linker, NLL, *scores]; insert the full assembled
            # PROTAC SMILES right after the linker column so it is visible/copyable.
            full = [row[0], row[1], candidates[r].protac_smiles] + list(row[2:])
            for c, value in enumerate(full):
                self.table.setItem(r, c, QTableWidgetItem(value))
        self.export_btn.setEnabled(True)
        self._log(f"Done: {len(candidates)} linkers generated.")
        if self._console is not None:
            self._console.set_status("Linker designer: complete")
        ranked = "ranked by pocket fit" if with_fit else "ranked by model NLL (set 3D fit top > 0 to score pocket fit)"
        self.status.setText(f"{len(candidates)} linkers, {ranked}. Select a row to see the full structure.")

    def _molblock_from_smiles(self, smiles: str) -> str | None:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            AllChem.Compute2DCoords(mol)
            return Chem.MolToMolBlock(mol)
        except Exception:
            return None

    def _row_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        cand = self._candidates[rows[0].row()]
        proteins = self._included_proteins()
        protein = proteins[0] if proteins else None
        if cand.fit_molblock:
            # Full assembled PROTAC posed in 3D (warheads + linker), in the pocket.
            self._load_html(build_pose_html(cand.fit_molblock, protein_pdb=protein))
            self.status.setText(f"{cand.id}: full 3D pose shown.")
            return
        # No 3D fit — still show the full assembled molecule as a 2D depiction.
        block = self._molblock_from_smiles(cand.protac_smiles)
        if block:
            self._load_html(build_pose_html(block))
            self.status.setText(f"{cand.id}: full molecule (2D). Run with 3D fit top > 0 for a posed 3D structure.")
        else:
            self.status.setText(f"{cand.id}: {cand.protac_smiles}")

    def _export(self) -> None:
        from chemflow.generative.reinvent.linker.designer import export_design

        if not self._candidates:
            return
        out = QFileDialog.getExistingDirectory(self, "Export to folder", self.working_directory)
        if not out:
            return
        files = export_design(self._candidates, out, with_fit=self._with_fit)
        QMessageBox.information(self, "Exported", "Wrote:\n" + "\n".join(str(p) for p in files.values()))
