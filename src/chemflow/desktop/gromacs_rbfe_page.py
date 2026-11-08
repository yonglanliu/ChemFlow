"""GROMACS Ligand-RBFE workflow page for ChemFlow Studio.

Tabbed workflow (same layout as the OpenFE RBFE page) for the PMX GROMACS
relative binding free-energy pipeline: Ligand Prep (charge) → Network (plan/view/
edit) → System → FEP simulation → Job (HPC) → Execution. The form writes one TOML
config; the actions run ``chemflow ligand-rbfe`` / ``chemflow rbfe`` through the
GROMACS conda env via the shared Activity console.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QFile, QObject, Qt, QUrl, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.gromacs_rbfe_config import (
    build_config,
    parse_list,
    to_toml,
    validate_setup,
)
from chemflow.desktop.network_edit import connectivity_issue, ligand_names, read_network
from chemflow.desktop.widgets.forms import (  # noqa: F811
    NoScrollComboBox as QComboBox,
    NoScrollDoubleSpinBox as QDoubleSpinBox,
    NoScrollSpinBox as QSpinBox,
    PageHeader,
    PathField,
    action_button,
)


class _EdgeBridge(QObject):
    edge_clicked = Signal(str, str)
    node_clicked = Signal(str)

    @Slot(str, str)
    def on_edge_clicked(self, a: str, b: str) -> None:
        self.edge_clicked.emit(a, b)

    @Slot(str)
    def on_node_clicked(self, name: str) -> None:
        self.node_clicked.emit(name)


def _web_view():
    from PySide6.QtWebEngineWidgets import QWebEngineView

    return QWebEngineView()


def _combo(items, current):
    box = QComboBox()
    box.addItems(items)
    if current in items:
        box.setCurrentText(current)
    return box


def _dspin(value, lo, hi, step=0.1, decimals=3):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    return s


def _spin(value, lo, hi):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


def _sum_itp_charges(itp: Path) -> int | None:
    """Net charge = round(sum of partial charges) from an acpype `_GMX.itp` [atoms]."""
    if not itp.is_file():
        return None
    total, in_atoms, found = 0.0, False, False
    for line in itp.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("["):
            in_atoms = s.replace(" ", "").lower().startswith("[atoms]")
            continue
        if in_atoms and s and not s.startswith(";"):
            parts = s.split()
            if len(parts) >= 7:
                try:
                    total += float(parts[6])
                    found = True
                except ValueError:
                    pass
    return round(total) if found else None


class GromacsRbfePage(QTabWidget):
    """Tabbed GROMACS ligand-RBFE setup (charge → network → system → FEP → job)."""

    def __init__(self, console=None, parent=None):
        super().__init__(parent)
        self._console = console
        self.setDocumentMode(True)
        self.addTab(self._ligand_tab(), "1 · Ligand Prep")
        self.addTab(self._network_tab(), "2 · Network")
        self.addTab(self._system_tab(), "3 · System")
        self.addTab(self._fep_tab(), "4 · FEP simulation")
        self.addTab(self._job_tab(), "5 · Job (HPC)")
        self.addTab(self._exec_tab(), "Execution")
        # Now that the Job tab's HREX fields exist, apply the sampling gating.
        self._on_sampling_changed(self.sampling.currentText())

    def _wrap(self, header, body):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(header)
        layout.addWidget(body)
        layout.addStretch()
        return page

    # -- 1 · Ligand Prep ---------------------------------------------------
    def _ligand_tab(self):
        form = QWidget()
        f = QFormLayout(form)
        self.library_sdf = PathField("ligands.sdf (one record per ligand, unique _Name)")
        self.charge_manifest = PathField("charges.tsv / .csv (optional per-ligand net charges)")
        self.ligand_output = PathField(
            "output folder (blank = <work>/<project>/ligands)", mode="directory")
        self.ligand_output.setToolTip(
            "Each ligand is written to <output>/<residue name>/<residue name>.{sdf,mol2,itp,gro,top} "
            "plus ligand_manifest.tsv.")
        f.addRow("Ligand library SDF:", self.library_sdf)
        f.addRow("Output folder:", self.ligand_output)
        f.addRow("Charge manifest:", self.charge_manifest)

        row = QHBoxLayout()
        calc = action_button("Calculate charges")
        calc.setToolTip("Run chemflow rbfe assign-charge now (BCC via Open Babel + acpype/antechamber).")
        calc.clicked.connect(self._calc_charges)
        refresh = action_button("Refresh table")
        refresh.clicked.connect(self._refresh_ligand_table)
        self.charge_overwrite = QCheckBox("Overwrite existing")
        self.charge_overwrite.setToolTip("Replace existing per-ligand charge output (re-run after a failure).")
        row.addWidget(calc)
        row.addWidget(refresh)
        row.addWidget(self.charge_overwrite)
        row.addStretch()

        # ligand _Name -> user-edited residue name (survives table refreshes)
        self._resname_edits: dict[str, str] = {}
        self._populating_ligands = False
        self.ligand_table = QTableWidget(0, 4)
        self.ligand_table.setHorizontalHeaderLabels(
            ["Ligand", "Residue name", "Charge (before)", "Net charge (after)"])
        self.ligand_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.ligand_table.setMinimumWidth(420)
        self.ligand_table.itemSelectionChanged.connect(self._on_ligand_selected)
        self.ligand_table.itemChanged.connect(self._on_resname_edited)
        self.library_sdf.edit.textChanged.connect(lambda _=None: self._refresh_ligand_table())
        self.ligand_output.edit.textChanged.connect(lambda _=None: self._refresh_ligand_table())

        self.structure_view = _web_view()
        self.structure_view.setMinimumWidth(360)

        split = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(form)
        left.addLayout(row)
        left.addWidget(self.ligand_table, 1)
        split.addLayout(left, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("Structure:"))
        right.addWidget(self.structure_view, 1)
        split.addLayout(right, 1)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("GROMACS RBFE", "1 · Ligand preparation",
                                    "Charge + parameterize the ligands (BCC/GAFF2 via acpype). "
                                    "Click a ligand to see its structure."))
        layout.addLayout(split, 1)
        self.charge_status = QLabel("Set the ligand SDF, then Calculate charges / Refresh table.")
        self.charge_status.setObjectName("Muted")
        self.charge_status.setWordWrap(True)
        layout.addWidget(self.charge_status)
        return page

    def _calc_charges(self) -> None:
        problem = self._resname_problem()
        if problem:
            self.charge_status.setText(problem)
            return
        path = self._write_config()
        if not path:
            return
        argv = ["rbfe", "assign-charge", path]
        if self.charge_overwrite.isChecked():
            argv.insert(2, "--overwrite")
        self._run(argv)
        self.charge_status.setText("Charging ligands in the Activity console… click Refresh table "
                                   "when it finishes to see the 'after' net charges.")

    # -- residue names (editable per ligand; drive the output folder + files) --
    @staticmethod
    def _default_resname(index: int, total: int) -> str:
        return f"L{index + 1:02d}" if total > 1 else "LIG"

    def _table_resnames(self) -> list[str]:
        out = []
        for r in range(self.ligand_table.rowCount()):
            item = self.ligand_table.item(r, 1)
            out.append(item.text().strip().upper() if item else "")
        return out

    def _resname_problem(self) -> str | None:
        import re
        names = [self.ligand_table.item(r, 0).text()
                 for r in range(self.ligand_table.rowCount())]
        resnames = self._table_resnames()
        for name, res in zip(names, resnames):
            if not re.fullmatch(r"[A-Z0-9]{1,5}", res):
                return f"Residue name for {name!r} must be 1-5 letters/numbers (got {res!r})."
        dupes = sorted({r for r in resnames if resnames.count(r) > 1})
        if dupes:
            return "Residue names must be unique; duplicated: " + ", ".join(dupes) + "."
        return None

    def _on_resname_edited(self, item) -> None:
        if self._populating_ligands or item.column() != 1:
            return
        text = item.text().strip().upper()
        self._populating_ligands = True
        item.setText(text)
        self._populating_ligands = False
        name_item = self.ligand_table.item(item.row(), 0)
        if name_item is not None:
            self._resname_edits[name_item.text()] = text
        problem = self._resname_problem()
        self.charge_status.setText(
            problem or "Residue names set. The output folder and files use them "
                       "(<residue>/<residue>.itp/.gro/.top), and the molecule's residue "
                       "name inside those files matches.")
        self._refresh_after_charges()

    def _refresh_after_charges(self) -> None:
        """Refill the 'after' column from the output folder using current resnames."""
        names = [self.ligand_table.item(r, 0).text()
                 for r in range(self.ligand_table.rowCount())]
        after = {**self._charged_net_charges(names, self._table_resnames()),
                 **self._manifest_net_charges()}
        self._populating_ligands = True
        for r, name in enumerate(names):
            net = after.get(name)
            self.ligand_table.setItem(r, 3, QTableWidgetItem("—" if net is None else f"{net:+d}"))
        self._populating_ligands = False

    def _ligands_output_dir(self) -> Path:
        cfg = self.config_path.text()
        root = Path(cfg).expanduser().resolve().parent if cfg else Path.cwd()
        explicit = self.ligand_output.text().strip()
        if explicit:
            base = Path(explicit).expanduser()
            return base if base.is_absolute() else root / base
        work = self.work_dir.text().strip() or "work"
        name = self.project_name.text().strip() or "rbfe_project"
        base = Path(work).expanduser()
        if not base.is_absolute():
            base = root / base
        return base / name / "ligands"

    def _charged_net_charges(self, names: list[str], resnames: list[str] | None = None) -> dict:
        base = self._ligands_output_dir()
        if not base.is_dir():
            return {}
        n = len(names)
        resnames = resnames or [self._default_resname(i, n) for i in range(n)]
        out: dict[str, int] = {}
        for name, resname in zip(names, resnames):
            for candidate in (resname, name):
                itp = base / candidate / f"{candidate}.acpype" / f"{candidate}_GMX.itp"
                net = _sum_itp_charges(itp)
                if net is not None:
                    out[name] = net
                    break
        return out

    def _manifest_net_charges(self) -> dict:
        path = self.charge_manifest.text()
        if not path or not Path(path).expanduser().is_file():
            return {}
        import csv
        source = Path(path).expanduser()
        delimiter = "\t" if source.suffix.casefold() == ".tsv" else ","
        out: dict[str, int] = {}
        try:
            with source.open(newline="", encoding="utf-8") as handle:
                for r in csv.DictReader(handle, delimiter=delimiter):
                    name = (r.get("ligand_name") or r.get("name") or "").strip()
                    value = (r.get("net_charge") or "").strip()
                    if name and value:
                        out[name] = int(float(value))
        except (OSError, ValueError):
            return {}
        return out

    def _refresh_ligand_table(self) -> None:
        src = self.library_sdf.text()
        if not src:
            self.charge_status.setText("Set the ligand library SDF first (then Refresh).")
            return
        if not Path(src).expanduser().is_file():
            self.charge_status.setText(f"Ligand SDF not found: {src}")
            return
        try:
            from chemflow.fe.openfe.charge_report import formal_charges
            before = formal_charges(src)
        except Exception as exc:
            self.charge_status.setText(f"Could not read ligands: {exc}")
            return
        names = [name for name, _ in before]
        resnames = [self._resname_edits.get(name) or self._default_resname(i, len(names))
                    for i, name in enumerate(names)]
        after = {**self._charged_net_charges(names, resnames), **self._manifest_net_charges()}
        self._populating_ligands = True
        self.ligand_table.setRowCount(len(before))
        for r, (name, formal) in enumerate(before):
            net = after.get(name)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            res_item = QTableWidgetItem(resnames[r])
            res_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            res_item.setToolTip("Editable 1-5 character residue name. Names the output folder/files "
                                "and the residue inside the .itp/.gro/.top.")
            formal_item = QTableWidgetItem(f"{formal:+d}")
            formal_item.setFlags(formal_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            net_item = QTableWidgetItem("—" if net is None else f"{net:+d}")
            net_item.setFlags(net_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.ligand_table.setItem(r, 0, name_item)
            self.ligand_table.setItem(r, 1, res_item)
            self.ligand_table.setItem(r, 2, formal_item)
            self.ligand_table.setItem(r, 3, net_item)
        self._populating_ligands = False
        self.ligand_table.resizeColumnsToContents()
        self.charge_status.setText(
            f"Loaded {len(before)} ligand(s)." +
            ("" if after else " 'after' net charges appear once you charge (acpype .itp) "
             "or provide a charge manifest."))

    def _on_ligand_selected(self) -> None:
        rows = self.ligand_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.ligand_table.item(rows[0].row(), 0)
        src = self.library_sdf.text()
        if item is None or not src:
            return
        try:
            from chemflow.fe.openfe.charge_report import molecule_svg
            svg = molecule_svg(src, item.text())
        except Exception as exc:
            self.charge_status.setText(f"Could not draw {item.text()}: {exc}")
            return
        body = svg or f"<p style='color:#888'>No depiction for {item.text()}</p>"
        self.structure_view.setHtml(
            f"<html><body style='margin:0;display:flex;align-items:center;"
            f"justify-content:center;height:100%'>{body}</body></html>")

    # -- 2 · Network -------------------------------------------------------
    def _network_tab(self):
        form = QWidget()
        f = QFormLayout(form)
        self.network_method = _combo(
            ["minimal_spanning", "minimal_redundant", "lomap", "radial"], "minimal_spanning")
        self.atom_mapper = _combo(["lomap", "kartograf"], "lomap")
        self.max3d = _dspin(1.0, 0.0, 20.0, 0.5, 2)
        self.radial_hub = QLineEdit("")
        self.radial_hub.setPlaceholderText("hub ligand _Name (required for radial)")
        self.element_change = QCheckBox("Allow element change")
        self.require_cycle = QCheckBox("Require cycle covering")
        self.require_cycle.setChecked(True)
        f.addRow("Network method:", self.network_method)
        f.addRow("Atom mapper:", self.atom_mapper)
        f.addRow("Max 3D (Å):", self.max3d)
        f.addRow("Radial hub:", self.radial_hub)
        f.addRow("", self.element_change)
        f.addRow("", self.require_cycle)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("GROMACS RBFE", "2 · Network creation",
                                    "Plan the perturbation network, then review the graph and "
                                    "add/remove edges. Click an edge to see its atom mapping."))
        layout.addWidget(form)

        # Plan from: an SDF file, OR a folder of per-ligand SDFs (e.g. the charged
        # ligands folder); blank uses the Ligand Prep library SDF.
        src_row = QHBoxLayout()
        self.plan_ligands = PathField("ligands .sdf  (blank = Ligand Prep SDF)")
        browse_folder = QPushButton("Browse folder…")
        browse_folder.setToolTip("Pick the charged-ligands folder (per-ligand SDFs are combined).")
        browse_folder.clicked.connect(self._browse_plan_folder)
        src_row.addWidget(QLabel("Plan from:"))
        src_row.addWidget(self.plan_ligands, 1)
        src_row.addWidget(browse_folder)
        plan_btn = action_button("Plan network")
        plan_btn.setToolTip("Run chemflow rbfe plan (writes the bundle into the project network dir).")
        plan_btn.clicked.connect(self._plan_network)
        src_row.addWidget(plan_btn)
        layout.addLayout(src_row)

        dir_row = QHBoxLayout()
        self.network_dir = PathField("network bundle dir (edges.csv + ligand_network.graphml)",
                                     mode="directory")
        load_btn = QPushButton("Load edges")
        load_btn.clicked.connect(self._load_edges)
        graph_btn = QPushButton("Show network graph")
        graph_btn.clicked.connect(self._show_graph)
        dir_row.addWidget(self.network_dir, 1)
        dir_row.addWidget(load_btn)
        dir_row.addWidget(graph_btn)
        layout.addLayout(dir_row)

        self.edge_table = QTableWidget(0, 3)
        self.edge_table.setHorizontalHeaderLabels(["Ligand A", "Ligand B", "Score"])
        self.edge_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.edge_table.setMinimumHeight(150)
        layout.addWidget(self.edge_table)

        edit_row = QHBoxLayout()
        self.edge_a = QComboBox(); self.edge_a.setEditable(True)
        self.edge_b = QComboBox(); self.edge_b.setEditable(True)
        add_btn = QPushButton("Add edge"); add_btn.clicked.connect(self._add_edge)
        rm_btn = QPushButton("Remove selected"); rm_btn.clicked.connect(self._remove_edges)
        edit_row.addWidget(QLabel("A:")); edit_row.addWidget(self.edge_a, 1)
        edit_row.addWidget(QLabel("B:")); edit_row.addWidget(self.edge_b, 1)
        edit_row.addWidget(add_btn); edit_row.addWidget(rm_btn)
        layout.addLayout(edit_row)

        self._graph_layout = QVBoxLayout()
        layout.addLayout(self._graph_layout)

        self.network_status = QLabel("Plan or load a network bundle to review its edges. "
                                     "Add/remove run in the Activity console; click Load edges to refresh.")
        self.network_status.setObjectName("Muted")
        self.network_status.setWordWrap(True)
        layout.addWidget(self.network_status)
        layout.addStretch()
        return page

    def _default_network_dir(self) -> str:
        work = self.work_dir.text().strip() or "work"
        name = self.project_name.text().strip() or "rbfe_project"
        return str(Path(work) / name / "network")

    def _browse_plan_folder(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        start = self.plan_ligands.text() or str(self._ligands_output_dir())
        chosen = QFileDialog.getExistingDirectory(self, "Choose ligands folder", start)
        if chosen:
            self.plan_ligands.setText(chosen)

    def _resolve_plan_sdf(self) -> str | None:
        """Ligand SDF to plan from: a chosen SDF, a folder of per-ligand SDFs
        (combined into one), or the Ligand Prep library SDF."""
        src = self.plan_ligands.text().strip()
        if not src:
            return self.library_sdf.text().strip() or None
        p = Path(src).expanduser()
        if p.is_dir():
            sdfs = sorted(q for q in p.glob("**/*.sdf")
                          if q.name != "planning_ligands.sdf")
            if not sdfs:
                self.network_status.setText(f"No .sdf files found under {p}")
                return None
            cfg = self.config_path.text()
            dest_dir = Path(cfg).expanduser().resolve().parent if cfg else p
            combined = dest_dir / "planning_ligands.sdf"
            combined.write_text(
                "".join(f.read_text(encoding="utf-8", errors="replace") for f in sdfs),
                encoding="utf-8")
            return str(combined)
        return str(p)

    def _plan_network(self) -> None:
        sdf = self._resolve_plan_sdf()
        if not sdf:
            self.network_status.setText("Set a ligands SDF/folder (or the Ligand Prep SDF) to plan from.")
            return
        # Point the config's library at the chosen SDF so the plan uses it.
        self.library_sdf.setText(sdf)
        path = self._write_config()
        if not path:
            return
        if not self.network_dir.text():
            self.network_dir.setText(self._default_network_dir())
        self._run(["rbfe", "plan", path])
        self.network_status.setText(f"Planning network from {Path(sdf).name} in the Activity console… "
                                    "click Load edges when it finishes.")

    def _show_graph(self) -> None:
        directory = self.network_dir.text() or self._default_network_dir()
        try:
            from chemflow.fe.openfe.network_view import render_network_html
            html = render_network_html(directory)
        except Exception as exc:
            self.network_status.setText(f"Could not draw network: {exc}")
            return
        if not hasattr(self, "_graph_view"):
            self._init_graph_view()
        self._graph_view.setHtml(html, QUrl("about:blank"))
        self.network_status.setText("Network drawn — click an edge to see its atom mapping.")

    def _init_graph_view(self) -> None:
        from PySide6.QtWebChannel import QWebChannel

        self._graph_view = _web_view()
        self._graph_view.setMinimumHeight(340)
        self._edge_bridge = _EdgeBridge()
        self._edge_bridge.edge_clicked.connect(self._on_edge_clicked)
        self._edge_bridge.node_clicked.connect(self._on_node_clicked)
        channel = QWebChannel(self._graph_view.page())
        channel.registerObject("bridge", self._edge_bridge)
        self._graph_view.page().setWebChannel(channel)
        self._graph_view.loadFinished.connect(self._inject_graph_bridge)
        self._graph_layout.addWidget(self._graph_view)

    def _inject_graph_bridge(self, ok: bool) -> None:
        if not ok:
            return
        resource = QFile(":/qtwebchannel/qwebchannel.js")
        if not resource.open(QFile.OpenModeFlag.ReadOnly):
            return
        library = bytes(resource.readAll()).decode("utf-8")
        resource.close()
        self._graph_view.page().runJavaScript(
            library + "\nnew QWebChannel(qt.webChannelTransport, function(c){window.bridge=c.objects.bridge;});")

    def _on_node_clicked(self, name: str) -> None:
        picks = getattr(self, "_node_picks", [])
        if name in picks:
            picks.remove(name)
        else:
            picks.append(name)
            picks = picks[-2:]
        self._node_picks = picks
        if picks:
            self.edge_a.setCurrentText(picks[0])
        if len(picks) > 1:
            self.edge_b.setCurrentText(picks[1])
        self.network_status.setText("Selected: " + " , ".join(picks) +
                                    ("  — click Add edge" if len(picks) == 2 else ""))

    def _on_edge_clicked(self, a: str, b: str) -> None:
        directory = self.network_dir.text() or self._default_network_dir()
        try:
            from chemflow.fe.openfe.network_view import render_atom_mapping_html
            html = render_atom_mapping_html(directory, a, b)
        except Exception as exc:
            self.network_status.setText(f"Mapping unavailable for {a}–{b}: {exc}")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Atom mapping: {a} ↔ {b}")
        dialog.resize(900, 560)
        lay = QVBoxLayout(dialog)
        view = _web_view()
        view.setHtml(html, QUrl("about:blank"))
        lay.addWidget(view)
        dialog.show()
        self._mapping_dialog = dialog

    def _load_edges(self) -> None:
        directory = self.network_dir.text() or self._default_network_dir()
        try:
            edges = read_network(directory)
        except (FileNotFoundError, ValueError) as exc:
            self.network_status.setText(f"Could not load network: {exc}")
            return
        self.edge_table.setRowCount(len(edges))
        for r, e in enumerate(edges):
            self.edge_table.setItem(r, 0, QTableWidgetItem(e.ligand_a))
            self.edge_table.setItem(r, 1, QTableWidgetItem(e.ligand_b))
            self.edge_table.setItem(r, 2, QTableWidgetItem(e.score))
        names = ligand_names(edges)
        for combo in (self.edge_a, self.edge_b):
            combo.clear()
            combo.addItems(names)
        issue = connectivity_issue(edges)
        self.network_status.setText(
            f"{len(edges)} edges over {len(names)} ligands." + (f"  ⚠ {issue}" if issue else " Connected."))

    def _add_edge(self) -> None:
        directory = self.network_dir.text() or self._default_network_dir()
        a, b = self.edge_a.currentText().strip(), self.edge_b.currentText().strip()
        if not a or not b or a == b:
            self.network_status.setText("Pick two different ligands to add an edge.")
            return
        argv = ["network", "add", directory, "--a", a, "--b", b]
        if self.library_sdf.text():
            argv += ["--ligand-sdf", self.library_sdf.text()]
        self._run(argv)
        self.network_status.setText(f"Adding edge {a}–{b}… click Load edges to refresh when it finishes.")

    def _remove_edges(self) -> None:
        directory = self.network_dir.text() or self._default_network_dir()
        rows = self.edge_table.selectionModel().selectedRows()
        if not rows:
            self.network_status.setText("Select one or more edge rows to remove.")
            return
        argv = ["network", "remove", directory]
        for idx in rows:
            a = self.edge_table.item(idx.row(), 0).text()
            b = self.edge_table.item(idx.row(), 1).text()
            argv += ["--drop", f"{a}:{b}"]
        self._run(argv)
        self.network_status.setText("Removing selected edges… click Load edges to refresh when it finishes.")

    # -- 3 · System --------------------------------------------------------
    def _system_tab(self):
        inner = QWidget()
        v = QVBoxLayout(inner)

        # -- Receptor & system type ------------------------------------------
        recv = QGroupBox("Receptor & system type")
        rf = QFormLayout(recv)
        self.project_name = QLineEdit("rbfe_project")
        self.work_dir = PathField("work (project dir = work_dir/name)", mode="directory")
        self.work_dir.setText("work")
        self.protein = PathField("protein.pdb (receptor)")
        self.system_type = _combo(["soluble", "membrane"], "soluble")
        self.system_type.currentTextChanged.connect(self._on_system_type_changed)
        detect_btn = QPushButton("Detect from PDB")
        detect_btn.setToolTip("Scan the protein PDB for lipid residues and set the type.")
        detect_btn.clicked.connect(self._detect_membrane)
        type_row = QWidget()
        tr = QHBoxLayout(type_row)
        tr.setContentsMargins(0, 0, 0, 0)
        tr.addWidget(self.system_type, 1)
        tr.addWidget(detect_btn)
        rf.addRow("Project name:", self.project_name)
        rf.addRow("Work dir:", self.work_dir)
        rf.addRow("Protein PDB:", self.protein)
        rf.addRow("System type:", type_row)
        v.addWidget(recv)

        # -- Membrane (PBC / semi-isotropic pressure coupling) ----------------
        self.membrane_group = QGroupBox("Membrane settings (z-normal, semi-isotropic PBC)")
        mf = QFormLayout(self.membrane_group)
        self.lipid_resnames = QLineEdit("POPC")
        self.lipid_resnames.setToolTip("Space/comma-separated lipid residue names.")
        self.minimum_lipids = _spin(20, 1, 100000)
        self.membrane_barostat = _combo(
            ["Parrinello-Rahman", "C-rescale", "Berendsen"], "Parrinello-Rahman")
        self.ref_p_xy = _dspin(1.0, 0.0, 1000.0, 0.1, 3)
        self.ref_p_z = _dspin(1.0, 0.0, 1000.0, 0.1, 3)
        self.compress_xy = _dspin(4.5e-5, 0.0, 1.0, 1e-5, 8)
        self.compress_z = _dspin(4.5e-5, 0.0, 1.0, 1e-5, 8)
        self.refcoord_scaling = _combo(["com", "all", "no"], "com")
        mf.addRow("Lipid resnames:", self.lipid_resnames)
        mf.addRow("Minimum lipids:", self.minimum_lipids)
        mf.addRow("Barostat:", self.membrane_barostat)
        mf.addRow("Ref. pressure xy / z (bar):", self._pair_row(self.ref_p_xy, self.ref_p_z))
        mf.addRow("Compressibility xy / z (1/bar):",
                  self._pair_row(self.compress_xy, self.compress_z))
        mf.addRow("Refcoord scaling:", self.refcoord_scaling)
        pbc_note = QLabel(
            "Membrane runs couple pressure semi-isotropically with the bilayer "
            "normal along z (xy and z scaled independently); this PBC pressure "
            "contract is written automatically into the complex MDP.")
        pbc_note.setObjectName("Muted")
        pbc_note.setWordWrap(True)
        mf.addRow(pbc_note)
        v.addWidget(self.membrane_group)

        # -- Component force fields (system preparation) ----------------------
        prep = QGroupBox("Component force fields")
        pf = QFormLayout(prep)
        self.prepare_enabled = QCheckBox(
            "Assemble the complex from these force fields (prepare-system step)")
        self.prepare_enabled.setToolTip(
            "On: build the solvated / membrane complex with the selected component "
            "force fields plus any added ITPs. Off: use the protein PDB (or a "
            "prepared complex) directly. Required for membrane and added components.")
        self.prepare_enabled.toggled.connect(self._on_prepare_toggled)
        self.protein_ff = _combo(["ff14SB", "ff19SB", "ff99SBildn"], "ff14SB")
        self.ligand_ff = _combo(["GAFF2"], "GAFF2")
        self.lipid_ff = _combo(["Lipid21", "Lipid17"], "Lipid21")
        self.dna_ff = _combo(["OL21", "bsc1", "None"], "OL21")
        self.rna_ff = _combo(["OL3", "None"], "OL3")
        self.ion_params = QLineEdit("Joung-Cheatham TIP3P")
        pf.addRow(self.prepare_enabled)
        pf.addRow("Protein force field:", self.protein_ff)
        pf.addRow("Ligand force field:", self.ligand_ff)
        pf.addRow("Lipid force field:", self.lipid_ff)
        pf.addRow("DNA force field:", self.dna_ff)
        pf.addRow("RNA force field:", self.rna_ff)
        pf.addRow("Ion parameters:", self.ion_params)

        # Added force fields: ITPs uploaded from the right-hand component inventory
        # for residues the base force fields do not cover (ligands/cofactors/ions).
        self.added_ff_list = QListWidget()
        self.added_ff_list.setMaximumHeight(120)
        self.added_ff_list.setToolTip(
            "ITP force fields attached in the component inventory (right).")
        add_ff_remove = QPushButton("Remove selected")
        add_ff_remove.clicked.connect(self._remove_added_ff)
        add_ff_hint = QLabel(
            "Scan the PDB (right) and attach an ITP to each “needs FF” component; "
            "they appear here.")
        add_ff_hint.setObjectName("Muted")
        add_ff_hint.setWordWrap(True)
        pf.addRow("Added force fields:", self.added_ff_list)
        pf.addRow("", add_ff_remove)
        pf.addRow("", add_ff_hint)
        self._prep_group = prep
        v.addWidget(prep)

        # -- Solvation --------------------------------------------------------
        solv = QGroupBox("Solvation")
        sf = QFormLayout(solv)
        self.force_field = QLineEdit("amber99sb-star-ildn-mut")
        self.water_model = QLineEdit("tip3p")
        self.gmx = QLineEdit("gmx")
        self.box_shape = _combo(["dodecahedron", "cube", "octahedron"], "dodecahedron")
        self.box_distance = _dspin(1.2, 0.1, 10.0, 0.1, 2)
        self.neutralize = QCheckBox("Neutralize")
        self.neutralize.setChecked(True)
        self.ion_conc = _dspin(0.15, 0.0, 2.0, 0.01, 3)
        sf.addRow("Force field:", self.force_field)
        sf.addRow("Water model:", self.water_model)
        sf.addRow("GROMACS command:", self.gmx)
        sf.addRow("Box shape:", self.box_shape)
        sf.addRow("Box padding (nm):", self.box_distance)
        sf.addRow("Ionic strength (M):", self.ion_conc)
        sf.addRow("", self.neutralize)
        v.addWidget(solv)

        # -- Reuse prepared inputs (network + charged ligands) ----------------
        reuse = QGroupBox("Reuse prepared inputs")
        ref = QFormLayout(reuse)
        self.existing_network_dir = PathField(
            "network bundle dir (edges.csv + ligand_network.graphml + manifest)",
            mode="directory")
        self.charged_ligands_dir = PathField(
            "charged ligands folder (prepared per-ligand parameters)",
            mode="directory")
        ref.addRow("Generated network folder:", self.existing_network_dir)
        ref.addRow("Charged ligands folder:", self.charged_ligands_dir)
        v.addWidget(reuse)
        v.addStretch()

        self._on_system_type_changed(self.system_type.currentText())
        self._on_prepare_toggled(self.prepare_enabled.isChecked())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        # Two columns: left = the setup form, right = the PDB component inventory
        # the user scans to verify what parameterizes and attach ITPs for gaps.
        columns = QWidget()
        cols = QHBoxLayout(columns)
        cols.setContentsMargins(0, 0, 0, 0)
        cols.addWidget(scroll, 3)
        cols.addWidget(self._inventory_panel(), 2)
        # Re-scan automatically whenever a different PDB is loaded so the
        # inventory/charge estimate never shows the previous structure.
        self.protein.edit.textChanged.connect(lambda _=None: self._scan_components())

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader(
            "GROMACS RBFE", "3 · System",
            "Receptor, system type (soluble/membrane), component force fields and "
            "cofactor, solvation, and reused network / charged-ligand folders."))
        layout.addWidget(columns, 1)
        return page

    # -- Right column: PDB component inventory ----------------------------------
    def _inventory_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("Component inventory")
        bv = QVBoxLayout(box)
        intro = QLabel(
            "Scan the protein PDB to list every residue and whether the selected "
            "force fields parameterize it. Rows marked “needs FF” are ligands, "
            "cofactors or non-standard ions — attach a prepared ITP for any that "
            "cannot be parameterized automatically.")
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        bv.addWidget(intro)

        scan_row = QHBoxLayout()
        scan_btn = action_button("Scan loaded PDB")
        scan_btn.setToolTip("Classify residues in the Protein PDB field against the force fields.")
        scan_btn.clicked.connect(self._scan_components)
        scan_row.addWidget(scan_btn)
        scan_row.addStretch()
        bv.addLayout(scan_row)

        self.inventory_table = QTableWidget(0, 5)
        self.inventory_table.setHorizontalHeaderLabels(
            ["Residue", "Class", "Count", "Coverage", "Force field / ITP"])
        self.inventory_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.inventory_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.inventory_table.setMinimumHeight(360)
        self.inventory_table.verticalHeader().setVisible(False)
        header = self.inventory_table.horizontalHeader()
        for col in range(4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        # The action column fills the remaining width so the button is never clipped.
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.inventory_table.setColumnWidth(4, 220)
        bv.addWidget(self.inventory_table, 1)

        self.inventory_status = QLabel(
            "No scan yet. Pick a Protein PDB, then Scan loaded PDB.")
        self.inventory_status.setObjectName("Muted")
        self.inventory_status.setWordWrap(True)
        bv.addWidget(self.inventory_status)

        self.charge_summary = QLabel("")
        self.charge_summary.setWordWrap(True)
        self.charge_summary.setVisible(False)
        bv.addWidget(self.charge_summary)

        v.addWidget(box, 1)
        # resname -> attached ITP path (gaps the user resolves manually).
        self._component_itps: dict[str, str] = {}
        # resname -> its "attach/change" button, so a later scan/clear can refresh it.
        self._itp_buttons: dict[str, QPushButton] = {}
        return panel

    def _scan_components(self) -> None:
        path = self.protein.text().strip()
        # Always drop the previous PDB's results first so a new/invalid PDB never
        # leaves a stale inventory on screen.
        self.inventory_table.setRowCount(0)
        self._itp_buttons.clear()
        self.charge_summary.setVisible(False)
        if not path or not Path(path).is_file():
            self.inventory_status.setText("Pick a Protein PDB first, then Scan loaded PDB.")
            return
        try:
            from chemflow.fe.gromacs.pdb_components import classify_pdb_components
            components = classify_pdb_components(Path(path))
        except Exception as exc:  # pragma: no cover - defensive
            self.inventory_status.setText(f"Could not scan PDB: {exc}")
            return
        # Keep attached ITPs only for residues that still exist in the new PDB.
        present = {c.resname for c in components}
        self._component_itps = {k: v for k, v in self._component_itps.items() if k in present}
        gaps = 0
        has_lipid = False
        for comp in components:
            if comp.kind == "lipid":
                has_lipid = True
            row = self.inventory_table.rowCount()
            self.inventory_table.insertRow(row)
            self.inventory_table.setItem(row, 0, QTableWidgetItem(comp.resname))
            self.inventory_table.setItem(row, 1, QTableWidgetItem(comp.kind))
            self.inventory_table.setItem(row, 2, QTableWidgetItem(str(comp.count)))
            coverage = QTableWidgetItem("✓ covered" if comp.covered else "needs FF")
            self.inventory_table.setItem(row, 3, coverage)
            if comp.covered:
                cell = QTableWidgetItem("force field")
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.inventory_table.setItem(row, 4, cell)
            else:
                gaps += 1
                self.inventory_table.setCellWidget(
                    row, 4, self._itp_cell(comp.resname, comp.suggestion))
        # Only the content columns auto-size; the action column stays stretched.
        for col in range(4):
            self.inventory_table.resizeColumnToContents(col)
        if has_lipid and self.system_type.currentText() != "membrane":
            self.system_type.setCurrentText("membrane")
        msg = (f"{len(components)} components: {len(components) - gaps} covered, "
               f"{gaps} need a force field.")
        if gaps:
            msg += " Attach a prepared ITP for any gap that fails to parameterize."
        self.inventory_status.setText(msg)
        self._update_charge_summary(components)

    def _update_charge_summary(self, components) -> None:
        from chemflow.fe.gromacs.pdb_components import estimate_formal_charge
        est = estimate_formal_charge(components)
        sign = "+" if est.total > 0 else ""
        parts = [f"Estimated system formal charge: {sign}{est.total} e"]
        if est.total == 0:
            parts.append("already neutral")
        elif self.system_type.currentText() == "membrane":
            parts.append("membrane: no ions are added; charge-changing edges "
                         "use far-away bulk water converted to alchemical co-ions")
        elif self.neutralize.isChecked():
            need = abs(est.total)
            ion = "Cl⁻" if est.total > 0 else "Na⁺"
            parts.append(f"neutralization will add ~{need} {ion}"
                         + (" (plus the background ionic strength)"
                            if self.ion_conc.value() > 0 else ""))
        else:
            parts.append("neutralize is OFF — the cell will carry this net charge")
        text = "  ·  ".join(parts) + "."
        if est.nucleic_residues:
            text += (f" Includes {est.nucleic_residues} nucleotide(s) at −1 each "
                     "(approx; ignores a 5′-OH terminus).")
        if est.unknown:
            text += (" Excludes " + ", ".join(est.unknown)
                     + " — add their net charge (SDF / ITP) for the true total.")
        self.charge_summary.setText(text)
        self.charge_summary.setVisible(True)

    def _itp_cell(self, resname: str, suggestion: str):
        cell = QWidget()
        h = QHBoxLayout(cell)
        h.setContentsMargins(6, 3, 6, 3)
        h.setSpacing(6)
        btn = QPushButton("Attach ITP…")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(suggestion)
        btn.setMinimumWidth(120)
        btn.clicked.connect(lambda _=None, r=resname: self._attach_itp(r))
        clear = QPushButton("✕")
        clear.setToolTip("Remove the attached ITP")
        clear.setFixedWidth(26)
        clear.setCursor(Qt.CursorShape.PointingHandCursor)
        clear.setVisible(False)
        clear.clicked.connect(lambda _=None, r=resname: self._clear_itp(r))
        h.addWidget(btn, 1)
        h.addWidget(clear, 0)
        self._itp_buttons[resname] = btn
        btn._clear_button = clear  # type: ignore[attr-defined]
        # Restore the attached state if this component already has an ITP (rescan).
        self._refresh_itp_button(resname)
        return cell

    def _refresh_itp_button(self, resname: str) -> None:
        btn = self._itp_buttons.get(resname)
        if btn is None:
            return
        path = self._component_itps.get(resname)
        clear = getattr(btn, "_clear_button", None)
        if path:
            name = Path(path).name
            btn.setText(f"✓ {name}")
            btn.setToolTip(f"{path}\n(click to change)")
            if clear is not None:
                clear.setVisible(True)
        else:
            btn.setText("Attach ITP…")
            btn.setToolTip("Attach a prepared ITP force field for this component")
            if clear is not None:
                clear.setVisible(False)

    def _attach_itp(self, resname: str) -> None:
        from PySide6.QtWidgets import QFileDialog
        chosen, _ = QFileDialog.getOpenFileName(
            self, f"Attach ITP force field for {resname}", "",
            "GROMACS include topology (*.itp);;All files (*)")
        if not chosen:
            return
        self._component_itps[resname] = chosen
        self._refresh_itp_button(resname)
        self._refresh_added_ff_list()
        if not self.prepare_enabled.isChecked():
            self.prepare_enabled.setChecked(True)
        attached = len(self._component_itps)
        self.inventory_status.setText(
            f"Attached {attached} ITP force field(s). These are included when the "
            "complex is assembled (prepare-system).")

    def _clear_itp(self, resname: str) -> None:
        self._component_itps.pop(resname, None)
        self._refresh_itp_button(resname)
        self._refresh_added_ff_list()
        self.inventory_status.setText(
            f"Removed the ITP for {resname}. "
            f"{len(self._component_itps)} ITP force field(s) attached.")

    def _refresh_added_ff_list(self) -> None:
        """Mirror the attached ITPs into the left-column Added force fields list."""
        self.added_ff_list.clear()
        for resname in sorted(self._component_itps):
            path = self._component_itps[resname]
            item = QListWidgetItem(f"{resname}  —  {Path(path).name}")
            item.setToolTip(path)
            item.setData(Qt.ItemDataRole.UserRole, resname)
            self.added_ff_list.addItem(item)

    def _remove_added_ff(self) -> None:
        item = self.added_ff_list.currentItem()
        if item is None:
            self.inventory_status.setText("Select an added force field to remove.")
            return
        resname = item.data(Qt.ItemDataRole.UserRole)
        self._clear_itp(resname)

    @staticmethod
    def _pair_row(a, b):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(b)
        return row

    def _on_system_type_changed(self, text: str) -> None:
        is_membrane = text == "membrane"
        self.membrane_group.setEnabled(is_membrane)
        # Membrane systems use the prepared box as-is: no neutralizing ions or
        # salt are added. Charge-changing edges instead convert far-away bulk
        # water to alchemical co-ions (count follows the charge change).
        self.neutralize.setEnabled(not is_membrane)
        self.ion_conc.setEnabled(not is_membrane)
        tip = ("Membrane: no ions are added. Charge-changing edges convert bulk "
               "water to alchemical Na+/Cl- co-ions." if is_membrane else "")
        self.neutralize.setToolTip(tip)
        self.ion_conc.setToolTip(tip)
        if hasattr(self, "inventory_table") and self.protein.text().strip():
            self._scan_components()
        # Membrane requires assembling a prepared complex.
        if is_membrane and not self.prepare_enabled.isChecked():
            self.prepare_enabled.setChecked(True)

    def _on_prepare_toggled(self, enabled: bool) -> None:
        for w in (self.protein_ff, self.ligand_ff, self.lipid_ff, self.dna_ff,
                  self.rna_ff, self.ion_params, self.added_ff_list):
            w.setEnabled(enabled)

    def _detect_membrane(self) -> None:
        """Set the system type by scanning the protein PDB for lipid residues."""
        path = self.protein.text().strip()
        if not path or not Path(path).is_file():
            self.status.setText("Pick a protein PDB first, then Detect from PDB.")
            return
        try:
            from chemflow.fe.gromacs.membrane import count_lipid_molecules_pdb
            lipids = count_lipid_molecules_pdb(Path(path))
        except Exception as exc:  # pragma: no cover - defensive
            self.status.setText(f"Could not scan PDB: {exc}")
            return
        if lipids > 0:
            self.system_type.setCurrentText("membrane")
            self.status.setText(
                f"Detected {lipids} lipid molecules → system type set to membrane.")
        else:
            self.system_type.setCurrentText("soluble")
            self.status.setText("No lipids detected → system type set to soluble.")


    # -- 4 · FEP simulation ------------------------------------------------
    def _fep_tab(self):
        self.backend = _combo(["pmx"], "pmx")
        self.charge_method = _combo(["bcc", "user"], "bcc")
        self.charge_change = _combo(["reject", "correction"], "reject")
        self.charge_correction_file = PathField("coions.csv (required for correction)")
        topology = QGroupBox("Topology & charge handling")
        tf = QFormLayout(topology)
        tf.addRow("Backend:", self.backend)
        tf.addRow("Charge method:", self.charge_method)
        tf.addRow("Charge-change method:", self.charge_change)
        tf.addRow("Charge correction file:", self.charge_correction_file)

        self.replicates = _spin(3, 1, 50)
        self.lambda_windows = _spin(21, 2, 200)
        self.timestep = _dspin(0.002, 0.0005, 0.005, 0.0005, 4)
        self.equilibration = _dspin(1.0, 0.0, 100.0, 0.5, 3)
        self.production = _dspin(5.0, 0.0, 1000.0, 0.5, 3)
        self.sampling = _combo(["independent", "hrex"], "independent")
        self.sampling.setToolTip(
            "independent: one task per λ window. hrex: Hamiltonian replica exchange "
            "across all λ windows of an edge/leg/replicate (GROMACS -replex).")
        self.sampling.currentTextChanged.connect(self._on_sampling_changed)
        self.exchange_interval = _spin(500, 1, 100000)
        self.exchange_interval.setToolTip(
            "HREX only: attempt a neighbour replica exchange every N integration "
            "steps (GROMACS -replex).")
        sim = QGroupBox("Simulation")
        sf = QFormLayout(sim)
        sf.addRow("Replicates:", self.replicates)
        sf.addRow("λ windows:", self.lambda_windows)
        sf.addRow("Timestep (ps):", self.timestep)
        sf.addRow("Equilibration (ns):", self.equilibration)
        sf.addRow("Production (ns):", self.production)
        sf.addRow("Sampling method:", self.sampling)
        sf.addRow("HREX exchange interval (steps):", self.exchange_interval)

        inner = QWidget()
        v = QVBoxLayout(inner)
        v.addWidget(topology)
        v.addWidget(sim)
        v.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(inner)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("GROMACS RBFE", "4 · FEP simulation",
                                    "Hybrid-topology backend, charge handling, and the λ schedule."))
        layout.addWidget(scroll, 1)
        self._on_sampling_changed(self.sampling.currentText())
        return page

    def _on_sampling_changed(self, method: str) -> None:
        """Enable HREX-only controls (exchange interval + Job-tab HREX fields)."""
        is_hrex = method == "hrex"
        self.exchange_interval.setEnabled(is_hrex)
        # The Job tab is built after this on first construction; guard with getattr.
        for name in ("hrex_module", "hrex_launcher", "hrex_gmx"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(is_hrex)

    # -- 5 · Job (HPC) -----------------------------------------------------
    def _job_tab(self):
        form = QWidget()
        f = QFormLayout(form)
        self.config_path = PathField("rbfe.toml", mode="save", save_filter="TOML (*.toml)")
        self.config_path.setText("rbfe.toml")
        self.mdrun_options = QLineEdit("-ntmpi 1")
        self.mdrun_options.setToolTip("Space-separated gmx mdrun flags.")
        self.partition = QLineEdit("gpu")
        self.nodes = _spin(1, 1, 64)
        self.cpus = _spin(8, 1, 128)
        self.gpus = _spin(1, 0, 16)
        self.memory = QLineEdit("16G")
        self.walltime = QLineEdit("24:00:00")
        self.conda_init = QLineEdit("")
        self.conda_init.setPlaceholderText("<conda-prefix>/etc/profile.d/conda.sh")
        self.conda_env = QLineEdit("")
        self.conda_env.setPlaceholderText("alchemforge")
        self.hrex_module = QLineEdit("")
        self.hrex_module.setPlaceholderText("gromacs/2026.3 (external MPI GROMACS, hrex only)")
        self.hrex_launcher = QLineEdit("")
        self.hrex_launcher.setPlaceholderText("mpirun (only for hrex)")
        self.hrex_gmx = QLineEdit("")
        self.hrex_gmx.setPlaceholderText("gmx_mpi or full path (external MPI build, hrex only)")
        external_tip = ("HREX (-multidir) cannot run on the conda env's thread-MPI gmx. "
                        "Point this at an external MPI GROMACS build (e.g. gmx_mpi from an "
                        "HPC module, or a full path).")
        self.hrex_module.setToolTip(external_tip)
        self.hrex_gmx.setToolTip(external_tip)
        for label, w in [("Config file to write:", self.config_path),
                         ("mdrun options:", self.mdrun_options),
                         ("SLURM partition:", self.partition),
                         ("Nodes:", self.nodes), ("CPUs per task:", self.cpus),
                         ("GPUs:", self.gpus), ("Memory:", self.memory),
                         ("Walltime:", self.walltime),
                         ("conda init script:", self.conda_init),
                         ("conda environment:", self.conda_env),
                         ("HREX module (external GROMACS):", self.hrex_module),
                         ("HREX launcher:", self.hrex_launcher),
                         ("HREX gmx command (external MPI):", self.hrex_gmx)]:
            f.addRow(label, w)
        return self._wrap(PageHeader("GROMACS RBFE", "5 · Job settings (HPC)",
                                     "Config path, mdrun options, and the SLURM parameters."), form)

    # -- Execution ---------------------------------------------------------
    def _exec_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("GROMACS RBFE", "Execution",
                                    "Write the config, validate, stage, then run. Fix PDB, "
                                    "charge ligands and plan the network on their own tabs."))
        row = QHBoxLayout()
        for label, slot in [("Write config", self._write_config),
                            ("Validate", self._validate),
                            ("Stage", self._stage),
                            ("Run", self._run_workflow)]:
            btn = action_button(label)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        row.addStretch()
        layout.addLayout(row)
        self.status = QLabel("Fill in the tabs, then Write config → Validate → Stage → Run. "
                             "Output appears in the Activity console below.")
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()
        return page

    # -- config assembly ---------------------------------------------------
    def _charge_manifest_for_config(self) -> str | None:
        """User-supplied manifest, else the ligand_manifest.tsv the prep step wrote."""
        explicit = self.charge_manifest.text().strip()
        if explicit:
            return explicit
        written = self._ligands_output_dir() / "ligand_manifest.tsv"
        return str(written) if written.is_file() else None

    def _overrides(self) -> dict:
        is_membrane = self.system_type.currentText() == "membrane"
        prep_on = self.prepare_enabled.isChecked()
        system = {
            "type": self.system_type.currentText(),
            "protein": self.protein.text() or None,
            "force_field": self.force_field.text().strip() or None,
            "water_model": self.water_model.text().strip() or None,
            "gromacs_command": self.gmx.text().strip() or None,
            "box_shape": self.box_shape.currentText(),
            "box_distance": self.box_distance.value(),
            "neutralize": False if is_membrane else self.neutralize.isChecked(),
            "ion_concentration": 0.0 if is_membrane else self.ion_conc.value(),
        }
        if is_membrane:
            system.update({
                "membrane_normal": "z",
                "lipid_resnames": parse_list(self.lipid_resnames.text()) or ["POPC"],
                "minimum_lipids": self.minimum_lipids.value(),
                "barostat": self.membrane_barostat.currentText(),
                "reference_pressure_bar": [self.ref_p_xy.value(), self.ref_p_z.value()],
                "compressibility_bar": [self.compress_xy.value(), self.compress_z.value()],
                "refcoord_scaling": self.refcoord_scaling.currentText(),
            })
        overrides = {
            "project": {"name": self.project_name.text().strip() or None,
                        "work_dir": self.work_dir.text().strip() or None},
            "system": system,
            "ligands": {"library_sdf": self.library_sdf.text() or None,
                        "charge_manifest": self._charge_manifest_for_config()},
            "ligand_prep": {
                "output_directory": self.ligand_output.text().strip() or None,
                "resnames": self._table_resnames() or None,
            },
            "mapping": {
                "network_method": self.network_method.currentText(),
                "atom_mapper": self.atom_mapper.currentText(),
                "element_change": self.element_change.isChecked(),
                "max3d_angstrom": self.max3d.value(),
                "require_cycle_covering": self.require_cycle.isChecked(),
                "radial_hub": self.radial_hub.text().strip() or None,
                "existing_network_dir": (
                    self.existing_network_dir.text().strip()
                    or self.network_dir.text().strip()
                    or None
                ),
            },
            "topology": {
                "backend": self.backend.currentText(),
                "charge_method": self.charge_method.currentText(),
                "charge_change_method": self.charge_change.currentText(),
                "charge_correction_file": self.charge_correction_file.text() or None,
            },
            "simulation": {
                "replicates": self.replicates.value(),
                "lambda_windows": self.lambda_windows.value(),
                "timestep_ps": self.timestep.value(),
                "equilibration_ns": self.equilibration.value(),
                "production_ns": self.production.value(),
                "sampling_method": self.sampling.currentText(),
                "exchange_interval_steps": (
                    self.exchange_interval.value()
                    if self.sampling.currentText() == "hrex" else None
                ),
            },
            "execution": {
                "mdrun_options": parse_list(self.mdrun_options.text()),
                "hrex_module": self.hrex_module.text().strip() or None,
                "hrex_mpi_launcher": self.hrex_launcher.text().strip() or None,
                "hrex_gromacs_command": self.hrex_gmx.text().strip() or None,
            },
            "slurm": {
                "partition": self.partition.text().strip() or None,
                "nodes": self.nodes.value(),
                "cpus_per_task": self.cpus.value(),
                "gpus": self.gpus.value(),
                "memory": self.memory.text().strip() or None,
                "walltime": self.walltime.text().strip() or None,
                "conda_init": self.conda_init.text().strip() or None,
                "conda_environment": self.conda_env.text().strip() or None,
            },
        }
        if prep_on:
            overrides["preparation"] = {
                "enabled": True,
                "protein_force_field": self.protein_ff.currentText(),
                "ligand_force_field": self.ligand_ff.currentText(),
                "lipid_force_field": self.lipid_ff.currentText(),
                "dna_force_field": self.dna_ff.currentText(),
                "rna_force_field": self.rna_ff.currentText(),
                "ion_parameters": self.ion_params.text().strip() or None,
                "prepared_ligand_directory": self.charged_ligands_dir.text().strip() or None,
                "component_itp_files": (
                    sorted(self._component_itps.values())
                    if self._component_itps else None
                ),
            }
        return overrides

    def _config_file(self) -> str | None:
        path = self.config_path.text()
        if not path:
            self.status.setText("Set a config file path on the Job settings tab first.")
            return None
        return str(Path(path).expanduser().resolve())

    def _write_config(self) -> str | None:
        conf = build_config(self._overrides())
        issues = validate_setup(conf)
        path = self._config_file()
        if path is None:
            return None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(to_toml(conf), encoding="utf-8")
        self.status.setText(("Wrote config with warnings: " + " ".join(issues))
                            if issues else f"Wrote config: {path}")
        return path

    def _run(self, argv) -> None:
        if self._console is not None:
            self._console.run_chemflow_gmx(argv)

    def _validate(self) -> None:
        path = self._write_config()
        if not path:
            return
        issues = validate_setup(build_config(self._overrides()))
        if issues:
            self.status.setText("Fix before running: " + " ".join(issues))
            return
        self._run(["ligand-rbfe", "validate", path])

    def _stage(self) -> None:
        path = self._write_config()
        if path:
            self._run(["ligand-rbfe", "stage", path])

    def _run_workflow(self) -> None:
        path = self._write_config()
        if path:
            self._run(["ligand-rbfe", "run", path])
