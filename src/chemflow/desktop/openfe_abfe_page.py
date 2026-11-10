"""OpenFE ABFE workflow page for ChemFlow Studio — the absolute-BFE setup that
mirrors AlchemForge's OpenFE ABFE config, driving the ``alchemforge openfe abfe``
CLI.

ABFE decouples ONE ligand, so there is no perturbation network. Sub-tabs:
Ligand Prep (charge the ligand + cofactor) → System (soluble or membrane; a
membrane needs explicit box vectors) → Restraints → FEP simulation (alchemical +
per-leg lambda schedules) → Job Setting (HPC) → Execution. The form writes one
create_json YAML; Execution runs validate / generate (create_json) / local run
through the shared Activity console, or writes a SLURM ``openfe quickrun`` array.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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

from chemflow.desktop.openfe_abfe_config import (
    build_abfe_lambda_schedule,
    build_config,
    format_box_vectors,
    format_float_list,
    parse_box_vectors,
    parse_float_list,
    slurm_script,
    to_yaml,
    validate_setup,
)
from chemflow.desktop.openfe_rbfe_config import HPC_JOB_HELP_HTML
from chemflow.desktop.widgets.forms import (  # noqa: F811
    NoScrollComboBox as QComboBox,
    NoScrollDoubleSpinBox as QDoubleSpinBox,
    NoScrollSpinBox as QSpinBox,
    PageHeader,
    PathField,
    action_button,
    with_help,
)


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


class OpenFEAbfeWorkflowPage(QTabWidget):
    """Five-step OpenFE ABFE setup (charge → system → restraints → FEP → job)."""

    def __init__(self, console=None, parent=None):
        super().__init__(parent)
        self._console = console
        self.setDocumentMode(True)
        self.addTab(self._ligand_tab(), "1 · Ligand Prep")
        self.addTab(self._system_tab(), "2 · System")
        self.addTab(self._restraints_tab(), "3 · Restraints")
        self.addTab(self._fep_tab(), "4 · FEP simulation")
        self.addTab(self._job_tab(), "5 · Job (HPC)")
        self.addTab(self._exec_tab(), "Execution")

    # -- helpers -----------------------------------------------------------
    def _wrap(self, header, form_widget):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(header)
        layout.addWidget(form_widget)
        layout.addStretch()
        return page

    def _run(self, argv) -> None:
        if self._console is not None:
            self._console.run_chemflow_fe(argv)

    # -- 1 · Ligand Prep ---------------------------------------------------
    def _ligand_tab(self):
        form = QWidget()
        f = QFormLayout(form)
        self.ligand_sdf = PathField("ligand.sdf (exactly one molecule for ABFE)")
        self.cofactor_sdf = PathField("cofactor.sdf (optional)")
        self.off_backend = _combo(["ambertools", "openeye", "rdkit"], "ambertools")
        self.charge_method = _combo(["am1bcc", "am1bccelf10", "espaloma", "nagl"], "am1bcc")
        self.skip_charge = QCheckBox("Skip charge assignment (use stored SDF charges — faster)")
        self.skip_charge.setToolTip("When the ligand/cofactor SDFs already carry partial charges, "
                                    "skip the charge check/generation at Setup FEP (create_json).")
        self.charge_cores = _spin(4, 1, 64)
        self.charged_out = PathField("charged_ligand.sdf (output)", mode="save",
                                     save_filter="SDF (*.sdf)")
        self.charged_cofactor_out = PathField("charged_cofactors.sdf (output, optional)",
                                              mode="save", save_filter="SDF (*.sdf)")
        self.overwrite_charges = QCheckBox("Overwrite existing partial charges")
        self.overwrite_charges.setChecked(True)
        self.smff = QLineEdit("openff-2.1.0")
        f.addRow("Ligand SDF:", self.ligand_sdf)
        f.addRow("Cofactor SDF:", self.cofactor_sdf)
        f.addRow("Charge toolkit backend:", self.off_backend)
        f.addRow("Partial charge method:", self.charge_method)
        f.addRow("Charging processes:", self.charge_cores)
        f.addRow("Charged ligand output:", self.charged_out)
        f.addRow("Charged cofactor output:", self.charged_cofactor_out)
        f.addRow("", self.overwrite_charges)
        f.addRow("", self.skip_charge)
        f.addRow("Small-molecule force field:", self.smff)

        row = QHBoxLayout()
        calc = action_button("Calculate ligand charges")
        calc.clicked.connect(self._calc_charges)
        calc_cof = action_button("Calculate cofactor charges")
        calc_cof.clicked.connect(self._calc_cofactor_charges)
        refresh = action_button("Refresh table")
        refresh.clicked.connect(self._refresh_ligand_table)
        row.addWidget(calc)
        row.addWidget(calc_cof)
        row.addWidget(refresh)
        row.addStretch()

        self.ligand_table = QTableWidget(0, 3)
        self.ligand_table.setHorizontalHeaderLabels(["Ligand", "Charge (before)", "Charge (after)"])
        self.ligand_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.ligand_table.setMinimumWidth(460)
        _hdr = self.ligand_table.horizontalHeader()
        _hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        _hdr.setMinimumSectionSize(120)
        self.ligand_table.itemSelectionChanged.connect(self._on_ligand_selected)
        self.ligand_sdf.edit.textChanged.connect(self._on_ligand_sdf_changed)

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
        layout.addWidget(PageHeader("OpenFE ABFE", "1 · Ligand preparation",
                                    "Charge the ligand (exactly one for ABFE) and any cofactor — "
                                    "run it directly here. Click a ligand to see its structure."))
        layout.addLayout(split, 1)
        self.charge_status = QLabel("Set the ligand SDF, choose a method, then Calculate charges.")
        self.charge_status.setObjectName("Muted")
        self.charge_status.setWordWrap(True)
        layout.addWidget(self.charge_status)
        self._refresh_ligand_table()
        return page

    def _on_ligand_sdf_changed(self, _text=None) -> None:
        """New ligand SDF → auto-set charged output to ``<stem>_charged.sdf``
        (always overwrite) and refresh the ligand table."""
        src = self.ligand_sdf.text().strip()
        if src:
            self.charged_out.setText(str(Path(src).with_name(Path(src).stem + "_charged.sdf")))
        self._refresh_ligand_table()

    def _on_ligand_selected(self) -> None:
        rows = self.ligand_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.ligand_table.item(rows[0].row(), 0)
        if item is None:
            return
        name = item.text()
        source = self._active_ligand_sdf()
        if not source:
            return
        try:
            from chemflow.fe.openfe.charge_report import molecule_svg
            svg = molecule_svg(source, name)
        except Exception as exc:
            self.charge_status.setText(f"Could not draw {name}: {exc}")
            return
        body = svg or f"<p style='color:#888'>No depiction for {name}</p>"
        self.structure_view.setHtml(
            f"<html><body style='margin:0;display:flex;align-items:center;"
            f"justify-content:center;height:100%'>{body}</body></html>")

    def _charge_argv(self, source: str, out: str) -> list[str]:
        argv = ["openfe", "charge", "--ligands", source,
                "--method", self.charge_method.currentText(),
                "--backend", self.off_backend.currentText(),
                "--cores", str(self.charge_cores.value())]
        if not self.overwrite_charges.isChecked():
            argv.append("--keep-charges")
        argv += ["--out", out]
        return argv

    def _calc_charges(self) -> None:
        source = self.ligand_sdf.text()
        if not source:
            self.charge_status.setText("Set the ligand SDF first.")
            return
        out = self.charged_out.text()
        if not out:
            out = str(Path(source).expanduser().with_name(Path(source).stem + "_charged.sdf"))
            self.charged_out.setText(out)
        self._run(self._charge_argv(source, out))
        self._refresh_ligand_table()
        self.charge_status.setText(
            "Charging the ligand… when it finishes, click “Refresh table” to see the "
            "after-charges. The charged SDF is used downstream.")

    def _calc_cofactor_charges(self) -> None:
        source = self.cofactor_sdf.text()
        if not source:
            self.charge_status.setText("Set the cofactor SDF first.")
            return
        out = self.charged_cofactor_out.text()
        if not out:
            out = str(Path(source).expanduser().with_name(Path(source).stem + "_charged.sdf"))
            self.charged_cofactor_out.setText(out)
        self._run(self._charge_argv(source, out))
        self.charge_status.setText("Charging the cofactor… the charged cofactor SDF is used downstream.")

    def _active_ligand_sdf(self) -> str:
        """Prefer the charged output if it exists, else the original ligand SDF."""
        charged = self.charged_out.text()
        if charged and Path(charged).expanduser().is_file():
            return charged
        return self.ligand_sdf.text()

    def _active_cofactor_sdf(self) -> str:
        charged = self.charged_cofactor_out.text()
        if charged and Path(charged).expanduser().is_file():
            return charged
        return self.cofactor_sdf.text()

    def _refresh_ligand_table(self) -> None:
        src = self.ligand_sdf.text()
        if not src or not Path(src).expanduser().is_file():
            return
        try:
            from chemflow.fe.openfe.charge_report import formal_charges, partial_charge_sums
            before = formal_charges(src)
            charged = self.charged_out.text()
            after = (partial_charge_sums(charged)
                     if charged and Path(charged).expanduser().is_file() else {})
        except Exception as exc:
            self.charge_status.setText(f"Could not read ligands: {exc}")
            return
        self.ligand_table.setRowCount(len(before))
        for r, (name, formal) in enumerate(before):
            aft = after.get(name)
            self.ligand_table.setItem(r, 0, QTableWidgetItem(name))
            self.ligand_table.setItem(r, 1, QTableWidgetItem(f"{formal:+d}"))
            self.ligand_table.setItem(r, 2, QTableWidgetItem("—" if aft is None else f"{aft:+.3f}"))

    # -- 2 · System --------------------------------------------------------
    def _system_tab(self):
        form = QWidget()
        f = QFormLayout(form)
        self.protein_pdb = PathField("complex.pdb (receptor / complex)")
        self.system_ligand_sdf = PathField("charged ligand SDF (defaults to Ligand Prep output)")
        self.system_cofactor_sdf = PathField("charged cofactor SDF (optional)")
        # Cofactor-from-template: one charged template SDF placed on every matching
        # residue in the (complex) protein PDB. When set, cofactor SDF is ignored.
        self.cofactor_template_sdf = PathField(
            "cofactor_template.sdf (optional; one charged copy, placed from the complex PDB)")
        self.cofactor_resnames = QLineEdit()
        self.cofactor_resnames.setPlaceholderText("cofactor PDB resname(s), e.g. 4DY  (comma-separated)")
        self.cofactor_resnames.setToolTip(
            "PDB residue name(s) of the cofactor in the protein PDB. With a template "
            "SDF, each matching residue is built from the template (identical charges) "
            "and stripped from the protein before parameterization.")
        self.membrane = QCheckBox("Membrane system (requires explicit box vectors)")
        self.membrane.toggled.connect(self._on_membrane_toggle)
        self.box_vectors = QPlainTextEdit()
        self.box_vectors.setPlaceholderText("x, y, z  (nm, diagonal box)  — or 9 numbers for a full 3×3")
        self.box_vectors.setFixedHeight(64)
        self.box_vectors.setEnabled(False)
        self.box_browse = QPushButton("Browse pbc.txt")
        self.box_browse.setToolTip("Load periodic box from a .txt file "
                                   "(comma/space separated: 3 numbers x,y,z or 9 for a full 3×3)")
        self.box_browse.setEnabled(False)
        self.box_browse.clicked.connect(self._browse_pbc)
        box_container = QWidget()
        box_layout = QVBoxLayout(box_container)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.addWidget(self.box_vectors)
        box_layout.addWidget(self.box_browse)
        self.box_shape = _combo(["dodecahedron", "cube", "octahedron"], "dodecahedron")
        self.complex_pad = _dspin(1.5, 0.5, 5.0, 0.1, 2)
        self.solvent_pad = _dspin(1.5, 0.5, 5.0, 0.1, 2)
        self.solvent_model = QLineEdit("tip3p")
        self.ion_conc = _dspin(0.15, 0.0, 2.0, 0.01, 3)
        self.pos_ion = QLineEdit("Na+")
        self.neg_ion = QLineEdit("Cl-")
        self.neutralize = QCheckBox("Neutralize net charge with counter-ions")
        self.neutralize.setChecked(True)
        f.addRow("Protein PDB:", self.protein_pdb)
        f.addRow("Charged ligand SDF:", self.system_ligand_sdf)
        f.addRow("Charged cofactor SDF:", self.system_cofactor_sdf)
        f.addRow("Cofactor template SDF (optional):", self.cofactor_template_sdf)
        f.addRow("Cofactor resname(s):", self.cofactor_resnames)
        f.addRow("", self.membrane)
        f.addRow("Complex box vectors (membrane):", box_container)
        f.addRow("Box shape (soluble + solvent leg):", self.box_shape)
        f.addRow("Complex solvation padding (nm):", self.complex_pad)
        f.addRow("Solvent solvation padding (nm):", self.solvent_pad)
        f.addRow("Solvent model:", self.solvent_model)
        f.addRow("Ionic strength (M):", self.ion_conc)
        f.addRow("Positive ion:", self.pos_ion)
        f.addRow("Negative ion:", self.neg_ion)
        f.addRow("", self.neutralize)

        self.pbc_note = QLabel("Soluble system: box is padded automatically (box shape used).")
        self.pbc_note.setObjectName("Muted")
        self.pbc_note.setWordWrap(True)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("OpenFE ABFE", "2 · System",
                                    "Protein complex and solvation. A membrane system needs explicit "
                                    "box vectors for a valid periodic box."))
        layout.addWidget(form)
        layout.addWidget(self.pbc_note)
        layout.addStretch()
        return page

    def _apply_hpc_preset(self, name: str) -> None:
        """Fill the SLURM fields from a named env/hpc.yaml preset ('(custom)' = no-op)."""
        if not name or name == "(custom)":
            return
        try:
            from chemflow.desktop.openfe_rbfe_config import apply_slurm_preset
            merged = apply_slurm_preset(self._hpc(), name)
        except Exception:
            return
        self.partition.setText(str(merged.get("partition", "gpu")))
        self.nodes.setValue(int(merged.get("nodes", 1) or 1))
        self.ntasks.setValue(int(merged.get("ntasks", 1) or 1))
        self.cpus.setValue(int(merged.get("cpus_per_task", 16) or 16))
        self.gpus.setValue(int(merged.get("gpus", 1) or 0))
        self.gpu_type.setCurrentText(str(merged.get("gpu_type", "") or ""))
        self.memory.setText(str(merged.get("memory", "32G")))
        self.walltime.setText(str(merged.get("walltime", "240:00:00")))
        if merged.get("conda_environment"):
            self.conda_env.setText(str(merged["conda_environment"]))

    def _on_membrane_toggle(self, on: bool) -> None:
        self.box_vectors.setEnabled(on)
        self.box_browse.setEnabled(on)
        # Membrane uses explicit box vectors; the box shape/padding are for the
        # soluble complex (and always the solvent leg).
        self.box_shape.setEnabled(not on)
        self.complex_pad.setEnabled(not on)
        self.pbc_note.setText(
            "Membrane system: set complex box vectors (nm) so the periodic box is valid. "
            "The solvent leg still uses the box shape + solvent padding."
            if on else "Soluble system: box is padded automatically (box shape used).")

    def _browse_pbc(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose PBC file", "", "Text/PBC (*.txt);;All files (*)")
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            self.pbc_note.setText(f"Could not read PBC file: {path}")
            return
        self.box_vectors.setPlainText(text)

    # -- 3 · Restraints ----------------------------------------------------
    def _restraints_tab(self):
        form = QWidget()
        f = QFormLayout(form)
        self.anchor_strategy = _combo(["multi-residue", "bonded"], "multi-residue")
        self.host_max = _dspin(2.0, 0.1, 10.0, 0.1, 2)
        self.host_min = _dspin(0.3, 0.0, 5.0, 0.1, 2)
        self.host_selection = _combo(["backbone", "all"], "backbone")
        self.rmsf_cutoff = _dspin(0.1, 0.0, 2.0, 0.05, 3)
        self.spring_constant = _dspin(1000.0, 0.0, 100000.0, 50.0, 1)
        f.addRow("Anchor finding strategy:", self.anchor_strategy)
        f.addRow("Host max distance (nm):", self.host_max)
        f.addRow("Host min distance (nm):", self.host_min)
        f.addRow("Host selection:", self.host_selection)
        f.addRow("RMSF cutoff (nm):", self.rmsf_cutoff)
        f.addRow("Solvent restraint spring constant (kJ/mol/nm²):", self.spring_constant)
        return self._wrap(PageHeader("OpenFE ABFE", "3 · Restraints",
                                     "Boresch-style host restraint search and the solvent-leg "
                                     "restraint spring constant."), form)

    # -- 4 · FEP simulation ------------------------------------------------
    def _group(self, title: str, rows) -> QGroupBox:
        box = QGroupBox(title)
        f = QFormLayout(box)
        for label, widget in rows:
            f.addRow(label, widget)
        return box

    def _fep_tab(self):
        # General / thermodynamics + force field
        self.protocol_repeats = _spin(1, 1, 50)
        self.pressure = _dspin(1.0, 0.0, 1000.0, 0.5, 2)
        self.temperature = _dspin(310.0, 0.0, 1000.0, 1.0, 1)
        self.platform = _combo(["CUDA", "OpenCL", "CPU"], "CUDA")
        self.nb_method = _combo(["PME", "CutoffPeriodic", "CutoffNonPeriodic"], "PME")
        self.nb_cutoff = _dspin(1.0, 0.5, 2.0, 0.1, 2)
        self.rigid_water = QCheckBox("Rigid water")
        self.rigid_water.setChecked(True)
        self.hmass = _dspin(3.0, 1.0, 4.0, 0.5, 2)

        general = self._group("General / thermodynamics", [
            ("Protocol repeats:", self.protocol_repeats),
            ("Pressure (bar):", self.pressure),
            ("Temperature (K):", self.temperature),
            ("Platform:", self.platform),
            ("Nonbonded method:", self.nb_method),
            ("Nonbonded cutoff (nm):", self.nb_cutoff),
            ("", self.rigid_water),
            ("Hydrogen mass (Da):", self.hmass),
        ])

        # Alchemical
        self.disable_disp = QCheckBox("Disable alchemical dispersion correction")
        self.annihilate_sterics = QCheckBox("Annihilate sterics")
        self.softcore_a = _dspin(1.0, 0.0, 10.0, 0.1, 3)
        self.softcore_b = _dspin(1.0, 0.0, 10.0, 0.1, 3)
        self.softcore_c = _dspin(6.0, 0.0, 20.0, 0.5, 3)
        self.softcore_alpha = _dspin(0.5, 0.0, 2.0, 0.05, 3)
        alchemical = self._group("Alchemical", [
            ("", self.disable_disp),
            ("", self.annihilate_sterics),
            ("Softcore a:", self.softcore_a),
            ("Softcore b:", self.softcore_b),
            ("Softcore c:", self.softcore_c),
            ("Softcore alpha:", self.softcore_alpha),
        ])

        # Equilibration
        self.c_eq_mini = _spin(5000, 0, 1000000)
        self.c_eq_len = _dspin(1.0, 0.0, 100.0, 0.1, 3)
        self.c_eq_nvt = _dspin(1.0, 0.0, 100.0, 0.1, 3)
        self.c_eq_prod = _dspin(2.0, 0.0, 1000.0, 0.5, 3)
        self.s_eq_mini = _spin(5000, 0, 1000000)
        self.s_eq_len = _dspin(1.0, 0.0, 100.0, 0.1, 3)
        self.s_eq_nvt = _dspin(1.0, 0.0, 100.0, 0.1, 3)
        self.s_eq_prod = _dspin(2.0, 0.0, 1000.0, 0.5, 3)
        equil = self._group("Equilibration", [
            ("Complex mini steps:", self.c_eq_mini),
            ("Complex equil length (ns):", self.c_eq_len),
            ("Complex equil NVT length (ns):", self.c_eq_nvt),
            ("Complex equil production (ns):", self.c_eq_prod),
            ("Solvent mini steps:", self.s_eq_mini),
            ("Solvent equil length (ns):", self.s_eq_len),
            ("Solvent equil NVT length (ns):", self.s_eq_nvt),
            ("Solvent equil production (ns):", self.s_eq_prod),
        ])

        # Production simulation
        self.early_term = _dspin(0.0, 0.0, 100.0, 0.1, 3)
        self.c_sim_mini = _spin(5000, 0, 1000000)
        self.c_sim_eq = _dspin(1.0, 0.0, 100.0, 0.5, 3)
        self.c_sim_prod = _dspin(10.0, 0.0, 1000.0, 1.0, 3)
        self.s_sim_mini = _spin(5000, 0, 1000000)
        self.s_sim_eq = _dspin(1.0, 0.0, 100.0, 0.5, 3)
        self.s_sim_prod = _dspin(10.0, 0.0, 1000.0, 1.0, 3)
        simulation = self._group("Production simulation", [
            ("Early-termination target error (kcal/mol):", self.early_term),
            ("Complex mini step:", self.c_sim_mini),
            ("Complex equil length (ns):", self.c_sim_eq),
            ("Complex production length (ns):", self.c_sim_prod),
            ("Solvent mini step:", self.s_sim_mini),
            ("Solvent equil length (ns):", self.s_sim_eq),
            ("Solvent production length (ns):", self.s_sim_prod),
        ])

        # Integrator
        self.c_timestep = _dspin(4.0, 0.5, 5.0, 0.5, 2)
        self.s_timestep = _dspin(4.0, 0.5, 5.0, 0.5, 2)
        self.barostat_freq = _dspin(25.0, 1.0, 1000.0, 1.0, 1)
        self.surface_tension = _dspin(300.0, 0.0, 10000.0, 10.0, 1)
        integrator = self._group("Integrator", [
            ("Complex timestep (fs):", self.c_timestep),
            ("Solvent timestep (fs):", self.s_timestep),
            ("Barostat frequency (steps):", self.barostat_freq),
            ("Surface tension (bar·nm, membrane):", self.surface_tension),
        ])

        # Lambda schedules
        defaults = build_config()
        self.lam_c_elec = QLineEdit(format_float_list(defaults["complex_lambda_elec"]))
        self.lam_c_vdw = QLineEdit(format_float_list(defaults["complex_lambda_vdw"]))
        self.lam_c_rest = QLineEdit(format_float_list(defaults["complex_lambda_restraints"]))
        self.lam_s_elec = QLineEdit(format_float_list(defaults["solvent_lambda_elec"]))
        self.lam_s_vdw = QLineEdit(format_float_list(defaults["solvent_lambda_vdw"]))
        self.lam_s_rest = QLineEdit(format_float_list(defaults["solvent_lambda_restraints"]))
        lam = self._group("Lambda schedules (each leg's lists must share one length)", [
            ("Complex λ elec:", self.lam_c_elec),
            ("Complex λ vdw:", self.lam_c_vdw),
            ("Complex λ restraints:", self.lam_c_rest),
            ("Solvent λ elec:", self.lam_s_elec),
            ("Solvent λ vdw:", self.lam_s_vdw),
            ("Solvent λ restraints:", self.lam_s_rest),
        ])

        # Build the schedules from per-phase window counts instead of typing them.
        self.n_c_rest = _spin(6, 1, 100)
        self.n_c_elec = _spin(11, 2, 100)
        self.n_c_vdw = _spin(13, 2, 100)
        self.n_s_elec = _spin(5, 2, 100)
        self.n_s_vdw = _spin(9, 2, 100)
        build_box = QGroupBox("Build λ windows from a function (staged decoupling)")
        bf = QFormLayout(build_box)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("restraints")); crow.addWidget(self.n_c_rest)
        crow.addWidget(QLabel("elec")); crow.addWidget(self.n_c_elec)
        crow.addWidget(QLabel("vdw")); crow.addWidget(self.n_c_vdw)
        crow.addStretch()
        srow = QHBoxLayout()
        srow.addWidget(QLabel("elec")); srow.addWidget(self.n_s_elec)
        srow.addWidget(QLabel("vdw")); srow.addWidget(self.n_s_vdw)
        srow.addStretch()
        crow_w = QWidget(); crow_w.setLayout(crow)
        srow_w = QWidget(); srow_w.setLayout(srow)
        bf.addRow("Complex windows:", crow_w)
        bf.addRow("Solvent windows:", srow_w)
        build_btn = action_button("Build λ schedules")
        build_btn.setToolTip("Generate the six lambda lists: complex ramps restraints → elec → vdw; "
                             "solvent ramps elec → vdw (no restraints).")
        build_btn.clicked.connect(self._build_lambdas)
        bf.addRow("", build_btn)
        self.lambda_note = QLabel("Complex: restraints on, then decouple elec, then vdw. "
                                  "Solvent: decouple elec, then vdw.")
        self.lambda_note.setObjectName("Muted")
        self.lambda_note.setWordWrap(True)
        bf.addRow("", self.lambda_note)

        inner = QWidget()
        grid = QVBoxLayout(inner)
        top = QHBoxLayout()
        col1 = QVBoxLayout()
        col1.addWidget(general)
        col1.addWidget(alchemical)
        col1.addWidget(integrator)
        col2 = QVBoxLayout()
        col2.addWidget(equil)
        col2.addWidget(simulation)
        top.addLayout(col1, 1)
        top.addLayout(col2, 1)
        grid.addLayout(top)
        grid.addWidget(build_box)
        grid.addWidget(lam)
        grid.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("OpenFE ABFE", "4 · FEP simulation",
                                    "Alchemical settings, equilibration/production lengths, integrator "
                                    "and the per-leg lambda schedules."))
        layout.addWidget(scroll, 1)
        return page

    def _build_lambdas(self) -> None:
        """Fill the six lambda lists from the per-phase window counts."""
        c = build_abfe_lambda_schedule("complex", self.n_c_rest.value(),
                                       self.n_c_elec.value(), self.n_c_vdw.value())
        s = build_abfe_lambda_schedule("solvent", 0,
                                       self.n_s_elec.value(), self.n_s_vdw.value())
        self.lam_c_elec.setText(format_float_list(c["complex_lambda_elec"]))
        self.lam_c_vdw.setText(format_float_list(c["complex_lambda_vdw"]))
        self.lam_c_rest.setText(format_float_list(c["complex_lambda_restraints"]))
        self.lam_s_elec.setText(format_float_list(s["solvent_lambda_elec"]))
        self.lam_s_vdw.setText(format_float_list(s["solvent_lambda_vdw"]))
        self.lam_s_rest.setText(format_float_list(s["solvent_lambda_restraints"]))
        n_c = len(c["complex_lambda_elec"])
        n_s = len(s["solvent_lambda_elec"])
        self.lambda_note.setText(
            f"Built {n_c} complex windows and {n_s} solvent windows. "
            "Complex: restraints on, then decouple elec, then vdw. "
            "Solvent: decouple elec, then vdw.")

    # -- 5 · Job (HPC) -----------------------------------------------------
    def _job_tab(self):
        form = QWidget()
        f = QFormLayout(form)
        self.name = QLineEdit("abfe_my_ligand")
        self.name.setToolTip("Transformation name — the output JSON basename.")
        self.config_path = PathField("abfe.yaml (written into <workdir>/<jobname>/)", mode="save",
                                     save_filter="YAML (*.yaml *.yml)")
        self.config_path.setText("abfe.yaml")
        self.workdir = PathField("base dir — project goes to <workdir>/<jobname>/",
                                 mode="directory")
        self.workdir.setText(".")
        self.jobname = QLineEdit("abfe")
        self.jobname.setToolTip("Project folder name. The config, inputs/, json/ and "
                                "results/ all live under <workdir>/<jobname>/.")
        self.partition = QLineEdit("gpu")
        self.hpc_preset = QComboBox()
        try:
            from chemflow.desktop.openfe_rbfe_config import single_gpu_preset_names
            _preset_names = single_gpu_preset_names()
        except Exception:
            _preset_names = []
        self.hpc_preset.addItems(["(custom)", *_preset_names])
        self.hpc_preset.setToolTip("Pick a preset from env/hpc.yaml (e.g. singlenode-singlegpu-long) "
                                   "to fill the SLURM fields below; '(custom)' keeps your own values.")
        self.hpc_preset.currentTextChanged.connect(self._apply_hpc_preset)
        self.nodes = _spin(1, 1, 64)
        self.ntasks = _spin(1, 1, 256)
        self.cpus = _spin(16, 1, 128)
        self.gpus = _spin(1, 0, 16)
        self.gpu_type = QComboBox()
        self.gpu_type.setEditable(True)
        self.gpu_type.addItems(["", "a100", "a100x", "v100", "v100x", "h100", "p100", "a40", "rtx8000"])
        self.gpu_type.setCurrentText("")
        self.gpu_type.setToolTip("GPU type for --gres=gpu:<type>:N (leave blank for any GPU). "
                                 "Editable — type your cluster's name if not listed.")
        self.replicates = _spin(3, 1, 64)
        self.replicates.setToolTip("Repeats per transformation. Array size = N_transformations × replicates.")
        self.array = QLineEdit("")
        self.array.setPlaceholderText("auto: transformations × replicates (override only if needed)")
        self.memory = QLineEdit("32G")
        self.walltime = QLineEdit("240:00:00")
        self.modules = QLineEdit("")
        self.modules.setPlaceholderText("module load names, e.g. cuda/12.4 gcc (HPC)")
        self.conda_init = QLineEdit("")
        self.conda_init.setPlaceholderText("<conda-prefix>/etc/profile.d/conda.sh")
        self.conda_env = QLineEdit("")
        self.conda_env.setPlaceholderText("alchemforge")
        for label, w in [("Transformation name:", self.name),
                         ("Config file name:", self.config_path),
                         ("Work dir (base):", self.workdir), ("Job name:", self.jobname),
                         ("HPC preset:", self.hpc_preset),
                         ("SLURM partition:", self.partition),
                         ("Nodes:", self.nodes), ("Tasks:", self.ntasks),
                         ("CPUs per task:", self.cpus), ("GPUs:", self.gpus),
                         ("GPU type (--gres):", self.gpu_type),
                         ("Replicates per edge:", self.replicates),
                         ("Array range (--array):", self.array),
                         ("Memory:", self.memory), ("Walltime:", self.walltime),
                         ("Modules (module load):", self.modules),
                         ("conda init script:", self.conda_init),
                         ("conda environment:", self.conda_env)]:
            f.addRow(label, w)
        return self._wrap(PageHeader("OpenFE ABFE", "5 · Job settings (HPC)",
                                     "Output locations and the SLURM parameters for submission. "
                                     "results/ and json/ are created under the work dir."),
                          with_help(form, HPC_JOB_HELP_HTML))

    # -- Execution ---------------------------------------------------------
    def _exec_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("OpenFE ABFE", "Execution",
                                    "Write the config, validate, generate the transformation JSON, "
                                    "then submit."))
        row = QHBoxLayout()
        for label, slot in [("Write config", self._write_config),
                            ("Validate", self._validate),
                            ("Generate JSON (create_json)", self._generate),
                            ("Write SLURM script", self._write_slurm),
                            ("Submit (sbatch)", self._submit_slurm),
                            ("Submit (local)", self._submit_local)]:
            btn = action_button(label)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        row.addStretch()
        layout.addLayout(row)
        self.status = QLabel("Fill in the tabs, then Write config → Validate → Generate JSON → Submit. "
                             "Output appears in the Activity console below.")
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()
        return page

    # -- config assembly ---------------------------------------------------
    def _overrides(self) -> dict:
        return {
            "name": self.name.text().strip() or None,
            "ligand_sdf": (self.system_ligand_sdf.text() or self._active_ligand_sdf()) or None,
            "cofactor_sdf": (self.system_cofactor_sdf.text() or self._active_cofactor_sdf()) or None,
            "cofactor_template_sdf": self.cofactor_template_sdf.text() or None,
            "cofactor_resnames": (
                [r.strip() for r in self.cofactor_resnames.text().replace(",", " ").split() if r.strip()]
                or None
            ),
            "protein_pdb": self.protein_pdb.text() or None,
            "off_toolkit_backend": self.off_backend.currentText(),
            "partial_charge_method": self.charge_method.currentText(),
            "skip_charge_assignment": self.skip_charge.isChecked(),
            "small_molecule_forcefield": self.smff.text().strip(),
            # System
            "membrane_system": self.membrane.isChecked(),
            "complex_box_vectors": parse_box_vectors(self.box_vectors.toPlainText()),
            "box_shape": self.box_shape.currentText(),
            "complex_solvation_padding": self.complex_pad.value(),
            "solvent_solvation_padding": self.solvent_pad.value(),
            "solvent_model": self.solvent_model.text().strip(),
            "ion_concentration": self.ion_conc.value(),
            "positive_ion": self.pos_ion.text().strip() or "Na+",
            "negative_ion": self.neg_ion.text().strip() or "Cl-",
            "neutralize": self.neutralize.isChecked(),
            # Restraints
            "anchor_finding_strategy": self.anchor_strategy.currentText(),
            "host_max_distance": self.host_max.value(),
            "host_min_distance": self.host_min.value(),
            "host_selection": self.host_selection.currentText(),
            "rmsf_cutoff": self.rmsf_cutoff.value(),
            "solvent_restraint_spring_constant": self.spring_constant.value(),
            # General / thermo / FF
            "protocol_repeats": self.protocol_repeats.value(),
            "pressure": self.pressure.value(),
            "temperature": self.temperature.value(),
            "platform": self.platform.currentText(),
            "nonbonded_method": self.nb_method.currentText(),
            "nonbonded_cutoff": self.nb_cutoff.value(),
            "rigid_water": self.rigid_water.isChecked(),
            "hydrogen_mass": self.hmass.value(),
            # Alchemical
            "disable_alchemical_dispersion_correction": self.disable_disp.isChecked(),
            "annihilate_sterics": self.annihilate_sterics.isChecked(),
            "softcore_a": self.softcore_a.value(),
            "softcore_b": self.softcore_b.value(),
            "softcore_c": self.softcore_c.value(),
            "softcore_alpha": self.softcore_alpha.value(),
            # Equilibration
            "complex_equil_mini_steps": self.c_eq_mini.value(),
            "complex_equil_equil_length": self.c_eq_len.value(),
            "complex_equil_equil_length_nvt": self.c_eq_nvt.value(),
            "complex_equil_production_length": self.c_eq_prod.value(),
            "solvent_equil_mini_steps": self.s_eq_mini.value(),
            "solvent_equil_equil_length": self.s_eq_len.value(),
            "solvent_equil_equil_length_nvt": self.s_eq_nvt.value(),
            "solvent_equil_production_length": self.s_eq_prod.value(),
            # Simulation
            "early_termination_target_error": self.early_term.value(),
            "complex_sim_mini_step": self.c_sim_mini.value(),
            "complex_sim_equil_length": self.c_sim_eq.value(),
            "complex_sim_prod_length": self.c_sim_prod.value(),
            "solvent_sim_mini_step": self.s_sim_mini.value(),
            "solvent_sim_equil_length": self.s_sim_eq.value(),
            "solvent_sim_prod_length": self.s_sim_prod.value(),
            # Integrator
            "complex_timestep": self.c_timestep.value(),
            "solvent_timestep": self.s_timestep.value(),
            "barostat_frequency": self.barostat_freq.value(),
            "surface_tension": self.surface_tension.value(),
            # Lambda schedules
            "complex_lambda_elec": parse_float_list(self.lam_c_elec.text()),
            "complex_lambda_vdw": parse_float_list(self.lam_c_vdw.text()),
            "complex_lambda_restraints": parse_float_list(self.lam_c_rest.text()),
            "solvent_lambda_elec": parse_float_list(self.lam_s_elec.text()),
            "solvent_lambda_vdw": parse_float_list(self.lam_s_vdw.text()),
            "solvent_lambda_restraints": parse_float_list(self.lam_s_rest.text()),
            # Output
            "workdir": self.workdir.text().strip() or ".",
            "jobname": self.jobname.text().strip() or "abfe",
            "result_dir": self._result_dir(),
            "json_dir": self._json_dir(),
        }

    def _hpc(self) -> dict:
        return {
            "partition": self.partition.text().strip(),
            "nodes": self.nodes.value(),
            "ntasks": self.ntasks.value(),
            "cpus_per_task": self.cpus.value(),
            "gpus": self.gpus.value(),
            "gpu_type": self.gpu_type.currentText().strip(),
            "replicates": self.replicates.value(),
            "array": self.array.text().strip(),
            "memory": self.memory.text().strip(),
            "walltime": self.walltime.text().strip(),
            "modules": self.modules.text().strip(),
            "conda_init": self.conda_init.text().strip(),
            "conda_environment": self.conda_env.text().strip(),
        }

    def _root(self) -> str:
        """Project folder: <workdir>/<jobname>."""
        base = self.workdir.text().strip() or "."
        job = self.jobname.text().strip() or "abfe"
        return str(Path(base) / job)

    def _workdir(self) -> str:
        return self.workdir.text().strip() or "."

    def _result_dir(self) -> str:
        return str(Path(self._root()) / "results")

    def _json_dir(self) -> str:
        return str(Path(self._root()) / "json")

    def _config_path(self) -> str | None:
        name = self.config_path.text().strip()
        if not name:
            self.status.setText("Set a config file name on the Job settings tab first.")
            return None
        # The config always lives inside the project folder (<workdir>/<jobname>/).
        return str((Path(self._root()) / Path(name).name).expanduser().resolve())

    def _write_config(self) -> str | None:
        try:
            conf = build_config(self._overrides())
        except ValueError as exc:
            self.status.setText(f"Cannot build config: {exc}")
            return None
        conf["workdir"] = str(Path(self._workdir()).expanduser().resolve())
        issues = validate_setup(conf)
        path = self._config_path()
        if path is None:
            return None
        try:
            from chemflow.fe.openfe.workflow_defaults import prepare_project
            prepare_project(
                conf,
                default_jobname=self.jobname.text().strip() or "abfe",
                optional_file_attrs=[
                    ("protein_pdb", "protein PDB"),
                    ("ligand_sdf", "ligand"),
                    ("cofactor_sdf", "cofactor"),
                    ("cofactor_template_sdf", "cofactor template"),
                ],
            )
        except (OSError, ValueError) as exc:
            self.status.setText(f"Could not stage inputs: {exc}")
            return None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(to_yaml(conf), encoding="utf-8")
        if issues:
            self.status.setText("Wrote config with warnings: " + " ".join(issues))
        else:
            self.status.setText(f"Wrote config + staged inputs under: {Path(path).parent}")
        return path

    def _validate(self) -> None:
        path = self._write_config()
        if not path:
            return
        conf = build_config(self._overrides())
        issues = validate_setup(conf)
        if issues:
            self.status.setText("Fix before running: " + " ".join(issues))
            return
        self._run(["openfe", "abfe", "validate", path])

    def _generate(self) -> None:
        path = self._write_config()
        if path:
            self._run(["openfe", "abfe", "generate", path])

    def _submit_local(self) -> None:
        path = self._write_config()
        if path:
            self._run(["openfe", "abfe", "run", path])

    def _write_slurm(self) -> str | None:
        path = self._write_config()
        if not path:
            return None
        conf = build_config(self._overrides())
        conf["workdir"] = str(Path(self._workdir()).expanduser().resolve())
        from chemflow.fe.openfe.workflow_defaults import resolve_layout
        layout = resolve_layout(conf, default_jobname=self.jobname.text().strip() or "abfe")
        conf["json_dir"] = str(layout["json"])
        conf["result_dir"] = str(layout["results"])
        hpc = self._hpc()
        replicates = max(1, int(hpc.get("replicates", 1)))
        n_json = self._count_json_files()
        if n_json > 0:
            total = n_json * replicates
            hpc["array"] = f"0-{total - 1}"
            note = (f"--array auto-set to 0-{total - 1} "
                    f"({n_json} transformations × {replicates} replicates = {total} tasks).")
        elif hpc.get("array"):
            note = f"--array kept as {hpc['array']} (no JSONs found yet to auto-size)."
        else:
            hpc["array"] = "0-0"
            note = ("No transformation JSONs found yet — run Generate JSON (create_json) first, "
                    "then re-write the SLURM script to auto-size --array.")
        script = slurm_script(path, hpc, conf)
        script_path = Path(path).expanduser().with_suffix(".slurm")
        script_path.write_text(script, encoding="utf-8")
        self.status.setText(f"Wrote SLURM array script: {script_path}. {note} "
                            f"Submit: sbatch {script_path}")
        return str(script_path)

    def _submit_slurm(self) -> None:
        """Write the SLURM script into the job dir and sbatch it from there."""
        script_path = self._write_slurm()
        if not script_path:
            return
        if self._console is None:
            return
        job_dir = str(Path(script_path).parent)
        self._console.run_program("sbatch", [Path(script_path).name], cwd=job_dir)

    def _count_json_files(self) -> int:
        try:
            json_dir = Path(self._json_dir()).expanduser()
            if not json_dir.is_absolute():
                json_dir = Path.cwd() / json_dir
            return len(list(json_dir.glob("*.json"))) if json_dir.is_dir() else 0
        except OSError:
            return 0
