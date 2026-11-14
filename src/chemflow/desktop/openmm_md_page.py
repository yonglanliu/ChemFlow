"""OpenMM Modeling & MD workflow page for ChemFlow Studio.

A native page for the ``chemflow openmm modeling`` pipeline, migrated from
AlchemForge: build a solvated system (soluble or membrane) -> relax a membrane
system -> run plain MD (minimize -> NVT -> NPT -> production). Each tab writes its
YAML config and runs the matching step through the OpenFE/OpenMM conda env via the
shared Activity console, or writes a single-job SLURM script.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.openmm_md_config import (
    DEFAULT_FORCEFIELDS,
    build_config,
    format_forcefields,
    parse_forcefields,
    slurm_script,
    to_yaml,
)
from chemflow.desktop.widgets.forms import (  # noqa: F811
    NoScrollComboBox as QComboBox,
    NoScrollDoubleSpinBox as QDoubleSpinBox,
    NoScrollSpinBox as QSpinBox,
    PageHeader,
    PathField,
    action_button,
)



def _hpc_gpu_types(module: str) -> list[str]:
    """Selectable --gres GPU types from the shared hpc presets (one list, repo-wide)."""
    try:
        from hpc import presets as _presets

        return _presets.gpu_types(module)
    except Exception:          # hpc not importable -> a safe built-in list
        return ["", "a100", "a100x", "v100", "v100x", "h100", "p100", "a40", "rtx8000"]

def _combo(items, current):
    box = QComboBox()
    box.addItems(items)
    if current in items:
        box.setCurrentText(current)
    return box


# Common small-molecule force fields for ligand/cofactor parameterization
# (openmmforcefields: GAFF via AmberTools, OpenFF/SMIRNOFF via openff-toolkit).
_SMFF_CHOICES = (
    "gaff-2.11",        # default
    "gaff-1.81",
    "openff-2.2.1",
    "openff-2.1.0",     # Sage
    "openff-2.0.0",
    "",                 # none (no ligand to parameterize)
)


def _smff_combo():
    # NOT editable: the theme styles QComboBox::drop-down, so an editable combo
    # renders like a plain text field and only its arrow opens the popup. A plain
    # (non-editable) combo opens the list wherever you click — a real dropdown.
    box = QComboBox()
    box.addItems(_SMFF_CHOICES)
    box.setCurrentText("gaff-2.11")
    box.setToolTip(
        "Small-molecule force field for any ligand/cofactor in the system "
        "(GAFF or OpenFF/SMIRNOFF). Ignored when no ligand SDF is given."
    )
    return box


def _pcm_combo():
    """Partial-charge method for ligands/cofactors."""
    box = QComboBox()
    box.addItems(("auto", "am1bcc", "nagl"))
    box.setCurrentText("auto")
    box.setToolTip(
        "Charges for any ligand/cofactor.\n"
        "auto   = AM1-BCC when AmberTools/OpenEye is available, else the NAGL "
        "GNN surrogate.\n"
        "am1bcc = require true AM1-BCC (needs AmberTools' sqm).\n"
        "nagl   = always use the fast NAGL surrogate.\n"
        "Charges already present in the ligand SDF always take priority."
    )
    return box


def _ecombo(items, current, tip=None):
    """Dropdown of common choices (non-editable, so the whole widget opens the list)."""
    box = QComboBox()
    box.addItems(list(items))
    box.setCurrentText(current)
    if tip:
        box.setToolTip(tip)
    return box


def _form(parent):
    """QFormLayout that degrades gracefully instead of overlapping when cramped.

    WrapLongRows moves the label above its field when the column is too narrow
    (the cause of label/field overlap); expanding fields keep inputs usable.
    """
    f = QFormLayout(parent)
    f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    f.setHorizontalSpacing(12)
    f.setVerticalSpacing(8)
    return f


# Common choices for the build tab's free-text fields.
_SOLVENT_MODELS = ("tip3p", "tip3pfb", "spce", "tip4pew", "tip5p", "opc", "opc3")
_POSITIVE_IONS = ("Na+", "K+", "Li+", "Rb+", "Cs+")
_NEGATIVE_IONS = ("Cl-", "Br-", "F-", "I-")
_LIPID_TYPES = ("POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC")


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


class OpenMMModelingPage(QTabWidget):
    """Build (soluble/membrane) -> relax (membrane) -> MD, via chemflow openmm."""

    def __init__(self, console=None, parent=None):
        super().__init__(parent)
        self._console = console
        self.setDocumentMode(True)
        self.addTab(self._build_tab(), "1 · System build")
        self.addTab(self._relax_tab(), "2 · Relax (membrane)")
        self.addTab(self._md_tab(), "3 · MD")
        self.addTab(self._job_tab(), "4 · Job (HPC)")

    # -- helpers -----------------------------------------------------------
    def _run(self, argv) -> None:
        if self._console is not None:
            self._console.run_chemflow_fe(argv)

    def _ff_field(self, text=None) -> QPlainTextEdit:
        w = QPlainTextEdit(format_forcefields(text or DEFAULT_FORCEFIELDS))
        w.setFixedHeight(88)
        w.setToolTip("One OpenMM force-field XML per line (protein, water, lipids, ions).")
        return w

    def _measure_membrane_center(self) -> None:
        """Read the bilayer midplane out of a structure that still has lipids.

        0 is only right for an OPM-oriented input. A protein lifted out of an
        existing membrane system keeps that system's frame, and building at 0
        puts the lipid patch below the protein instead of through it.
        """
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        start = self.b_input.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Structure containing the bilayer", start,
            "Structures (*.pdb *.cif *.mmcif);;All files (*)")
        if not path:
            return
        try:
            from chemflow.utils.structure import bilayer_midplane, load_structure

            topology, positions = load_structure(path)
            result = bilayer_midplane(topology, positions)
        except Exception as exc:
            QMessageBox.warning(
                self, "Could not measure the midplane",
                f"{Path(path).name}\n\n{exc}")
            return

        self.mem_center_z.setValue(round(result["midplane"], 3))
        QMessageBox.information(
            self, "Bilayer midplane",
            f"{Path(path).name}\n\n"
            f"midplane  {result['midplane']:.3f} nm   (set as Membrane center Z)\n"
            f"leaflets  {result['n_lower']} / {result['n_upper']} lipids at "
            f"{result['lower']:.3f} / {result['upper']:.3f} nm\n"
            f"separation {result['separation']:.2f} nm "
            f"(POPC is ~3.8 nm)\n"
            f"core gap  {result['core_gap']:.2f} nm\n\n"
            f"The mean of all phosphates would have said "
            f"{result['phosphate_mean']:.3f} nm — biased when the leaflets are "
            f"unequal, as here.")

    def _exec_row(self, writer, step, status: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        for label, slot in [
            ("Write config", lambda: self._write_config(writer, step, status)),
            ("Validate", lambda: self._cli(writer, step, "validate", status)),
            ("Run (local)", lambda: self._cli(writer, step, "run", status)),
            ("Write SLURM", lambda: self._write_slurm(writer, step, status)),
            ("Submit (sbatch)", lambda: self._submit_slurm(writer, step, status)),
        ]:
            btn = action_button(label)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        row.addStretch()
        return row

    @staticmethod
    def _resolve_step(step) -> str:
        """``step`` may be a fixed name or a callable (build tab: soluble/membrane)."""
        return step() if callable(step) else step

    def _write_config(self, writer, step, status: QLabel):
        step = self._resolve_step(step)
        path = self._config_path(step)
        if not path:
            status.setText("Set a config file path on the Job (HPC) tab first.")
            return None
        try:
            conf = build_config(step, writer())
        except ValueError as exc:
            status.setText(f"Cannot build config: {exc}")
            return None
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        Path(path).expanduser().resolve().write_text(to_yaml(conf), encoding="utf-8")
        status.setText(f"Wrote config: {path}")
        return str(Path(path).expanduser().resolve())

    def _cli(self, writer, step, verb: str, status: QLabel):
        step = self._resolve_step(step)
        path = self._write_config(writer, step, status)
        if not path:
            return
        from chemflow.desktop.openmm_md_config import STEPS
        subcommand = STEPS[step][1]
        self._run(["openmm", "modeling", subcommand, verb, path])
        status.setText(f"Running: chemflow openmm modeling {subcommand} {verb}")

    def _write_slurm(self, writer, step, status: QLabel):
        step = self._resolve_step(step)
        path = self._write_config(writer, step, status)
        if not path:
            return None
        script = slurm_script(step, path, self._hpc())
        script_path = Path(path).expanduser().with_suffix(".slurm")
        script_path.write_text(script, encoding="utf-8")
        status.setText(f"Wrote SLURM script: {script_path}  (sbatch {script_path})")
        return str(script_path)

    def _submit_slurm(self, writer, step, status: QLabel):
        step = self._resolve_step(step)
        script_path = self._write_slurm(writer, step, status)
        if not script_path or self._console is None:
            return
        job_dir = str(Path(script_path).parent)
        self._console.run_program("sbatch", [Path(script_path).name], cwd=job_dir)

    # -- 1 · System build --------------------------------------------------
    def _build_tab(self):
        self.b_system = _combo(["soluble", "membrane"], "soluble")
        self.b_system.currentTextChanged.connect(self._on_build_system_changed)
        self.b_input = PathField("protein .pdb or .cif (receptor / complex; use .cif if >99,999 atoms)")
        self.b_output = QLineEdit("soluble_complex.pdb")
        self.b_platform = _combo(["CUDA", "OpenCL", "CPU"], "CUDA")
        self.b_ff = self._ff_field()
        self.b_solvent = _ecombo(_SOLVENT_MODELS, "tip3p", "Water model for addSolvent")
        self.b_pos = _ecombo(_POSITIVE_IONS, "Na+", "Positive counter-ion")
        self.b_neg = _ecombo(_NEGATIVE_IONS, "Cl-", "Negative counter-ion")
        self.b_ionic = _dspin(0.15, 0.0, 2.0, 0.01, 3)
        self.b_neutralize = QCheckBox("Neutralize")
        self.b_neutralize.setChecked(True)
        self.b_ligand = PathField("ligand.sdf (optional, if a ligand is in the PDB)")
        self.b_smff = _smff_combo()
        self.b_pcm = _pcm_combo()
        self.b_fix = QCheckBox("Auto-fix structure (rename HIS by protonation, add missing H)")
        self.b_fix.setChecked(True)
        self.b_ph = _dspin(7.0, 0.0, 14.0, 0.5, 1)

        common = QGroupBox("System")
        cf = _form(common)
        cf.addRow("System type:", self.b_system)
        cf.addRow("Input PDB:", self.b_input)
        cf.addRow("Output PDB:", self.b_output)
        cf.addRow("Platform (membrane):", self.b_platform)
        cf.addRow("Solvent model:", self.b_solvent)
        cf.addRow("Positive ion:", self.b_pos)
        cf.addRow("Negative ion:", self.b_neg)
        cf.addRow("Ionic strength (M):", self.b_ionic)
        cf.addRow("", self.b_neutralize)
        cf.addRow("Ligand SDF:", self.b_ligand)
        cf.addRow("Small-molecule FF:", self.b_smff)
        cf.addRow("Partial charges:", self.b_pcm)
        cf.addRow("", self.b_fix)
        cf.addRow("pH (for added H):", self.b_ph)
        cf.addRow("Force fields:", self.b_ff)

        # Soluble box
        self.sol_box_mode = _combo(["padding", "boxsize", "vectors"], "padding")
        self.sol_padding = _dspin(1.0, 0.5, 5.0, 0.1, 2)
        self.sol_box_shape = _combo(["cube", "dodecahedron", "octahedron"], "cube")
        self.sol_boxsize = QLineEdit("")
        self.sol_boxsize.setPlaceholderText("Lx, Ly, Lz (nm) — boxsize mode")
        self.sol_vectors = QLineEdit("")
        self.sol_vectors.setPlaceholderText("9 numbers (full 3×3, nm) — vectors mode")
        self.sol_group = QGroupBox("Soluble box")
        sf = _form(self.sol_group)
        sf.addRow("Box mode:", self.sol_box_mode)
        sf.addRow("Padding (nm):", self.sol_padding)
        sf.addRow("Box shape:", self.sol_box_shape)
        sf.addRow("Box size:", self.sol_boxsize)
        sf.addRow("Box vectors:", self.sol_vectors)

        # Membrane mode selector (build a new patch, or solvate an existing membrane)
        self.mem_mode = _combo(["build", "existing"], "build")
        self.mem_mode.currentTextChanged.connect(self._on_mem_mode_changed)
        self.mem_mode_group = QGroupBox("Membrane mode")
        mmf = _form(self.mem_mode_group)
        mmf.addRow("Mode:", self.mem_mode)
        _mode_help = QLabel(
            "build = tile a lipid patch around the protein and trim it.\n"
            "existing = the input already has a membrane; just solvate the empty "
            "space + add ions (original waters are preserved)."
        )
        _mode_help.setObjectName("Muted")
        _mode_help.setWordWrap(True)
        mmf.addRow("", _mode_help)
        self.mem_mode_group.setVisible(False)

        # Membrane build
        self.mem_lipid = _ecombo(_LIPID_TYPES, "POPC", "Lipid type for the tiled patch")
        self.mem_scale = _dspin(0.6, 0.1, 1.0, 0.05, 2)
        self.mem_center_z = _dspin(0.0, -100.0, 100.0, 0.1, 3)
        self.mem_center_z.setToolTip(
            "Z of the BILAYER MIDPLANE, in the input structure's own frame.\n\n"
            "0 is correct only for an OPM-oriented structure. If the protein came "
            "out of an existing membrane system its coordinates are usually all "
            "positive, and leaving this at 0 packs the lipid patch below the "
            "protein instead of through it — the build then fails with a NaN.\n\n"
            "Use Measure… on a structure that still has its lipids. The midplane "
            "is the midpoint between the two phosphate planes; neither the mean of "
            "all phosphates nor the protein centre is right when the system is "
            "asymmetric.")
        self.mem_center_measure = action_button("Measure…")
        self.mem_center_measure.setToolTip(
            "Pick a PDB/mmCIF that still contains the bilayer and read the midplane "
            "out of its lipid phosphate positions.")
        self.mem_center_measure.clicked.connect(self._measure_membrane_center)
        self._mem_center_row = QWidget()
        _mcz = QHBoxLayout(self._mem_center_row)
        _mcz.setContentsMargins(0, 0, 0, 0)
        _mcz.addWidget(self.mem_center_z, 1)
        _mcz.addWidget(self.mem_center_measure)
        self.mem_xpad = _dspin(1.0, 0.0, 5.0, 0.1, 2)
        self.mem_ypad = _dspin(1.0, 0.0, 5.0, 0.1, 2)
        self.mem_zpad = _dspin(1.0, 0.0, 5.0, 0.1, 2)
        self.mem_zpad.setToolTip(
            "Water beyond the PROTEIN's z extent on each side — the box becomes "
            "protein_z + 2 x this, not bilayer + 2 x this.")
        self.mem_group = QGroupBox("Membrane build (tile + trim a patch)")
        mf = _form(self.mem_group)
        mf.addRow("Lipid type:", self.mem_lipid)
        mf.addRow("Scale factor:", self.mem_scale)
        mf.addRow("Membrane center Z (nm):", self._mem_center_row)
        mf.addRow("X padding (nm):", self.mem_xpad)
        mf.addRow("Y padding (nm):", self.mem_ypad)
        mf.addRow("Z padding (nm):", self.mem_zpad)
        self.mem_group.setVisible(False)

        # Existing membrane: solvate empty space + ions (box preserved by default)
        self.memx_padding = _dspin(1.0, 0.5, 5.0, 0.1, 2)
        self.memx_boxsize = QLineEdit("")
        self.memx_boxsize.setPlaceholderText("Lx, Ly, Lz (nm) — optional; blank = use the input PDB's box")
        self.memx_vectors = QLineEdit("")
        self.memx_vectors.setPlaceholderText("9 numbers (3×3, nm) — optional box override")
        self.memx_pbcxy = QLineEdit("")
        self.memx_align = QCheckBox("Align bilayer normal to Z (fixes a tilted Desmond frame)")
        self.memx_align.setChecked(True)
        self.memx_wcut = _dspin(2.8, 0.0, 10.0, 0.1, 1)
        self.memx_wcut.setToolTip("Newly added water closer than this to a lipid atom is removed.")
        self.memx_pbcxy.setPlaceholderText("Lx, Ly (nm) — optional in-plane override")
        self.memx_group = QGroupBox("Existing membrane — solvate empty space")
        xf = _form(self.memx_group)
        _memx_help = QLabel(
            "The box is read from the input PDB's CRYST1/box by default — fill the "
            "fields below only to override it. Padding is a fallback for the Z length "
            "when no box is otherwise given. Original waters (incl. inside the "
            "membrane) are preserved; only new core waters are removed."
        )
        _memx_help.setObjectName("Muted")
        _memx_help.setWordWrap(True)
        xf.addRow("", _memx_help)
        xf.addRow("Padding (nm, Z fallback):", self.memx_padding)
        xf.addRow("Box size:", self.memx_boxsize)
        xf.addRow("Box vectors:", self.memx_vectors)
        xf.addRow("In-plane (x,y):", self.memx_pbcxy)
        xf.addRow("", self.memx_align)
        xf.addRow("Water-lipid cutoff (A):", self.memx_wcut)
        self.memx_group.setVisible(False)

        self.b_status = QLabel("Choose soluble or membrane, set the PDB, then Write config → Run.")
        self.b_status.setObjectName("Muted")
        self.b_status.setWordWrap(True)

        inner = QWidget()
        v = QVBoxLayout(inner)
        v.addWidget(common)
        v.addWidget(self.sol_group)
        v.addWidget(self.mem_mode_group)
        v.addWidget(self.mem_group)
        v.addWidget(self.memx_group)
        v.addLayout(self._exec_row(self._build_overrides, self._current_build_step, self.b_status))
        v.addWidget(self.b_status)
        v.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(inner)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("OpenMM Modeling", "1 · System build",
                                    "Build a solvated system — soluble (box + ions) or a membrane "
                                    "patch (lipids + water). Output feeds relaxation or MD."))
        layout.addWidget(scroll, 1)
        return page

    def _on_build_system_changed(self, value: str) -> None:
        soluble = value == "soluble"
        self.sol_group.setVisible(soluble)
        self.mem_mode_group.setVisible(not soluble)
        if soluble:
            self.mem_group.setVisible(False)
            self.memx_group.setVisible(False)
        else:
            self._on_mem_mode_changed(self.mem_mode.currentText())
        if self.b_output.text() in ("soluble_complex.pdb", "membrane_complex.pdb"):
            self.b_output.setText("soluble_complex.pdb" if soluble else "membrane_complex.pdb")
        # Rewire the exec row's step by recreating is overkill; the exec buttons
        # read the step from _build_overrides' companion via _current_build_step.

    def _on_mem_mode_changed(self, value: str) -> None:
        if self.b_system.currentText() == "soluble":
            return
        existing = value == "existing"
        self.mem_group.setVisible(not existing)
        self.memx_group.setVisible(existing)

    def _current_build_step(self) -> str:
        return "soluble" if self.b_system.currentText() == "soluble" else "membrane"

    def _parse_triplet(self, text):
        text = text.strip()
        if not text:
            return None
        parts = [p for p in text.replace(",", " ").split()]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return None

    def _build_common(self) -> dict:
        over = {
            "input_pdb": self.b_input.text() or None,
            "output_pdb": self.b_output.text().strip() or None,
            "solvent_model": self.b_solvent.currentText().strip() or None,
            "positive_ion": self.b_pos.currentText().strip() or None,
            "negative_ion": self.b_neg.currentText().strip() or None,
            "ionic_strength": self.b_ionic.value(),
            "neutralize": self.b_neutralize.isChecked(),
            "forcefields": parse_forcefields(self.b_ff.toPlainText()),
        }
        if self.b_ligand.text():
            over["ligand_sdf"] = self.b_ligand.text()
        if self.b_smff.currentText().strip():
            over["small_molecule_forcefield"] = self.b_smff.currentText().strip()
        over["partial_charge_method"] = self.b_pcm.currentText()
        over["fix_structure"] = self.b_fix.isChecked()
        over["add_missing_hydrogens"] = self.b_fix.isChecked()
        over["ph"] = self.b_ph.value()
        return over

    def _build_overrides(self) -> dict:
        over = self._build_common()
        if self._current_build_step() == "soluble":
            box = {"mode": self.sol_box_mode.currentText(),
                   "padding": self.sol_padding.value(),
                   "box_shape": self.sol_box_shape.currentText()}
            bs = self._parse_triplet(self.sol_boxsize.text())
            if bs and len(bs) == 3:
                box["boxsize"] = bs
            vec = self._parse_triplet(self.sol_vectors.text())
            if vec and len(vec) == 9:
                box["box_vectors"] = [vec[0:3], vec[3:6], vec[6:9]]
            over["box"] = box
        else:
            over["platform"] = self.b_platform.currentText()
            if self.mem_mode.currentText() == "existing":
                over["mode"] = "existing"
                existing = {"padding": self.memx_padding.value()}
                bs = self._parse_triplet(self.memx_boxsize.text())
                if bs and len(bs) == 3:
                    existing["boxsize"] = bs
                vec = self._parse_triplet(self.memx_vectors.text())
                if vec and len(vec) == 9:
                    existing["box_vectors"] = [vec[0:3], vec[3:6], vec[6:9]]
                xy = self._parse_triplet(self.memx_pbcxy.text())
                if xy and len(xy) == 2:
                    existing["pbc_x"] = xy[0]
                    existing["pbc_y"] = xy[1]
                over["existing"] = existing
                over["align_membrane"] = self.memx_align.isChecked()
                over["water_lipid_cutoff"] = self.memx_wcut.value() / 10.0  # A -> nm
            else:
                over["mode"] = "build"
                over["build"] = {
                    "lipid_type": self.mem_lipid.currentText().strip() or "POPC",
                    "scale_factor": self.mem_scale.value(),
                    "membrane_center_z": self.mem_center_z.value(),
                    "x_padding": self.mem_xpad.value(),
                    "y_padding": self.mem_ypad.value(),
                    "z_padding": self.mem_zpad.value(),
                }
        return over

    # -- 2 · Relax (membrane) ---------------------------------------------
    def _relax_tab(self):
        self.r_input = PathField("membrane complex .pdb or .cif (use .cif if >99,999 atoms)")
        self.r_protocol = _combo(["full", "equilibrated"], "full")
        self.r_protocol.setToolTip(
            "full = 6-stage cold-start ladder (freshly built membrane).\n"
            "equilibrated = minimization -> NVT -> NPgT only, for an already-equilibrated "
            "membrane (e.g. after re-solvation/wrapping)."
        )
        self.r_platform = _combo(["CUDA", "OpenCL", "CPU"], "CUDA")
        self.r_friction = _dspin(1.0, 0.1, 10.0, 0.1, 2)
        self.r_seed = _spin(1, 0, 10_000_000)
        self.r_mini = _spin(10000, 1, 1_000_000)
        self.r_ff = self._ff_field()
        self.r_ligand = PathField("ligand.sdf (optional)")
        self.r_smff = _smff_combo()
        self.r_pcm = _pcm_combo()
        self.r_protocol.currentTextChanged.connect(self._on_relax_protocol_changed)

        # Equilibrated (2-stage) parameters — shown only when protocol == equilibrated.
        self.re_nvt_time = _dspin(250.0, 1.0, 100000.0, 50.0, 1)
        self.re_nvt_restrain = QCheckBox("Restrain solute (backbone + ligand) during NVT")
        self.re_nvt_restrain.setChecked(True)
        self.re_nprt_time = _dspin(2000.0, 1.0, 1000000.0, 100.0, 1)
        self.r_equil_group = QGroupBox("Equilibrated protocol (NVT → NPγT)")
        ef = _form(self.r_equil_group)
        ef.addRow("NVT time (ps):", self.re_nvt_time)
        ef.addRow("", self.re_nvt_restrain)
        ef.addRow("NPγT time (ps):", self.re_nprt_time)
        self.r_equil_group.setVisible(False)
        form = QWidget()
        f = _form(form)
        f.addRow("Input PDB:", self.r_input)
        f.addRow("Protocol:", self.r_protocol)
        f.addRow("Platform:", self.r_platform)
        f.addRow("Friction (1/ps):", self.r_friction)
        f.addRow("Seed:", self.r_seed)
        f.addRow("Minimization steps:", self.r_mini)
        f.addRow("Ligand SDF:", self.r_ligand)
        f.addRow("Small-molecule FF:", self.r_smff)
        f.addRow("Partial charges:", self.r_pcm)
        f.addRow("Force fields:", self.r_ff)

        self.r_status = QLabel("Gradual, restrained heating/relaxation of a membrane system. "
                               "The 6-stage restraint schedule uses sensible defaults — edit the "
                               "written YAML for fine control.")
        self.r_status.setObjectName("Muted")
        self.r_status.setWordWrap(True)

        inner = QWidget()
        v = QVBoxLayout(inner)
        v.addWidget(form)
        v.addWidget(self.r_equil_group)
        v.addLayout(self._exec_row(self._relax_overrides, "relax", self.r_status))
        v.addWidget(self.r_status)
        v.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(inner)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("OpenMM Modeling", "2 · Relax (membrane)",
                                    "Multi-stage restrained equilibration that gently relaxes a "
                                    "freshly built membrane system before production MD."))
        layout.addWidget(scroll, 1)
        return page

    def _on_relax_protocol_changed(self, value: str) -> None:
        self.r_equil_group.setVisible(value == "equilibrated")

    def _relax_overrides(self) -> dict:
        over = {
            "input_pdb": self.r_input.text() or None,
            "protocol": self.r_protocol.currentText(),
            "platform": self.r_platform.currentText(),
            "friction": self.r_friction.value(),
            "seed": self.r_seed.value(),
            "forcefields": parse_forcefields(self.r_ff.toPlainText()),
            "minimization": {"steps": self.r_mini.value()},
        }
        if self.r_ligand.text():
            over["ligand_sdf"] = self.r_ligand.text()
        if self.r_smff.currentText().strip():
            over["small_molecule_forcefield"] = self.r_smff.currentText().strip()
        over["partial_charge_method"] = self.r_pcm.currentText()
        if self.r_protocol.currentText() == "equilibrated":
            nvt = {"simulation_time": self.re_nvt_time.value()}
            if not self.re_nvt_restrain.isChecked():
                nvt["restraints"] = False
            over["equil_nvt"] = nvt
            over["equil_nprt"] = {"simulation_time": self.re_nprt_time.value()}
        return over

    # -- 3 · MD ------------------------------------------------------------
    def _md_tab(self):
        self.m_input = PathField("relaxed/solvated complex .pdb or .cif (must carry a periodic box)")
        self.m_system = _combo(["soluble", "membrane"], "soluble")
        self.m_platform = _combo(["CUDA", "OpenCL", "CPU"], "CUDA")
        self.m_friction = _dspin(1.0, 0.1, 10.0, 0.1, 2)
        self.m_seed = _spin(1, 0, 10_000_000)
        self.m_temp = _dspin(300.0, 0.0, 1000.0, 1.0, 1)
        self.m_pressure = _dspin(1.0, 0.0, 1000.0, 0.5, 2)
        self.m_surface = _dspin(0.0, 0.0, 10000.0, 10.0, 1)
        self.m_report = _spin(5000, 1, 10_000_000)
        self.m_checkpoint = _spin(50000, 1, 10_000_000)
        self.m_prefix = QLineEdit("md")
        self.m_ff = self._ff_field()
        self.m_ligand = PathField("ligand.sdf (optional)")
        self.m_smff = _smff_combo()
        self.m_pcm = _pcm_combo()

        self.m_mini = _spin(10000, 1, 1_000_000)
        self.m_nvt_dt = _dspin(0.002, 0.0005, 0.005, 0.0005, 4)
        self.m_nvt_t = _dspin(250.0, 0.0, 1_000_000.0, 50.0, 1)
        self.m_npt_dt = _dspin(0.002, 0.0005, 0.005, 0.0005, 4)
        self.m_npt_t = _dspin(500.0, 0.0, 1_000_000.0, 50.0, 1)
        self.m_prod_dt = _dspin(0.004, 0.0005, 0.005, 0.0005, 4)
        self.m_prod_t = _dspin(10000.0, 0.0, 100_000_000.0, 1000.0, 1)

        general = QGroupBox("Ensemble & output")
        gf = _form(general)
        gf.addRow("Input PDB:", self.m_input)
        gf.addRow("System type:", self.m_system)
        gf.addRow("Platform:", self.m_platform)
        gf.addRow("Friction (1/ps):", self.m_friction)
        gf.addRow("Seed:", self.m_seed)
        gf.addRow("Temperature (K):", self.m_temp)
        gf.addRow("Pressure (bar):", self.m_pressure)
        gf.addRow("Surface tension (bar·nm, membrane):", self.m_surface)
        gf.addRow("Report interval (steps):", self.m_report)
        gf.addRow("Checkpoint interval (steps):", self.m_checkpoint)
        gf.addRow("Output prefix:", self.m_prefix)
        gf.addRow("Ligand SDF:", self.m_ligand)
        gf.addRow("Small-molecule FF:", self.m_smff)
        gf.addRow("Partial charges:", self.m_pcm)
        gf.addRow("Force fields:", self.m_ff)

        stages = QGroupBox("Stages (ps)")
        sf = _form(stages)
        sf.addRow("Minimization steps:", self.m_mini)
        sf.addRow("NVT timestep (ps):", self.m_nvt_dt)
        sf.addRow("NVT time (ps):", self.m_nvt_t)
        sf.addRow("NPT timestep (ps):", self.m_npt_dt)
        sf.addRow("NPT time (ps):", self.m_npt_t)
        sf.addRow("Production timestep (ps):", self.m_prod_dt)
        sf.addRow("Production time (ps):", self.m_prod_t)

        self.m_status = QLabel("Minimize → NVT → NPT → production. Input must carry a periodic box "
                               "(CRYST1). 0.004 ps = 4 fs (with hydrogen-mass repartitioning).")
        self.m_status.setObjectName("Muted")
        self.m_status.setWordWrap(True)

        inner = QWidget()
        v = QVBoxLayout(inner)
        # Stacked, not side-by-side: two group boxes sharing a narrow panel squeezed
        # each other until labels and fields overlapped. The scroll area handles height.
        v.addWidget(general)
        v.addWidget(stages)
        v.addLayout(self._exec_row(self._md_overrides, "md", self.m_status))
        v.addWidget(self.m_status)
        v.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(inner)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("OpenMM Modeling", "3 · Molecular dynamics",
                                    "Unrestrained production MD for a prepared system."))
        layout.addWidget(scroll, 1)
        return page

    def _md_overrides(self) -> dict:
        over = {
            "input_pdb": self.m_input.text() or None,
            "system_type": self.m_system.currentText(),
            "platform": self.m_platform.currentText(),
            "friction": self.m_friction.value(),
            "seed": self.m_seed.value(),
            "temperature": self.m_temp.value(),
            "pressure": self.m_pressure.value(),
            "surface_tension": self.m_surface.value(),
            "report_interval": self.m_report.value(),
            "checkpoint_interval": self.m_checkpoint.value(),
            "output_prefix": self.m_prefix.text().strip() or "md",
            "forcefields": parse_forcefields(self.m_ff.toPlainText()),
            "minimization": {"steps": self.m_mini.value()},
            "nvt": {"timestep": self.m_nvt_dt.value(), "simulation_time": self.m_nvt_t.value()},
            "npt": {"timestep": self.m_npt_dt.value(), "simulation_time": self.m_npt_t.value()},
            "production": {"timestep": self.m_prod_dt.value(),
                           "simulation_time": self.m_prod_t.value()},
        }
        if self.m_ligand.text():
            over["ligand_sdf"] = self.m_ligand.text()
        if self.m_smff.currentText().strip():
            over["small_molecule_forcefield"] = self.m_smff.currentText().strip()
        over["partial_charge_method"] = self.m_pcm.currentText()
        return over

    # -- 4 · Job (HPC) -----------------------------------------------------
    def _job_tab(self):
        self.cfg_soluble = PathField("soluble.yaml", mode="save", save_filter="YAML (*.yaml *.yml)")
        self.cfg_soluble.setText("openmm_soluble.yaml")
        self.cfg_membrane = PathField("membrane.yaml", mode="save", save_filter="YAML (*.yaml *.yml)")
        self.cfg_membrane.setText("openmm_membrane.yaml")
        self.cfg_relax = PathField("relax.yaml", mode="save", save_filter="YAML (*.yaml *.yml)")
        self.cfg_relax.setText("openmm_relax.yaml")
        self.cfg_md = PathField("md.yaml", mode="save", save_filter="YAML (*.yaml *.yml)")
        self.cfg_md.setText("openmm_md.yaml")

        self.workdir = PathField("base dir — each step's files go to <workdir>/<jobname>/",
                                 mode="directory")
        self.workdir.setText(".")
        self.jobname = QLineEdit("md")
        self.jobname.setToolTip("Project folder name. Each step's config and SLURM script "
                                "are written under <workdir>/<jobname>/.")

        self.partition = QLineEdit("gpu")
        self.hpc_preset = QComboBox()
        try:
            from chemflow.desktop.openfe_rbfe_config import single_gpu_preset_names
            _preset_names = single_gpu_preset_names("openmm")
        except Exception:
            _preset_names = []
        self.hpc_preset.addItems(["(custom)", *_preset_names])
        self.hpc_preset.setToolTip("Pick an OpenMM preset (src/hpc/presets/openmm.yaml, or the openmm: section of env/hpc.yaml) "
                                   "to fill the SLURM fields below; '(custom)' keeps your own values.")
        self.hpc_preset.currentTextChanged.connect(self._apply_hpc_preset)
        self.cpus = _spin(8, 1, 128)
        self.gpus = _spin(1, 0, 16)
        self.gpu_type = QComboBox()
        self.gpu_type.setEditable(True)
        self.gpu_type.addItems(_hpc_gpu_types("openmm"))
        self.gpu_type.setCurrentText("")
        self.memory = QLineEdit("32G")
        self.walltime = QLineEdit("48:00:00")
        self.modules = QLineEdit("")
        self.modules.setPlaceholderText("module load names, e.g. cuda/12.4 (HPC)")
        self.conda_init = QLineEdit("")
        self.conda_init.setPlaceholderText("<conda-prefix>/etc/profile.d/conda.sh")
        self.conda_env = QLineEdit("")
        self.conda_env.setPlaceholderText("alchemforge")

        form = QWidget()
        f = _form(form)
        for label, w in [("Work dir (base):", self.workdir),
                         ("Job name:", self.jobname),
                         ("Soluble config:", self.cfg_soluble),
                         ("Membrane config:", self.cfg_membrane),
                         ("Relax config:", self.cfg_relax),
                         ("MD config:", self.cfg_md),
                         ("HPC preset:", self.hpc_preset),
                         ("SLURM partition:", self.partition),
                         ("CPUs per task:", self.cpus), ("GPUs:", self.gpus),
                         ("GPU type (--gres):", self.gpu_type),
                         ("Memory:", self.memory), ("Walltime:", self.walltime),
                         ("Modules (module load):", self.modules),
                         ("conda init script:", self.conda_init),
                         ("conda environment:", self.conda_env)]:
            f.addRow(label, w)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("OpenMM Modeling", "4 · Job settings (HPC)",
                                    "Where each step's config is written, and the SLURM parameters "
                                    "for the Write-SLURM buttons on each tab."))
        from chemflow.desktop.openfe_rbfe_config import HPC_JOB_HELP_HTML
        from chemflow.desktop.widgets.forms import with_help
        layout.addWidget(with_help(form, HPC_JOB_HELP_HTML))
        layout.addStretch()
        return page

    def _root(self) -> str:
        """Project folder: <workdir>/<jobname>."""
        base = self.workdir.text().strip() or "."
        job = self.jobname.text().strip() or "md"
        return str(Path(base) / job)

    def _config_path(self, step: str):
        field = {
            "soluble": self.cfg_soluble, "membrane": self.cfg_membrane,
            "relax": self.cfg_relax, "md": self.cfg_md,
        }[step]
        name = field.text()
        if not name:
            return None
        # Each step's config (and its .slurm) lives in the project folder.
        return str((Path(self._root()) / Path(name).name).expanduser().resolve())

    def _apply_hpc_preset(self, name: str) -> None:
        """Fill the SLURM fields from a named env/hpc.yaml preset ('(custom)' = no-op)."""
        if not name or name == "(custom)":
            return
        try:
            from chemflow.desktop.openfe_rbfe_config import apply_slurm_preset
            merged = apply_slurm_preset(self._hpc(), name, "openmm")
        except Exception:
            return
        self.partition.setText(str(merged.get("partition", "gpu")))
        self.cpus.setValue(int(merged.get("cpus_per_task", 8) or 8))
        self.gpus.setValue(int(merged.get("gpus", 1) or 0))
        self.gpu_type.setCurrentText(str(merged.get("gpu_type", "") or ""))
        self.memory.setText(str(merged.get("memory", "32G")))
        self.walltime.setText(str(merged.get("walltime", "48:00:00")))
        if merged.get("conda_environment"):
            self.conda_env.setText(str(merged["conda_environment"]))

    def _hpc(self) -> dict:
        return {
            "partition": self.partition.text().strip(),
            "cpus_per_task": self.cpus.value(),
            "gpus": self.gpus.value(),
            "gpu_type": self.gpu_type.currentText().strip(),
            "memory": self.memory.text().strip(),
            "walltime": self.walltime.text().strip(),
            "modules": self.modules.text().strip(),
            "conda_init": self.conda_init.text().strip(),
            "conda_environment": self.conda_env.text().strip(),
        }
