"""OpenMM Modeling & MD workflow page for ChemFlow Studio.

A native page for the ``chemflow openmm modeling`` pipeline, migrated from
AlchemForge: build a solvated system (soluble or membrane) -> relax a membrane
system -> run plain MD (minimize -> NVT -> NPT -> production). Each tab writes its
YAML config and runs the matching step through the OpenFE/OpenMM conda env via the
shared Activity console, or writes a single-job SLURM script.
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
        self.b_input = PathField("protein.pdb (receptor / complex)")
        self.b_output = QLineEdit("soluble_complex.pdb")
        self.b_platform = _combo(["CUDA", "OpenCL", "CPU"], "CUDA")
        self.b_ff = self._ff_field()
        self.b_solvent = QLineEdit("tip3p")
        self.b_pos = QLineEdit("Na+")
        self.b_neg = QLineEdit("Cl-")
        self.b_ionic = _dspin(0.15, 0.0, 2.0, 0.01, 3)
        self.b_neutralize = QCheckBox("Neutralize")
        self.b_neutralize.setChecked(True)
        self.b_ligand = PathField("ligand.sdf (optional, if a ligand is in the PDB)")
        self.b_smff = QLineEdit("")
        self.b_smff.setPlaceholderText("gaff-2.11  or  openff-2.1.0 (if ligand present)")

        common = QGroupBox("System")
        cf = QFormLayout(common)
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
        sf = QFormLayout(self.sol_group)
        sf.addRow("Box mode:", self.sol_box_mode)
        sf.addRow("Padding (nm):", self.sol_padding)
        sf.addRow("Box shape:", self.sol_box_shape)
        sf.addRow("Box size:", self.sol_boxsize)
        sf.addRow("Box vectors:", self.sol_vectors)

        # Membrane build
        self.mem_lipid = QLineEdit("POPC")
        self.mem_scale = _dspin(0.6, 0.1, 1.0, 0.05, 2)
        self.mem_center_z = _dspin(0.0, -10.0, 10.0, 0.5, 2)
        self.mem_xpad = _dspin(1.0, 0.0, 5.0, 0.1, 2)
        self.mem_ypad = _dspin(1.0, 0.0, 5.0, 0.1, 2)
        self.mem_zpad = _dspin(1.0, 0.0, 5.0, 0.1, 2)
        self.mem_group = QGroupBox("Membrane build (tile + trim a patch)")
        mf = QFormLayout(self.mem_group)
        mf.addRow("Lipid type:", self.mem_lipid)
        mf.addRow("Scale factor:", self.mem_scale)
        mf.addRow("Membrane center Z (nm):", self.mem_center_z)
        mf.addRow("X padding (nm):", self.mem_xpad)
        mf.addRow("Y padding (nm):", self.mem_ypad)
        mf.addRow("Z padding (nm):", self.mem_zpad)
        self.mem_group.setVisible(False)

        self.b_status = QLabel("Choose soluble or membrane, set the PDB, then Write config → Run.")
        self.b_status.setObjectName("Muted")
        self.b_status.setWordWrap(True)

        inner = QWidget()
        v = QVBoxLayout(inner)
        v.addWidget(common)
        v.addWidget(self.sol_group)
        v.addWidget(self.mem_group)
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
        self.mem_group.setVisible(not soluble)
        if self.b_output.text() in ("soluble_complex.pdb", "membrane_complex.pdb"):
            self.b_output.setText("soluble_complex.pdb" if soluble else "membrane_complex.pdb")
        # Rewire the exec row's step by recreating is overkill; the exec buttons
        # read the step from _build_overrides' companion via _current_build_step.

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
            "solvent_model": self.b_solvent.text().strip() or None,
            "positive_ion": self.b_pos.text().strip() or None,
            "negative_ion": self.b_neg.text().strip() or None,
            "ionic_strength": self.b_ionic.value(),
            "neutralize": self.b_neutralize.isChecked(),
            "forcefields": parse_forcefields(self.b_ff.toPlainText()),
        }
        if self.b_ligand.text():
            over["ligand_sdf"] = self.b_ligand.text()
        if self.b_smff.text().strip():
            over["small_molecule_forcefield"] = self.b_smff.text().strip()
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
            over["build"] = {
                "lipid_type": self.mem_lipid.text().strip() or "POPC",
                "scale_factor": self.mem_scale.value(),
                "membrane_center_z": self.mem_center_z.value(),
                "x_padding": self.mem_xpad.value(),
                "y_padding": self.mem_ypad.value(),
                "z_padding": self.mem_zpad.value(),
            }
        return over

    # -- 2 · Relax (membrane) ---------------------------------------------
    def _relax_tab(self):
        self.r_input = PathField("membrane_complex.pdb (built membrane system)")
        self.r_platform = _combo(["CUDA", "OpenCL", "CPU"], "CUDA")
        self.r_friction = _dspin(1.0, 0.1, 10.0, 0.1, 2)
        self.r_seed = _spin(1, 0, 10_000_000)
        self.r_mini = _spin(10000, 1, 1_000_000)
        self.r_ff = self._ff_field()
        self.r_ligand = PathField("ligand.sdf (optional)")
        self.r_smff = QLineEdit("")
        self.r_smff.setPlaceholderText("gaff-2.11 or openff-2.1.0 (if ligand present)")
        form = QWidget()
        f = QFormLayout(form)
        f.addRow("Input PDB:", self.r_input)
        f.addRow("Platform:", self.r_platform)
        f.addRow("Friction (1/ps):", self.r_friction)
        f.addRow("Seed:", self.r_seed)
        f.addRow("Minimization steps:", self.r_mini)
        f.addRow("Ligand SDF:", self.r_ligand)
        f.addRow("Small-molecule FF:", self.r_smff)
        f.addRow("Force fields:", self.r_ff)

        self.r_status = QLabel("Gradual, restrained heating/relaxation of a membrane system. "
                               "The 6-stage restraint schedule uses sensible defaults — edit the "
                               "written YAML for fine control.")
        self.r_status.setObjectName("Muted")
        self.r_status.setWordWrap(True)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(PageHeader("OpenMM Modeling", "2 · Relax (membrane)",
                                    "Multi-stage restrained equilibration that gently relaxes a "
                                    "freshly built membrane system before production MD."))
        layout.addWidget(form)
        layout.addLayout(self._exec_row(self._relax_overrides, "relax", self.r_status))
        layout.addWidget(self.r_status)
        layout.addStretch()
        return page

    def _relax_overrides(self) -> dict:
        over = {
            "input_pdb": self.r_input.text() or None,
            "platform": self.r_platform.currentText(),
            "friction": self.r_friction.value(),
            "seed": self.r_seed.value(),
            "forcefields": parse_forcefields(self.r_ff.toPlainText()),
            "minimization": {"steps": self.r_mini.value()},
        }
        if self.r_ligand.text():
            over["ligand_sdf"] = self.r_ligand.text()
        if self.r_smff.text().strip():
            over["small_molecule_forcefield"] = self.r_smff.text().strip()
        return over

    # -- 3 · MD ------------------------------------------------------------
    def _md_tab(self):
        self.m_input = PathField("relaxed/solvated complex .pdb (with a periodic box)")
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
        self.m_smff = QLineEdit("")
        self.m_smff.setPlaceholderText("gaff-2.11 or openff-2.1.0 (if ligand present)")

        self.m_mini = _spin(10000, 1, 1_000_000)
        self.m_nvt_dt = _dspin(0.002, 0.0005, 0.005, 0.0005, 4)
        self.m_nvt_t = _dspin(250.0, 0.0, 1_000_000.0, 50.0, 1)
        self.m_npt_dt = _dspin(0.002, 0.0005, 0.005, 0.0005, 4)
        self.m_npt_t = _dspin(500.0, 0.0, 1_000_000.0, 50.0, 1)
        self.m_prod_dt = _dspin(0.004, 0.0005, 0.005, 0.0005, 4)
        self.m_prod_t = _dspin(10000.0, 0.0, 100_000_000.0, 1000.0, 1)

        general = QGroupBox("Ensemble & output")
        gf = QFormLayout(general)
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
        gf.addRow("Force fields:", self.m_ff)

        stages = QGroupBox("Stages (ps)")
        sf = QFormLayout(stages)
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
        top = QHBoxLayout()
        top.addWidget(general, 1)
        top.addWidget(stages, 1)
        v.addLayout(top)
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
        if self.m_smff.text().strip():
            over["small_molecule_forcefield"] = self.m_smff.text().strip()
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
            _preset_names = single_gpu_preset_names()
        except Exception:
            _preset_names = []
        self.hpc_preset.addItems(["(custom)", *_preset_names])
        self.hpc_preset.setToolTip("Pick a preset from env/hpc.yaml (e.g. singlenode-singlegpu-long) "
                                   "to fill the SLURM fields below; '(custom)' keeps your own values.")
        self.hpc_preset.currentTextChanged.connect(self._apply_hpc_preset)
        self.cpus = _spin(8, 1, 128)
        self.gpus = _spin(1, 0, 16)
        self.gpu_type = QComboBox()
        self.gpu_type.setEditable(True)
        self.gpu_type.addItems(["", "a100", "a100x", "v100", "v100x", "h100", "p100", "a40", "rtx8000"])
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
        f = QFormLayout(form)
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
            merged = apply_slurm_preset(self._hpc(), name)
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
