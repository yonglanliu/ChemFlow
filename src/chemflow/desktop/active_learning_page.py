"""Active-learning virtual screening page for ChemFlow Studio.

Drives ``chemflow active-learn`` (Deep-Docking-style loop): a Chemprop surrogate
scores the whole sharded library each round; the chosen docking **oracle**
(Vina or Glide) only docks the acquisition batch. HPC environment **modules** can
be ``module load``-ed before the oracle runs.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.widgets.forms import (
    NoScrollComboBox,
    NoScrollSpinBox,
    PageHeader,
    PathField,
    action_button,
)


class ActiveLearningPage(QWidget):
    """Configure + launch an active-learning docking screen (Vina / Glide oracle)."""

    def __init__(self, console=None, parent=None):
        super().__init__(parent)
        self._console = console
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(12)

        layout.addWidget(PageHeader(
            "AI screening", "Active learning virtual screening",
            "A Chemprop surrogate scores the whole library each round; the docking "
            "oracle (Vina / Glide) only docks the batch active learning selects. "
            "Runs `chemflow active-learn`."))

        # Config path
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Config:"))
        self.config = PathField("al.yaml", mode="save", save_filter="YAML (*.yaml *.yml)")
        self.config.setText("al.yaml")
        cfg_row.addWidget(self.config, 1)
        layout.addLayout(cfg_row)

        # --- Oracle ---
        layout.addWidget(self._section("Docking oracle"))
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Oracle:"))
        self.oracle = NoScrollComboBox()
        self.oracle.addItems(["glide", "vsw", "vina"])
        self.oracle.setToolTip("Docking oracle. Each wrapper does ligand prep THEN docks:\n"
                               "  glide = LigPrep → Glide SP/XP\n"
                               "  vsw   = Schrödinger Virtual Screening Workflow (prep + HTVS→SP→XP)\n"
                               "  vina  = Meeko/OpenBabel → AutoDock Vina")
        orow.addWidget(self.oracle)
        orow.addWidget(QLabel("Receptor:"))
        self.receptor = PathField("Glide grid .zip  (Vina: receptor .pdbqt)", mode="file")
        orow.addWidget(self.receptor, 1)
        layout.addLayout(orow)

        mod_row = QHBoxLayout()
        mod_row.addWidget(QLabel("module load:"))
        self.modules = QLineEdit()
        self.modules.setPlaceholderText("HPC modules for the oracle, e.g. schrodinger  (or: autodock-vina openbabel)")
        self.modules.setToolTip("Environment modules `module load`-ed in a login shell before the "
                                "oracle runs (space/comma separated). Leave blank if not on an lmod cluster.")
        mod_row.addWidget(self.modules, 1)
        layout.addLayout(mod_row)

        ocmd_row = QHBoxLayout()
        ocmd_row.addWidget(QLabel("Oracle command (optional override):"))
        self.oracle_command = QLineEdit()
        self.oracle_command.setPlaceholderText(
            "leave blank → glide_dock.sh / vsw_dock.sh / vina_dock.sh {batch} {receptor} {out}")
        self.oracle_command.setToolTip(
            "Wrapper that reads the batch .smi (SMILES+id), runs ligand prep, docks against "
            "the receptor, and writes {out} as CSV 'id,score'. Placeholders: {batch} {receptor} {out}.")
        ocmd_row.addWidget(self.oracle_command, 1)
        layout.addLayout(ocmd_row)

        # --- Library + surrogate ---
        layout.addWidget(self._section("Library & surrogate"))
        lib_row = QHBoxLayout()
        lib_row.addWidget(QLabel("Library:"))
        self.library_preset = NoScrollComboBox()
        self.library_preset.addItems(["enamine_real", "zinc20", "custom"])
        self.library_preset.setToolTip(
            "Vendor preset fills the shard glob + SMILES/id columns:\n"
            "  enamine_real = screen_db/*.cxsmiles\n"
            "  zinc20       = screen_db/*.smi\n"
            "  custom       = type your own glob")
        lib_row.addWidget(self.library_preset)
        lib_row.addWidget(QLabel("Shards (glob):"))
        self.library = QLineEdit("screen_db/enamine_real/*.cxsmiles.bz2")
        self.library.setToolTip("Sharded library files (SMILES + id per line; .bz2/.gz ok). "
                                "For 10^8–10^9 compounds, shard across many files.")
        lib_row.addWidget(self.library, 1)
        self._lib_browse = QPushButton("Browse…")
        self._lib_browse.setToolTip("Custom library: locate a folder of shards or pick files.")
        _menu = QMenu(self._lib_browse)
        _menu.addAction("Choose folder…", self._browse_library_folder)
        _menu.addAction("Choose files…", self._browse_library_files)
        self._lib_browse.setMenu(_menu)
        lib_row.addWidget(self._lib_browse)
        layout.addLayout(lib_row)
        self._PRESET_GLOBS = {
            "enamine_real": "screen_db/enamine_real/*.cxsmiles.bz2",
            "zinc20": "screen_db/zinc20/*.smi",
        }
        self._PRESET_EXTS = {
            "enamine_real": ["cxsmiles.bz2", "cxsmiles.gz", "cxsmiles"],
            "zinc20": ["smi", "smi.gz", "smi.bz2"],
        }
        self.library_preset.currentTextChanged.connect(self._on_library_preset)
        self._on_library_preset(self.library_preset.currentText())

        tr_row = QHBoxLayout()
        tr_row.addWidget(QLabel("Train cmd:"))
        self.train_command = QLineEdit("chemflow train chemprop --data {labels} --out {model}")
        self.train_command.setToolTip("Surrogate training command. Placeholders: {labels} {model}.")
        tr_row.addWidget(self.train_command, 1)
        layout.addLayout(tr_row)
        pr_row = QHBoxLayout()
        pr_row.addWidget(QLabel("Predict cmd:"))
        self.predict_command = QLineEdit(
            "chemflow predict --model {model} --library {library} --out {out} --uncertainty")
        self.predict_command.setToolTip("Library-wide prediction command. Placeholders: {model} {library} {out}.")
        pr_row.addWidget(self.predict_command, 1)
        layout.addLayout(pr_row)

        # --- Loop settings ---
        layout.addWidget(self._section("Active-learning loop"))
        s1 = QHBoxLayout()
        s1.addWidget(QLabel("Workdir:"))
        self.workdir = QLineEdit(".")
        self.workdir.setMaximumWidth(120)
        self.workdir.setToolTip("Run from here (default: the current directory).")
        s1.addWidget(self.workdir)
        s1.addWidget(QLabel("Job name:"))
        self.job_name = QLineEdit("al_run")
        self.job_name.setMaximumWidth(160)
        self.job_name.setToolTip("All outputs are written into <workdir>/<job_name>/.")
        s1.addWidget(self.job_name)
        s1.addWidget(QLabel("Rounds:"))
        self.rounds = self._spin(5, 1, 100)
        s1.addWidget(self.rounds)
        s1.addWidget(QLabel("Seed size:"))
        self.seed_size = self._spin(2000, 1, 10_000_000)
        s1.addWidget(self.seed_size)
        s1.addWidget(QLabel("Batch size:"))
        self.batch_size = self._spin(2000, 1, 10_000_000)
        s1.addWidget(self.batch_size)
        s1.addStretch()
        layout.addLayout(s1)

        s2 = QHBoxLayout()
        s2.addWidget(QLabel("Acquisition:"))
        self.strategy = NoScrollComboBox()
        self.strategy.addItems(["ucb", "greedy", "thompson", "random"])
        self.strategy.setToolTip("ucb/thompson use surrogate uncertainty (explore); greedy = exploit.")
        s2.addWidget(self.strategy)
        s2.addWidget(QLabel("Top hits:"))
        self.top_n = self._spin(500, 1, 1_000_000)
        s2.addWidget(self.top_n)
        self.lower_is_better = QCheckBox("Lower score = better")
        self.lower_is_better.setChecked(True)
        self.lower_is_better.setToolTip("Docking scores: more negative = stronger. Uncheck for a higher-is-better score.")
        s2.addWidget(self.lower_is_better)
        s2.addStretch()
        layout.addLayout(s2)

        # Seed (round-0) selection row.
        s3 = QHBoxLayout()
        s3.addWidget(QLabel("Seed method:"))
        self.seed_method = NoScrollComboBox()
        self.seed_method.addItems(["diverse", "random", "scaffold"])  # diverse = default
        self.seed_method.setToolTip(
            "Round-0 selection: diverse = MaxMin fingerprint pick (default), "
            "random = fast offset sample, scaffold = Butina cluster representatives.")
        s3.addWidget(self.seed_method)
        s3.addWidget(QLabel("Seed pool:"))
        self.seed_pool = self._spin(0, 0, 50_000_000)
        self.seed_pool.setToolTip("Oversample the diverse/scaffold pick draws from (0 = ~20x seed size).")
        s3.addWidget(self.seed_pool)
        s3.addWidget(QLabel("Known actives:"))
        self.seed_actives = PathField("optional SMILES of known binders (warm-start)", mode="file")
        s3.addWidget(self.seed_actives, 1)
        layout.addLayout(s3)

        # --- Actions ---
        btns = QHBoxLayout()
        self._btn_scaffold = action_button("Scaffold helpers")
        self._btn_seed = action_button("Create seed")
        self._btn_generate = action_button("Write config")
        self._btn_dry = action_button("Dry run")
        self._btn_run = action_button("Run")
        self._btn_scaffold.clicked.connect(lambda: self._scaffold())
        self._btn_seed.clicked.connect(lambda: self._seed())
        self._btn_generate.clicked.connect(lambda: self._write_config())
        self._btn_dry.clicked.connect(lambda: self._run(dry=True))
        self._btn_run.clicked.connect(lambda: self._run(dry=False))
        for b in (self._btn_scaffold, self._btn_seed, self._btn_generate,
                  self._btn_dry, self._btn_run):
            btns.addWidget(b)
        btns.addStretch()
        layout.addLayout(btns)

        self.status = QLabel("Set the oracle + receptor, point at your library shards, then Dry run.")
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()

    # -- helpers ----------------------------------------------------------
    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight:700;margin-top:6px;")
        return lbl

    def _spin(self, value: int, lo: int, hi: int) -> NoScrollSpinBox:
        sp = NoScrollSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(value)
        sp.setMinimumWidth(90)
        return sp

    def _modules_list(self) -> list[str]:
        return [m for m in self.modules.text().replace(",", " ").split() if m]

    def _on_library_preset(self, name: str) -> None:
        """Preset → auto-find the shard folder and lock the field; custom → Browse."""
        if name in self._PRESET_GLOBS:
            detected = self._autodetect_library(name)
            self.library.setText(detected)
            self.library.setEnabled(False)
            self._lib_browse.setEnabled(False)
            found = detected != self._PRESET_GLOBS[name]
            msg = (f"{name}: found library → {detected}" if found else
                   f"{name}: folder not auto-found, using default '{detected}' "
                   "(switch to custom to Browse).")
            self._set_status(msg)
        else:  # custom
            self.library.setEnabled(True)
            self._lib_browse.setEnabled(True)
            self._set_status("Custom library — Browse to a folder or files, or type a glob.")

    def _set_status(self, text: str) -> None:
        # _on_library_preset runs once during __init__ before self.status exists.
        if getattr(self, "status", None) is not None:
            self.status.setText(text)

    def _library_bases(self, preset: str) -> list[Path]:
        """Dirs to search for a preset library: config dir + cwd, each plus
        screen_db/ and screen_db/<preset>/ (the on-disk layout)."""
        cfg_dir = Path(self.config.text().strip() or "al.yaml").expanduser().parent
        bases: list[Path] = []
        for b in (cfg_dir, Path.cwd()):
            bases.extend([b, b / "screen_db", b / "screen_db" / preset])
        return bases

    def _autodetect_library(self, preset: str) -> str:
        """Find the first dir holding files of the preset's type → 'dir/*.ext' glob."""
        exts = self._PRESET_EXTS.get(preset, [])
        seen: set[Path] = set()
        for d in self._library_bases(preset):
            try:
                d = d.resolve()
            except OSError:
                continue
            if d in seen or not d.is_dir():
                continue
            seen.add(d)
            for ext in exts:
                if any(d.glob(f"*.{ext}")):
                    return str(d / f"*.{ext}")
        return self._PRESET_GLOBS.get(preset, "")

    def _browse_start(self) -> str:
        t = self.library.text().strip()
        if t:
            p = Path(t.split("*")[0]).expanduser()
            if p.exists():
                return str(p)
            if p.parent.exists():
                return str(p.parent)
        return str(Path.cwd())

    def _browse_library_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose library folder", self._browse_start())
        if not d:
            return
        ext = self._dominant_ext(Path(d)) or "smi"
        self.library.setText(str(Path(d) / f"*.{ext}"))
        self._set_status(f"Custom library → {self.library.text()}")

    def _browse_library_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose library files", self._browse_start(),
            "SMILES libraries (*.cxsmiles *.smi *.cxsmiles.bz2 *.cxsmiles.gz *.smi.bz2 *.smi.gz);;"
            "All files (*)")
        if not files:
            return
        p = Path(files[0])
        ext = "".join(p.suffixes).lstrip(".") or "smi"   # 'foo.smi.gz' → 'smi.gz'
        # A shard glob over the selected files' folder (the loop globs, not single files).
        self.library.setText(str(p.parent / f"*.{ext}"))
        self._set_status(f"Custom library → {self.library.text()} ({len(files)} file(s) selected)")

    @staticmethod
    def _dominant_ext(d: Path) -> str | None:
        from collections import Counter

        c: Counter = Counter()
        for ext in ("cxsmiles.bz2", "cxsmiles.gz", "cxsmiles", "smi.bz2", "smi.gz", "smi"):
            n = len(list(d.glob(f"*.{ext}")))
            if n:
                c[ext] += n
        return c.most_common(1)[0][0] if c else None

    def _config_dict(self) -> dict:
        return {
            "workdir": self.workdir.text().strip() or ".",
            "job_name": self.job_name.text().strip() or "al_run",
            "library": self.library_preset.currentText(),
            "library_shards": self.library.text().strip() or "screen_db/*.cxsmiles",
            "rounds": self.rounds.value(),
            "seed_size": self.seed_size.value(),
            "batch_size": self.batch_size.value(),
            "strategy": self.strategy.currentText(),
            "seed_method": self.seed_method.currentText(),
            "seed_pool": self.seed_pool.value(),
            "seed_actives": self.seed_actives.text().strip(),
            "lower_is_better": self.lower_is_better.isChecked(),
            "top_n": self.top_n.value(),
            "oracle": self.oracle.currentText(),
            "receptor": self.receptor.text().strip(),
            "modules": self._modules_list(),
            "oracle_command": self.oracle_command.text().strip(),
            "train_command": self.train_command.text().strip(),
            "predict_command": self.predict_command.text().strip(),
        }

    def _write_config(self) -> str | None:
        import yaml

        cfg = self.config.text().strip() or "al.yaml"
        path = Path(cfg).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(self._config_dict(), sort_keys=False), encoding="utf-8")
        except OSError as exc:
            self.status.setText(f"Could not write config: {exc}")
            return None
        self.status.setText(f"Wrote {path}. Review it, then Dry run / Run.")
        return str(path)

    def _run(self, *, dry: bool) -> None:
        cfg = self._write_config()
        if cfg is None:
            return
        if self._console is None:
            self.status.setText("No console attached; run from a terminal: "
                                f"chemflow active-learn run {cfg}" + (" --dry-run" if dry else ""))
            return
        argv = ["active-learn", "run", cfg]
        if dry:
            argv.append("--dry-run")
        self._console.run(argv)
        self.status.setText(("Dry run: " if dry else "Running: ")
                            + f"active-learning loop ({self.oracle.currentText()} oracle). "
                            "Watch the Activity console.")

    def _scaffold(self) -> None:
        """Write the VSW template + oracle wrapper + monitor next to the config."""
        if self._console is None:
            self.status.setText("No console attached; run: chemflow active-learn scaffold")
            return
        dest = str(Path(self.config.text().strip() or "al.yaml").expanduser().parent)
        self._console.run(["active-learn", "scaffold", "--dir", dest])
        self.status.setText(f"Writing helper scripts (vsw_template.inp, vsw_dock.sh, …) to {dest}. "
                            "Edit vsw_template.inp (grid/constraints) before running.")

    def _seed(self) -> None:
        """Create seed.smi by sampling the library with the chosen method."""
        if self._console is None:
            self.status.setText("No console attached; run: chemflow active-learn seed --db …")
            return
        db = self.library.text().strip()
        if not db:
            self.status.setText("Set 'Library' first (preset auto-finds, or Browse a custom folder).")
            return
        base = Path(self.config.text().strip() or "al.yaml").expanduser().parent
        job_dir = base / (self.workdir.text().strip() or ".") / (self.job_name.text().strip() or "al_run")
        seed_out = str(job_dir / "seed.smi")
        argv = ["active-learn", "seed", "--library", self.library_preset.currentText(),
                "--db", db,
                "--n", str(self.seed_size.value()),
                "--method", self.seed_method.currentText(),
                "--pool", str(self.seed_pool.value()),
                "-o", seed_out]
        actives = self.seed_actives.text().strip()
        if actives:
            argv += ["--actives", actives]
        self._console.run(argv)
        self.status.setText(f"Selecting {self.seed_size.value()} seed compounds "
                            f"({self.seed_method.currentText()}) → {seed_out}. Watch the console.")
