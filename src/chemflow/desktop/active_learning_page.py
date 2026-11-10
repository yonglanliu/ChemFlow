"""Active-learning virtual screening page for ChemFlow Studio.

A **workflow view**: the pipeline is drawn as a strip of clickable nodes
(Library → Acquire → Oracle → Train → Predict → Escalate), mirroring the
active-learning figure. Click a node to configure it in the panel below (database,
method, oracle tool(s), commands); optional nodes carry an "Include this step"
toggle. Global run settings and the actions live at the bottom. Drives
``chemflow active-learn``.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QStackedWidget,
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

# Node accent colours (match the workflow figure: ML steps teal, oracle orange).
_ACCENTS = {
    "library": "#94a3b8",
    "seed": "#2bb79a",
    "acquire": "#2bb79a",
    "oracle": "#c2791f",
    "train": "#2bb79a",
    "predict": "#2bb79a",
    "escalate": "#6da8ff",
}

# name -> (checkbox label, input hint, default metric column, default direction)
_ORACLE_META = {
    "glide":  ("Glide (SP→XP)",             "Glide grid .zip",          "glide_score", "lower"),
    "vina":   ("AutoDock Vina",             "receptor .pdbqt + box",    "vina_kcal",   "lower"),
    "vsw":    ("Schrödinger VSW",           "Glide grid .zip",          "glide_score", "lower"),
    "boltz2": ("Boltz-2 co-fold (affinity)", "protein FASTA / sequence", "pIC50",       "higher"),
}


class ActiveLearningPage(QWidget):
    """Workflow-style configuration + launch for an active-learning docking screen."""

    def __init__(self, console=None, parent=None):
        super().__init__(parent)
        self._console = console
        self._node_buttons: dict[str, QPushButton] = {}
        self._node_enabled: dict[str, QCheckBox] = {}

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
            "Click a node to configure that step. A Chemprop surrogate scores the "
            "whole library each round; the docking oracle(s) only dock the batch "
            "active learning selects. Runs `chemflow active-learn`."))

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Config:"))
        self.config = PathField("al.yaml", mode="save", save_filter="YAML (*.yaml *.yml)")
        self.config.setText("al.yaml")
        cfg_row.addWidget(self.config, 1)
        layout.addLayout(cfg_row)

        # --- Build the per-node widgets (reused by the config panel) ----------
        self._build_library_widgets()
        self._build_acquire_widgets()
        self._build_oracle_widgets()
        self._build_train_widgets()
        self._build_predict_widgets()
        self._build_escalate_widgets()

        # --- Workflow node strip ---------------------------------------------
        self._nodes = [
            ("library", "Chemical\nlibrary", False),
            ("seed", "Seed\n(round 0)", False),
            ("acquire", "① Acquire\nbatch", False),
            ("oracle", "② Oracle\n(dock)", False),
            ("train", "③ Train\nsurrogate", True),
            ("predict", "④ Predict\nlibrary", True),
            ("escalate", "Escalate\n/ Buy", True),
        ]
        strip = QHBoxLayout()
        strip.setSpacing(4)
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._stack = QStackedWidget()
        page_builders = {
            "library": self._library_page,
            "seed": self._seed_page,
            "acquire": self._acquire_page,
            "oracle": self._oracle_page,
            "train": self._train_page,
            "predict": self._predict_page,
            "escalate": self._escalate_page,
        }
        loop_keys = {"acquire", "oracle", "train", "predict"}

        # The acquire→oracle→train→predict steps repeat each round, so draw them
        # inside a dashed "loop box" carrying the Rounds control.
        loop_frame = QFrame()
        loop_frame.setObjectName("LoopBox")
        loop_frame.setStyleSheet("#LoopBox{border:2px dashed #2bb79a;border-radius:12px;}")
        lf = QVBoxLayout(loop_frame)
        lf.setContentsMargins(10, 6, 10, 10)
        lf.setSpacing(4)
        loop_head = QHBoxLayout()
        loop_title = QLabel("⟳ Active-learning loop   ×")
        loop_title.setStyleSheet("color:#0f766e;font-weight:700;")
        loop_head.addWidget(loop_title)
        loop_head.addWidget(self.rounds)
        loop_head.addWidget(QLabel("rounds"))
        loop_head.addStretch()
        lf.addLayout(loop_head)
        loop_strip = QHBoxLayout()
        loop_strip.setSpacing(4)
        lf.addLayout(loop_strip)

        main_count = 0
        loop_frame_added = False

        def _add_main(widget):
            nonlocal main_count
            if main_count:
                a = QLabel("→")
                a.setStyleSheet("color:#94a3b8;font-size:18px;")
                strip.addWidget(a)
            strip.addWidget(widget)
            main_count += 1

        for i, (key, label, optional) in enumerate(self._nodes):
            btn = self._make_node_button(key, label)
            self._btn_group.addButton(btn, i)
            self._node_buttons[key] = btn
            self._stack.addWidget(page_builders[key](key, optional))
            if key in loop_keys:
                if not loop_frame_added:
                    _add_main(loop_frame)
                    loop_frame_added = True
                if loop_strip.count():
                    a = QLabel("→")
                    a.setStyleSheet("color:#2bb79a;font-size:16px;")
                    loop_strip.addWidget(a)
                loop_strip.addWidget(btn)
            else:
                _add_main(btn)
        back = QLabel("↺")
        back.setStyleSheet("color:#2bb79a;font-size:20px;font-weight:700;")
        back.setToolTip("Repeats back to ① Acquire each round until budget/convergence.")
        loop_strip.addWidget(back)
        strip.addStretch()
        layout.addLayout(strip)
        self._btn_group.idClicked.connect(self._stack.setCurrentIndex)
        self._node_buttons["library"].setChecked(True)

        panel = QFrame()
        panel.setObjectName("Card")
        panel.setStyleSheet("#Card{border:1px solid #e2e8f0;border-radius:10px;}")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(16, 12, 16, 16)
        pl.addWidget(self._stack)
        layout.addWidget(panel)

        # --- Global run settings ---------------------------------------------
        layout.addWidget(self._section("Run settings"))
        s1 = QHBoxLayout()
        s1.addWidget(QLabel("Workdir:"))
        self.workdir = QLineEdit(".")
        self.workdir.setMaximumWidth(110)
        self.workdir.setToolTip("Run from here (default: the current directory).")
        s1.addWidget(self.workdir)
        s1.addWidget(QLabel("Job name:"))
        self.job_name = QLineEdit("al_run")
        self.job_name.setMaximumWidth(150)
        self.job_name.setToolTip("All outputs are written into <workdir>/<job_name>/.")
        s1.addWidget(self.job_name)
        s1.addWidget(QLabel("Batch size:"))
        self.batch_size = self._spin(10000, 1, 10_000_000)
        s1.addWidget(self.batch_size)
        s1.addWidget(QLabel("Top hits:"))
        s1.addWidget(self.top_n)
        s1.addStretch()
        layout.addLayout(s1)

        # --- Actions ----------------------------------------------------------
        btns = QHBoxLayout()
        self._btn_scaffold = action_button("Scaffold helpers")
        self._btn_generate = action_button("Write config")
        self._btn_dry = action_button("Dry run")
        self._btn_run = action_button("Run")
        self._btn_scaffold.clicked.connect(lambda: self._scaffold())
        self._btn_generate.clicked.connect(lambda: self._write_config())
        self._btn_dry.clicked.connect(lambda: self._run(dry=True))
        self._btn_run.clicked.connect(lambda: self._run(dry=False))
        for b in (self._btn_scaffold, self._btn_generate,
                  self._btn_dry, self._btn_run):
            btns.addWidget(b)
        btns.addStretch()
        layout.addLayout(btns)

        self.status = QLabel("Click a node to configure it, then Write config → Dry run → Run.")
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()

        self._on_library_preset(self.library_preset.currentText())

    # ======================================================================
    # Node strip
    # ======================================================================
    def _make_node_button(self, key: str, label: str) -> QPushButton:
        accent = _ACCENTS.get(key, "#94a3b8")
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setMinimumSize(104, 60)
        btn.setStyleSheet(
            f"QPushButton{{border:2px solid {accent};border-radius:10px;"
            f"background:#ffffff;color:#0f172a;font-weight:600;padding:4px;}}"
            f"QPushButton:checked{{background:{accent};color:#ffffff;}}"
            f"QPushButton:disabled{{border-color:#cbd5e1;color:#cbd5e1;}}")
        return btn

    def _refresh_node_style(self, key: str) -> None:
        """Dim a node button when its step is excluded."""
        btn = self._node_buttons.get(key)
        cb = self._node_enabled.get(key)
        if btn is None or cb is None:
            return
        on = cb.isChecked()
        btn.setText(btn.text())  # no-op keep
        f = btn.font()
        f.setStrikeOut(not on)
        btn.setFont(f)

    # ======================================================================
    # Per-node widgets
    # ======================================================================
    def _build_library_widgets(self) -> None:
        self.library_preset = NoScrollComboBox()
        self.library_preset.addItems(["enamine_real", "zinc20", "custom"])
        self.library_preset.setToolTip(
            "Vendor preset fills the shard glob + SMILES/id columns:\n"
            "  enamine_real = screen_db/enamine_real/*.cxsmiles.bz2\n"
            "  zinc20       = screen_db/zinc20/*.smi\n"
            "  custom       = type your own glob / Browse")
        self.library = QLineEdit("screen_db/enamine_real/*.cxsmiles.bz2")
        self.library.setToolTip("Sharded library files (SMILES + id per line; .bz2/.gz ok).")
        self._lib_browse = QPushButton("Browse…")
        self._lib_browse.setToolTip("Custom library: locate a folder of shards or pick files.")
        _menu = QMenu(self._lib_browse)
        _menu.addAction("Choose folder…", self._browse_library_folder)
        _menu.addAction("Choose files…", self._browse_library_files)
        self._lib_browse.setMenu(_menu)
        self._PRESET_GLOBS = {
            "enamine_real": "screen_db/enamine_real/*.cxsmiles.bz2",
            "zinc20": "screen_db/zinc20/*.smi",
        }
        self._PRESET_EXTS = {
            "enamine_real": ["cxsmiles.bz2", "cxsmiles.gz", "cxsmiles"],
            "zinc20": ["smi", "smi.gz", "smi.bz2"],
        }
        self.library_preset.currentTextChanged.connect(self._on_library_preset)

    def _build_acquire_widgets(self) -> None:
        self.strategy = NoScrollComboBox()
        self.strategy.addItems(["ucb", "greedy", "thompson", "random"])
        self.strategy.setToolTip("ucb/thompson use surrogate uncertainty (explore); greedy = exploit.")
        self.seed_method = NoScrollComboBox()
        self.seed_method.addItems(["diverse", "random", "scaffold"])
        self.seed_method.setToolTip(
            "Round-0 selection: diverse = MaxMin fingerprint pick (default), "
            "random = fast offset sample, scaffold = Butina cluster representatives.")
        self.seed_pool = self._spin(0, 0, 50_000_000)
        self.seed_pool.setToolTip("Oversample the diverse/scaffold pick draws from (0 = ~20x seed size).")
        self.seed_size = self._spin(10000, 1, 10_000_000)
        self.seed_size.setToolTip("Round-0 seed count (compounds docked before any model exists).")
        self.rounds = self._spin(5, 1, 100)
        self.rounds.setToolTip("Active-learning iterations: how many times the loop "
                               "(acquire → dock → train → predict) repeats.")
        self.seed_actives = PathField("optional SMILES of known binders (warm-start)", mode="file")
        self.top_n = self._spin(500, 1, 1_000_000)

    def _build_oracle_widgets(self) -> None:
        self.oracle_checks: dict = {}
        self.oracle_cfg: dict = {}        # name -> {input, metric, dir, modules, cmd}
        for name, (label, inhint, metric, direction) in _ORACLE_META.items():
            self.oracle_checks[name] = QCheckBox(label)
            inp = PathField(inhint, mode="file")
            met = QLineEdit(metric)
            met.setMaximumWidth(130)
            met.setToolTip("Target column name for this oracle's score (a training task).")
            dirc = NoScrollComboBox()
            dirc.addItems(["lower = better", "higher = better"])
            dirc.setCurrentIndex(0 if direction == "lower" else 1)
            mod = QLineEdit()
            mod.setPlaceholderText("module load … (optional)")
            cmd = QLineEdit()
            cmd.setPlaceholderText(f"override → {name}_*.sh {{batch}} {{receptor}} {{out}} (optional)")
            self.oracle_cfg[name] = {"input": inp, "metric": met, "dir": dirc,
                                     "modules": mod, "cmd": cmd}
        self.oracle_checks["glide"].setChecked(True)
        self.oracle_checks["boltz2"].setToolTip(
            "ML co-folding affinity (AlchemFold / Boltz-2): predicts pIC50 per ligand — "
            "GPU, no docking grid; its input is the protein sequence/FASTA.")
        self.primary_oracle = NoScrollComboBox()
        self.primary_oracle.setToolTip(
            "Which oracle's metric drives batch selection each round; the others still "
            "train the surrogate as auxiliary (multitask) targets.")

    def _build_train_widgets(self) -> None:
        self.train_command = QLineEdit("chemflow train chemprop --data {labels} --out {model}")
        self.train_command.setToolTip("Surrogate training command. Placeholders: {labels} {model}.")

    def _build_predict_widgets(self) -> None:
        self.predict_command = QLineEdit(
            "chemflow predict --model {model} --library {library} --out {out} --uncertainty")
        self.predict_command.setToolTip("Library-wide prediction. Placeholders: {model} {library} {out}.")

    def _build_escalate_widgets(self) -> None:
        self.escalate_note = QLabel(
            "Top-ranked hits branch out: order purchasable compounds directly (Buy), "
            "and/or escalate the best to MM-GBSA → Boltz-2 co-fold → FEP (AlchemForge) "
            "for quantitative ΔG. 'Top hits' count = Run settings · Top hits.")
        self.escalate_note.setWordWrap(True)
        self.escalate_note.setObjectName("Muted")

    # ======================================================================
    # Per-node config pages
    # ======================================================================
    def _node_page(self, key: str, optional: bool, title: str, rows: list) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        head = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet(f"font-weight:700;color:{_ACCENTS.get(key, '#0f172a')};font-size:15px;")
        head.addWidget(lbl)
        head.addStretch()
        cb = QCheckBox("Include this step")
        cb.setChecked(True)
        if optional:
            cb.toggled.connect(lambda _=None, k=key: self._refresh_node_style(k))
        else:
            cb.setEnabled(False)
            cb.setToolTip("Required step — always run.")
        self._node_enabled[key] = cb
        head.addWidget(cb)
        v.addLayout(head)
        for row in rows:
            if isinstance(row, QWidget):
                v.addWidget(row)
            else:
                v.addLayout(row)
        v.addStretch()
        return page

    def _labeled(self, text: str, widget: QWidget, stretch: int = 1) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(text))
        row.addWidget(widget, stretch)
        return row

    def _library_page(self, key, optional) -> QWidget:
        lib_row = QHBoxLayout()
        lib_row.addWidget(QLabel("Database:"))
        lib_row.addWidget(self.library_preset)
        lib_row.addWidget(QLabel("Shards (glob):"))
        lib_row.addWidget(self.library, 1)
        lib_row.addWidget(self._lib_browse)
        note = QLabel("Enamine REAL / ZINC20 auto-find their folder; Custom → type a glob or Browse.")
        note.setObjectName("Muted"); note.setWordWrap(True)
        return self._node_page(key, optional, "Chemical library", [lib_row, note])

    def _seed_page(self, key, optional) -> QWidget:
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Method:")); r1.addWidget(self.seed_method)
        r1.addWidget(QLabel("Seed size:")); r1.addWidget(self.seed_size)
        r1.addWidget(QLabel("Pool:")); r1.addWidget(self.seed_pool)
        r1.addStretch()
        r2 = self._labeled("Known actives:", self.seed_actives)
        make = action_button("Create seed")
        make.clicked.connect(lambda: self._seed())
        r3 = QHBoxLayout()
        r3.addWidget(make)
        r3.addStretch()
        note = QLabel("Round 0 has no model yet, so pick a diverse seed (MaxMin) for coverage; "
                      "random = fast, scaffold = Butina clusters. Known actives are force-included "
                      "to warm-start the surrogate. 'Create seed' samples the library into seed.smi.")
        note.setObjectName("Muted"); note.setWordWrap(True)
        return self._node_page(key, optional, "Seed (round 0 selection)", [r1, r2, r3, note])

    def _acquire_page(self, key, optional) -> QWidget:
        r1 = self._labeled("Acquisition:", self.strategy, 0)
        r1.addStretch()
        note = QLabel("From round 1 on, the surrogate ranks the whole library and this picks the next "
                      "batch — UCB/Thompson balance explore/exploit via the model's uncertainty; "
                      "greedy = pure exploit. Batch size is in Run settings. With multiple oracles, "
                      "the primary oracle's metric (set on the Oracle node) drives selection.")
        note.setObjectName("Muted"); note.setWordWrap(True)
        return self._node_page(key, optional, "① Acquire next batch", [r1, note])

    def _oracle_page(self, key, optional) -> QWidget:
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Oracle tool(s):"))
        for cb in self.oracle_checks.values():
            orow.addWidget(cb)
        orow.addStretch()

        prow = QHBoxLayout()
        prow.addWidget(QLabel("Primary metric (drives selection):"))
        prow.addWidget(self.primary_oracle)
        prow.addStretch()

        # Per-oracle config, one card each, shown only when that oracle is checked.
        self._oracle_rows: dict = {}
        rows = QVBoxLayout()
        rows.setSpacing(6)
        for name, w in self.oracle_cfg.items():
            card = QWidget()
            cl = QVBoxLayout(card)
            cl.setContentsMargins(8, 6, 8, 6)
            cl.setSpacing(3)
            title = QLabel(f"▸ {_ORACLE_META[name][0]}")
            title.setStyleSheet("font-weight:600;color:#9a5b12;")
            cl.addWidget(title)
            l1 = QHBoxLayout()
            l1.addWidget(QLabel("Input:")); l1.addWidget(w["input"], 1)
            l1.addWidget(QLabel("Metric:")); l1.addWidget(w["metric"]); l1.addWidget(w["dir"])
            cl.addLayout(l1)
            l2 = QHBoxLayout()
            l2.addWidget(QLabel("module load:")); l2.addWidget(w["modules"], 1)
            l2.addWidget(QLabel("cmd:")); l2.addWidget(w["cmd"], 1)
            cl.addLayout(l2)
            card.setStyleSheet("background:#fffaf2;border:1px solid #f0d9b8;border-radius:8px;")
            self._oracle_rows[name] = card
            rows.addWidget(card)

        note = QLabel("Pick one, or several for multi-fidelity. Each oracle has its own input and output "
                      "metric; every metric is a training target for the (multitask) surrogate, while the "
                      "primary metric drives which compounds are docked next. Glide/Vina/VSW dock a "
                      "grid/receptor; Boltz-2 co-folds a FASTA (GPU). Note: the loop runs all selected "
                      "oracles per batch and records each metric, but surrogate training/acquisition on "
                      "multiple targets needs the multitask backend — until then it uses the primary metric.")
        note.setObjectName("Muted"); note.setWordWrap(True)

        for cb in self.oracle_checks.values():
            cb.toggled.connect(lambda _=None: self._refresh_oracle_rows())
        self._refresh_oracle_rows()
        return self._node_page(key, optional, "② Dock / score the batch (oracle)",
                               [orow, prow, rows, note])

    def _refresh_oracle_rows(self) -> None:
        """Show config only for checked oracles; keep the primary selector in sync."""
        sel = self._selected_oracles()
        for name, card in getattr(self, "_oracle_rows", {}).items():
            card.setVisible(self.oracle_checks[name].isChecked())
        cur = self.primary_oracle.currentText()
        self.primary_oracle.blockSignals(True)
        self.primary_oracle.clear()
        self.primary_oracle.addItems(sel or ["glide"])
        idx = self.primary_oracle.findText(cur)
        if idx >= 0:
            self.primary_oracle.setCurrentIndex(idx)
        self.primary_oracle.blockSignals(False)

    def _train_page(self, key, optional) -> QWidget:
        r = self._labeled("Train cmd:", self.train_command)
        note = QLabel(
            "Command run once per round to (re)fit the ML surrogate on every compound docked "
            "so far. Placeholders filled in each round:\n"
            "  • {labels} — cumulative CSV of id,score (all oracle-docked compounds to date)\n"
            "  • {model}  — output path the trained model/checkpoint is written to\n"
            "Train on ALL labels (good and bad scores), not just the top hits, and emit an "
            "uncertainty (σ) so UCB/Thompson acquisition can explore — e.g. Chemprop with "
            "MC-dropout or an ensemble. Example: "
            "chemflow train chemprop --data {labels} --out {model}\n"
            "Uncheck 'Include this step' to plan a brute-force dock with no ML surrogate.")
        note.setObjectName("Muted"); note.setWordWrap(True)
        return self._node_page(key, optional, "③ Train surrogate", [r, note])

    def _predict_page(self, key, optional) -> QWidget:
        r = self._labeled("Predict cmd:", self.predict_command)
        note = QLabel("Scores the whole library (ŷ ± σ) so acquisition can pick the next batch.")
        note.setObjectName("Muted"); note.setWordWrap(True)
        return self._node_page(key, optional, "④ Predict over full library", [r, note])

    def _escalate_page(self, key, optional) -> QWidget:
        return self._node_page(key, optional, "Escalate / Buy", [self.escalate_note])

    # ======================================================================
    # Helpers (library autodetect / browse / config / run) — unchanged logic
    # ======================================================================
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
        # Modules are now per-oracle (see _oracles_spec); kept for API compatibility.
        return []

    def _selected_oracles(self) -> list[str]:
        return [name for name, cb in self.oracle_checks.items() if cb.isChecked()]

    def _on_library_preset(self, name: str) -> None:
        if name in self._PRESET_GLOBS:
            detected = self._autodetect_library(name)
            self.library.setText(detected)
            self.library.setEnabled(False)
            self._lib_browse.setEnabled(False)
            found = detected != self._PRESET_GLOBS[name]
            self._set_status(f"{name}: found library → {detected}" if found else
                             f"{name}: folder not auto-found, using default '{detected}' "
                             "(switch to custom to Browse).")
        else:
            self.library.setEnabled(True)
            self._lib_browse.setEnabled(True)
            self._set_status("Custom library — Browse to a folder or files, or type a glob.")

    def _set_status(self, text: str) -> None:
        if getattr(self, "status", None) is not None:
            self.status.setText(text)

    def _library_bases(self, preset: str) -> list[Path]:
        cfg_dir = Path(self.config.text().strip() or "al.yaml").expanduser().parent
        bases: list[Path] = []
        for b in (cfg_dir, Path.cwd()):
            bases.extend([b, b / "screen_db", b / "screen_db" / preset])
        return bases

    def _autodetect_library(self, preset: str) -> str:
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
        ext = "".join(p.suffixes).lstrip(".") or "smi"
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

    def _oracles_spec(self) -> tuple[list[dict], dict]:
        """Return (list of per-oracle dicts, primary dict) from the selected oracles."""
        sel = self._selected_oracles() or ["glide"]
        specs = []
        for name in sel:
            w = self.oracle_cfg[name]
            specs.append({
                "name": name,
                "receptor": w["input"].text().strip(),
                "metric": w["metric"].text().strip() or name,
                "lower_is_better": w["dir"].currentIndex() == 0,
                "modules": [m for m in w["modules"].text().replace(",", " ").split() if m],
                "command": w["cmd"].text().strip(),
            })
        primary = self.primary_oracle.currentText() or sel[0]
        if primary not in sel:
            primary = sel[0]
        prim = next(o for o in specs if o["name"] == primary)
        return specs, prim

    def _config_dict(self) -> dict:
        specs, prim = self._oracles_spec()
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
            "top_n": self.top_n.value(),
            # Multi-oracle: each with its own input + metric (training targets).
            "oracles": specs,
            "primary_oracle": prim["name"],
            "primary_metric": prim["metric"],
            # Single-oracle back-compat (the current loop reads these = primary).
            "oracle": prim["name"],
            "receptor": prim["receptor"],
            "modules": prim["modules"],
            "oracle_command": prim["command"],
            "lower_is_better": prim["lower_is_better"],
            "train_command": self.train_command.text().strip(),
            "predict_command": self.predict_command.text().strip(),
            "nodes_enabled": {k: cb.isChecked() for k, cb in self._node_enabled.items()},
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
                            + f"active-learning loop (oracle: {', '.join(self._selected_oracles()) or 'glide'}). "
                            "Watch the Activity console.")

    def _scaffold(self) -> None:
        if self._console is None:
            self.status.setText("No console attached; run: chemflow active-learn scaffold")
            return
        dest = str(Path(self.config.text().strip() or "al.yaml").expanduser().parent)
        self._console.run(["active-learn", "scaffold", "--dir", dest])
        self.status.setText(f"Writing helper scripts (vsw_template.inp, vsw_dock.sh, …) to {dest}. "
                            "Edit vsw_template.inp (grid/constraints) before running.")

    def _seed(self) -> None:
        if self._console is None:
            self.status.setText("No console attached; run: chemflow active-learn seed --db …")
            return
        db = self.library.text().strip()
        if not db:
            self.status.setText("Set the Library node first (preset auto-finds, or Browse a custom folder).")
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
