"""ChemFlow Studio PROTAC module pages (PySide6).

Ternary search (PyRosetta / RosettaDock), structure fixing, merge, screen,
linkability, and the linker designer — each a scrollable form that runs through
the app's shared process console.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.widgets.forms import Card, PageHeader, PathField, action_button, field


def _scrollable(widget: QWidget, content: QWidget) -> None:
    """Wrap ``content`` in a scroll area filling ``widget`` (the WorkflowPage look)."""
    scroll = QScrollArea(widget)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(content)
    outer = QVBoxLayout(widget)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.addWidget(scroll)


class RosettaDockPage(QWidget):
    """Linker-free global protein-protein docking (RosettaDock) for the ternary search."""

    def __init__(self, runner=None, parent=None):
        super().__init__(parent)
        self._runner = runner
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(18)
        layout.addWidget(PageHeader(
            "Ternary search", "RosettaDock",
            "Linker-free protein-protein docking: sample how the two warhead-bound "
            "proteins come together, then rank by interface score and linkability."))

        card = Card("Inputs", "One PDB with both partners as separate chains, each with its warhead.")
        self.rosetta_cmd = QLineEdit("docking_protocol.default.linuxgccrelease")
        card.body.addWidget(field("Rosetta command", self.rosetta_cmd,
                                  "Path/name of your Rosetta docking app"))
        self.input_pdb = PathField("Both partners in one PDB (chains A and B)")
        card.body.addWidget(field("Input structure", self.input_pdb))
        self.partners = QLineEdit("A_B")
        self.mode = QComboBox()
        self.mode.addItems(["global", "local"])
        card.body.addWidget(self._row("Partners (e.g. A_B)", self.partners, "Mode", self.mode))
        self.nstruct = QSpinBox()
        self.nstruct.setRange(1, 1000000)
        self.nstruct.setValue(1000)
        self.out_dir = PathField("Output folder", mode="directory")
        self.out_dir.setText("rosettadock_output")
        card.body.addWidget(self._row("Models (nstruct)", self.nstruct, "Output folder", self.out_dir))
        self.target_params = PathField("Target warhead .params (Rosetta)")
        self.e3_params = PathField("E3 warhead .params (Rosetta)")
        card.body.addWidget(field("Warhead parameter files (from molfile_to_params.py)",
                                  self.target_params,
                                  "Rosetta needs a .params per warhead (non-standard residue)"))
        card.body.addWidget(field("", self.e3_params))
        layout.addWidget(card)

        buttons = QHBoxLayout()
        v = action_button("Validate"); v.setObjectName(""); v.clicked.connect(self._validate)
        s = action_button("Stage"); s.setObjectName(""); s.clicked.connect(self._stage)
        self.run_btn = action_button("Run RosettaDock"); self.run_btn.clicked.connect(self._run)
        buttons.addWidget(v); buttons.addWidget(s); buttons.addStretch(); buttons.addWidget(self.run_btn)
        layout.addLayout(buttons)
        layout.addStretch()
        _scrollable(self, content)

    def _row(self, l1, w1, l2, w2) -> QWidget:
        box = QWidget(); row = QHBoxLayout(box); row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(field(l1, w1), 1); row.addWidget(field(l2, w2), 1)
        return box

    def _write_yaml(self) -> Path:
        if not self.input_pdb.text():
            raise ValueError("Select the input PDB (both partners as separate chains).")
        out_dir = Path(self.out_dir.text() or "rosettadock_output").expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "rosetta_command": self.rosetta_cmd.text().strip() or "docking_protocol.default.linuxgccrelease",
            "workdir": str(out_dir.parent),
            "out_dir": str(out_dir),
            "input_pdb": self.input_pdb.text(),
            "partners": self.partners.text().strip() or "A_B",
            "mode": self.mode.currentText(),
            "nstruct": self.nstruct.value(),
        }
        params = [p for p in (self.target_params.text(), self.e3_params.text()) if p]
        if params:
            data["ligand_params"] = params
        path = out_dir / "rosettadock.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    def _validate(self) -> None:
        from chemflow.protac.rosettadock import load_config, validate_config

        try:
            validate_config(load_config(self._write_yaml()))
        except Exception as exc:
            QMessageBox.warning(self, "Invalid configuration", str(exc))
            return
        QMessageBox.information(self, "Valid", "The RosettaDock configuration is valid.")

    def _stage(self) -> None:
        from chemflow.protac.rosettadock import load_config, stage, validate_config

        try:
            conf = load_config(self._write_yaml())
            validate_config(conf)
            runner, out_dir = stage(conf)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot stage", str(exc))
            return
        QMessageBox.information(self, "Staged", f"Runner: {runner}\nOutput: {out_dir}")

    def _run(self) -> None:
        try:
            yaml_path = self._write_yaml()
        except Exception as exc:
            QMessageBox.warning(self, "Cannot run", str(exc))
            return
        if self._runner is None:
            QMessageBox.information(self, "Run from a terminal",
                                   f"Config written to:\n{yaml_path}\n\n"
                                   f"Run: chemflow protac rosettadock run {yaml_path}")
            return
        try:
            self._runner(["protac", "rosettadock", "run", str(yaml_path)])
        except Exception as exc:
            QMessageBox.warning(self, "Cannot start task", str(exc))


class LinkabilityPage(QWidget):
    """Is a candidate ternary complex linkable? (exit-vector distance/orientation)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(18)
        layout.addWidget(PageHeader(
            "Linkability", "Can these two warheads be linked?",
            "Before designing a linker, check that the two posed warheads sit close "
            "enough and point toward each other. Both files must share one frame."))

        card = Card("Posed warheads", "From the same candidate complex (same coordinate frame).")
        self.head1 = PathField("Target warhead (SDF/MOL/PDB)")
        self.atom1 = QSpinBox(); self.atom1.setRange(0, 100000)
        card.body.addWidget(field("Target warhead", self.head1))
        card.body.addWidget(field("Attachment atom index (head 1)", self.atom1))
        self.head2 = PathField("E3 warhead (SDF/MOL/PDB)")
        self.atom2 = QSpinBox(); self.atom2.setRange(0, 100000)
        card.body.addWidget(field("E3 warhead", self.head2))
        card.body.addWidget(field("Attachment atom index (head 2)", self.atom2))
        self.min_dist = QDoubleSpinBox(); self.min_dist.setRange(0.1, 50.0); self.min_dist.setValue(3.0)
        self.max_dist = QDoubleSpinBox(); self.max_dist.setRange(0.1, 100.0); self.max_dist.setValue(20.0)
        row = QWidget(); r = QHBoxLayout(row); r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(field("Min distance (A)", self.min_dist), 1)
        r.addWidget(field("Max distance (A)", self.max_dist), 1)
        card.body.addWidget(row)
        layout.addWidget(card)

        assess = action_button("Assess linkability")
        assess.clicked.connect(self._assess)
        layout.addWidget(assess)
        self.result = QLabel("Load two posed warheads and pick an attachment atom on each.")
        self.result.setWordWrap(True)
        self.result.setObjectName("Muted")
        layout.addWidget(self.result)
        layout.addStretch()
        _scrollable(self, content)

    def _assess(self) -> None:
        from chemflow.protac.linkability import assess_from_files

        if not self.head1.text() or not self.head2.text():
            QMessageBox.warning(self, "Missing input", "Select both warhead files.")
            return
        try:
            res = assess_from_files(
                self.head1.text(), self.atom1.value(), self.head2.text(), self.atom2.value(),
                min_dist=self.min_dist.value(), max_dist=self.max_dist.value())
        except Exception as exc:
            QMessageBox.warning(self, "Cannot assess", str(exc))
            return
        verdict = res.verdict.replace("_", " ").upper()
        self.result.setText(
            f"Verdict: {verdict}" + (f" — {res.reason}" if res.reason else "")
            + f"\nAttachment-atom distance: {res.distance:.2f} A"
            + f"\nExit vectors toward partner (cosine): head1 {res.toward1:+.2f}, head2 {res.toward2:+.2f}"
            + ("\n\nGood candidate — design a linker on the PROTAC linker page."
               if res.linkable else "\n\nNot ideal; try another complex or attachment atoms."))


class MergePdbPage(QWidget):
    """Merge two binary complexes into one two-chain PDB for docking."""

    def __init__(self, parent=None):
        super().__init__(parent)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(18)
        layout.addWidget(PageHeader("Merge PDB", "Build the two-chain docking input",
                                    "Combine target+warhead and E3+warhead into one PDB."))
        card = Card("Inputs", "Each should be one protein + its warhead.")
        self.pdb1 = PathField("Target protein + warhead (PDB)")
        self.pdb2 = PathField("E3 ligase + warhead (PDB)")
        self.chain1 = QLineEdit("A")
        self.chain2 = QLineEdit("B")
        self.out = PathField("Output PDB", mode="save", save_filter="PDB (*.pdb)")
        self.out.setText("ternary_partners.pdb")
        card.body.addWidget(field("Target structure", self.pdb1))
        card.body.addWidget(field("E3 structure", self.pdb2))
        row = QWidget(); r = QHBoxLayout(row); r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(field("Target chain", self.chain1), 1)
        r.addWidget(field("E3 chain", self.chain2), 1)
        card.body.addWidget(row)
        card.body.addWidget(field("Output PDB", self.out))
        layout.addWidget(card)
        btn = action_button("Merge")
        btn.clicked.connect(self._merge)
        layout.addWidget(btn)
        self.status = QLabel(""); self.status.setObjectName("Muted"); self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()
        _scrollable(self, content)

    def _merge(self):
        from chemflow.protac.merge_pdb import merge_pdbs

        if not self.pdb1.text() or not self.pdb2.text():
            QMessageBox.warning(self, "Missing input", "Select both PDB files.")
            return
        pdb1 = Path(self.pdb1.text()).expanduser().resolve()
        out = Path(self.out.text().strip() or "ternary_partners.pdb").expanduser()
        if not out.is_absolute():
            out = Path.cwd() / out                # the working folder
        self.out.setText(str(out))
        try:
            res = merge_pdbs(str(pdb1), self.pdb2.text(), str(out),
                             chain1=self.chain1.text() or "A", chain2=self.chain2.text() or "B")
        except Exception as exc:
            QMessageBox.warning(self, "Cannot merge", str(exc))
            return
        self.status.setText(f"Wrote {res['out']}  (partners {res['partners']}).")


class FixStructurePage(QWidget):
    """Fill missing atoms/residues (PDBFixer) or rebuild loops (Rosetta KIC)."""

    def __init__(self, runner=None, parent=None):
        super().__init__(parent)
        self._runner = runner
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(18)
        layout.addWidget(PageHeader("Fix structure", "Fill missing atoms / residues",
                                    "PDBFixer for atoms + short loops; Rosetta KIC for interface loops."))
        card = Card("Input", "")
        self.input = PathField("Structure (PDB)")
        self.method = QComboBox(); self.method.addItems(["pdbfixer", "kic"])
        self.output = PathField("Output PDB", mode="save", save_filter="PDB (*.pdb)")
        self.output.setText("fixed.pdb")
        self.include_termini = QCheckBox("Also model terminal residues (pdbfixer)")
        self.add_h = QCheckBox("Add hydrogens (pdbfixer)")
        self.max_gap = QSpinBox(); self.max_gap.setRange(0, 1000); self.max_gap.setSpecialValueText("no limit")
        self.loops = QLineEdit(); self.loops.setPlaceholderText("kic refine existing loops: A:45-52, A:88-93")
        # Reference sequence (to fill missing residues) with three sources.
        self.seq_source = QComboBox(); self.seq_source.addItems(["Paste sequence", "FASTA file", "UniProt ID"])
        self.seq_paste = QPlainTextEdit(); self.seq_paste.setPlaceholderText("paste the one-letter sequence…")
        self.seq_paste.setMaximumHeight(90)
        self.fasta = PathField("Browse a FASTA / .fa file", save_filter="FASTA (*.fa *.fasta)")
        self.uniprot = QLineEdit(); self.uniprot.setPlaceholderText("e.g. Q96SW2")
        self.chain = QLineEdit(); self.chain.setPlaceholderText("chain to build (only if several chains)")
        card.body.addWidget(field("Structure", self.input))
        gaps_row = QHBoxLayout()
        gaps = action_button("List gaps"); gaps.setObjectName(""); gaps.clicked.connect(self._list_gaps)
        gaps_row.addWidget(gaps); gaps_row.addStretch()
        card.body.addLayout(gaps_row)
        self.status = QLabel(""); self.status.setObjectName("Muted"); self.status.setWordWrap(True)
        card.body.addWidget(self.status)
        card.body.addWidget(field("Method", self.method))
        card.body.addWidget(field("Output", self.output))
        card.body.addWidget(self.include_termini)
        card.body.addWidget(self.add_h)
        card.body.addWidget(field("Max gap length (pdbfixer; 0 = no limit)", self.max_gap))
        card.body.addWidget(field("Reference sequence to fill missing residues", self.seq_source))
        self._seq_paste_row = field("Paste one-letter sequence", self.seq_paste)
        self._seq_fasta_row = field("FASTA / .fa file", self.fasta)
        self._seq_uniprot_row = field("UniProt accession", self.uniprot)
        card.body.addWidget(self._seq_paste_row)
        card.body.addWidget(self._seq_fasta_row)
        card.body.addWidget(self._seq_uniprot_row)
        card.body.addWidget(field("Chain to build (if several chains)", self.chain))
        card.body.addWidget(field("Refine existing loops instead (kic)", self.loops))
        self.seq_source.currentTextChanged.connect(self._update_seq_source)
        self._update_seq_source(self.seq_source.currentText())
        layout.addWidget(card)
        row = QHBoxLayout()
        run = action_button("Run"); run.clicked.connect(self._run)
        row.addStretch(); row.addWidget(run)
        layout.addLayout(row)
        layout.addStretch()
        _scrollable(self, content)

    def _list_gaps(self):
        from chemflow.protac.fix_structure import find_gaps

        if not self.input.text():
            QMessageBox.warning(self, "Missing input", "Select a structure.")
            return
        try:
            gaps = find_gaps(Path(self.input.text()).read_text(errors="replace"))
        except Exception as exc:
            QMessageBox.warning(self, "Cannot read", str(exc))
            return
        if not gaps:
            self.status.setText("No internal chain gaps found (terminal gaps aren't detectable).")
            return
        self.status.setText("Internal gaps: " + "; ".join(
            f"{g.chain}:{g.after_resseq}-{g.before_resseq} (missing {g.length})" for g in gaps))

    def _update_seq_source(self, text: str) -> None:
        self._seq_paste_row.setVisible(text == "Paste sequence")
        self._seq_fasta_row.setVisible(text == "FASTA file")
        self._seq_uniprot_row.setVisible(text == "UniProt ID")

    def _sequence_args(self) -> list[str]:
        """CLI args for the chosen reference-sequence source (may be empty)."""
        source = self.seq_source.currentText()
        chain = self.chain.text().strip()
        prefix = f"{chain}=" if chain else ""
        if source == "Paste sequence":
            seq = "".join(self.seq_paste.toPlainText().split())
            return ["--sequence", prefix + seq] if seq else []
        if source == "FASTA file":
            path = self.fasta.text().strip()
            return ["--fasta", prefix + path] if path else []
        acc = self.uniprot.text().strip()
        return ["--uniprot", prefix + acc] if acc else []

    def _run(self):
        if not self.input.text():
            QMessageBox.warning(self, "Missing input", "Select a structure.")
            return
        # Resolve the output to an absolute path next to the input, so it isn't
        # written to the app's launch directory where it can't be found.
        in_path = Path(self.input.text()).expanduser().resolve()
        out_text = self.output.text().strip() or f"{in_path.stem}_fixed.pdb"
        out_path = Path(out_text).expanduser()
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path      # the working folder
        self.output.setText(str(out_path))
        args = ["protac", "fix-structure", "--input", str(in_path),
                "-o", str(out_path), "--method", self.method.currentText()]
        seq_args = self._sequence_args()
        if self.method.currentText() == "pdbfixer":
            if self.include_termini.isChecked():
                args.append("--include-termini")
            if self.add_h.isChecked():
                args.append("--add-hydrogens")
            if self.max_gap.value() > 0:
                args += ["--max-gap", str(self.max_gap.value())]
            args += seq_args
        else:  # kic
            if seq_args:
                args += seq_args
                if self.chain.text().strip():
                    args += ["--chain", self.chain.text().strip()]
            else:
                loops = [s.strip() for s in self.loops.text().split(",") if s.strip()]
                if not loops:
                    QMessageBox.warning(self, "KIC needs input",
                                        "Give a reference sequence (paste / FASTA / UniProt) to build "
                                        "missing residues, or loops (e.g. A:45-52) to refine existing ones.")
                    return
                for lp in loops:
                    args += ["--loop", lp]
        if self._runner is None:
            QMessageBox.information(self, "Run from terminal", "chemflow " + " ".join(args))
            return
        try:
            self._runner(args)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot start task", str(exc))


class PyRosettaDockPage(QWidget):
    """In-process protein-protein docking with PyRosetta (ternary search)."""

    def __init__(self, runner=None, parent=None):
        super().__init__(parent)
        self._runner = runner
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(18)
        layout.addWidget(PageHeader("Ternary search (PyRosetta)", "Protein-protein docking",
                                    "In-process docking; proteins-only by default (no ligand params)."))
        card = Card("Inputs", "One PDB with both partners as chains (see Merge PDB).")
        self.input_pdb = PathField("Both partners in one PDB")
        self.partners = QLineEdit("A_B")
        self.mode = QComboBox(); self.mode.addItems(["global", "local"])
        self.nstruct = QSpinBox(); self.nstruct.setRange(1, 1000000); self.nstruct.setValue(100)
        self.proteins_only = QCheckBox("Dock proteins only (no ligand params)"); self.proteins_only.setChecked(True)
        self.pre_relax = QCheckBox("Pre-relax input (FastRelax)")
        self.post_relax = QCheckBox("Post-relax each model (slow)")
        self.out_dir = PathField("Output folder", mode="directory"); self.out_dir.setText("pyrosetta_dock_output")
        # Residue-residue distance restraint: steer docking toward an arrangement.
        self.restraint_on = QCheckBox("Distance restraint between two protein residues")
        self.r_atom1 = QLineEdit(); self.r_atom1.setPlaceholderText("residue on partner 1: chain resseq, e.g. A 123")
        self.r_atom2 = QLineEdit(); self.r_atom2.setPlaceholderText("residue on partner 2: e.g. B 456")
        self.r_dist = QDoubleSpinBox(); self.r_dist.setRange(0.1, 100.0); self.r_dist.setValue(10.0)
        self.r_std = QDoubleSpinBox(); self.r_std.setRange(0.1, 20.0); self.r_std.setValue(2.0)
        card.body.addWidget(field("Input structure", self.input_pdb))
        row = QWidget(); r = QHBoxLayout(row); r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(field("Partners", self.partners), 1)
        r.addWidget(field("Mode", self.mode), 1)
        r.addWidget(field("Models", self.nstruct), 1)
        card.body.addWidget(row)
        card.body.addWidget(self.proteins_only)
        card.body.addWidget(self.pre_relax)
        card.body.addWidget(self.post_relax)
        card.body.addWidget(field("Output folder", self.out_dir))
        card.body.addWidget(self.restraint_on)
        card.body.addWidget(field("Restraint residue on partner 1", self.r_atom1))
        card.body.addWidget(field("Restraint residue on partner 2", self.r_atom2))
        rrow = QWidget(); rr = QHBoxLayout(rrow); rr.setContentsMargins(0, 0, 0, 0)
        rr.addWidget(field("Target distance (Å)", self.r_dist), 1)
        rr.addWidget(field("Std dev (Å)", self.r_std), 1)
        card.body.addWidget(rrow)
        layout.addWidget(card)
        row2 = QHBoxLayout()
        val = action_button("Validate"); val.setObjectName(""); val.clicked.connect(self._validate)
        run = action_button("Run docking"); run.clicked.connect(self._run)
        row2.addWidget(val); row2.addStretch(); row2.addWidget(run)
        layout.addLayout(row2)
        layout.addStretch()
        _scrollable(self, content)

    def _write_yaml(self) -> Path:
        if not self.input_pdb.text():
            raise ValueError("Select the input PDB.")
        in_path = Path(self.input_pdb.text()).expanduser().resolve()
        out_dir = Path(self.out_dir.text().strip() or "pyrosetta_dock_output").expanduser()
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir        # the working folder
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.setText(str(out_dir))
        data = {
            "input_pdb": str(in_path),
            "partners": self.partners.text() or "A_B",
            "mode": self.mode.currentText(),
            "nstruct": self.nstruct.value(),
            "dock_proteins_only": self.proteins_only.isChecked(),
            "pre_relax": self.pre_relax.isChecked(),
            "post_relax": self.post_relax.isChecked(),
            "workdir": str(out_dir.parent),
            "out_dir": str(out_dir),
        }
        if self.restraint_on.isChecked():
            data["restraint"] = {
                "res1": self._parse_residue(self.r_atom1.text(), "partner 1"),
                "res2": self._parse_residue(self.r_atom2.text(), "partner 2"),
                "distance": self.r_dist.value(),
                "stddev": self.r_std.value(),
            }
        path = out_dir / "pyrosetta_dock.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    @staticmethod
    def _parse_residue(text: str, label: str) -> dict:
        parts = text.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError(f"{label} residue must be 'chain resseq', e.g. A 123 (got {text!r})")
        chain, resseq = parts
        return {"chain": chain, "resseq": int(resseq)}

    def _validate(self):
        from chemflow.protac.pyrosetta_dock import load_config, validate_config

        try:
            validate_config(load_config(self._write_yaml()))
        except Exception as exc:
            QMessageBox.warning(self, "Invalid configuration", str(exc))
            return
        QMessageBox.information(self, "Valid", "The docking configuration is valid.")

    def _run(self):
        try:
            yaml_path = self._write_yaml()
        except Exception as exc:
            QMessageBox.warning(self, "Cannot run", str(exc))
            return
        if self._runner is None:
            QMessageBox.information(self, "Run from terminal",
                                   f"chemflow protac pyrosetta-dock run {yaml_path}")
            return
        try:
            self._runner(["protac", "pyrosetta-dock", "run", str(yaml_path)])
        except Exception as exc:
            QMessageBox.warning(self, "Cannot start task", str(exc))


class ScreenPage(QWidget):
    """Rank docked models by I_sc and check warhead linkability."""

    def __init__(self, parent=None):
        super().__init__(parent)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(18)
        layout.addWidget(PageHeader("Screen", "Rank models + linkability",
                                    "Rank docked models by I_sc and measure warhead distance."))
        card = Card("Inputs", "A docking output folder, or a score.csv (+ model PDBs).")
        self.dir = PathField("Output folder or score.csv", mode="file")
        self.models_dir = PathField("Models folder (if not next to the CSV)", mode="directory")
        self.atom1 = QLineEdit(); self.atom1.setPlaceholderText("target warhead atom, e.g. LIG:C7")
        self.atom2 = QLineEdit(); self.atom2.setPlaceholderText("E3 warhead atom, e.g. VH1:C3")
        self.top = QSpinBox(); self.top.setRange(1, 100000); self.top.setValue(20)
        card.body.addWidget(field("Output folder or score.csv", self.dir))
        card.body.addWidget(field("Models folder (optional, if separate from the CSV)", self.models_dir))
        row = QWidget(); r = QHBoxLayout(row); r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(field("Target warhead atom", self.atom1), 1)
        r.addWidget(field("E3 warhead atom", self.atom2), 1)
        r.addWidget(field("Top N", self.top), 1)
        card.body.addWidget(row)
        layout.addWidget(card)
        btn = action_button("Screen")
        btn.clicked.connect(self._screen)
        layout.addWidget(btn)
        self.result = QPlainTextEdit(); self.result.setReadOnly(True); self.result.setMinimumHeight(180)
        layout.addWidget(self.result, 1)
        _scrollable(self, content)

    def _screen(self):
        from chemflow.protac.screen import screen_models

        if not self.dir.text() or not self.atom1.text() or not self.atom2.text():
            QMessageBox.warning(self, "Missing input", "Select the folder and both warhead atoms.")
            return
        try:
            rows = screen_models(self.dir.text(), self.atom1.text(), self.atom2.text(),
                                 top=self.top.value(),
                                 models_dir=self.models_dir.text() or None)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot screen", str(exc))
            return
        lines = [f"{'model':<18}{'I_sc':>9}{'dist(A)':>9}  verdict"]
        for r in rows:
            isc = "n/a" if r.i_sc is None else f"{r.i_sc:.2f}"
            dist = "n/a" if r.distance is None else f"{r.distance:.2f}"
            lines.append(f"{r.model:<18}{isc:>9}{dist:>9}  {r.verdict}")
        linkable = sum(1 for r in rows if r.verdict == "linkable")
        lines.append(f"\n{linkable} of {len(rows)} are linkable; their PDBs were copied to a "
                     "'linkable' folder next to the scores.")
        self.result.setPlainText("\n".join(lines))


class RosettaEnvSetupPage(QWidget):
    """Create/link the PyRosetta conda env (one-click), runs `chemflow rosetta-env`."""

    def __init__(self, console=None, parent=None):
        super().__init__(parent)
        self._console = console
        layout = QVBoxLayout(self)
        layout.addWidget(PageHeader(
            "PROTAC design", "PyRosetta environment",
            "PyRosetta runs in its own conda env (CPU). Install it once; the PyRosetta "
            "steps (ternary dock, KIC fix-structure) then run through it. Needs a "
            "RosettaCommons license."))
        row = QHBoxLayout()
        row.addWidget(QLabel("Env name:"))
        self.env_name = QLineEdit("pyrosetta")
        row.addWidget(self.env_name)
        row.addWidget(QLabel("Method:"))
        self.method = QComboBox()
        self.method.addItems(["pip-installer", "conda"])
        row.addWidget(self.method)
        row.addStretch()
        layout.addLayout(row)
        buttons = QHBoxLayout()
        for label, slot in [("Install PyRosetta env", self._install),
                            ("Preview commands", self._dry),
                            ("Check environment", self._status)]:
            btn = action_button(label)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.note = QLabel("Install streams to the Activity console. 'pip-installer' uses the "
                           "pyrosetta-installer helper; 'conda' uses the RosettaCommons channel.")
        self.note.setObjectName("Muted")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        layout.addStretch()

    def _argv(self, dry: bool) -> list[str]:
        argv = ["rosetta-env", "install", "--env", self.env_name.text().strip() or "pyrosetta",
                "--method", self.method.currentText()]
        if dry:
            argv.append("--dry-run")
        return argv

    def _install(self):
        if self._console is not None:
            self._console.run(self._argv(dry=False))

    def _dry(self):
        if self._console is not None:
            self._console.run(self._argv(dry=True))

    def _status(self):
        if self._console is not None:
            self._console.run(["rosetta-env", "status"])


class ProtacModulePage(QWidget):
    """PROTAC design module: all sub-modules as tabs (each built on first open)."""

    def __init__(self, runner=None, parent=None, console=None, rosetta_runner=None):
        super().__init__(parent)
        self._runner = runner
        self._console = console
        # PyRosetta-backed steps run through the PyRosetta env's interpreter.
        rose = rosetta_runner or runner
        # (tab title, factory). Order follows the workflow.
        self._subs = [
            ("Fix structure", lambda: FixStructurePage(rose)),
            ("Merge PDB", lambda: MergePdbPage()),
            ("Ternary search · PyRosetta", lambda: PyRosettaDockPage(rose)),
            ("Ternary search · RosettaDock", lambda: RosettaDockPage(runner)),
            ("Screen", lambda: ScreenPage()),
            ("Linkability", lambda: LinkabilityPage()),
            ("Linker designer", self._make_linker),
        ]
        self._built: set[int] = set()
        self._containers: list[QWidget] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        for title, _factory in self._subs:
            container = QWidget()
            box = QVBoxLayout(container)
            box.setContentsMargins(0, 0, 0, 0)
            self._tabs.addTab(container, title)
            self._containers.append(container)
        self._tabs.currentChanged.connect(self._ensure_tab)
        layout.addWidget(self._tabs)
        self._ensure_tab(0)

    def _make_linker(self):
        from chemflow.desktop.linker_designer_page import LinkerDesignerPage
        return LinkerDesignerPage(console=self._console)

    def _ensure_tab(self, index: int) -> None:
        if index in self._built or index < 0 or index >= len(self._subs):
            return
        try:
            widget = self._subs[index][1]()
        except Exception as exc:  # a heavy import failing shows, not crashes the module
            widget = QLabel(f"This sub-module could not be loaded:\n{exc}")
            widget.setWordWrap(True)
            widget.setObjectName("Muted")
        self._containers[index].layout().addWidget(widget)
        self._built.add(index)
