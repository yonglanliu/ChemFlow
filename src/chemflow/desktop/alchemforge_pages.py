"""Binding-free-energy modules for ChemFlow Studio (drive the ``alchemforge`` CLI).

Two dashboard modules bring AlchemForge's alchemical workflows into ChemFlow as
native PySide6 pages, each a tab group of sub-workflows:

* **Binding Free Energy · GROMACS** — RBFE, ABFE, SepTop, resFEP, Alanine scan,
  MM-GBSA (GROMACS/PMX engine; TOML/YAML configs).
* **Binding Free Energy · OpenFE** — RBFE, ABFE, SepTop, OpenMM modeling/MD
  (OpenFE engine; native YAML configs).

Rather than re-implement AlchemForge's science in the GUI, each tab builds a
command line for the ``alchemforge`` CLI and runs it through ChemFlow's shared
Activity console (the same ``python -m`` subprocess mechanism used elsewhere).
This avoids the PyQt5/PySide6 split between the two desktops.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.cli_bridge import ALCHEMFORGE_MODULE, render_step
from chemflow.desktop.widgets.forms import PageHeader, PathField, action_button

_ALCHEMFORGE_MODULE = ALCHEMFORGE_MODULE


class FeEnvSetupPage(QWidget):
    """Create/link the one compute env (OpenFE + GROMACS + OpenMM) for free energy."""

    def __init__(self, console=None, parent=None):
        super().__init__(parent)
        self._console = console
        layout = QVBoxLayout(self)
        layout.addWidget(PageHeader(
            "Binding Free Energy", "Environment setup",
            "One conda env holds the whole compute stack — OpenFE, OpenMM, RDKit, "
            "Open Babel, ACPYPE, AmberTools, GROMACS (+ PMX) — kept separate from "
            "ChemFlow's ML stack. Install it once; the desktop auto-detects it."))

        row = QHBoxLayout()
        row.addWidget(QLabel("Env name:"))
        self.env_name = QLineEdit("alchemforge")
        row.addWidget(self.env_name)
        self.gpu = QCheckBox("GPU (CUDA)")
        self.gpu.setToolTip("Tick for HPC/GPU machines; sets CUDA-enabled OpenMM/GROMACS builds")
        row.addWidget(self.gpu)
        row.addWidget(QLabel("CUDA version:"))
        self.cuda = QLineEdit("12.4")
        self.cuda.setMaximumWidth(80)
        row.addWidget(self.cuda)
        self.recreate = QCheckBox("Recreate")
        self.recreate.setToolTip("Remove the env first and rebuild cleanly (else update in place).")
        row.addWidget(self.recreate)
        row.addStretch()
        layout.addLayout(row)

        buttons = QHBoxLayout()
        for label, slot in [("Install compute env", self._install),
                            ("Preview commands", self._dry_run),
                            ("Check environment", self._status)]:
            btn = action_button(label)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.note = QLabel("Install runs mamba/conda to create the env and link this package into it, "
                           "streaming to the Activity console. On a laptop leave GPU unticked "
                           "(CPU/OpenCL); on HPC tick GPU and set the cluster's CUDA version.")
        self.note.setObjectName("Muted")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        layout.addStretch()

    def _argv(self, dry: bool) -> list[str]:
        argv = ["fe-env", "install", "--env", self.env_name.text().strip() or "alchemforge"]
        if self.gpu.isChecked() and self.cuda.text().strip():
            argv += ["--cuda", self.cuda.text().strip()]
        if self.recreate.isChecked():
            argv.append("--recreate")
        if dry:
            argv.append("--dry-run")
        return argv

    def _install(self) -> None:
        if self._console is not None:
            self._console.run(self._argv(dry=False))

    def _dry_run(self) -> None:
        if self._console is not None:
            self._console.run(self._argv(dry=True))

    def _status(self) -> None:
        if self._console is not None:
            self._console.run(["fe-env", "status", "--env",
                               self.env_name.text().strip() or "alchemforge"])


class GmxEnvSetupPage(QWidget):
    """Create/link the GROMACS/PMX conda env (openbabel, acpype, ambertools, gromacs, pmx)."""

    def __init__(self, console=None, parent=None):
        super().__init__(parent)
        self._console = console
        layout = QVBoxLayout(self)
        layout.addWidget(PageHeader(
            "Binding Free Energy · GROMACS", "Environment setup",
            "The GROMACS/PMX pipeline needs its own conda env (Open Babel, ACPYPE, "
            "AmberTools, GROMACS, PMX). Install it once; then point the FE interpreter "
            "at it or declare it in chemflow_tools.yaml."))

        row = QHBoxLayout()
        row.addWidget(QLabel("Env name:"))
        self.env_name = QLineEdit("alchemforge")
        row.addWidget(self.env_name)
        self.gpu = QCheckBox("GPU (CUDA)")
        self.gpu.setToolTip("Tick for an HPC GPU GROMACS build; sets cuda-version.")
        row.addWidget(self.gpu)
        row.addWidget(QLabel("CUDA version:"))
        self.cuda = QLineEdit("12.4")
        self.cuda.setMaximumWidth(80)
        row.addWidget(self.cuda)
        self.with_pmx = QCheckBox("Install PMX")
        self.with_pmx.setChecked(True)
        row.addWidget(self.with_pmx)
        self.recreate = QCheckBox("Recreate")
        self.recreate.setToolTip("Remove the env first and rebuild it cleanly. "
                                 "Leave unticked to install/update an existing env (safe retry).")
        row.addWidget(self.recreate)
        row.addStretch()
        layout.addLayout(row)

        buttons = QHBoxLayout()
        for label, slot in [("Install GROMACS env", self._install),
                            ("Preview commands", self._dry_run),
                            ("Check environment", self._status)]:
            btn = action_button(label)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.note = QLabel("Install runs mamba/conda to create the env (openbabel, acpype, "
                           "ambertools, gromacs [+ pmx]) and link chemflow into it, streaming to "
                           "the Activity console. On HPC tick GPU and set the CUDA version; "
                           "on a laptop leave it unticked (CPU GROMACS).")
        self.note.setObjectName("Muted")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        layout.addStretch()

    def _argv(self, dry: bool) -> list[str]:
        argv = ["gmx-env", "install", "--env", self.env_name.text().strip() or "alchemforge"]
        if self.gpu.isChecked() and self.cuda.text().strip():
            argv += ["--cuda", self.cuda.text().strip()]
        if not self.with_pmx.isChecked():
            argv.append("--no-pmx")
        if self.recreate.isChecked():
            argv.append("--recreate")
        if dry:
            argv.append("--dry-run")
        return argv

    def _install(self) -> None:
        if self._console is not None:
            self._console.run(self._argv(dry=False))

    def _dry_run(self) -> None:
        if self._console is not None:
            self._console.run(self._argv(dry=True))

    def _status(self) -> None:
        if self._console is not None:
            self._console.run(["gmx-env", "status", "--env", self.env_name.text().strip() or "alchemforge"])


class CliWorkflowPage(QWidget):
    """One alchemical sub-workflow: a config path + generate/validate/run buttons."""

    def __init__(
        self,
        *,
        eyebrow: str,
        title: str,
        subtitle: str,
        config_placeholder: str,
        steps: list[tuple[str, list[str]]],
        console=None,
        generate_argv: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._console = console
        self._steps = steps
        self._generate_argv = generate_argv

        layout = QVBoxLayout(self)
        layout.addWidget(PageHeader(eyebrow, title, subtitle))

        self.config = PathField(config_placeholder, mode="save")
        layout.addWidget(QLabel("Configuration file:"))
        layout.addWidget(self.config)

        if generate_argv is not None:
            gen = action_button("Generate example config")
            gen.setToolTip("Write a ready-to-edit config at the path above, then edit it")
            gen.clicked.connect(self._generate)
            layout.addWidget(gen)

        buttons = QHBoxLayout()
        for label, argv in steps:
            btn = action_button(label)
            btn.clicked.connect(lambda _=False, a=argv: self._run(a))
            buttons.addWidget(btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.note = QLabel("Set a config path, Generate it, edit the inputs, then Validate → Run. "
                           "Output appears in the Activity console below.")
        self.note.setObjectName("Muted")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        layout.addStretch()

    def _config_or_warn(self) -> str | None:
        cfg = self.config.text()
        if not cfg:
            self.note.setText("Choose a configuration file path first.")
            return None
        return cfg

    def _generate(self) -> None:
        cfg = self._config_or_warn()
        if cfg is None or self._console is None:
            return
        self._console.run_alchemforge(render_step(self._generate_argv, cfg))

    def _run(self, argv_template: list[str]) -> None:
        cfg = self._config_or_warn()
        if cfg is None or self._console is None:
            return
        self._console.run_alchemforge(render_step(argv_template, cfg))


# ---------------------------------------------------------------------------
# GROMACS / PMX module
# ---------------------------------------------------------------------------

class BindingFreeEnergyGromacsModule(QTabWidget):
    """Binding Free Energy · GROMACS — RBFE/ABFE/SepTop/resFEP/alanine/MM-GBSA."""

    def __init__(self, console=None, parent=None):
        super().__init__(parent)
        self._console = console
        self.setDocumentMode(True)
        self._subs = [
            ("RBFE", self._rbfe),
            ("ABFE", self._abfe),
            ("SepTop", self._septop),
            ("resFEP", self._resfep),
            ("Alanine scan", self._alanine),
            ("MM-GBSA", self._mmgbsa),
        ]
        self._built: dict[int, QWidget] = {}
        self._containers: list[QWidget] = []
        for title, _factory in self._subs:
            container = QWidget()
            QVBoxLayout(container).setContentsMargins(0, 0, 0, 0)
            self.addTab(container, title)
            self._containers.append(container)
        self.currentChanged.connect(self._ensure)
        self._ensure(0)

    def _ensure(self, index: int) -> None:
        if index in self._built or index < 0 or index >= len(self._subs):
            return
        page = self._subs[index][1]()
        self._containers[index].layout().addWidget(page)
        self._built[index] = page

    def _rbfe(self):
        from chemflow.desktop.gromacs_rbfe_page import GromacsRbfePage

        return GromacsRbfePage(console=self._console)

    def _abfe(self):
        return CliWorkflowPage(
            eyebrow="Binding Free Energy · GROMACS", title="Absolute BFE (ABFE)",
            subtitle="Absolute binding free energy for a single ligand (GROMACS/PMX).",
            config_placeholder="abfe.toml", console=self._console,
            generate_argv=["generate", "abfe", "{name}", "--engine", "gromacs"],
            steps=[("Validate", ["abfe-gromacs", "validate", "{config}"]),
                   ("Stage", ["abfe-gromacs", "stage", "{config}"]),
                   ("Run", ["abfe-gromacs", "run", "{config}"]),
                   ("Analyze", ["abfe-gromacs", "analyze", "{config}"])])

    def _septop(self):
        return CliWorkflowPage(
            eyebrow="Binding Free Energy · GROMACS", title="Separated Topologies (SepTop)",
            subtitle="Separated-topology RBFE (GROMACS).",
            config_placeholder="septop.toml", console=self._console,
            generate_argv=["generate", "septop", "{name}", "--engine", "gromacs"],
            steps=[("Validate", ["septop-gromacs", "validate", "{config}"]),
                   ("Staging", ["septop-gromacs", "staging", "{config}"]),
                   ("Prepare", ["septop-gromacs", "prepare", "{config}"]),
                   ("Submit", ["septop-gromacs", "submit", "{config}"])])

    def _resfep(self):
        return CliWorkflowPage(
            eyebrow="Binding Free Energy · GROMACS", title="Residue FEP (resFEP)",
            subtitle="Protein residue-mutation free energy (GROMACS/PMX).",
            config_placeholder="resfep.toml", console=self._console,
            generate_argv=["generate", "res_rbfe", "{name}", "--engine", "gromacs"],
            steps=[("Validate", ["resfep", "validate", "{config}"]),
                   ("Prepare ligand", ["resfep", "prepare-ligand", "{config}"]),
                   ("Prepare system", ["resfep", "prepare-system", "{config}"]),
                   ("Stage", ["resfep", "stage", "{config}"]),
                   ("Submit", ["resfep", "submit", "{config}"])])

    def _alanine(self):
        return CliWorkflowPage(
            eyebrow="Binding Free Energy · GROMACS", title="Alanine scan",
            subtitle="Per-residue → ALA resFEP batch over an interface.",
            config_placeholder="alanine_scan.toml", console=self._console,
            generate_argv=None,
            steps=[("Stage", ["alanine-scan", "stage", "{config}"]),
                   ("Submit", ["alanine-scan", "submit", "{config}"]),
                   ("Analyze", ["alanine-scan", "analyze", "{config}"])])

    def _mmgbsa(self):
        return CliWorkflowPage(
            eyebrow="Binding Free Energy · GROMACS", title="MM-GBSA endpoint",
            subtitle="Fast single-trajectory MM-GBSA triage from an MD trajectory.",
            config_placeholder="mmgbsa.yaml", console=self._console,
            generate_argv=["mmgbsa", "generate", "{config}"],
            steps=[("Validate", ["mmgbsa", "validate", "{config}"]),
                   ("Stage", ["mmgbsa", "stage", "{config}"]),
                   ("Run", ["mmgbsa", "run", "{config}"])])

    def _env(self):
        # One compute env (OpenFE + GROMACS) serves both modules — reuse the same page.
        return FeEnvSetupPage(console=self._console)


# ---------------------------------------------------------------------------
# OpenFE module
# ---------------------------------------------------------------------------

class BindingFreeEnergyOpenFEModule(QTabWidget):
    """Binding Free Energy · OpenFE — RBFE/ABFE/SepTop + OpenMM modeling/MD."""

    def __init__(self, console=None, parent=None):
        super().__init__(parent)
        self._console = console
        self.setDocumentMode(True)
        self._subs = [
            ("RBFE", self._rbfe),
            ("ABFE", self._abfe),
            ("SepTop", self._septop),
            ("OpenMM modeling / MD", self._openmm),
        ]
        self._built: dict[int, QWidget] = {}
        self._containers: list[QWidget] = []
        for title, _factory in self._subs:
            container = QWidget()
            QVBoxLayout(container).setContentsMargins(0, 0, 0, 0)
            self.addTab(container, title)
            self._containers.append(container)
        self.currentChanged.connect(self._ensure)
        self._ensure(0)

    def _ensure(self, index: int) -> None:
        if index in self._built or index < 0 or index >= len(self._subs):
            return
        page = self._subs[index][1]()
        self._containers[index].layout().addWidget(page)
        self._built[index] = page

    def _openfe_page(self, workflow: str, title: str, subtitle: str):
        return CliWorkflowPage(
            eyebrow="Binding Free Energy · OpenFE", title=title, subtitle=subtitle,
            config_placeholder=f"{workflow}.yaml", console=self._console,
            generate_argv=["generate", workflow, "{name}", "--engine", "openfe"],
            steps=[("Validate", ["openfe", workflow, "validate", "{config}"]),
                   ("Generate JSON", ["openfe", workflow, "generate", "{config}"]),
                   ("Run", ["openfe", workflow, "run", "{config}"])])

    def _env(self):
        return FeEnvSetupPage(console=self._console)

    def _rbfe(self):
        from chemflow.desktop.openfe_rbfe_page import OpenFERbfeWorkflowPage

        return OpenFERbfeWorkflowPage(console=self._console)

    def _abfe(self):
        from chemflow.desktop.openfe_abfe_page import OpenFEAbfeWorkflowPage

        return OpenFEAbfeWorkflowPage(console=self._console)

    def _septop(self):
        from chemflow.desktop.openfe_septop_page import OpenFESeptopWorkflowPage

        return OpenFESeptopWorkflowPage(console=self._console)

    def _openmm(self):
        from chemflow.desktop.openmm_md_page import OpenMMModelingPage

        return OpenMMModelingPage(console=self._console)
