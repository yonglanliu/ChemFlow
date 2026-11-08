"""Environments module for ChemFlow Studio — install the compute conda envs.

A single place to provision the external environments ChemFlow drives, each as its
own card: pick the env, options (GPU/CUDA, recreate), and Install / Preview / Check.
Everything runs ``chemflow <tool>-env …`` through the shared Activity console.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.widgets.forms import Card, PageHeader, action_button


@dataclass
class _EnvSpec:
    key: str                    # CLI subcommand, e.g. "fe-env"
    title: str
    description: str
    default_env: str
    gpu: bool = False           # offer a GPU/CUDA option
    recreate: bool = False      # offer a recreate option
    status_env: bool = True     # does `status` accept --env?
    extra: list[tuple[str, str]] = field(default_factory=list)  # (flag, label) toggles


ENVIRONMENTS = [
    _EnvSpec(
        "fe-env", "AlchemForge — OpenFE + GROMACS",
        "Binding free energy: OpenFE (RBFE/ABFE/SepTop), OpenMM MD, and the GROMACS/PMX "
        "RBFE pipeline. Includes RDKit, Open Babel, ACPYPE, AmberTools, GROMACS (+ PMX).",
        default_env="alchemforge", gpu=True, recreate=True),
    _EnvSpec(
        "alchemfold-env", "AlchemFold — Boltz-2 + Chai-1",
        "Co-folding / docking structure prediction (Boltz-2 and Chai-1). Pulls a Torch "
        "stack; set GPU for a CUDA build.",
        default_env="alchemfold", gpu=True, recreate=True),
    _EnvSpec(
        "rosetta-env", "PyRosetta — PROTAC docking",
        "PyRosetta for PROTAC ternary docking and KIC loop building.",
        default_env="pyrosetta", gpu=False, recreate=False, status_env=False),
    _EnvSpec(
        "reinvent-env", "REINVENT4 — generative design",
        "REINVENT4 for generative molecule / linker design.",
        default_env="reinvent", gpu=False, recreate=False, status_env=False),
]


class EnvironmentsPage(QWidget):
    """Browse and install the compute environments ChemFlow links to."""

    def __init__(self, console=None, parent=None):
        super().__init__(parent)
        self._console = console
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(20)
        layout.addWidget(PageHeader(
            "Environments", "Install compute environments",
            "Each heavy toolchain lives in its own conda env, linked to ChemFlow. "
            "Pick one and install — it streams to the Activity console. On a laptop "
            "leave GPU off; on HPC tick GPU and set the CUDA version."))
        for spec in ENVIRONMENTS:
            layout.addWidget(self._env_card(spec))
        layout.addStretch()

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _env_card(self, spec: _EnvSpec) -> Card:
        card = Card(spec.title, spec.description)
        state: dict = {}

        row = QHBoxLayout()
        row.addWidget(QLabel("Env name:"))
        env_name = QLineEdit(spec.default_env)
        env_name.setMaximumWidth(180)
        row.addWidget(env_name)
        state["env"] = env_name

        if spec.gpu:
            gpu = QCheckBox("GPU (CUDA)")
            row.addWidget(gpu)
            cuda = QLineEdit("12.6")
            cuda.setMaximumWidth(70)
            row.addWidget(QLabel("CUDA:"))
            row.addWidget(cuda)
            state["gpu"], state["cuda"] = gpu, cuda
        if spec.recreate:
            recreate = QCheckBox("Recreate")
            recreate.setToolTip("Remove the env first and rebuild cleanly (else update in place).")
            row.addWidget(recreate)
            state["recreate"] = recreate
        for flag, label in spec.extra:
            box = QCheckBox(label)
            row.addWidget(box)
            state.setdefault("extra", []).append((flag, box))
        row.addStretch()
        card.body.addLayout(row)

        buttons = QHBoxLayout()
        install = action_button("Install")
        install.clicked.connect(lambda _=False, s=spec, st=state: self._run(s, st, dry=False))
        preview = action_button("Preview commands")
        preview.clicked.connect(lambda _=False, s=spec, st=state: self._run(s, st, dry=True))
        check = action_button("Check")
        check.clicked.connect(lambda _=False, s=spec, st=state: self._status(s, st))
        buttons.addWidget(install)
        buttons.addWidget(preview)
        buttons.addWidget(check)
        buttons.addStretch()
        card.body.addLayout(buttons)
        return card

    def _install_argv(self, spec: _EnvSpec, state: dict, dry: bool) -> list[str]:
        argv = [spec.key, "install", "--env", state["env"].text().strip() or spec.default_env]
        if spec.gpu and state["gpu"].isChecked() and state["cuda"].text().strip():
            argv += ["--cuda", state["cuda"].text().strip()]
        if spec.recreate and state["recreate"].isChecked():
            argv.append("--recreate")
        for flag, box in state.get("extra", []):
            if box.isChecked():
                argv.append(flag)
        if dry:
            argv.append("--dry-run")
        return argv

    def _run(self, spec: _EnvSpec, state: dict, dry: bool) -> None:
        if self._console is not None:
            self._console.run(self._install_argv(spec, state, dry))

    def _status(self, spec: _EnvSpec, state: dict) -> None:
        if self._console is None:
            return
        argv = [spec.key, "status"]
        if spec.status_env:
            argv += ["--env", state["env"].text().strip() or spec.default_env]
        self._console.run(argv)
