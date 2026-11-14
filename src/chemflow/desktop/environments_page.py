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
    cuda_default: str = "12.6"  # prefilled CUDA version for the GPU option
    path_flag: str = ""         # optional folder option, e.g. "--dir"
    path_label: str = ""
    path_default: str = ""
    warmup: bool = False        # offer a "Warm-up" button (`<key> warmup`)


def _default_diffdock_dir() -> str:
    """<ChemFlow root>/DiffDock — a sibling of src/ (like REINVENT4/), not in HOME."""
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        if (parent / "env" / "environment-diffdock.yaml").is_file():
            return str(parent / "DiffDock")
    return "~/DiffDock"


ENVIRONMENTS = [
    _EnvSpec(
        "fe-env", "AlchemForge — OpenFE + GROMACS",
        "Binding free energy: OpenFE (RBFE/ABFE/SepTop), OpenMM MD, and the GROMACS/PMX "
        "RBFE pipeline. Includes RDKit, Open Babel, ACPYPE, AmberTools, GROMACS (+ PMX).",
        default_env="alchemforge", gpu=True, recreate=True),
    _EnvSpec(
        "boltz-env", "Boltz-2 — co-folding / docking (own env)",
        "Boltz-2 with the optimized cuEquivariance triangle kernels: faster and much "
        "lighter on memory for big complexes. Its own env because the kernels need a "
        "recent PyTorch that Chai-1 forbids. Check prints the boltz path and whether "
        "the kernels import; the docking page finds this env automatically.",
        default_env="boltz", gpu=True, recreate=True),
    _EnvSpec(
        "chai-env", "Chai-1 — co-folding / docking (own env)",
        "Chai-1 in its own env (it pins torch<2.7, so it cannot share with Boltz-2). "
        "Point CHAI_DOWNLOADS_DIR at /data or scratch — the weights are large and "
        "download on first use.",
        default_env="chai", gpu=True, recreate=True),
    _EnvSpec(
        "diffdock-env", "DiffDock-L — blind docking (own env)",
        "DiffDock-L (gcorso/DiffDock) in its OWN env, separate from AlchemFold: creates the conda "
        "env from env/environment-diffdock.yaml and clones the repo. After installing, run "
        "Warm-up ONCE on a node with internet (downloads the DiffDock-L models + ESM2 weights). "
        "Then pick engine 'diffdock' on the AI docking & screening page. GPU wheels are CUDA "
        "12.1 by default (A100/L40/H200); tick GPU to choose 11.8 / 12.1 / 12.4.",
        default_env="diffdock", gpu=True, recreate=True, cuda_default="12.1",
        path_flag="--dir", path_label="DiffDock folder:", path_default=_default_diffdock_dir(),
        extra=[("--profile repo", "Use DiffDock's own environment.yml (older torch)")],
        warmup=True),
    _EnvSpec(
        "rosetta-env", "Rosetta — PROTAC docking",
        "Classic Rosetta C++ apps (rosetta_scripts, docking_protocol, molfile_to_params) "
        "AND PyRosetta, from the RosettaCommons conda channel — one-click, no license tarball. "
        "Used for PROTAC ternary docking and KIC loop building.",
        default_env="rosetta", gpu=False, recreate=False, status_env=False),
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
            cuda = QLineEdit(spec.cuda_default)
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

        if spec.path_flag:
            prow = QHBoxLayout()
            prow.addWidget(QLabel(spec.path_label))
            path = QLineEdit(spec.path_default)
            path.setToolTip("Where the code is cloned (put it on a disk with room — /data, not HOME quota).")
            prow.addWidget(path, 1)
            state["path"] = path
            card.body.addLayout(prow)

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
        if spec.warmup:
            warm = action_button("Warm-up")
            warm.setToolTip("First run: download the models/weights and build lookup tables. "
                            "Needs internet — run on a login/data node.")
            warm.clicked.connect(lambda _=False, s=spec, st=state: self._warmup(s, st))
            buttons.addWidget(warm)
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
                argv.extend(flag.split())
        if spec.path_flag and state.get("path") is not None and state["path"].text().strip():
            argv += [spec.path_flag, state["path"].text().strip()]
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
        if spec.path_flag and state.get("path") is not None and state["path"].text().strip():
            argv += [spec.path_flag, state["path"].text().strip()]
        self._console.run(argv)

    def _warmup(self, spec: _EnvSpec, state: dict) -> None:
        if self._console is None:
            return
        argv = [spec.key, "warmup", "--env", state["env"].text().strip() or spec.default_env]
        if spec.path_flag and state.get("path") is not None and state["path"].text().strip():
            argv += [spec.path_flag, state["path"].text().strip()]
        self._console.run(argv)
