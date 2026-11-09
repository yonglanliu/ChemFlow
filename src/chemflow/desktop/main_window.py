"""Main native ChemFlow workspace."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.pages import (
    DashboardPage,
    GeneratePage,
    PredictPage,
    SearchPage,
)
# PropertyPredictionPage and LinkerDesignerPage are imported lazily inside their
# page factories: importing them pulls in torch / chemprop / QtWebEngine, which
# we do not want to pay for at launch.
from chemflow.desktop.widgets.process_console import ProcessConsole


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Apply external-tools links (chemflow_tools.yaml) so the GUI resolves the
        # FE/Rosetta interpreters and tool commands without manual env exports.
        try:
            from chemflow.tools_config import apply_env
            apply_env()
        except Exception:
            pass
        self.setWindowTitle("ChemFlow Studio")
        self.resize(1380, 1040)
        self.working_directory = str(Path.cwd())
        self.setMinimumSize(1050, 820)
        # Note: mouse-wheel-over-value-field is blocked per-widget via the
        # NoScroll* classes in widgets.forms (used by every page's _combo/_spin/
        # _dspin helpers). An application-wide event filter was tried here but it
        # crashed QtWebEngine (SIGSEGV), so it must NOT be reintroduced.

        root = QWidget()
        root.setObjectName("Root")
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self._sidebar())

        self.pages = QStackedWidget()
        runner = self._run

        # Single registry of workflows: drives both the launcher cards on the
        # dashboard and the lazy page factories below (indices must match).
        self.functions = [
            {"index": 1, "title": "ADMET", "detail": "Physicochemical & ADME/Tox endpoints"},
            {"index": 5, "title": "PROTAC design", "detail": "Ternary search, linker design, screening"},
            {"index": 3, "title": "LBVS", "detail": "Ligand-based virtual screening (chemical similarity)"},
            {"index": 9, "title": "AI docking & screening", "detail": "Dock or screen with Boltz-2 / Chai-1 + 3D pose"},
            {"index": 2, "title": "Predict", "detail": "Run a saved model on a molecule set"},
            {"index": 4, "title": "Generate", "detail": "Sample novel SMILES"},
            {"index": 6, "title": "Binding Free Energy · GROMACS", "detail": "RBFE, ABFE, SepTop, resFEP, alanine scan, MM-GBSA"},
            {"index": 7, "title": "Binding Free Energy · OpenFE", "detail": "RBFE, ABFE, SepTop, OpenMM modeling & MD"},
        ]
        dashboard = DashboardPage(self.functions)
        dashboard.navigate.connect(self.select_page)

        # Only the dashboard (cheap) is built at launch. Every other page is
        # built the first time it is opened, so starting the app does not import
        # torch / chemprop / QtWebEngine or touch any model weights.
        def _make_property():
            from chemflow.desktop.property_prediction_page import PropertyPredictionPage
            return PropertyPredictionPage()

        def _make_protac_module():
            from chemflow.desktop.protac_page import ProtacModulePage
            return ProtacModulePage(runner, console=self.console, rosetta_runner=self._run_rosetta)

        def _make_bfe_gromacs():
            from chemflow.desktop.alchemforge_pages import BindingFreeEnergyGromacsModule
            return BindingFreeEnergyGromacsModule(console=self.console)

        def _make_bfe_openfe():
            from chemflow.desktop.alchemforge_pages import BindingFreeEnergyOpenFEModule
            return BindingFreeEnergyOpenFEModule(console=self.console)

        def _make_environments():
            from chemflow.desktop.environments_page import EnvironmentsPage
            return EnvironmentsPage(console=self.console)

        self._make_environments = _make_environments

        def _make_cofold_screening():
            from chemflow.desktop.cofold_screening_page import CofoldScreeningPage
            return CofoldScreeningPage(console=self.console)

        self._page_factories = {
            1: _make_property,
            2: lambda: PredictPage(runner),
            3: lambda: SearchPage(runner),
            4: lambda: GeneratePage(runner),
            5: _make_protac_module,
            6: _make_bfe_gromacs,
            7: _make_bfe_openfe,
            # Not a dashboard card — reached via the top-right "Working environment" button.
            8: _make_environments,
            9: _make_cofold_screening,
        }
        self._environments_index = 8
        self._page_built: set[int] = set()
        self._page_containers: list[QWidget] = []
        for _ in range(1 + len(self._page_factories)):
            container = QWidget()
            box = QVBoxLayout(container)
            box.setContentsMargins(0, 0, 0, 0)
            self.pages.addWidget(container)
            self._page_containers.append(container)
        self._page_containers[0].layout().addWidget(dashboard)
        self._page_built.add(0)

        self.console = ProcessConsole()
        self.console.state_changed.connect(self._task_state_changed)
        self.console.setMinimumHeight(150)
        self.console.output.setMinimumHeight(120)

        shell.addWidget(self.pages, 1)
        self.setCentralWidget(root)

        # Activity console as a movable/floatable dock: drag it to the top/bottom,
        # pop it out as its own window, or resize the split — and reopen from View.
        # The console sits straight in the dock (its title bar is the only header).
        self.console_dock = QDockWidget("Activity console", self)
        self.console_dock.setObjectName("ActivityConsoleDock")
        self.console_dock.setWidget(self.console)
        self.console_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.console_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)
        self.resizeDocks([self.console_dock], [300], Qt.Orientation.Vertical)
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.console_dock.toggleViewAction())

        # Top-right entry point for installing/checking the compute conda envs.
        # Use a window toolbar (not the menu bar, which is native/global on macOS)
        # so the button is visibly at the window's top-right.
        env_button = QPushButton("⚙  Working environment")
        env_button.setObjectName("NavButton")
        env_button.setToolTip("Working environment checking and installation")
        env_button.setCursor(Qt.CursorShape.PointingHandCursor)
        env_button.clicked.connect(self._open_environments)
        topbar = QToolBar("Top")
        topbar.setMovable(False)
        topbar.setFloatable(False)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        topbar.addWidget(spacer)
        topbar.addWidget(env_button)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, topbar)

        self.statusBar().showMessage("Ready · local ChemFlow environment")

    def _sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(224)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 24, 18, 18)
        layout.setSpacing(8)
        brand_row = QHBoxLayout()
        mark = QLabel()
        mark.setFixedSize(42, 42)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        from chemflow.config import PROJECT_ROOT
        _logo = PROJECT_ROOT / "assets" / "images" / "chemflow_logo.svg"
        if _logo.is_file():
            mark.setPixmap(QIcon(str(_logo)).pixmap(42, 42))
        else:
            mark.setObjectName("BrandMark")
            mark.setText("CF")
        brand = QLabel("ChemFlow")
        brand.setObjectName("Brand")
        brand_row.addWidget(mark)
        brand_row.addWidget(brand)
        brand_row.addStretch()
        layout.addLayout(brand_row)
        caption = QLabel("MOLECULAR AI STUDIO")
        caption.setObjectName("Eyebrow")
        layout.addWidget(caption)
        layout.addSpacing(22)

        # The workflow list now lives on the main page (the dashboard launcher).
        # The sidebar keeps only a Home button back to it.
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        home = QPushButton("Home")
        home.setObjectName("NavButton")
        home.setCheckable(True)
        home.setChecked(True)
        home.setCursor(Qt.CursorShape.PointingHandCursor)
        home.clicked.connect(lambda: self.select_page(0))
        self.nav_group.addButton(home, 0)
        layout.addWidget(home)

        layout.addStretch()

        # Working folder: where runs execute and generated files are written.
        wf_caption = QLabel("WORKING FOLDER")
        wf_caption.setObjectName("Eyebrow")
        layout.addWidget(wf_caption)
        self.workdir_label = QLabel(self._short_path(self.working_directory))
        self.workdir_label.setObjectName("Muted")
        self.workdir_label.setWordWrap(True)
        self.workdir_label.setToolTip(self.working_directory)
        layout.addWidget(self.workdir_label)
        change_dir = QPushButton("Change…")
        change_dir.setObjectName("NavButton")
        change_dir.setCursor(Qt.CursorShape.PointingHandCursor)
        change_dir.clicked.connect(self._choose_working_directory)
        layout.addWidget(change_dir)
        layout.addSpacing(14)

        version = QLabel("CHEMFLOW  ·  DESKTOP PREVIEW")
        version.setObjectName("Muted")
        version.setWordWrap(True)
        layout.addWidget(version)
        return sidebar

    @staticmethod
    def _short_path(path: str) -> str:
        home = str(Path.home())
        return "~" + path[len(home):] if path.startswith(home) else path

    def _choose_working_directory(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose working folder", self.working_directory)
        if not chosen:
            return
        self.working_directory = chosen
        os.chdir(chosen)                          # in-process writes default here too
        self.console.set_working_directory(chosen)
        self.workdir_label.setText(self._short_path(chosen))
        self.workdir_label.setToolTip(chosen)
        self.statusBar().showMessage(f"Working folder: {chosen}", 5000)

    def _open_environments(self) -> None:
        """Show the compute-environment installer/checker in the main window."""
        self.select_page(self._environments_index)

    def select_page(self, index: int) -> None:
        self._ensure_page(index)
        self.pages.setCurrentIndex(index)
        button = self.nav_group.button(index)
        if button:
            button.setChecked(True)

    def _ensure_page(self, index: int) -> None:
        """Build a page the first time it is opened (deferred heavy imports)."""
        if index in self._page_built:
            return
        factory = self._page_factories.get(index)
        if factory is None:
            self._page_built.add(index)
            return
        self.statusBar().showMessage("Loading…")
        self.statusBar().repaint()
        try:
            widget = factory()
        except Exception as exc:  # a heavy import or model failure shows, not crashes
            widget = QLabel(f"This page could not be loaded:\n{exc}")
            widget.setWordWrap(True)
            widget.setObjectName("Muted")
        self._page_containers[index].layout().addWidget(widget)
        self._page_built.add(index)
        self.statusBar().showMessage("Ready", 2000)

    def _run(self, arguments: list[str]) -> None:
        self.console.run(arguments)

    def _run_rosetta(self, arguments: list[str]) -> None:
        self.console.run_rosetta(arguments)
        self.statusBar().showMessage("Running: chemflow " + " ".join(arguments[:2]))

    def _task_state_changed(self, running: bool) -> None:
        if not running:
            self.statusBar().showMessage("Ready · task finished", 5000)
