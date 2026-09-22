"""Main native ChemFlow workspace."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.pages import (
    DashboardPage,
    GeneratePage,
    PredictPage,
    SearchPage,
    UncertaintyPage,
)
from chemflow.desktop.adme_page import ADMEDeploymentPage
from chemflow.desktop.training_center import TrainingCenterPage
from chemflow.desktop.widgets.forms import Card
from chemflow.desktop.widgets.process_console import ProcessConsole


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChemFlow Studio")
        self.resize(1380, 900)
        self.setMinimumSize(1050, 720)

        root = QWidget()
        root.setObjectName("Root")
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self._sidebar())

        self.pages = QStackedWidget()
        runner = self._run
        dashboard = DashboardPage()
        dashboard.navigate.connect(self.select_page)
        for page in [
            dashboard,
            TrainingCenterPage(),
            PredictPage(runner),
            ADMEDeploymentPage(),
            SearchPage(runner),
            GeneratePage(runner),
            UncertaintyPage(runner),
        ]:
            self.pages.addWidget(page)

        self.console = ProcessConsole()
        self.console.state_changed.connect(self._task_state_changed)
        self.console_card = Card("Activity console", "Live output from the current ChemFlow process.")
        self.console_card.body.addWidget(self.console)
        self.console_card.setMinimumHeight(205)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.pages)
        splitter.addWidget(self.console_card)
        splitter.setSizes([650, 250])
        splitter.setCollapsible(0, False)
        content_layout.addWidget(splitter)
        shell.addWidget(content, 1)
        self.setCentralWidget(root)
        self.pages.currentChanged.connect(self._page_changed)
        self.statusBar().showMessage("Ready · local ChemFlow environment")

    def _sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(224)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 24, 18, 18)
        layout.setSpacing(8)
        brand_row = QHBoxLayout()
        mark = QLabel("CF")
        mark.setObjectName("BrandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        for index, label in enumerate([
            "Overview",
            "Train",
            "Predict",
            "ADME clearance",
            "Similarity search",
            "Generate",
            "Uncertainty",
        ]):
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, page=index: self.select_page(page))
            self.nav_group.addButton(button, index)
            layout.addWidget(button)
            if index == 0:
                button.setChecked(True)
        layout.addStretch()
        version = QLabel("CHEMFLOW  ·  DESKTOP PREVIEW")
        version.setObjectName("Muted")
        version.setWordWrap(True)
        layout.addWidget(version)
        return sidebar

    def select_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        button = self.nav_group.button(index)
        if button:
            button.setChecked(True)

    def _run(self, arguments: list[str]) -> None:
        self.console.run(arguments)
        self.statusBar().showMessage("Running: chemflow " + " ".join(arguments[:2]))

    def _task_state_changed(self, running: bool) -> None:
        if not running:
            self.statusBar().showMessage("Ready · task finished", 5000)

    def _page_changed(self, index: int) -> None:
        """The training center owns its job output; other pages use the shared console."""
        self.console_card.setVisible(index != 1)
