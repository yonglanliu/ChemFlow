"""Standalone ADMET Desktop application entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.adme_page import ADMEDeploymentPage
from chemflow.desktop.app import _dark_palette
from chemflow.desktop.physchem_page import PhysChemDeploymentPage
from chemflow.desktop.theme import stylesheet


class ADMETHelpDialog(QDialog):
    """In-application viewer for the packaged ADMET deployment guide."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("HelpDialog")
        self.setWindowTitle("ADMET Desktop help")
        self.resize(920, 760)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        guide = Path(__file__).with_name("ADMET_GUIDE.md")
        browser = QTextBrowser()
        browser.setObjectName("HelpBrowser")
        browser.setOpenExternalLinks(True)
        browser.setMarkdown(guide.read_text(encoding="utf-8"))
        layout.addWidget(browser, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ADMETMainWindow(QMainWindow):
    """Focused desktop shell for deployed ADMET prediction models."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("ADMET Desktop · ChemFlow")
        self.resize(1320, 900)
        self.setMinimumSize(980, 700)
        root = QWidget()
        root.setObjectName("Root")
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        self.page = ADMEDeploymentPage(root, config_path=config_path)
        self.physchem_page = PhysChemDeploymentPage(root)
        self.pages = QStackedWidget(root)
        self.pages.addWidget(self.physchem_page)
        self.pages.addWidget(self.page)
        shell.addWidget(self._property_panel())
        shell.addWidget(self.pages, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage(
            "Ready · LogD, kinetic and thermodynamic solubility"
        )

    def _property_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Sidebar")
        panel.setFixedWidth(270)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 24, 18, 18)
        layout.setSpacing(8)

        brand_row = QHBoxLayout()
        mark = QLabel("AD")
        mark.setObjectName("BrandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand = QLabel("ADMET Desktop")
        brand.setObjectName("Brand")
        brand_row.addWidget(mark)
        brand_row.addWidget(brand)
        brand_row.addStretch()
        help_button = QPushButton("?")
        help_button.setObjectName("HelpButton")
        help_button.setFixedSize(34, 34)
        help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        help_button.setToolTip("Open prediction and uncertainty guide")
        help_button.setAccessibleName("ADMET help")
        help_button.clicked.connect(self._show_help)
        brand_row.addWidget(help_button)
        layout.addLayout(brand_row)
        caption = QLabel("PROPERTY DEPLOYMENT")
        caption.setObjectName("Eyebrow")
        layout.addWidget(caption)
        layout.addSpacing(22)

        heading = QLabel("Select prediction")
        heading.setStyleSheet("font-size: 14px; font-weight: 700;")
        layout.addWidget(heading)
        detail = QLabel("Run every deployed model or focus on one clearance endpoint.")
        detail.setObjectName("Muted")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        layout.addSpacing(8)

        select_all = QPushButton("Select all properties")
        select_all.setObjectName("SelectAllButton")
        select_all.setCursor(Qt.CursorShape.PointingHandCursor)
        select_all.clicked.connect(self._select_all_properties)
        layout.addWidget(select_all)

        category = QFrame()
        category.setObjectName("PropertyCategory")
        category_layout = QVBoxLayout(category)
        category_layout.setContentsMargins(12, 12, 12, 12)
        category_layout.setSpacing(7)
        category_name = QLabel("METABOLISM")
        category_name.setObjectName("PropertyCategoryTitle")
        category_layout.addWidget(category_name)
        category_detail = QLabel("Microsomal intrinsic clearance")
        category_detail.setObjectName("Muted")
        category_layout.addWidget(category_detail)

        self._updating_property_selection = False
        self.property_buttons: dict[str, QPushButton] = {}
        for property_name, label in (
            ("HLM", "Human liver microsomes (HLM)"),
            ("MLM", "Mouse liver microsomes (MLM)"),
            ("RLM", "Rat liver microsomes (RLM)"),
        ):
            button = QPushButton(label)
            button.setObjectName("PropertyButton")
            button.setCheckable(True)
            button.setChecked(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.toggled.connect(self._property_selection_changed)
            self.property_buttons[property_name] = button
            category_layout.addWidget(button)
        physchem_category = QFrame()
        physchem_category.setObjectName("PropertyCategory")
        physchem_layout = QVBoxLayout(physchem_category)
        physchem_layout.setContentsMargins(12, 12, 12, 12)
        physchem_layout.setSpacing(7)
        physchem_name = QLabel("PHYSICOCHEMICAL")
        physchem_name.setObjectName("PropertyCategoryTitle")
        physchem_layout.addWidget(physchem_name)
        physchem_detail = QLabel("Bundled Chemprop deployment")
        physchem_detail.setObjectName("Muted")
        physchem_layout.addWidget(physchem_detail)
        open_physchem = QPushButton("LogD and solubility")
        open_physchem.setObjectName("PropertyButton")
        open_physchem.setCursor(Qt.CursorShape.PointingHandCursor)
        open_physchem.clicked.connect(self._show_physchem)
        physchem_layout.addWidget(open_physchem)
        layout.addWidget(physchem_category)
        layout.addWidget(category)

        layout.addStretch()
        future = QLabel("MODEL CATALOG\nMetabolism\nMore ADMET categories can be added here")
        future.setObjectName("Muted")
        future.setWordWrap(True)
        layout.addWidget(future)
        return panel

    def _show_help(self) -> None:
        ADMETHelpDialog(self).exec()

    def _selected_properties(self) -> tuple[str, ...]:
        return tuple(
            property_name
            for property_name, button in self.property_buttons.items()
            if button.isChecked()
        )

    def _select_all_properties(self) -> None:
        self._updating_property_selection = True
        try:
            for button in self.property_buttons.values():
                button.setChecked(True)
        finally:
            self._updating_property_selection = False
        self._select_properties(self._selected_properties())

    def _show_physchem(self) -> None:
        self.pages.setCurrentWidget(self.physchem_page)
        self.statusBar().showMessage(
            "Active prediction: LogD, kinetic and thermodynamic solubility", 4000
        )

    def _property_selection_changed(self, _checked: bool) -> None:
        if self._updating_property_selection:
            return
        selected = self._selected_properties()
        if not selected:
            # Prediction requires at least one endpoint. Keep the last one selected.
            button = self.sender()
            if isinstance(button, QPushButton):
                self._updating_property_selection = True
                button.setChecked(True)
                self._updating_property_selection = False
            return
        self._select_properties(selected)

    def _select_properties(self, properties: tuple[str, ...]) -> None:
        self.pages.setCurrentWidget(self.page)
        self.page.set_selected_properties(properties)
        self.statusBar().showMessage(
            f"Active prediction: {', '.join(properties)}", 4000
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the ADMET deployment desktop.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional override for the automatically discovered deployment TOML.",
    )
    arguments, qt_arguments = parser.parse_known_args(
        sys.argv[1:] if argv is None else argv
    )
    app = QApplication.instance() or QApplication([sys.argv[0], *qt_arguments])
    app.setApplicationName("ADMET Desktop")
    app.setOrganizationName("ChemFlow")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(stylesheet())
    window = ADMETMainWindow(config_path=arguments.config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
