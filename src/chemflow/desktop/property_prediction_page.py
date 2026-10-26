"""Unified workspace for deployed physicochemical and ADME predictors."""

from __future__ import annotations

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from chemflow.desktop.adme_page import ADMEDeploymentPage
from chemflow.desktop.physchem_page import PhysChemDeploymentPage


class PropertyPredictionPage(QWidget):
    """Group deployed property models while keeping physicochemical first."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.physchem_page = PhysChemDeploymentPage(self)
        self.adme_page = ADMEDeploymentPage(self)
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("PropertyPredictionTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.physchem_page, "Physicochemical")
        self.tabs.addTab(self.adme_page, "Clearance")
        self.tabs.setCurrentIndex(0)
        self._last_tab = 0
        self.tabs.currentChanged.connect(self._tab_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

    def _tab_changed(self, index: int) -> None:
        """Carry molecular input between both predictors when changing tabs."""
        source = self.physchem_page if self._last_tab == 0 else self.adme_page
        target = self.physchem_page if index == 0 else self.adme_page
        if source is not target:
            target.input_tabs.setCurrentIndex(source.input_tabs.currentIndex())
            target.smiles_input.setPlainText(source.smiles_input.toPlainText())
            target.input_file.setText(source.input_file.text())
            target.structure_column.setText(source.structure_column.text())
        self._last_tab = index
