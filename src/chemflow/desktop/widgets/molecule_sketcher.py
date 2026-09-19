"""Small native molecule sketcher and RDKit structure preview widgets."""

from __future__ import annotations

import copy
import math

from PySide6.QtCore import QByteArray, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from rdkit import Chem
from rdkit.Chem import rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D


ATOM_COLORS = {
    "C": "#edf6ff",
    "N": "#6da8ff",
    "O": "#ff6b7a",
    "S": "#ffd166",
    "F": "#43e0c2",
    "Cl": "#43e0c2",
    "Br": "#d79bff",
    "P": "#ffad66",
}


class MoleculePreview(QSvgWidget):
    """Render a SMILES string as a scalable RDKit SVG depiction."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MoleculePreview")
        self.setMinimumHeight(220)
        self.set_smiles("")

    def set_smiles(self, smiles: str) -> bool:
        molecule = Chem.MolFromSmiles(str(smiles).strip()) if str(smiles).strip() else None
        if molecule is None:
            self.load(QByteArray(self._empty_svg().encode("utf-8")))
            return False
        molecule = Chem.Mol(molecule)
        rdCoordGen.AddCoords(molecule)
        drawer = rdMolDraw2D.MolDraw2DSVG(720, 400)
        options = drawer.drawOptions()
        options.clearBackground = True
        options.setBackgroundColour((0.97, 0.985, 1.0, 1.0))
        options.padding = 0.12
        options.bondLineWidth = 2.2
        options.minFontSize = 16
        options.maxFontSize = 30
        options.setAtomPalette(
            {
                6: (0.08, 0.12, 0.18),
                7: (0.10, 0.30, 0.82),
                8: (0.86, 0.12, 0.18),
                9: (0.00, 0.56, 0.36),
                15: (0.92, 0.42, 0.08),
                16: (0.72, 0.52, 0.00),
                17: (0.00, 0.56, 0.36),
                35: (0.55, 0.18, 0.70),
                53: (0.42, 0.16, 0.58),
            }
        )
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        self.load(QByteArray(drawer.GetDrawingText().encode("utf-8")))
        return True

    @staticmethod
    def _empty_svg() -> str:
        return """<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400">
<rect width="100%" height="100%" rx="14" fill="#f7fbff"/>
<text x="360" y="190" text-anchor="middle" fill="#536a80"
 font-family="sans-serif" font-size="17">Enter or draw a valid molecule</text>
<text x="360" y="220" text-anchor="middle" fill="#8295a8"
 font-family="sans-serif" font-size="12">The 2D structure will appear here</text>
</svg>"""


class SketchCanvas(QWidget):
    smiles_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 310)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.atom_symbol = "C"
        self.bond_order = 1
        self.atoms: list[tuple[str, QPointF]] = []
        self.bonds: list[tuple[int, int, int]] = []
        self.selected: int | None = None
        self._history: list[tuple[list, list, int | None]] = []

    def set_atom_symbol(self, symbol: str) -> None:
        self.atom_symbol = symbol

    def set_bond_order(self, label: str) -> None:
        self.bond_order = {"Single": 1, "Double": 2, "Triple": 3}[label]

    def _snapshot(self) -> None:
        self._history.append((copy.deepcopy(self.atoms), copy.deepcopy(self.bonds), self.selected))
        self._history = self._history[-50:]

    def undo(self) -> None:
        if not self._history:
            return
        self.atoms, self.bonds, self.selected = self._history.pop()
        self._changed()

    def clear(self) -> None:
        if not self.atoms:
            return
        self._snapshot()
        self.atoms.clear()
        self.bonds.clear()
        self.selected = None
        self._changed()

    def _hit_atom(self, point: QPointF) -> int | None:
        for index, (_symbol, position) in enumerate(self.atoms):
            if math.hypot(point.x() - position.x(), point.y() - position.y()) <= 22:
                return index
        return None

    def mousePressEvent(self, event) -> None:
        point = event.position()
        hit = self._hit_atom(point)
        if event.button() == Qt.MouseButton.RightButton:
            if hit is not None:
                self._snapshot()
                self.atoms.pop(hit)
                self.bonds = [
                    (first - (first > hit), second - (second > hit), order)
                    for first, second, order in self.bonds
                    if first != hit and second != hit
                ]
                self.selected = None
                self._changed()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return
        if hit is not None:
            if self.selected is not None and self.selected != hit:
                self._snapshot()
                pair = {self.selected, hit}
                self.bonds = [
                    bond for bond in self.bonds if {bond[0], bond[1]} != pair
                ]
                self.bonds.append((self.selected, hit, self.bond_order))
                self.selected = hit
                self._changed()
            else:
                self.selected = hit
                self.update()
            return

        self._snapshot()
        new_index = len(self.atoms)
        self.atoms.append((self.atom_symbol, point))
        if self.selected is not None:
            self.bonds.append((self.selected, new_index, self.bond_order))
        self.selected = new_index
        self._changed()

    def _changed(self) -> None:
        self.update()
        self.smiles_changed.emit(self.to_smiles())

    def to_smiles(self) -> str:
        if not self.atoms:
            return ""
        editable = Chem.RWMol()
        for symbol, _position in self.atoms:
            editable.AddAtom(Chem.Atom(symbol))
        bond_types = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }
        try:
            for first, second, order in self.bonds:
                editable.AddBond(first, second, bond_types[order])
            molecule = editable.GetMol()
            Chem.SanitizeMol(molecule)
            return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        except (RuntimeError, ValueError):
            return ""

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#081522"))
        painter.setPen(QPen(QColor("#162b43"), 1))
        for x in range(20, self.width(), 20):
            painter.drawLine(x, 0, x, self.height())
        for y in range(20, self.height(), 20):
            painter.drawLine(0, y, self.width(), y)

        for first, second, order in self.bonds:
            start = self.atoms[first][1]
            end = self.atoms[second][1]
            dx, dy = end.x() - start.x(), end.y() - start.y()
            length = max(1.0, math.hypot(dx, dy))
            perpendicular = QPointF(-dy / length * 4.0, dx / length * 4.0)
            offsets = {1: [0.0], 2: [-0.7, 0.7], 3: [-1.2, 0.0, 1.2]}[order]
            painter.setPen(QPen(QColor("#8ea5bd"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for offset in offsets:
                delta = perpendicular * offset
                painter.drawLine(start + delta, end + delta)

        for index, (symbol, position) in enumerate(self.atoms):
            radius = 18
            painter.setBrush(QColor("#16344a" if index != self.selected else "#245f6c"))
            painter.setPen(QPen(QColor("#43e0c2" if index == self.selected else "#294b68"), 2))
            painter.drawEllipse(position, radius, radius)
            painter.setPen(QColor(ATOM_COLORS.get(symbol, "#edf6ff")))
            painter.drawText(
                int(position.x() - radius),
                int(position.y() - radius),
                radius * 2,
                radius * 2,
                Qt.AlignmentFlag.AlignCenter,
                symbol,
            )


class MoleculeSketcher(QWidget):
    """Native click-to-add atom and bond editor with SMILES export."""

    smiles_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        atoms = QComboBox()
        atoms.addItems(list(ATOM_COLORS))
        bonds = QComboBox()
        bonds.addItems(["Single", "Double", "Triple"])
        undo = QPushButton("Undo")
        clear = QPushButton("Clear")
        help_text = QLabel("Left click: add/select/connect · Right click: delete atom")
        help_text.setObjectName("Muted")
        controls.addWidget(QLabel("Atom"))
        controls.addWidget(atoms)
        controls.addWidget(QLabel("Bond"))
        controls.addWidget(bonds)
        controls.addWidget(undo)
        controls.addWidget(clear)
        controls.addStretch()
        controls.addWidget(help_text)
        layout.addLayout(controls)

        self.canvas = SketchCanvas()
        layout.addWidget(self.canvas)
        self.smiles = QLineEdit()
        self.smiles.setReadOnly(True)
        self.smiles.setPlaceholderText("A valid SMILES appears as the structure is completed")
        layout.addWidget(self.smiles)

        atoms.currentTextChanged.connect(self.canvas.set_atom_symbol)
        bonds.currentTextChanged.connect(self.canvas.set_bond_order)
        undo.clicked.connect(self.canvas.undo)
        clear.clicked.connect(self.canvas.clear)
        self.canvas.smiles_changed.connect(self._update_smiles)

    def _update_smiles(self, value: str) -> None:
        self.smiles.setText(value)
        self.smiles_changed.emit(value)

    def value(self) -> str:
        return self.smiles.text().strip()
