"""Portable, interactive molecule-inspired dashboard scene."""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QPointF, QTimer, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QWidget


ATOMS = (
    (-1.75, 0.15, 0.10, "#43e0c2"),
    (-0.95, 0.75, -0.05, "#6da8ff"),
    (-0.05, 0.32, 0.18, "#edf6ff"),
    (0.80, 0.88, 0.02, "#6da8ff"),
    (1.72, 0.40, -0.16, "#43e0c2"),
    (1.58, -0.62, 0.12, "#edf6ff"),
    (0.56, -0.92, -0.08, "#6da8ff"),
    (-0.30, -0.42, 0.10, "#edf6ff"),
    (-1.22, -0.72, -0.14, "#43e0c2"),
    (0.12, 1.42, 0.28, "#ff6b7a"),
    (2.45, 0.88, 0.20, "#ffbf57"),
    (-2.48, 0.58, -0.22, "#ff6b7a"),
)

BONDS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 2),
    (7, 8),
    (8, 0),
    (3, 9),
    (4, 10),
    (0, 11),
)


class MoleculeView(QWidget):
    """Animated molecular constellation rendered without an OpenGL surface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._x_angle = -15.0
        self._y_angle = 20.0
        self._zoom = 7.2
        self._last_mouse = QPoint()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(30)

    def _transform(self, atom) -> tuple[float, float, float, str]:
        x, y, z, color = atom
        x_angle = math.radians(self._x_angle)
        y_angle = math.radians(self._y_angle)
        y, z = (
            y * math.cos(x_angle) - z * math.sin(x_angle),
            y * math.sin(x_angle) + z * math.cos(x_angle),
        )
        x, z = (
            x * math.cos(y_angle) + z * math.sin(y_angle),
            -x * math.sin(y_angle) + z * math.cos(y_angle),
        )
        return x, y, z, color

    def _project(self, atom) -> tuple[QPointF, float, str]:
        x, y, z, color = self._transform(atom)
        depth = max(3.0, self._zoom - z)
        scale = min(self.width(), self.height()) * 1.15 / depth
        point = QPointF(
            self.width() / 2.0 + x * scale,
            self.height() / 2.0 - y * scale,
        )
        return point, z, color

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        background = QLinearGradient(0, 0, self.width(), self.height())
        background.setColorAt(0.0, QColor("#081522"))
        background.setColorAt(1.0, QColor("#10283a"))
        painter.fillRect(self.rect(), background)

        projected = [self._project(atom) for atom in ATOMS]
        painter.setPen(
            QPen(
                QColor(73, 105, 133, 180),
                3.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        for first, second in BONDS:
            painter.drawLine(projected[first][0], projected[second][0])

        for index in sorted(range(len(projected)), key=lambda item: projected[item][1]):
            point, depth, color = projected[index]
            radius = max(7.0, 12.0 + depth * 1.8)
            base = QColor(color)
            painter.setPen(QPen(base.lighter(145), 1.4))
            painter.setBrush(base)
            painter.drawEllipse(point, radius, radius)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 115))
            painter.drawEllipse(
                QPointF(point.x() - radius * 0.3, point.y() - radius * 0.35),
                radius * 0.28,
                radius * 0.22,
            )

    def mousePressEvent(self, event) -> None:
        self._last_mouse = event.position().toPoint()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        point = event.position().toPoint()
        delta = point - self._last_mouse
        self._x_angle += delta.y() * 0.45
        self._y_angle += delta.x() * 0.45
        self._last_mouse = point
        self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event) -> None:
        self._zoom = min(12.0, max(4.5, self._zoom - event.angleDelta().y() / 480.0))
        self.update()

    def _animate(self) -> None:
        if not self.underMouse():
            self._y_angle += 0.16
            self.update()
