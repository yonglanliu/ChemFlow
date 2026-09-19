"""Small reusable form widgets for the desktop client."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PathField(QWidget):
    def __init__(
        self,
        placeholder: str,
        *,
        mode: str = "file",
        save_filter: str = "All files (*)",
        parent=None,
    ):
        super().__init__(parent)
        self.mode = mode
        self.save_filter = save_filter
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.edit, 1)
        layout.addWidget(browse)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:
        self.edit.setText(value)

    def _browse(self) -> None:
        start = self.text() or str(Path.cwd())
        if self.mode == "directory":
            selected = QFileDialog.getExistingDirectory(self, "Choose directory", start)
        elif self.mode == "save":
            selected, _ = QFileDialog.getSaveFileName(
                self, "Choose output", start, self.save_filter
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(self, "Choose file", start)
        if selected:
            self.setText(selected)


class Card(QFrame):
    def __init__(self, title: str | None = None, subtitle: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(22, 20, 22, 22)
        self.body.setSpacing(13)
        if title:
            heading = QLabel(title)
            heading.setObjectName("CardTitle")
            self.body.addWidget(heading)
        if subtitle:
            detail = QLabel(subtitle)
            detail.setObjectName("Muted")
            detail.setWordWrap(True)
            self.body.addWidget(detail)


class PageHeader(QWidget):
    def __init__(self, eyebrow: str, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(5)
        top = QLabel(eyebrow.upper())
        top.setObjectName("Eyebrow")
        heading = QLabel(title)
        heading.setObjectName("PageTitle")
        detail = QLabel(subtitle)
        detail.setObjectName("PageSubtitle")
        detail.setWordWrap(True)
        layout.addWidget(top)
        layout.addWidget(heading)
        layout.addWidget(detail)


def field(label: str, widget: QWidget, hint: str | None = None) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    caption = QLabel(label)
    caption.setStyleSheet("font-weight: 600;")
    layout.addWidget(caption)
    layout.addWidget(widget)
    if hint:
        helper = QLabel(hint)
        helper.setObjectName("Muted")
        helper.setWordWrap(True)
        layout.addWidget(helper)
    return container


def action_button(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("Primary")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setMinimumHeight(39)
    return button
