"""Helpers for presenting long workflow forms as ResFEP-style step tabs."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFormLayout, QTabWidget, QVBoxLayout, QWidget


def form_step_tabs(
    source: QFormLayout,
    sections: Sequence[tuple[str, int]],
) -> QTabWidget:
    """Move sequential rows from one form into named, clickable tabs."""

    expected = sum(row_count for _, row_count in sections)
    if source.rowCount() != expected:
        raise ValueError(
            f"Tabbed form defines {expected} rows but contains {source.rowCount()}"
        )

    tabs = QTabWidget()
    for title, row_count in sections:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        for _ in range(row_count):
            row = source.takeRow(0)
            label = row.labelItem.widget() if row.labelItem is not None else None
            field = row.fieldItem.widget() if row.fieldItem is not None else None
            if label is not None and field is not None:
                form.addRow(label, field)
            elif field is not None:
                form.addRow(field)
        page_layout.addLayout(form)
        page_layout.addStretch()
        tabs.addTab(page, title)
    return tabs
