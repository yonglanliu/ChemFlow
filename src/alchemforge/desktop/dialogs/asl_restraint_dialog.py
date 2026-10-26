"""Stage-specific ASL restraint editor for GROMACS workflows."""

from __future__ import annotations

import re
from copy import deepcopy

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

_COMPONENTS = ("protein", "ligand", "cofactor")


def parse_restraint_asl(expression: str) -> dict[str, object]:
    """Parse the topology-stable PyMOL-like subset used by the RBFE builder."""
    text = " ".join(expression.strip().split())
    if not text:
        raise ValueError("ASL selection cannot be empty")
    lowered = text.casefold()
    if re.search(r"\b(index|atom\.num|chain(?:\.name)?)\b", lowered):
        raise ValueError(
            "PDB atom indices and chain-only selectors are not topology-stable "
            "after PMX/GROMACS assembly. Use resid, resname, and name instead."
        )
    if re.search(r"\bor\b", lowered):
        raise ValueError("Use one component per rule; OR expressions are not supported")
    lowered = lowered.replace("polymer.protein", "protein")
    if re.search(r"\borganic\b", lowered):
        raise ValueError(
            "'organic' is ambiguous in an assembled system; use ligand or "
            "cofactor explicitly"
        )
    components = [name for name in _COMPONENTS if re.search(rf"\b{name}\b", lowered)]
    if len(components) != 1:
        raise ValueError("Start the ASL with exactly one of: protein, ligand, cofactor")

    residue_ids: list[int] = []
    residue_match = re.search(
        r"\b(?:resi|resid(?:ue)?(?:\.number)?)\s+(.+?)(?=\s+and\s+|$)",
        lowered,
    )
    if residue_match:
        for token in re.split(r"[\s,+]+", residue_match.group(1).strip()):
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                try:
                    start, end = int(start_text), int(end_text)
                except ValueError as error:
                    raise ValueError(f"Invalid residue range: {token}") from error
                if start < 1 or end < start:
                    raise ValueError(f"Invalid residue range: {token}")
                residue_ids.extend(range(start, end + 1))
            else:
                try:
                    residue_ids.append(int(token))
                except ValueError as error:
                    raise ValueError(f"Invalid residue number: {token}") from error

    def names_after(keyword: str) -> list[str]:
        match = re.search(
            rf"\b(?:{keyword})\s+(.+?)(?=\s+and\s+|$)", text, re.IGNORECASE
        )
        if not match:
            return []
        return [
            value.upper()
            for value in re.split(r"[\s,+]+", match.group(1).strip())
            if value
        ]

    residue_names = names_after(r"resn|resname|residue\.name")
    atom_names = names_after(r"name|atom\.name")
    recognized = re.sub(r"\b(?:protein|ligand|cofactor)\b", "", lowered)
    recognized = re.sub(
        r"\b(?:resi|resid(?:ue)?(?:\.number)?|resn|resname|residue\.name|name|atom\.name)\s+.+?(?=\s+and\s+|$)",
        "",
        recognized,
    )
    recognized = re.sub(
        r"\b(?:and|heavy|not\s+(?:elem(?:ent)?\s+h|hydro))\b",
        "",
        recognized,
    )
    recognized = recognized.replace("(", "").replace(")", "").strip()
    if recognized:
        raise ValueError(f"Unsupported ASL term: {recognized!r}")
    return {
        "component": components[0],
        "residue_ids": sorted(set(residue_ids)),
        "residue_names": sorted(set(residue_names)),
        "atom_names": sorted(set(atom_names)),
    }


class ASLRestraintDialog(QDialog):
    """Edit all positional-restraint rules for one equilibration stage."""

    rules_saved = pyqtSignal(str, object)

    def __init__(
        self,
        *,
        stage_key: str,
        stage_label: str,
        rules: list[dict[str, object]],
        structure_description: str,
        parent=None,
    ):
        super().__init__(parent)
        self.stage_key = stage_key
        self.stage_label = stage_label
        self.setWindowTitle(f"{stage_label} · PyMOL-style Restraints")
        self.resize(850, 430)

        root = QVBoxLayout(self)
        source = QLabel(
            f"Structure source: {structure_description}\n"
            "This editor reuses the structure selected in the System tab."
        )
        source.setWordWrap(True)
        root.addWidget(source)
        help_text = QLabel(
            "PyMOL-style selection: protein | ligand | cofactor, optionally "
            "followed by and resi 10-50, and resn LIG, and name CA+C+N+O, or "
            "and not elem H. Example: protein and resi 20-80 and name CA+C+N+O. "
            "Changing a component selection updates that selection in the other "
            "stages; each stage keeps its own force constant."
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color:#52708f")
        root.addWidget(help_text)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Rule name", "PyMOL-style selection", "Force (kJ mol⁻¹ nm⁻²)", "Status"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._validate_rows)
        root.addWidget(self.table, 1)

        controls = QHBoxLayout()
        add = QPushButton("Add restraint")
        add.clicked.connect(self._add_empty_rule)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_selected)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.close)
        save = QPushButton("Save restraints")
        save.clicked.connect(self._save)
        controls.addWidget(add)
        controls.addWidget(remove)
        controls.addStretch()
        controls.addWidget(cancel)
        controls.addWidget(save)
        root.addLayout(controls)

        for rule in deepcopy(rules):
            self._append_rule(rule)
        self._validate_rows()

    def _append_rule(self, rule: dict[str, object]) -> None:
        was_blocked = self.table.blockSignals(True)
        row = self.table.rowCount()
        try:
            self.table.insertRow(row)
            self.table.setItem(
                row, 0, QTableWidgetItem(str(rule.get("name", "Restraint")))
            )
            self.table.setItem(
                row, 1, QTableWidgetItem(str(rule.get("asl", "protein")))
            )
            force = QDoubleSpinBox()
            force.setRange(0.0, 100000.0)
            force.setDecimals(1)
            force.setValue(float(rule.get("force", 0.0)))
            force.setSuffix(" kJ mol⁻¹ nm⁻²")
            force.valueChanged.connect(self._validate_rows)
            self.table.setCellWidget(row, 2, force)
            status = QTableWidgetItem()
            status.setFlags(status.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 3, status)
        finally:
            self.table.blockSignals(was_blocked)

    def _add_empty_rule(self) -> None:
        counts = {name: 0 for name in _COMPONENTS}
        component_forces: dict[str, float] = {}
        for rule in self.rules():
            try:
                component = str(
                    parse_restraint_asl(str(rule["asl"]))["component"]
                )
                counts[component] += 1
                component_forces.setdefault(component, float(rule["force"]))
            except ValueError:
                continue
        component = min(_COMPONENTS, key=lambda name: counts[name])
        self._append_rule(
            {
                "name": f"{component.title()} restraint {counts[component] + 1}",
                "asl": f"{component} and not element H",
                "force": component_forces.get(component, 100.0),
            }
        )
        self.table.selectRow(self.table.rowCount() - 1)
        self._validate_rows()

    def _remove_selected(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
        self._validate_rows()

    def rules(self) -> list[dict[str, object]]:
        rules: list[dict[str, object]] = []
        for row in range(self.table.rowCount()):
            force = self.table.cellWidget(row, 2)
            name_item = self.table.item(row, 0)
            asl_item = self.table.item(row, 1)
            if name_item is None or asl_item is None or force is None:
                continue
            rules.append(
                {
                    "name": name_item.text().strip() or "Restraint",
                    "asl": asl_item.text().strip(),
                    "force": force.value(),
                }
            )
        return rules

    def _validate_rows(self, *_args) -> bool:
        valid = True
        component_forces: dict[str, float] = {}
        for row, rule in enumerate(self.rules()):
            status = self.table.item(row, 3)
            try:
                parsed = parse_restraint_asl(str(rule["asl"]))
                component = str(parsed["component"])
                force = float(rule["force"])
                previous_force = component_forces.get(component)
                if (
                    force > 0
                    and previous_force is not None
                    and force != previous_force
                ):
                    raise ValueError(
                        f"All {component} groups in one stage must use the same "
                        "force. Separate per-group forces are not representable "
                        "by the current GROMACS topology."
                    )
                if force > 0:
                    component_forces[component] = force
                if force <= 0:
                    status.setText("Released")
                else:
                    status.setText("Valid")
                status.setForeground(Qt.darkGreen)
                status.setToolTip("")
            except ValueError as error:
                valid = False
                status.setText("Invalid")
                status.setForeground(Qt.red)
                status.setToolTip(str(error))
        return valid

    def _save(self) -> None:
        if not self._validate_rows():
            QMessageBox.warning(
                self,
                "Invalid Restraint Selection",
                "Correct the invalid selection rules. Hover over an Invalid status for details.",
            )
            return
        self.rules_saved.emit(self.stage_key, self.rules())
        self.close()
