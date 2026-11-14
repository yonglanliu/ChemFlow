"""Advanced Boltz-2 constraints for the AI-docking page, in their own window.

The docking page is already dense, so the binding-site pocket, structural
template, pairwise contact restraints and covalent bonds live here and are
opened with one button. This dialog OWNS those widgets (the page keeps
references), so their state survives closing and reopening the window.

Everything here maps onto Boltz's ``constraints:`` / ``templates:`` schema and
is emitted as ``alchemfold dock generate`` flags by :meth:`ConstraintsDialog.argv`.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.widgets.forms import PathField, action_button


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Muted")
    label.setWordWrap(True)
    return label


class _EditableTable(QWidget):
    """A small add/remove row table over free-text columns."""

    def __init__(self, headers: list[str], placeholders: list[str], parent=None):
        super().__init__(parent)
        self._placeholders = placeholders
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
            | QAbstractItemView.EditKeyPressed | QAbstractItemView.AnyKeyPressed)
        header = self.table.horizontalHeader()
        for column in range(len(headers)):
            header.setSectionResizeMode(column, QHeaderView.Stretch)
        self.table.setMinimumHeight(96)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        self.add_button = action_button("Add row")
        self.remove_button = action_button("Remove selected")
        self.add_button.clicked.connect(self.add_row)
        self.remove_button.clicked.connect(self.remove_selected)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    def add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, placeholder in enumerate(self._placeholders):
            item = QTableWidgetItem("")
            item.setToolTip(placeholder)
            self.table.setItem(row, column, item)

    def remove_selected(self) -> None:
        for index in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(index)

    def rows(self) -> list[list[str]]:
        """Non-empty rows as lists of stripped strings."""
        out = []
        for row in range(self.table.rowCount()):
            values = []
            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)
                values.append((item.text() if item else "").strip())
            if any(values):
                out.append(values)
        return out


class ConstraintsDialog(QDialog):
    """Pocket / template / contact / bond settings for one docking job."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Constraints & templates")
        self.setMinimumWidth(620)
        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        outer.addWidget(_muted(
            "Optional ways to steer the prediction. All of these are Boltz-2 features; "
            "Chai-1 supports only the pocket (as contact restraints) and DiffDock-L none "
            "of them."))

        # --- binding-site pocket ----------------------------------------
        self.pocket_group = QGroupBox("Binding pocket")
        self.pocket_group.setCheckable(True)
        self.pocket_group.setChecked(False)
        pocket_layout = QVBoxLayout(self.pocket_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Binder:"))
        self.pocket_binder = QLineEdit("L")
        self.pocket_binder.setFixedWidth(48)
        self.pocket_binder.setToolTip("Ligand id (chain) that binds the pocket.")
        row.addWidget(self.pocket_binder)
        row.addWidget(QLabel("Contacts:"))
        self.pocket_contacts = QLineEdit()
        self.pocket_contacts.setPlaceholderText("pocket residues, e.g. A:123, A:145")
        self.pocket_contacts.setToolTip(
            "Receptor residues lining the pocket as CHAIN:RES, comma-separated (bare "
            "numbers use the first protein chain). Numbering must match the sequence. "
            "On a ligand chain an atom name may be used instead of an index.")
        row.addWidget(self.pocket_contacts, 1)
        row.addWidget(QLabel("Max dist (Å):"))
        self.pocket_max_distance = QLineEdit("6.0")
        self.pocket_max_distance.setFixedWidth(52)
        row.addWidget(self.pocket_max_distance)
        pocket_layout.addLayout(row)
        self.pocket_force = QCheckBox("Enforce with a steering potential")
        self.pocket_force.setToolTip(
            "Off: the pocket is a soft feature the model may ignore. On: a potential "
            "pushes the ligand into those contacts during diffusion.")
        pocket_layout.addWidget(self.pocket_force)
        pocket_layout.addWidget(_muted(
            "Leave the residues blank to write a placeholder block you can edit in the YAML."))
        outer.addWidget(self.pocket_group)

        # --- structural template ----------------------------------------
        self.template_group = QGroupBox("Receptor template")
        self.template_group.setCheckable(True)
        self.template_group.setChecked(False)
        template_layout = QVBoxLayout(self.template_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Structure:"))
        self.template_file = PathField(
            "reference structure (.cif / .pdb) — e.g. an experimental open-state model",
            mode="file")
        row.addWidget(self.template_file, 1)
        row.addWidget(QLabel("Chains:"))
        self.template_chains = QLineEdit()
        self.template_chains.setFixedWidth(110)
        self.template_chains.setPlaceholderText("A,B,C,D")
        self.template_chains.setToolTip(
            "Which chains of THIS job to template, comma-separated. Blank = Boltz picks "
            "the best matching chains.")
        row.addWidget(self.template_chains)
        template_layout.addLayout(row)
        row = QHBoxLayout()
        self.template_force = QCheckBox("Enforce with a steering potential")
        self.template_force.setToolTip(
            "Off: an input feature the model may overrule — often not enough for a "
            "conformation it disprefers. On: a potential holds the backbone near the "
            "template during diffusion.")
        row.addWidget(self.template_force)
        row.addWidget(QLabel("Max deviation (Å):"))
        self.template_threshold = QLineEdit("2.0")
        self.template_threshold.setFixedWidth(52)
        self.template_threshold.setToolTip(
            "~2 Å pins the conformation; 4-6 Å keeps the fold but allows shifts.")
        row.addWidget(self.template_threshold)
        row.addStretch(1)
        template_layout.addLayout(row)
        template_layout.addWidget(_muted(
            "Protein chains only — the ligand pose is still predicted. Templating imposes "
            "the conformation, so the result is a starting structure, not evidence that "
            "the ligand induces that state."))
        outer.addWidget(self.template_group)
        self.template_threshold.setEnabled(False)
        self.template_force.toggled.connect(self.template_threshold.setEnabled)

        # --- pairwise contact restraints --------------------------------
        self.contact_group = QGroupBox("Contact restraints")
        self.contact_group.setCheckable(True)
        self.contact_group.setChecked(False)
        contact_layout = QVBoxLayout(self.contact_group)
        self.contacts_table = _EditableTable(
            ["Token 1", "Token 2", "Max dist (Å)"],
            ["CHAIN:RESIDUE, or CHAIN:ATOM_NAME on a ligand chain",
             "CHAIN:RESIDUE, or CHAIN:ATOM_NAME on a ligand chain",
             "optional — blank uses the default below"])
        contact_layout.addWidget(self.contacts_table)
        row = QHBoxLayout()
        self.contact_force = QCheckBox("Enforce with a steering potential")
        row.addWidget(self.contact_force)
        row.addWidget(QLabel("Default max dist (Å):"))
        self.contact_max_distance = QLineEdit("5.0")
        self.contact_max_distance.setFixedWidth(52)
        row.addWidget(self.contact_max_distance)
        row.addStretch(1)
        contact_layout.addLayout(row)
        contact_layout.addWidget(_muted(
            "A distance restraint between two tokens — one known interaction rather than "
            "a whole pocket."))
        outer.addWidget(self.contact_group)

        # --- covalent bonds ----------------------------------------------
        self.bond_group = QGroupBox("Covalent bonds")
        self.bond_group.setCheckable(True)
        self.bond_group.setChecked(False)
        bond_layout = QVBoxLayout(self.bond_group)
        self.bonds_table = _EditableTable(
            ["Atom 1", "Atom 2"],
            ["CHAIN:RESIDUE:ATOM_NAME, e.g. A:123:SG",
             "CHAIN:RESIDUE:ATOM_NAME, e.g. L:1:C12"])
        bond_layout.addWidget(self.bonds_table)
        bond_layout.addWidget(_muted(
            "An explicit bond between two atoms — the covalent-docking path. Atom names "
            "must match the CCD/SMILES naming of that component."))
        outer.addWidget(self.bond_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        outer.addWidget(buttons)

    # -- engine gating ---------------------------------------------------
    def set_engine(self, engine: str, label: str) -> None:
        """Enable only what ``engine`` supports (Boltz-2 has all of it)."""
        is_boltz = engine == "boltz"
        self.pocket_group.setEnabled(engine in {"boltz", "chai"})
        self.pocket_force.setEnabled(is_boltz)
        self.pocket_force.setToolTip(
            "Off: the pocket is a soft feature the model may ignore. On: a potential "
            "pushes the ligand into those contacts during diffusion."
            if is_boltz else
            f"Boltz-2 only — {label} writes the pocket as plain contact restraints.")
        for group in (self.template_group, self.contact_group, self.bond_group):
            group.setEnabled(is_boltz)
            if not is_boltz:
                group.setChecked(False)
                group.setToolTip(f"Boltz-2 only — {label} has no equivalent.")
            else:
                group.setToolTip("")
        if engine == "diffdock":
            self.pocket_group.setChecked(False)
            self.pocket_group.setToolTip("DiffDock-L docks blind — no pocket constraint.")
        else:
            self.pocket_group.setToolTip("")

    # -- what the page needs ---------------------------------------------
    def summary(self) -> str:
        """One line describing what is switched on (for the page)."""
        parts = []
        if self.pocket_group.isChecked():
            contacts = [c for c in self.pocket_contacts.text().replace(";", ",").split(",")
                        if c.strip()]
            detail = f"{len(contacts)} residues" if contacts else "placeholder"
            parts.append(f"pocket ({detail}{', forced' if self.pocket_force.isChecked() else ''})")
        if self.template_group.isChecked():
            from pathlib import Path

            name = Path(self.template_file.text().strip()).name or "no file"
            parts.append(f"template {name}{', forced' if self.template_force.isChecked() else ''}")
        rows = len(self.contacts_table.rows()) if self.contact_group.isChecked() else 0
        if rows:
            parts.append(f"{rows} contact restraint{'s' if rows != 1 else ''}")
        rows = len(self.bonds_table.rows()) if self.bond_group.isChecked() else 0
        if rows:
            parts.append(f"{rows} covalent bond{'s' if rows != 1 else ''}")
        return " · ".join(parts) if parts else "none set"

    def argv(self, engine: str) -> list[str]:
        """CLI flags for ``alchemfold dock generate``.

        Raises ``ValueError`` with a message meant for the page's status line.
        """
        argv: list[str] = []
        if self.pocket_group.isChecked() and engine in {"boltz", "chai"}:
            contacts = self.pocket_contacts.text().strip()
            if contacts:
                argv += ["--pocket-contacts", contacts]
                binder = self.pocket_binder.text().strip()
                if binder:
                    argv += ["--pocket-binder", binder]
                distance = self.pocket_max_distance.text().strip()
                if distance:
                    argv += ["--pocket-max-distance", distance]
            else:
                argv.append("--pocket")   # placeholder block to edit by hand
            # Chai-1 writes the pocket as contact restraints and has no steering
            # potential, so `force` only means something for Boltz-2.
            if self.pocket_force.isChecked() and engine == "boltz":
                argv.append("--pocket-force")

        if engine != "boltz":
            return argv

        if self.template_group.isChecked():
            path = self.template_file.text().strip()
            if not path:
                raise ValueError(
                    "Receptor template is on but no structure was chosen — browse for a "
                    ".cif/.pdb in Constraints & templates, or switch it off.")
            argv += ["--template", path]
            chains = self.template_chains.text().strip()
            if chains:
                argv += ["--template-chains", chains]
            if self.template_force.isChecked():
                argv.append("--template-force")
                threshold = self.template_threshold.text().strip()
                if threshold:
                    argv += ["--template-threshold", threshold]

        if self.contact_group.isChecked():
            for index, row in enumerate(self.contacts_table.rows(), start=1):
                token1, token2, distance = row[0], row[1], row[2]
                if not token1 or not token2:
                    raise ValueError(
                        f"Contact restraint {index} needs both tokens "
                        "(CHAIN:RESIDUE), or remove the row.")
                spec = f"{token1},{token2}" + (f",{distance}" if distance else "")
                argv += ["--contact", spec]
            if self.contact_force.isChecked():
                argv.append("--contact-force")
            distance = self.contact_max_distance.text().strip()
            if distance:
                argv += ["--contact-max-distance", distance]

        if self.bond_group.isChecked():
            for index, row in enumerate(self.bonds_table.rows(), start=1):
                atom1, atom2 = row[0], row[1]
                if not atom1 or not atom2:
                    raise ValueError(
                        f"Covalent bond {index} needs both atoms "
                        "(CHAIN:RESIDUE:ATOM_NAME), or remove the row.")
                argv += ["--bond", f"{atom1},{atom2}"]
        return argv
