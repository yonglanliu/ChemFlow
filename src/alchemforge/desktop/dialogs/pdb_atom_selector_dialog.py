"""Structure-aware PDB atom selector used by restraint editors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


@dataclass(frozen=True)
class PDBAtom:
    serial: int
    atom_name: str
    resname: str
    chain: str
    residue_number: int
    insertion_code: str
    element: str
    record: str

    @property
    def key(self) -> tuple[str, int, str, str, int]:
        return (
            self.chain,
            self.residue_number,
            self.insertion_code,
            self.atom_name,
            self.serial,
        )


def read_pdb_atoms(path: str | Path) -> list[PDBAtom]:
    """Read selectable ATOM/HETATM records from a PDB file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"PDB file does not exist: {source}")
    atoms: list[PDBAtom] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        record = line[:6].strip().upper()
        if record not in {"ATOM", "HETATM"}:
            continue
        altloc = line[16:17].strip()
        if altloc not in {"", "A"}:
            continue
        try:
            serial = int(line[6:11])
            residue_number = int(line[22:26])
        except ValueError:
            continue
        atom_name = line[12:16].strip().upper()
        resname = line[17:20].strip().upper()
        if not atom_name or not resname:
            continue
        atoms.append(
            PDBAtom(
                serial=serial,
                atom_name=atom_name,
                resname=resname,
                chain=line[21:22].strip() or "(blank)",
                residue_number=residue_number,
                insertion_code=line[26:27].strip(),
                element=(line[76:78].strip() or atom_name[:1]).upper(),
                record=record,
            )
        )
    if not atoms:
        raise ValueError(f"No ATOM or HETATM records were found in {source}")
    return atoms


class PDBAtomSelectorDialog(QDialog):
    """Filter and select atoms while retaining full PDB identity metadata."""

    selection_applied = pyqtSignal(str, object)

    def __init__(
        self,
        *,
        component: str,
        structure_path: str = "",
        working_directory: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.component = component
        self.working_directory = str(Path(working_directory or ".").resolve())
        self.atoms: list[PDBAtom] = []
        self.selected_keys: set[tuple[str, int, str, str, int]] = set()
        self.setWindowTitle(f"Select {component.title()} Restraint Atoms")
        self.resize(900, 620)

        root = QVBoxLayout(self)
        source_row = QHBoxLayout()
        self.source = QLineEdit(structure_path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        load = QPushButton("Read PDB")
        load.clicked.connect(self.load_pdb)
        source_row.addWidget(QLabel("Structure PDB:"))
        source_row.addWidget(self.source, 1)
        source_row.addWidget(browse)
        source_row.addWidget(load)
        root.addLayout(source_row)

        filters = QHBoxLayout()
        self.chain_filter = QComboBox()
        self.resname_filter = QComboBox()
        self.residue_filter = QComboBox()
        self.atom_filter = QLineEdit()
        self.atom_filter.setPlaceholderText("Atom name contains…")
        for label, widget in (
            ("Chain:", self.chain_filter),
            ("Residue name:", self.resname_filter),
            ("Residue number:", self.residue_filter),
            ("Atom:", self.atom_filter),
        ):
            filters.addWidget(QLabel(label))
            filters.addWidget(widget)
        root.addLayout(filters)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Use", "Serial", "Chain", "Resname", "Residue", "I-code", "Atom", "Element"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._item_changed)
        root.addWidget(self.table, 1)

        controls = QHBoxLayout()
        select_visible = QPushButton("Select visible")
        select_visible.clicked.connect(self._select_visible)
        select_rows = QPushButton("Select highlighted")
        select_rows.clicked.connect(self._select_highlighted)
        clear = QPushButton("Clear selection")
        clear.clicked.connect(self._clear_selection)
        apply = QPushButton("Apply Selection")
        apply.clicked.connect(self._apply)
        controls.addWidget(select_visible)
        controls.addWidget(select_rows)
        controls.addWidget(clear)
        controls.addStretch()
        self.status = QLabel("Read a PDB to inspect its atoms.")
        controls.addWidget(self.status)
        controls.addWidget(apply)
        root.addLayout(controls)

        self.chain_filter.currentTextChanged.connect(self._refresh_table)
        self.resname_filter.currentTextChanged.connect(self._refresh_table)
        self.residue_filter.currentTextChanged.connect(self._refresh_table)
        self.atom_filter.textChanged.connect(self._refresh_table)
        if structure_path and Path(structure_path).is_file():
            self.load_pdb()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select restraint structure", self.working_directory, "PDB files (*.pdb)"
        )
        if path:
            self.source.setText(path)
            self.load_pdb()

    def load_pdb(self) -> None:
        try:
            self.atoms = read_pdb_atoms(self.source.text().strip())
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot Read PDB", str(error))
            return
        self.selected_keys.clear()
        self._set_filter(self.chain_filter, sorted({atom.chain for atom in self.atoms}))
        self._set_filter(
            self.resname_filter, sorted({atom.resname for atom in self.atoms})
        )
        self._set_filter(
            self.residue_filter,
            [str(value) for value in sorted({atom.residue_number for atom in self.atoms})],
        )
        self._refresh_table()

    @staticmethod
    def _set_filter(widget: QComboBox, values: list[str]) -> None:
        widget.blockSignals(True)
        widget.clear()
        widget.addItem("All")
        widget.addItems(values)
        widget.blockSignals(False)

    def _filtered_atoms(self) -> list[PDBAtom]:
        chain = self.chain_filter.currentText()
        resname = self.resname_filter.currentText()
        residue = self.residue_filter.currentText()
        atom_text = self.atom_filter.text().strip().upper()
        return [
            atom
            for atom in self.atoms
            if (chain in {"", "All"} or atom.chain == chain)
            and (resname in {"", "All"} or atom.resname == resname)
            and (residue in {"", "All"} or atom.residue_number == int(residue))
            and (not atom_text or atom_text in atom.atom_name)
        ]

    def _refresh_table(self, *_args) -> None:
        atoms = self._filtered_atoms()
        self.table.blockSignals(True)
        self.table.setRowCount(len(atoms))
        for row, atom in enumerate(atoms):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            use.setCheckState(Qt.Checked if atom.key in self.selected_keys else Qt.Unchecked)
            use.setData(Qt.UserRole, atom.key)
            self.table.setItem(row, 0, use)
            for column, value in enumerate(
                (
                    atom.serial,
                    atom.chain,
                    atom.resname,
                    atom.residue_number,
                    atom.insertion_code,
                    atom.atom_name,
                    atom.element,
                ),
                start=1,
            ):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        self._update_status(len(atoms))

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        key = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self.selected_keys.add(key)
        else:
            self.selected_keys.discard(key)
        self._update_status(self.table.rowCount())

    def _select_visible(self) -> None:
        for atom in self._filtered_atoms():
            self.selected_keys.add(atom.key)
        self._refresh_table()

    def _select_highlighted(self) -> None:
        for index in self.table.selectionModel().selectedRows():
            item = self.table.item(index.row(), 0)
            self.selected_keys.add(item.data(Qt.UserRole))
        self._refresh_table()

    def _clear_selection(self) -> None:
        self.selected_keys.clear()
        self._refresh_table()

    def _update_status(self, visible: int) -> None:
        self.status.setText(f"{visible} visible · {len(self.selected_keys)} selected")

    def _apply(self) -> None:
        selected = [atom for atom in self.atoms if atom.key in self.selected_keys]
        if not selected:
            QMessageBox.information(self, "No Atoms Selected", "Select at least one atom.")
            return
        self.selection_applied.emit(
            self.component, [asdict(atom) for atom in selected]
        )
        self.close()
