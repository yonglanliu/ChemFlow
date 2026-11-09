"""Workflow pages backed by the existing ChemFlow command-line interface."""

from __future__ import annotations

import shlex
from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from chemflow.desktop.widgets.forms import Card, PageHeader, PathField, action_button, field
from chemflow.desktop.widgets.molecule_view import MoleculeView


Runner = Callable[[list[str]], None]


def launcher_view(functions, query, expanded, max_visible):
    """Decide which workflow cards to show and the 'more' button mode.

    Returns ``(visible, mode, hidden)`` where ``mode`` is one of ``"none"``,
    ``"explore"`` (collapsed with extras hidden) or ``"fewer"`` (expanded).
    Search (a non-empty query) shows every match and hides the button.
    """
    q = (query or "").strip().lower()
    if q:
        matches = [f for f in functions
                   if q in f["title"].lower() or q in f.get("detail", "").lower()]
        return matches, "none", 0
    matches = list(functions)
    if len(matches) <= max_visible:
        return matches, "none", 0
    if expanded:
        return matches, "fewer", 0
    return matches[:max_visible], "explore", len(matches) - max_visible



class WorkflowPage(QWidget):
    def __init__(self, header: PageHeader, card: Card, parent=None):
        super().__init__(parent)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(20)
        layout.addWidget(header)
        layout.addWidget(card)
        layout.addStretch()
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    @staticmethod
    def launch(runner: Runner, arguments: list[str]) -> None:
        try:
            runner(arguments)
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(None, "Cannot start task", str(error))


class PredictPage(WorkflowPage):
    def __init__(self, runner: Runner, parent=None):
        card = Card("Inference setup", "Run a saved model against one molecule or a molecular dataset.")
        model = QComboBox()
        model.addItems(["chemeleon", "ml"])
        source_type = QComboBox()
        source_type.addItems(["Dataset file", "Single SMILES"])
        source = PathField("CSV, Parquet, SMI, or TXT input")
        smiles = QLineEdit()
        smiles.setPlaceholderText("For example: CC(=O)Oc1ccccc1C(=O)O")
        structure = QLineEdit("SMILES")
        checkpoint = PathField("Model checkpoint or .pkl package")
        tasks = QLineEdit()
        tasks.setPlaceholderText("Space-separated task names")
        output = PathField("predictions.csv", mode="save", save_filter="CSV (*.csv);;Parquet (*.parquet)")
        output.setText("predictions.csv")
        batch = QSpinBox()
        batch.setRange(1, 65536)
        batch.setValue(64)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(14)
        grid.addWidget(field("Model family", model), 0, 0)
        grid.addWidget(field("Input type", source_type), 0, 1)
        source_box = field("Dataset", source)
        smiles_box = field("SMILES", smiles)
        grid.addWidget(source_box, 1, 0, 1, 2)
        grid.addWidget(smiles_box, 2, 0, 1, 2)
        grid.addWidget(field("Checkpoint / model", checkpoint), 3, 0, 1, 2)
        grid.addWidget(field("Structure column", structure), 4, 0)
        grid.addWidget(field("Batch size", batch), 4, 1)
        grid.addWidget(field("Task names", tasks, "Optional for CheMeleon."), 5, 0, 1, 2)
        grid.addWidget(field("Output", output), 6, 0, 1, 2)
        card.body.addLayout(grid)
        run = action_button("Run prediction")
        card.body.addWidget(run, alignment=Qt.AlignmentFlag.AlignRight)
        super().__init__(PageHeader("Inference", "Predict molecular properties", "Use trained models without leaving the desktop workspace."), card, parent)

        def input_changed(index: int) -> None:
            source_box.setVisible(index == 0)
            smiles_box.setVisible(index == 1)

        def model_changed(name: str) -> None:
            tasks.setPlaceholderText("Output column name" if name == "ml" else "Space-separated task names")
            batch.setEnabled(name != "ml")

        def start() -> None:
            input_value = source.text() if source_type.currentIndex() == 0 else smiles.text().strip()
            if not input_value or not checkpoint.text() or not output.text():
                raise_or_warn(self, "Input, model checkpoint, and output are required.")
                return
            name = model.currentText()
            args = ["predict", name]
            args.extend(["--input" if source_type.currentIndex() == 0 else "--smiles", input_value])
            args.extend(["--structure-column", structure.text().strip() or "SMILES"])
            task_names = shlex.split(tasks.text())
            if name == "ml":
                args.extend(["--model", checkpoint.text(), "--task-name", task_names[0] if task_names else "prediction"])
            else:
                if task_names:
                    args.extend(["--task-names", *task_names])
                args.extend(["--model-checkpoint", checkpoint.text(), "--batch-size", str(batch.value())])
            args.extend(["--output", output.text()])
            self.launch(runner, args)

        source_type.currentIndexChanged.connect(input_changed)
        model.currentTextChanged.connect(model_changed)
        run.clicked.connect(start)
        input_changed(source_type.currentIndex())


class SearchPage(WorkflowPage):
    def __init__(self, runner: Runner, parent=None):
        card = Card("Similarity search", "Rank a molecular database against a SMILES query or query file.")
        query_type = QComboBox()
        query_type.addItems(["Single SMILES", "Query file"])
        query = QLineEdit()
        query.setPlaceholderText("Enter a query SMILES")
        query_file = PathField("Query molecular file")
        database = PathField("Database CSV, Parquet, or SMI file")
        representation = QComboBox()
        representation.addItems(["ecfp4", "ecfp6", "fcfp4", "fcfp6", "maccs", "2d_descriptor"])
        metric = QComboBox()
        metric.addItems(["tanimoto", "dice", "cosine", "euclidean", "manhattan", "mcconnaughey"])
        workers = QSpinBox()
        workers.setRange(1, 256)
        workers.setValue(1)
        job = QLineEdit("similarity")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(14)
        grid.addWidget(field("Query type", query_type), 0, 0)
        grid.addWidget(field("Job name", job), 0, 1)
        query_box = field("Query SMILES", query)
        file_box = field("Query file", query_file)
        grid.addWidget(query_box, 1, 0, 1, 2)
        grid.addWidget(file_box, 2, 0, 1, 2)
        grid.addWidget(field("Database", database), 3, 0, 1, 2)
        grid.addWidget(field("Representation", representation), 4, 0)
        grid.addWidget(field("Metric", metric), 4, 1)
        grid.addWidget(field("Workers", workers), 5, 0)
        card.body.addLayout(grid)
        run = action_button("Search database")
        card.body.addWidget(run, alignment=Qt.AlignmentFlag.AlignRight)
        super().__init__(PageHeader("Discovery", "Search chemical space", "Fingerprint and descriptor similarity search powered by ChemFlow."), card, parent)

        def type_changed(index: int) -> None:
            query_box.setVisible(index == 0)
            file_box.setVisible(index == 1)

        def start() -> None:
            query_value = query.text().strip() if query_type.currentIndex() == 0 else query_file.text()
            if not query_value or not database.text():
                raise_or_warn(self, "Query and database are required.")
                return
            args = ["search", "similarity"]
            args.extend(["--query_smiles" if query_type.currentIndex() == 0 else "--query_file", query_value])
            args.extend(["--database", database.text(), "--rep_type", representation.currentText(), "--metric", metric.currentText(), "--num_workers", str(workers.value()), "--job_name", job.text().strip() or "similarity"])
            self.launch(runner, args)

        query_type.currentIndexChanged.connect(type_changed)
        run.clicked.connect(start)
        type_changed(query_type.currentIndex())


class GeneratePage(WorkflowPage):
    def __init__(self, runner: Runner, parent=None):
        card = Card("GPT generation", "Sample new molecular strings from a trained ChemFlow language model.")
        checkpoint = PathField("GPT checkpoint")
        tokenizer = PathField("Tokenizer file")
        adapter = PathField("Optional LoRA adapter checkpoint")
        output = PathField("generated_smiles.txt", mode="save", save_filter="Text (*.txt);;SMILES (*.smi)")
        output.setText("generated_smiles.txt")
        prompt = QLineEdit()
        prompt.setPlaceholderText("Optional generation prompt")
        samples = QSpinBox(); samples.setRange(1, 1_000_000); samples.setValue(10)
        tokens = QSpinBox(); tokens.setRange(1, 4096); tokens.setValue(100)
        temperature = QDoubleSpinBox(); temperature.setRange(0.01, 10); temperature.setSingleStep(0.1); temperature.setValue(0.8)
        top_k = QSpinBox(); top_k.setRange(1, 10000); top_k.setValue(20)
        grid = QGridLayout(); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(14)
        grid.addWidget(field("Checkpoint", checkpoint), 0, 0, 1, 2)
        grid.addWidget(field("Tokenizer", tokenizer), 1, 0, 1, 2)
        grid.addWidget(field("LoRA adapter", adapter), 2, 0, 1, 2)
        grid.addWidget(field("Prompt", prompt), 3, 0, 1, 2)
        grid.addWidget(field("Samples", samples), 4, 0)
        grid.addWidget(field("Maximum new tokens", tokens), 4, 1)
        grid.addWidget(field("Temperature", temperature), 5, 0)
        grid.addWidget(field("Top-k", top_k), 5, 1)
        grid.addWidget(field("Output", output), 6, 0, 1, 2)
        card.body.addLayout(grid)
        run = action_button("Generate molecules")
        card.body.addWidget(run, alignment=Qt.AlignmentFlag.AlignRight)
        super().__init__(PageHeader("Generative chemistry", "Design new molecules", "Configure and run GPT molecular generation."), card, parent)

        def start() -> None:
            if not checkpoint.text() or not tokenizer.text() or not output.text():
                raise_or_warn(self, "Checkpoint, tokenizer, and output are required.")
                return
            args = ["generate", "gpt", "--checkpoint", checkpoint.text(), "--tokenizer", tokenizer.text(), "--output", output.text(), "--num_samples", str(samples.value()), "--max_new_tokens", str(tokens.value()), "--temperature", str(temperature.value()), "--top_k", str(top_k.value())]
            if adapter.text(): args.extend(["--adapter_checkpoint", adapter.text()])
            if prompt.text().strip(): args.extend(["--prompt", prompt.text().strip()])
            self.launch(runner, args)

        run.clicked.connect(start)


class DashboardPage(QWidget):
    navigate = Signal(int)

    MAX_VISIBLE = 10          # show at most this many before "Explore more"

    def __init__(self, functions=None, parent=None):
        super().__init__(parent)
        # functions: list of {"index": int, "title": str, "detail": str}
        self._functions = list(functions or [])
        self._expanded = False
        self._query = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(18)
        layout.addWidget(PageHeader(
            "ChemFlow desktop", "Molecular intelligence, one workspace",
            "Pick a workflow to begin. Search to jump straight to one."))

        search_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search workflows…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_search)
        search_row.addWidget(self._search)
        layout.addLayout(search_row)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(14)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._grid_host)
        layout.addWidget(scroll, 1)

        self._more_button = QPushButton()
        self._more_button.clicked.connect(self._toggle_expand)
        layout.addWidget(self._more_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._empty = QLabel("No workflows match your search.")
        self._empty.setObjectName("Muted")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty)

        self._rebuild()

    def _on_search(self, text: str) -> None:
        self._query = text.strip().lower()
        self._rebuild()

    def _toggle_expand(self) -> None:
        self._expanded = not self._expanded
        self._rebuild()

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _rebuild(self) -> None:
        self._clear_grid()
        visible, mode, hidden = launcher_view(
            self._functions, self._query, self._expanded, self.MAX_VISIBLE)

        self._empty.setVisible(not visible)
        columns = 3
        for position, function in enumerate(visible):
            card = Card(function["title"], function.get("detail", ""))
            button = QPushButton("Open workflow  →")
            button.clicked.connect(lambda _checked=False, page=function["index"]: self.navigate.emit(page))
            card.body.addWidget(button)
            self._grid.addWidget(card, position // columns, position % columns)

        if mode == "explore":
            self._more_button.setText(f"Explore more ({hidden} more)")
            self._more_button.setVisible(True)
        elif mode == "fewer":
            self._more_button.setText("Show fewer")
            self._more_button.setVisible(True)
        else:
            self._more_button.setVisible(False)



def raise_or_warn(parent: QWidget, message: str) -> None:
    QMessageBox.warning(parent, "Missing information", message)
