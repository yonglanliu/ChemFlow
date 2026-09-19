"""Workflow pages backed by the existing ChemFlow command-line interface."""

from __future__ import annotations

import shlex
from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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

from src.desktop.widgets.forms import Card, PageHeader, PathField, action_button, field
from src.desktop.widgets.molecule_view import MoleculeView


Runner = Callable[[list[str]], None]


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


class TrainPage(WorkflowPage):
    def __init__(self, runner: Runner, parent=None):
        card = Card("Training setup", "Select a model family and its reproducible configuration file.")
        model = QComboBox()
        model.addItems(["chemeleon", "graphormer", "ml", "gpt"])
        config = PathField("TOML, YAML, or JSON configuration")
        seed = QSpinBox()
        seed.setRange(0, 2_147_483_647)
        seed.setValue(42)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(14)
        grid.addWidget(field("Model family", model), 0, 0)
        grid.addWidget(field("Random seed", seed, "Used by Graphormer and GPT."), 0, 1)
        grid.addWidget(field("Configuration", config), 1, 0, 1, 2)
        card.body.addLayout(grid)
        run = action_button("Start training")
        card.body.addWidget(run, alignment=Qt.AlignmentFlag.AlignRight)
        super().__init__(
            PageHeader("Model studio", "Train a model", "Launch classical ML, Graphormer, CheMeleon, or GPT experiments."),
            card,
            parent,
        )

        def start() -> None:
            if not config.text():
                raise_or_warn(self, "Choose a training configuration first.")
                return
            args = ["train", model.currentText(), config.text()]
            if model.currentText() in {"graphormer", "gpt"}:
                args.extend(["--seed", str(seed.value())])
            self.launch(runner, args)

        run.clicked.connect(start)


class PredictPage(WorkflowPage):
    def __init__(self, runner: Runner, parent=None):
        card = Card("Inference setup", "Run a saved model against one molecule or a molecular dataset.")
        model = QComboBox()
        model.addItems(["chemeleon", "graphormer", "ml"])
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
        grid.addWidget(field("Task names", tasks, "Required for Graphormer; optional for CheMeleon."), 5, 0, 1, 2)
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
                if name == "graphormer" and not task_names:
                    raise_or_warn(self, "Graphormer requires at least one task name.")
                    return
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


class UncertaintyPage(WorkflowPage):
    def __init__(self, runner: Runner, parent=None):
        card = Card("Bootstrap evaluation", "Estimate confidence intervals for multitask prediction metrics.")
        input_file = PathField("Prediction CSV or Parquet")
        tasks = QLineEdit(); tasks.setPlaceholderText("task:target_column:prediction_column; task2:target2:prediction2")
        output = PathField("Output directory", mode="directory")
        iterations = QSpinBox(); iterations.setRange(10, 1_000_000); iterations.setValue(1000)
        confidence = QDoubleSpinBox(); confidence.setRange(0.5, 0.999); confidence.setDecimals(3); confidence.setSingleStep(0.01); confidence.setValue(0.95)
        distributions = QCheckBox("Save individual bootstrap distributions")
        grid = QGridLayout(); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(14)
        grid.addWidget(field("Predictions", input_file), 0, 0, 1, 2)
        grid.addWidget(field("Task mappings", tasks, "Separate multiple task specifications with semicolons."), 1, 0, 1, 2)
        grid.addWidget(field("Bootstrap samples", iterations), 2, 0)
        grid.addWidget(field("Confidence level", confidence), 2, 1)
        grid.addWidget(field("Output directory", output), 3, 0, 1, 2)
        grid.addWidget(distributions, 4, 0, 1, 2)
        card.body.addLayout(grid)
        run = action_button("Evaluate uncertainty")
        card.body.addWidget(run, alignment=Qt.AlignmentFlag.AlignRight)
        super().__init__(PageHeader("Evaluation", "Quantify uncertainty", "Bootstrap model metrics and generate publication-ready summaries."), card, parent)

        def start() -> None:
            specs = [value.strip() for value in tasks.text().split(";") if value.strip()]
            if not input_file.text() or not output.text() or not specs:
                raise_or_warn(self, "Predictions, at least one task mapping, and output directory are required.")
                return
            args = ["uncertainty", "bootstrap", "--input", input_file.text()]
            for spec in specs: args.extend(["--task", spec])
            args.extend(["--n-bootstrap", str(iterations.value()), "--confidence-level", str(confidence.value()), "--output-dir", output.text()])
            if distributions.isChecked(): args.append("--plot-distributions")
            self.launch(runner, args)

        run.clicked.connect(start)


class DashboardPage(QWidget):
    navigate = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(34, 28, 34, 34)
        layout.setSpacing(18)
        layout.addWidget(PageHeader("ChemFlow desktop", "Molecular intelligence, one workspace", "Train, predict, search, generate, and evaluate from a focused native interface."))
        hero = Card()
        hero.setObjectName("AccentCard")
        hero_layout = QHBoxLayout()
        copy = QVBoxLayout()
        tag = QLabel("END-TO-END DISCOVERY")
        tag.setObjectName("Eyebrow")
        title = QLabel("Move from molecules to decisions")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        text = QLabel("Configure experiments visually while ChemFlow runs the same reproducible CLI workflows underneath.")
        text.setObjectName("Muted"); text.setWordWrap(True)
        start = action_button("Start a training run")
        start.clicked.connect(lambda: self.navigate.emit(1))
        copy.addWidget(tag); copy.addWidget(title); copy.addWidget(text); copy.addSpacing(8); copy.addWidget(start, alignment=Qt.AlignmentFlag.AlignLeft); copy.addStretch()
        hero_layout.addLayout(copy, 3)
        hero_layout.addWidget(MoleculeView(), 4)
        hero.body.addLayout(hero_layout)
        layout.addWidget(hero, 1)
        shortcuts = QHBoxLayout(); shortcuts.setSpacing(14)
        for index, title_text, detail in [
            (2, "Predict", "Score molecular properties"),
            (3, "ADME", "Predict HLM, RLM, and MLM clearance"),
            (4, "Search", "Explore chemical similarity"),
            (5, "Generate", "Sample novel SMILES"),
            (6, "Evaluate", "Bootstrap uncertainty"),
        ]:
            item = Card(title_text, detail)
            button = QPushButton("Open workflow  →")
            button.clicked.connect(lambda _checked=False, page=index: self.navigate.emit(page))
            item.body.addWidget(button)
            shortcuts.addWidget(item)
        layout.addLayout(shortcuts)


def raise_or_warn(parent: QWidget, message: str) -> None:
    QMessageBox.warning(parent, "Missing information", message)
