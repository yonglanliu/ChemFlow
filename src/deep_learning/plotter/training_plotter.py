from __future__ import annotations
from typing import Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay



class ClassificationPlotter:
    def __init__(self, config: dict[str, Any]) -> None:

        self.config = config

    def plot_classification_curves(self, data: dict[str, Any], output_dir: str | Path, prefix: str = "test") -> None:
        """
        Plot ROC curve, precision-recall curve, and confusion matrix.
        Parameters
        ----------
        data
            Curve data returned by ClassificationPlotter.compute_curve_data().
        output_dir
            Directory in which plots will be saved.
        prefix
            Filename prefix, such as ``"test"``.
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dpi = int(getattr(self.config, "plot_dpi", 300))

        prefix = prefix.rstrip("_")

        if "roc" in data:
            self._plot_roc_curve(roc_data=data["roc"], output_path=output_dir / f"{prefix}_roc_curve.png", dpi=dpi)

        if "pr" in data:
            self._plot_precision_recall_curve(pr_data=data["pr"], output_path=output_dir / f"{prefix}_pr_curve.png", dpi=dpi)

        if "confusion_matrix" in data:
            self._plot_confusion_matrix(
                matrix=data["confusion_matrix"],
                output_path=output_dir / f"{prefix}_confusion_matrix.png",
                dpi=dpi,
                class_names=data.get("class_names"),
            )

    @staticmethod
    def _plot_roc_curve(roc_data: dict[str, Any], output_path: str | Path, dpi: int = 300) -> None:
        """
        Plot a binary ROC curve.
        """
        fpr = np.asarray(roc_data["fpr"], dtype=np.float64)
        tpr = np.asarray(roc_data["tpr"], dtype=np.float64)

        if fpr.shape != tpr.shape:
            raise ValueError(
                "ROC fpr and tpr must have the same shape, "
                f"got {fpr.shape} and {tpr.shape}."
            )

        auc_value = roc_data.get("auc")

        plt.figure(figsize=(7, 6))

        if auc_value is None:
            plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
        else:
            plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {float(auc_value):.4f})")

        # Random-classifier baseline.
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.5, label="Random classifier")

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()


    @staticmethod
    def _plot_precision_recall_curve(pr_data: dict[str, Any], output_path: str | Path, dpi: int = 300) -> None:
        """
        Plot a binary precision-recall curve.
        """
        precision = np.asarray(pr_data["precision"], dtype=np.float64, )
        recall = np.asarray(pr_data["recall"], dtype=np.float64)

        if precision.shape != recall.shape:
            raise ValueError(
                "PR precision and recall must have the same shape, "
                f"got {precision.shape} and {recall.shape}."
            )

        average_precision = pr_data.get("average_precision")
        positive_prevalence = pr_data.get("positive_prevalence")

        plt.figure(figsize=(7, 6))

        if average_precision is None:
            plt.plot(recall, precision, linewidth=2, label="Precision–recall curve")
        else:
            plt.plot(recall, precision, linewidth=2, label=f"Precision–recall curve (AP = {float(average_precision):.4f})")

        # For PR curves, the random baseline is the positive-class prevalence.
        if positive_prevalence is not None:
            prevalence = float(positive_prevalence)
            plt.axhline(y=prevalence, linestyle="--", linewidth=1.5, label=f"Baseline = {prevalence:.4f}")

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()


    @staticmethod
    def _plot_confusion_matrix(matrix: Any, output_path: str | Path, dpi: int = 300, class_names: list[str] | None = None) -> None:
        """
        Plot a confusion matrix.
        """
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Confusion matrix must be a square 2D array, got shape {matrix.shape}.")
        num_classes = matrix.shape[0]
        if class_names is None:
            class_names = [str(index) for index in range(num_classes)]

        if len(class_names) != num_classes:
            raise ValueError(f"class_names length must match confusion-matrix size: {len(class_names)} versus {num_classes}.")

        figure_size = max(6.0, num_classes * 1.2)
        figure, axis = plt.subplots(figsize=(figure_size, figure_size))
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
        display.plot(ax=axis, values_format="d", colorbar=False)

        axis.set_title("Confusion Matrix")
        figure.tight_layout()
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)