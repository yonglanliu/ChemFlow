import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)

from sklearn.preprocessing import label_binarize


def evaluate_model(model, X_test, y_test, task_type):
    task_type = task_type.lower()

    if task_type not in ["regression", "classification"]:
        raise ValueError(f"Invalid task type: {task_type}")

    y_pred = model.predict(X_test)

    results = {
        "y_test": pd.Series(y_test).tolist(),
        "y_pred": pd.Series(y_pred).tolist(),
    }

    if task_type == "regression":
        results.update(
            {
                "test_rmse": float(root_mean_squared_error(y_test, y_pred)),
                "test_mae": float(mean_absolute_error(y_test, y_pred)),
                "test_r2": float(r2_score(y_test, y_pred)),
            }
        )
        return results

    labels = sorted(pd.Series(y_test).dropna().unique())

    results.update(
        {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "classification_report": classification_report(y_test, y_pred, output_dict=True,zero_division=0,),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels,).tolist(),
            "confusion_matrix_labels": [str(x) for x in labels],
        }
    )

    if not hasattr(model, "predict_proba"):
        return results

    try:
        y_prob = model.predict_proba(X_test)
        results["y_proba"] = y_prob.tolist()

        # Compute ROC AUC and PR AUC for binary and multi-class classification
        if y_prob.shape[1] == 2: # Binary classification
            y_score = y_prob[:, 1]

            fpr, tpr, roc_thresholds = roc_curve(y_test, y_score)
            precision, recall, pr_thresholds = precision_recall_curve(y_test, y_score)

            results["test_roc_auc"] = float(roc_auc_score(y_test, y_score))
            results["test_average_precision"] = float(average_precision_score(y_test, y_score))
            results["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist(),}
            results["pr_curve"] = {"precision": precision.tolist(),"recall": recall.tolist(), "thresholds": pr_thresholds.tolist(),}

        else: # Multi-class classification

            # Compute macro-average ROC AUC score
            results["test_roc_auc"] = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro", ))

            # Compute ROC and PR curves for each class
            y_test_bin = label_binarize(y_test, classes=labels)

            roc_curves = {}
            pr_curves = {}

            for i, label in enumerate(labels):
                fpr, tpr, roc_thresholds = roc_curve(y_test_bin[:, i], y_prob[:, i],)
                precision, recall, pr_thresholds = precision_recall_curve(y_test_bin[:, i], y_prob[:, i],)
                ap = average_precision_score(y_test_bin[:, i], y_prob[:, i])

                roc_curves[str(label)] = {"fpr": fpr.tolist(), 
                                          "tpr": tpr.tolist(), 
                                          "thresholds": roc_thresholds.tolist(),}

                pr_curves[str(label)] = {"precision": precision.tolist(), 
                                         "recall": recall.tolist(), 
                                         "thresholds": pr_thresholds.tolist(), 
                                         "average_precision": float(ap),}

            results["roc_curves_ovr"] = roc_curves
            results["pr_curves_ovr"] = pr_curves

    except Exception as e:
        results["test_roc_auc"] = None
        results["auc_error"] = str(e)

    return results