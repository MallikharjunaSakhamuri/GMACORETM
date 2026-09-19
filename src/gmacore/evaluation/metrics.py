"""Evaluation metrics for the downstream benchmarks.

Classification benchmarks report ROC-AUC and PR-AUC; regression benchmarks
report RMSE, MAE and the coefficient of determination. All metrics are computed
per task and then averaged over tasks, with missing labels masked out. Tasks
containing only one class in the evaluation split are skipped for ROC-AUC and
PR-AUC, since those metrics are undefined there.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

CLASSIFICATION_METRICS = ("roc_auc", "pr_auc")
REGRESSION_METRICS = ("rmse", "mae", "r2")


def _valid_mask(targets: np.ndarray) -> np.ndarray:
    return ~np.isnan(targets)


def classification_metrics(targets: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    """ROC-AUC and PR-AUC averaged over tasks.

    `logits` are raw model outputs; the sigmoid is monotonic, so ROC-AUC is
    unaffected by applying it, but PR-AUC is reported on probabilities for
    interpretability.
    """
    targets = np.asarray(targets, dtype=float)
    logits = np.asarray(logits, dtype=float)
    if targets.ndim == 1:
        targets = targets[:, None]
        logits = logits[:, None]

    probabilities = 1.0 / (1.0 + np.exp(-logits))

    roc_scores: List[float] = []
    pr_scores: List[float] = []

    for task in range(targets.shape[1]):
        mask = _valid_mask(targets[:, task])
        y_true = targets[mask, task]
        y_score = probabilities[mask, task]

        if y_true.size == 0 or np.unique(y_true).size < 2:
            continue

        roc_scores.append(float(roc_auc_score(y_true, y_score)))
        pr_scores.append(float(average_precision_score(y_true, y_score)))

    return {
        "roc_auc": float(np.mean(roc_scores)) if roc_scores else float("nan"),
        "pr_auc": float(np.mean(pr_scores)) if pr_scores else float("nan"),
        "num_tasks_scored": len(roc_scores),
    }


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE and R-squared averaged over tasks."""
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if targets.ndim == 1:
        targets = targets[:, None]
        predictions = predictions[:, None]

    rmse_scores: List[float] = []
    mae_scores: List[float] = []
    r2_scores: List[float] = []

    for task in range(targets.shape[1]):
        mask = _valid_mask(targets[:, task])
        y_true = targets[mask, task]
        y_pred = predictions[mask, task]
        if y_true.size < 2:
            continue

        rmse_scores.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
        mae_scores.append(float(mean_absolute_error(y_true, y_pred)))
        r2_scores.append(float(r2_score(y_true, y_pred)))

    return {
        "rmse": float(np.mean(rmse_scores)) if rmse_scores else float("nan"),
        "mae": float(np.mean(mae_scores)) if mae_scores else float("nan"),
        "r2": float(np.mean(r2_scores)) if r2_scores else float("nan"),
        "num_tasks_scored": len(rmse_scores),
    }


def compute_metrics(task_type: str, targets: np.ndarray, outputs: np.ndarray) -> Dict[str, float]:
    """Dispatch to the metric set appropriate for the task type."""
    if task_type == "classification":
        return classification_metrics(targets, outputs)
    if task_type == "regression":
        return regression_metrics(targets, outputs)
    raise ValueError(f"Unknown task type '{task_type}'")


def primary_metric(task_type: str) -> str:
    """Metric used for model selection and for the headline result table."""
    return "roc_auc" if task_type == "classification" else "rmse"


def is_higher_better(metric: str) -> bool:
    return metric in {"roc_auc", "pr_auc", "r2"}


def summarize_runs(results: List[Dict[str, float]]) -> Dict[str, str]:
    """Aggregate repeated runs into `mean +/- standard deviation` strings."""
    if not results:
        return {}

    summary: Dict[str, str] = {}
    for key in results[0]:
        values = np.array([run[key] for run in results], dtype=float)
        if np.all(np.isnan(values)):
            continue
        summary[key] = f"{np.nanmean(values):.4f} +/- {np.nanstd(values):.4f}"
    return summary
