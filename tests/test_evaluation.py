"""Tests for the downstream metrics and representation analysis."""

import numpy as np
import pytest

from gmacore.evaluation.metrics import (
    classification_metrics,
    compute_metrics,
    primary_metric,
    regression_metrics,
    summarize_runs,
)
from gmacore.evaluation.representation import (
    centroid_trajectory,
    encoder_key_alignment,
    representation_drift,
)


def test_perfect_classifier_scores_one():
    targets = np.array([[0.0], [0.0], [1.0], [1.0]])
    logits = np.array([[-5.0], [-4.0], [4.0], [5.0]])
    assert classification_metrics(targets, logits)["roc_auc"] == pytest.approx(1.0)


def test_missing_labels_are_masked():
    targets = np.array([[0.0, np.nan], [1.0, np.nan], [0.0, 1.0], [1.0, 0.0]])
    logits = np.array([[-1.0, 0.5], [1.0, 0.5], [-1.0, 0.9], [1.0, -0.9]])
    result = classification_metrics(targets, logits)
    assert result["num_tasks_scored"] == 2
    assert np.isfinite(result["roc_auc"])


def test_single_class_task_is_skipped():
    targets = np.array([[1.0], [1.0], [1.0]])
    logits = np.array([[0.1], [0.2], [0.3]])
    assert classification_metrics(targets, logits)["num_tasks_scored"] == 0


def test_perfect_regression_scores():
    targets = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = regression_metrics(targets, targets.copy())
    assert result["rmse"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_primary_metric_selection():
    assert primary_metric("classification") == "roc_auc"
    assert primary_metric("regression") == "rmse"


def test_compute_metrics_rejects_unknown_task():
    with pytest.raises(ValueError):
        compute_metrics("ranking", np.zeros((2, 1)), np.zeros((2, 1)))


def test_summary_reports_mean_and_deviation():
    runs = [{"roc_auc": 0.90}, {"roc_auc": 0.92}, {"roc_auc": 0.94}]
    assert summarize_runs(runs)["roc_auc"].startswith("0.9200 +/-")


# ----------------------------------------------------------------------
def test_identical_checkpoints_show_no_drift():
    embeddings = np.random.default_rng(0).normal(size=(50, 16))
    result = representation_drift(embeddings, embeddings.copy())
    assert result["drift_mean"] == pytest.approx(0.0, abs=1e-9)


def test_alignment_is_one_for_identical_embeddings():
    embeddings = np.random.default_rng(0).normal(size=(50, 16))
    result = encoder_key_alignment(embeddings, embeddings.copy())
    assert result["alignment_mean"] == pytest.approx(1.0, abs=1e-6)


def test_drift_requires_matching_shapes():
    with pytest.raises(ValueError):
        representation_drift(np.zeros((10, 4)), np.zeros((10, 8)))


def test_monotone_trajectory_is_directionally_consistent():
    """A centroid moving steadily in one direction scores close to one."""
    sequence = [np.full((20, 4), step, dtype=float) for step in range(5)]
    result = centroid_trajectory(sequence)
    assert result["directional_consistency"] == pytest.approx(1.0, abs=1e-6)


def test_oscillating_trajectory_is_directionally_inconsistent():
    offsets = [0.0, 1.0, 0.0, 1.0, 0.0]
    sequence = [np.full((20, 4), offset) for offset in offsets]
    result = centroid_trajectory(sequence)
    assert result["directional_consistency"] == pytest.approx(-1.0, abs=1e-6)
