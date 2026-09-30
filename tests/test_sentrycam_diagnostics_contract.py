"""Contract for src.diagnostics (SentryCam-inspired instability signal, arXiv:2405.15135).

src/diagnostics.py does not exist yet: this whole module is expected to fail to collect
(ImportError) until the implementer adds it. That failure is correct and intentional for a
frozen-test-first workflow -- do not "fix" it by loosening these tests.

Width: solitary for cluster_drift_metrics and sentrycam_alert_epoch (pure functions, hand
computed oracles); one cluster test at the bottom cross-checks TrainResult against the standalone
function so the two can never silently diverge.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from src.diagnostics import cluster_drift_metrics, sentrycam_alert_epoch
from src.experiment import ExperimentConfig
from src.train import train_classifier

# --------------------------------------------------------------------------- builders
# Deliberately duplicated (not imported) from tests/test_train_contract.py's private builders:
# importing across test modules would make this file's collectability depend on import order of
# an unrelated file, and would tie an ImportError in test_train_contract.py to this file too.
# Kept intentionally tiny.


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))


def _config(**overrides) -> ExperimentConfig:
    settings = dict(condition="clean", seed=17, max_epochs=60, patience=10, microbatch_size=8, accumulation_steps=2)
    settings.update(overrides)
    return ExperimentConfig(**settings)


def _separable(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x = 2.0 * np.random.default_rng(seed).normal(size=(10 * n, 2)).astype(np.float32)
    x = x[np.abs(x.sum(axis=1)) > 1.0][:n]
    assert x.shape[0] == n
    return x, (x.sum(axis=1) > 0).astype(np.int64)


# --------------------------------------------------------------------------- cluster_drift_metrics


def test_cluster_drift_metrics_matches_hand_computed_centroids_and_distance_for_two_classes() -> None:
    """class 0: (0,0),(2,0) -> centroid (1,0), intra-var mean(1,1)=1.
    class 1: (0,4),(0,6) -> centroid (0,5), intra-var mean(1,1)=1.
    inter-cluster distance: ||(1,0)-(0,5)|| = sqrt(1+25) = sqrt(26)."""
    logits = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [0.0, 6.0]])
    labels = np.array([0, 0, 1, 1])

    inter, intra = cluster_drift_metrics(logits, labels)

    assert intra == pytest.approx(1.0)
    assert inter == pytest.approx(math.sqrt(26.0))


def test_cluster_drift_intra_variance_weights_each_present_class_equally_not_by_row_count() -> None:
    """class 0 has 2 rows at (0,0)/(4,0) -> centroid (2,0), intra-var 4.0.
    class 1 has 6 rows split evenly between (0,0) and (2,0) -> centroid (1,0), intra-var 1.0.
    Equal-per-class weighting gives mean(4.0, 1.0) = 2.5; row-weighted would give
    (2*4.0 + 6*1.0) / 8 = 1.75 -- the two must NOT be confused."""
    logits = np.array([
        [0.0, 0.0], [4.0, 0.0],
        [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        [2.0, 0.0], [2.0, 0.0], [2.0, 0.0],
    ])
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 1])

    inter, intra = cluster_drift_metrics(logits, labels)

    assert intra == pytest.approx(2.5)
    assert inter == pytest.approx(1.0)


def test_cluster_drift_inter_distance_is_the_mean_of_every_pairwise_centroid_distance() -> None:
    """Three classes with centroids (0,0), (3,0), (0,4): a 3-4-5 triangle, so the mean pairwise
    centroid distance is (3+4+5)/3 = 4.0. Every row already sits exactly at its centroid, so
    intra-class variance is exactly zero."""
    logits = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
    labels = np.array([0, 0, 1, 1, 2, 2])

    inter, intra = cluster_drift_metrics(logits, labels)

    assert inter == pytest.approx(4.0)
    assert intra == pytest.approx(0.0)


def test_cluster_drift_inter_distance_is_exactly_zero_with_fewer_than_two_present_classes() -> None:
    logits = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 0.0]])
    labels = np.array([5, 5, 5])

    inter, intra = cluster_drift_metrics(logits, labels)

    assert inter == 0.0
    assert intra == pytest.approx(np.mean([
        (1.0 - 4.0 / 3) ** 2 + (2.0 - 1.0 / 3) ** 2,
        (3.0 - 4.0 / 3) ** 2 + (-1.0 - 1.0 / 3) ** 2,
        (0.0 - 4.0 / 3) ** 2 + (0.0 - 1.0 / 3) ** 2,
    ]))


@pytest.mark.parametrize("logits,labels", [
    (np.zeros(3), np.array([0, 1, 0])),                      # logits.ndim != 2
    (np.zeros((3, 2)), np.array([[0], [1], [0]])),           # labels.ndim != 1
    (np.zeros((3, 2)), np.array([0, 1])),                    # row count mismatch
    (np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)),      # empty input
])
def test_cluster_drift_metrics_rejects_malformed_shapes(logits, labels) -> None:
    with pytest.raises(ValueError):
        cluster_drift_metrics(logits, labels)


# --------------------------------------------------------------------------- sentrycam_alert_epoch
# k=2, alpha=0.25, window=10 (module defaults) unless a test overrides a kwarg explicitly.
# Every history below was checked against the spec's own formula before being frozen here.


def test_sentrycam_alert_fires_at_the_earliest_epoch_with_sustained_decrease_and_a_significant_jump() -> None:
    """Flat for 6 epochs (0..5), then a two-step sustained decrease (10->9->8) at epoch 6, then a
    much larger drop that would extend the decrease further at epoch 7. Epoch 6 is the earliest
    epoch where the k=2 sustained-decrease condition holds AND the jump clears alpha*sigma of the
    preceding window; epochs 2-5 fail the direction check because the history is still flat there."""
    distance_history = [10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 8.0, 2.0]
    variance_history = [1.0] * len(distance_history)

    assert sentrycam_alert_epoch(distance_history, variance_history) == 6


def test_sentrycam_alert_fires_on_the_variance_condition_with_a_sustained_increase_and_a_significant_jump() -> None:
    """Mirror of the distance case: distance_history stays flat (never fires), variance_history
    sustains an increase and then jumps at epoch 6."""
    distance_history = [5.0] * 8
    variance_history = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 9.0]

    assert sentrycam_alert_epoch(distance_history, variance_history) == 6


def test_sentrycam_alert_returns_none_for_a_flat_noisy_history_with_no_sustained_direction() -> None:
    distance_history = [5.0, 5.2, 5.0, 5.3, 5.1, 5.2, 5.0, 5.3]
    variance_history = [1.0, 1.1, 1.0, 1.05, 1.0, 1.02, 1.0, 1.03]

    assert sentrycam_alert_epoch(distance_history, variance_history) is None


def test_sentrycam_alert_does_not_fire_when_the_direction_holds_but_the_jump_is_not_significant() -> None:
    """Epochs 4-6 do sustain a decrease (30.0 -> 29.9 -> 29.8 -> 29.7), which alone might look
    like a trigger, but the preceding window (10, 20, 5, 30, 29.9, ...) has a large sigma from
    the earlier swings, so no single small step clears alpha*sigma. Direction without
    significance must not fire."""
    distance_history = [10.0, 20.0, 5.0, 30.0, 29.9, 29.8, 29.7]
    variance_history = [1.0] * len(distance_history)

    assert sentrycam_alert_epoch(distance_history, variance_history) is None


def test_sentrycam_alert_never_fires_from_a_degenerate_significance_window() -> None:
    """Same clear sustained-decrease-plus-jump history that fires under the default window=10
    (proving the direction+jump alone would otherwise trigger); overriding window=1 forces every
    significance window to have fewer than 2 points, which must suppress the alert entirely."""
    distance_history = [10.0, 10.0, 10.0, 9.0, 3.0]
    variance_history = [1.0] * len(distance_history)

    assert sentrycam_alert_epoch(distance_history, variance_history) is not None, (
        "sanity check: this history must fire under the default window so the window=1 case below "
        "is shown to suppress a real trigger, not merely fail to find one"
    )
    assert sentrycam_alert_epoch(distance_history, variance_history, window=1) is None


def test_sentrycam_alert_returns_the_earlier_epoch_when_both_conditions_would_fire() -> None:
    """distance_history alone fires at epoch 6 (see the sustained-decrease test above);
    variance_history alone fires at epoch 5. When combined, the smaller (earlier) of the two
    must win."""
    distance_history = [10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 8.0, 2.0]
    variance_history = [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 9.0, 9.0]

    assert sentrycam_alert_epoch(distance_history, variance_history) == 5


@pytest.mark.parametrize("distance_history,variance_history", [
    ([1.0, 2.0], [1.0]),   # mismatched lengths
    ([], []),              # empty
])
def test_sentrycam_alert_epoch_rejects_mismatched_or_empty_histories(distance_history, variance_history) -> None:
    with pytest.raises(ValueError):
        sentrycam_alert_epoch(distance_history, variance_history)


# --------------------------------------------------------------------------- cross-check with TrainResult
# Width: cluster (train_classifier + diagnostics.sentrycam_alert_epoch across their real seam).


def test_train_result_instability_alert_epoch_never_diverges_from_the_standalone_sentrycam_function() -> None:
    """The most important contract in this file: TrainResult must compute its
    instability_alert_epoch by calling sentrycam_alert_epoch on its own two histories, so this
    and any other caller (e.g. NB4) can never see a different answer for the same histories."""
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=20, patience=20))

    expected = sentrycam_alert_epoch(result.inter_cluster_distance_history, result.intra_cluster_variance_history)
    assert result.instability_alert_epoch == expected
