from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import (
    aurc,
    bootstrap_interval,
    brier_score,
    expected_calibration_error,
    nll,
    risk_coverage_curve,
    selective_risk,
)


def test_probability_metrics_match_hand_calculated_truth() -> None:
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int64)

    assert nll(probabilities, labels) == pytest.approx(-(math.log(0.8) + math.log(0.9)) / 2)
    assert brier_score(probabilities, labels) == pytest.approx(0.05)
    assert expected_calibration_error(np.eye(2), labels, bins=10) == pytest.approx(0.0)


def test_ece_uses_confidence_weighted_bin_error() -> None:
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4]], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int64)

    assert expected_calibration_error(probabilities, labels, bins=10) == pytest.approx(0.35)


def test_selective_risk_curve_orders_high_confidence_examples_first() -> None:
    confidence = np.array([0.99, 0.80, 0.70, 0.50])
    correct = np.array([True, False, True, False])
    coverage, risk = risk_coverage_curve(confidence, correct)

    assert np.allclose(coverage, [0.25, 0.50, 0.75, 1.00])
    assert np.allclose(risk, [0.0, 0.5, 1 / 3, 0.5])
    assert selective_risk(confidence, correct, coverage=0.5) == pytest.approx(0.5)
    assert aurc(coverage, risk) == pytest.approx(np.trapezoid(risk, coverage))


def test_metric_inputs_fail_loudly_for_non_probabilities_and_empty_selection() -> None:
    with pytest.raises(ValueError, match="probability"):
        nll(np.array([[0.8, 0.8]]), np.array([0]))
    with pytest.raises(ValueError, match="coverage"):
        selective_risk(np.array([0.9]), np.array([True]), coverage=0.0)


def test_bootstrap_interval_is_repeatable_and_contains_sample_estimate() -> None:
    values = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    first = bootstrap_interval(values, statistic=np.mean, seed=11, resamples=2_000)
    second = bootstrap_interval(values, statistic=np.mean, seed=11, resamples=2_000)

    assert first == second
    assert first.lower <= first.point <= first.upper
    assert first.point == pytest.approx(float(values.mean()))
