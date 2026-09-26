"""Small, deterministic metrics used by the research notebooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float


def _probability_inputs(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels)
    if p.ndim != 2 or y.ndim != 1 or p.shape[0] != y.size or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty (n, classes) array aligned with labels")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError("probability rows must sum to one for each example")
    if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("labels must be integer class indices within the probability columns")
    return p, y.astype(np.int64, copy=False)


def confidence(probabilities: np.ndarray) -> np.ndarray:
    """Return each row's maximum class probability."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2 or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be a finite non-empty matrix")
    if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("probabilities must lie in [0, 1] and sum to one")
    return np.max(p, axis=1)


def nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p, y = _probability_inputs(probabilities, labels)
    return float(-np.log(np.clip(p[np.arange(y.size), y], np.finfo(float).tiny, 1.0)).mean())


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score, averaged across examples and classes."""
    p, y = _probability_inputs(probabilities, labels)
    target = np.eye(p.shape[1], dtype=float)[y]
    return float(np.mean(np.sum((p - target) ** 2, axis=1)))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Top-label ECE with equal-width bins on [0, 1], weighted by bin mass."""
    p, y = _probability_inputs(probabilities, labels)
    if not isinstance(bins, (int, np.integer)) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    conf = np.max(p, axis=1)
    correct = np.argmax(p, axis=1) == y
    assignment = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for index in range(bins):
        selected = assignment == index
        if np.any(selected):
            total += float(selected.mean()) * abs(float(conf[selected].mean()) - float(correct[selected].mean()))
    return total


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    if conf.ndim != 1 or ok.ndim != 1 or conf.size == 0 or conf.shape != ok.shape:
        raise ValueError("confidence and correct must be non-empty aligned vectors")
    if not np.all(np.isfinite(conf)):
        raise ValueError("confidence must be finite")
    order = np.argsort(-conf, kind="stable")
    errors = (~ok[order]).astype(float)
    count = np.arange(1, conf.size + 1, dtype=float)
    return count / conf.size, np.cumsum(errors) / count


def selective_risk(confidence: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    if not np.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    if conf.ndim != 1 or ok.ndim != 1 or conf.size == 0 or conf.shape != ok.shape:
        raise ValueError("confidence and correct must be non-empty aligned vectors")
    if not np.all(np.isfinite(conf)):
        raise ValueError("confidence must be finite")
    retained = max(1, int(np.ceil(coverage * conf.size)))
    order = np.argsort(-conf, kind="stable")[:retained]
    return float(np.mean(~ok[order]))


def aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    x, y = np.asarray(coverage, dtype=float), np.asarray(risk, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size < 2 or x.shape != y.shape:
        raise ValueError("coverage and risk must be aligned vectors with at least two points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(np.diff(x) < 0):
        raise ValueError("coverage must be finite and ordered; risk must be finite")
    if np.any(x <= 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        raise ValueError("coverage and risk values must lie in (0, 1]")
    return float(np.trapezoid(y, x))


def bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int = 0,
    resamples: int = 2_000,
    confidence_level: float = 0.95,
) -> BootstrapInterval:
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("values must be a non-empty finite vector")
    if resamples <= 0 or not 0 < confidence_level < 1:
        raise ValueError("resamples must be positive and confidence_level must be in (0, 1)")
    point = float(statistic(sample))
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for i in range(resamples):
        draws[i] = statistic(sample[rng.integers(0, sample.size, size=sample.size)])
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return BootstrapInterval(point=point, lower=float(lower), upper=float(upper))
