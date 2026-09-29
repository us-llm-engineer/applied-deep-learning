"""Post-hoc calibration and prediction-set utilities."""
from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a positive scalar temperature by minimizing calibration NLL."""
    values = np.asarray(logits, dtype=float)
    y = np.asarray(labels)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2 or y.shape != (values.shape[0],):
        raise ValueError("logits must be a non-empty matrix aligned with labels")
    if not np.all(np.isfinite(values)) or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= values.shape[1]):
        raise ValueError("logits must be finite and labels valid integer class indices")

    def objective(log_temperature: float) -> float:
        probs = _softmax(values / np.exp(log_temperature))
        return float(-np.log(np.clip(probs[np.arange(y.size), y], np.finfo(float).tiny, 1.0)).mean())

    # Golden-section search over log(T) keeps T positive without optional dependencies.
    left, right = -5.0, 5.0
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(120):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = objective(x2)
    candidates = [1.0, float(np.exp((left + right) / 2.0))]
    return min(candidates, key=lambda temperature: objective(float(np.log(temperature))))


def prediction_sets(logits: np.ndarray, temperature: float, threshold: float) -> list[set[int]]:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("logits must be a finite non-empty matrix")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    probabilities = _softmax(values / temperature)
    return [set(np.flatnonzero(row >= threshold).astype(int).tolist()) for row in probabilities]


def crc_threshold(true_label_scores: np.ndarray, alpha: float) -> float:
    """Conservative order statistic for indicator miscoverage on calibration scores."""
    scores = np.asarray(true_label_scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("true_label_scores must be a non-empty finite vector in [0, 1]")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    # The CRC correction bounds the allowable empirical miscoverage count.
    allowed = int(np.floor(alpha * (scores.size + 1) - 1.0 + 1e-12))
    if allowed < 0:
        return 0.0
    index = min(max(allowed, 0), scores.size - 1)
    return float(np.sort(scores)[index])
