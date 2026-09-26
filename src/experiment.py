"""Training configuration and prediction-only paired evaluation helpers.

This module deliberately does not train models. Heavy feature extraction and
optimization belong in the explicitly offloaded notebook cells; predictions
can then be summarized locally and reproducibly here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .metrics import BootstrapInterval, aurc, bootstrap_interval, brier_score, expected_calibration_error, nll, risk_coverage_curve


TRAINING_CONDITIONS = ("clean", "masking", "acoustic")
_ECE_BINS = (10, 15, 20)


@dataclass(frozen=True)
class ExperimentConfig:
    """Shared optimization settings for one member of the training matrix."""

    condition: str = "clean"
    seed: int = 17
    max_epochs: int = 6
    patience: int = 2
    microbatch_size: int = 64
    accumulation_steps: int = 2

    def __post_init__(self) -> None:
        if self.condition not in TRAINING_CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}; condition must be one of {TRAINING_CONDITIONS}")
        for name in ("max_epochs", "patience", "microbatch_size", "accumulation_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")

    @property
    def optimization_budget(self) -> tuple[int, int, int, int]:
        """Budget-defining settings, equal across augmentation conditions."""
        return (self.seed, self.max_epochs, self.microbatch_size, self.accumulation_steps)


@dataclass(frozen=True)
class PredictionMetrics:
    accuracy: float
    macro_f1: float
    per_class_recall: np.ndarray
    accuracy_interval: BootstrapInterval
    macro_f1_interval: BootstrapInterval
    nll: float
    brier_score: float
    ece_by_bins: dict[int, float]
    risk_coverage: tuple[np.ndarray, np.ndarray]
    aurc: float


@dataclass(frozen=True)
class PairedEvaluation:
    identifiers: tuple[str, ...]
    clean: PredictionMetrics
    shifted: PredictionMetrics
    accuracy_change: float
    macro_f1_change: float


@dataclass(frozen=True)
class ResourceRecord:
    condition: str
    device: str
    trainable_parameters: int
    wall_seconds: float
    peak_memory_bytes: int


def _macro_f1(predicted: np.ndarray, labels: np.ndarray, class_count: int) -> float:
    scores: list[float] = []
    for label in range(class_count):
        true_positive = int(np.sum((predicted == label) & (labels == label)))
        false_positive = int(np.sum((predicted == label) & (labels != label)))
        false_negative = int(np.sum((predicted != label) & (labels == label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


def _prediction_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> PredictionMetrics:
    prediction = np.argmax(probabilities, axis=1)
    correct = prediction == labels
    accuracy = float(np.mean(correct))
    f1 = _macro_f1(prediction, labels, probabilities.shape[1])

    def f1_statistic(sample_labels: np.ndarray, sample_prediction: np.ndarray) -> float:
        return _macro_f1(sample_prediction, sample_labels, probabilities.shape[1])

    rng = np.random.default_rng(bootstrap_seed)
    sampled_indices = rng.integers(0, labels.size, size=(bootstrap_resamples, labels.size))
    f1_draws = np.fromiter(
        (f1_statistic(labels[index], prediction[index]) for index in sampled_indices),
        dtype=float,
        count=bootstrap_resamples,
    )
    tail = 0.025
    f1_interval = BootstrapInterval(
        point=f1,
        lower=float(np.quantile(f1_draws, tail)),
        upper=float(np.quantile(f1_draws, 1.0 - tail)),
    )
    accuracy_interval = bootstrap_interval(
        correct.astype(float), seed=bootstrap_seed, resamples=bootstrap_resamples
    )
    coverage, risk = risk_coverage_curve(np.max(probabilities, axis=1), correct)
    recalls = np.zeros(probabilities.shape[1], dtype=float)
    present = np.bincount(labels, minlength=probabilities.shape[1]) > 0
    for label in np.flatnonzero(present):
        recalls[label] = float(np.mean(prediction[labels == label] == label))
    return PredictionMetrics(
        accuracy=accuracy,
        macro_f1=f1,
        per_class_recall=recalls,
        accuracy_interval=accuracy_interval,
        macro_f1_interval=f1_interval,
        nll=nll(probabilities, labels),
        brier_score=brier_score(probabilities, labels),
        ece_by_bins={bins: expected_calibration_error(probabilities, labels, bins=bins) for bins in _ECE_BINS},
        risk_coverage=(coverage, risk),
        aurc=aurc(coverage, risk) if coverage.size > 1 else 0.0,
    )


def evaluate_predictions(
    identifiers: Sequence[str],
    labels: np.ndarray,
    clean_probabilities: np.ndarray,
    shifted_probabilities: np.ndarray,
    *,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 2_000,
) -> PairedEvaluation:
    """Summarize aligned clean and shifted probabilities without model fitting."""
    ids = tuple(identifiers)
    y = np.asarray(labels)
    clean = np.asarray(clean_probabilities, dtype=float)
    shifted = np.asarray(shifted_probabilities, dtype=float)
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("identifiers must be unique, non-empty strings")
    if y.ndim != 1 or len(ids) != y.size or clean.ndim != 2 or shifted.ndim != 2:
        raise ValueError("identifiers, labels, and probability rows must be aligned")
    if clean.shape != shifted.shape or clean.shape[0] != y.size:
        raise ValueError("clean and shifted predictions must have aligned rows and classes")
    if not isinstance(bootstrap_resamples, int) or isinstance(bootstrap_resamples, bool) or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        raise ValueError("bootstrap_seed must be an integer")
    # Delegate probability and label validation to the shared metrics implementation.
    clean_summary = _prediction_metrics(
        clean, y, bootstrap_seed=bootstrap_seed, bootstrap_resamples=bootstrap_resamples
    )
    shifted_summary = _prediction_metrics(
        shifted, y, bootstrap_seed=bootstrap_seed + 1, bootstrap_resamples=bootstrap_resamples
    )
    return PairedEvaluation(
        identifiers=ids,
        clean=clean_summary,
        shifted=shifted_summary,
        accuracy_change=shifted_summary.accuracy - clean_summary.accuracy,
        macro_f1_change=shifted_summary.macro_f1 - clean_summary.macro_f1,
    )


def record_resources(
    condition: str,
    device: str,
    trainable_parameters: int,
    wall_seconds: float,
    peak_memory_bytes: int,
) -> ResourceRecord:
    """Create a validated record of measurements collected by the caller."""
    if not isinstance(condition, str) or not condition:
        raise ValueError("condition must be a non-empty string")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if not isinstance(trainable_parameters, int) or isinstance(trainable_parameters, bool) or trainable_parameters < 0:
        raise ValueError("trainable_parameters must be non-negative")
    if not isinstance(peak_memory_bytes, int) or isinstance(peak_memory_bytes, bool) or peak_memory_bytes < 0:
        raise ValueError("peak_memory_bytes must be non-negative")
    if not isinstance(wall_seconds, (int, float)) or isinstance(wall_seconds, bool) or not math.isfinite(wall_seconds) or wall_seconds < 0:
        raise ValueError("wall_seconds must be finite and non-negative")
    return ResourceRecord(
        condition=condition,
        device=device,
        trainable_parameters=trainable_parameters,
        wall_seconds=float(wall_seconds),
        peak_memory_bytes=peak_memory_bytes,
    )
