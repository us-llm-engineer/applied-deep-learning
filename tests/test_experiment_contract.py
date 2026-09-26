from __future__ import annotations

import numpy as np
import pytest
import torch

from src.experiment import ExperimentConfig, evaluate_predictions, record_resources
from src.models import LogMelCNN, log_mel


def test_log_mel_and_cnn_evaluation_are_deterministic_on_tiny_waveforms() -> None:
    torch.manual_seed(17)
    waveforms = torch.linspace(-0.25, 0.25, 640).reshape(2, 320)
    first = log_mel(waveforms, sample_rate=16_000, n_mels=16, hop_length=80)
    second = log_mel(waveforms, sample_rate=16_000, n_mels=16, hop_length=80)
    model = LogMelCNN(n_mels=16, num_classes=3).eval()

    with torch.no_grad():
        logits_a = model(first)
        logits_b = model(second)
    assert first.shape == (2, 1, 16, 5)
    assert torch.isfinite(first).all()
    assert logits_a.shape == (2, 3)
    assert torch.allclose(first, second)
    assert torch.allclose(logits_a, logits_b)


def test_config_conditions_share_budget_and_reject_unknown_condition() -> None:
    clean = ExperimentConfig(condition="clean", seed=17, max_epochs=6, patience=2, microbatch_size=64, accumulation_steps=2)
    masking = ExperimentConfig(condition="masking", seed=17, max_epochs=6, patience=2, microbatch_size=64, accumulation_steps=2)
    acoustic = ExperimentConfig(condition="acoustic", seed=17, max_epochs=6, patience=2, microbatch_size=64, accumulation_steps=2)

    assert clean.optimization_budget == masking.optimization_budget == acoustic.optimization_budget
    with pytest.raises(ValueError, match="condition"):
        ExperimentConfig(condition="unknown")


def test_paired_prediction_summary_keeps_identifiers_and_reports_known_metric_change() -> None:
    identifiers = ("a", "b", "c", "d")
    labels = np.array([0, 1, 2, 0])
    clean = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.8, .1, .1]])
    shifted = np.array([[.8, .1, .1], [.2, .7, .1], [.5, .4, .1], [.2, .2, .6]])
    summary = evaluate_predictions(identifiers, labels, clean, shifted, bootstrap_seed=17, bootstrap_resamples=500)

    assert summary.identifiers == identifiers
    assert summary.clean.accuracy == pytest.approx(1.0)
    assert summary.shifted.accuracy == pytest.approx(0.5)
    assert summary.shifted.macro_f1 == pytest.approx(0.5)
    assert np.allclose(summary.shifted.per_class_recall, [0.5, 1.0, 0.0])
    assert summary.clean.accuracy_interval.point == pytest.approx(1.0)
    assert summary.shifted.accuracy_interval.point == pytest.approx(0.5)
    assert summary.clean.ece_by_bins.keys() == {10, 15, 20}


def test_paired_prediction_summary_rejects_misaligned_rows_and_resource_record_preserves_measurement() -> None:
    probabilities = np.array([[.9, .1], [.1, .9]])
    with pytest.raises(ValueError, match="identifier|aligned"):
        evaluate_predictions(("a",), np.array([0, 1]), probabilities, probabilities)

    record = record_resources(
        condition="clean", device="cuda", trainable_parameters=123, wall_seconds=4.5, peak_memory_bytes=4096
    )
    assert record.condition == "clean"
    assert record.device == "cuda"
    assert record.trainable_parameters == 123
    assert record.wall_seconds == pytest.approx(4.5)
    assert record.peak_memory_bytes == 4096
    with pytest.raises(ValueError, match="non-negative|positive"):
        record_resources("clean", "cpu", trainable_parameters=-1, wall_seconds=1.0, peak_memory_bytes=0)
