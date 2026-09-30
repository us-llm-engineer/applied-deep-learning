"""Contract for the in-training probes (src.diagnostics) and train_classifier's probe_fn hook.

Grounding: SentryCam (arXiv:2405.15135) probes the penultimate layer and computes its two cluster
metrics in a low-dimensional projection, not in raw representation space; MM-PHATE
(arXiv:2406.01969) needs a (samples, sequence-step, unit) activation tensor per epoch, z-scored
per unit, with the expensive embedding done post-hoc.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.diagnostics import penultimate_and_sequence, project_2d, sentrycam_probe
from src.experiment import ExperimentConfig
from src.models import ConvGRU, LogMelCNN, TimePoolCNN, WideCNN
from src.train import train_classifier


def _probe_batch(n=24, n_mels=40, frames=101):
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(n, 1, n_mels, frames)).astype(np.float32))
    return x, (np.arange(n) % 12)


# ------------------------------------------------------------------ project_2d


def test_project_2d_returns_two_components_and_centres_them() -> None:
    rng = np.random.default_rng(1)
    acts = rng.normal(size=(50, 17))
    out = project_2d(acts)
    assert out.shape == (50, 2)
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-5), "PCA scores are centred by construction"


def test_project_2d_recovers_the_dominant_plane_of_a_rank_two_cloud() -> None:
    """Points living in a 2-plane embedded in 8D must survive the projection with distances intact."""
    rng = np.random.default_rng(2)
    plane = rng.normal(size=(60, 2)) * np.array([10.0, 5.0])
    basis = np.linalg.qr(rng.normal(size=(8, 2)))[0]
    out = project_2d(plane @ basis.T)
    original = np.linalg.norm(plane[:, None] - plane[None], axis=-1)
    projected = np.linalg.norm(out[:, None] - out[None], axis=-1)
    assert np.allclose(original, projected, atol=1e-3), "an isometric embedding must be undone exactly"


@pytest.mark.parametrize("bad", [np.zeros((0, 4)), np.zeros((5,)), np.zeros((5, 1))])
def test_project_2d_rejects_degenerate_inputs(bad) -> None:
    with pytest.raises(ValueError):
        project_2d(bad)


# ------------------------------------------------------------------ penultimate / sequence capture


def test_penultimate_capture_matches_the_classifier_input_width() -> None:
    x, _ = _probe_batch()
    for cls in (LogMelCNN, TimePoolCNN, WideCNN, ConvGRU):
        model = cls().eval()
        penult, _ = penultimate_and_sequence(model, x)
        assert penult.shape == (x.shape[0], model.classifier.in_features), cls.__name__


def test_globally_pooled_baseline_exposes_no_sequence_axis_but_the_new_models_do() -> None:
    """LogMelCNN's AdaptiveAvgPool2d((1,1)) destroys the time axis -- the structural point of the
    whole architecture comparison -- so MM-PHATE simply does not apply to it."""
    x, _ = _probe_batch()
    assert penultimate_and_sequence(LogMelCNN().eval(), x)[1] is None
    for cls in (TimePoolCNN, WideCNN, ConvGRU):
        seq = penultimate_and_sequence(cls().eval(), x)[1]
        assert seq is not None and seq.ndim == 3 and seq.shape[0] == x.shape[0], cls.__name__
        assert seq.shape[1] >= 2, f"{cls.__name__} must keep at least two sequence steps"


# ------------------------------------------------------------------ sentrycam_probe


def test_sentrycam_probe_records_2d_coords_labels_and_metrics_computed_in_2d() -> None:
    x, y = _probe_batch()
    rec = sentrycam_probe(ConvGRU().eval(), x, y)
    assert rec["coords_2d"].shape == (x.shape[0], 2)
    assert np.array_equal(rec["labels"], y)
    assert isinstance(rec["inter_cluster_distance_2d"], float)
    assert isinstance(rec["intra_cluster_variance_2d"], float)
    assert np.isfinite([rec["inter_cluster_distance_2d"], rec["intra_cluster_variance_2d"]]).all()


def test_sequence_activations_are_z_scored_per_unit() -> None:
    """MM-PHATE's tensor is z-scored; per-unit mean ~0 and std ~1 over (samples x steps)."""
    x, y = _probe_batch()
    seq = sentrycam_probe(ConvGRU().eval(), x, y)["sequence_activations"]
    flat = seq.reshape(-1, seq.shape[-1])
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-4)


def test_sequence_can_be_switched_off_for_cheap_probing() -> None:
    x, y = _probe_batch()
    assert "sequence_activations" not in sentrycam_probe(ConvGRU().eval(), x, y, keep_sequence=False)


# ------------------------------------------------------------------ train_classifier integration


def _tiny_training_set(n=48, n_mels=40, frames=24):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, 1, n_mels, frames)).astype(np.float32)
    y = (np.arange(n) % 12).astype(np.int64)
    return x, y


def test_train_classifier_collects_one_probe_record_per_epoch() -> None:
    train_x, train_y = _tiny_training_set()
    val_x, val_y = _tiny_training_set(n=24)
    cfg = ExperimentConfig(condition="clean", seed=17, max_epochs=3, patience=3,
                           microbatch_size=16, accumulation_steps=1)
    probe_x = torch.from_numpy(val_x)
    calls = []

    def probe_fn(model, epoch):
        calls.append(epoch)
        return sentrycam_probe(model, probe_x, val_y, keep_sequence=False)

    result = train_classifier(TimePoolCNN(n_mels=40), train_x, train_y, val_x, val_y, cfg,
                              probe_fn=probe_fn)
    assert calls == list(range(result.epochs_run))
    assert len(result.probe_history) == result.epochs_run
    assert all(r["coords_2d"].shape == (24, 2) for r in result.probe_history)


def test_probe_history_is_empty_when_no_probe_is_supplied() -> None:
    train_x, train_y = _tiny_training_set()
    val_x, val_y = _tiny_training_set(n=24)
    cfg = ExperimentConfig(condition="clean", seed=17, max_epochs=2, patience=2,
                           microbatch_size=16, accumulation_steps=1)
    result = train_classifier(TimePoolCNN(n_mels=40), train_x, train_y, val_x, val_y, cfg)
    assert result.probe_history == []
