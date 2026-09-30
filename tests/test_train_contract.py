from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.experiment import ExperimentConfig
from src.train import TrainResult, apply_condition, predict_logits, spec_mask, train_classifier


# --------------------------------------------------------------------------- builders


def _logmel(seed: int = 0, n_mels: int = 8, frames: int = 10) -> np.ndarray:
    return (np.random.default_rng(seed).normal(size=(n_mels, frames)) + 3.0).astype(np.float32)


def _waveforms(batch: int = 4, samples: int = 2_000) -> np.ndarray:
    t = np.arange(samples, dtype=np.float64) / 16_000.0
    rows = [0.9 * np.sin(2 * np.pi * (300.0 + 150.0 * i) * t) for i in range(batch)]
    return np.stack(rows).astype(np.float32)


def _separable(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """n two-feature points labelled by the sign of x0 + x1, with a margin around the boundary."""
    x = 2.0 * np.random.default_rng(seed).normal(size=(10 * n, 2)).astype(np.float32)
    x = x[np.abs(x.sum(axis=1)) > 1.0][:n]
    assert x.shape[0] == n
    return x, (x.sum(axis=1) > 0).astype(np.int64)


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))


def _config(**overrides) -> ExperimentConfig:
    settings = dict(condition="clean", seed=17, max_epochs=60, patience=10, microbatch_size=8, accumulation_steps=2)
    settings.update(overrides)
    return ExperimentConfig(**settings)


def _val_loss(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    logits = predict_logits(model, x)
    return float(F.cross_entropy(torch.from_numpy(np.asarray(logits, dtype=np.float64)), torch.from_numpy(y)))


# --------------------------------------------------------------------------- spec_mask


@pytest.mark.parametrize("freq_mask,time_mask", [(3, 0), (0, 4), (3, 4), (2, 5)])
def test_spec_mask_fills_exactly_the_requested_contiguous_rows_and_columns_with_the_array_mean(freq_mask, time_mask) -> None:
    """Given an (8, 10) log-mel, then exactly freq_mask contiguous rows and time_mask contiguous frames
    equal the ORIGINAL mean, and every cell outside those bands is untouched."""
    x = _logmel(1)
    fill = float(x.mean(dtype=np.float64))
    out = spec_mask(x, freq_mask=freq_mask, time_mask=time_mask, rng=np.random.default_rng(0))

    is_fill = np.isclose(out, fill, rtol=1e-5, atol=1e-6)
    fill_rows = is_fill.all(axis=1)
    fill_cols = is_fill.all(axis=0)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert int(fill_rows.sum()) == freq_mask
    assert int(fill_cols.sum()) == time_mask
    for flags in (fill_rows, fill_cols):
        idx = np.flatnonzero(flags)
        if idx.size:
            assert idx.tolist() == list(range(idx[0], idx[0] + idx.size)), "masked band must be contiguous"
    untouched = ~fill_rows[:, None] & ~fill_cols[None, :]
    assert np.array_equal(out[untouched], x[untouched])


def test_spec_mask_can_cover_the_whole_frequency_axis_or_the_whole_time_axis() -> None:
    x = _logmel(2)
    fill = float(x.mean(dtype=np.float64))
    all_rows = spec_mask(x, freq_mask=8, time_mask=0, rng=np.random.default_rng(0))
    all_cols = spec_mask(x, freq_mask=0, time_mask=10, rng=np.random.default_rng(0))
    assert np.allclose(all_rows, fill, rtol=1e-5)
    assert np.allclose(all_cols, fill, rtol=1e-5)


def test_spec_mask_is_deterministic_per_rng_seed_and_start_positions_cover_every_legal_offset() -> None:
    x = _logmel(3)
    first = spec_mask(x, freq_mask=3, time_mask=0, rng=np.random.default_rng(5))
    again = spec_mask(x, freq_mask=3, time_mask=0, rng=np.random.default_rng(5))
    assert np.array_equal(first, again)

    fill = float(x.mean(dtype=np.float64))
    starts = set()
    for seed in range(200):
        out = spec_mask(x, freq_mask=3, time_mask=0, rng=np.random.default_rng(seed))
        starts.add(int(np.flatnonzero(np.isclose(out, fill, rtol=1e-5).all(axis=1))[0]))
    assert starts == set(range(0, 8 - 3 + 1)), "a 3-row band in 8 rows has 6 legal offsets; each must be reachable"


@pytest.mark.parametrize("freq_mask,time_mask", [(3, 4), (0, 0), (8, 10)])
def test_spec_mask_never_mutates_its_input(freq_mask, time_mask) -> None:
    x = _logmel(4)
    before = x.copy()
    out = spec_mask(x, freq_mask=freq_mask, time_mask=time_mask, rng=np.random.default_rng(0))
    assert np.array_equal(x, before)
    assert not np.shares_memory(out, x)


def test_spec_mask_with_zero_widths_returns_an_equal_copy() -> None:
    x = _logmel(5)
    out = spec_mask(x, freq_mask=0, time_mask=0, rng=np.random.default_rng(0))
    assert np.array_equal(out, x)
    assert not np.shares_memory(out, x)


def test_spec_mask_rejects_a_frequency_mask_wider_than_n_mels_and_leaves_the_input_intact() -> None:
    x = _logmel(6)
    before = x.copy()
    with pytest.raises(ValueError):
        spec_mask(x, freq_mask=9, time_mask=0, rng=np.random.default_rng(0))
    assert np.array_equal(x, before)


def test_spec_mask_rejects_a_time_mask_longer_than_the_frame_count_and_leaves_the_input_intact() -> None:
    x = _logmel(7)
    before = x.copy()
    with pytest.raises(ValueError):
        spec_mask(x, freq_mask=0, time_mask=11, rng=np.random.default_rng(0))
    assert np.array_equal(x, before)


# --------------------------------------------------------------------------- apply_condition


def test_clean_condition_returns_an_equal_copy() -> None:
    w = _waveforms()
    before = w.copy()
    out = apply_condition(w, "clean", np.random.default_rng(0))
    assert np.array_equal(out, w) and out.dtype == w.dtype
    assert not np.shares_memory(out, w)
    assert np.array_equal(w, before)


def test_masking_condition_leaves_the_waveform_unchanged_because_masking_happens_in_feature_space() -> None:
    w = _waveforms()
    before = w.copy()
    out = apply_condition(w, "masking", np.random.default_rng(0))
    assert np.array_equal(out, before)
    assert np.array_equal(w, before)


def test_acoustic_condition_changes_every_example_keeps_shape_float32_finite_and_clips_to_unit_range() -> None:
    """Inputs peak at 0.9 and the perturbation is at most 25 dB SNR, so unclipped output would exceed 1.0
    for at least one example; the assertion on |peak| <= 1 therefore bites."""
    w = _waveforms()
    before = w.copy()
    out = apply_condition(w, "acoustic", np.random.default_rng(0))

    assert out.shape == w.shape and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 1.0
    assert all(not np.array_equal(out[i], w[i]) for i in range(w.shape[0]))
    assert np.array_equal(w, before), "the input batch must not be modified"


def test_acoustic_condition_perturbs_but_does_not_replace_the_signal() -> None:
    w = _waveforms()
    out = apply_condition(w, "acoustic", np.random.default_rng(0))
    for i in range(w.shape[0]):
        correlation = float(np.dot(out[i], w[i]) / (np.linalg.norm(out[i]) * np.linalg.norm(w[i])))
        assert correlation > 0.3, f"example {i} lost its signal (correlation {correlation:.3f})"


def test_acoustic_condition_is_deterministic_per_seed_and_varies_across_seeds() -> None:
    w = _waveforms()
    a = apply_condition(w, "acoustic", np.random.default_rng(3))
    b = apply_condition(w, "acoustic", np.random.default_rng(3))
    c = apply_condition(w, "acoustic", np.random.default_rng(4))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_acoustic_condition_draws_independent_perturbations_for_identical_examples() -> None:
    row = _waveforms(batch=1)[0]
    w = np.stack([row, row, row])
    out = apply_condition(w, "acoustic", np.random.default_rng(0))
    assert not np.array_equal(out[0], out[1])
    assert not np.array_equal(out[1], out[2])


@pytest.mark.parametrize("condition", ["unknown", "", "Clean"])
def test_unknown_condition_is_rejected_and_the_input_is_untouched(condition) -> None:
    w = _waveforms()
    before = w.copy()
    with pytest.raises(ValueError):
        apply_condition(w, condition, np.random.default_rng(0))
    assert np.array_equal(w, before)


# --------------------------------------------------------------------------- train_classifier


def test_training_reduces_validation_loss_and_reaches_ninety_percent_accuracy_on_a_separable_problem() -> None:
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    model = _model()
    untrained_loss = _val_loss(model, val_x, val_y)

    result = train_classifier(model, train_x, train_y, val_x, val_y, _config())

    assert isinstance(result, TrainResult)
    best = result.history[result.best_epoch]
    assert best["val_loss"] < result.history[0]["val_loss"], "best epoch must beat the first epoch"
    assert best["val_loss"] < untrained_loss
    assert best["val_accuracy"] >= 0.9
    logits = predict_logits(result.model, val_x)
    assert float((np.argmax(logits, axis=1) == val_y).mean()) >= 0.9


def test_history_records_one_row_per_epoch_with_finite_losses_and_best_epoch_is_the_argmin_of_val_loss() -> None:
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=12, patience=12))

    assert len(result.history) == result.epochs_run == 12
    assert [row["epoch"] for row in result.history] == list(range(12))
    for row in result.history:
        assert {"train_loss", "val_loss", "val_accuracy"} <= set(row)
        assert np.isfinite([row["train_loss"], row["val_loss"]]).all()
        assert 0.0 <= row["val_accuracy"] <= 1.0
    losses = [row["val_loss"] for row in result.history]
    assert result.best_epoch == int(np.argmin(losses))
    assert isinstance(result.wall_seconds, float) and result.wall_seconds >= 0.0


def test_restored_model_is_the_best_epoch_not_the_last_epoch() -> None:
    """Train labels are all 0 and validation labels all 1, so validation loss rises as training proceeds.
    The returned model must then score exactly the best epoch's validation loss, not the (worse) last one."""
    train_x, _ = _separable(64, 1)
    val_x, _ = _separable(32, 2)
    train_y = np.zeros(64, dtype=np.int64)
    val_y = np.ones(32, dtype=np.int64)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=30, patience=4))

    best_loss = result.history[result.best_epoch]["val_loss"]
    assert result.history[-1]["val_loss"] > best_loss, "scenario must make the last epoch strictly worse"
    assert _val_loss(result.model, val_x, val_y) == pytest.approx(best_loss, abs=1e-5)


@pytest.mark.parametrize("patience", [1, 3])
def test_early_stopping_ends_exactly_patience_epochs_after_the_best_epoch(patience) -> None:
    """Validation loss cannot improve after epoch 0 (train pushes toward class 0, validation is all class 1)."""
    train_x, _ = _separable(64, 1)
    val_x, _ = _separable(32, 2)
    train_y = np.zeros(64, dtype=np.int64)
    val_y = np.ones(32, dtype=np.int64)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=100, patience=patience))

    assert result.epochs_run == result.best_epoch + 1 + patience
    assert len(result.history) == result.epochs_run
    assert result.epochs_run < 100


def test_training_never_exceeds_max_epochs() -> None:
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=2, patience=10))
    assert result.epochs_run == 2 and len(result.history) == 2


def test_same_config_and_initial_weights_give_identical_logits_history_and_counters() -> None:
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    a = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=8))
    b = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=8))

    assert torch.equal(torch.from_numpy(np.asarray(predict_logits(a.model, val_x))),
                       torch.from_numpy(np.asarray(predict_logits(b.model, val_x))))
    assert a.history == b.history
    assert (a.best_epoch, a.epochs_run, a.optimizer_steps) == (b.best_epoch, b.epochs_run, b.optimizer_steps)


def test_different_config_seeds_give_different_trained_weights_from_identical_initial_weights() -> None:
    """The seed must drive the epoch shuffling (initial weights are held equal here)."""
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    a = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(seed=1, max_epochs=3, patience=3))
    b = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(seed=2, max_epochs=3, patience=3))
    assert not np.array_equal(predict_logits(a.model, val_x), predict_logits(b.model, val_x))


@pytest.mark.parametrize("accumulation_steps", [1, 2, 4])
def test_optimizer_steps_equal_epochs_times_microbatches_per_epoch_over_accumulation(accumulation_steps) -> None:
    """64 samples / microbatch 8 = 8 microbatches per epoch; 3 epochs; one step per `accumulation_steps` microbatches."""
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    cfg = _config(max_epochs=3, patience=3, accumulation_steps=accumulation_steps)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, cfg)
    assert result.epochs_run == 3
    assert result.optimizer_steps == 3 * 8 // accumulation_steps


def test_all_three_conditions_receive_the_same_optimizer_budget() -> None:
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    steps = {}
    for condition in ("clean", "masking", "acoustic"):
        cfg = _config(condition=condition, max_epochs=4, patience=4)
        result = train_classifier(_model(), train_x, train_y, val_x, val_y, cfg)
        assert result.epochs_run == 4
        steps[condition] = result.optimizer_steps
    assert steps == {"clean": 16, "masking": 16, "acoustic": 16}


def test_accumulating_two_microbatches_of_eight_matches_one_batch_of_sixteen() -> None:
    """Gradient accumulation must be equivalent to the larger batch it stands in for."""
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    accumulated = train_classifier(
        _model(), train_x, train_y, val_x, val_y, _config(max_epochs=5, patience=5, microbatch_size=8, accumulation_steps=2))
    single = train_classifier(
        _model(), train_x, train_y, val_x, val_y, _config(max_epochs=5, patience=5, microbatch_size=16, accumulation_steps=1))

    assert accumulated.optimizer_steps == single.optimizer_steps == 20
    assert np.allclose(predict_logits(accumulated.model, val_x), predict_logits(single.model, val_x), atol=1e-4)


def test_the_model_passed_in_is_not_mutated_and_the_result_is_a_trained_copy() -> None:
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    model = _model()
    before = copy.deepcopy(model.state_dict())

    result = train_classifier(model, train_x, train_y, val_x, val_y, _config(max_epochs=5, patience=5))

    assert result.model is not model
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items()), "input model must be untouched"
    assert any(not torch.equal(before[k], v) for k, v in result.model.state_dict().items()), "result must be trained"


def test_augment_fn_output_replaces_the_training_inputs() -> None:
    """The model must see augment_fn's output, not the raw batch, during gradient steps.

    Oracle: a spy wrapper records the absolute sum of every input batch seen in training mode.
    With an erasing augmentation every training batch is all zeros; without it none is.
    """
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)

    class Spy(nn.Module):
        """Records into a shared list: train_classifier trains a deep copy of the model."""

        def __init__(self, inner: nn.Module, sink: list[float]) -> None:
            super().__init__()
            self.inner = inner
            self.sink = sink

        def __deepcopy__(self, memo):
            import copy

            return Spy(copy.deepcopy(self.inner, memo), self.sink)  # deep copy keeps the shared sink

        def forward(self, x):
            if self.training:
                self.sink.append(float(x.abs().sum()))
            return self.inner(x)

    def erase(batch, *args, **kwargs):
        return batch * 0

    erased_sums: list[float] = []
    normal_sums: list[float] = []
    train_classifier(Spy(_model(), erased_sums), train_x, train_y, val_x, val_y, _config(), augment_fn=erase)
    train_classifier(Spy(_model(), normal_sums), train_x, train_y, val_x, val_y, _config())

    assert erased_sums and all(total == 0.0 for total in erased_sums)
    assert normal_sums and all(total > 0.0 for total in normal_sums)


# --------------------------------------------------------------------------- AutoClip (arXiv:2007.14469)


def test_autoclip_histories_have_exactly_one_entry_per_real_optimizer_step() -> None:
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=5, patience=5))
    assert len(result.grad_norm_history) == result.optimizer_steps
    assert len(result.clip_threshold_history) == result.optimizer_steps


def test_first_autoclip_threshold_is_unbounded_because_the_history_starts_empty() -> None:
    """AutoClip's own cumulative-history formulation starts from an empty list, so the very
    first real optimizer step has nothing to compute a percentile over and must not clip."""
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=5, patience=5))
    assert result.optimizer_steps >= 1
    assert result.clip_threshold_history[0] == float("inf")


def test_autoclip_thresholds_after_the_first_step_equal_the_percentile_of_the_prior_grad_norm_history() -> None:
    """Every threshold at index i>0 must equal np.percentile of the returned history's own
    prefix [:i] at config.autoclip_percentile -- computed here with the same call the
    implementation must make, not hand-derived."""
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    cfg = _config(max_epochs=5, patience=5, autoclip_percentile=37.0)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, cfg)
    assert result.optimizer_steps > 1, "need at least one post-first step to exercise the percentile branch"
    for i in range(1, result.optimizer_steps):
        expected = np.percentile(result.grad_norm_history[:i], cfg.autoclip_percentile)
        assert result.clip_threshold_history[i] == pytest.approx(expected), f"index {i}"


def test_autoclip_grad_norm_history_values_are_finite_and_non_negative() -> None:
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=5, patience=5))
    assert np.isfinite(result.grad_norm_history).all()
    assert (np.asarray(result.grad_norm_history) >= 0).all()


def test_changing_autoclip_percentile_changes_thresholds_but_not_the_number_of_optimizer_steps() -> None:
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    low = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=6, patience=6, autoclip_percentile=1.0))
    high = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=6, patience=6, autoclip_percentile=50.0))

    assert len(low.grad_norm_history) == len(high.grad_norm_history) == low.optimizer_steps == high.optimizer_steps
    assert low.clip_threshold_history != high.clip_threshold_history, (
        "a 1st vs 50th percentile clip threshold must diverge once at least two real steps have run"
    )


def test_training_with_autoclip_still_reaches_high_accuracy_on_a_separable_problem() -> None:
    """Relaxed re-run of the pre-AutoClip 90% convergence check: the default percentile=10.0
    must not break real learning on this easy toy problem, and predictions must not collapse
    to a single always-predicted class."""
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, _config())

    logits = predict_logits(result.model, val_x)
    predictions = np.argmax(logits, axis=1)
    accuracy = float((predictions == val_y).mean())
    assert accuracy >= 0.85
    assert len(set(predictions.tolist())) > 1, "must not degenerate to always predicting one class"


# --------------------------------------------------------------------------- SentryCam-inspired cluster drift (arXiv:2405.15135)
# (cluster_drift_metrics and sentrycam_alert_epoch themselves are tested directly, with hand
# computed numbers, in test_sentrycam_diagnostics_contract.py; these tests only check that
# train_classifier wires per-epoch tracking into TrainResult's own new fields correctly.)


def test_cluster_drift_histories_have_one_entry_per_epoch_and_are_finite() -> None:
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=6, patience=6))

    assert len(result.inter_cluster_distance_history) == result.epochs_run
    assert len(result.intra_cluster_variance_history) == result.epochs_run
    assert np.isfinite(result.inter_cluster_distance_history).all()
    assert np.isfinite(result.intra_cluster_variance_history).all()


def test_instability_alert_epoch_is_none_or_a_valid_epoch_index() -> None:
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    result = train_classifier(_model(), train_x, train_y, val_x, val_y, _config(max_epochs=20, patience=20))

    assert result.instability_alert_epoch is None or (
        isinstance(result.instability_alert_epoch, int) and 0 <= result.instability_alert_epoch < result.epochs_run
    )


# --------------------------------------------------------------------------- predict_logits


class _Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        self.grad_enabled: list[bool] = []
        self.inner_training: list[bool] = []

    def forward(self, x):
        self.grad_enabled.append(torch.is_grad_enabled())
        self.inner_training.append(self.inner.training)
        return self.inner(x)


def _probe() -> _Probe:
    torch.manual_seed(0)
    probe = _Probe()
    bn = probe.inner[1]
    bn.running_mean.copy_(torch.tensor([0.5, -0.5, 1.0, 0.0]))
    bn.running_var.copy_(torch.tensor([2.0, 0.5, 1.5, 3.0]))
    return probe


def test_predict_logits_returns_rows_in_input_order_whatever_the_batch_size() -> None:
    x = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    probe = _probe().train()  # train mode would make BatchNorm outputs depend on the batch
    with torch.no_grad():
        expected = probe.eval()(torch.from_numpy(x)).numpy()
    probe.train()

    for batch_size in (1, 3, 7, 256):
        logits = predict_logits(probe, x, batch_size=batch_size)
        assert logits.shape == (10, 2) and logits.dtype in (np.float32, np.float64)
        assert np.allclose(logits, expected, atol=1e-5), f"batch_size={batch_size}"


@pytest.mark.parametrize("start_in_training_mode", [True, False])
def test_predict_logits_runs_in_eval_without_grad_and_restores_the_original_mode(start_in_training_mode) -> None:
    probe = _probe().train(start_in_training_mode)
    running_before = probe.inner[1].running_mean.clone()
    x = np.random.default_rng(1).normal(size=(5, 3)).astype(np.float32)

    predict_logits(probe, x, batch_size=2)

    assert probe.grad_enabled == [False] * 3  # ceil(5 / 2) batches
    assert probe.inner_training == [False] * 3
    assert probe.training is start_in_training_mode
    assert torch.equal(probe.inner[1].running_mean, running_before), "prediction must not update BatchNorm statistics"


def test_predict_logits_applies_the_feature_function_before_the_model() -> None:
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    x = np.random.default_rng(2).normal(size=(6, 3)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x) * 2).numpy()
    logits = predict_logits(model, x, feature_fn=lambda t: t * 2)
    assert np.allclose(logits, expected, atol=1e-6)


# --------------------------------------------------------------------------- CUDA device placement


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_predict_logits_on_a_cuda_model_returns_a_plain_finite_cpu_array() -> None:
    """A model explicitly moved to cuda, given plain numpy input, must not require the
    caller to place any tensor: predict_logits infers the device from the model itself."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2)).to("cuda")
    x = np.random.default_rng(0).normal(size=(9, 3)).astype(np.float32)

    logits = predict_logits(model, x)

    assert type(logits) is np.ndarray
    assert logits.shape == (9, 2)
    assert logits.dtype in (np.float32, np.float64)
    assert np.isfinite(logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_train_classifier_on_a_cuda_model_produces_python_float_losses_keeps_the_model_on_cuda_and_still_learns() -> None:
    """Given a model moved to cuda, when trained for a few epochs, then every history row's
    train_loss/val_loss is a genuine Python float (not a 0-dim CUDA tensor -- the line-302 bug
    this test targets raises on CUDA even though it silently "works" on CPU), the returned
    model's parameters are still on cuda, and validation accuracy improves over the untrained
    baseline on this easily separable problem."""
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    model = _model().to("cuda")
    untrained_loss = _val_loss(model, val_x, val_y)
    untrained_logits = predict_logits(model, val_x)
    untrained_accuracy = float((np.argmax(untrained_logits, axis=1) == val_y).mean())

    result = train_classifier(model, train_x, train_y, val_x, val_y, _config(max_epochs=5, patience=5))

    assert isinstance(result, TrainResult)
    for row in result.history:
        assert type(row["train_loss"]) is float, row
        assert type(row["val_loss"]) is float, row
    assert next(result.model.parameters()).device.type == "cuda"
    best = result.history[result.best_epoch]
    assert best["val_loss"] < untrained_loss
    assert best["val_accuracy"] > untrained_accuracy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_train_classifier_on_cuda_with_an_augment_fn_exercises_the_cpu_numpy_round_trip_without_error() -> None:
    """augment_fn forces train_classifier to detach a training batch to numpy and rebuild a
    tensor from it (the line 252/254 round-trip); on a CUDA model this must move the
    rebuilt tensor back to cuda, and history rows must still be genuine Python floats."""
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    model = _model().to("cuda")

    def add_fixed_offset(batch, _condition, _rng):
        return batch + 0.01

    result = train_classifier(
        model, train_x, train_y, val_x, val_y, _config(max_epochs=3, patience=3), augment_fn=add_fixed_offset,
    )

    assert isinstance(result, TrainResult)
    assert len(result.history) == result.epochs_run
    for row in result.history:
        assert type(row["train_loss"]) is float, row
        assert type(row["val_loss"]) is float, row
    assert next(result.model.parameters()).device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cluster_drift_histories_on_cuda_are_genuine_python_floats() -> None:
    """Regression guard on top of "doesn't crash": cluster_drift_metrics already returns
    Python floats on CPU; the .cpu() fix needed to call it on CUDA logits/labels must
    preserve that float contract, not just avoid a device-mismatch exception."""
    train_x, train_y = _separable(64, 1)
    val_x, val_y = _separable(32, 2)
    model = _model().to("cuda")

    result = train_classifier(model, train_x, train_y, val_x, val_y, _config(max_epochs=5, patience=5))

    assert len(result.inter_cluster_distance_history) == result.epochs_run
    assert len(result.intra_cluster_variance_history) == result.epochs_run
    for value in result.inter_cluster_distance_history:
        assert type(value) is float, value
    for value in result.intra_cluster_variance_history:
        assert type(value) is float, value
