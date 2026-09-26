from __future__ import annotations

import dataclasses
import time

import pytest
import torch
from torch import nn

from src.latency import LatencyRecord, count_parameters, measure_batch1_latency, model_size_bytes
from src.models import LogMelCNN


class _Recorder(nn.Module):
    """Wraps a module and records what every forward call saw."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.batch_sizes: list[int] = []
        self.shapes: list[tuple[int, ...]] = []
        self.inner_training: list[bool] = []
        self.grad_enabled: list[bool] = []
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(x.shape[0]))
        self.shapes.append(tuple(x.shape))
        self.inner_training.append(self.inner.training)
        self.grad_enabled.append(torch.is_grad_enabled())
        self.inputs.append(x.detach().clone())
        return self.inner(x)


class _ScriptedClock(nn.Module):
    """A module whose forward advances a fake clock by a scripted duration (seconds)."""

    def __init__(self, durations_s: list[float]) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))  # gives the model a device
        self.durations = list(durations_s)
        self.now = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.now += self.durations.pop(0)
        return x


def _linear_recorder() -> _Recorder:
    torch.manual_seed(0)
    return _Recorder(nn.Linear(4, 3))


# --------------------------------------------------------------------------- measure_batch1_latency


def test_latency_runs_exactly_warmup_plus_repeats_forwards_each_with_batch_one_in_eval_without_grad() -> None:
    model = _linear_recorder()
    model.train()
    measure_batch1_latency(model, (4,), warmup=3, repeats=7, seed=0)

    assert len(model.batch_sizes) == 3 + 7
    assert model.shapes == [(1, 4)] * 10
    assert model.inner_training == [False] * 10
    assert model.grad_enabled == [False] * 10


def test_latency_with_zero_warmup_runs_only_the_repeats() -> None:
    model = _linear_recorder()
    measure_batch1_latency(model, (4,), warmup=0, repeats=5, seed=0)
    assert len(model.batch_sizes) == 5


def test_latency_reuses_one_seeded_input_and_the_seed_selects_it() -> None:
    first, again, other = _linear_recorder(), _linear_recorder(), _linear_recorder()
    measure_batch1_latency(first, (4,), warmup=1, repeats=3, seed=5)
    measure_batch1_latency(again, (4,), warmup=1, repeats=3, seed=5)
    measure_batch1_latency(other, (4,), warmup=1, repeats=3, seed=6)

    assert all(torch.equal(x, first.inputs[0]) for x in first.inputs), "all forwards must see the same ONE input"
    assert torch.equal(first.inputs[0], again.inputs[0]), "same seed -> same input"
    assert not torch.equal(first.inputs[0], other.inputs[0]), "different seed -> different input"


@pytest.mark.parametrize("start_in_training_mode", [True, False])
def test_latency_restores_the_models_original_train_or_eval_mode(start_in_training_mode: bool) -> None:
    torch.manual_seed(0)
    model = LogMelCNN(n_mels=8, num_classes=3, width=2)
    model.train(start_in_training_mode)
    measure_batch1_latency(model, (1, 8, 10), warmup=1, repeats=2, seed=0)
    assert model.training is start_in_training_mode
    assert all(m.training is start_in_training_mode for m in model.modules())


def test_latency_fields_are_positive_ordered_and_describe_the_run() -> None:
    torch.manual_seed(0)
    record = measure_batch1_latency(nn.Linear(4, 3), (4,), warmup=1, repeats=6, seed=0)

    assert isinstance(record, LatencyRecord)
    assert record.median_ms > 0 and record.mean_ms > 0
    assert record.p95_ms >= record.median_ms
    assert record.repeats == 6
    assert record.num_threads == torch.get_num_threads()
    assert record.device == "cpu"


def test_latency_summarises_only_the_timed_repeats_with_exact_median_p95_and_mean(monkeypatch) -> None:
    """Given scripted per-forward durations on a fake perf_counter, when latency is measured,
    then warm-up forwards are excluded and median/p95/mean (in ms) are the exact statistics.

    Timed durations (ms): 1..18 plus two 30s, in a scrambled order -> sorted[9:11] = 10, 11 (median 10.5),
    sorted[18:] = 30, 30 (p95 = 30 under every common percentile definition), mean = 231/20 = 11.55.
    Warm-up forwards take a full second each so including them would be unmistakable.
    """
    timed_ms = [7, 30, 2, 18, 11, 1, 15, 30, 4, 9, 17, 3, 12, 6, 16, 5, 13, 8, 14, 10]
    model = _ScriptedClock([1.0] * 3 + [ms / 1000.0 for ms in timed_ms])
    monkeypatch.setattr(time, "perf_counter", lambda: model.now)

    record = measure_batch1_latency(model, (2,), warmup=3, repeats=20, seed=0)

    assert record.median_ms == pytest.approx(10.5, rel=1e-6)
    assert record.p95_ms == pytest.approx(30.0, rel=1e-6)
    assert record.mean_ms == pytest.approx(11.55, rel=1e-6)
    assert record.repeats == 20


@pytest.mark.parametrize(
    "kwargs",
    [{"repeats": 0}, {"repeats": -3}, {"warmup": -1}],
    ids=["repeats-zero", "repeats-negative", "warmup-negative"],
)
def test_latency_rejects_bad_counts_before_running_any_forward(kwargs) -> None:
    model = _linear_recorder()
    model.train()
    with pytest.raises(ValueError):
        measure_batch1_latency(model, (4,), **kwargs)
    assert model.batch_sizes == []
    assert model.training is True


@pytest.mark.parametrize("bad_shape", [[4], 4, "4"], ids=["list", "int", "str"])
def test_latency_rejects_input_shape_that_is_not_a_tuple(bad_shape) -> None:
    model = _linear_recorder()
    with pytest.raises(ValueError):
        measure_batch1_latency(model, bad_shape)
    assert model.batch_sizes == []


def test_latency_record_is_a_frozen_value_with_the_declared_fields() -> None:
    record = LatencyRecord(median_ms=1.0, p95_ms=2.0, mean_ms=1.5, repeats=20, num_threads=4, device="cpu")
    assert (record.median_ms, record.p95_ms, record.mean_ms) == (1.0, 2.0, 1.5)
    assert (record.repeats, record.num_threads, record.device) == (20, 4, "cpu")
    assert record == LatencyRecord(1.0, 2.0, 1.5, 20, 4, "cpu")
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.median_ms = 9.0  # type: ignore[misc]
    assert record.median_ms == 1.0


# --------------------------------------------------------------------------- count_parameters / model_size_bytes


def test_linear_3_to_2_has_eight_trainable_parameters_and_thirty_two_bytes() -> None:
    model = nn.Linear(3, 2)
    assert count_parameters(model) == 8  # 3*2 weights + 2 biases
    assert model_size_bytes(model) == 32  # 8 float32 * 4 bytes


def test_size_bytes_follows_the_element_size_of_the_dtype() -> None:
    assert model_size_bytes(nn.Linear(3, 2).double()) == 64
    assert model_size_bytes(nn.Linear(3, 2).half()) == 16


def test_count_excludes_frozen_parameters_but_size_includes_them_and_batchnorm_buffers() -> None:
    """Sequential(Linear(3,2), BatchNorm1d(2)) with the linear weight frozen.

    trainable = linear.bias 2 + bn.weight 2 + bn.bias 2 = 6
    bytes = params (6+2+2+2 = 12 float32 = 48) + buffers (running_mean 2 + running_var 2 float32 = 16,
            num_batches_tracked one int64 = 8) = 72
    """
    model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
    model[0].weight.requires_grad_(False)

    assert count_parameters(model) == 6
    assert model_size_bytes(model) == 72


def test_model_without_parameters_or_buffers_counts_zero() -> None:
    assert count_parameters(nn.ReLU()) == 0
    assert model_size_bytes(nn.ReLU()) == 0


def test_logmel_cnn_trainable_parameter_count_matches_the_hand_computation() -> None:
    """width=2, n_mels=8, 3 classes.
    conv1 1*2*3*3=18, bn1 2*2=4, conv2 2*4*9=72, bn2 2*4=8, conv3 4*4*9=144, bn3 8, fc 4*3+3=15 -> 269.
    """
    assert count_parameters(LogMelCNN(n_mels=8, num_classes=3, width=2)) == 269


def test_counting_and_sizing_do_not_change_the_model() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2)).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    count_parameters(model)
    model_size_bytes(model)
    assert model.training is True
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
