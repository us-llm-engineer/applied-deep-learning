"""Contract for src.pipeline.run_arm and summarise_arm (width: cluster; tiny CPU workloads)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from pipeline_builders import build_dataset
from src.calibration import crc_threshold, temperature_scale
from src.experiment import ExperimentConfig
from src.latency import count_parameters, model_size_bytes
from src.models import LogMelCNN, log_mel
from src.pipeline import assemble_bundle, run_arm, summarise_arm
from src.stress import apply_stress

STRESS = ("clean", "noise_10", "gain_+10")
CONDITIONS = ("clean", "masking", "acoustic")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = build_dataset(tmp_path_factory.mktemp("tiny"), command_clips=8, aux_clips=2)
    return assemble_bundle(root, seed=17, cap_per_label=8)


def _cfg(condition, seed):
    return ExperimentConfig(condition=condition, seed=seed, max_epochs=2)


def _run(bundle, condition="clean", seed=17, **kw):
    return run_arm(bundle, condition=condition, seed=seed, config=_cfg(condition, seed),
                   stress_names=STRESS, latency_repeats=1, **kw)


def _float(waves):
    return np.asarray(waves, dtype=np.float32) / 32768.0


class FixedModel(nn.Module):
    """Logits are a fixed function of the features; training cannot change them (dummy param x 0)."""
    train_rows: list = []      # class-level so copy.deepcopy inside train_classifier keeps recording here
    eval_rows: list = []

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(123)
        self.lin = nn.Linear(40, 12)
        self.lin.weight = nn.Parameter(torch.randn(12, 40, generator=g), requires_grad=False)
        self.lin.bias = nn.Parameter(torch.randn(12, generator=g), requires_grad=False)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        (FixedModel.train_rows if self.training else FixedModel.eval_rows).append(x.detach().clone())
        return self.lin(x.mean(dim=(1, 3))) + 0.0 * self.dummy


@pytest.fixture()
def fixed_runs(bundle):
    def go(condition="clean", seed=17):
        FixedModel.train_rows, FixedModel.eval_rows = [], []
        return _run(bundle, condition, seed, model_factory=FixedModel)
    return go


def _expected(waves_int16, stress=None, seed=17):
    x = _float(waves_int16)
    if stress:
        x, _ = apply_stress(x, stress, seed=seed)
    with torch.no_grad():
        return FixedModel().eval()(log_mel(torch.from_numpy(x))).numpy()


# ------------------------------------------------------------------ run_arm

@pytest.fixture(scope="module")
def clean_run(bundle):
    return _run(bundle)


def test_arm_run_shapes_keys_and_resources(bundle, clean_run):
    """Logits have one 12-way row per cal/test/ood clip for every requested stress; resources match src.latency."""
    r, sp = clean_run, bundle.splits
    assert (r.condition, r.seed) == ("clean", 17)
    assert r.logits_cal.shape == (sp["cal"].y.size, 12)
    assert set(r.logits_test) == set(STRESS) and set(r.logits_ood) == set(STRESS)
    for name in STRESS:
        assert r.logits_test[name].shape == (sp["test"].y.size, 12), name
        assert r.logits_ood[name].shape == (sp["ood"].y.size, 12), name
        assert np.isfinite(r.logits_test[name]).all() and np.isfinite(r.logits_ood[name]).all()
    assert r.ids_test == sp["test"].ids
    assert r.train["epochs_run"] == 2 and 0 <= r.train["best_epoch"] < 2
    assert r.train["optimizer_steps"] > 0 and r.train["wall_seconds"] >= 0
    ref = LogMelCNN()
    assert r.resources["parameters"] == count_parameters(ref)
    assert r.resources["model_size_bytes"] == model_size_bytes(ref)
    assert r.resources["latency_cpu_median_ms"] > 0
    assert r.resources["peak_memory_mb"] == 0.0 and r.resources["device"] == "cpu"


def test_two_runs_with_the_same_seed_give_identical_logits(bundle, clean_run):
    again = _run(bundle)
    assert np.array_equal(again.logits_cal, clean_run.logits_cal)
    for name in STRESS:
        assert np.array_equal(again.logits_test[name], clean_run.logits_test[name]), name
        assert np.array_equal(again.logits_ood[name], clean_run.logits_ood[name]), name


def test_different_seeds_give_different_logits(bundle, clean_run):
    other = _run(bundle, seed=18)
    assert not np.allclose(other.logits_cal, clean_run.logits_cal)
    assert other.ids_test == clean_run.ids_test


def test_optimizer_steps_are_equal_across_the_three_conditions_and_match_the_budget(bundle):
    """Same train size, epochs and accumulation => same step count; steps == epochs * ceil(ceil(n/64)/2) (a final partial accumulation group is flushed, never discarded)."""
    n = bundle.splits["train"].y.size
    runs = {c: _run(bundle, c) for c in CONDITIONS}
    steps = {c: r.train["optimizer_steps"] for c, r in runs.items()}
    assert len(set(steps.values())) == 1, steps
    r = runs["clean"]
    microbatches = -(-n // 64)
    assert steps["clean"] == r.train["epochs_run"] * (-(-microbatches // 2))


def test_clean_stress_logits_equal_the_plain_forward_pass_and_stressed_rows_are_paired(bundle, fixed_runs):
    """With a training-insensitive model, every row of every stress equals model(stress(test wave i)); cal is clean."""
    r = fixed_runs()
    test_w = bundle.splits["test"].waveforms
    assert np.allclose(r.logits_cal, _expected(bundle.splits["cal"].waveforms), atol=1e-5)
    assert np.allclose(r.logits_test["clean"], _expected(test_w), atol=1e-5)
    assert np.allclose(r.logits_test["noise_10"], _expected(test_w, "noise_10", seed=17), atol=1e-5)
    assert np.allclose(r.logits_test["gain_+10"], _expected(test_w, "gain_+10"), atol=1e-5)
    ood_w = bundle.splits["ood"].waveforms
    assert np.allclose(r.logits_ood["clean"], _expected(ood_w), atol=1e-5)
    assert np.allclose(r.logits_ood["gain_+10"], _expected(ood_w, "gain_+10"), atol=1e-5)


@pytest.mark.parametrize("condition", ["masking", "acoustic"])
def test_augmentation_is_training_only_so_evaluation_logits_match_the_clean_arm(bundle, fixed_runs, condition):
    ref = fixed_runs("clean")
    aug = fixed_runs(condition)
    assert np.array_equal(aug.logits_cal, ref.logits_cal)
    for name in STRESS:
        assert np.array_equal(aug.logits_test[name], ref.logits_test[name]), name


def test_training_forward_passes_see_exactly_the_train_split_once_per_epoch(bundle, fixed_runs):
    """Given clean training, then rows seen in training mode == epochs * n_train and each is a train-split feature."""
    r = fixed_runs("clean")
    seen = torch.cat(FixedModel.train_rows)
    sp = bundle.splits
    assert seen.shape[0] == r.train["epochs_run"] * sp["train"].y.size
    pool, owner = [], []
    for name in ("train", "val", "cal", "test"):
        f = log_mel(torch.from_numpy(_float(sp[name].waveforms)))
        pool.append(f.flatten(1)); owner += [name] * f.shape[0]
    dist = torch.cdist(seen.flatten(1), torch.cat(pool))
    nearest = dist.argmin(dim=1)
    assert dist.min(dim=1).values.max() < 1e-3, "a training row matches no clip of the bundle"
    assert {owner[i] for i in nearest.tolist()} == {"train"}


@pytest.mark.parametrize("condition", ["masking", "acoustic"])
def test_augmentation_actually_changes_training_inputs(bundle, fixed_runs, condition):
    """The multiset of training features differs from the clean arm's (column sums are order-independent)."""
    fixed_runs("clean"); clean_sum = torch.cat(FixedModel.train_rows).double().sum().item()
    fixed_runs(condition); aug_sum = torch.cat(FixedModel.train_rows).double().sum().item()
    assert not math.isclose(clean_sum, aug_sum, rel_tol=1e-6)


# ------------------------------------------------------------------ summarise_arm

K = 5


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _synthetic(n, seed, scale=1.0):
    """Well specified when scale == 1: labels are drawn from softmax(z); scale > 1 makes logits over-confident."""
    rng = np.random.default_rng(seed)
    z = 1.5 * rng.normal(size=(n, K))
    p = _softmax(z)
    y = (rng.random((n, 1)) > np.cumsum(p, axis=1)).sum(axis=1).clip(0, K - 1).astype(np.int64)
    return scale * z, y


def _ood(n=40, seed=5):
    return 0.3 * np.random.default_rng(seed).normal(size=(n, K))


def _summ(cal, cal_y, test, test_y, ood, **kw):
    kw.setdefault("seed", 3); kw.setdefault("bootstrap_resamples", 20)
    return summarise_arm(cal, cal_y, {"clean": test}, test_y, {"clean": ood}, **kw)


def _cell(out, mode, cond="clean"):
    root = out["conditions"] if "conditions" in out else out
    return root[cond][mode]


HAND = np.log(3.0) * np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
HAND_Y = np.array([0, 1, 1, 1])


def _hand_summary():
    cal, cal_y = _synthetic(300, 1)
    return _summ(cal[:, :3], cal_y % 3, HAND, HAND_Y, _ood()[:, :3])


def test_top_level_fields_and_json_serialisability():
    cal, cal_y = _synthetic(200, 1); test, test_y = _synthetic(50, 2)
    out = _summ(cal, cal_y, test, test_y, _ood())
    json.dumps(out)
    assert (out["seed"], out["n_cal"], out["n_test"], out["n_ood"]) == (3, 200, 50, 40)
    assert out["temperature"] > 0


def test_hand_computed_accuracy_nll_and_brier_on_a_four_row_example():
    """probabilities (.6,.2,.2)-style rows: acc 2/4; nll = -(ln .6 + ln .2)/2; Brier = (.24+.24+1.04+1.04)/4."""
    cell = _cell(_hand_summary(), "uncalibrated")
    assert cell["accuracy"] == pytest.approx(0.5)
    assert cell["nll"] == pytest.approx(-0.5 * (math.log(0.6) + math.log(0.2)), abs=1e-9)
    assert cell["brier"] == pytest.approx(0.64, abs=1e-9)
    assert cell["selective_risk_0.8"] == pytest.approx(0.5) and cell["selective_risk_0.9"] == pytest.approx(0.5)


def test_accuracy_is_identical_calibrated_and_uncalibrated():
    cal, cal_y = _synthetic(500, 1, 3.0); test, test_y = _synthetic(300, 2, 3.0)
    out = _summ(cal, cal_y, test, test_y, _ood())
    assert _cell(out, "calibrated")["accuracy"] == _cell(out, "uncalibrated")["accuracy"]


def test_temperature_uses_only_cal_logits_and_equals_the_reference_fit():
    cal, cal_y = _synthetic(600, 1, 3.0)
    a = _summ(cal, cal_y, *_synthetic(80, 2), _ood())
    b = _summ(cal, cal_y, *_synthetic(120, 9, 0.5), _ood(seed=8))
    assert a["temperature"] == b["temperature"] == temperature_scale(cal, cal_y)


def test_temperature_tracks_calibration_overconfidence_and_improves_shifted_nll():
    """Well specified cal -> T near 1; logits inflated x3 -> T near 3, and calibrated NLL beats uncalibrated."""
    well = _summ(*_synthetic(2000, 1), *_synthetic(200, 2), _ood())["temperature"]
    cal, cal_y = _synthetic(2000, 1, 3.0); test, test_y = _synthetic(1000, 2, 3.0)
    over = _summ(cal, cal_y, test, test_y, _ood())
    assert 0.85 < well < 1.15
    assert 2.5 < over["temperature"] < 3.6
    assert _cell(over, "calibrated")["nll"] < _cell(over, "uncalibrated")["nll"]


def _auroc(id_a, ood_a, mode="uncalibrated"):
    rows = lambda a: np.array([[v, 0.0, 0.0] for v in a])
    cal, cal_y = _synthetic(300, 1)
    out = _summ(cal[:, :3], cal_y % 3, rows(id_a), np.zeros(len(id_a), dtype=np.int64), rows(ood_a))
    return _cell(out, mode)["ood_auroc"]


def test_ood_auroc_is_one_for_separable_confidences_in_both_modes():
    assert _auroc([4.0, 5.0], [1.0, 2.0], "uncalibrated") == 1.0
    assert _auroc([4.0, 5.0], [1.0, 2.0], "calibrated") == 1.0


def test_ood_auroc_is_half_when_confidences_are_identical_and_counts_ties_as_half():
    assert _auroc([2.0, 2.0], [2.0, 2.0]) == 0.5
    # ID conf {3,2} vs OOD {2,1}: pairs 1,1,tie(.5),1 -> 3.5/4
    assert _auroc([3.0, 2.0], [2.0, 1.0]) == pytest.approx(0.875)


def test_summary_is_deterministic_per_seed_and_bootstrap_depends_on_seed():
    cal, cal_y = _synthetic(300, 1); test, test_y = _synthetic(60, 2)
    a = _summ(cal, cal_y, test, test_y, _ood(), seed=1)
    assert json.dumps(a) == json.dumps(_summ(cal, cal_y, test, test_y, _ood(), seed=1))
    b = _summ(cal, cal_y, test, test_y, _ood(), seed=2)
    assert {k: v for k, v in a.items() if k != "seed"} != {k: v for k, v in b.items() if k != "seed"}


def test_crc_threshold_matches_reference_and_miscoverage_is_near_alpha_on_a_well_specified_problem():
    """n=2000 cal and test rows: miscoverage <= alpha + 0.1 (loose); thresholds/sets are monotone in alpha."""
    cal, cal_y = _synthetic(2000, 1); test, test_y = _synthetic(2000, 2)
    out = _summ(cal, cal_y, test, test_y, _ood(), alphas=(0.05, 0.10))
    for mode in ("uncalibrated", "calibrated"):
        crc = _cell(out, mode)["crc"]
        for alpha in (0.05, 0.10):
            assert crc[str(alpha)]["miscoverage"] <= alpha + 0.1, (mode, alpha)
        assert crc["0.05"]["threshold"] <= crc["0.1"]["threshold"]
        assert crc["0.05"]["mean_set_size"] >= crc["0.1"]["mean_set_size"]
    t = out["temperature"]
    true_scores = _softmax(cal / t)[np.arange(cal_y.size), cal_y]
    assert _cell(out, "calibrated")["crc"]["0.05"]["threshold"] == pytest.approx(crc_threshold(true_scores, 0.05))
