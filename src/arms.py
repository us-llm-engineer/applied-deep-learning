"""Per-arm training, prediction, and evaluation (run_arm, summarise_arm).

Design reference: plans/round-02/training-pipeline.md sections 2, 4, 5, 6.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from .calibration import _softmax, crc_threshold, prediction_sets, temperature_scale
from .experiment import _macro_f1
from .latency import count_parameters, measure_batch1_latency, model_size_bytes
from .metrics import (
    aurc,
    bootstrap_interval,
    brier_score,
    confidence,
    expected_calibration_error,
    nll,
    risk_coverage_curve,
    selective_risk,
)
from .models import LogMelCNN, log_mel
from .stress import apply_stress
from .train import apply_condition, predict_logits, spec_mask, train_classifier

# Masking arm: fixed band widths (design section 2: freq_mask <= 8 mel rows, time_mask <= 20 frames).
MASK_FREQ_ROWS = 8
MASK_TIME_FRAMES = 20
_FEATURE_CHUNK = 256


@dataclass
class ArmRun:
    """Result of running one arm (training condition, seed) of the training pipeline.

    Fields:
        condition: training condition, one of clean / masking / acoustic.
        seed: seed used for model init, shuffling, augmentation and stress noise.
        logits_cal: (n_cal, classes) logits on the clean calibration split.
        logits_test: stress name -> (n_test, classes) logits; row i is stress(test clip i),
            so rows are paired across stress conditions and across arms.
        logits_ood: stress name -> (n_ood, classes) logits on the held-out-word OOD set.
        ids_test: identifiers of the test clips, aligned with the rows of logits_test.
        train: epochs_run, best_epoch (0-based), optimizer_steps, wall_seconds of training.
        resources: parameters, model_size_bytes, latency_cpu_median_ms (batch-1 CPU),
            peak_memory_mb (real CUDA measurement when device is a CUDA device, else 0.0), device.
    """
    condition: str
    seed: int
    logits_cal: np.ndarray
    logits_test: dict[str, np.ndarray]
    logits_ood: dict[str, np.ndarray]
    ids_test: tuple[str, ...]
    train: dict
    resources: dict


def _to_float(int16_waveforms: np.ndarray) -> np.ndarray:
    """int16 PCM -> float32 in [-1, 1) (divide by 32768)."""
    return np.asarray(int16_waveforms, dtype=np.float32) / 32768.0


def _features(waveforms: np.ndarray) -> np.ndarray:
    """float32 (n, samples) waveforms -> float32 log-mel (n, 1, n_mels, frames), computed in chunks."""
    chunks = [
        log_mel(torch.from_numpy(np.ascontiguousarray(waveforms[i:i + _FEATURE_CHUNK]))).numpy()
        for i in range(0, len(waveforms), _FEATURE_CHUNK)
    ]
    return np.concatenate(chunks, axis=0)


def _feature_stats(features: np.ndarray) -> tuple[float, float]:
    """Global mean/std of a log-mel feature array, for input normalization.

    Global (whole-array) rather than per-utterance on purpose: per-utterance normalization
    divides out each clip's overall level, which would make the gain_* stress conditions
    trivially invariant and silently nullify three of the eleven conditions this study measures.
    """
    mean = float(np.mean(features))
    std = float(np.std(features))
    return mean, (std if std > 0.0 else 1.0)


def _normalize(features: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardize log-mel features by precomputed statistics (train-split only: no leakage)."""
    return ((np.asarray(features, dtype=np.float32) - mean) / std).astype(np.float32)


def _masking_augment() -> Callable:
    """Training-only augment_fn: spec_mask on every log-mel example of the batch.

    Randomness comes from the rng that train_classifier supplies per microbatch,
    so masks change from batch to batch and epoch to epoch.
    """
    def augment(features: np.ndarray, _condition: str, rng: np.random.Generator) -> np.ndarray:
        out = features.copy()
        for i in range(out.shape[0]):
            out[i, 0] = spec_mask(out[i, 0], freq_mask=MASK_FREQ_ROWS, time_mask=MASK_TIME_FRAMES, rng=rng)
        return out
    return augment


def _acoustic_augment(seed: int, mean: float = 0.0, std: float = 1.0) -> Callable:
    """Training-only augment_fn: takes a batch of float waveforms, returns their log-mel features.

    apply_condition("acoustic") draws a fresh random gain / white-noise / reverb per
    example. The generator is keyed on (seed, call counter) rather than on the rng
    train_classifier supplies, because that one is seeded from epoch*1000 + batch offset
    and can repeat across epochs on datasets larger than 1000 clips.
    """
    calls = [0]

    def augment(waveforms: np.ndarray, _condition: str, _rng: np.random.Generator) -> np.ndarray:
        rng = np.random.default_rng((seed, calls[0]))
        calls[0] += 1
        return _normalize(_features(apply_condition(waveforms, "acoustic", rng)), mean, std)
    return augment


def run_arm(
    bundle,
    *,
    condition: str = "clean",
    seed: int = 17,
    config,
    stress_names: tuple[str, ...],
    latency_repeats: int = 1,
    model_factory: Optional[Callable[[], nn.Module]] = None,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
    probe_fn=None,
) -> ArmRun:
    """Train one model on the train split and predict cal, stressed test and stressed OOD.

    Training inputs by condition:
      * clean: log_mel(float32(int16) / 32768) of the train split, unmodified.
      * masking: the same features, with spec_mask applied per example per microbatch
        (fresh random bands every batch).
      * acoustic: the float waveforms are given to train_classifier and every microbatch
        is passed through apply_condition (fresh gain / noise / reverb per example) and
        then log_mel, so each epoch sees new random augmentations.
    Early stopping uses the clean val split; cal, test and OOD never see augmentation.
    Stress conditions are applied to test and OOD waveforms via apply_stress(seed=seed),
    so row i of every stress condition derives from clip i.

    Args:
        bundle: DatasetBundle with .splits train/val/cal/test/ood (int16 waveforms).
        condition: training condition (clean, masking, acoustic).
        seed: seed for model init, shuffling, augmentation and stress noise.
        config: ExperimentConfig (epochs, patience, microbatch, accumulation).
        stress_names: names from src.stress.STRESS_CONDITIONS to evaluate on test and OOD.
        latency_repeats: repeats for the batch-1 CPU latency measurement.
        model_factory: zero-argument callable building a fresh model (default LogMelCNN).
        device: device to train on (default 'cpu', can be 'cuda' or 'cuda:0' etc.).
        probe_fn: optional per-epoch probe forwarded to train_classifier; whatever it returns
            lands in ArmRun.train["probe_history"]. See src.diagnostics for the probes.
        checkpoint_path: if given, the trained model's state_dict is saved there via torch.save
            (after training, before prediction) -- the checkpoint is not loaded back or used by
            run_arm itself, it is purely a side-effect for external reuse.

    Returns:
        ArmRun (see its docstring).
    """
    if condition not in ("clean", "masking", "acoustic"):
        raise ValueError(f"unknown condition {condition!r}")
    model_factory = model_factory or LogMelCNN
    device_obj = torch.device(device)
    is_cuda = device_obj.type == "cuda"
    splits = bundle.splits

    torch.manual_seed(seed)  # model initialisation; train_classifier reseeds for shuffling

    train_waves = _to_float(splits["train"].waveforms)
    train_feats = _features(train_waves)
    # Normalization statistics come from the train split only -- never from val/cal/test/ood.
    feat_mean, feat_std = _feature_stats(train_feats)

    val_x = _normalize(_features(_to_float(splits["val"].waveforms)), feat_mean, feat_std)
    cal_x = _normalize(_features(_to_float(splits["cal"].waveforms)), feat_mean, feat_std)

    if condition == "acoustic":
        train_x, augment_fn = train_waves, _acoustic_augment(seed, feat_mean, feat_std)
    else:
        train_x = _normalize(train_feats, feat_mean, feat_std)
        augment_fn = _masking_augment() if condition == "masking" else None

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device_obj)

    train_result = train_classifier(
        model_factory().to(device_obj), train_x, splits["train"].y, val_x, splits["val"].y, config,
        augment_fn=augment_fn, probe_fn=probe_fn,
    )
    model = train_result.model
    peak_memory_mb = (torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)) if is_cuda else 0.0

    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)

    logits_cal = predict_logits(model, cal_x)

    def stressed_logits(split: str) -> dict[str, np.ndarray]:
        waves = _to_float(splits[split].waveforms)
        return {
            name: predict_logits(
                model, _normalize(_features(apply_stress(waves, name, seed=seed)[0]), feat_mean, feat_std)
            )
            for name in stress_names
        }

    logits_test = stressed_logits("test")
    logits_ood = stressed_logits("ood")

    latency = measure_batch1_latency(
        model, input_shape=tuple(cal_x.shape[1:]), warmup=1, repeats=latency_repeats, seed=seed,
    )
    return ArmRun(
        condition=condition,
        seed=seed,
        logits_cal=logits_cal,
        logits_test=logits_test,
        logits_ood=logits_ood,
        ids_test=splits["test"].ids,
        train={
            "epochs_run": train_result.epochs_run,
            "best_epoch": train_result.best_epoch,
            "optimizer_steps": train_result.optimizer_steps,
            "wall_seconds": train_result.wall_seconds,
            "history": train_result.history,
            "grad_norm_history": train_result.grad_norm_history,
            "clip_threshold_history": train_result.clip_threshold_history,
            "inter_cluster_distance_history": train_result.inter_cluster_distance_history,
            "intra_cluster_variance_history": train_result.intra_cluster_variance_history,
            "instability_alert_epoch": train_result.instability_alert_epoch,
            "probe_history": train_result.probe_history,
        },
        resources={
            "parameters": count_parameters(model),
            "model_size_bytes": model_size_bytes(model),
            "latency_cpu_median_ms": latency.median_ms,
            "peak_memory_mb": peak_memory_mb,
            "device": latency.device,
        },
    )


def _ood_auroc(id_conf: np.ndarray, ood_conf: np.ndarray) -> float:
    """AUROC of max-softmax confidence separating ID (positive) from OOD (negative).

    Rank formula: P(conf_id > conf_ood) + 0.5 * P(conf_id == conf_ood) over all ID/OOD pairs.
    """
    ood_sorted = np.sort(ood_conf)
    below = np.searchsorted(ood_sorted, id_conf, side="left")
    below_or_tied = np.searchsorted(ood_sorted, id_conf, side="right")
    return float(((below + below_or_tied) / 2.0).sum() / (id_conf.size * ood_conf.size))


def _interval(statistic: Callable[[np.ndarray], float], n: int, seed: int, resamples: int) -> dict:
    """Seeded bootstrap of a statistic of test-row indices; returns {point, lower, upper}.

    The resampled indices depend only on (n, seed, resamples), so intervals are paired
    across modes, conditions and arms that share the test set and seed.
    """
    interval = bootstrap_interval(
        np.arange(n, dtype=float),
        statistic=lambda idx: statistic(idx.astype(np.int64)),
        seed=seed,
        resamples=resamples,
    )
    return asdict(interval)


def _mode_metrics(
    logits_test: np.ndarray,
    labels_test: np.ndarray,
    logits_ood: np.ndarray,
    logits_cal: np.ndarray,
    labels_cal: np.ndarray,
    temperature: float,
    *,
    seed: int,
    bootstrap_resamples: int,
    alphas: tuple[float, ...],
) -> dict:
    """Metrics of one (stress condition, mode) cell; every probability is softmax(logits / temperature).

    temperature is 1.0 for the uncalibrated mode and the cal-fit value for the calibrated
    mode; it is applied to the cal, test and OOD logits alike (CRC threshold included).
    """
    probs = _softmax(logits_test / temperature)
    probs_ood = _softmax(logits_ood / temperature)
    probs_cal = _softmax(logits_cal / temperature)
    n = labels_test.size

    prediction = np.argmax(probs, axis=1)
    correct = prediction == labels_test
    conf = confidence(probs)
    coverage, risk = risk_coverage_curve(conf, correct)

    crc = {}
    cal_true_scores = probs_cal[np.arange(labels_cal.size), labels_cal]
    for alpha in alphas:
        threshold = crc_threshold(cal_true_scores, alpha)
        sets = prediction_sets(logits_test, temperature=temperature, threshold=threshold)
        crc[str(alpha)] = {
            "threshold": float(threshold),
            "miscoverage": float(np.mean([int(y) not in s for y, s in zip(labels_test, sets)])),
            "mean_set_size": float(np.mean([len(s) for s in sets])),
        }

    def aurc_of(idx: np.ndarray) -> float:
        cov, rsk = risk_coverage_curve(conf[idx], correct[idx])
        return aurc(cov, rsk) if cov.size > 1 else float(rsk[0])

    def ci(statistic: Callable[[np.ndarray], float]) -> dict:
        return _interval(statistic, n, seed, bootstrap_resamples)

    return {
        "accuracy": float(correct.mean()),
        "macro_f1": _macro_f1(prediction, labels_test, probs.shape[1]),
        "nll": nll(probs, labels_test),
        "brier": brier_score(probs, labels_test),
        "ece_10": expected_calibration_error(probs, labels_test, bins=10),
        "ece_15": expected_calibration_error(probs, labels_test, bins=15),
        "ece_20": expected_calibration_error(probs, labels_test, bins=20),
        "aurc": aurc(coverage, risk) if coverage.size > 1 else float(risk[0]),
        "selective_risk_0.8": selective_risk(conf, correct, coverage=0.8),
        "selective_risk_0.9": selective_risk(conf, correct, coverage=0.9),
        "ood_auroc": _ood_auroc(conf, confidence(probs_ood)),
        "crc": crc,
        "ci": {
            "accuracy": ci(lambda idx: float(correct[idx].mean())),
            "ece_10": ci(lambda idx: expected_calibration_error(probs[idx], labels_test[idx], bins=10)),
            "nll": ci(lambda idx: nll(probs[idx], labels_test[idx])),
            "aurc": ci(aurc_of),
        },
    }


def summarise_arm(
    logits_cal: np.ndarray,
    labels_cal: np.ndarray,
    logits_test: dict[str, np.ndarray],
    labels_test: np.ndarray,
    logits_ood: dict[str, np.ndarray],
    *,
    seed: int = 3,
    bootstrap_resamples: int = 2000,
    alphas: tuple[float, ...] = (0.05, 0.10),
) -> dict:
    """Summarise one arm's predictions in uncalibrated and temperature-scaled modes.

    The temperature is fit on the (clean) cal logits only. Uncalibrated mode uses raw
    softmax for everything, CRC threshold included; calibrated mode uses
    softmax(logits / T) for cal (CRC threshold), test and OOD. CRC results are empirical
    measurements on the stressed test set, not shift-valid guarantees.

    Args:
        logits_cal: (n_cal, classes) clean calibration logits.
        labels_cal: (n_cal,) integer calibration labels.
        logits_test: stress name -> (n_test, classes) logits of that stressed test set.
        labels_test: (n_test,) labels shared by every stress condition.
        logits_ood: stress name -> (n_ood, classes) logits of the OOD-unknown set; must
            have the same keys as logits_test.
        seed: seed of the bootstrap (identical resample indices in both modes).
        bootstrap_resamples: number of bootstrap resamples of test rows.
        alphas: CRC target miscoverage levels.

    Returns:
        JSON-serialisable dict:
          seed, n_cal, n_test, n_ood, temperature (fit on cal), and
          conditions[stress_name][mode] for mode in "uncalibrated" / "calibrated", each with
            accuracy, macro_f1, nll, brier: point metrics on the test rows;
            ece_10 / ece_15 / ece_20: top-label ECE with that many equal-width bins;
            aurc: area under the risk-coverage curve (confidence = max softmax);
            selective_risk_0.8 / _0.9: error rate on the most confident 80% / 90% of rows;
            ood_auroc: AUROC of max-softmax, ID test rows (positive) vs OOD rows;
            crc[str(alpha)]: threshold (fit on cal true-label probabilities), miscoverage
              and mean_set_size of the prediction sets on the test rows;
            ci: 95% seeded bootstrap {point, lower, upper} for accuracy, ece_10, nll, aurc.
    """
    if set(logits_test) != set(logits_ood):
        raise ValueError("logits_test and logits_ood must cover the same stress conditions")
    logits_cal = np.asarray(logits_cal, dtype=float)
    labels_cal = np.asarray(labels_cal)
    labels_test = np.asarray(labels_test)

    temperature = temperature_scale(logits_cal, labels_cal)
    conditions = {}
    for name in sorted(logits_test):
        test = np.asarray(logits_test[name], dtype=float)
        ood = np.asarray(logits_ood[name], dtype=float)
        conditions[name] = {
            mode: _mode_metrics(
                test, labels_test, ood, logits_cal, labels_cal, t,
                seed=seed, bootstrap_resamples=bootstrap_resamples, alphas=alphas,
            )
            for mode, t in (("uncalibrated", 1.0), ("calibrated", temperature))
        }
    return {
        "seed": seed,
        "n_cal": int(labels_cal.size),
        "n_test": int(labels_test.size),
        "n_ood": int(next(iter(logits_ood.values())).shape[0]),
        "temperature": float(temperature),
        "conditions": conditions,
    }
