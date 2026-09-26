"""Notebook plotting helpers with a consistent, color-blind friendly palette."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

PALETTE = {"blue": "#3568A8", "orange": "#E58B2A", "green": "#3A8D68", "gray": "#667085", "light": "#D7DEE8"}


def plot_reliability(confidence: np.ndarray, correct: np.ndarray, bins: tuple[int, ...] = (10, 15, 20)):
    conf, ok = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=bool)
    if conf.ndim != 1 or ok.shape != conf.shape or conf.size == 0 or not np.all(np.isfinite(conf)) or np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidence and correctness must be aligned; confidence must lie in [0, 1]")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot([0, 1], [0, 1], color=PALETTE["gray"], linestyle="--", label="Perfect calibration")
    colors = (PALETTE["blue"], PALETTE["orange"], PALETTE["green"])
    for n_bins, color in zip(bins, colors, strict=False):
        if n_bins <= 0:
            raise ValueError("bin counts must be positive")
        index = np.minimum((conf * n_bins).astype(int), n_bins - 1)
        x, y, counts = [], [], []
        for b in range(n_bins):
            mask = index == b
            if np.any(mask):
                x.append(float(conf[mask].mean()))
                y.append(float(ok[mask].mean()))
                counts.append(int(mask.sum()))
        ax.plot(x, y, marker="o", color=color, label=f"{n_bins} bins")
        for cx, cy, count in zip(x, y, counts, strict=True):
            ax.annotate(str(count), (cx, cy), xytext=(3, 3), textcoords="offset points", fontsize=7, color=color)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence in bin", ylabel="Empirical accuracy", title="Reliability across bin counts")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_risk_coverage(coverage: np.ndarray, selective_risk: np.ndarray, random_reject_reference: np.ndarray):
    cov, risk, random = (np.asarray(x, dtype=float) for x in (coverage, selective_risk, random_reject_reference))
    if cov.ndim != 1 or risk.shape != cov.shape or random.shape != cov.shape or cov.size == 0:
        raise ValueError("coverage and both risk series must be aligned non-empty vectors")
    if not np.all(np.isfinite(cov)) or not np.all(np.isfinite(risk)) or not np.all(np.isfinite(random)) or np.any(np.diff(cov) < 0):
        raise ValueError("curve values must be finite and coverage ordered")
    plotted_coverage = np.concatenate(([0.0], cov))
    plotted_risk = np.concatenate(([0.0], risk))
    plotted_random = np.concatenate(([0.0], random))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(plotted_coverage, plotted_risk, color=PALETTE["blue"], linewidth=2.2, label="Confidence-ranked retention")
    ax.plot(plotted_coverage, plotted_random, color=PALETTE["orange"], linestyle="--", label="Random-reject reference")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Retained coverage", ylabel="Selective risk", title="Risk as retained coverage changes")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_waveform_shift(time: np.ndarray, waveform: np.ndarray, transformed_waveform: np.ndarray, *, snr_db: float, gain_db: float, reverb_decay_seconds: float):
    t, clean, changed = (np.asarray(x, dtype=float) for x in (time, waveform, transformed_waveform))
    if t.ndim != 1 or clean.ndim != 1 or changed.shape != clean.shape or clean.shape != t.shape or t.size == 0:
        raise ValueError("time and waveform arrays must be aligned vectors")
    if not all(np.all(np.isfinite(x)) for x in (t, clean, changed)):
        raise ValueError("waveforms and time must be finite")
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(t, clean, color=PALETTE["blue"], linewidth=1, label="Original")
    ax.plot(t, changed, color=PALETTE["orange"], linewidth=1, alpha=0.85, label="Synthetic shifted")
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title="Known acoustic severity audit")
    ax.text(0.99, 0.97, f"SNR {snr_db:.1f} dB\nGain {gain_db:+.1f} dB\nDecay {reverb_decay_seconds:.2f} s", transform=ax.transAxes, va="top", ha="right", bbox={"facecolor": "white", "edgecolor": PALETTE["light"], "alpha": 0.9})
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax
