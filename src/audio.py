"""Deterministic waveform transforms used for controlled acoustic shifts."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


def _waveform(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("waveform must be a one-dimensional numeric array")
    if not np.isfinite(array).all():
        raise ValueError("waveform must contain only finite samples")
    return array.astype(np.float32, copy=False)


def add_white_noise(
    waveform: np.ndarray, *, sample_rate: int, snr_db: float, seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    samples = _waveform(waveform)
    if sample_rate <= 0 or not np.isfinite(snr_db):
        raise ValueError("sample_rate must be positive and snr_db finite")
    signal_rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    noise_rms = signal_rms / (10.0 ** (float(snr_db) / 20.0))
    noise = np.random.default_rng(seed).standard_normal(samples.shape).astype(np.float32)
    output = samples + noise * np.float32(noise_rms)
    return output.astype(np.float32), {
        "kind": "additive_noise", "snr_db": float(snr_db), "seed": int(seed),
    }


def apply_gain(waveform: np.ndarray, *, gain_db: float) -> tuple[np.ndarray, dict[str, object]]:
    samples = _waveform(waveform)
    if not np.isfinite(gain_db):
        raise ValueError("gain_db must be finite")
    gain = np.float32(10.0 ** (float(gain_db) / 20.0))
    return (samples * gain).astype(np.float32), {"kind": "gain", "gain_db": float(gain_db)}


def apply_reverb(
    waveform: np.ndarray, *, sample_rate: int, decay_seconds: float
) -> tuple[np.ndarray, dict[str, object]]:
    samples = _waveform(waveform)
    if sample_rate <= 0 or not np.isfinite(decay_seconds) or decay_seconds <= 0:
        raise ValueError("sample_rate and decay_seconds must be positive")
    if samples.size == 0:
        return samples.copy(), {"kind": "reverb", "decay_seconds": float(decay_seconds), "sample_rate": int(sample_rate)}
    length = max(1, int(round(sample_rate * decay_seconds)))
    time = np.arange(length, dtype=np.float64) / sample_rate
    impulse = np.exp(-6.907755278982137 * time / decay_seconds)
    impulse /= impulse.sum()
    impulse[0] += 0.5  # direct path plus a decaying, normalized late response
    output = np.convolve(samples, impulse.astype(np.float32), mode="full")[: samples.size]
    return output.astype(np.float32), {
        "kind": "reverb", "decay_seconds": float(decay_seconds), "sample_rate": int(sample_rate),
    }


def apply_rir_reverb(
    waveform: np.ndarray, *, sample_rate: int, rt60_seconds: float, drr_db: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, dict[str, object]]:
    """Synthetic room impulse response reverb with DRR control.

    Args:
        waveform: Input waveform (1-D float32 array).
        sample_rate: Sample rate in Hz (must be positive).
        rt60_seconds: Reverberation time in seconds (must be positive).
        drr_db: Direct-to-reverberant ratio in dB (default 0.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (output, record) where:
        - output: Reverberated waveform (float32, same shape as input)
        - record: Dict with kind, rt60_seconds, drr_db, seed, sample_rate, level_scale,
                  measured_drr_db, measured_rt60_seconds
    """
    samples = _waveform(waveform)

    # Validate parameters
    if sample_rate <= 0 or rt60_seconds <= 0:
        raise ValueError("sample_rate and rt60_seconds must be positive")

    # Handle empty/silent input
    if samples.size == 0:
        return samples.copy(), {
            "kind": "rir_reverb",
            "rt60_seconds": float(rt60_seconds),
            "drr_db": float(drr_db),
            "seed": int(seed),
            "sample_rate": int(sample_rate),
            "level_scale": 1.0,
            "measured_drr_db": float(drr_db),
            "measured_rt60_seconds": float(rt60_seconds),
        }

    # Construct the RIR with seeded randomness for the tail
    rir_length = math.ceil(rt60_seconds * sample_rate) + 1
    rng = np.random.default_rng(seed)
    tail = rng.standard_normal(rir_length - 1).astype(np.float64)

    # Apply exponential decay to the tail
    time_indices = np.arange(1, rir_length, dtype=np.float64) / sample_rate
    decay_envelope = np.exp(-np.log(1000.0) * time_indices / rt60_seconds)
    tail = tail * decay_envelope

    # Construct full RIR: direct path at h[0]=1.0, then decaying tail
    h_full = np.concatenate([[1.0], tail])

    # Adjust tail energy to achieve target DRR
    # DRR = 10*log10(direct_power / tail_power)
    direct_power = 1.0
    target_tail_power = direct_power / (10.0 ** (float(drr_db) / 10.0))
    tail_power = float(np.sum(h_full[1:] ** 2))

    if tail_power > 1e-10:  # Avoid division by zero
        tail_scale = np.sqrt(target_tail_power / tail_power)
        h_full[1:] *= tail_scale

    # Measure achieved DRR
    measured_tail_power = float(np.sum(h_full[1:] ** 2))
    if measured_tail_power > 1e-10:
        measured_drr_db = 10.0 * np.log10(h_full[0] ** 2 / measured_tail_power)
    else:
        measured_drr_db = float(drr_db)

    # Measure RT60 using Schroeder backward-integrated EDC method
    h_for_rt60 = h_full[: math.ceil(rt60_seconds * sample_rate) + 1]
    measured_rt60_seconds = _measure_rt60_schroeder(h_for_rt60, sample_rate)

    # Convolve input with RIR
    convolved = np.convolve(samples.astype(np.float64), h_full, mode="full")[: samples.size]

    # Scale to preserve input RMS
    input_rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    conv_rms = float(np.sqrt(np.mean(np.square(convolved))))

    if conv_rms > 1e-10 and input_rms > 0:
        level_scale = input_rms / conv_rms
    elif input_rms == 0:
        level_scale = 1.0
    else:
        level_scale = 1.0

    output = (convolved * level_scale).astype(np.float32)

    return output, {
        "kind": "rir_reverb",
        "rt60_seconds": float(rt60_seconds),
        "drr_db": float(drr_db),
        "seed": int(seed),
        "sample_rate": int(sample_rate),
        "level_scale": float(level_scale),
        "measured_drr_db": float(measured_drr_db),
        "measured_rt60_seconds": float(measured_rt60_seconds),
    }


def _measure_rt60_schroeder(h: np.ndarray, sr: int) -> float:
    """Measure RT60 using Schroeder backward-integrated EDC method.

    Args:
        h: Impulse response array.
        sr: Sample rate in Hz.

    Returns:
        Estimated RT60 in seconds, or 0.0 if insufficient decay data.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size == 0 or np.sum(h ** 2) == 0:
        return 0.0

    # Backward-integrated energy decay curve
    energy = np.cumsum((h ** 2)[::-1])[::-1]
    edc = 10.0 * np.log10(energy / energy[0] + 1e-12)

    # Find points in the -5 to -35 dB range for linear fit
    idx = np.where((edc <= -5.0) & (edc >= -35.0))[0]

    if len(idx) < 2:
        # Fallback: not enough points in linear region
        return 0.0

    # Linear fit: slope and intercept
    slope, _ = np.polyfit(idx / sr, edc[idx], 1)

    # Extrapolate to -60 dB
    if slope >= -1e-6:  # Avoid division issues with near-zero slope
        return 0.0

    return -60.0 / slope


def measure_snr_db(reference: np.ndarray, degraded: np.ndarray) -> float:
    """Measure signal-to-noise ratio in dB.

    SNR = 10*log10(P_signal / P_noise) where noise = degraded - reference.
    """
    ref = np.asarray(reference, dtype=np.float64)
    deg = np.asarray(degraded, dtype=np.float64)
    if ref.shape != deg.shape:
        raise ValueError("reference and degraded must have the same shape")
    signal_power = float(np.mean(np.square(ref)))
    if signal_power == 0.0:
        raise ValueError("reference signal power must be non-zero")
    noise = deg - ref
    noise_power = float(np.mean(np.square(noise)))
    if noise_power == 0.0:
        raise ValueError("noise power must be non-zero")
    return float(10.0 * math.log10(signal_power / noise_power))
