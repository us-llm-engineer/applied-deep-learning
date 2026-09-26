"""Deterministic stress conditions for robustness evaluation."""

from __future__ import annotations

import numpy as np

from src.audio import add_white_noise, apply_gain, apply_rir_reverb, measure_snr_db

STRESS_CONDITIONS = (
    "clean", "noise_20", "noise_10", "noise_5", "noise_0",
    "reverb_short", "reverb_mid", "reverb_long", "gain_-20", "gain_-10", "gain_+10",
)

SAMPLE_RATE = 16_000

# Reverb RT60 times in seconds
REVERB_RT60_TIMES = {
    "reverb_short": 0.2,
    "reverb_mid": 0.5,
    "reverb_long": 1.0,
}

NOISE_SNR_VALUES = {
    "noise_20": 20.0,
    "noise_10": 10.0,
    "noise_5": 5.0,
    "noise_0": 0.0,
}

GAIN_VALUES = {
    "gain_-20": -20.0,
    "gain_-10": -10.0,
    "gain_+10": 10.0,
}


def apply_stress(
    waveforms: np.ndarray,
    name: str,
    seed: int,
    index_offset: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a stress condition to a batch of waveforms.

    Args:
        waveforms: (num_examples, samples) float32 array
        name: condition name from STRESS_CONDITIONS
        seed: base random seed
        index_offset: offset for per-example seed computation

    Returns:
        (transformed_waveforms, record) where record includes:
        - name: condition name
        - seed: base seed used
        - requested: requested SNR/gain (None for reverb/clean)
        - realised_snr_db: realised SNR (None for gain/reverb/clean)
        - peak: max absolute value in output
        - clipped_fraction: fraction of samples at ±1.0
        - n: number of examples
        - decay: reverb decay times dict (only for reverb)
    """
    if name not in STRESS_CONDITIONS:
        raise ValueError(f"Unknown stress condition: {name}")

    # Validate input
    if waveforms.ndim != 2:
        raise ValueError(f"waveforms must be 2D, got shape {waveforms.shape}")
    if not np.isfinite(waveforms).all():
        raise ValueError("waveforms must contain only finite samples")

    num_examples, num_samples = waveforms.shape
    output = np.zeros_like(waveforms)

    if name == "clean":
        # Return independent copy
        output = waveforms.copy()
        record = {
            "name": name,
            "seed": seed,
            "requested": None,
            "realised_snr_db": None,
            "peak": float(np.abs(output).max()),
            "clipped_fraction": 0.0,
            "n": num_examples,
        }
    elif name in NOISE_SNR_VALUES:
        # Noise condition
        snr_db = NOISE_SNR_VALUES[name]
        realised_snrs = []
        for i in range(num_examples):
            per_example_seed = seed + index_offset + i
            noisy, _ = add_white_noise(
                waveforms[i], sample_rate=SAMPLE_RATE, snr_db=snr_db, seed=per_example_seed
            )
            output[i] = np.clip(noisy, -1.0, 1.0)
            # Measure realised SNR
            realised_snr = measure_snr_db(waveforms[i], output[i])
            realised_snrs.append(realised_snr)

        clipped_samples = np.sum(np.abs(output) >= 1.0)
        total_samples = num_examples * num_samples
        record = {
            "name": name,
            "seed": seed,
            "requested": snr_db,
            "realised_snr_db": float(np.mean(realised_snrs)),
            "peak": float(np.abs(output).max()),
            "clipped_fraction": float(clipped_samples / total_samples),
            "n": num_examples,
        }
    elif name in REVERB_RT60_TIMES:
        # Reverb condition
        rt60_seconds = REVERB_RT60_TIMES[name]
        measured_rt60s = []
        for i in range(num_examples):
            # Per-example seed
            per_example_seed = seed + index_offset + i
            reverbed, rev_record = apply_rir_reverb(
                waveforms[i], sample_rate=SAMPLE_RATE, rt60_seconds=rt60_seconds, drr_db=0.0, seed=per_example_seed
            )
            output[i] = np.clip(reverbed, -1.0, 1.0)
            measured_rt60s.append(rev_record["measured_rt60_seconds"])

        clipped_samples = np.sum(np.abs(output) >= 1.0)
        total_samples = num_examples * num_samples
        record = {
            "name": name,
            "seed": seed,
            "requested": None,
            "realised_snr_db": None,
            "peak": float(np.abs(output).max()),
            "clipped_fraction": float(clipped_samples / total_samples),
            "n": num_examples,
            "decay": rt60_seconds,
            "rt60_seconds": rt60_seconds,
            "drr_db": 0.0,
            "mean_measured_rt60_seconds": float(np.mean(measured_rt60s)),
        }
    elif name in GAIN_VALUES:
        # Gain condition
        gain_db = GAIN_VALUES[name]
        for i in range(num_examples):
            gained, _ = apply_gain(waveforms[i], gain_db=gain_db)
            output[i] = np.clip(gained, -1.0, 1.0)

        clipped_samples = np.sum(np.abs(output) >= 1.0)
        total_samples = num_examples * num_samples
        record = {
            "name": name,
            "seed": seed,
            "requested": gain_db,
            "realised_snr_db": None,
            "peak": float(np.abs(output).max()),
            "clipped_fraction": float(clipped_samples / total_samples),
            "n": num_examples,
        }
    else:
        raise ValueError(f"Unknown stress condition: {name}")

    return output.astype(np.float32), record
