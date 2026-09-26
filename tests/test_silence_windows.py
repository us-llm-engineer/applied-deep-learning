"""Test deterministic silence window generation."""

from __future__ import annotations

import numpy as np
import pytest

from src.dataset import silence_windows


def _noise_array(seed: int, size: int = 160000) -> np.ndarray:
    """Generate a noise array with a given seed (default 10 seconds at 16 kHz)."""
    return np.random.default_rng(seed).standard_normal(size).astype(np.float32)


def test_silence_windows_returns_correct_length() -> None:
    """Given n and a list of noise arrays, when silence_windows is called,
    then it returns n windows, each with 16000 samples."""
    noise_arrays = [_noise_array(i, 32000) for i in range(3)]
    windows = silence_windows(noise_arrays, n=5, seed=42)
    assert windows.shape == (5, 16000)
    assert windows.dtype == np.float32


def test_silence_windows_is_deterministic() -> None:
    """Given the same seed, silence_windows returns identical windows."""
    noise_arrays = [_noise_array(i, 48000) for i in range(4)]
    result1 = silence_windows(noise_arrays, n=3, seed=17)
    result2 = silence_windows(noise_arrays, n=3, seed=17)
    assert np.array_equal(result1, result2)


def test_silence_windows_varies_with_seed() -> None:
    """Given different seeds, silence_windows returns different windows."""
    noise_arrays = [_noise_array(i, 48000) for i in range(4)]
    result1 = silence_windows(noise_arrays, n=3, seed=17)
    result2 = silence_windows(noise_arrays, n=3, seed=18)
    assert not np.array_equal(result1, result2)


def test_silence_windows_respects_train_split() -> None:
    """Given split='train', all windows are drawn from the first 70% of files."""
    # Create a file with a distinguishable pattern: first 70% is high, rest is low
    noise = np.concatenate([
        np.full(112000, 1.0, dtype=np.float32),  # first 70% high
        np.full(48000, -1.0, dtype=np.float32),  # remaining 30% low
    ])
    windows = silence_windows([noise], n=20, seed=42, split="train")
    # Windows from train split (first 70%) should be mostly high
    means = np.mean(windows, axis=1)
    assert np.all(means > 0.5), "train split windows should be from high region"


def test_silence_windows_respects_cal_split() -> None:
    """Given split='cal', all windows are drawn from the 70-85% region."""
    noise = np.concatenate([
        np.full(112000, -1.0, dtype=np.float32),  # first 70%
        np.full(24000, 1.0, dtype=np.float32),    # next 15% (cal)
        np.full(24000, -1.0, dtype=np.float32),   # last 15%
    ])
    windows = silence_windows([noise], n=20, seed=42, split="cal")
    # Windows from cal split should be mostly high
    means = np.mean(windows, axis=1)
    assert np.all(means > 0.5), "cal split windows should be from high region"


def test_silence_windows_respects_test_split() -> None:
    """Given split='test', all windows are drawn from the last 15% region."""
    noise = np.concatenate([
        np.full(112000, -1.0, dtype=np.float32),  # first 70%
        np.full(24000, -1.0, dtype=np.float32),   # next 15% (cal)
        np.full(24000, 1.0, dtype=np.float32),    # last 15% (test)
    ])
    windows = silence_windows([noise], n=20, seed=42, split="test")
    # Windows from test split should be mostly high
    means = np.mean(windows, axis=1)
    assert np.all(means > 0.5), "test split windows should be from high region"


def test_silence_windows_rejects_invalid_split() -> None:
    """Given an invalid split name, silence_windows raises ValueError."""
    noise_arrays = [_noise_array(0, 32000)]
    with pytest.raises(ValueError, match="split must be"):
        silence_windows(noise_arrays, n=5, seed=42, split="invalid")


def test_silence_windows_rejects_invalid_n() -> None:
    """Given n <= 0, silence_windows raises ValueError."""
    noise_arrays = [_noise_array(0, 32000)]
    with pytest.raises(ValueError, match="n must be positive"):
        silence_windows(noise_arrays, n=0, seed=42)
    with pytest.raises(ValueError, match="n must be positive"):
        silence_windows(noise_arrays, n=-1, seed=42)
