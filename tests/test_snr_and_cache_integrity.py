from __future__ import annotations

import math
import zipfile
import zlib

import numpy as np
import pytest

from src.audio import add_white_noise, measure_snr_db
from src.cache import CacheProvenance, CacheStore

# Every way a damaged .npz can surface: bad zip structure, bad CRC/deflate stream,
# an empty file, or the store's own checksum ValueError.
CORRUPTION_ERRORS = (ValueError, zipfile.BadZipFile, zlib.error, EOFError, OSError)


def _tone(samples: int = 16_000) -> np.ndarray:
    t = np.arange(samples, dtype=np.float64) / 16_000.0
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _provenance(identifier: str) -> CacheProvenance:
    return CacheProvenance(identifier=identifier, transform={"kind": "gain", "gain_db": 3.0}, source_checksum="abc123")


def _incompressible(seed: int) -> np.ndarray:
    # ~16 KB of Gaussian floats: the compressed payload dwarfs every other archive member.
    return np.random.default_rng(seed).standard_normal((32, 128)).astype(np.float32)


# --------------------------------------------------------------------------- measure_snr_db


def test_snr_of_unit_signal_with_point_one_offset_is_twenty_db() -> None:
    """Given reference ones and degraded = ones + 0.1, when SNR is measured, then it is 10*log10(1/0.01) = 20 dB."""
    reference = np.ones(1000, dtype=np.float64)
    assert measure_snr_db(reference, reference + 0.1) == pytest.approx(20.0, abs=1e-4)


def test_snr_of_alternating_signal_with_half_amplitude_error_is_10log10_of_four() -> None:
    """Given degraded = 1.5*reference, noise = 0.5*reference so P(ref)/P(noise) = 4."""
    reference = np.array([1.0, -1.0, 1.0, -1.0])
    assert measure_snr_db(reference, reference * 1.5) == pytest.approx(10.0 * math.log10(4.0), abs=1e-6)


def test_snr_of_all_zero_degraded_signal_is_zero_db() -> None:
    """Given degraded = 0, noise = -reference, so signal and noise power are equal."""
    reference = np.array([0.5, -0.25, 1.0, 0.0])
    assert measure_snr_db(reference, np.zeros_like(reference)) == pytest.approx(0.0, abs=1e-9)


def test_snr_treats_first_argument_as_the_signal_not_the_second() -> None:
    """Swapping the arguments changes the answer: ref=1, deg=2 -> 0 dB but ref=2, deg=1 -> 10*log10(4)."""
    ones, twos = np.ones(8), 2.0 * np.ones(8)
    assert measure_snr_db(ones, twos) == pytest.approx(0.0, abs=1e-9)
    assert measure_snr_db(twos, ones) == pytest.approx(10.0 * math.log10(4.0), abs=1e-6)


def test_snr_is_invariant_to_scaling_both_signals_and_returns_a_python_float() -> None:
    reference = np.array([1.0, -2.0, 0.5, 3.0])
    degraded = reference + np.array([0.1, -0.2, 0.05, 0.3])
    base = measure_snr_db(reference, degraded)
    assert isinstance(base, float)
    assert base == pytest.approx(20.0, abs=1e-6)  # noise is exactly 0.1 * reference
    assert measure_snr_db(7.0 * reference, 7.0 * degraded) == pytest.approx(base, abs=1e-9)


def test_snr_does_not_modify_its_inputs() -> None:
    reference = np.array([1.0, -1.0, 1.0])
    degraded = np.array([1.1, -0.9, 1.2])
    ref_before, deg_before = reference.copy(), degraded.copy()
    measure_snr_db(reference, degraded)
    assert np.array_equal(reference, ref_before)
    assert np.array_equal(degraded, deg_before)


def test_snr_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        measure_snr_db(np.ones(4), np.ones(5) * 1.1)


def test_snr_rejects_zero_power_reference() -> None:
    with pytest.raises(ValueError):
        measure_snr_db(np.zeros(4), np.full(4, 0.1))


def test_snr_rejects_identical_arrays_because_noise_power_is_zero() -> None:
    signal = np.array([0.25, -0.5, 0.75])
    with pytest.raises(ValueError):
        measure_snr_db(signal, signal.copy())


@pytest.mark.parametrize("seed", [1, 7, 123])
@pytest.mark.parametrize("snr_db", [0.0, 5.0, 15.0, 25.0])
def test_add_white_noise_realises_the_requested_snr_within_five_hundredths_of_a_db(snr_db: float, seed: int) -> None:
    """Given a 1 s tone, when noise is added at snr_db, then the realised SNR is within 0.05 dB of snr_db.

    A generator that scales unit Gaussian noise by the nominal RMS has ~0.05 dB standard error at
    16 000 samples, so this pins that the realised noise power (not just its expectation) is set.
    """
    waveform = _tone()
    noisy, _ = add_white_noise(waveform, sample_rate=16_000, snr_db=snr_db, seed=seed)
    assert measure_snr_db(waveform, noisy) == pytest.approx(snr_db, abs=0.05)


# --------------------------------------------------------------------------- cache corruption


@pytest.mark.parametrize("fraction", [0.3, 0.5, 0.7])
def test_flipping_one_byte_inside_the_stored_payload_makes_read_raise_and_leaves_other_entries_intact(tmp_path, fraction) -> None:
    """Given two stored entries, when one byte of entry A's file is flipped, then reading A raises
    (never returns data) and entry B still reads back bit-exact."""
    store = CacheStore(tmp_path)
    a_value, b_value = _incompressible(1), _incompressible(2)
    store.write("a", a_value, _provenance("a"))
    store.write("b", b_value, _provenance("b"))
    assert np.array_equal(store.read("a", _provenance("a")), a_value)  # pristine read succeeds: corruption is the only cause

    path = tmp_path / "a.npz"
    data = bytearray(path.read_bytes())
    data[int(len(data) * fraction)] ^= 0xFF
    path.write_bytes(bytes(data))

    with pytest.raises(CORRUPTION_ERRORS):
        store.read("a", _provenance("a"))
    assert np.array_equal(store.read("b", _provenance("b")), b_value)


@pytest.mark.parametrize("keep_fraction", [0.0, 0.1, 0.5, 0.95])
def test_truncated_cache_file_makes_read_raise_and_leaves_other_entries_intact(tmp_path, keep_fraction) -> None:
    store = CacheStore(tmp_path)
    b_value = _incompressible(4)
    store.write("a", _incompressible(3), _provenance("a"))
    store.write("b", b_value, _provenance("b"))

    path = tmp_path / "a.npz"
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * keep_fraction)])

    with pytest.raises(CORRUPTION_ERRORS):
        store.read("a", _provenance("a"))
    assert np.array_equal(store.read("b", _provenance("b")), b_value)


def test_payload_replaced_behind_the_stored_checksum_is_reported_as_a_checksum_mismatch(tmp_path) -> None:
    """Given a well-formed archive whose value was altered but whose provenance and checksum were kept,
    when it is read, then a ValueError naming the checksum is raised (the digest covers the payload)."""
    store = CacheStore(tmp_path)
    original = _incompressible(5)
    store.write("a", original, _provenance("a"))
    path = tmp_path / "a.npz"
    with np.load(path, allow_pickle=False) as stored:
        provenance, checksum = stored["provenance"], stored["checksum"]
    tampered = original.copy()
    tampered[0, 0] += 1.0
    np.savez_compressed(path, value=tampered, provenance=provenance, checksum=checksum)

    with pytest.raises(ValueError, match="checksum"):
        store.read("a", _provenance("a"))
