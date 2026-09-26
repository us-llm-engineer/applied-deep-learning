"""Contract for `src.audio.apply_rir_reverb` (synthetic room impulse response reverb).

Width: solitary (pure function, no collaborators). The module is accessed as an attribute so
that the legacy `apply_reverb` test still runs while the new function is absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import src.audio as audio

SR = 8_000  # small rate keeps h short; the RT60 tests use 16 kHz as specified.


def _rir(rt60: float, sr: int = SR, drr_db: float = 0.0, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Recover h from a unit impulse: output == level_scale * h[:n], so h == output / level_scale."""
    n = math.ceil(rt60 * sr) + 1 + 50
    impulse = np.zeros(n, dtype=np.float32)
    impulse[0] = 1.0
    out, record = audio.apply_rir_reverb(impulse, sample_rate=sr, rt60_seconds=rt60, drr_db=drr_db, seed=seed)
    return out.astype(np.float64) / record["level_scale"], record


def _schroeder_rt60(h: np.ndarray, sr: int) -> float:
    """Independent oracle: backward-integrated EDC, -5..-35 dB fit extrapolated to -60 dB."""
    energy = np.cumsum((h**2)[::-1])[::-1]
    edc = 10.0 * np.log10(energy / energy[0])
    idx = np.where((edc <= -5.0) & (edc >= -35.0))[0]
    slope, _ = np.polyfit(idx / sr, edc[idx], 1)
    return -60.0 / slope


def _signal(n: int = 4_000, amplitude: float = 0.3, seed: int = 5) -> np.ndarray:
    return (amplitude * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_same_seed_gives_identical_output_and_record() -> None:
    """Given equal arguments incl. seed; When applied twice; Then output and record are identical."""
    x = _signal()
    a, ra = audio.apply_rir_reverb(x, sample_rate=SR, rt60_seconds=0.1, drr_db=3.0, seed=4)
    b, rb = audio.apply_rir_reverb(x, sample_rate=SR, rt60_seconds=0.1, drr_db=3.0, seed=4)
    assert np.array_equal(a, b) and ra == rb, "same seed must reproduce the same RIR"


def test_different_seed_changes_the_output() -> None:
    """Given equal arguments except seed; When applied; Then the outputs differ (tail is seeded noise)."""
    x = _signal()
    a, _ = audio.apply_rir_reverb(x, sample_rate=SR, rt60_seconds=0.1, seed=1)
    b, _ = audio.apply_rir_reverb(x, sample_rate=SR, rt60_seconds=0.1, seed=2)
    assert not np.array_equal(a, b), "seed is ignored: outputs identical for seeds 1 and 2"


def test_record_reports_the_requested_parameters() -> None:
    """Given a call; When reading the record; Then kind and echoed parameters match the request."""
    _, rec = audio.apply_rir_reverb(_signal(), sample_rate=SR, rt60_seconds=0.15, drr_db=-2.5, seed=9)
    assert rec["kind"] == "rir_reverb"
    assert rec["rt60_seconds"] == 0.15
    assert rec["drr_db"] == -2.5
    assert rec["seed"] == 9
    assert rec["sample_rate"] == SR


@pytest.mark.parametrize("drr_db", [-10.0, -3.0, 0.0, 6.5, 20.0])
def test_requested_drr_is_realised_exactly_in_the_rir(drr_db: float) -> None:
    """Given drr_db; When building the RIR; Then 10*log10(1/sum(tail^2)) equals it to 1e-6 dB."""
    h, rec = _rir(0.1, drr_db=drr_db, seed=3)
    assert abs(rec["measured_drr_db"] - drr_db) < 1e-6, f"record DRR {rec['measured_drr_db']} != {drr_db}"
    # Independent check on the RIR recovered through the output (float32 limits precision).
    recovered = 10.0 * np.log10(h[0] ** 2 / np.sum(h[1:] ** 2))
    assert abs(recovered - drr_db) < 1e-3, f"RIR recovered from output has DRR {recovered}, wanted {drr_db}"


def test_unit_impulse_output_is_a_scaled_copy_of_the_rir() -> None:
    """Given a unit impulse; When reverberated; Then output == level_scale * h, zero beyond len(h)."""
    rt60 = 0.1
    length = math.ceil(rt60 * SR) + 1
    h, rec = _rir(rt60, drr_db=0.0, seed=7)
    assert abs(h[0] - 1.0) < 1e-5, "direct path h[0] must be 1"
    assert np.all(h[length:] == 0.0), "output is nonzero beyond len(h) = ceil(rt60*sr)+1"
    assert np.any(h[1:length] != 0.0), "tail is missing"
    # Same-seed re-run at a different input length reproduces the same h prefix (proportionality).
    impulse = np.zeros(length + 500, dtype=np.float32)
    impulse[0] = 1.0
    out, rec2 = audio.apply_rir_reverb(impulse, sample_rate=SR, rt60_seconds=rt60, drr_db=0.0, seed=7)
    assert np.allclose(out / np.float32(rec2["level_scale"]), h[:length].tolist() + [0.0] * 500, atol=1e-5)


def test_output_rms_equals_input_rms_and_level_scale_is_the_applied_factor() -> None:
    """Given a signal; When reverberated; Then RMS(out)==RMS(in) and level_scale == RMS(in)/RMS(conv)."""
    x = _signal()
    out, rec = audio.apply_rir_reverb(x, sample_rate=SR, rt60_seconds=0.1, drr_db=0.0, seed=2)
    assert _rms(out) == pytest.approx(_rms(x), rel=1e-5)
    h, _ = _rir(0.1, drr_db=0.0, seed=2)
    h = h[: math.ceil(0.1 * SR) + 1]
    conv = np.convolve(x.astype(np.float64), h)[: x.size]
    assert rec["level_scale"] == pytest.approx(_rms(x) / _rms(conv), rel=1e-4)


def test_output_is_float32_finite_same_shape_and_not_clipped() -> None:
    """Given a loud input (peak >> 1); When reverberated; Then dtype/shape hold and nothing is clipped."""
    x = (10.0 * np.sin(2 * np.pi * 300.0 * np.arange(4_000) / SR)).astype(np.float32)
    out, _ = audio.apply_rir_reverb(x, sample_rate=SR, rt60_seconds=0.1, seed=1)
    assert out.dtype == np.float32 and out.shape == x.shape and np.isfinite(out).all()
    assert np.abs(out).max() > 1.0, "output was clipped to [-1, 1]; clipping must not happen here"
    assert _rms(out) == pytest.approx(_rms(x), rel=1e-5)


@pytest.mark.parametrize("rt60", [0.2, 0.5, 1.0])
def test_measured_rt60_is_within_20_percent_of_requested(rt60: float) -> None:
    """Given rt60 at 16 kHz; When building the RIR; Then Schroeder RT60 (record and independent) is within 20%."""
    h, rec = _rir(rt60, sr=16_000, drr_db=0.0, seed=11)
    h = h[: math.ceil(rt60 * 16_000) + 1]
    assert rec["measured_rt60_seconds"] == pytest.approx(rt60, rel=0.2)
    assert _schroeder_rt60(h, 16_000) == pytest.approx(rt60, rel=0.2), "RIR actually returned does not decay at rt60"


def test_rt60_is_independent_of_drr() -> None:
    """Given the same rt60 and seed; When drr_db changes; Then the decay time stays within 20% of rt60."""
    _, quiet_tail = _rir(0.5, sr=16_000, drr_db=15.0, seed=11)
    _, loud_tail = _rir(0.5, sr=16_000, drr_db=-10.0, seed=11)
    assert quiet_tail["measured_rt60_seconds"] == pytest.approx(0.5, rel=0.2)
    assert loud_tail["measured_rt60_seconds"] == pytest.approx(0.5, rel=0.2)


def test_at_equal_drr_a_longer_rt60_keeps_more_tail_energy_late_in_absolute_time() -> None:
    """Given DRR fixed at 0 dB; When rt60 grows 0.2 -> 0.5 -> 1.0; Then the energy after 100 ms grows.

    Note: the requested 'last-quarter of h' fraction is NOT used; the envelope is defined relative to
    rt60 and len(h) is proportional to rt60, so that fraction is constant (~3e-5) by construction.
    """
    sr, late = 16_000, 1_600
    fractions = []
    for rt60 in (0.2, 0.5, 1.0):
        h, _ = _rir(rt60, sr=sr, drr_db=0.0, seed=13)
        h = h[: math.ceil(rt60 * sr) + 1]
        fractions.append(float(np.sum(h[late:] ** 2) / np.sum(h**2)))
    assert fractions[0] < fractions[1] < fractions[2], f"late-energy fractions not increasing: {fractions}"


def test_silent_input_returns_zeros_with_unit_level_scale() -> None:
    """Given all-zero input; When reverberated; Then zeros come back and level_scale is exactly 1.0."""
    out, rec = audio.apply_rir_reverb(np.zeros(1_000, dtype=np.float32), sample_rate=SR, rt60_seconds=0.1)
    assert out.shape == (1_000,) and out.dtype == np.float32
    assert np.all(out == 0.0)
    assert rec["level_scale"] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0, "rt60_seconds": 0.2},
        {"sample_rate": -16_000, "rt60_seconds": 0.2},
        {"sample_rate": SR, "rt60_seconds": 0.0},
        {"sample_rate": SR, "rt60_seconds": -0.5},
    ],
    ids=["sr_zero", "sr_negative", "rt60_zero", "rt60_negative"],
)
def test_non_positive_sample_rate_or_rt60_is_rejected_without_touching_input(kwargs: dict) -> None:
    """Given exactly one non-positive parameter; When applied; Then ValueError and input is unchanged."""
    x = _signal()
    before = x.copy()
    with pytest.raises(ValueError):
        audio.apply_rir_reverb(x, **kwargs)
    assert np.array_equal(x, before)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_non_finite_input_is_rejected_without_touching_input(bad: float) -> None:
    """Given a waveform with one non-finite sample; When applied; Then ValueError and input is unchanged."""
    x = _signal()
    x[10] = bad
    before = x.copy()
    with pytest.raises(ValueError):
        audio.apply_rir_reverb(x, sample_rate=SR, rt60_seconds=0.1)
    assert np.array_equal(x, before, equal_nan=True)


def test_two_dimensional_input_is_rejected_without_touching_input() -> None:
    """Given a 2-D array of finite values; When applied; Then ValueError and input is unchanged."""
    x = np.ones((2, 500), dtype=np.float32)
    with pytest.raises(ValueError):
        audio.apply_rir_reverb(x, sample_rate=SR, rt60_seconds=0.1)
    assert np.array_equal(x, np.ones((2, 500), dtype=np.float32))


def test_legacy_apply_reverb_is_unchanged() -> None:
    """Given the old exponential-kernel reverb; When fed an impulse; Then output is the documented kernel.

    Oracle: kernel = exp(-ln(1000)*t/decay) normalised to unit sum, plus 0.5 on the direct tap.
    """
    decay, sr = 0.01, SR
    length = int(round(sr * decay))
    kernel = np.exp(-6.907755278982137 * (np.arange(length) / sr) / decay)
    kernel /= kernel.sum()
    kernel[0] += 0.5
    impulse = np.zeros(200, dtype=np.float32)
    impulse[0] = 1.0
    out, rec = audio.apply_reverb(impulse, sample_rate=sr, decay_seconds=decay)
    assert np.allclose(out[:length], kernel, atol=1e-6)
    assert np.all(out[length:] == 0.0)
    assert rec == {"kind": "reverb", "decay_seconds": decay, "sample_rate": sr}
