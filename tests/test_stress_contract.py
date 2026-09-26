from __future__ import annotations

import numpy as np
import pytest

from src.audio import measure_snr_db
from src.stress import STRESS_CONDITIONS, apply_stress

NOISE = {"noise_20": 20.0, "noise_10": 10.0, "noise_5": 5.0, "noise_0": 0.0}
RECORD_KEYS = {"name", "seed", "requested", "realised_snr_db", "peak", "clipped_fraction", "n"}
N = 16_000


def _batch(rows: int = 3, amplitude: float = 0.2) -> np.ndarray:
    """Distinct, non-clipping, non-periodic-aligned test signals (amplitude <= 0.2 by default)."""
    t = np.arange(N, dtype=np.float64) / N
    out = [amplitude * np.sin(2 * np.pi * (200.0 + 137.0 * i) * t + i) for i in range(rows)]
    return np.stack(out).astype(np.float32)


def _snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Independent oracle, computed from the definition."""
    c = clean.astype(np.float64)
    e = noisy.astype(np.float64) - c
    return float(10.0 * np.log10(np.sum(c**2) / np.sum(e**2)))


def _tail_ratio(out_row: np.ndarray, onset: int, skip: int = 160) -> float:
    return float(np.sum(out_row[onset + skip :].astype(np.float64) ** 2) / np.sum(out_row.astype(np.float64) ** 2))


def _impulse(amplitude: float = 0.5, onset: int = 100) -> np.ndarray:
    x = np.zeros((1, N), dtype=np.float32)
    x[0, onset] = amplitude
    return x


def test_condition_names_are_the_eleven_declared_in_order() -> None:
    """Given the module; When reading STRESS_CONDITIONS; Then it is exactly the specified sequence."""
    assert tuple(STRESS_CONDITIONS) == (
        "clean", "noise_20", "noise_10", "noise_5", "noise_0",
        "reverb_short", "reverb_mid", "reverb_long", "gain_-20", "gain_-10", "gain_+10",
    )


@pytest.mark.parametrize("name", list(STRESS_CONDITIONS))
def test_every_condition_returns_finite_float32_same_shape_within_unit_range(name: str) -> None:
    """Given a batch; When any condition is applied; Then output is float32, same shape, finite, in [-1, 1]."""
    x = _batch(3, amplitude=0.9)
    out, record = apply_stress(x, name, seed=3)
    assert out.dtype == np.float32 and out.shape == x.shape
    assert np.isfinite(out).all()
    assert out.min() >= -1.0 and out.max() <= 1.0
    assert RECORD_KEYS <= set(record)  # reverb records may carry extra severity keys


@pytest.mark.parametrize("name", list(STRESS_CONDITIONS))
def test_result_is_deterministic_for_name_and_seed(name: str) -> None:
    """Given the same (name, seed); When applied twice; Then outputs and records are identical."""
    x = _batch()
    a, ra = apply_stress(x, name, seed=11)
    b, rb = apply_stress(x, name, seed=11)
    assert np.array_equal(a, b)
    assert ra == rb


def test_noise_realisation_depends_on_seed() -> None:
    """Given noise_10; When the seed changes; Then the noise realisation changes."""
    x = _batch()
    a, _ = apply_stress(x, "noise_10", seed=1)
    b, _ = apply_stress(x, "noise_10", seed=2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("name", ["noise_10", "noise_0", "reverb_mid", "gain_-10"])
def test_row_result_is_independent_of_batch_composition(name: str) -> None:
    """Given a full batch; When row i is re-run alone with index_offset=i; Then it equals row i of the full result."""
    x = _batch(4)
    full, _ = apply_stress(x, name, seed=5)
    for i in range(4):
        single, _ = apply_stress(x[i : i + 1], name, seed=5, index_offset=i)
        assert np.array_equal(single[0], full[i]), f"{name}: row {i} depends on batch composition"


def test_noise_rows_are_distinct_realisations_per_example_index() -> None:
    """Given identical rows; When noise is added; Then rows get different noise (seeded per index, not shared)."""
    row = _batch(1)[0]
    x = np.stack([row, row, row])
    out, _ = apply_stress(x, "noise_10", seed=5)
    assert not np.array_equal(out[0], out[1])
    assert not np.array_equal(out[1], out[2])


def test_clean_returns_an_equal_independent_copy() -> None:
    """Given clean; When applied; Then values equal the input, memory is not shared, and the record has no requested value."""
    x = _batch()
    original = x.copy()
    out, record = apply_stress(x, "clean", seed=0)
    assert np.array_equal(out, original)
    assert not np.shares_memory(out, x)
    assert record["requested"] is None and record["realised_snr_db"] is None
    out[:] = 0.0
    assert np.array_equal(x, original)


@pytest.mark.parametrize(("name", "snr"), list(NOISE.items()))
def test_noise_condition_realises_requested_snr_per_row(name: str, snr: float) -> None:
    """Given unclipped 0.2-amplitude signals; When noise_K is applied; Then each row's SNR is within 0.05 dB of K."""
    x = _batch(3, amplitude=0.2)
    out, record = apply_stress(x, name, seed=7)
    assert np.abs(out).max() < 1.0, "test premise: signal must not be clipped"
    realised = [_snr_db(x[i], out[i]) for i in range(3)]
    for i, r in enumerate(realised):
        assert abs(r - snr) <= 0.05, f"{name} row {i}: realised {r:.3f} dB, wanted {snr}"
    assert record["requested"] == snr
    assert abs(record["realised_snr_db"] - float(np.mean(realised))) <= 0.05


def test_audio_snr_helper_agrees_with_independent_definition() -> None:
    """Given a noisy pair; When measure_snr_db(clean, noisy) is called; Then it matches the definition (guards the oracle itself)."""
    x = _batch(1)[0]
    out, _ = apply_stress(x[None, :], "noise_5", seed=2)
    assert abs(measure_snr_db(x, out[0]) - _snr_db(x, out[0])) < 1e-3


@pytest.mark.parametrize(("name", "gain_db"), [("gain_-20", -20.0), ("gain_-10", -10.0), ("gain_+10", 10.0)])
def test_gain_multiplies_then_clips(name: str, gain_db: float) -> None:
    """Given amplitude-0.5 input; When gain_G is applied; Then output equals clip(x * 10**(G/20))."""
    x = _batch(2, amplitude=0.5)
    out, record = apply_stress(x, name, seed=0)
    expected = np.clip(x.astype(np.float64) * 10.0 ** (gain_db / 20.0), -1.0, 1.0)
    assert np.allclose(out, expected, atol=1e-6)
    assert record["requested"] == gain_db
    assert record["realised_snr_db"] is None


def test_gain_plus_10_clips_and_gain_minus_20_does_not() -> None:
    """Given amplitude-0.5 input; When gain is applied; Then +10 dB reports clipping and -20 dB reports none, matching the output."""
    x = _batch(2, amplitude=0.5)
    loud, loud_rec = apply_stress(x, "gain_+10", seed=0)
    quiet, quiet_rec = apply_stress(x, "gain_-20", seed=0)
    assert loud_rec["clipped_fraction"] > 0.0
    assert quiet_rec["clipped_fraction"] == 0.0
    assert loud_rec["clipped_fraction"] == pytest.approx(float(np.mean(np.abs(loud) >= 1.0)))
    assert loud_rec["peak"] == pytest.approx(1.0)
    assert quiet_rec["peak"] == pytest.approx(float(np.abs(quiet).max()))
    assert quiet_rec["peak"] == pytest.approx(0.05, abs=1e-3)


def test_record_reports_seed_name_and_sample_count() -> None:
    """Given a (3, N) batch; When applied; Then the record carries name, seed and n equal to the number of examples."""
    x = _batch(3)
    _, record = apply_stress(x, "noise_20", seed=42)
    assert record["name"] == "noise_20"
    assert record["seed"] == 42
    assert record["n"] == 3


@pytest.mark.parametrize("name", ["reverb_short", "reverb_mid", "reverb_long"])
def test_reverb_reshapes_the_signal_at_matched_level(name: str) -> None:
    """Given an impulse; When reverb is applied; Then the output differs, carries late energy, and keeps the input level.

    Level matching isolates reverberation from gain: total energy stays equal to the input's.
    """
    x = _impulse()
    out, _ = apply_stress(x, name, seed=0)
    assert not np.allclose(out, x)
    assert np.sum(out.astype(np.float64) ** 2) == pytest.approx(np.sum(x.astype(np.float64) ** 2), rel=1e-3)
    assert _tail_ratio(out[0], 100) > 0.0


def test_longer_reverb_decay_gives_larger_tail_energy_ratio() -> None:
    """Given an impulse; When short, mid, long reverb are applied; Then tail energy ratio is strictly ordered short < mid < long."""
    x = _impulse()
    ratios = [_tail_ratio(apply_stress(x, n, seed=0)[0][0], 100) for n in ("reverb_short", "reverb_mid", "reverb_long")]
    assert ratios[0] < ratios[1] < ratios[2], f"tail ratios not ordered: {ratios}"


def test_unknown_name_is_rejected_and_input_is_unchanged() -> None:
    x = _batch()
    before = x.copy()
    with pytest.raises(ValueError):
        apply_stress(x, "noise_15", seed=0)
    assert np.array_equal(x, before)


@pytest.mark.parametrize("shape", [(N,), (2, 1, N)])
def test_wrong_ndim_is_rejected(shape: tuple[int, ...]) -> None:
    x = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError):
        apply_stress(x, "clean", seed=0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_is_rejected_and_input_is_unchanged(bad: float) -> None:
    x = _batch(2)
    x[1, 10] = bad
    before = x.copy()
    with pytest.raises(ValueError):
        apply_stress(x, "gain_-10", seed=0)
    assert np.array_equal(x, before, equal_nan=True)
