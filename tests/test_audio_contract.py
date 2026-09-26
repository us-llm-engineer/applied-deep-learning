from __future__ import annotations

import numpy as np

from src.audio import add_white_noise, apply_gain, apply_reverb


def test_acoustic_transforms_are_seeded_shape_stable_and_finite() -> None:
    waveform = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
    noise_a, record_a = add_white_noise(waveform, sample_rate=16_000, snr_db=8.0, seed=19)
    noise_b, record_b = add_white_noise(waveform, sample_rate=16_000, snr_db=8.0, seed=19)
    gain, gain_record = apply_gain(waveform, gain_db=6.0)
    reverb, reverb_record = apply_reverb(waveform, sample_rate=16_000, decay_seconds=0.08)

    for transformed in (noise_a, gain, reverb):
        assert transformed.shape == waveform.shape
        assert transformed.dtype == np.float32
        assert np.isfinite(transformed).all()
    assert np.array_equal(noise_a, noise_b)
    assert record_a == record_b
    assert record_a["kind"] == "additive_noise"
    assert record_a["snr_db"] == 8.0
    assert gain_record == {"kind": "gain", "gain_db": 6.0}
    assert reverb_record["kind"] == "reverb"


def test_gain_has_the_declared_multiplicative_effect_without_clipping() -> None:
    waveform = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
    transformed, _ = apply_gain(waveform, gain_db=20.0)

    assert np.allclose(transformed, waveform * 10.0, atol=1e-6)
