from __future__ import annotations

import numpy as np
import pytest

from src.cache import CacheProvenance, CacheStore
from src.calibration import crc_threshold, prediction_sets, temperature_scale


def test_temperature_scaling_preserves_argmax_and_prediction_sets_expand_monotonically() -> None:
    logits = np.array([[4.0, 0.0], [0.0, 4.0], [2.0, 1.0]])
    labels = np.array([0, 1, 0])
    temperature = temperature_scale(logits, labels)
    low = prediction_sets(logits, temperature=temperature, threshold=0.55)
    high = prediction_sets(logits, temperature=temperature, threshold=0.15)

    assert temperature > 0
    assert np.array_equal(np.argmax(logits, axis=1), np.argmax(logits / temperature, axis=1))
    assert all(low_set.issubset(high_set) for low_set, high_set in zip(low, high, strict=True))


def test_crc_threshold_is_conservative_for_indicator_miscoverage() -> None:
    true_label_scores = np.array([0.92, 0.83, 0.74, 0.66, 0.51])
    threshold = crc_threshold(true_label_scores, alpha=0.25)
    empirical_miscoverage = np.mean(true_label_scores < threshold)

    assert threshold in true_label_scores
    assert empirical_miscoverage <= (0.25 * (len(true_label_scores) + 1) - 1) / len(true_label_scores)


def test_cache_round_trip_requires_matching_provenance(tmp_path) -> None:
    store = CacheStore(tmp_path)
    provenance = CacheProvenance(
        identifier="left-7",
        transform={"kind": "gain", "gain_db": 3.0},
        source_checksum="abc123",
    )
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    store.write("left-7", value, provenance)

    restored = store.read("left-7", provenance)
    assert np.array_equal(restored, value)
    with pytest.raises(ValueError, match="provenance|checksum"):
        store.read("left-7", CacheProvenance("left-7", {"kind": "gain", "gain_db": 4.0}, "abc123"))
