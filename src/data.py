"""Manifest records and reproducible, label-stratified dataset partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import math

import numpy as np

from .config import DEFAULT_SPLIT_RATIOS, SPLIT_NAMES


@dataclass(frozen=True)
class ManifestRecord:
    identifier: str
    label: str
    path: str

    def __post_init__(self) -> None:
        if not self.identifier or not self.label or not self.path:
            raise ValueError("identifier, label, and path must be non-empty")


def _counts_for_class(size: int, ratios: Sequence[float]) -> tuple[int, ...]:
    exact = [size * ratio for ratio in ratios]
    counts = [math.floor(part) for part in exact]
    remainder = size - sum(counts)
    order = sorted(range(len(ratios)), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def stratified_split(
    records: Iterable[ManifestRecord],
    seed: int,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, list[ManifestRecord]]:
    """Split each label independently; every input record appears exactly once."""
    rows = list(records)
    if len(ratios) != 3 or any(not np.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("ratios must contain three finite, non-negative values")
    if not np.isclose(sum(ratios), 1.0, rtol=0, atol=1e-10):
        raise ValueError("ratios must sum to 1")
    identifiers = [row.identifier for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("record identifiers must be unique")

    by_label: dict[str, list[ManifestRecord]] = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)

    result = {name: [] for name in SPLIT_NAMES}
    # Sorted labels and per-label RNG streams make results independent of mapping order.
    seed_sequence = np.random.SeedSequence(seed)
    label_seeds = seed_sequence.spawn(len(by_label))
    for label, child_seed in zip(sorted(by_label), label_seeds):
        group = by_label[label]
        permutation = np.random.default_rng(child_seed).permutation(len(group))
        shuffled = [group[int(index)] for index in permutation]
        counts = _counts_for_class(len(group), ratios)
        start = 0
        for name, count in zip(SPLIT_NAMES, counts):
            result[name].extend(shuffled[start : start + count])
            start += count
    return result
