"""Compact, deterministic audio manifests for the speech command study."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


STUDY_LABELS = (
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "unknown", "silence",
)

AUXILIARY_WORDS = (
    "backward", "bed", "bird", "cat", "dog", "eight", "five", "follow", "forward", "four",
    "happy", "house", "learn", "marvin", "nine", "one", "seven", "sheila", "six", "three",
    "tree", "two", "visual", "wow", "zero",
)


@dataclass(frozen=True)
class AudioRecord:
    identifier: str
    label: str
    path: Path
    checksum: str
    sample_rate: int
    num_frames: int

    def __post_init__(self) -> None:
        if not self.identifier or not self.label or not self.checksum:
            raise ValueError("identifier, label, and checksum must be non-empty")
        if self.sample_rate <= 0 or self.num_frames < 0:
            raise ValueError("sample_rate must be positive and num_frames non-negative")
        object.__setattr__(self, "path", Path(self.path))


def _record_key(record: AudioRecord) -> tuple[str, str, str, str, int, int]:
    return (record.identifier, record.label, str(record.path), record.checksum,
            int(record.sample_rate), int(record.num_frames))


def build_compact_manifest(
    records: Iterable[AudioRecord], *, cap_per_label: int = 600, seed: int = 17
) -> list[AudioRecord]:
    """Select at most ``cap_per_label`` rows per study label reproducibly.

    Sorting before sampling makes selection and output independent of input order.
    Labels outside the declared 12-class population are omitted.
    """
    if isinstance(cap_per_label, bool) or not isinstance(cap_per_label, int) or cap_per_label < 0:
        raise ValueError("cap_per_label must be a non-negative integer")
    groups: dict[str, list[AudioRecord]] = defaultdict(list)
    ids: set[str] = set()
    for row in records:
        if not isinstance(row, AudioRecord):
            raise TypeError("records must contain AudioRecord values")
        if row.identifier in ids:
            raise ValueError(f"duplicate identifier: {row.identifier}")
        ids.add(row.identifier)
        if row.label in STUDY_LABELS:
            groups[row.label].append(row)

    result: list[AudioRecord] = []
    root = np.random.SeedSequence(int(seed))
    child_seeds = root.spawn(len(STUDY_LABELS))
    for label, child in zip(STUDY_LABELS, child_seeds):
        group = sorted(groups[label], key=_record_key)
        if len(group) > cap_per_label:
            indices = np.random.default_rng(child).choice(len(group), cap_per_label, replace=False)
            group = [group[int(i)] for i in sorted(indices)]
        result.extend(group)
    return result


def manifest_checksum(records: Iterable[AudioRecord]) -> str:
    """Return a stable SHA-256 digest of all manifest fields, independent of row order."""
    rows = sorted((_record_key(row) for row in records))
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def find_duplicate_checksums(records: Iterable[AudioRecord]) -> dict[str, tuple[str, ...]]:
    """Map each repeated content checksum to all colliding record identifiers."""
    by_checksum: dict[str, list[str]] = defaultdict(list)
    for row in records:
        by_checksum[row.checksum].append(row.identifier)
    return {
        checksum: tuple(sorted(identifiers))
        for checksum, identifiers in sorted(by_checksum.items())
        if len(identifiers) > 1
    }


def holdout_unknown_words(
    words: Iterable[str], n_holdout: int, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split words into train and holdout sets deterministically.

    Returns (train, heldout) where heldout contains n_holdout words.
    The split is independent of input order.
    """
    word_list = sorted(set(words))
    total = len(word_list)
    if len(set(words)) != len(list(words)):
        raise ValueError("words must not contain duplicates")
    if n_holdout <= 0 or n_holdout >= total:
        raise ValueError(f"n_holdout must be in range [1, {total-1}]")
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    holdout_indices = set(indices[:n_holdout])
    train = tuple(word_list[i] for i in range(total) if i not in holdout_indices)
    heldout = tuple(word_list[i] for i in range(total) if i in holdout_indices)
    return (train, heldout)


def silence_windows(
    noise_arrays: list[np.ndarray], n: int, seed: int, split: str = "train"
) -> np.ndarray:
    """Generate n deterministic 16000-sample windows from noise arrays.

    Each file is split by time: first 70% (train), next 15% (cal), last 15% (test).
    Window offsets are seeded and deterministic.
    """
    if split not in {"train", "cal", "test"}:
        raise ValueError("split must be one of 'train', 'cal', 'test'")
    if n <= 0:
        raise ValueError("n must be positive")

    # Map split to region
    split_ranges = {"train": (0.0, 0.7), "cal": (0.7, 0.85), "test": (0.85, 1.0)}
    start_frac, end_frac = split_ranges[split]

    rng = np.random.default_rng(seed)
    windows = []

    for _ in range(n):
        # Select a file uniformly at random
        file_idx = rng.integers(0, len(noise_arrays))
        noise = noise_arrays[file_idx]

        # Compute region boundaries
        length = len(noise)
        region_start = int(start_frac * length)
        region_end = int(end_frac * length)

        # Ensure we can fit a 16000-sample window
        max_offset = max(0, region_end - 16000)
        if region_start >= region_end or max_offset < region_start:
            # Fallback: use the region as-is
            offset = region_start
        else:
            offset = rng.integers(region_start, max_offset + 1)

        window = noise[offset : offset + 16000]
        # Pad if necessary
        if len(window) < 16000:
            window = np.pad(window, (0, 16000 - len(window)), mode="constant")
        windows.append(window[:16000])

    return np.array(windows, dtype=np.float32)

