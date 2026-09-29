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
    noise_arrays: list[np.ndarray],
    n: int,
    seed: int,
    split: str = "train",
    *,
    return_provenance: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[tuple[int, int]]]:
    """Generate n deterministic 16000-sample windows from noise arrays.

    Each file is split by time: first 70% (train), next 15% (cal), last 15% (test).
    Window offsets are seeded and deterministic.

    Args:
        noise_arrays: List of noise audio arrays.
        n: Number of windows to generate.
        seed: Random seed for reproducibility.
        split: One of "train", "cal", "test".
        return_provenance: If True, return (windows, provenance) tuple where provenance
            is a list of (file_idx, offset) pairs; if False, return only windows array.

    Returns:
        If return_provenance=False: (n, 16000) float32 ndarray of windows.
        If return_provenance=True: (windows, provenance) where provenance is list of (file_idx, offset).

    Invariant: For every returned window with provenance (file_idx, offset), the window satisfies:
        region_start <= offset and offset + 16000 <= region_end
    where region_start, region_end are computed from the split's fractional boundaries.
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
    provenance = [] if return_provenance else None

    for _ in range(n):
        # Find a file whose region can accommodate a 16000-sample window
        file_idx = None
        offset = None
        found = False

        # Keep trying to find a feasible file, with a generous attempt limit
        max_attempts = len(noise_arrays) * 100
        for attempt in range(max_attempts):
            candidate_idx = rng.integers(0, len(noise_arrays))
            candidate = noise_arrays[candidate_idx]
            candidate_length = len(candidate)
            candidate_region_start = int(start_frac * candidate_length)
            candidate_region_end = int(end_frac * candidate_length)

            if candidate_region_end - candidate_region_start >= 16000:
                file_idx = candidate_idx
                max_offset = max(0, candidate_region_end - 16000)
                offset = rng.integers(candidate_region_start, max_offset + 1)
                found = True
                break

        if not found:
            raise ValueError(f"No file in noise_arrays has a {split} region large enough for a 16000-sample window")

        # Extract window
        window = noise_arrays[file_idx][offset : offset + 16000]
        assert len(window) == 16000, f"Window extraction failed: got {len(window)} samples instead of 16000"

        windows.append(window)
        if return_provenance:
            provenance.append((int(file_idx), int(offset)))

    result_array = np.array(windows, dtype=np.float32)

    if return_provenance:
        return result_array, provenance
    else:
        return result_array

