"""Behavioural contract for speaker-aware bundle auditing."""
from __future__ import annotations

import numpy as np
import pytest

from src.bundle import (
    DatasetBundle,
    SplitArrays,
    audit_bundle,
    speaker_of,
    speaker_overlap,
    speakers_per_split,
)
import src.pipeline as pipeline


def _split(*ids: str) -> SplitArrays:
    """Build a deliberately tiny split without relying on dataset fixtures."""
    return SplitArrays(
        waveforms=np.zeros((len(ids), 1), dtype=np.int16),
        y=np.zeros(len(ids), dtype=np.int64),
        ids=ids,
        words=tuple("yes" for _ in ids),
    )


def _bundle(**splits: SplitArrays) -> DatasetBundle:
    return DatasetBundle(
        labels=("yes", "silence"),
        splits=splits,
        train_unknown_words=(),
        heldout_words=(),
        manifest_sha256="manual-test-bundle",
        shortfalls={},
    )


def test_speaker_of_extracts_only_a_real_source_stem_and_ignores_prefixes() -> None:
    """A label/directory may surround a source stem, but cannot become its speaker."""
    assert speaker_of("0a7c2a8d_nohash_3") == "0a7c2a8d"
    assert speaker_of("yes/0a7c2a8d_nohash_3") == "0a7c2a8d"
    assert speaker_of("archive/unknown/fedcba98_nohash_17") == "fedcba98"

    # These are intentionally plausible identifiers, but not source clips.
    assert speaker_of("silence/train/0") is None
    assert speaker_of("yes_000") is None
    assert speaker_of("yes/0a7c2a8d_nohash_not_a_take") is None
    assert speaker_of("yes/3") is None


def test_speakers_per_split_counts_distinct_real_speakers_including_ood() -> None:
    bundle = _bundle(
        train=_split("yes/0a7c2a8d_nohash_0", "no/0a7c2a8d_nohash_1", "silence/train/0"),
        cal=_split("up/fedcba98_nohash_0", "malformed_12"),
        test=_split("down/11111111_nohash_0"),
        ood=_split("tree/22222222_nohash_0", "tree/22222222_nohash_1", "silence/ood/0"),
    )

    assert speakers_per_split(bundle) == {"train": 1, "cal": 1, "test": 1, "ood": 1}


def test_speaker_overlap_uses_every_test_clip_as_its_denominator() -> None:
    """Silence and malformed test IDs dilute the leakage fraction rather than disappearing."""
    bundle = _bundle(
        train=_split("yes/0a7c2a8d_nohash_0", "no/11111111_nohash_0"),
        cal=_split("up/fedcba98_nohash_0"),
        test=_split(
            "yes/0a7c2a8d_nohash_9",       # seen in train
            "up/fedcba98_nohash_4",        # seen in cal
            "down/22222222_nohash_0",      # unseen source speaker
            "silence/test/0",               # not a speaker, still in denominator
            "yes_004",                      # malformed, still in denominator
        ),
    )

    assert speaker_overlap(bundle) == {
        "test_seen_in_train": pytest.approx(1 / 5),
        "test_seen_in_cal": pytest.approx(1 / 5),
    }


def test_speaker_overlap_is_zero_for_empty_or_missing_test_and_missing_references() -> None:
    assert speaker_overlap(_bundle(train=_split("yes/0a7c2a8d_nohash_0"))) == {
        "test_seen_in_train": 0.0,
        "test_seen_in_cal": 0.0,
    }
    assert speaker_overlap(_bundle(test=_split())) == {
        "test_seen_in_train": 0.0,
        "test_seen_in_cal": 0.0,
    }
    assert speaker_overlap(_bundle(test=_split("yes/0a7c2a8d_nohash_0"))) == {
        "test_seen_in_train": 0.0,
        "test_seen_in_cal": 0.0,
    }


def test_audit_surfaces_the_exact_standalone_speaker_results() -> None:
    bundle = _bundle(
        train=_split("yes/0a7c2a8d_nohash_0", "silence/train/0"),
        cal=_split("up/fedcba98_nohash_0"),
        test=_split("yes/0a7c2a8d_nohash_1", "silence/test/0"),
        ood=_split("tree/22222222_nohash_0"),
    )

    audit = audit_bundle(bundle)
    assert audit["speakers_per_split"] == speakers_per_split(bundle)
    assert audit["speaker_overlap"] == speaker_overlap(bundle)


def test_pipeline_reexports_all_speaker_audit_helpers() -> None:
    assert pipeline.speaker_of is speaker_of
    assert pipeline.speakers_per_split is speakers_per_split
    assert pipeline.speaker_overlap is speaker_overlap
