"""Speaker-audit contract at the dataset-assembly boundary."""
from __future__ import annotations

import numpy as np

from pipeline_builders import CLIP, build_dataset
from src.dataset import STUDY_LABELS
from src.pipeline import (
    DatasetBundle,
    SplitArrays,
    assemble_bundle,
    audit_bundle,
    speaker_of,
    speaker_overlap,
    speakers_per_split,
)


def _split(ids: tuple[str, ...]) -> SplitArrays:
    """Minimal aligned split; speaker helpers depend only on identifiers."""
    return SplitArrays(
        waveforms=np.zeros((len(ids), CLIP), dtype=np.int16),
        y=np.zeros(len(ids), dtype=np.int64),
        ids=ids,
        words=tuple("source" for _ in ids),
    )


def _speaker_bundle() -> DatasetBundle:
    return DatasetBundle(
        labels=STUDY_LABELS,
        splits={
            "train": _split(("yes/a1b2c3d4_nohash_0", "silence/train/0", "opaque_0")),
            "cal": _split(("no/a1b2c3d4_nohash_1", "up/f1e2d3c4_nohash_0")),
            "test": _split(("down/a1b2c3d4_nohash_9", "silence/test/0", "test_999")),
        },
        train_unknown_words=(),
        heldout_words=(),
        manifest_sha256="test-only",
        shortfalls={},
    )


def test_source_stems_are_preserved_through_assembly_including_same_word_same_suffix(tmp_path):
    """Two real source stems sharing ``_nohash_0`` must not collapse to one numeric ID."""
    root = build_dataset(tmp_path / "speech", command_clips=2, aux_clips=2)
    yes = root / "yes"
    (yes / "yes_000.wav").rename(yes / "a1b2c3d4_nohash_0.wav")
    (yes / "yes_001.wav").rename(yes / "f1e2d3c4_nohash_0.wav")

    # Every command has exactly cap clips, so these two source files are selected without sampling.
    bundle = assemble_bundle(root, seed=17, cap_per_label=2)
    yes_ids = tuple(
        identifier
        for split in bundle.splits.values()
        for identifier, word in zip(split.ids, split.words)
        if word == "yes"
    )

    assert len(yes_ids) == len(set(yes_ids)) == 2
    assert {speaker_of(identifier) for identifier in yes_ids} == {"a1b2c3d4", "f1e2d3c4"}


def test_speaker_audit_counts_real_source_speakers_but_not_silence_or_opaque_ids():
    bundle = _speaker_bundle()

    assert speaker_of("a1b2c3d4_nohash_0") == "a1b2c3d4"
    assert speaker_of("yes/a1b2c3d4_nohash_0") == "a1b2c3d4"
    assert speaker_of("silence/test/0") is None
    assert speaker_of("yes_000") is None
    assert speaker_of("yes/a1b2c3d4_notnohash_0") is None

    expected_counts = {"train": 1, "cal": 2, "test": 1}
    expected_overlap = {"test_seen_in_train": 1 / 3, "test_seen_in_cal": 1 / 3}
    assert speakers_per_split(bundle) == expected_counts
    assert speaker_overlap(bundle) == expected_overlap

    audit = audit_bundle(bundle)
    assert audit["speakers_per_split"] == expected_counts
    assert audit["speaker_overlap"] == expected_overlap
    # Speaker reuse is distinct from duplicated clip IDs: all source IDs remain separate.
    assert audit["id_overlap"]["train-test"] == 0
