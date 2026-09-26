"""Contract for src.pipeline.assemble_bundle / audit_bundle (width: cluster, real files in tmp_path)."""
from __future__ import annotations

import dataclasses
import os
import shutil
from collections import Counter

import numpy as np
import pytest

from pipeline_builders import COMMANDS, CLIP, build_dataset, noise_arrays
from src.dataset import AUXILIARY_WORDS, STUDY_LABELS, holdout_unknown_words
from src.pipeline import assemble_bundle, audit_bundle

CAP = 25
SPLITS = ("train", "val", "cal", "test", "ood")
IN_DIST = ("train", "val", "cal", "test")
UNKNOWN, SILENCE = STUDY_LABELS.index("unknown"), STUDY_LABELS.index("silence")
REGIONS = {"train": (0.0, 0.7), "cal": (0.7, 0.85), "test": (0.85, 1.0)}


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    """30 clips per command (so the cap binds); 'yes' has exactly CAP clips incl. a short and a long one."""
    from pipeline_builders import write_wav
    root = tmp_path_factory.mktemp("speech")
    build_dataset(root, command_clips=30, aux_clips=6, clips_by_word={"yes": CAP})
    write_wav(root / "yes" / "yes_000.wav", np.full(8_000, 1000))    # short: must be zero-padded
    write_wav(root / "yes" / "yes_001.wav", np.full(20_000, 2000))   # long: must be trimmed
    return root


@pytest.fixture(scope="module")
def bundle(dataset_root):
    return assemble_bundle(dataset_root, seed=17, cap_per_label=CAP)


def _label_totals(b, splits=IN_DIST):
    out = {}
    for s in splits:
        for lab, n in Counter(b.splits[s].y.tolist()).items():
            out.setdefault(lab, {})[s] = n
    return out


def _all_ids(b):
    return [i for s in SPLITS for i in b.splits[s].ids]


def test_bundle_declares_the_twelve_study_labels_and_five_split_keys(bundle):
    """Given a built bundle, then labels == STUDY_LABELS and splits has exactly the five named keys."""
    assert tuple(bundle.labels) == STUDY_LABELS
    assert set(bundle.splits) == set(SPLITS)


def test_split_arrays_are_int16_16k_aligned_and_ids_are_globally_unique(bundle):
    """Every split has int16 (n,16000) waveforms, int64 y, and aligned ids/words; ids appear in exactly one split."""
    for name in SPLITS:
        s = bundle.splits[name]
        assert s.waveforms.dtype == np.int16 and s.waveforms.shape[1:] == (CLIP,), name
        assert s.y.dtype == np.int64 and s.y.shape == (s.waveforms.shape[0],), name
        assert isinstance(s.ids, tuple) and isinstance(s.words, tuple), name
        assert len(s.ids) == len(s.words) == s.y.size, name
    ids = _all_ids(bundle)
    assert len(ids) == len(set(ids)), "a clip id appears in more than one split (or twice in one)"


def test_each_study_label_holds_exactly_cap_clips_across_in_distribution_splits(bundle):
    """Given 30 clips per command and a cap of 25, then each of the 12 labels totals exactly 25 (cap binds)."""
    totals = {lab: sum(v.values()) for lab, v in _label_totals(bundle).items()}
    assert totals == {i: CAP for i in range(len(STUDY_LABELS))}


def test_per_label_counts_follow_a_70_15_15_partition_within_one_clip(bundle):
    """train+val, cal, test counts of every label are within 1 clip of 70/15/15 of the label total."""
    for lab, per in _label_totals(bundle).items():
        total = sum(per.values())
        got = (per.get("train", 0) + per.get("val", 0), per.get("cal", 0), per.get("test", 0))
        for g, ratio, name in zip(got, (0.70, 0.15, 0.15), ("train+val", "cal", "test")):
            assert abs(g - total * ratio) <= 1, f"label {STUDY_LABELS[lab]} {name}: {g} vs {total * ratio}"


def test_validation_is_ten_percent_of_each_labels_train_portion(bundle):
    """val is carved from the 70% portion: per label |val - 0.10*(train+val)| <= 1 and val is non-empty overall."""
    per_label = _label_totals(bundle, ("train", "val"))
    for lab, per in per_label.items():
        portion = per.get("train", 0) + per.get("val", 0)
        assert abs(per.get("val", 0) - 0.10 * portion) <= 1, STUDY_LABELS[lab]
    assert bundle.splits["val"].y.size > 0


def test_unknown_label_is_sampled_only_from_train_unknown_words_and_spread_evenly(bundle):
    """unknown rows come solely from the 20 train-unknown words, cover all of them, counts differ by <= 1."""
    train_words, held = holdout_unknown_words(AUXILIARY_WORDS, 5, 17)
    assert tuple(bundle.train_unknown_words) == train_words and tuple(bundle.heldout_words) == held
    words = [w for s in IN_DIST for w, y in zip(bundle.splits[s].words, bundle.splits[s].y) if y == UNKNOWN]
    per_word = Counter(words)
    assert set(per_word) == set(train_words)
    assert max(per_word.values()) - min(per_word.values()) <= 1
    assert sum(per_word.values()) == CAP


def test_heldout_words_appear_only_in_ood_split_with_label_minus_one_spread_evenly(bundle):
    """ood holds cap//2 clips, y == -1, only held-out words (spread within 1); no held-out word elsewhere."""
    held = set(bundle.heldout_words)
    ood = bundle.splits["ood"]
    assert ood.y.size == CAP // 2 and set(ood.y.tolist()) == {-1}
    per_word = Counter(ood.words)
    assert set(per_word) <= held
    assert max(per_word.values()) - min(per_word.values()) <= 1 and len(per_word) == len(held)
    for s in IN_DIST:
        assert not (set(bundle.splits[s].words) & held), f"held-out word leaked into {s}"


def test_labels_agree_with_the_word_each_clip_came_from(bundle):
    """Command words map to their own label, auxiliary words to 'unknown', and nothing else is 'silence'-mislabelled."""
    for s in IN_DIST:
        for word, y in zip(bundle.splits[s].words, bundle.splits[s].y.tolist()):
            if word in COMMANDS:
                assert y == STUDY_LABELS.index(word)
            elif word in AUXILIARY_WORDS:
                assert y == UNKNOWN
            else:
                assert y == SILENCE, f"unexpected source {word!r} labelled {y}"


def _locate(window, arrays):
    """Return (file_index, offset) of an int16 window inside the noise arrays, or None."""
    for k, arr in enumerate(arrays):
        for off in np.flatnonzero(arr[: arr.size - CLIP + 1] == window[0]):
            if np.array_equal(arr[off:off + CLIP], window):
                return k, int(off)
    return None


def test_silence_windows_lie_in_their_splits_time_region_so_splits_never_overlap_in_time(bundle):
    """Each silence window is found in the noise files inside its split's 70/15/15 time region."""
    arrays = noise_arrays()
    spans = {s: [] for s in REGIONS}
    for s, (lo, hi) in REGIONS.items():
        sp = bundle.splits[s]
        for row in sp.waveforms[sp.y == SILENCE]:
            hit = _locate(row, arrays)
            assert hit is not None, f"{s}: a silence window is not a slice of any background noise file"
            k, off = hit
            n = arrays[k].size
            assert int(lo * n) <= off and off + CLIP <= int(hi * n), f"{s}: window outside its time region"
            spans[s].append((k, off))
    assert all(len(v) > 0 for v in spans.values())


def test_short_clip_is_zero_padded_and_long_clip_is_trimmed_to_16000(bundle):
    """A 8000-sample clip becomes its samples then zeros; a 20000-sample clip becomes its first 16000 samples."""
    rows = np.concatenate([bundle.splits[s].waveforms for s in IN_DIST])
    short = [r for r in rows if (r[:8_000] == 1000).all() and (r[8_000:] == 0).all()]
    long_ = [r for r in rows if (r == 2000).all()]
    assert len(short) == 1, "zero-padded short clip not found exactly once"
    assert len(long_) == 1, "trimmed long clip not found exactly once"


def test_assembly_is_deterministic_for_a_seed_and_selection_changes_with_seed(dataset_root, bundle):
    """Same seed -> identical ids, labels, waveforms and digest; a different seed selects different clips."""
    again = assemble_bundle(dataset_root, seed=17, cap_per_label=CAP)
    for s in SPLITS:
        assert again.splits[s].ids == bundle.splits[s].ids
        assert np.array_equal(again.splits[s].waveforms, bundle.splits[s].waveforms)
        assert np.array_equal(again.splits[s].y, bundle.splits[s].y)
    assert again.manifest_sha256 == bundle.manifest_sha256
    other = assemble_bundle(dataset_root, seed=18, cap_per_label=CAP)
    assert other.manifest_sha256 != bundle.manifest_sha256
    assert set(_all_ids(other)) != set(_all_ids(bundle))


def test_manifest_digest_is_independent_of_filesystem_listing_order(dataset_root, bundle, monkeypatch):
    """Reversing every directory listing must not change the digest or the split membership."""
    real_listdir, real_scandir = os.listdir, os.scandir

    class _Rev(list):
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def close(self): pass

    monkeypatch.setattr(os, "listdir", lambda p=".": list(reversed(real_listdir(p))))
    monkeypatch.setattr(os, "scandir", lambda p=".": _Rev(reversed(list(real_scandir(p)))))
    reordered = assemble_bundle(dataset_root, seed=17, cap_per_label=CAP)
    assert reordered.manifest_sha256 == bundle.manifest_sha256
    assert {s: reordered.splits[s].ids for s in SPLITS} == {s: bundle.splits[s].ids for s in SPLITS}


def test_word_with_too_few_clips_contributes_all_it_has_and_records_shortfall(tmp_path):
    """Given 'yes' has 10 clips and cap 25, then all 10 are used and shortfalls == {'yes': 15}."""
    root = build_dataset(tmp_path / "d", command_clips=25, aux_clips=6, clips_by_word={"yes": 10})
    b = assemble_bundle(root, seed=17, cap_per_label=CAP)
    yes_rows = sum(int(np.sum(b.splits[s].y == STUDY_LABELS.index("yes"))) for s in IN_DIST)
    assert yes_rows == 10
    assert dict(b.shortfalls) == {"yes": 15}


def test_complete_dataset_reports_no_shortfall(bundle):
    assert dict(bundle.shortfalls) == {}


def test_missing_dataset_root_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        assemble_bundle(tmp_path / "nope", seed=17, cap_per_label=CAP)


def test_missing_background_noise_directory_raises_value_error(dataset_root, tmp_path):
    root = tmp_path / "c"
    shutil.copytree(dataset_root, root)
    shutil.rmtree(root / "_background_noise_")
    with pytest.raises(ValueError):
        assemble_bundle(root, seed=17, cap_per_label=CAP)


def test_missing_word_directory_raises_value_error(dataset_root, tmp_path):
    root = tmp_path / "c"
    shutil.copytree(dataset_root, root)
    shutil.rmtree(root / "stop")
    with pytest.raises(ValueError):
        assemble_bundle(root, seed=17, cap_per_label=CAP)


# ------------------------------------------------------------------ audit_bundle

def test_audit_of_a_correct_bundle_is_clean_and_counts_match_the_arrays(bundle):
    a = audit_bundle(bundle)
    assert a["id_overlap"] and all(v == 0 for v in a["id_overlap"].values())
    assert a["duplicate_checksums"] == {}
    assert a["heldout_word_leak"] is False and a["silence_time_overlap"] is False
    def norm(k):  # label counts may be keyed by index or by label name
        return STUDY_LABELS.index(k) if k in STUDY_LABELS else int(k)

    for s in SPLITS:
        expected = {int(k): int(v) for k, v in Counter(bundle.splits[s].y.tolist()).items()}
        assert {norm(k): v for k, v in a["label_counts"][s].items()} == expected, s


def test_audit_reports_an_id_shared_between_train_and_test(bundle):
    """Given one train id copied into test, then exactly one split pair reports a positive overlap of 1."""
    test = bundle.splits["test"]
    bad = dataclasses.replace(bundle, splits={**bundle.splits, "test": dataclasses.replace(
        test, ids=(bundle.splits["train"].ids[0],) + test.ids[1:])})
    overlap = audit_bundle(bad)["id_overlap"]
    assert sorted(v for v in overlap.values() if v) == [1]


def test_audit_reports_a_heldout_word_in_a_training_split(bundle):
    train = bundle.splits["train"]
    bad = dataclasses.replace(bundle, splits={**bundle.splits, "train": dataclasses.replace(
        train, words=(bundle.heldout_words[0],) + train.words[1:])})
    assert audit_bundle(bad)["heldout_word_leak"] is True


def test_audit_reports_planted_identical_wav_files_with_both_ids(tmp_path):
    """Given yes/x.wav and no/y.wav are byte-identical (both selected), one digest lists exactly those two clips."""
    root = build_dataset(tmp_path / "d", command_clips=25, aux_clips=6)
    shutil.copyfile(root / "yes" / "yes_000.wav", root / "no" / "no_000.wav")
    b = assemble_bundle(root, seed=17, cap_per_label=CAP)
    dups = audit_bundle(b)["duplicate_checksums"]
    assert len(dups) == 1
    (ids,) = dups.values()
    word_of = {i: w for s in SPLITS for i, w in zip(b.splits[s].ids, b.splits[s].words)}
    assert len(ids) == 2 and sorted(word_of[i] for i in ids) == ["no", "yes"]
