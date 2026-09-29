"""Frozen contract for silence-window provenance and real `silence_time_overlap` auditing.

(R4.2, claims C43/C44 in plans/round-04/plan-a.md.) This file specifies the exact API
extension the implementer must build against. It does not implement anything and must
not be edited by the implementer.

THE BUG BEING CLOSED (verified by direct inspection of the current, unfixed code):
`src/dataset.py:124-168`'s `silence_windows` picks a random source noise file and a
random offset inside that split's fractional time region of the file (train=[0,0.7),
cal=[0.7,0.85), test=[0.85,1.0) of the file's own length). When the chosen file is too
short for a full 16000-sample window to fit in that region (`max_offset < region_start`,
equivalently `region_end - region_start < 16000`), the old code falls back to
`offset = region_start` "as-is" WITHOUT checking `region_start + 16000 <= region_end` --
so a short file can produce a window that reads past its region boundary into a
neighbouring split's region of the SAME file. No provenance is returned today, so
`src/bundle.py:421` cannot detect this and hard-codes `silence_time_overlap = False`
unconditionally.

===========================================================================
FROZEN CONTRACT -- PART 1: `src.dataset.silence_windows`
===========================================================================

    def silence_windows(
        noise_arrays: list[np.ndarray],
        n: int,
        seed: int,
        split: str = "train",
        *,
        return_provenance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[tuple[int, int]]]:

- `return_provenance=False` (the default, unchanged call signature): behaves exactly as
  today except for the invariant below -- returns only the (n, 16000) float32 ndarray of
  windows.
- `return_provenance=True`: returns a 2-tuple `(windows, provenance)`. `windows` is
  bit-identical to what `return_provenance=False` would return for the same
  `(noise_arrays, n, seed, split)` -- toggling the flag must never change which windows
  are drawn, only whether a second value is returned. `provenance` is a
  `list[tuple[int, int]]` of length `n`; `provenance[i] = (file_idx, offset)` is the
  index into `noise_arrays` and the sample offset used to build `windows[i]`, i.e.
  `windows[i] == noise_arrays[file_idx][offset:offset + 16000]` exactly (no padding --
  the invariant below guarantees a full 16000-sample slice is always available for any
  window that is actually returned).

INVARIANT (holds regardless of `return_provenance`, and is the actual bug fix): for
every returned window, with `(file_idx, offset)` its provenance and `length =
len(noise_arrays[file_idx])`, `region_start = int(start_frac * length)`, `region_end =
int(end_frac * length)` for the requested split's `(start_frac, end_frac)` (train
0.0/0.7, cal 0.7/0.85, test 0.85/1.0) -- it must hold that
`region_start <= offset and offset + 16000 <= region_end`.

Resolution when the randomly-drawn file cannot satisfy this (`region_end - region_start
< 16000` for that file): `silence_windows` must deterministically pick a DIFFERENT file
from `noise_arrays` that CAN satisfy it for the requested split, and draw the window
from that file instead (never silently spill over). If NO file in `noise_arrays` can
satisfy it for the requested split (every file's region is too short), `silence_windows`
must raise `ValueError` (message naming the split) instead of returning any window.
This is a "resample-else-raise" contract, not "always raise" -- a single short file
among otherwise-long files must not break generation. The tests below are written
against this behavioural invariant only, not against any particular resampling
algorithm/order, so any implementation satisfying the invariant passes.

===========================================================================
FROZEN CONTRACT -- PART 2: `DatasetBundle` provenance fields + real `audit_bundle`
===========================================================================

`DatasetBundle` (src/bundle.py) gains two new fields, both with defaults so every
existing hand-built `DatasetBundle(...)` call in the rest of the suite (which passes
none of this) keeps constructing a valid, vacuously-clean bundle:

    silence_provenance: dict[str, tuple[tuple[int, int], ...]] = dataclasses.field(default_factory=dict)
    silence_source_lengths: tuple[int, ...] = ()

- `silence_provenance` maps split name in {"train", "val", "cal", "test"} to a tuple of
  `(file_idx, offset)` pairs, one per silence window belonging to that split, in the
  same order as that split's `silence/<split>/<i>` ids (ascending `i`). A split with no
  silence windows maps to `()`. `file_idx` values are only meaningful because
  `assemble_bundle` calls `silence_windows` with the SAME `noise_arrays` list (same
  objects, same order) for every split's generation -- this is the explicit assumption
  that makes `file_idx` comparable across the train/cal/test calls; it is not re-derived
  or re-validated anywhere, it is simply relied upon.
- `silence_source_lengths` is `tuple(len(a) for a in noise_arrays)` for that same
  canonical `noise_arrays` list, indexed identically to the `file_idx` values above.
- Special case: `val`'s silence windows are architecturally a stratified SUBSET of the
  windows generated for the "train" region (see `assemble_bundle`'s `silence_array_70`
  / `train_silence_indices` / `val_silence_indices` machinery) -- there is no
  independently-sampled "val region". Therefore `val` provenance entries must be
  validated against the TRAIN region fractions (0.0, 0.7), not a separate val region.
  This is a load-bearing, easy-to-get-wrong design decision and is tested explicitly
  below.

`audit_bundle`'s `silence_time_overlap` (currently a hard-coded `False` literal at
src/bundle.py:421) must instead be computed for real: for every split in
`{"train", "val", "cal", "test"}` and every `(file_idx, offset)` in
`bundle.silence_provenance.get(split, ())`, using `length =
bundle.silence_source_lengths[file_idx]` and that split's region fractions (val uses
train's fractions per above), the window is a violation if
`not (region_start <= offset and offset + 16000 <= region_end)`. `silence_time_overlap`
is reported as a plain `bool`: `True` if at least one violation exists anywhere, `False`
otherwise (including vacuously False when `silence_provenance` is empty/absent, which
preserves every existing hand-built bundle in the rest of the suite that never sets it).
This shape choice keeps `tests/test_pipeline_data_contract.py`'s existing
`a["silence_time_overlap"] is False` assertion meaningful rather than replacing it with
a different type.

===========================================================================
Expected state against the CURRENT (unfixed) code
===========================================================================
Every test below that calls `return_provenance=True`, or constructs a `DatasetBundle`
with `silence_provenance=`/`silence_source_lengths=`, will fail today with a
`TypeError` (unexpected keyword argument) -- the feature does not exist yet. That is
the correct, expected state: it proves the contract is not accidentally already
satisfied.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.dataset import silence_windows
from src.bundle import DatasetBundle, SplitArrays, assemble_bundle, audit_bundle
from pipeline_builders import build_dataset

CLIP = 16_000
REGION_FRACTIONS = {
    "train": (0.0, 0.7),
    "val": (0.0, 0.7),  # val shares the train region -- see module docstring
    "cal": (0.7, 0.85),
    "test": (0.85, 1.0),
}


def _noise_array(seed: int, size: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(size).astype(np.float32)


def _region_bounds(length: int, split: str) -> tuple[int, int]:
    start_frac, end_frac = REGION_FRACTIONS[split]
    return int(start_frac * length), int(end_frac * length)


# ===========================================================================
# Group A: silence_windows provenance on a normal (long-enough) noise array
# ===========================================================================


def test_default_call_without_return_provenance_still_returns_a_plain_array() -> None:
    """Given the legacy call signature, when silence_windows is called, then it
    returns only the (n, 16000) waveform array -- the new keyword is additive."""
    arrays = [_noise_array(i, 160_000) for i in range(3)]
    result = silence_windows(arrays, n=5, seed=1, split="train")
    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 16000)


def test_toggling_return_provenance_does_not_change_which_windows_are_drawn() -> None:
    """Given the same seed, when called with and without return_provenance, then the
    returned waveforms are bit-identical -- the flag only adds a second value."""
    arrays = [_noise_array(i, 160_000) for i in range(3)]
    plain = silence_windows(arrays, n=6, seed=7, split="cal")
    windows, provenance = silence_windows(arrays, n=6, seed=7, split="cal", return_provenance=True)
    assert np.array_equal(plain, windows)
    assert len(provenance) == 6


@pytest.mark.parametrize("split", ["train", "cal", "test"])
def test_provenance_offsets_lie_in_the_requested_splits_region_and_match_window_content(split: str) -> None:
    """Given several long noise files, when return_provenance=True, then every
    (file_idx, offset) both lies inside the split's region of its own file and
    reproduces the exact window content -- provenance cannot be fabricated/misaligned."""
    arrays = [_noise_array(i, 160_000 + i * 4_000) for i in range(4)]
    windows, provenance = silence_windows(arrays, n=25, seed=3, split=split, return_provenance=True)
    assert len(provenance) == 25
    for window, (file_idx, offset) in zip(windows, provenance):
        assert 0 <= file_idx < len(arrays)
        length = len(arrays[file_idx])
        region_start, region_end = _region_bounds(length, split)
        assert region_start <= offset, f"{split}: offset {offset} before region start {region_start}"
        assert offset + CLIP <= region_end, f"{split}: window [{offset},{offset + CLIP}) exceeds region end {region_end}"
        assert np.array_equal(window, arrays[file_idx][offset:offset + CLIP]), "provenance does not reproduce the window"


def test_provenance_is_deterministic_for_a_fixed_seed() -> None:
    """Given the same seed, when called twice with return_provenance=True, then the
    provenance lists (not just the waveforms) are identical."""
    arrays = [_noise_array(i, 200_000) for i in range(3)]
    _, prov1 = silence_windows(arrays, n=10, seed=99, split="test", return_provenance=True)
    _, prov2 = silence_windows(arrays, n=10, seed=99, split="test", return_provenance=True)
    assert prov1 == prov2


# ===========================================================================
# Group B: regression -- a short file must never spill past its region
# ===========================================================================
# We require: if at least one file's region is large enough, generation must succeed
# by resampling to a feasible file (never spill); if NO file's region is large enough,
# generation must raise ValueError. Both branches are tested below.


def test_short_file_regression_resamples_to_a_feasible_file_instead_of_spilling() -> None:
    """Given one file too short for the 'test' region to fit a window, and one file
    long enough, when silence_windows is called for split='test', then every returned
    window comes from the feasible file and lies inside its region -- the fixed
    behavior REQUIRED here is silent resampling to a valid file, not an exception,
    because a feasible alternative exists.

    Length 50_000: test region = [42500, 50000), size 7500 < 16000 -> infeasible
    (mirrors the exact old-code trigger condition max_offset < region_start).
    Length 200_000: test region = [170000, 200000), size 30000 >= 16000 -> feasible.
    """
    short_file = _noise_array(11, 50_000)
    long_file = _noise_array(12, 200_000)
    arrays = [short_file, long_file]

    windows, provenance = silence_windows(arrays, n=40, seed=5, split="test", return_provenance=True)

    assert len(provenance) == 40
    region_start, region_end = _region_bounds(len(long_file), "test")
    for window, (file_idx, offset) in zip(windows, provenance):
        assert file_idx == 1, "the infeasible short file (index 0) must never be used for the 'test' split here"
        assert region_start <= offset and offset + CLIP <= region_end, "window spilled outside its region"
        assert np.array_equal(window, long_file[offset:offset + CLIP])


def test_all_files_too_short_for_the_split_raises_value_error() -> None:
    """Given every file too short for the 'test' region to fit a window, when
    silence_windows is called for split='test', then it raises ValueError instead of
    returning a spilled-over window.

    Length 50_000: test region size 7500 < 16000. Length 45_000: test region =
    [38250, 45000), size 6750 < 16000. Neither file can produce a valid 'test' window.
    """
    arrays = [_noise_array(21, 50_000), _noise_array(22, 45_000)]
    with pytest.raises(ValueError):
        silence_windows(arrays, n=5, seed=6, split="test")


# ===========================================================================
# Group C: audit_bundle computes silence_time_overlap for real from provenance
# ===========================================================================


def _silence_split(n: int, name: str) -> SplitArrays:
    ids = tuple(f"silence/{name}/{i}" for i in range(n))
    return SplitArrays(
        waveforms=np.zeros((n, CLIP), dtype=np.int16),
        y=np.zeros(n, dtype=np.int64),
        ids=ids,
        words=tuple("silence" for _ in ids),
    )


def _bundle_with_provenance(
    *,
    counts: dict[str, int],
    provenance: dict[str, tuple[tuple[int, int], ...]],
    source_lengths: tuple[int, ...],
) -> DatasetBundle:
    return DatasetBundle(
        labels=("silence",),
        splits={name: _silence_split(n, name) for name, n in counts.items()},
        train_unknown_words=(),
        heldout_words=(),
        manifest_sha256="silence-provenance-test",
        shortfalls={},
        silence_provenance=provenance,
        silence_source_lengths=source_lengths,
    )


def test_audit_reports_false_when_every_silence_window_is_inside_its_region() -> None:
    """Given a bundle whose provenance places every window inside its own split's
    (val -> train) region, when audit_bundle is called, then silence_time_overlap
    is False.

    File length 160_000: train/val region [0, 112000), cal region [112000, 136000),
    test region [136000, 160000). All offsets chosen well inside their region.
    """
    bundle = _bundle_with_provenance(
        counts={"train": 1, "val": 1, "cal": 1, "test": 1},
        provenance={
            "train": ((0, 1_000),),
            "val": ((0, 50_000),),      # inside train region [0, 112000)
            "cal": ((0, 120_000),),     # inside cal region [112000, 136000)
            "test": ((0, 140_000),),    # inside test region [136000, 160000): window [140000,156000)
        },
        source_lengths=(160_000,),
    )
    assert audit_bundle(bundle)["silence_time_overlap"] is False


def test_audit_reports_true_when_a_window_spills_into_the_next_splits_region() -> None:
    """Given a bundle where one cal-split window's range extends past the cal region
    boundary into the test region of the same file, when audit_bundle is called, then
    silence_time_overlap is True.

    Cal region for length 160_000 is [112000, 136000). offset=130_000 gives a window
    of [130000, 146000), which exceeds region_end=136000 by 10000 samples.
    """
    bundle = _bundle_with_provenance(
        counts={"train": 0, "val": 0, "cal": 1, "test": 0},
        provenance={"train": (), "val": (), "cal": ((0, 130_000),), "test": ()},
        source_lengths=(160_000,),
    )
    assert audit_bundle(bundle)["silence_time_overlap"] is True


def test_audit_checks_val_windows_against_the_train_region_not_a_separate_val_region() -> None:
    """Given a val-split window whose offset sits inside the CAL fractional region of
    its file (not the train region), when audit_bundle is called, then
    silence_time_overlap is True -- val has no independently-sampled region of its
    own, so any val offset outside [0, 0.7*length) is a real defect, even though it
    would look perfectly valid if (incorrectly) checked against a separate val region.

    File length 160_000: train region [0, 112000). offset=120_000 lies in the cal
    region [112000, 136000), i.e. strictly outside the train region val must be
    checked against.
    """
    bad = _bundle_with_provenance(
        counts={"train": 0, "val": 1, "cal": 0, "test": 0},
        provenance={"train": (), "val": ((0, 120_000),), "cal": (), "test": ()},
        source_lengths=(160_000,),
    )
    assert audit_bundle(bad)["silence_time_overlap"] is True

    good = _bundle_with_provenance(
        counts={"train": 0, "val": 1, "cal": 0, "test": 0},
        provenance={"train": (), "val": ((0, 90_000),), "cal": (), "test": ()},
        source_lengths=(160_000,),
    )
    assert audit_bundle(good)["silence_time_overlap"] is False


def test_audit_uses_each_windows_own_file_length_not_the_first_files_length() -> None:
    """Given two source files of very different lengths, when a window's file_idx
    points at the SHORT file but its offset is only valid relative to the LONG file's
    region, then audit_bundle must still catch it -- proving the region bounds are
    recomputed per (file_idx, its own length), not reused from file 0.

    File 0 length 160_000: train region [0, 112000) (offset=50000 is fine here).
    File 1 length 40_000: train region [0, 28000). offset=20000 gives a window of
    [20000, 36000), which exceeds file 1's own region_end=28000 -- a violation only
    visible if file 1's own (short) length is used, not file 0's.
    """
    bundle = _bundle_with_provenance(
        counts={"train": 2, "val": 0, "cal": 0, "test": 0},
        provenance={"train": ((0, 50_000), (1, 20_000)), "val": (), "cal": (), "test": ()},
        source_lengths=(160_000, 40_000),
    )
    assert audit_bundle(bundle)["silence_time_overlap"] is True


def test_audit_is_vacuously_false_when_no_silence_provenance_was_ever_recorded() -> None:
    """Given a DatasetBundle built the old way (no silence_provenance/source_lengths
    keywords at all, matching every other hand-built bundle already in this suite),
    when audit_bundle is called, then silence_time_overlap defaults to False rather
    than raising or defaulting to True -- backward compatibility for bundles that
    predate this feature."""
    bundle = DatasetBundle(
        labels=("silence",),
        splits={"train": _silence_split(1, "train")},
        train_unknown_words=(),
        heldout_words=(),
        manifest_sha256="no-provenance-at-all",
        shortfalls={},
    )
    assert audit_bundle(bundle)["silence_time_overlap"] is False


# ===========================================================================
# Group D: end-to-end wiring through assemble_bundle (cluster-width check)
# ===========================================================================


def test_assemble_bundle_populates_provenance_with_one_entry_per_silence_window(tmp_path) -> None:
    """Given a real dataset built on disk, when assemble_bundle runs, then
    bundle.silence_provenance carries exactly one (file_idx, offset) pair per silence
    id actually present in each split, and silence_source_lengths has one entry per
    background-noise file -- proving assemble_bundle threads real provenance through,
    not an empty/placeholder structure."""
    root = build_dataset(tmp_path / "speech", command_clips=12, aux_clips=4, noise_files=3, noise_len=160_000)
    bundle = assemble_bundle(root, seed=17, cap_per_label=10)

    assert len(bundle.silence_source_lengths) == 3
    for split_name in ("train", "val", "cal", "test"):
        n_silence_ids = sum(1 for i in bundle.splits[split_name].ids if i.startswith("silence/"))
        provenance = bundle.silence_provenance.get(split_name, ())
        assert len(provenance) == n_silence_ids, f"{split_name}: provenance count does not match silence id count"
        for file_idx, offset in provenance:
            assert 0 <= file_idx < len(bundle.silence_source_lengths)
            assert isinstance(offset, (int, np.integer))


def test_audit_bundle_reports_false_for_a_genuinely_correct_real_assembly(tmp_path) -> None:
    """Given a real, correctly-assembled bundle (background files long enough that no
    split's region is infeasible), when audit_bundle is called, then
    silence_time_overlap is False -- computed for real from the wired-through
    provenance, not by virtue of the field being empty."""
    root = build_dataset(tmp_path / "speech", command_clips=12, aux_clips=4, noise_files=3, noise_len=160_000)
    bundle = assemble_bundle(root, seed=17, cap_per_label=10)

    assert any(bundle.silence_provenance.get(s) for s in ("train", "val", "cal", "test")), \
        "test setup invariant: this bundle must actually contain silence windows to be a meaningful check"
    assert audit_bundle(bundle)["silence_time_overlap"] is False
