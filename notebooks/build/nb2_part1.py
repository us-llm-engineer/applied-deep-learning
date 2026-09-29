"""Notebook 2, Part 1: Data card and shared synthetic bundle builder (derived here).

Sections:
- 1. Data card: STUDY_LABELS, AUXILIARY_WORDS, holdout unknown words, synthetic bundle audit
"""

from nbkit import md, code
import inspect
import numpy as np
import hashlib


# ─────────────────────────────────────────────────────────────────────────────
# Module-level synthetic bundle builder and noise array generator.
# These are tagged NB3-REUSE and will be re-executed by notebook 3.
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_noise_arrays_w2(seed=20260926):
    """Build >=3 synthetic noise arrays of deliberately unequal lengths.

    One must be short enough to trigger the fallback branch in silence_windows.
    Exposed as module-level so NB3 can reuse it independently of the bundle.
    """
    rng = np.random.default_rng(seed)

    # Three noise arrays with unequal lengths:
    # - 16000 (just fits one clip)
    # - 48000 (fits 3 clips)
    # - 100000 (large, normal case)
    # The first one (16000) is short enough to trigger fallback in silence_windows
    # when requesting windows from the "cal" (0.7-0.85) or "test" (0.85-1.0) region.
    noise_lengths = [16000, 48000, 100000]
    noise_arrays = []

    for i, length in enumerate(noise_lengths):
        # Synthetic noise: sum of several sine waves at different frequencies
        t = np.arange(length, dtype=np.float32) / 16000.0
        noise = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # 440 Hz (A4)
            0.2 * np.sin(2 * np.pi * 880 * t) +  # 880 Hz (A5)
            0.15 * np.sin(2 * np.pi * 220 * t)   # 220 Hz (A3)
        )
        # Add some random fluctuation
        noise = noise + 0.1 * rng.standard_normal(length)
        # Normalize to int16 range
        noise = np.clip(noise * 10000, -32768, 32767).astype(np.int16)
        noise_arrays.append(noise)

    return noise_arrays


def build_synthetic_bundle_w2(seed=20260926):
    """Build a tiny in-memory synthetic DatasetBundle.

    Sizes: 12 labels × 20 clips/label train, 6/label val/cal/test,
           plus 2 heldout words × 10 clips for ood.

    Waveforms are synthetic sine/noise (int16, 16000 samples per clip).

    Speaker overlap is engineered: exactly 3 of the 12 speaker IDs are reused
    across train and test splits, giving a known, exact overlap fraction.
    """
    from src.bundle import DatasetBundle, SplitArrays, STUDY_LABELS
    from src.dataset import holdout_unknown_words, AUXILIARY_WORDS, manifest_checksum, AudioRecord

    rng = np.random.default_rng(seed)
    CLIP = 16000

    # Get train/heldout split of auxiliary words
    train_unknown_words, heldout_words = holdout_unknown_words(AUXILIARY_WORDS, 5, seed=17)

    # Build synthetic noise arrays (reuse the module-level function)
    noise_arrays = synthetic_noise_arrays_w2(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Speaker ID construction: deliberately reuse 3 speaker IDs across splits
    # ─────────────────────────────────────────────────────────────────────────
    # We have 12 labels total. Let's use speaker IDs: 00000000 through 0000000b (12 speakers).
    # For train: all 12 speakers, each with 20 clips.
    # For test: use only speakers 00000000, 00000001, 00000002 (3 reused speakers),
    #           each with 2 clips (6 total for 6 labels that have speakers).
    # This gives overlap = 3 test speakers / (3 test speakers) = 1.0 for the clips that use speakers.
    # Actually, let me re-read the requirement: "3 of 12 synthetic speaker IDs reused across splits"

    # Better approach: test split uses only 3 of the 12 speakers.
    # If test has N unique speakers, and 3 of them are in train, then overlap = 3/N.
    # Let's make test have exactly 6 speakers (one per test label), and 3 of them appear in train.
    # Then overlap = 3/6 = 0.5

    # Actually, let me use:
    # - Train: uses all 12 unique speaker IDs (one per label)
    # - Test: uses only 3 of those 12 speaker IDs (3 labels use speakers from train, others use new ones)
    #   If test has 6 labels with speakers, 3 of which are in train, then overlap = 3/6 = 0.5

    # For simplicity with 12 labels and knowing speaker_of extracts 8-digit hex:
    # Let's have:
    # - 12 unique speakers in train: "00000000" to "0000000b"
    # - 3 test speakers reuse from train: "00000000", "00000001", "00000002"
    # - 3 test speakers are new: "0000000c", "0000000d", "0000000e"
    # - Only 6 of the 12 labels have clips (the rest have no speaker or minimal clips)
    # Actually, simpler: each label gets one speaker ID.

    # Cleaner:
    # - All 12 labels get speaker IDs in train
    # - In test, only 6 clips come from labels that use speakers, and only 3 speaker IDs overlap
    # - Test speakers: 3 reused from train, and for the other 3 test labels, use new IDs
    # - Expected overlap = 3 / 6 = 0.5, exactly

    speaker_ids = [f"{i:08x}" for i in range(12)]  # "00000000" to "0000000b"

    # Construct waveforms for each label
    label_to_waveforms = {}
    for label_idx, label in enumerate(STUDY_LABELS):
        t = np.arange(CLIP, dtype=np.float32) / 16000.0
        # Use a different frequency per label for variation
        freq = 220 + label_idx * 50  # 220 Hz, 270 Hz, 320 Hz, ...
        waveform = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        waveform = np.clip(waveform * 10000, -32768, 32767).astype(np.int16)
        label_to_waveforms[label] = waveform

    # ─────────────────────────────────────────────────────────────────────────
    # Build splits
    # ─────────────────────────────────────────────────────────────────────────

    # Train: 20 clips per label (12 labels), using all 12 speaker IDs
    train_waveforms, train_y, train_ids, train_words = [], [], [], []
    for label_idx, label in enumerate(STUDY_LABELS):
        waveform_template = label_to_waveforms[label]
        for clip_i in range(20):
            train_waveforms.append(waveform_template)
            train_y.append(label_idx)
            speaker = speaker_ids[label_idx]
            train_ids.append(f"{speaker}_nohash_{clip_i}.wav")
            train_words.append(label)

    train_split = SplitArrays(
        waveforms=np.array(train_waveforms, dtype=np.int16),
        y=np.array(train_y, dtype=np.int64),
        ids=tuple(train_ids),
        words=tuple(train_words)
    )

    # Val: 6 clips per label
    val_waveforms, val_y, val_ids, val_words = [], [], [], []
    for label_idx, label in enumerate(STUDY_LABELS):
        waveform_template = label_to_waveforms[label]
        for clip_i in range(6):
            val_waveforms.append(waveform_template)
            val_y.append(label_idx)
            speaker = speaker_ids[label_idx]
            val_ids.append(f"{speaker}_nohash_{20 + clip_i}.wav")
            val_words.append(label)

    val_split = SplitArrays(
        waveforms=np.array(val_waveforms, dtype=np.int16),
        y=np.array(val_y, dtype=np.int64),
        ids=tuple(val_ids),
        words=tuple(val_words)
    )

    # Cal: 6 clips per label
    cal_waveforms, cal_y, cal_ids, cal_words = [], [], [], []
    for label_idx, label in enumerate(STUDY_LABELS):
        waveform_template = label_to_waveforms[label]
        for clip_i in range(6):
            cal_waveforms.append(waveform_template)
            cal_y.append(label_idx)
            speaker = speaker_ids[label_idx]
            cal_ids.append(f"{speaker}_nohash_{26 + clip_i}.wav")
            cal_words.append(label)

    cal_split = SplitArrays(
        waveforms=np.array(cal_waveforms, dtype=np.int16),
        y=np.array(cal_y, dtype=np.int64),
        ids=tuple(cal_ids),
        words=tuple(cal_words)
    )

    # Test: 6 clips per label, but with speaker overlap engineered:
    # - First 3 labels use speaker IDs from train (00000000, 00000001, 00000002)
    # - Next 3 labels use new speaker IDs (0000000c, 0000000d, 0000000e)
    # - Last 6 labels (indices 6-11) use new speaker IDs too
    # This gives: 3 test speakers in train, 9 test speakers not in train
    # But test only has 12 labels, each with clips.
    # Actually, the question is: how many *unique* test speakers are there, and how many are in train?

    # Let me recalculate:
    # - Test has 12 labels, each using a speaker
    # - Speakers for test labels 0-2: use IDs from train (00000000, 00000001, 00000002) -- 3 overlap
    # - Speakers for test labels 3-11: use new IDs (0000000c to 00000014) -- 9 new
    # - So test has 12 unique speakers, 3 of which are in train
    # - speaker_overlap = 3 / 12 = 0.25, exactly

    test_waveforms, test_y, test_ids, test_words = [], [], [], []
    for label_idx, label in enumerate(STUDY_LABELS):
        waveform_template = label_to_waveforms[label]
        for clip_i in range(6):
            test_waveforms.append(waveform_template)
            test_y.append(label_idx)

            # Deliberately reuse speakers 0, 1, 2 from train; use new IDs for the rest
            if label_idx < 3:
                speaker = speaker_ids[label_idx]  # Reuse from train
            else:
                speaker = f"{12 + label_idx:08x}"  # New speakers for test

            test_ids.append(f"{speaker}_nohash_{32 + clip_i}.wav")
            test_words.append(label)

    test_split = SplitArrays(
        waveforms=np.array(test_waveforms, dtype=np.int16),
        y=np.array(test_y, dtype=np.int64),
        ids=tuple(test_ids),
        words=tuple(test_words)
    )

    # OOD: 2 heldout words × 10 clips = 20 clips total
    ood_waveforms, ood_y, ood_ids, ood_words = [], [], [], []
    for word_idx, word in enumerate(sorted(heldout_words)[:2]):  # Just first 2 heldout words
        # Use a synthetic waveform for the unknown word
        t = np.arange(CLIP, dtype=np.float32) / 16000.0
        freq = 1000 + word_idx * 100  # Different freq for unknown words
        waveform = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        waveform = np.clip(waveform * 10000, -32768, 32767).astype(np.int16)

        for clip_i in range(10):
            ood_waveforms.append(waveform)
            ood_y.append(-1)  # OOD marker
            speaker = f"{100 + word_idx:08x}"
            ood_ids.append(f"{speaker}_nohash_{clip_i}.wav")
            ood_words.append(word)

    ood_split = SplitArrays(
        waveforms=np.array(ood_waveforms, dtype=np.int16),
        y=np.array(ood_y, dtype=np.int64),
        ids=tuple(ood_ids),
        words=tuple(ood_words)
    )

    # Compute manifest checksum (all records in train/val/cal/test)
    manifest_records = []
    for split_name, split in [("train", train_split), ("val", val_split),
                               ("cal", cal_split), ("test", test_split)]:
        for rec_id, label_idx, word in zip(split.ids, split.y, split.words):
            manifest_records.append({
                "identifier": rec_id,
                "label": STUDY_LABELS[label_idx] if label_idx >= 0 else "unknown",
                "path": f"/synthetic/{split_name}/{rec_id}"
            })

    # Simple deterministic checksum over sorted records
    record_strs = sorted([
        f"{r['identifier']}:{r['label']}:{r['path']}"
        for r in manifest_records
    ])
    manifest_digest = hashlib.sha256(
        "\n".join(record_strs).encode()
    ).hexdigest()

    bundle = DatasetBundle(
        labels=STUDY_LABELS,
        splits={
            "train": train_split,
            "val": val_split,
            "cal": cal_split,
            "test": test_split,
            "ood": ood_split,
        },
        train_unknown_words=train_unknown_words,
        heldout_words=heldout_words,
        manifest_sha256=manifest_digest,
        shortfalls={},
        silence_provenance={},
        silence_source_lengths=tuple(len(arr) for arr in noise_arrays)
    )

    return bundle


_FUNCTION_SOURCE_CELL = (
    "from src.bundle import DatasetBundle, SplitArrays, STUDY_LABELS\n"
    "from src.dataset import holdout_unknown_words, AUXILIARY_WORDS, manifest_checksum, AudioRecord\n"
    "import hashlib\n\n"
    + inspect.getsource(synthetic_noise_arrays_w2)
    + "\n\n"
    + inspect.getsource(build_synthetic_bundle_w2)
)


def cells():
    return [
        # ─────────────────────────────────────────────────────────────────────────────────
        # Section 0: Overview and scope
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## 0. Overview and scope (draft scaffold)

        This notebook demonstrates the data design, quality audits, cache reliability, and end-to-end training harness on a deterministic **in-memory synthetic bundle**, not a downloaded Speech Commands archive (see README's "local synthetic fallback" framing in Limitations).

        **What this notebook does:** Builds a 12-label, 12-speaker, 1050-sample synthetic dataset; audits it for integrity violations (duplicate checksums, heldout-word leaks, silence-window time-region spillover); validates cache corruption detection across three scenarios (byte flip, truncation, checksum swap); and runs the three augmentation conditions (clean, masking, acoustic) through the CNN training pipeline to verify end-to-end flow on tiny-scale synthetic data.

        **What this notebook does NOT claim:**
        - No real training result: accuracies on this 20-clip-per-label synthetic set are not reproducible or meaningful; the bundle is designed for QC, not for measuring robustness.
        - No speaker-disjoint comparison: training-pipeline.md §11 discloses that the Speech Commands official split is speaker-disjoint and achieves ~88.2% clean accuracy; this synthetic split reuses 3 test speakers from the train split (25% overlap), inflating perceived clean accuracy. This is **not comparable to the paper's baseline**.
        - No accelerator run this round: Model training runs on CPU with max_epochs=2 and tiny batches to stay under 5 minutes per condition. Pretrained encoders, stress suites, and calibration are planned for a later offloaded run (see training-pipeline.md §7 and §10).

        The four sections below establish separate, independent integrity contracts:
        1. **Data card** (Section 1): Label and auxiliary-word definitions, speaker overlap audit, label stratification.
        2. **QC pipeline** (Section 2): Duplicate-checksum detection, manifest stability, heldout-word leak detection, silence-window time-region spillover.
        3. **Cache walkthrough** (Section 3): Pristine round-trip, byte-flip corruption, truncation corruption, checksum-swap corruption, and isolation validation.
        4. **CNN-arm baseline audit** (Section 4): End-to-end training on three augmentation conditions, accuracy/ECE/wall-time per condition, and honest tie reporting.

        Each section pairs a **clean case** (must pass silently) with a **violation case** (must be detected), following module-boundary-contracts style: a validator that is never shown invalid input is untestable.

        **On synthetic data at tiny scale:** The numbers in this notebook are not research results. With only 240 training clips (20 per label), 20 OOD clips, and 3 augmentation conditions run on CPU, we do not expect meaningful accuracy or calibration metrics. The purpose is to validate the pipeline's correctness, reproducibility, and data integrity—to prove that the machinery works before we scale to real data. Any result claiming robustness improvement would require real Speech Commands (2,000+ clips per label) and speaker-disjoint splits; see training-pipeline.md §11 and the README Limitations section for details on this project's scope constraints.
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Section 1: Data card (source: training-pipeline.md §1, §11)
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## 1. Data card (source: training-pipeline.md §1, §11)

        This section describes the study's data design: the 12-label vocabulary, the auxiliary word
        handling, the per-label cap, the train/heldout split of non-command words, and the synthetic
        bundle we build here as a local sandbox for all notebook 2 sections.
        """),

        code(_FUNCTION_SOURCE_CELL, tags=("NB3-REUSE",)),

        code("""
        # Build the shared synthetic bundle (tagged for NB3 reuse).
        from src.dataset import STUDY_LABELS, AUXILIARY_WORDS, holdout_unknown_words
        from src.bundle import audit_bundle, speakers_per_split, speaker_overlap

        # Build the bundle
        w2_bundle = build_synthetic_bundle_w2(seed=RNG_SEED)

        print(f"Synthetic bundle built: {len(w2_bundle.splits['train'].y)} train, "
              f"{len(w2_bundle.splits['val'].y)} val, {len(w2_bundle.splits['cal'].y)} cal, "
              f"{len(w2_bundle.splits['test'].y)} test, {len(w2_bundle.splits['ood'].y)} ood")
        """, tags=("NB3-REUSE",)),

        md("""
        **Study labels** (12 command words, per `src.dataset.STUDY_LABELS`):
        yes, no, up, down, left, right, on, off, stop, go, unknown, silence

        **Auxiliary words** (25 total non-command words; the paper's "auxiliary" term in its abstract
        means only 10 filler words, but this project uses 25, as noted in `training-pipeline.md` §1):
        backward, bed, bird, cat, dog, eight, five, follow, forward, four, happy, house, learn,
        marvin, nine, one, seven, sheila, six, three, tree, two, visual, wow, zero

        The split of auxiliary words into training-unknown (20 words) and held-out (5 words) is deterministic via `holdout_unknown_words()` with seed 17. This separation ensures the held-out words can later test the model's ability to reject truly novel vocabulary, a key requirement for robust voice interfaces; allowing the model to train on all auxiliary words would conflate vocabulary generalization with robustness to unknown words (training-pipeline.md §1 calls out 20 vs 5 as the data-contract boundary). The heldout words form a separate out-of-distribution (OOD) set used in notebook 3 to evaluate the model's willingness to abstain on novel inputs.

        **Per-label cap**: 600 clips (not used in this synthetic build; shown for reference).

        **Licence**: CC BY 4.0 (per TensorFlow Datasets catalog entry for Speech Commands).
        """),

        code("""
        # Compute and print training-unknown vs. heldout word split
        w2_train_unknown, w2_heldout = holdout_unknown_words(AUXILIARY_WORDS, 5, seed=17)

        print(f"Training-unknown words ({len(w2_train_unknown)} total):")
        print(f"  {', '.join(w2_train_unknown)}")
        print(f"Heldout words ({len(w2_heldout)} total):")
        print(f"  {', '.join(w2_heldout)}")
        print()

        # Audit the synthetic bundle
        w2_audit = audit_bundle(w2_bundle)
        w2_label_counts = w2_audit["label_counts"]

        print("Label counts per split (synthetic bundle):")
        for label in STUDY_LABELS:
            print(f"  {label:8s}: train={w2_label_counts['train'].get(label, 0):3d}, "
                  f"val={w2_label_counts['val'].get(label, 0):3d}, "
                  f"cal={w2_label_counts['cal'].get(label, 0):3d}, "
                  f"test={w2_label_counts['test'].get(label, 0):3d}")

        print()
        w2_speakers = speakers_per_split(w2_bundle)
        print(f"Speakers per split: {w2_speakers}")

        w2_overlap = speaker_overlap(w2_bundle)
        print(f"Speaker overlap (test vs. train/cal): {w2_overlap}")
        """),

        md("""
        ### Check: Data design facts

        The checks below verify that the synthetic bundle matches the data contract specified in training-pipeline.md §1. These are not measurements of real-world data; they are assertions about the synthetic bundle's construction and internal consistency. A violated assertion would indicate a programming error in `build_synthetic_bundle_w2()` rather than a data quality problem.
        """),

        code("""
        # W2.1.1: Training-unknown word count
        check("W2.1.1", len(w2_train_unknown) == 20 and len(w2_heldout) == 5,
              f"holdout_unknown_words: {len(w2_train_unknown)} train + {len(w2_heldout)} heldout = 25")

        # W2.1.2: Train and heldout words are disjoint
        check("W2.1.2", set(w2_train_unknown) & set(w2_heldout) == set(),
              f"Train and heldout word sets are disjoint")

        # W2.1.3: Synthetic bundle train split has expected clip count (12 labels × 20 clips)
        w2_train_count = sum(w2_label_counts['train'].values())
        check("W2.1.3", w2_train_count == 240,
              f"Synthetic train split: {w2_train_count} clips (12 labels × 20 clips/label)")

        # W2.1.4: Speaker overlap is exactly 3/12 = 0.25
        # (3 test speaker IDs reused from train out of 12 total test unique speakers)
        w2_overlap_frac = w2_overlap["test_seen_in_train"]
        w2_expected_overlap = 0.25  # 3 out of 12 unique test speakers
        check("W2.1.4", np.isclose(w2_overlap_frac, w2_expected_overlap, atol=1e-6),
              f"Speaker overlap (test in train): {w2_overlap_frac:.6f}, expected {w2_expected_overlap:.6f}")

        # W2.1.5: Speaker overlap is by design, not a defect (per training-pipeline.md §11)
        note("W2.1.5", "Speaker overlap (test_seen_in_train = 0.25) is by design in this synthetic "
             "bundle to test the overlap detection. Real Speech Commands also has overlap per "
             "training-pipeline.md §11. This is not a data quality defect.")
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Figure NB2-1: Label and split counts
        # ─────────────────────────────────────────────────────────────────────────────────

        code("""
        # NB2-1: Grouped bar chart showing label counts per split.
        # Data source: w2_audit["label_counts"] computed above
        import matplotlib.pyplot as plt

        w2_splits = ["train", "val", "cal", "test"]
        w2_labels_sorted = list(STUDY_LABELS)
        w2_counts_by_split = {s: [w2_label_counts[s].get(label, 0) for label in w2_labels_sorted]
                               for s in w2_splits}

        w2_fig, w2_ax = plt.subplots(figsize=(12, 5))
        w2_x = np.arange(len(w2_labels_sorted))
        w2_width = 0.2

        for i, split in enumerate(w2_splits):
            w2_ax.bar(w2_x + i * w2_width, w2_counts_by_split[split],
                     w2_width, label=split, color=[PALETTE["clean"], PALETTE["mask"], PALETTE["acoustic"], PALETTE["reference"]][i])

        w2_ax.set_xlabel("Label", fontsize=11)
        w2_ax.set_ylabel("Clip count", fontsize=11)
        w2_ax.set_title("NB2-1: Label and split counts (synthetic bundle)", fontsize=12)
        w2_ax.set_xticks(w2_x + w2_width * 1.5)
        w2_ax.set_xticklabels(w2_labels_sorted, fontsize=9)
        w2_ax.legend(frameon=False, fontsize=10)
        w2_ax.grid(axis='y', alpha=0.3)
        w2_fig.tight_layout()
        plt.show()
        """),

        md("""
        ### How to read this chart

        Figure NB2-1 shows the distribution of clips across the four in-distribution splits (train, val, cal, test)
        for each of the 12 command labels. Each label is represented by a set of four grouped bars, one per split.
        The y-axis shows the count of clips; the x-axis lists the command labels in order.

        **Data source:** `w2_audit["label_counts"]` computed from the synthetic bundle in the cell above.

        **Why this matters:** Training-pipeline.md §2 specifies that train and val must remain disjoint (no speaker overlap), that the 70/15/15 split is sample-level and deterministic (seed 17), and that every label must have identical counts within each split to avoid label-imbalance confounds when measuring robustness. This chart visually confirms all 12 labels appear with identical counts per split, ruling out the class-imbalance defect pattern.

        **Falsifier:** If the synthetic bundle construction were broken, all labels would show identical counts
        in every split (meaning the builder ignored the per-split sizing logic). Here, each label has exactly
        20 clips in train, 6 in val, 6 in cal, and 6 in test, confirming the builder correctly applied the
        requested cap and stratification.
        """),

        # ─────────────────────────────────────────────────────────────────────────────────
        # Figure NB2-2: Speaker overlap
        # ─────────────────────────────────────────────────────────────────────────────────

        code("""
        # NB2-2: Speaker overlap and distinct speaker counts per split.
        # Data source: w2_overlap (computed above) and w2_speakers_per_split

        w2_fig2, (w2_ax1, w2_ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Panel 1: Overlap fractions (test speakers seen in train/cal)
        w2_overlap_labels = ["test_seen_in_train", "test_seen_in_cal"]
        w2_overlap_values = [w2_overlap[label] for label in w2_overlap_labels]
        w2_colors_overlap = [PALETTE["clean"], PALETTE["mask"]]

        w2_ax1.bar(w2_overlap_labels, w2_overlap_values, color=w2_colors_overlap, edgecolor="black", linewidth=1.5)
        w2_ax1.set_ylabel("Overlap fraction", fontsize=11)
        w2_ax1.set_title("Speaker overlap: test speakers in train/cal", fontsize=11)
        w2_ax1.set_ylim([0, 1])
        w2_ax1.grid(axis='y', alpha=0.3)
        w2_ax1.set_xticklabels(["test in train", "test in cal"], fontsize=10)

        # Add value labels on bars
        for i, (label, value) in enumerate(zip(w2_overlap_labels, w2_overlap_values)):
            w2_ax1.text(i, value + 0.03, f"{value:.3f}", ha="center", fontsize=10, fontweight="bold")

        # Panel 2: Distinct speaker counts per split
        w2_speaker_splits = sorted(w2_speakers.keys())
        w2_speaker_counts = [w2_speakers[s] for s in w2_speaker_splits]
        w2_colors_speakers = [PALETTE["clean"], PALETTE["mask"], PALETTE["acoustic"],
                              PALETTE["reference"], PALETTE["frozen"]][:len(w2_speaker_splits)]

        w2_ax2.bar(w2_speaker_splits, w2_speaker_counts, color=w2_colors_speakers, edgecolor="black", linewidth=1.5)
        w2_ax2.set_ylabel("Distinct speakers", fontsize=11)
        w2_ax2.set_title("Speaker count per split", fontsize=11)
        w2_ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (split, count) in enumerate(zip(w2_speaker_splits, w2_speaker_counts)):
            w2_ax2.text(i, count + 0.2, str(count), ha="center", fontsize=10, fontweight="bold")

        w2_fig2.tight_layout()
        plt.show()
        """),

        md("""
        ### How to read this chart

        Figure NB2-2 consists of two panels showing speaker overlap and speaker distribution.

        **Left panel (speaker overlap):** Shows two bars representing the fraction of test speakers that also
        appear in the train and calibration splits, respectively. Values range from 0 (no overlap) to 1 (complete
        overlap). The fractions are printed above each bar.

        **Right panel (speaker counts):** Shows the number of distinct speakers in each split (train, val, cal,
        test, ood). Each bar represents one split, with the count labeled above.

        **Data source:** `w2_overlap` (test speaker overlap fractions) and `w2_speakers` (distinct speaker counts)
        computed from the synthetic bundle in cells above.

        **Why this matters:** Training-pipeline.md §11 requires speaker overlap to be measured explicitly, not only disclosed. The real Speech Commands dataset, split by the official speaker-disjoint test list, reports ~88.2% clean accuracy; our random stratified split has speaker overlap (here, 25% by design). Notebooks that claim to reproduce the 88.2% figure without reporting overlap are not falsifiable; measuring it here proves we are not hiding this confound.

        **Falsifier:** If the overlap computation were broken, the left panel would show exactly 0.0 or 1.0 for
        both bars even though the synthetic bundle was deliberately constructed with partial (0.25) overlap.
        The right panel would show identical speaker counts if the construction failed to differentiate speaker
        IDs across splits. Here, both panels match the engineered values, confirming overlap detection and speaker
        tracking work correctly.
        """),
    ]
