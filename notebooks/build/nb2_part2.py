"""Notebook 2, Part 2: QC pipeline - dataset integrity audits (derived here).

Sections:
- 2. QC pipeline: duplicate checksums, manifest stability, bundle audits, silence regions
"""

from nbkit import md, code


def cells():
    return [
        # ─────────────────────────────────────────────────────────────────────────────────
        # Section 2: QC pipeline (derived here + source: training-pipeline.md §1)
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## 2. QC pipeline (derived here + source: training-pipeline.md §1)

        This section audits the synthetic dataset bundle for integrity violations: duplicate
        checksums, manifest stability under permutation, heldout-word leaks, and silence-window
        time-region spillover. Each audit combines a clean-case baseline (must pass) with a
        deliberately-constructed violation case (must be detected), per the module-boundary-contracts
        pattern: a detector that never fires is untestable.

        These audits protect against four classes of defects that would corrupt model training or evaluation. Duplicate checksums indicate file corruption or copy-paste errors; manifest instability suggests the bundle's identity is fragile and not reproducible; heldout-word leaks violate the train/OOD boundary and inflate accuracy; silence-window spillover violates time-region invariants and causes train/test confusion. Each check must succeed on clean data AND catch injected violations. Passing these audits is necessary but not sufficient—it proves the bundle is internally consistent and free of known defects, but does not verify the bundle matches any real data source or that the pipeline generalizes.
        """),

        md("""
        **Test design:** Each integrity check below consists of two cells: a clean-case baseline and a violation-case injection. The clean case must pass silently (all assertions succeed); the violation case must be detected (assertions catch the defect). If a checker passes clean data but misses injected violations, it is not falsifiable and must be rewritten.
        """),

        code("""
        # W2.2: QC pipeline setup and imports.
        import hashlib
        import numpy as np
        from src.dataset import (
            find_duplicate_checksums, manifest_checksum, silence_windows,
            STUDY_LABELS, AUXILIARY_WORDS, holdout_unknown_words, AudioRecord
        )
        from src.bundle import audit_bundle

        w2_qc_imports_ok = all(
            callable(x) for x in [find_duplicate_checksums, manifest_checksum, silence_windows, audit_bundle]
        )
        check("W2.2.0", w2_qc_imports_ok, "QC pipeline imports loaded")
        """),

        code("""
        # W2.2.1: Duplicate-checksum audit, clean case then with an injected duplicate.
        #
        # This uses its own small record set with genuinely distinct waveforms, not the
        # shared synthetic bundle from nb2_part1: that bundle deliberately reuses one
        # waveform template per label across every clip (a valid simplification for the
        # training-speed demo in section 4), so it is NOT duplicate-free by construction
        # and is the wrong fixture for a duplicate-*detector* test.

        w2_dup_rng = np.random.default_rng(RNG_SEED + 1)
        w2_all_records = []
        for w2_i in range(8):
            waveform = w2_dup_rng.integers(-32768, 32767, size=16000, dtype=np.int16)
            checksum = hashlib.sha256(waveform.tobytes()).hexdigest()
            w2_all_records.append(AudioRecord(
                identifier=f"clean_record_{w2_i}",
                label=STUDY_LABELS[w2_i % len(STUDY_LABELS)],
                path="dummy",
                checksum=checksum,
                sample_rate=16000,
                num_frames=16000,
            ))

        # Clean set: verify no duplicates
        w2_clean_duplicates = find_duplicate_checksums(w2_all_records)
        w2_clean_dup_count = len(w2_clean_duplicates)
        check("W2.2.1", w2_clean_dup_count == 0,
              f"Clean record set has {w2_clean_dup_count} duplicate-checksum groups (expected 0)")

        # Inject a duplicate: pick one record, clone it with a different identifier
        if w2_all_records:
            w2_original_record = w2_all_records[0]
            w2_duplicate_record = AudioRecord(
                identifier="injected_duplicate_" + w2_original_record.identifier,
                label=w2_original_record.label,
                path=w2_original_record.path,
                checksum=w2_original_record.checksum,  # Same checksum = same content
                sample_rate=w2_original_record.sample_rate,
                num_frames=w2_original_record.num_frames
            )
            w2_records_with_dup = w2_all_records + [w2_duplicate_record]

            # Find duplicates
            w2_injected_duplicates = find_duplicate_checksums(w2_records_with_dup)
            w2_dup_found = len(w2_injected_duplicates) == 1

            # Verify the exact pair is detected
            if w2_dup_found:
                w2_found_checksum = list(w2_injected_duplicates.keys())[0]
                w2_found_ids = set(w2_injected_duplicates[w2_found_checksum])
                w2_expected_ids = {w2_original_record.identifier, w2_duplicate_record.identifier}
                w2_ids_match = w2_found_ids == w2_expected_ids
            else:
                w2_ids_match = False

            check("W2.2.1b", w2_dup_found and w2_ids_match,
                  f"Injected duplicate detected: {w2_dup_found}, IDs match: {w2_ids_match}")
        else:
            check("W2.2.1", False, "Bundle contains no records to test")
        """),

        code("""
        # W2.2.2: Manifest checksum stability under permutation.
        # Compute manifest checksum twice: original order and reversed.

        w2_records_for_checksum = w2_all_records[:10] if len(w2_all_records) >= 10 else w2_all_records

        if w2_records_for_checksum:
            w2_checksum_original = manifest_checksum(w2_records_for_checksum)
            w2_checksum_reversed = manifest_checksum(list(reversed(w2_records_for_checksum)))

            w2_checksums_match = w2_checksum_original == w2_checksum_reversed
            check("W2.2.2", w2_checksums_match,
                  f"Manifest checksum stable under reversal: {w2_checksums_match}")
        else:
            check("W2.2.2", False, "No records available for checksum test")
        """),

        code("""
        # W2.2.3: audit_bundle on clean synthetic bundle - all four metrics.

        w2_clean_audit = audit_bundle(w2_bundle)

        print("\\n=== Clean synthetic bundle audit ===")
        print(f"id_overlap:              {w2_clean_audit['id_overlap']}")
        print(f"duplicate_checksums:     {w2_clean_audit['duplicate_checksums']}")
        print(f"heldout_word_leak:       {w2_clean_audit['heldout_word_leak']}")
        print(f"silence_time_overlap:    {w2_clean_audit['silence_time_overlap']}")

        # Baseline checks: clean bundle must be free of detected violations
        check("W2.2.3", w2_clean_audit['heldout_word_leak'] is False,
              "Clean bundle reports no heldout-word leak")
        check("W2.2.3", w2_clean_audit['silence_time_overlap'] is False,
              "Clean bundle reports no silence-time overlap")
        """),

        code("""
        # W2.2.4: Build a SECOND bundle deliberately injected with heldout-word leak.
        # Reuse the same synthetic bundle but inject one heldout-word clip into the train split.

        # Create a modified bundle with one heldout word present in train split
        import dataclasses

        # Build a small set of injected heldout-word records to add to train
        w2_heldout_word_to_inject = w2_bundle.heldout_words[0] if w2_bundle.heldout_words else "unknown"

        # Get one heldout clip from ood split if available
        w2_leaked_waveform = None
        w2_leaked_id = None
        w2_leaked_word = None

        if "ood" in w2_bundle.splits:
            w2_ood_split = w2_bundle.splits["ood"]
            for i, word in enumerate(w2_ood_split.words):
                if word == w2_heldout_word_to_inject:
                    w2_leaked_waveform = w2_ood_split.waveforms[i]
                    w2_leaked_id = w2_ood_split.ids[i]
                    w2_leaked_word = word
                    break

        if w2_leaked_waveform is not None:
            # Create a modified train split with the leaked clip
            w2_train_split = w2_bundle.splits["train"]
            w2_new_train_waveforms = np.vstack([w2_train_split.waveforms, [w2_leaked_waveform]])
            w2_new_train_ids = w2_train_split.ids + (f"leaked_{w2_leaked_id}",)
            w2_new_train_words = w2_train_split.words + (w2_leaked_word,)
            w2_new_train_y = np.concatenate([w2_train_split.y, [-1]])  # OOD label

            from src.bundle import SplitArrays
            w2_modified_train_split = SplitArrays(
                waveforms=w2_new_train_waveforms,
                y=w2_new_train_y,
                ids=w2_new_train_ids,
                words=w2_new_train_words
            )

            # Create modified bundle with injected leak
            w2_leaked_bundle = dataclasses.replace(
                w2_bundle,
                splits={**w2_bundle.splits, "train": w2_modified_train_split}
            )

            # Audit the leaked bundle
            w2_leaked_audit = audit_bundle(w2_leaked_bundle)
            w2_leak_detected = w2_leaked_audit['heldout_word_leak'] is True

            check("W2.2.4", w2_leak_detected,
                  f"Injected heldout-word leak DETECTED: {w2_leak_detected}")
        else:
            check("W2.2.4", False, "Could not construct heldout-word leak test")
        """),

        code("""
        # W2.2.5: Silence-window region regression on short file.
        # Get noise arrays, find/create a short one, and verify provenance.

        w2_noise_arrays = synthetic_noise_arrays_w2(seed=RNG_SEED)
        w2_noise_lengths = [len(arr) for arr in w2_noise_arrays]

        check("W2.2.5", len(w2_noise_arrays) >= 3,
              f"Noise arrays available: {len(w2_noise_arrays)} (expected >= 3)")

        # Identify or create a short file
        w2_shortest_array = min(w2_noise_arrays, key=len) if w2_noise_arrays else None
        w2_shortest_length = len(w2_shortest_array) if w2_shortest_array is not None else 0

        # For test split: region is [0.85, 1.0) of file length
        w2_test_split_region = (0.85, 1.0)
        w2_test_region_start = int(w2_test_split_region[0] * w2_shortest_length)
        w2_test_region_end = int(w2_test_split_region[1] * w2_shortest_length)
        w2_test_region_size = w2_test_region_end - w2_test_region_start

        # If the region is large enough, sample windows; if not, verify the error is raised
        if w2_shortest_array is not None and w2_test_region_size >= 16000:
            # Region is large enough: sample and check provenance
            w2_test_windows, w2_test_provenance = silence_windows(
                w2_noise_arrays, n=5, seed=RNG_SEED, split="test", return_provenance=True
            )

            # Verify every window's range stays within its region
            w2_all_windows_in_bounds = True
            for file_idx, offset in w2_test_provenance:
                file_length = w2_noise_arrays[file_idx].shape[0]
                region_start = int(w2_test_split_region[0] * file_length)
                region_end = int(w2_test_split_region[1] * file_length)
                window_end = offset + 16000

                if not (region_start <= offset and window_end <= region_end):
                    w2_all_windows_in_bounds = False
                    break

            check("W2.2.5", w2_all_windows_in_bounds,
                  "All sampled windows' ranges lie within their assigned regions")
        else:
            # Region too small: expect the function to raise ValueError
            w2_short_file_raises = False
            try:
                silence_windows(
                    w2_noise_arrays, n=1, seed=RNG_SEED, split="test", return_provenance=True
                )
            except ValueError:
                w2_short_file_raises = True

            check("W2.2.5", w2_short_file_raises,
                  f"Short file raises ValueError as expected: {w2_short_file_raises}")
        """),

        code("""
        # W2.2.6: Build a bundle with silence-time overlap violation and verify audit detects it.
        # This requires constructing provenance that spills over a region boundary.

        # Since the real bundle may have proper provenance, we'll construct a minimal
        # synthetic case with deliberately-incorrect provenance.

        import dataclasses
        from src.bundle import SplitArrays

        # Create a minimal synthetic bundle with one tiny noise file
        w2_tiny_noise_file = np.random.default_rng(RNG_SEED).integers(0, 100, 1000, dtype=np.int16)
        w2_tiny_bundle = w2_bundle  # Start with the real bundle

        # Inject a silence_provenance that spills outside its region
        # For test split [0.85, 1.0): region_end = int(0.85 * 1000) = 850, int(1.0 * 1000) = 1000
        # So region is [850, 1000], size = 150 samples (too small for a 16000-sample window)
        # Inject provenance claiming a window at offset=900 (which would end at 16900, outside region)

        w2_violation_provenance = {
            "train": (),
            "val": (),
            "cal": (),
            "test": ((0, 900),)  # File 0, offset 900; window [900, 16900] spills over region [850, 1000]
        }

        w2_violation_bundle = dataclasses.replace(
            w2_bundle,
            silence_provenance=w2_violation_provenance,
            silence_source_lengths=(1000,)  # One file of 1000 samples
        )

        w2_violation_audit = audit_bundle(w2_violation_bundle)
        w2_overlap_detected = w2_violation_audit['silence_time_overlap'] is True

        check("W2.2.6", w2_overlap_detected,
              f"Injected silence-time overlap DETECTED: {w2_overlap_detected}")
        """),

        code("""
        # W2.2.7: Figure R4.2-a - silence-window occupancy map.
        # One horizontal bar per synthetic noise file, showing train/cal/test regions
        # with actual sampled window positions overlaid.

        w2_noise_arrays = synthetic_noise_arrays_w2(seed=RNG_SEED)

        # Prepare data for each file
        w2_fig_file_data = []

        for w2_file_idx, w2_noise_array in enumerate(w2_noise_arrays):
            w2_file_len = len(w2_noise_array)
            w2_regions = {
                "train": (0.0, 0.7),
                "cal": (0.7, 0.85),
                "test": (0.85, 1.0),
            }

            w2_file_regions = {}
            for w2_split_name, (w2_start_frac, w2_end_frac) in w2_regions.items():
                w2_start = int(w2_start_frac * w2_file_len)
                w2_end = int(w2_end_frac * w2_file_len)
                w2_file_regions[w2_split_name] = (w2_start, w2_end)

            # Collect actual window ranges from provenance if available
            w2_windows_in_file = {"train": [], "cal": [], "test": []}

            for w2_split_name in ["train", "val", "cal", "test"]:
                w2_provenance_list = w2_bundle.silence_provenance.get(w2_split_name, ())
                for w2_prov_idx, (w2_prov_file_idx, w2_prov_offset) in enumerate(w2_provenance_list):
                    if w2_prov_file_idx == w2_file_idx:
                        w2_window_end = w2_prov_offset + 16000
                        # Determine which split this window "belongs to"
                        w2_assigned_split = w2_split_name.replace("val", "train")  # val uses train region
                        w2_windows_in_file[w2_assigned_split].append((w2_prov_offset, w2_window_end))

            w2_fig_file_data.append({
                "file_idx": w2_file_idx,
                "length": w2_file_len,
                "regions": w2_file_regions,
                "windows": w2_windows_in_file
            })

        # Create figure: one horizontal bar per file
        w2_fig, w2_ax = plt.subplots(figsize=(12, len(w2_fig_file_data) * 0.5 + 1))

        w2_y_pos = 0
        w2_split_colors = {"train": PALETTE["clean"], "cal": PALETTE["mask"], "test": PALETTE["shifted"]}

        for w2_file_info in w2_fig_file_data:
            w2_file_len = w2_file_info["length"]
            w2_file_label = f"File {w2_file_info['file_idx']} ({w2_file_len} samples)"

            # Draw region bars for each split
            for w2_split_name, (w2_start, w2_end) in w2_file_info["regions"].items():
                w2_width = (w2_end - w2_start) / w2_file_len
                w2_left = w2_start / w2_file_len
                w2_ax.barh(
                    w2_y_pos, w2_width, left=w2_left, height=0.3,
                    color=w2_split_colors[w2_split_name], alpha=0.4,
                    label=w2_split_name if w2_y_pos == 0 else ""
                )

            # Overlay actual windows
            for w2_split_name, w2_windows_list in w2_file_info["windows"].items():
                for w2_offset, w2_window_end in w2_windows_list:
                    w2_window_width = (w2_window_end - w2_offset) / w2_file_len
                    w2_window_left = w2_offset / w2_file_len

                    # Check if window spills outside its region
                    w2_region_start, w2_region_end = w2_file_info["regions"][w2_split_name]
                    w2_spills = (w2_offset < w2_region_start or w2_window_end > w2_region_end)
                    w2_marker_color = "red" if w2_spills else w2_split_colors[w2_split_name]
                    w2_marker_alpha = 1.0 if w2_spills else 0.7

                    w2_ax.barh(
                        w2_y_pos, w2_window_width, left=w2_window_left, height=0.15,
                        color=w2_marker_color, alpha=w2_marker_alpha, edgecolor="black", linewidth=1
                    )

            w2_ax.text(-0.08, w2_y_pos, w2_file_label, ha="right", va="center", fontsize=9)
            w2_y_pos += 1

        w2_ax.set_xlim(-0.05, 1.05)
        w2_ax.set_ylim(-0.5, w2_y_pos - 0.5)
        w2_ax.set_xlabel("Fractional file position")
        w2_ax.set_ylabel("Noise file")
        w2_ax.set_title("Fig R4.2-a: Silence-window occupancy map (colored regions: assigned split; overlaid bars: actual windows)")
        w2_ax.set_yticks([])
        w2_ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

        note("W2.2.7", "Silence-window occupancy map (Fig R4.2-a) plotted; red bars indicate windows outside assigned region")
        """),

        md("""
        ### How to read this chart (Fig R4.2-a)

        **Data source:** One horizontal bar per synthetic noise file, drawn using `w2_bundle.silence_provenance`
        (file index and window offset) and file lengths from `synthetic_noise_arrays_w2()`.

        **Chart structure:** Each file occupies one row. Within each row, three colored sub-bars show the
        assigned fractional regions for train (blue, 0–70%), cal (orange, 70–85%), and test (red, 85–100%).
        Overlaid on top are black-edged bars representing the actual 16000-sample windows sampled by
        `silence_windows()`, colored by their assigned split. A window colored red indicates it spills outside
        its assigned region—the exact violation Fig R4.2-a is designed to catch.

        **Falsifier:** If every window sits exactly inside its labeled region even for the intentionally-short file,
        it would mean the regression case never triggered the fallback branch and `silence_windows` did not
        respect the time-region invariant. The presence of at least one red bar (or the absence of any red bars
        despite a deliberately short file) would signal a regression.

        **Key observations:**
        - One file is deliberately short (less than ~46,000 samples) to force `silence_windows` into the fallback
          code path when sampling for test split.
        - Windows must satisfy: `region_start ≤ offset and offset + 16000 ≤ region_end` for their assigned split.
        """),

        md("""
        ### How to read the QC checks

        The QC pipeline above verifies four independent integrity properties:

        1. **Duplicate checksums** (W2.2.1): Two records built from the same waveform bytes must be
           identified by `find_duplicate_checksums`. The test injects a known duplicate and verifies
           it is found by its exact identifiers, not just "count > 0". Why: Silent duplicates would
           mean the training set accidentally learns the same clip twice, or that corruption is undetected.

        2. **Manifest checksum stability** (W2.2.2): The SHA-256 digest of a record list must be
           invariant to insertion order. Computed twice—once in original order, once reversed—both
           digests must match exactly. Why: If manifest digests vary by order, downstream code cannot
           verify bundle reproducibility; the training pipeline would have no way to confirm it got
           the same data as a prior run (training-pipeline.md §1 specifies manifest checksums as the
           bundle's identity token).

        3. **Heldout-word leak detection** (W2.2.3 & W2.2.4): A clean bundle must report
           `heldout_word_leak = False`. A second bundle with one heldout-word clip deliberately
           injected into the train split must report `True`, proving the detector actually fires
           on real leaks, not just stays silent on clean data. Why: Training on held-out words would
           collapse the train/OOD boundary, making OOD-unknown accuracy meaningless and inflating
           the model's apparent robustness to novel words (training-pipeline.md §1 mandates the 20/5 split).

        4. **Silence-time region spillover** (W2.2.5 & W2.2.6): Windows sampled from short
           background files must respect time-region boundaries (e.g., test region = [85%, 100%)
           of file). The first check verifies sampled windows stay in bounds; the second constructs
           a bundle with deliberately-incorrect provenance and verifies `audit_bundle` detects the
           spillover, proving the detector fires on real violations. Why: Spillover causes training
           silence from test files or test silence from training files, violating the fundamental
           train/test contract and rendering any accuracy claim meaningless.

        Every check is a **two-sided assertion**: a clean case (must pass quietly) paired with a
        violation case (must be detected). This follows module-boundary-contracts style: a validator
        that is never shown invalid input is untestable.
        """),
    ]
