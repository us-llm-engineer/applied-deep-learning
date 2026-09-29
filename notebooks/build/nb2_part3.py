"""Notebook 2, Part 3: Cache walkthrough (derived here).

Sections:
- 3. Cache walkthrough (derived here)
"""

from nbkit import md, code


def cells():
    return [
        # ─────────────────────────────────────────────────────────────────────────────────
        # Section 3: Cache walkthrough (derived here)
        # ─────────────────────────────────────────────────────────────────────────────────

        md("""
        ## 3. Cache walkthrough (derived here)

        This section demonstrates the reliability of the cache storage system by performing a pristine round-trip (write and read back a feature array) and then corrupting cache files in three ways to verify that the cache correctly detects corruption and leaves unaffected entries intact.

        Training-pipeline.md §1 specifies that waveform caches must include per-file checksums and raise exceptions on corrupted reads, never silently returning corrupted data. This section validates that `CacheStore` honors that contract: unaffected entries remain readable even when a sibling entry is corrupted, proving corruption is isolated at the file level, not leaked.
        """),

        md("""
        **Test structure:** Like the QC pipeline (Section 2), this section uses a two-sided test design: a clean round-trip (must succeed) followed by three corruption scenarios (must be caught). The isolation check (verifying sibling entries are unaffected) is critical because a cache system that detects corruption but spreads it to neighboring entries is worse than useless. Data integrity is a precondition for any reproducible result; silent corruption is the worst possible failure mode.
        """),

        # Pristine round-trip
        code("""
        # W2.3.1: Pristine cache round-trip — write a small feature array and read it back.
        import tempfile
        from pathlib import Path
        from src.cache import CacheStore, CacheProvenance

        w2_tmp_dir = Path(tempfile.mkdtemp(prefix="cache_walkthrough_"))
        w2_store = CacheStore(w2_tmp_dir)

        # Create a small synthetic feature array.
        w2_original_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        w2_provenance = CacheProvenance(
            identifier="pristine_test",
            transform={"kind": "identity"},
            source_checksum="abc123"
        )

        # Write to cache.
        w2_store.write("pristine_key", w2_original_value, w2_provenance)

        # Read back from cache.
        w2_read_value = w2_store.read("pristine_key", w2_provenance)

        # Check bit-exact equality.
        w2_pristine_equal = np.array_equal(w2_read_value, w2_original_value)
        check("W2.3.1", w2_pristine_equal, f"Pristine round-trip: original shape {w2_original_value.shape}, read back identical")
        """),

        # Corruption scenario 1: byte flip
        code("""
        # W2.3.2: Byte-flip corruption — flip one byte inside entry A's file, verify read raises.
        import zipfile
        import zlib

        w2_store_1 = CacheStore(w2_tmp_dir)

        # Write two entries.
        w2_a_value = np.random.default_rng(42).standard_normal((16, 64)).astype(np.float32)
        w2_b_value = np.random.default_rng(43).standard_normal((16, 64)).astype(np.float32)
        w2_prov_a = CacheProvenance(identifier="corrupt_a", transform={"kind": "gain", "gain_db": 3.0}, source_checksum="a1")
        w2_prov_b = CacheProvenance(identifier="corrupt_b", transform={"kind": "gain", "gain_db": 3.0}, source_checksum="b2")

        w2_store_1.write("entry_a", w2_a_value, w2_prov_a)
        w2_store_1.write("entry_b", w2_b_value, w2_prov_b)

        # Corrupt entry A's file by flipping one byte at 30% offset.
        w2_a_path = w2_tmp_dir / "entry_a.npz"
        w2_a_data = bytearray(w2_a_path.read_bytes())
        w2_a_data[int(len(w2_a_data) * 0.3)] ^= 0xFF
        w2_a_path.write_bytes(bytes(w2_a_data))

        # Try to read corrupted entry A — expect exception.
        w2_byte_flip_raised = False
        w2_byte_flip_exception_type = None
        try:
            w2_store_1.read("entry_a", w2_prov_a)
        except (ValueError, zipfile.BadZipFile, zlib.error, EOFError, OSError) as w2_e:
            w2_byte_flip_raised = True
            w2_byte_flip_exception_type = type(w2_e).__name__

        check("W2.3.2", w2_byte_flip_raised, f"Byte-flip corruption raised {w2_byte_flip_exception_type}")
        """),

        code("""
        # W2.3.3: Byte-flip scenario — verify sibling entry B still reads correctly.
        w2_b_read_value = w2_store_1.read("entry_b", w2_prov_b)
        w2_b_unaffected = np.array_equal(w2_b_read_value, w2_b_value)
        check("W2.3.3", w2_b_unaffected, "Sibling entry B unaffected by byte-flip corruption to A")
        """),

        # Corruption scenario 2: truncation
        code("""
        # W2.3.4: Truncated cache file — truncate entry A's file, verify read raises.
        w2_store_2 = CacheStore(w2_tmp_dir)

        # Write two entries for truncation test.
        w2_a2_value = np.random.default_rng(44).standard_normal((16, 64)).astype(np.float32)
        w2_b2_value = np.random.default_rng(45).standard_normal((16, 64)).astype(np.float32)
        w2_prov_a2 = CacheProvenance(identifier="trunc_a", transform={"kind": "none"}, source_checksum="a3")
        w2_prov_b2 = CacheProvenance(identifier="trunc_b", transform={"kind": "none"}, source_checksum="b4")

        w2_store_2.write("entry_a2", w2_a2_value, w2_prov_a2)
        w2_store_2.write("entry_b2", w2_b2_value, w2_prov_b2)

        # Truncate entry A2's file to 50% of its size.
        w2_a2_path = w2_tmp_dir / "entry_a2.npz"
        w2_a2_data = w2_a2_path.read_bytes()
        w2_a2_path.write_bytes(w2_a2_data[: int(len(w2_a2_data) * 0.5)])

        # Try to read truncated entry A2 — expect exception.
        w2_truncation_raised = False
        w2_truncation_exception_type = None
        try:
            w2_store_2.read("entry_a2", w2_prov_a2)
        except (ValueError, zipfile.BadZipFile, zlib.error, EOFError, OSError) as w2_e:
            w2_truncation_raised = True
            w2_truncation_exception_type = type(w2_e).__name__

        check("W2.3.4", w2_truncation_raised, f"Truncation corruption raised {w2_truncation_exception_type}")
        """),

        code("""
        # W2.3.5: Truncation scenario — verify sibling entry B2 still reads correctly.
        w2_b2_read_value = w2_store_2.read("entry_b2", w2_prov_b2)
        w2_b2_unaffected = np.array_equal(w2_b2_read_value, w2_b2_value)
        check("W2.3.5", w2_b2_unaffected, "Sibling entry B2 unaffected by truncation of A2")
        """),

        # Corruption scenario 3: checksum-swap
        code("""
        # W2.3.6: Payload-swap-behind-checksum — modify payload but keep old checksum.
        w2_store_3 = CacheStore(w2_tmp_dir)

        # Write two entries for checksum-swap test.
        w2_a3_original = np.random.default_rng(46).standard_normal((16, 64)).astype(np.float32)
        w2_b3_value = np.random.default_rng(47).standard_normal((16, 64)).astype(np.float32)
        w2_prov_a3 = CacheProvenance(identifier="checksum_a", transform={"kind": "test"}, source_checksum="a5")
        w2_prov_b3 = CacheProvenance(identifier="checksum_b", transform={"kind": "test"}, source_checksum="b6")

        w2_store_3.write("entry_a3", w2_a3_original, w2_prov_a3)
        w2_store_3.write("entry_b3", w2_b3_value, w2_prov_b3)

        # Extract provenance and checksum from entry A3's npz.
        w2_a3_path = w2_tmp_dir / "entry_a3.npz"
        with np.load(w2_a3_path, allow_pickle=False) as w2_stored:
            w2_stored_prov = w2_stored["provenance"]
            w2_stored_checksum = w2_stored["checksum"]

        # Modify the payload.
        w2_a3_tampered = w2_a3_original.copy()
        w2_a3_tampered[0, 0] += 1.0

        # Save tampered payload with original checksum.
        np.savez_compressed(w2_a3_path, value=w2_a3_tampered, provenance=w2_stored_prov, checksum=w2_stored_checksum)

        # Try to read — should raise checksum ValueError.
        w2_checksum_raised = False
        w2_checksum_exception_type = None
        try:
            w2_store_3.read("entry_a3", w2_prov_a3)
        except (ValueError, zipfile.BadZipFile, zlib.error, EOFError, OSError) as w2_e:
            w2_checksum_raised = True
            w2_checksum_exception_type = type(w2_e).__name__

        check("W2.3.6", w2_checksum_raised, f"Checksum-swap corruption raised {w2_checksum_exception_type}")
        """),

        code("""
        # W2.3.7: Checksum-swap scenario — verify sibling entry B3 still reads correctly.
        w2_b3_read_value = w2_store_3.read("entry_b3", w2_prov_b3)
        w2_b3_unaffected = np.array_equal(w2_b3_read_value, w2_b3_value)
        check("W2.3.7", w2_b3_unaffected, "Sibling entry B3 unaffected by checksum-swap of A3")
        """),

        # Summary table
        code("""
        # W2.3.8: Summary table of cache corruption outcomes.
        w2_corruption_results = [
            {"scenario": "Byte flip", "corrupted_raised": w2_byte_flip_raised, "sibling_ok": w2_b_unaffected, "exception": w2_byte_flip_exception_type},
            {"scenario": "Truncation", "corrupted_raised": w2_truncation_raised, "sibling_ok": w2_b2_unaffected, "exception": w2_truncation_exception_type},
            {"scenario": "Checksum swap", "corrupted_raised": w2_checksum_raised, "sibling_ok": w2_b3_unaffected, "exception": w2_checksum_exception_type},
        ]

        print("\\nCache Corruption Test Results:")
        print("─" * 80)
        print(f"{'Scenario':<20} | {'Corrupted Read Raised':<21} | {'Sibling OK':<12} | {'Exception Type':<20}")
        print("─" * 80)
        for w2_r in w2_corruption_results:
            print(f"{w2_r['scenario']:<20} | {str(w2_r['corrupted_raised']):<21} | {str(w2_r['sibling_ok']):<12} | {w2_r['exception']:<20}")
        print("─" * 80)
        """),

        # Figure NB2-3: cache corruption outcomes grid
        code("""
        # Figure NB2-3: Cache corruption detection outcomes grid.
        # Rows: 3 corruption scenarios; Columns: {corrupted key, sibling key}
        # Cell value: 1 if read raised (corruption detected), 0 if read succeeded (not corrupted)

        w2_scenario_names = ["Byte flip", "Truncation", "Checksum swap"]
        # First column: 1 if corrupted key raised (expected), 0 if succeeded (unexpected)
        # Second column: 0 if sibling read succeeded (expected), 1 if raised (corruption leaked—bad)
        w2_outcomes = np.array([
            [int(w2_byte_flip_raised), int(not w2_b_unaffected)],
            [int(w2_truncation_raised), int(not w2_b2_unaffected)],
            [int(w2_checksum_raised), int(not w2_b3_unaffected)],
        ])

        w2_fig, w2_ax = plt.subplots(figsize=(7, 4))
        w2_im = w2_ax.imshow(w2_outcomes, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

        w2_ax.set_xticks([0, 1])
        w2_ax.set_xticklabels(["Corrupted Key", "Sibling Key"], fontsize=11)
        w2_ax.set_yticks(range(len(w2_scenario_names)))
        w2_ax.set_yticklabels(w2_scenario_names, fontsize=11)
        w2_ax.set_title("Figure NB2-3: Cache Corruption Detection Outcomes", fontsize=12, fontweight="bold")

        # Annotate cells with 0/1 values.
        for w2_i in range(len(w2_scenario_names)):
            for w2_j in range(2):
                w2_text_color = "white" if w2_outcomes[w2_i, w2_j] == 1 else "black"
                w2_ax.text(w2_j, w2_i, str(w2_outcomes[w2_i, w2_j]),
                          ha="center", va="center", color=w2_text_color,
                          fontsize=16, fontweight="bold")

        w2_cbar = plt.colorbar(w2_im, ax=w2_ax)
        w2_cbar.set_label("1 = Read Raised, 0 = Read Succeeded", fontsize=10)

        w2_fig.tight_layout()
        plt.show()

        # Check: all sibling keys should show 0 (no exception), confirming isolation.
        w2_sibling_column = w2_outcomes[:, 1]
        w2_isolation_verified = np.all(w2_sibling_column == 0)
        check("W2.3.8", w2_isolation_verified, f"All sibling keys show 0 (succeeded), confirming corruption isolation: {w2_sibling_column}")
        """),

        # How to read this chart
        md("""
        ### How to read this chart

        This grid visualizes read outcomes for three cache file corruption scenarios. Each row represents a corruption type (byte flip, truncation, checksum swap); columns show outcomes for the corrupted entry and its unmodified sibling.

        **Cell values:** 1 = read raised (corruption detected), 0 = read succeeded (no corruption detected).

        **Expected pattern:** Corrupted-key column should show all 1s (read raises, detecting tampering). Sibling-key column should show all 0s (read succeeds), confirming corruption does not leak across files.

        **Data source:** Each cell value is computed directly from live corruption tests (cells W2.3.2–W2.3.7). The 1/0 outcomes are derived from actual exception events, not hand-typed.

        **Why this matters:** Training-pipeline.md §1 specifies that corrupted cache entries must raise exceptions during read; a training loop that silently reads corrupted audio would fit a model to garbage and report false accuracy gains. This table proves that (a) corruption is reliably detected, (b) one corrupted entry does not poison sibling entries, and (c) the cache can be trusted as a fallback after the first data load.

        **Falsifier:** If any sibling-key cell showed 1, corruption would have leaked across files (violation of isolation). If any corrupted-key cell showed 0, the cache failed to detect corruption. The actual pattern validates that `CacheStore` isolates corruption at the file level through independent per-entry checksum validation.
        """),
    ]
