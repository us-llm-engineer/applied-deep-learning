from __future__ import annotations

from pathlib import Path

from src.dataset import (
    AudioRecord,
    build_compact_manifest,
    find_duplicate_checksums,
    manifest_checksum,
)


def _records(root: Path) -> list[AudioRecord]:
    labels = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
    rows = [
        AudioRecord(
            identifier=f"{label}-{index}",
            label=label,
            path=root / f"{label}-{index}.wav",
            checksum=f"{label}-digest-{index}",
            sample_rate=16_000,
            num_frames=16_000,
        )
        for label in labels
        for index in range(3)
    ]
    rows.extend(
        [
            AudioRecord("unknown-0", "unknown", root / "unknown-0.wav", "unknown-a", 16_000, 16_000),
            AudioRecord("silence-0", "silence", root / "silence-0.wav", "silence-a", 16_000, 16_000),
            AudioRecord("other-0", "bird", root / "other-0.wav", "other-a", 16_000, 16_000),
        ]
    )
    return rows


def test_compact_manifest_is_seeded_limited_and_uses_only_study_labels(tmp_path: Path) -> None:
    rows = _records(tmp_path)
    first = build_compact_manifest(rows, cap_per_label=2, seed=17)
    second = build_compact_manifest(list(reversed(rows)), cap_per_label=2, seed=17)

    expected = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"}
    assert {row.label for row in first} == expected
    assert all(sum(row.label == label for row in first) <= 2 for label in expected)
    assert [row.identifier for row in first] == [row.identifier for row in second]
    assert "other-0" not in {row.identifier for row in first}


def test_manifest_digest_is_order_invariant_and_duplicate_audit_names_all_collisions(tmp_path: Path) -> None:
    rows = _records(tmp_path)
    assert manifest_checksum(rows) == manifest_checksum(list(reversed(rows)))
    changed = list(rows)
    changed[0] = AudioRecord("yes-0", "yes", tmp_path / "yes-0.wav", "changed", 16_000, 16_000)
    assert manifest_checksum(rows) != manifest_checksum(changed)

    collisions = find_duplicate_checksums(rows + [AudioRecord("copy", "yes", tmp_path / "copy.wav", "yes-digest-0", 16_000, 16_000)])
    assert collisions == {"yes-digest-0": ("copy", "yes-0")}
