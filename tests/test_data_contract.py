from __future__ import annotations

from collections import Counter

from src.data import ManifestRecord, stratified_split


def _records(per_label: int = 20) -> list[ManifestRecord]:
    return [
        ManifestRecord(identifier=f"{label}-{index}", label=label, path=f"/{label}/{index}.wav")
        for label in ("left", "right", "up")
        for index in range(per_label)
    ]


def test_stratified_split_is_disjoint_reproducible_and_label_preserving() -> None:
    records = _records()
    first = stratified_split(records, seed=17, ratios=(0.70, 0.15, 0.15))
    second = stratified_split(records, seed=17, ratios=(0.70, 0.15, 0.15))

    first_ids = {name: {record.identifier for record in rows} for name, rows in first.items()}
    assert first_ids == {name: {record.identifier for record in rows} for name, rows in second.items()}
    assert first_ids["train"].isdisjoint(first_ids["calibration"])
    assert first_ids["train"].isdisjoint(first_ids["test"])
    assert first_ids["calibration"].isdisjoint(first_ids["test"])
    assert set().union(*first_ids.values()) == {record.identifier for record in records}

    for label in ("left", "right", "up"):
        counts = {name: Counter(row.label for row in rows)[label] for name, rows in first.items()}
        assert counts == {"train": 14, "calibration": 3, "test": 3}


def test_stratified_split_changes_selection_but_not_partition_sizes_with_seed() -> None:
    records = _records(per_label=21)
    first = stratified_split(records, seed=1, ratios=(0.70, 0.15, 0.15))
    second = stratified_split(records, seed=2, ratios=(0.70, 0.15, 0.15))

    assert {name: len(rows) for name, rows in first.items()} == {
        name: len(rows) for name, rows in second.items()
    }
    assert {record.identifier for record in first["train"]} != {
        record.identifier for record in second["train"]
    }
